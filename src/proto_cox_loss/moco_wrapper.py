import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MoCoCoxWrapper(nn.Module):
    """
    MoCo-style Wrapper for Survival Models (inspired by torchsurv.Momentum / He et al.).

    Unlike the simple Memory Bank, this wraps the **Backbone Model**.
    It maintains a 'Momentum Encoder' (EMA copy of the backbone) to generate
    consistent features for the Queue, ensuring stability even with Batch Size=4.

    Args:
        backbone (nn.Module): Your 3D ResNet/CNN.
        dim (int): Feature dimension (output of backbone).
        queue_size (int): Number of negatives to store (e.g., 1024).
        m (float): Momentum coefficient (0.999 is standard for MoCo).
        temperature (float): Temperature for the contrastive logits.
    """
    def __init__(self, backbone, dim, queue_size=1024, m=0.999, temperature=0.07):
        super().__init__()
        
        self.backbone_q = backbone  # Query Encoder (Main Model)
        self.backbone_k = copy.deepcopy(backbone) # Key Encoder (Momentum Copy)
        
        # Freeze Key Encoder (No Gradients)
        for param in self.backbone_k.parameters():
            param.requires_grad = False
            
        self.queue_size = queue_size
        self.m = m
        self.temperature = temperature
        
        # The Queue (Features + Times + Events)
        self.register_buffer("queue_feats", torch.randn(queue_size, dim))
        self.register_buffer("queue_times", torch.zeros(queue_size))
        self.register_buffer("queue_events", torch.zeros(queue_size)) # Standard MoCo doesn't need events, but Cox does
        self.queue_feats = F.normalize(self.queue_feats, dim=1)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder: 
        param_k = m * param_k + (1 - m) * param_q
        """
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, times, events):
        """Update queue with new keys (from momentum encoder)."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # If batch is larger than remaining space, strictly handle wrap-around 
        # (Simplification: assume queue_size % batch_size == 0 usually)
        if ptr + batch_size > self.queue_size:
            ptr = 0 # Reset for simplicity in this snippet
            
        self.queue_feats[ptr:ptr + batch_size] = keys
        self.queue_times[ptr:ptr + batch_size] = times
        self.queue_events[ptr:ptr + batch_size] = events
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x, times, events, loss_fn):
        """
        Args:
            x: Input images (B, C, D, H, W)
            times: (B,)
            events: (B,)
            loss_fn: The ProtoCoxLoss function (expects logits or feats)
        """
        # 1. Compute Query Features (Gradients flow here)
        q = self.backbone_q(x)  # (B, dim)
        q = F.normalize(q, dim=1)

        # 2. Compute Key Features (No Gradients) using Momentum Encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()  # Update weights first
            k = self.backbone_k(x)  # (B, dim)
            k = F.normalize(k, dim=1)

        # 3. Compute Loss
        # We pass the Query (q) as the "Current Batch"
        # We pass the Queue (queue_feats) as the "Risk Set History"
        
        # NOTE: We need to slightly adapt how we pass this to ProtoCoxLoss.
        # Standard ProtoCox expects just (feats, times).
        # Here we manually construct the "Large Denominator".
        
        # Get Queue State
        queue_feats = self.queue_feats.clone().detach()
        queue_times = self.queue_times.clone().detach()
        
        # Calculate Logits for Query vs Prototype (Numerator)
        # We assume loss_fn has the 'risk_prototype'
        proto = F.normalize(loss_fn.risk_prototype, dim=0)
        logits_q = torch.matmul(q, proto) / self.temperature
        
        # Calculate Logits for Queue vs Prototype (Denominator context)
        logits_queue = torch.matmul(queue_feats, proto) / self.temperature
        
        # --- Cox Partial Likelihood with MoCo Queue ---
        
        # Combine: [Current Batch (q)] + [History (queue)]
        # Note: Unlike standard MoCo (Instance Discrim), in Cox we just need
        # the Queue to populate the Risk Set Denominator.
        
        all_logits = torch.cat([logits_q, logits_queue], dim=0) # (B + Q)
        all_times = torch.cat([times, queue_times], dim=0)      # (B + Q)
        
        # Mask: For each patient 'i' in Batch, who in 'all' is in risk set?
        # i = (B,), j = (B+Q,)
        time_i = times.unsqueeze(1)          # (B, 1)
        time_j = all_times.unsqueeze(0)      # (1, B+Q)
        risk_mask = (time_j >= time_i).float()
        
        # Denominator
        exp_all = torch.exp(all_logits).unsqueeze(0) # (1, B+Q)
        denom = (exp_all * risk_mask).sum(dim=1)
        log_denom = torch.log(denom + 1e-8)
        
        # Loss (Events in Batch only)
        loss_vec = -(logits_q - log_denom)
        loss = loss_vec[events.bool()].mean() if events.bool().any() else torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # 4. Update Queue with the STABLE keys (k), not the changing queries (q)
        self._dequeue_and_enqueue(k, times, events)
        
        return loss