import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ProtoCoxLoss(nn.Module):
    """
    Unified 'Proto-Cox' Loss: A Geometric Contrastive formulation of Cox Partial Likelihood.
    
    Instead of a linear risk score f(x) = w^T x, this loss defines risk as the 
    Cosine Similarity between a patient's embedding and a learnable 'Event Prototype'.
    
    This unifies Contrastive Learning (InfoNCE) and Survival Analysis (Cox) into a 
    single mathematical framework without auxiliary terms.
    
    Args:
        input_dim (int): Dimension of the input feature vectors (D).
        temperature (float): Scaling factor for logits. Low values (<0.1) sharpen the distribution.
        reduction (str): 'mean', 'sum', or 'none'.
        use_ipcw (bool): Whether to apply Inverse Probability of Censoring Weighting.
    """
    def __init__(
        self,
        input_dim: int,
        temperature: float = 0.07,
        reduction: str = 'mean',
        use_ipcw: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.use_ipcw = use_ipcw
        
        # Learnable 'Risk Prototype' Vector (The concept of "Death/High Risk")
        # We initialize it randomly. It will rotate during training to find the 
        # direction of maximum risk in the hypersphere.
        self.risk_prototype = nn.Parameter(torch.randn(input_dim))
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def forward(
        self,
        feats: torch.Tensor,                  # (N, D): Multimodal embeddings
        times: torch.Tensor,                  # (N,): Survival times
        events: torch.Tensor,                 # (N,): Event indicators (1=Event, 0=Censored)
        ipcw_weights: Optional[torch.Tensor] = None # (N,): Optional instance weights
    ) -> torch.Tensor:
        """
        Compute the Geometric Contrastive Cox Loss.
        
        Formula:
            L = - sum_{i: Event} [ sim(z_i, w)/tau - log( sum_{j in RiskSet} exp(sim(z_j, w)/tau) ) ]
        """
        N, D = feats.shape
        device = feats.device
        
        # 1. Normalize Inputs & Prototype to hypersphere (Geometric constraint)
        feats_norm = F.normalize(feats, p=2, dim=1)
        proto_norm = F.normalize(self.risk_prototype, p=2, dim=0)
        
        # 2. Compute "Geometric Risk Scores" (Cosine Similarity)
        # Shape: (N,)
        # This replaces the standard linear f(x) = w*x with cos(theta)
        logits = torch.matmul(feats_norm, proto_norm) / self.temperature
        
        # 3. Construct Risk Set Mask
        # riskmask[i, j] = 1 if t_j >= t_i (j is in risk set of i)
        # Note: handling ties can be complex; here we use standard inclusive definition
        times = times.view(-1, 1)
        riskmask = (times.T >= times).float()  # (N, N)
        
        # Mask out self-comparison in denominator? 
        # Standard Cox includes the event itself in the denominator.
        # InfoNCE usually excludes it. 
        # We stick to Standard Cox definition (include self) for statistical validity.
        
        # 4. Compute Log-Sum-Exp over Risk Sets
        # We want: log( sum_{j in R_i} exp(logit_j) )
        exp_logits = torch.exp(logits) * riskmask # (N, N) - masked
        
        # Numerical stability trick: sum might be 0 for empty risk sets (last event)
        # We add epsilon, though technically the last event always has itself in risk set.
        denom = exp_logits.sum(dim=1) # (N,)
        log_denom = torch.log(denom + 1e-8)
        
        # 5. Calculate Partial Likelihood for EVENTS only
        # L_i = - (logit_i - log_denom_i)
        loss_vec = -(logits - log_denom)
        
        # 6. Apply Event Mask (Loss is only defined for observed events)
        events = events.bool()
        loss_vec = loss_vec * events.float()
        
        # 7. Apply IPCW (Inverse Probability of Censoring Weights)
        if self.use_ipcw and ipcw_weights is not None:
            loss_vec = loss_vec * ipcw_weights
            
        # 8. Reduction
        if self.reduction == 'sum':
            return loss_vec.sum()
        elif self.reduction == 'mean':
            # Divide by number of events (or sum of weights), not batch size
            normalizer = (events.float() * ipcw_weights).sum() if (self.use_ipcw and ipcw_weights is not None) else events.sum()
            return loss_vec.sum() / (normalizer + 1e-8)
        else:
            return loss_vec