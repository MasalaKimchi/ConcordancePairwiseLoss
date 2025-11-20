"""
ContrastiveCoxLoss implementation.

This module contains the main ContrastiveCoxLoss class, which implements a Cox-style
maximum partial likelihood loss where learnable multimodal pairwise similarity substitutes
the risk score, integrating classic contrastive learning into survival likelihood for more
meaningful feature alignment. Uses torchsurv's built-in IPCW weighting for proper handling
of censoring.

Based on unified "Contrastive Cox Likelihood" (ConCox) formulation:
    For each event (uncensored) patient i and all at-risk j,
    use a learnable similarity S(z_i, z_j) in place of log-risk, with Cox-style denominator.

**Key Features**:
- Drop-in compatible with other pairwise/interchangeable survival losses (Cox, DPCL, etc.)
- Computes valid risk sets with correct censoring
- Uses cosine similarity for pairwise feature comparison
- Proper support for minibatch and full-batch mode
- Weighting by IPCW only on event samples (per Uno et al.)
- Option to use temperature scaling for flexible margins (like InfoNCE)
- Handles gradient stability in degenerate cases

References: See AI analysis for mathematical foundation, [Su et al. 2024][web:68], [PCLSurv][web:63][web:66].

"""

import torch
import torch.nn.functional as F
from typing import Optional
from torchsurv.stats.ipcw import get_ipcw

class ContrastiveCoxLoss:
    """
    ContrastiveCoxLoss for survival analysis with Cox-style likelihood
    using pairwise learned cosine similarity.

    Args:
        temperature: Temperature for InfoNCE-style denominator (float)
        reduction: 'mean', 'sum', or 'none'
        ipcw_weights: (optional) Pre-computed IPCW weights
        use_ipcw: Whether to use IPCW weighting (default: True)
    """
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = 'mean',
        ipcw_weights: Optional[torch.Tensor]=None,
        use_ipcw: bool = True,
    ):
        self.temperature = temperature
        self.reduction = reduction
        self.ipcw_weights = ipcw_weights
        self.use_ipcw = use_ipcw
        if reduction not in ['mean','sum','none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def sim(self, zi, zj):
        """Compute cosine similarity between feature vectors."""
        return F.cosine_similarity(zi, zj, dim=-1)

    def __call__(
        self,
        feats: torch.Tensor,      # (N, D): multimodal feature representation for each patient
        times: torch.Tensor,      # (N,)
        events: torch.Tensor,     # (N,) - bool or float {0,1}
        sample_indices: Optional[torch.Tensor] = None,  # for batch/subset
    ) -> torch.Tensor:
        """
        Compute the Contrastive Cox loss.

        Args:
            feats: Feature matrix (N,D)
            times: Survival times (N,)
            events: Event indicators (N,)
            sample_indices: Selects subset in batch (optional)
        Returns:
            Loss tensor
        """
        N = feats.size(0)
        device = feats.device
        times = times.to(device)
        events = events.to(device)
        if events.dtype != torch.bool:
            events = events.bool()

        # Compute pairwise similarity matrix (N,N)
        zi = feats.unsqueeze(1).expand(-1,N,-1)   # (N,N,D)
        zj = feats.unsqueeze(0).expand(N,-1,-1)   # (N,N,D)
        sim_mat = self.sim(zi, zj) / self.temperature # (N,N)

        # Valid risk set: for i (event), all j with t_j >= t_i
        time = times.unsqueeze(0)  # (1,N)
        ti = times.unsqueeze(1)    # (N,1)
        riskmask = (time >= ti)    # (N,N) -> risk set for row (i,*)

        # If no events, avoid crash
        if not events.any():
            return feats.sum() * 0.0

        # Optionally get IPCW weights (as in ConcordancePairwiseLoss)
        if self.use_ipcw and self.ipcw_weights is None:
            times_cpu = times.cpu()
            events_cpu = events.cpu()
            ipcw = get_ipcw(events_cpu, times_cpu).to(device)
        elif self.use_ipcw:
            if sample_indices is not None:
                ipcw = self.ipcw_weights[sample_indices].to(device)
            else:
                ipcw = self.ipcw_weights.to(device)
        else:
            ipcw = torch.ones(N, device=device)

        # Numerator: sim(i,i). We'll use diagonal for each patient/event anchor
        diag_idx = torch.arange(N, device=device)
        sim_diag = sim_mat[diag_idx, diag_idx]  # (N,)

        # Denominator: sum over all risk set for anchor i
        # Note: If risk set only contains the event itself, loss contribution is 0
        # (this is mathematically correct for Cox partial likelihood)
        denom = (torch.exp(sim_mat) * riskmask.float()).sum(dim=1) + 1e-12 # (N,)

        # Loss vector: -[sim_diag - log denom] for event anchors; zero for censored
        loss_vec = torch.zeros(N, device=device)
        pos_mask = events
        loss_vec[pos_mask] = -(sim_diag[pos_mask] - torch.log(denom[pos_mask]))

        # Multiply loss by IPCW (row-wise, events only)
        weighted_loss = loss_vec * ipcw

        # Reduction
        if self.reduction == 'mean':
            total_weight = ipcw[events].sum()
            return weighted_loss[events].sum() / total_weight if total_weight > 0 else weighted_loss.sum() * 0.0
        elif self.reduction == 'sum':
            return weighted_loss[events].sum()
        else:  # "none"
            return weighted_loss

# Example Usage:
# feats: torch.FloatTensor [N,D] from your multimodal model's last hidden layer
# times: torch.FloatTensor [N]
# events: torch.BoolTensor [N]
# Use in your training loop as you would Cox partial likelihood or ConcordancePairwiseLoss.

