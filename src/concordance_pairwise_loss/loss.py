"""
ConcordancePairwiseLoss implementation.

This module contains the main ConcordancePairwiseLoss class that implements
a pairwise loss function for survival analysis to improve concordance.
Uses torchsurv's built-in IPCW implementation for inverse probability of censoring weights.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Import torchsurv's IPCW implementation
from torchsurv.stats.ipcw import get_ipcw


class ConcordancePairwiseLoss:
    """
    ConcordancePairwiseLoss for survival analysis with proper comparable pair handling.
    
    This loss function improves concordance between predicted and actual survival times
    by comparing pairs of samples and penalizing incorrect rankings. Uses the correct
    definition of comparable pairs: (i,j) is comparable only if t_i < t_j AND δ_i = 1
    (earlier sample had an event).
    
    Key features:
    - Proper comparable pair definition for right-censoring
    - IPCW weights applied to earlier index only (Uno's method)
    - Weight-normalized reduction for proper loss scaling
    - Explicit tie handling (time ties excluded)
    - Maintains gradient flow in degenerate cases
    
    Args:
        temperature: Temperature parameter for the logistic function (default: 1.0)
        reduction: Reduction method ('mean', 'sum', 'none') - 'mean' uses weight normalization
        ipcw_weights: Pre-computed IPCW weights (optional)
        temp_scaling: Temperature scaling strategy ('linear', 'log', 'sqrt')
        pairwise_sampling: Pairwise sampling strategy:
                          - 'all': All comparable pairs
                          - 'event_only': Only pairs where both samples have events
                          - 'balanced': All comparable pairs with higher weight for event-event pairs
        use_ipcw: Whether to use IPCW weights (default: True)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
        ipcw_weights: Optional[torch.Tensor] = None,
        temp_scaling: str = "linear",
        pairwise_sampling: str = "balanced",
        use_ipcw: bool = True,
    ):
        self.temperature = temperature
        self.reduction = reduction
        self.ipcw_weights = ipcw_weights
        self.temp_scaling = temp_scaling
        self.pairwise_sampling = pairwise_sampling
        self.use_ipcw = use_ipcw
        
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        
        if temp_scaling not in ["linear", "log", "sqrt"]:
            raise ValueError("temp_scaling must be 'linear', 'log', or 'sqrt'")
        
        if pairwise_sampling not in ["all", "event_only", "balanced"]:
            raise ValueError("pairwise_sampling must be 'all', 'event_only', or 'balanced'")
    
    def __call__(
        self,
        log_risks: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the ConcordancePairwiseLoss.
        
        Args:
            log_risks: Log hazard predictions (N, 1) or (N,)
            times: Survival times (N,)
            events: Event indicators (N,) - boolean or float
            
        Returns:
            Loss tensor
        """
        # Handle input shapes
        if log_risks.dim() == 2 and log_risks.size(1) == 1:
            log_risks = log_risks.squeeze(1)
        elif log_risks.dim() != 1:
            raise ValueError("log_risks must be a 1D tensor or 2D tensor with shape (N, 1)")
        
        if log_risks.size(0) != times.size(0) or log_risks.size(0) != events.size(0):
            raise ValueError("All inputs must have the same batch size")
        
        # Convert events to boolean if needed
        if events.dtype != torch.bool:
            events = events.bool()
        
        n = log_risks.size(0)
        
        # Get IPCW weights if needed
        if self.use_ipcw and self.ipcw_weights is None:
            # Handle edge case where all samples are censored
            if not events.any():
                # If all censored, no comparable pairs exist anyway, use unit weights
                ipcw = torch.ones(n, device=log_risks.device)
            else:
                # torchsurv's get_ipcw has device issues, so we compute on CPU then move to GPU
                events_cpu = events.cpu()
                times_cpu = times.cpu()
                ipcw = get_ipcw(events_cpu, times_cpu)
                # Move result back to the same device as log_risks
                ipcw = ipcw.to(log_risks.device)
        elif self.use_ipcw and self.ipcw_weights is not None:
            # Use precomputed weights
            if sample_indices is not None:
                # Use specific indices for batch samples
                ipcw = self.ipcw_weights[sample_indices].to(log_risks.device)
            else:
                # Fall back to computing IPCW on current batch
                if not events.any():
                    ipcw = torch.ones(n, device=log_risks.device)
                else:
                    events_cpu = events.cpu()
                    times_cpu = times.cpu()
                    ipcw = get_ipcw(events_cpu, times_cpu)
                    ipcw = ipcw.to(log_risks.device)
        else:
            ipcw = torch.ones(n, device=log_risks.device)
        
        # Compute pairwise differences
        log_risk_diff = log_risks.unsqueeze(1) - log_risks.unsqueeze(0)  # (n, n)
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)  # (n, n)
        
        # Create comparable pairs mask: t_i < t_j AND δ_i = 1 (earlier sample had event)
        # This is the fundamental definition for survival analysis concordance
        time_ordering = time_diff < 0  # t_i < t_j (row i < col j)
        earlier_event = events.unsqueeze(1).expand(-1, n)  # δ_i = 1 for earlier sample
        comparable_pairs = time_ordering & earlier_event
        
        # Remove diagonal (self-pairs) and ties in time
        not_diagonal = ~torch.eye(n, device=log_risks.device, dtype=torch.bool)
        not_time_ties = time_diff != 0  # Exclude time ties
        comparable_pairs = comparable_pairs & not_diagonal & not_time_ties
        
        # Apply pairwise sampling strategy on top of comparable pairs
        if self.pairwise_sampling == "all":
            # Use all comparable pairs
            valid_pairs = comparable_pairs
        elif self.pairwise_sampling == "event_only":
            # Only pairs where both samples have events (most informative)
            both_events = events.unsqueeze(0) & events.unsqueeze(1)
            valid_pairs = comparable_pairs & both_events
        elif self.pairwise_sampling == "balanced":
            # Use all comparable pairs (already filtered for δ_i = 1)
            valid_pairs = comparable_pairs
        else:
            valid_pairs = comparable_pairs
        
        if not valid_pairs.any():
            # No valid pairs - maintain gradient flow
            return log_risks.sum() * 0.0
        
        # Apply temperature scaling
        if self.temp_scaling == "linear":
            temp = self.temperature
        elif self.temp_scaling == "log":
            temp = torch.log(torch.tensor(self.temperature + 1.0, device=log_risks.device))
        elif self.temp_scaling == "sqrt":
            temp = torch.sqrt(torch.tensor(self.temperature, device=log_risks.device))
        else:
            temp = self.temperature
        
        # Compute logits
        logits = log_risk_diff / temp
        
        # Apply IPCW weights - only to earlier index (i) as per Uno's IPCW
        # ipcw[i] applies to pair (i,j) where i is the earlier sample with event
        ipcw_weights = ipcw.unsqueeze(1).expand(-1, n)  # Shape: (n, n)
        
        # Compute pairwise weights for balanced sampling
        if self.pairwise_sampling == "balanced":
            # Weight event-event pairs more heavily (both samples have events)
            both_events = events.unsqueeze(0) & events.unsqueeze(1)
            balance_weights = torch.ones_like(valid_pairs, dtype=torch.float)
            balance_weights[both_events & valid_pairs] = 2.0  # Higher weight for event-event pairs
            
            # Combine IPCW weights with balance weights
            combined_weights = ipcw_weights * balance_weights
        else:
            # Use IPCW weights directly
            combined_weights = ipcw_weights
        
        # Apply weights only to valid pairs
        weighted_valid_pairs = combined_weights * valid_pairs.float()
        
        # Compute loss
        loss_matrix = F.softplus(-logits) * weighted_valid_pairs
        
        # Apply reduction with proper weight normalization
        if self.reduction == "mean":
            # Normalize by sum of weights over valid pairs (not raw count)
            total_weight = weighted_valid_pairs.sum()
            if total_weight > 0:
                return loss_matrix.sum() / total_weight
            else:
                return log_risks.sum() * 0.0
        elif self.reduction == "sum":
            return loss_matrix.sum()
        else:  # "none"
            return loss_matrix.sum(dim=1)
