"""
Dynamic weighting for combining Cox loss (NLL) and Pairwise loss.

Kept minimal and practical:
- Scale-balanced normalization (simple, robust default)
- Optional GradNorm-inspired weighting (no hard-coded factors)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Sequence
import numpy as np


class NormalizedLossCombination:
    """
    Scale-balanced loss combination with fixed normalization factors.
    
    This class combines Negative Partial Log-Likelihood (NLL) and Pairwise Ranking Loss
    by normalizing both losses to similar scales using predetermined factors, then
    applying inverse weighting based on the normalized loss values.
    
    Args:
        total_epochs: Total number of training epochs
        nll_norm_factor: Normalization factor for NLL loss (default: 25.0)
        pairwise_norm_factor: Normalization factor for Pairwise Ranking Loss (default: 2.0)
    """
    
    def __init__(
        self, 
        total_epochs: int = 100,
        nll_norm_factor: float = 25.0,
        pairwise_norm_factor: float = 2.0
    ):
        self.total_epochs = total_epochs
        self.nll_norm_factor = nll_norm_factor
        self.pairwise_norm_factor = pairwise_norm_factor
        self.weight_history = []
    
    def get_weights_scale_balanced(
        self, 
        epoch: int, 
        nll_loss: float, 
        pairwise_loss: float
    ) -> Tuple[float, float]:
        """
        Scale-balanced weighting strategy - best performing approach.
        
        This method normalizes both losses to [0, 1] range using predetermined factors
        and then applies inverse weighting based on normalized loss values.
        
        Args:
            epoch: Current training epoch
            nll_loss: Current Negative Partial Log-Likelihood loss value
            pairwise_loss: Current Pairwise Ranking Loss value
            
        Returns:
            Tuple of (nll_weight, pairwise_weight)
        """
        # Normalize both losses to [0, 1] range using predetermined factors
        nll_norm = nll_loss / self.nll_norm_factor
        pairwise_norm = pairwise_loss / self.pairwise_norm_factor
        
        # Inverse weighting: higher normalized loss gets lower weight
        total_norm = nll_norm + pairwise_norm
        if total_norm > 0:
            nll_w = pairwise_norm / total_norm
            pairwise_w = nll_norm / total_norm
        else:
            nll_w, pairwise_w = 0.5, 0.5
        
        self.weight_history.append((nll_w, pairwise_w))
        return nll_w, pairwise_w

    def get_weights_gradnorm(
        self,
        epoch: int,
        nll_loss_tensor: torch.Tensor,
        pairwise_loss_tensor: torch.Tensor,
        shared_parameters: Sequence[nn.Parameter],
        alpha: float = 1.0,
        eps: float = 1e-8,
    ) -> Tuple[float, float]:
        """
        Gradient-norm based weighting (GradNorm-inspired) without hard-coded scale factors.

        This computes the L2 norm of gradients of each loss with respect to a set of
        shared parameters (typically the shared backbone), and assigns weights inversely
        proportional to those norms (optionally with exponent ``alpha``). The intuition
        is to balance contributions so that losses with larger gradients receive smaller
        weights and vice versa.

        Args:
            epoch: Current epoch (not used but kept for symmetry)
            nll_loss_tensor: Differentiable tensor for NLL loss
            pairwise_loss_tensor: Differentiable tensor for Pairwise loss
            shared_parameters: Iterable of shared nn.Parameters to compute grad norms on
            alpha: Exponent controlling strength of rebalancing (1.0 = inverse proportional)
            eps: Small constant to avoid division by zero

        Returns:
            Tuple of (nll_weight, pairwise_weight)

        Notes:
            - Caller must ensure ``retain_graph=True`` in upstream usage if needed.
            - This method uses ``torch.autograd.grad`` and does not call ``backward``.
        """
        params = [p for p in shared_parameters if p is not None and p.requires_grad]
        if len(params) == 0:
            # Fallback to equal weights if no parameters provided
            return 0.5, 0.5

        def grad_l2_norm(loss_t: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(
                loss_t, params, retain_graph=True, allow_unused=True
            )
            sq_sum = torch.zeros((), device=loss_t.device)
            for g in grads:
                if g is not None:
                    sq_sum = sq_sum + (g.detach()**2).sum()
            return torch.sqrt(sq_sum + eps)

        g_nll = grad_l2_norm(nll_loss_tensor)
        g_pair = grad_l2_norm(pairwise_loss_tensor)

        # Inverse-gradient weighting with exponent alpha
        inv_nll = (g_nll + eps).pow(-alpha)
        inv_pair = (g_pair + eps).pow(-alpha)
        s = inv_nll + inv_pair + eps
        w_nll = (inv_nll / s).clamp_(0.0, 1.0)
        w_pair = (inv_pair / s).clamp_(0.0, 1.0)

        # Record for analysis (store floats)
        self.weight_history.append((float(w_nll.item()), float(w_pair.item())))
        return float(w_nll.item()), float(w_pair.item())
