"""
Dynamic weighting strategies for combining Cox loss and ConcordancePairwiseLoss.

This module provides various dynamic weighting strategies to optimally combine
Cox loss and ConcordancePairwiseLoss during training.
"""

import torch
from typing import Tuple, List, Dict, Any
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
        self.detailed_log = []
    
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
        
        # Log detailed information
        log_entry = {
            'epoch': epoch,
            'nll_loss': nll_loss,
            'pairwise_loss': pairwise_loss,
            'nll_norm': nll_norm,
            'pairwise_norm': pairwise_norm,
            'nll_w': nll_w,
            'pairwise_w': pairwise_w
        }
        self.detailed_log.append(log_entry)
        
        self.weight_history.append((nll_w, pairwise_w))
        return nll_w, pairwise_w
    
    def get_weights_adaptive(
        self, 
        epoch: int, 
        nll_loss: float, 
        pairwise_loss: float
    ) -> Tuple[float, float]:
        """
        Adaptive weighting strategy based on loss ratios.
        
        Args:
            epoch: Current training epoch
            nll_loss: Current Negative Partial Log-Likelihood loss value
            pairwise_loss: Current Pairwise Ranking Loss value
            
        Returns:
            Tuple of (nll_weight, pairwise_weight)
        """
        # Adaptive weights based on loss ratios
        total_loss = nll_loss + pairwise_loss
        if total_loss > 0:
            nll_w = pairwise_loss / total_loss
            pairwise_w = nll_loss / total_loss
        else:
            nll_w, pairwise_w = 0.5, 0.5
        
        # Apply epoch-based adjustment
        epoch_factor = min(epoch / self.total_epochs, 1.0)
        nll_w = nll_w * (1 - 0.3 * epoch_factor) + 0.3 * epoch_factor * 0.5
        pairwise_w = pairwise_w * (1 - 0.3 * epoch_factor) + 0.3 * epoch_factor * 0.5
        
        self.weight_history.append((nll_w, pairwise_w))
        return nll_w, pairwise_w
    
    def get_weights_linear(
        self, 
        epoch: int, 
        nll_loss: float, 
        pairwise_loss: float
    ) -> Tuple[float, float]:
        """
        Linear weighting strategy that changes over epochs.
        
        Args:
            epoch: Current training epoch
            nll_loss: Current Negative Partial Log-Likelihood loss value
            pairwise_loss: Current Pairwise Ranking Loss value
            
        Returns:
            Tuple of (nll_weight, pairwise_weight)
        """
        # Linear progression from equal weights to loss-based weights
        progress = epoch / self.total_epochs
        
        # Start with equal weights, gradually move to loss-based weights
        nll_w = 0.5 * (1 - progress) + (pairwise_loss / (nll_loss + pairwise_loss)) * progress
        pairwise_w = 0.5 * (1 - progress) + (nll_loss / (nll_loss + pairwise_loss)) * progress
        
        self.weight_history.append((nll_w, pairwise_w))
        return nll_w, pairwise_w
    
    def get_weights_cosine(
        self, 
        epoch: int, 
        nll_loss: float, 
        pairwise_loss: float
    ) -> Tuple[float, float]:
        """
        Cosine annealing weighting strategy.
        
        Args:
            epoch: Current training epoch
            nll_loss: Current Negative Partial Log-Likelihood loss value
            pairwise_loss: Current Pairwise Ranking Loss value
            
        Returns:
            Tuple of (nll_weight, pairwise_weight)
        """
        # Cosine annealing between equal weights and loss-based weights
        progress = epoch / self.total_epochs
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        loss_based_nll = pairwise_loss / (nll_loss + pairwise_loss)
        loss_based_pairwise = nll_loss / (nll_loss + pairwise_loss)
        
        nll_w = 0.5 * cosine_factor + loss_based_nll * (1 - cosine_factor)
        pairwise_w = 0.5 * cosine_factor + loss_based_pairwise * (1 - cosine_factor)
        
        self.weight_history.append((nll_w, pairwise_w))
        return nll_w, pairwise_w
    
    def get_weights(
        self, 
        epoch: int, 
        nll_loss: float, 
        pairwise_loss: float,
        strategy: str = "scale_balanced"
    ) -> Tuple[float, float]:
        """
        Get weights using specified strategy.
        
        Args:
            epoch: Current training epoch
            nll_loss: Current Negative Partial Log-Likelihood loss value
            pairwise_loss: Current Pairwise Ranking Loss value
            strategy: Weighting strategy ('scale_balanced', 'adaptive', 'linear', 'cosine')
            
        Returns:
            Tuple of (nll_weight, pairwise_weight)
        """
        if strategy == "scale_balanced":
            return self.get_weights_scale_balanced(epoch, nll_loss, pairwise_loss)
        elif strategy == "adaptive":
            return self.get_weights_adaptive(epoch, nll_loss, pairwise_loss)
        elif strategy == "linear":
            return self.get_weights_linear(epoch, nll_loss, pairwise_loss)
        elif strategy == "cosine":
            return self.get_weights_cosine(epoch, nll_loss, pairwise_loss)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about weight evolution.
        
        Returns:
            Dictionary containing weight statistics
        """
        if not self.weight_history:
            return {}
        
        nll_weights = [w[0] for w in self.weight_history]
        pairwise_weights = [w[1] for w in self.weight_history]
        
        return {
            'nll_weight_mean': np.mean(nll_weights),
            'nll_weight_std': np.std(nll_weights),
            'nll_weight_min': np.min(nll_weights),
            'nll_weight_max': np.max(nll_weights),
            'pairwise_weight_mean': np.mean(pairwise_weights),
            'pairwise_weight_std': np.std(pairwise_weights),
            'pairwise_weight_min': np.min(pairwise_weights),
            'pairwise_weight_max': np.max(pairwise_weights),
            'total_epochs': len(self.weight_history)
        }
