"""
ProtoCoxLoss: Geometric Contrastive Formulation of Cox Partial Likelihood

This package provides:
- ProtoCoxLoss: A geometric contrastive formulation of Cox partial likelihood
- MoCoCoxWrapper: Momentum contrast wrapper for ProtoCoxLoss
- ProtoCoxLossTrainer: Custom training loop for ProtoCoxLoss models
"""

from .loss import ProtoCoxLoss
from .moco_wrapper import MoCoCoxWrapper
from .trainer import ProtoCoxLossTrainer, save_protocox_model, load_protocox_model

__all__ = [
    'ProtoCoxLoss',
    'MoCoCoxWrapper',
    'ProtoCoxLossTrainer',
    'save_protocox_model',
    'load_protocox_model'
]

