"""
ConcordancePairwiseLoss: A pairwise loss function for survival analysis.

This package provides a pairwise loss function that improves concordance
between predicted and actual survival times by comparing pairs of samples.
"""

from .loss import ConcordancePairwiseLoss
from .dynamic_weighting import NormalizedLossCombination

__version__ = "1.0.0"
__author__ = "JustinNKim"
__email__ = "justin@example.com"

__all__ = [
    "ConcordancePairwiseLoss",
    "NormalizedLossCombination",
]
