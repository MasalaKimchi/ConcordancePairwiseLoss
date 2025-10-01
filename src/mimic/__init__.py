"""
MIMIC-IV Chest X-ray Dataset Module

This module provides a complete pipeline for survival analysis using MIMIC-IV chest X-ray images.
It includes data preprocessing, stratified patient-level splitting, and data loading utilities.

Classes:
    MIMICDataLoader: Main data loader for MIMIC-IV chest X-ray data
    MIMICImageDataset: PyTorch dataset for MIMIC images with survival targets
    MIMICPreprocessor: Handles data preprocessing and stratified splitting

Functions:
    preprocess_mimic_data: Main preprocessing function
    get_efficientnet_transforms: Image transforms optimized for EfficientNet
"""

from .mimic_data_loader import MIMICDataLoader
from .dataset import MIMICImageDataset
from .preprocessor import MIMICPreprocessor, preprocess_mimic_data
from .transforms import get_efficientnet_transforms

__all__ = [
    'MIMICDataLoader',
    'MIMICImageDataset', 
    'MIMICPreprocessor',
    'preprocess_mimic_data',
    'get_efficientnet_transforms'
]

__version__ = "1.0.0"
