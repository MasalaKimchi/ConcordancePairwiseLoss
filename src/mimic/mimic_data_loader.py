"""
MIMIC-IV Data Loader Module

This module provides data loading utilities for MIMIC-IV chest X-ray data
with survival analysis targets.
"""

import os
import warnings
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import MIMICImageDataset
from .transforms import get_efficientnet_transforms
from data_loaders import AbstractDataLoader

# Suppress MONAI deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# MONAI imports for optimized data loading
from monai.data import CacheDataset, ThreadDataLoader, DataLoader as MonaiDataLoader
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityD, RandFlipD, RandRotateD, RandZoomD
from monai.utils import set_determinism

set_determinism(seed=42, use_deterministic_algorithms=False)


class MIMICDataLoader(AbstractDataLoader):
    """
    MIMIC-IV Chest X-ray dataset loader implementation.
    
    This class provides a complete data loading pipeline for MIMIC-IV chest X-ray data,
    including stratified train/validation/test splits and proper image preprocessing.
    """
    
    def __init__(
        self, 
        batch_size: int = 128,
        data_dir: str = "Y:/mimic-cxr-jpg-2.1.0.physionet.org/",
        csv_path: str = "data/mimic/mimic_cxr_splits.csv",
        target_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize MIMIC data loader.
        
        Args:
            batch_size: Batch size for data loaders
            data_dir: Base directory containing MIMIC data
            csv_path: Path to preprocessed CSV file
            target_size: Target image size for EfficientNet-B0
            use_augmentation: Whether to use data augmentation for training
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Load MIMIC-IV Chest X-ray dataset.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_features)
            For image data, num_features represents the number of channels (3 for RGB)
        """
        # Load preprocessed data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Preprocessed CSV not found at {self.csv_path}. "
                "Please run preprocessing first to generate the CSV file."
            )
        
        df = pd.read_csv(self.csv_path)
        
        # Filter out images that don't exist
        df = df[df['exists'] == True].copy()
        
        # Split data
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        print(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        print(f"Event rates - Train: {df_train['event'].mean():.3f}, Val: {df_val['event'].mean():.3f}, Test: {df_test['event'].mean():.3f}")
        
        # Create transforms
        train_transform = get_efficientnet_transforms(
            target_size=self.target_size,
            is_training=self.use_augmentation
        )
        val_test_transform = get_efficientnet_transforms(
            target_size=self.target_size,
            is_training=False
        )
        
        # Create datasets
        train_dataset = MIMICImageDataset(
            df_train, 
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=train_transform,
            target_size=self.target_size
        )
        
        val_dataset = MIMICImageDataset(
            df_val,
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=val_test_transform,
            target_size=self.target_size
        )
        
        test_dataset = MIMICImageDataset(
            df_test,
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=val_test_transform,
            target_size=self.target_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        # For image data, num_features represents the number of channels (3 for RGB)
        num_features = 3
        
        return train_loader, val_loader, test_loader, num_features
    
    def get_dataset_stats(self) -> dict:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Preprocessed CSV not found at {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        df = df[df['exists'] == True].copy()
        
        stats = {
            'total_samples': len(df),
            'train_samples': len(df[df['split'] == 'train']),
            'val_samples': len(df[df['split'] == 'val']),
            'test_samples': len(df[df['split'] == 'test']),
            'overall_event_rate': df['event'].mean(),
            'train_event_rate': df[df['split'] == 'train']['event'].mean(),
            'val_event_rate': df[df['split'] == 'val']['event'].mean(),
            'test_event_rate': df[df['split'] == 'test']['event'].mean(),
            'mean_survival_time': df['tte'].mean(),
            'median_survival_time': df['tte'].median()
        }
        
        return stats


class OptimizedMIMICDataLoader:
    """MONAI-optimized MIMIC data loader for maximum performance."""
    
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "Y:/mimic-cxr-jpg-2.1.0.physionet.org/",
        csv_path: str = "data/mimic/mimic_cxr_splits.csv",
        target_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = True,
        cache_rate: float = 0.1,  # Cache 10% of data in memory
        num_workers: int = 8,
        pin_memory: bool = True
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def _create_monai_transforms(self, is_training: bool = True):
        """Create MONAI-optimized transforms."""
        # MONAI transforms for better performance
        from monai.data import PILReader
        
        def convert_to_rgb(image):
            """Convert image to RGB format for EfficientNet compatibility."""
            return image.convert("RGB")
        
        transforms = [
            LoadImageD(keys=["image"], reader=PILReader(converter=convert_to_rgb)),
            EnsureChannelFirstD(keys=["image"]),
            ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        # No augmentation by default for large datasets
        # if is_training and self.use_augmentation:
        #     transforms.extend([
        #         RandFlipD(keys=["image"], prob=0.5, spatial_axis=1),
        #         RandRotateD(keys=["image"], range_x=15, prob=0.5),
        #         RandZoomD(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        #     ])
        
        # Resize to target size
        from monai.transforms import ResizeD
        transforms.append(ResizeD(keys=["image"], spatial_size=self.target_size))
        
        # Create Compose transform
        try:
            return Compose(transforms)
        except Exception as e:
            print(f"Warning: MONAI Compose error: {e}")
            # Fallback to standard transforms
            from .transforms import get_efficientnet_transforms
            return get_efficientnet_transforms(self.target_size, is_training)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load data with MONAI optimizations."""
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        df = df[df['exists'] == True].copy()
        
        # Create splits
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        print(f"Dataset sizes - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        
        # Create MONAI data dictionaries
        train_data = []
        for _, row in df_train.iterrows():
            train_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        val_data = []
        for _, row in df_val.iterrows():
            val_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        test_data = []
        for _, row in df_test.iterrows():
            test_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        # Create transforms
        train_transforms = self._create_monai_transforms(is_training=True)
        val_test_transforms = self._create_monai_transforms(is_training=False)
        
        # Create MONAI datasets with caching
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        val_dataset = CacheDataset(
            data=val_data,
            transform=val_test_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        test_dataset = CacheDataset(
            data=test_data,
            transform=val_test_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        # Create MONAI data loaders
        train_loader = ThreadDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = ThreadDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = ThreadDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        print(f"MONAI datasets created with cache_rate={self.cache_rate}")
        
        return train_loader, val_loader, test_loader, 3  # 3 channels for RGB
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics."""
        df = pd.read_csv(self.csv_path)
        df = df[df['exists'] == True].copy()
        
        return {
            'total_samples': len(df),
            'train_samples': len(df[df['split'] == 'train']),
            'val_samples': len(df[df['split'] == 'val']),
            'test_samples': len(df[df['split'] == 'test']),
            'overall_event_rate': df['event'].mean(),
            'train_event_rate': df[df['split'] == 'train']['event'].mean(),
            'val_event_rate': df[df['split'] == 'val']['event'].mean(),
            'test_event_rate': df[df['split'] == 'test']['event'].mean(),
            'mean_survival_time': df['tte'].mean(),
            'median_survival_time': df['tte'].median()
        }
