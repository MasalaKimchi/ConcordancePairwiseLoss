"""
MIMIC-IV Preprocessed Data Loader Module

This module provides data loading utilities for preprocessed MIMIC-IV chest X-ray data.
The preprocessed images are already converted to RGB and resized to 224x224.
"""

import os
import warnings
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .dataset import MIMICImageDataset
from abstract_data_loader import AbstractDataLoader

# Suppress MONAI deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# MONAI imports for optimized data loading (optional)
try:
    from monai.data import CacheDataset, ThreadDataLoader, DataLoader as MonaiDataLoader
    from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityD
    from monai.utils import set_determinism
    MONAI_AVAILABLE = True
    set_determinism(seed=42, use_deterministic_algorithms=False)
except ImportError:
    MONAI_AVAILABLE = False


class PreprocessedMIMICDataLoader(AbstractDataLoader):
    """
    Data loader for preprocessed MIMIC-IV chest X-ray images.
    
    This loader is optimized for preprocessed images that are already:
    - Converted to RGB format
    - Resized to target size (224x224)
    - Saved as JPEG files
    
    This eliminates the need for expensive transforms during training.
    """
    
    def __init__(
        self, 
        batch_size: int = 128,
        data_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
        csv_path: str = "data/mimic/mimic_cxr_splits_preprocessed.csv",
        use_augmentation: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        data_fraction: float = 1.0  # Fraction of data to use (for testing)
    ):
        """
        Initialize preprocessed MIMIC data loader.
        
        Args:
            batch_size: Batch size for data loaders
            data_dir: Directory containing preprocessed MIMIC images
            csv_path: Path to preprocessed CSV file
            use_augmentation: Whether to use data augmentation for training
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            data_fraction: Fraction of data to use (1.0 = all data, 0.01 = 1%)
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.use_augmentation = use_augmentation
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_fraction = data_fraction

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Load preprocessed MIMIC-IV Chest X-ray dataset.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_features)
            For preprocessed RGB images, num_features = 3
        """
        # Load preprocessed data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Preprocessed CSV not found at {self.csv_path}. "
                "Please run preprocessing first: python src/mimic/preprocess_images.py"
            )
        
        df = pd.read_csv(self.csv_path)
        
        # Filter out images that don't exist (should be already filtered in preprocessing)
        if 'preprocessed_exists' in df.columns:
            df = df[df['preprocessed_exists'] == True].copy()
        
        # Split data
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        # Apply data fraction for testing
        if self.data_fraction < 1.0:
            train_size = max(1, int(len(df_train) * self.data_fraction))
            val_size = max(1, int(len(df_val) * self.data_fraction))
            test_size = max(1, int(len(df_test) * self.data_fraction))
            
            df_train = df_train.sample(n=train_size, random_state=42).reset_index(drop=True)
            df_val = df_val.sample(n=val_size, random_state=42).reset_index(drop=True)
            df_test = df_test.sample(n=test_size, random_state=42).reset_index(drop=True)
            
            print(f"Dataset sizes ({self.data_fraction*100:.1f}% subset) - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        else:
            print(f"Dataset sizes (full) - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        
        print(f"Event rates - Train: {df_train['event'].mean():.3f}, Val: {df_val['event'].mean():.3f}, Test: {df_test['event'].mean():.3f}")
        
        # Create transforms for preprocessed images (minimal transforms needed)
        train_transform = self._get_preprocessed_transforms(is_training=self.use_augmentation)
        val_test_transform = self._get_preprocessed_transforms(is_training=False)
        
        # Create datasets
        train_dataset = MIMICImageDataset(
            df_train, 
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='preprocessed_path',  # Use preprocessed path column
            transform=train_transform,
            target_size=(224, 224)  # Already preprocessed to this size
        )
        
        val_dataset = MIMICImageDataset(
            df_val, 
            self.data_dir,
            time_col='tte',
            event_col='event', 
            path_col='preprocessed_path',
            transform=val_test_transform,
            target_size=(224, 224)
        )
        
        test_dataset = MIMICImageDataset(
            df_test, 
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='preprocessed_path', 
            transform=val_test_transform,
            target_size=(224, 224)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        # Return 3 for RGB channels
        return train_loader, val_loader, test_loader, 3
    
    def _get_preprocessed_transforms(self, is_training: bool = True):
        """
        Get transforms for preprocessed images.
        
        Since images are already RGB and resized, we only need:
        1. Convert to tensor
        2. Normalize
        3. Optional augmentation for training
        
        Args:
            is_training: Whether to apply training augmentations
            
        Returns:
            Composed transforms
        """
        transform_list = []
        
        # Add training augmentations (lighter since images are already processed)
        if is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # Reduced rotation for medical images
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        
        # Essential transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        return transforms.Compose(transform_list)
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics."""
        if not os.path.exists(self.csv_path):
            return {}
        
        df = pd.read_csv(self.csv_path)
        
        if 'preprocessed_exists' in df.columns:
            df = df[df['preprocessed_exists'] == True].copy()
        
        # Get stats for the full dataset (preprocessing processes all images)
        full_stats = {
            'total_samples': len(df),
            'train_samples': len(df[df['split'] == 'train']),
            'val_samples': len(df[df['split'] == 'val']),
            'test_samples': len(df[df['split'] == 'test']),
            'overall_event_rate': df['event'].mean(),
            'train_event_rate': df[df['split'] == 'train']['event'].mean(),
            'val_event_rate': df[df['split'] == 'val']['event'].mean(),
            'test_event_rate': df[df['split'] == 'test']['event'].mean(),
        }
        
        # If using data fraction, also calculate stats for the subset being used
        if self.data_fraction < 1.0:
            df_train_subset = df[df['split'] == 'train'].sample(n=max(1, int(len(df[df['split'] == 'train']) * self.data_fraction)), random_state=42)
            df_val_subset = df[df['split'] == 'val'].sample(n=max(1, int(len(df[df['split'] == 'val']) * self.data_fraction)), random_state=42)
            df_test_subset = df[df['split'] == 'test'].sample(n=max(1, int(len(df[df['split'] == 'test']) * self.data_fraction)), random_state=42)
            df_subset = pd.concat([df_train_subset, df_val_subset, df_test_subset])
            
            # Add subset stats (what's actually being used for training)
            full_stats.update({
                'used_total_samples': len(df_subset),
                'used_train_samples': len(df_train_subset),
                'used_val_samples': len(df_val_subset),
                'used_test_samples': len(df_test_subset),
                'data_fraction': self.data_fraction
            })
        
        return full_stats


class OptimizedPreprocessedMIMICDataLoader:
    """MONAI-optimized data loader for preprocessed MIMIC images."""
    
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
        csv_path: str = "data/mimic/mimic_cxr_splits_preprocessed.csv",
        use_augmentation: bool = True,
        cache_rate: float = 0.4,  # Cache 40% of preprocessed data in memory
        num_workers: int = 12,
        pin_memory: bool = True,
        data_fraction: float = 1.0
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.use_augmentation = use_augmentation
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_fraction = data_fraction
        
    def _create_monai_transforms(self, is_training: bool = True):
        """Create MONAI transforms for preprocessed images."""
        if not MONAI_AVAILABLE:
            # Fallback to standard loader
            standard_loader = PreprocessedMIMICDataLoader(
                batch_size=self.batch_size,
                data_dir=self.data_dir,
                csv_path=self.csv_path,
                use_augmentation=self.use_augmentation,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                data_fraction=self.data_fraction
            )
            return standard_loader._get_preprocessed_transforms(is_training)
        
        # MONAI transforms for preprocessed images (no RGB conversion needed)
        transforms = [
            LoadImageD(keys=["image"]),  # Load as-is (already RGB and resized)
            EnsureChannelFirstD(keys=["image"]),
            ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        # Create Compose transform
        try:
            return Compose(transforms)
        except Exception as e:
            print(f"Warning: MONAI Compose error: {e}")
            # Fallback to standard transforms
            standard_loader = PreprocessedMIMICDataLoader()
            return standard_loader._get_preprocessed_transforms(is_training)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load data with MONAI optimizations for preprocessed images."""
        if not MONAI_AVAILABLE:
            # Fallback to standard loader
            standard_loader = PreprocessedMIMICDataLoader(
                batch_size=self.batch_size,
                data_dir=self.data_dir,
                csv_path=self.csv_path,
                use_augmentation=self.use_augmentation,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                data_fraction=self.data_fraction
            )
            return standard_loader.load_data()
        
        # Load CSV data
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        
        if 'preprocessed_exists' in df.columns:
            df = df[df['preprocessed_exists'] == True].copy()
        
        # Create splits
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        # Apply data fraction
        if self.data_fraction < 1.0:
            train_size = max(1, int(len(df_train) * self.data_fraction))
            val_size = max(1, int(len(df_val) * self.data_fraction))
            test_size = max(1, int(len(df_test) * self.data_fraction))
            
            df_train = df_train.sample(n=train_size, random_state=42).reset_index(drop=True)
            df_val = df_val.sample(n=val_size, random_state=42).reset_index(drop=True)
            df_test = df_test.sample(n=test_size, random_state=42).reset_index(drop=True)
            
            print(f"Dataset sizes ({self.data_fraction*100:.1f}% subset) - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        else:
            print(f"Dataset sizes (full) - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        
        # Create MONAI data dictionaries
        train_data = []
        for _, row in df_train.iterrows():
            train_data.append({
                "image": os.path.join(self.data_dir, row['preprocessed_path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        val_data = []
        for _, row in df_val.iterrows():
            val_data.append({
                "image": os.path.join(self.data_dir, row['preprocessed_path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        test_data = []
        for _, row in df_test.iterrows():
            test_data.append({
                "image": os.path.join(self.data_dir, row['preprocessed_path']),
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
            num_workers=0,  # ThreadDataLoader handles threading
            pin_memory=self.pin_memory,
            buffer_size=2,
            buffer_timeout=30
        )
        
        val_loader = ThreadDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            buffer_size=2,
            buffer_timeout=30
        )
        
        test_loader = ThreadDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            buffer_size=2,
            buffer_timeout=30
        )
        
        return train_loader, val_loader, test_loader, 3
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics."""
        standard_loader = PreprocessedMIMICDataLoader(
            csv_path=self.csv_path,
            data_fraction=self.data_fraction
        )
        return standard_loader.get_dataset_stats()
