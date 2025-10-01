"""
MIMIC-IV Image Dataset Module

This module provides PyTorch dataset classes for MIMIC-IV chest X-ray images
with survival analysis targets.
"""

import os
from typing import Tuple, Optional, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MIMICImageDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-IV Chest X-ray images with survival analysis targets.
    
    This dataset loads chest X-ray images from MIMIC-IV and provides them with
    corresponding survival analysis targets (event indicators and survival times).
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        data_dir: str,
        time_col: str = "tte", 
        event_col: str = "event",
        path_col: str = "path",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize MIMIC image dataset.
        
        Args:
            df: DataFrame containing image paths and survival data
            data_dir: Base directory containing the MIMIC data
            time_col: Column name for time to event
            event_col: Column name for event indicator
            path_col: Column name for image path
            transform: Optional transform to apply to images
            target_size: Target size for image resizing
        """
        self.df = df
        self.data_dir = data_dir
        self.time_col = time_col
        self.event_col = event_col
        self.path_col = path_col
        self.target_size = target_size
        
        # Set up default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transform
            
        # Verify required columns exist
        required_cols = [time_col, event_col, path_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, (event_tensor, time_tensor))
        """
        sample = self.df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, sample[self.path_col])
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
        except (OSError, IOError) as e:
            # If image loading fails, create a black image as fallback
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new('RGB', self.target_size, (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get survival targets (more efficient tensor creation)
        event = torch.tensor(sample[self.event_col], dtype=torch.bool)
        time = torch.tensor(sample[self.time_col], dtype=torch.float32)
        
        return image, (event, time)
    
    def get_survival_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all survival data as tensors.
        
        Returns:
            Tuple of (events, times) tensors
        """
        events = torch.tensor(self.df[self.event_col].values, dtype=torch.bool)
        times = torch.tensor(self.df[self.time_col].values, dtype=torch.float32)
        return events, times
    
    def get_event_rate(self) -> float:
        """
        Get the event rate (proportion of deaths) in the dataset.
        
        Returns:
            Event rate as a float between 0 and 1
        """
        return self.df[self.event_col].mean()
    
    def get_survival_stats(self) -> dict:
        """
        Get survival statistics for the dataset.
        
        Returns:
            Dictionary containing survival statistics
        """
        return {
            'n_samples': len(self.df),
            'event_rate': self.df[self.event_col].mean(),
            'mean_survival_time': self.df[self.time_col].mean(),
            'median_survival_time': self.df[self.time_col].median(),
            'min_survival_time': self.df[self.time_col].min(),
            'max_survival_time': self.df[self.time_col].max()
        }
