"""
SurvivalMNIST Dataset Implementation

A simple synthetic dataset based on MNIST digits adapted for survival analysis.
This is perfect for quick prototyping and testing imaging survival methods.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional


class SurvivalMNISTDataset(Dataset):
    """
    SurvivalMNIST dataset - MNIST digits adapted for survival analysis.
    
    Each digit class is assigned a different "survival risk" profile:
    - Digits 0-2: Low risk (longer survival)
    - Digits 3-5: Medium risk 
    - Digits 6-9: High risk (shorter survival)
    
    Survival times are generated based on digit class and some randomness.
    """
    
    def __init__(
        self, 
        root: str = './data',
        train: bool = True,
        download: bool = True,
        transform: Optional[transforms.Compose] = None,
        max_survival_time: float = 100.0,
        min_survival_time: float = 1.0,
        event_rate: float = 0.7
    ):
        """
        Initialize SurvivalMNIST dataset.
        
        Args:
            root: Root directory for MNIST data
            train: Whether to use training set
            download: Whether to download MNIST if not present
            transform: Image transforms to apply
            max_survival_time: Maximum survival time
            min_survival_time: Minimum survival time
            event_rate: Probability of having an event (not censored)
        """
        self.max_survival_time = max_survival_time
        self.min_survival_time = min_survival_time
        self.event_rate = event_rate
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=root, 
            train=train, 
            download=download,
            transform=transform
        )
        
        # Generate survival data based on digit classes
        self._generate_survival_data()
    
    def _generate_survival_data(self):
        """Generate survival times and events based on digit classes."""
        n_samples = len(self.mnist)
        
        # Define risk levels for each digit (0-9)
        # Lower numbers = lower risk = longer survival
        risk_levels = {
            0: 0.1, 1: 0.2, 2: 0.3,  # Low risk
            3: 0.5, 4: 0.6, 5: 0.7,  # Medium risk  
            6: 0.8, 7: 0.9, 8: 0.95, 9: 1.0  # High risk
        }
        
        # Generate survival times
        survival_times = []
        events = []
        
        for i in range(n_samples):
            digit_class = self.mnist.targets[i].item()
            risk = risk_levels[digit_class]
            
            # Generate survival time based on risk level
            # Higher risk = shorter survival time
            base_time = self.min_survival_time + (self.max_survival_time - self.min_survival_time) * (1 - risk)
            
            # Add some randomness
            noise = np.random.exponential(scale=base_time * 0.3)
            survival_time = base_time + noise
            
            # Clip to valid range
            survival_time = np.clip(survival_time, self.min_survival_time, self.max_survival_time)
            
            # Determine if event occurred (not censored)
            event = np.random.random() < self.event_rate
            
            survival_times.append(survival_time)
            events.append(event)
        
        self.survival_times = torch.tensor(survival_times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.bool)
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """Get image and survival data."""
        image, _ = self.mnist[idx]  # We don't need the original label
        survival_time = self.survival_times[idx]
        event = self.events[idx]
        
        return image, (event, survival_time)
    
    def get_survival_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all survival data as tensors."""
        return self.events, self.survival_times


def get_survival_mnist_transforms(
    target_size: Tuple[int, int] = (28, 28),
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard transforms for SurvivalMNIST.
    
    Args:
        target_size: Target image size
        normalize: Whether to normalize images
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        # MNIST normalization
        transform_list.append(
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        )
    
    return transforms.Compose(transform_list)


def create_survival_mnist_loaders(
    batch_size: int = 128,
    root: str = './data',
    target_size: Tuple[int, int] = (28, 28),
    max_survival_time: float = 100.0,
    min_survival_time: float = 1.0,
    event_rate: float = 0.7
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Create train and test data loaders for SurvivalMNIST.
    
    Returns:
        Tuple of (train_loader, test_loader, num_features)
    """
    from torch.utils.data import DataLoader
    
    transform = get_survival_mnist_transforms(target_size=target_size)
    
    # Create datasets
    train_dataset = SurvivalMNISTDataset(
        root=root,
        train=True,
        transform=transform,
        max_survival_time=max_survival_time,
        min_survival_time=min_survival_time,
        event_rate=event_rate
    )
    
    test_dataset = SurvivalMNISTDataset(
        root=root,
        train=False,
        transform=transform,
        max_survival_time=max_survival_time,
        min_survival_time=min_survival_time,
        event_rate=event_rate
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, 1  # 1 channel for grayscale


if __name__ == "__main__":
    # Quick test
    train_loader, test_loader, num_features = create_survival_mnist_loaders(batch_size=32)
    
    print(f"Number of features: {num_features}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, (events, times) in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Events shape: {events.shape}")
        print(f"Times shape: {times.shape}")
        print(f"Event rate: {events.float().mean():.3f}")
        print(f"Time range: {times.min():.2f} - {times.max():.2f}")
        break
