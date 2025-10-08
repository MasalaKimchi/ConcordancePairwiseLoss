"""
SurvivalMNIST Dataset Implementation

A simple synthetic dataset based on MNIST digits adapted for survival analysis.
This is perfect for quick prototyping and testing imaging survival methods.

Dataset Configuration:
- Default censoring rate: 30% (0.3)
- Censoring applied reproducibly using sample index as seed
- Censoring times randomly drawn between 1 and true survival time
- Creates realistic survival data where IPCW weights are meaningful
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


class TorchSurvMNISTDataset(Dataset):
    """
    TorchSurv-compatible MNIST dataset where digit value becomes survival time.
    
    This follows the exact pattern from the TorchSurv tutorial:
    - Digit 0 becomes time=10 (to prevent log(0))
    - Digits 1-9 become time=1-9
    - Introduces censoring to make IPCW meaningful
    
    Censoring Mechanism (Default: 30%):
    - Reproducible censoring using sample index as random seed
    - Censoring times drawn uniformly from [1, survival_time]
    - Creates realistic survival data where IPCW is necessary
    """
    
    def __init__(
        self, 
        root: str = './data',
        train: bool = True,
        download: bool = True,
        transform: Optional[transforms.Compose] = None,
        censoring_rate: float = 0.3
    ):
        """
        Initialize TorchSurv MNIST dataset.
        
        Args:
            root: Root directory for MNIST data
            train: Whether to use training set
            download: Whether to download MNIST if not present
            transform: Image transforms to apply
            censoring_rate: Fraction of samples to censor (0.0 = no censoring, 1.0 = all censored)
        """
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=root, 
            train=train, 
            download=download,
            transform=transform
        )
        
        # Store censoring rate
        self.censoring_rate = censoring_rate
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """Get image and survival data following TorchSurv pattern with censoring."""
        image, digit = self.mnist[idx]
        
        # Convert digit to survival time (0 -> 10, 1-9 -> 1-9)
        # digit is already an integer from MNIST dataset
        survival_time = 10 if digit == 0 else digit
        
        # Introduce censoring based on censoring_rate
        # Use a deterministic approach based on idx to ensure reproducibility
        import random
        random.seed(idx)  # Ensure reproducible censoring
        is_censored = random.random() < self.censoring_rate
        
        if is_censored:
            # Censor the sample: event=False, time=observed_time (could be < survival_time)
            # For simplicity, use a random censoring time between 1 and survival_time
            censoring_time = random.uniform(1.0, float(survival_time))
            event = False
            observed_time = censoring_time
        else:
            # No censoring: event=True, time=survival_time
            event = True
            observed_time = survival_time
        
        # Convert to proper tensor types
        observed_time = torch.tensor(observed_time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.bool)
        
        return image, (event, observed_time)
    
    def get_survival_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all survival data as tensors."""
        # Convert all digits to survival times
        times = []
        for i in range(len(self.mnist)):
            digit = self.mnist.targets[i].item()
            survival_time = 10 if digit == 0 else digit
            times.append(survival_time)
        
        events = torch.ones(len(self.mnist), dtype=torch.bool)
        times = torch.tensor(times, dtype=torch.float32)
        
        return events, times


def get_survival_mnist_transforms(
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard transforms for SurvivalMNIST matching TorchSurv example.
    
    Args:
        target_size: Target image size (default 224x224 for ResNet)
        normalize: Whether to normalize images
    
    Returns:
        Composed transforms
    """
    from torchvision.transforms import v2
    
    transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(target_size, antialias=True),
    ]
    
    if normalize:
        # TorchSurv normalization (mean=0, std=1)
        transform_list.append(
            v2.Normalize(mean=(0,), std=(1,))
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


def create_torchsurv_mnist_loaders(
    batch_size: int = 128,
    root: str = './data',
    target_size: Tuple[int, int] = (224, 224),
    censoring_rate: float = 0.3
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Create train and test data loaders for TorchSurv MNIST.
    
    Args:
        batch_size: Batch size for data loaders
        root: Root directory for MNIST data
        target_size: Target image size (height, width)
        censoring_rate: Fraction of samples to censor (0.0 = no censoring, 1.0 = all censored)
    
    Returns:
        Tuple of (train_loader, test_loader, num_features)
    """
    from torch.utils.data import DataLoader
    
    transform = get_survival_mnist_transforms(target_size=target_size)
    
    # Create datasets
    train_dataset = TorchSurvMNISTDataset(
        root=root,
        train=True,
        transform=transform,
        censoring_rate=censoring_rate
    )
    
    test_dataset = TorchSurvMNISTDataset(
        root=root,
        train=False,
        transform=transform,
        censoring_rate=censoring_rate
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