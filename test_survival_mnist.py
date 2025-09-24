#!/usr/bin/env python3
"""
Quick test script for SurvivalMNIST dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loaders import SurvivalMNISTDataLoader
import torch

def test_survival_mnist():
    """Test SurvivalMNIST dataset loading and basic functionality."""
    print("Testing SurvivalMNIST dataset...")
    
    # Create data loader
    loader = SurvivalMNISTDataLoader(
        batch_size=32,
        target_size=(28, 28),
        max_survival_time=100.0,
        min_survival_time=1.0,
        event_rate=0.7
    )
    
    # Load data
    train_loader, val_loader, test_loader, num_features = loader.load_data()
    
    print(f"Number of features: {num_features}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test a batch
    print("\nTesting batch loading...")
    for i, (images, (events, times)) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Events shape: {events.shape}")
        print(f"  Times shape: {times.shape}")
        print(f"  Event rate: {events.float().mean():.3f}")
        print(f"  Time range: {times.min():.2f} - {times.max():.2f}")
        print(f"  Time mean: {times.mean():.2f}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("\nSurvivalMNIST test completed successfully!")

if __name__ == "__main__":
    test_survival_mnist()
