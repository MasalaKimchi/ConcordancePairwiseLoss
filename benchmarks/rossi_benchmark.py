#!/usr/bin/env python3
"""
Rossi Dataset Benchmark using Shared Framework

This script demonstrates how to use the shared benchmark framework
for the Rossi dataset comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple

# Import framework components
from benchmark_framework import (
    BenchmarkRunner, 
    AbstractDataLoader, 
    DATASET_CONFIGS
)

# Import dataset utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from flexible_dataset import FlexibleDataset


class RossiDataLoader(AbstractDataLoader):
    """Rossi dataset loader implementation."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size  # Smaller batch size for smaller dataset
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load Rossi dataset and return dataloaders."""
        import lifelines
        
        # Load and preprocess data
        df = lifelines.datasets.load_rossi()
        
        # Handle missing values
        df = df.dropna()
        
        # Ensure proper column names
        time_col = 'week'
        event_col = 'arrest'
        
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset")
        if event_col not in df.columns:
            raise ValueError(f"Event column '{event_col}' not found in dataset")
        
        # One-hot encode categorical variables if needed
        categorical_cols = []
        for col in df.columns:
            if col not in [time_col, event_col] and df[col].dtype == 'object':
                categorical_cols.append(col)
        
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Convert to float
        feature_cols = [col for col in df.columns if col not in [time_col, event_col]]
        df[feature_cols] = df[feature_cols].astype(float)
        
        # Train/validation/test split
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)
        
        print(f"Training: {len(df_train)}, Validation: {len(df_val)}, Testing: {len(df_test)}")
        
        # Create dataloaders
        dataloader_train = DataLoader(
            FlexibleDataset(df_train, time_col=time_col, event_col=event_col), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        dataloader_val = DataLoader(
            FlexibleDataset(df_val, time_col=time_col, event_col=event_col), 
            batch_size=len(df_val), 
            shuffle=False
        )
        dataloader_test = DataLoader(
            FlexibleDataset(df_test, time_col=time_col, event_col=event_col), 
            batch_size=len(df_test), 
            shuffle=False
        )
        
        # Get number of features
        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


def main():
    """Main function to run Rossi benchmark."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = RossiDataLoader(batch_size=32)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['rossi']
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=32,
        epochs=50,
        learning_rate=5e-2
    )
    
    results = runner.run_comparison()
    return results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Rossi survival analysis benchmark')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    print(f"Running Rossi benchmark with {args.runs} run(s)")
    if args.runs > 1:
        print("Multiple runs will provide statistical robustness (mean ± std)")
    
    # Create data loader
    data_loader = RossiDataLoader()
    dataset_config = DATASET_CONFIGS['rossi']
    
    # Create runner with command line arguments
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=64,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save
    )
    
    # Run the comparison
    results = runner.run_comparison(num_runs=args.runs)
    
    print("\n" + "="*80)
    print("ROSSI BENCHMARK COMPLETE!")
    if args.runs > 1:
        print(f"Completed {args.runs} independent runs with statistical aggregation")
    print("="*80)
