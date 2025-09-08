#!/usr/bin/env python3
"""
WHAS500 Dataset Benchmark using Shared Framework

This script demonstrates how to use the shared benchmark framework
for the WHAS500 dataset comparison. The WHAS500 dataset comprises 500 samples 
from the Worcester Heart Attack Study with 14 features, focusing on 
post-myocardial infarction survival.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


class WHAS500DataLoader(AbstractDataLoader):
    """WHAS500 dataset loader implementation."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size  # Smaller batch size for smaller dataset
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load WHAS500 dataset and return dataloaders."""
        from sksurv.datasets import load_whas500
        
        # Load dataset
        X, y = load_whas500()
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        
        # Extract time and event from structured array
        df['time'] = y['lenfol']  # follow-up time
        df['event'] = y['fstat'].astype(int)  # event indicator
        
        # Handle missing values for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        # Standardize features (excluding time and event)
        feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Train/validation/test split
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        
        print(f"Training: {len(df_train)}, Validation: {len(df_val)}, Testing: {len(df_test)}")
        print(f"Event rate - Train: {df_train['event'].mean():.3f}, Val: {df_val['event'].mean():.3f}, Test: {df_test['event'].mean():.3f}")
        
        # Create dataloaders
        dataloader_train = DataLoader(
            FlexibleDataset(df_train, time_col='time', event_col='event'), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        dataloader_val = DataLoader(
            FlexibleDataset(df_val, time_col='time', event_col='event'), 
            batch_size=len(df_val), 
            shuffle=False
        )
        dataloader_test = DataLoader(
            FlexibleDataset(df_test, time_col='time', event_col='event'), 
            batch_size=len(df_test), 
            shuffle=False
        )
        
        # Get number of features
        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)
        
        print(f"Number of features: {num_features}")
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


def main():
    """Main function to run WHAS500 benchmark."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = WHAS500DataLoader(batch_size=32)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['whas500']
    
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
    parser = argparse.ArgumentParser(description='Run WHAS500 survival analysis benchmark')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    print(f"Running WHAS500 benchmark with {args.runs} run(s)")
    if args.runs > 1:
        print("Multiple runs will provide statistical robustness (mean ± std)")
    
    # Create data loader
    data_loader = WHAS500DataLoader()
    dataset_config = DATASET_CONFIGS['whas500']
    
    # Create runner with command line arguments
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=32,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save
    )
    
    # Run the comparison
    results = runner.run_comparison(num_runs=args.runs)
    
    print("\n" + "="*80)
    print("WHAS500 BENCHMARK COMPLETE!")
    if args.runs > 1:
        print(f"Completed {args.runs} independent runs with statistical aggregation")
    print("="*80)
