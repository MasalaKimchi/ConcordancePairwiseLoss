#!/usr/bin/env python3
"""
GBSG2 Dataset Benchmark using Shared Framework

This script demonstrates how to use the shared benchmark framework
for the GBSG2 dataset comparison.
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
from flexible_dataset import Custom_dataset


class GBSG2DataLoader(AbstractDataLoader):
    """GBSG2 dataset loader implementation."""
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load GBSG2 dataset and return dataloaders."""
        import lifelines
        
        # Load and preprocess data
        df = lifelines.datasets.load_gbsg2()
        categorical_cols = ["horTh", "menostat", "tgrade"]
        drop_cols = ["horTh_no", "menostat_Post", "tgrade_I"]
        
        # One-hot encoding
        df_onehot = pd.get_dummies(df, columns=categorical_cols).astype("float")
        
        # Remove reference categories
        for col in drop_cols:
            if col in df_onehot.columns:
                df_onehot.drop(col, axis=1, inplace=True)
        
        # Train/validation/test split
        df_train, df_test = train_test_split(df_onehot, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)
        
        print(f"Training: {len(df_train)}, Validation: {len(df_val)}, Testing: {len(df_test)}")
        
        # Create dataloaders
        dataloader_train = DataLoader(Custom_dataset(df_train), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(Custom_dataset(df_val), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(Custom_dataset(df_test), batch_size=len(df_test), shuffle=False)
        
        # Get number of features
        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


def main():
    """Main function to run GBSG2 benchmark."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = GBSG2DataLoader(batch_size=64)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['gbsg2']
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=64,
        epochs=50,
        learning_rate=5e-2
    )
    
    results = runner.run_comparison()
    return results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run GBSG2 survival analysis benchmark')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible results (default: None)')
    
    args = parser.parse_args()
    
    print(f"Running GBSG2 benchmark with {args.runs} run(s)")
    if args.runs > 1:
        print("Multiple runs will provide statistical robustness (mean ± std)")
    
    # Create data loader
    data_loader = GBSG2DataLoader()
    dataset_config = DATASET_CONFIGS['gbsg2']
    
    # Create runner with command line arguments
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=64,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        random_seed=args.seed
    )
    
    # Run the comparison
    results = runner.run_comparison(num_runs=args.runs)
    
    print("\n" + "="*80)
    print("GBSG2 BENCHMARK COMPLETE!")
    if args.runs > 1:
        print(f"Completed {args.runs} independent runs with statistical aggregation")
    print("="*80)
