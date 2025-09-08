#!/usr/bin/env python3
"""
FLChain Dataset Benchmark using Shared Framework

This script demonstrates how to use the shared benchmark framework
for the FLChain dataset comparison. The FLChain dataset contains serum 
free light chain measurements for 7,874 subjects with 2,169 recorded deaths.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


class FLChainDataLoader(AbstractDataLoader):
    """FLChain dataset loader implementation."""
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load FLChain dataset and return dataloaders."""
        from sksurv.datasets import load_flchain
        
        # Load dataset
        X, y = load_flchain()
        
        # Convert structured array to DataFrame
        df = pd.DataFrame(X)
        
        # Extract time and event from structured array
        df['time'] = y['futime']
        df['event'] = y['death'].astype(int)
        
        # Ensure proper 0/1 encoding for events
        if df['event'].min() != 0 or df['event'].max() != 1:
            # Convert to binary encoding if needed
            df['event'] = (df['event'] > 0).astype(int)
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode and encode
        # Include both 'object' and 'category' dtypes
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            if col not in ['time', 'event']:
                # Fill missing values with mode
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna('unknown')
                
                # Label encode categorical variables
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        # Ensure all columns except time and event are numeric
        feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values introduced by conversion
        df = df.dropna()
        
        # Standardize all feature columns (excluding time and event)
        numerical_feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        
        if len(numerical_feature_cols) > 0:
            scaler = StandardScaler()
            df[numerical_feature_cols] = scaler.fit_transform(df[numerical_feature_cols])
        
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
    """Main function to run FLChain benchmark."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = FLChainDataLoader(batch_size=64)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['flchain']
    
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
    parser = argparse.ArgumentParser(description='Run FLChain survival analysis benchmark')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    print(f"Running FLChain benchmark with {args.runs} run(s)")
    if args.runs > 1:
        print("Multiple runs will provide statistical robustness (mean ± std)")
    
    # Create data loader
    data_loader = FLChainDataLoader()
    dataset_config = DATASET_CONFIGS['flchain']
    
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
    print("FLCHAIN BENCHMARK COMPLETE!")
    if args.runs > 1:
        print(f"Completed {args.runs} independent runs with statistical aggregation")
    print("="*80)
