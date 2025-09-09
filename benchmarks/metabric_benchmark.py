#!/usr/bin/env python3
"""
METABRIC Dataset Benchmark using Shared Framework

This script demonstrates how to use the shared benchmark framework
for the METABRIC dataset comparison. METABRIC is a breast cancer genomics
dataset with molecular subtypes and clinical variables from pycox.
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


class METABRICDataLoader(AbstractDataLoader):
    """METABRIC dataset loader implementation using pycox."""
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load METABRIC dataset and return dataloaders."""
        from pycox.datasets import metabric
        
        # Load METABRIC dataset from pycox
        df = metabric.read_df()
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # METABRIC dataset typically has these columns:
        # - time: survival time in days
        # - event: death indicator (1=death, 0=censored)
        # - Various clinical and molecular features
        
        # METABRIC dataset from pycox has 'duration' and 'event' columns
        # Rename 'duration' to 'time' for consistency with our framework
        if 'duration' in df.columns:
            df = df.rename(columns={'duration': 'time'})
        
        # Ensure we have the required columns
        if 'time' not in df.columns or 'event' not in df.columns:
            raise ValueError(f"Expected 'time' and 'event' columns not found. Available columns: {df.columns.tolist()}")
        
        # Handle missing values
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Identify categorical and continuous columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        continuous_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove time and event from feature columns
        feature_categorical_cols = [col for col in categorical_cols if col not in ['time', 'event']]
        feature_continuous_cols = [col for col in continuous_cols if col not in ['time', 'event']]
        
        print(f"Categorical features: {feature_categorical_cols}")
        print(f"Continuous features: {feature_continuous_cols}")
        
        # Process categorical variables
        label_encoders = {}
        for col in feature_categorical_cols:
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
        
        # Process continuous variables
        for col in feature_continuous_cols:
            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        # Ensure event column is binary (0/1)
        if df['event'].min() < 0 or df['event'].max() > 1:
            # If events are coded differently, convert to 0/1
            unique_events = sorted(df['event'].unique())
            if len(unique_events) == 2:
                df['event'] = (df['event'] == unique_events[1]).astype(int)
            else:
                # For multiple event types, convert to binary (any event vs censored)
                df['event'] = (df['event'] > 0).astype(int)
        
        # Ensure time is positive
        df = df[df['time'] > 0]
        
        # Standardize continuous features
        if len(feature_continuous_cols) > 0:
            scaler = StandardScaler()
            df[feature_continuous_cols] = scaler.fit_transform(df[feature_continuous_cols])
        
        # Train/validation/test split with stratification
        df_train, df_test = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df['event']
        )
        df_train, df_val = train_test_split(
            df_train, test_size=0.3, random_state=42, stratify=df_train['event']
        )
        
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
        print(f"Feature names: {[col for col in df.columns if col not in ['time', 'event']]}")
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


def main():
    """Main function to run METABRIC benchmark."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = METABRICDataLoader(batch_size=64)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['metabric']
    
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
    parser = argparse.ArgumentParser(description='Run METABRIC survival analysis benchmark')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible results (default: None)')
    
    args = parser.parse_args()
    
    print(f"Running METABRIC benchmark with {args.runs} run(s)")
    if args.runs > 1:
        print("Multiple runs will provide statistical robustness (mean ± std)")
    
    # Create data loader
    data_loader = METABRICDataLoader()
    dataset_config = DATASET_CONFIGS['metabric']
    
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
    print("METABRIC BENCHMARK COMPLETE!")
    if args.runs > 1:
        print(f"Completed {args.runs} independent runs with statistical aggregation")
    print("="*80)
