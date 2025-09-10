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

import numpy as np
import torch

# Import framework components
from benchmark_framework import (
    BenchmarkRunner,
    DATASET_CONFIGS
)

# Import dataset utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from datasets.whas500 import WHAS500DataLoader


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
