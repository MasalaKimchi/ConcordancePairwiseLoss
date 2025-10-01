#!/usr/bin/env python3
"""
MIMIC-IV Chest X-ray Survival Analysis Benchmark

This benchmark compares core loss functions on MIMIC-IV chest X-ray dataset:
1. NLL (Negative Log-Likelihood)
2. CPL (Concordance Pairwise Loss)
3. CPL (ipcw) (CPL with IPCW weighting computed per batch)
4. CPL (ipcw batch) (CPL with IPCW weighting precomputed from full training set)

Uses EfficientNet-B0 backbone with fixed hyperparameters:
- Learning rate: 1e-4
- Weight decay: 1e-5
- Early stopping with patience of 20 epochs
- Maximum 100,000 training steps

Usage:
    # Install MONAI for optimized data loading (optional but recommended)
    pip install monai[all]
    
    # Basic usage (no augmentation by default)
    conda activate concordance-pairwise-loss
    python benchmarks/benchmark_MIMIC.py --epochs 50 --batch-size 64
    
    # With augmentation enabled
    python benchmarks/benchmark_MIMIC.py --epochs 50 --batch-size 64 --enable-augmentation
    
    # Maximum performance
    python benchmarks/benchmark_MIMIC.py --epochs 30 --batch-size 128 --cache-rate 0.3 --num-workers 12
"""

import argparse
import os
import sys

# Ensure we can import the original framework
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

# Import MIMIC components from refactored modules
from mimic.util import MIMICBenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Chest X-ray Survival Analysis Benchmark")
    parser.add_argument('--data-dir', type=str, 
                       default='Y:\\MIMIC-CXR-JPG\\mimic-cxr-jpg-2.1.0.physionet.org\\',
                       help='Path to MIMIC data directory')
    parser.add_argument('--csv-path', type=str, default='data/mimic/mimic_cxr_splits.csv',
                       help='Path to preprocessed CSV file')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (will be auto-optimized)')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--cache-rate', type=float, default=0.4, help='MONAI cache rate (0.0-1.0)')
    parser.add_argument('--use-monai', action='store_true', default=True, help='Use MONAI optimizations')
    parser.add_argument('--enable-augmentation', action='store_true', help='Enable data augmentation (disabled by default for large datasets)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of independent runs per loss type')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.csv_path):
        print(f"❌ CSV file not found: {args.csv_path}")
        print("   Run preprocessing first: python -m src.mimic.preprocess")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Please update the --data-dir path")
        return
    
    # Run benchmark
    runner = MIMICBenchmarkRunner(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_results=True,
        random_seed=args.seed,
        num_runs=args.num_runs
    )
    
    results = runner.run_comparison(args)


if __name__ == "__main__":
    main()