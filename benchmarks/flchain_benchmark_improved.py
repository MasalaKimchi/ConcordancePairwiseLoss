#!/usr/bin/env python3
"""
FLChain Dataset Benchmark using Improved Framework

This script demonstrates how to use the improved benchmark framework
for the FLChain dataset comparison with horizon-weighted losses.
The FLChain dataset contains serum free light chain measurements 
for 7,874 subjects with 2,169 recorded deaths.

Uses horizon_kind="exp", hetero_tau=True, and use_uncertainty_weighting=True
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

# Import improved framework components
from benchmark_framework_improved import (
    BenchmarkRunnerImproved,
    DATASET_CONFIGS
)

# Import dataset utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from datasets.flchain import FLChainDataLoader


def main():
    """Main function to run FLChain benchmark with improved framework."""
    # Set random seeds for reproducibility
    seed = int(os.environ.get('PYTHONHASHSEED', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    data_loader = FLChainDataLoader(batch_size=64)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS['flchain']
    
    print("=" * 80)
    print(f"COMPREHENSIVE SURVIVAL ANALYSIS COMPARISON - {dataset_config.name.upper()} DATASET")
    print("Using Improved Framework with Horizon-Weighted Losses")
    print("=" * 80)
    
    # Define all loss combinations with horizon_kind="exp"
    loss_types = [
        'nll',
        'pairwise', 
        'pairwise_ipcw',
        'normalized_combination',
        'normalized_combination_ipcw', 
        'cphl_exp',
        'hybrid_exp'
    ]
    
    print(f"Parameters: horizon_kind='exp', hetero_tau=True, use_uncertainty_weighting=True")
    print(f"Loss types: {loss_types}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create and run benchmark with improved parameters
    runner = BenchmarkRunnerImproved(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=64,
        epochs=50,
        learning_rate=5e-2,
        horizon_kind="exp",
        hetero_tau=True,
        use_uncertainty_weighting=True
    )
    
    results = runner.run_comparison(loss_types=loss_types)
    return results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FLChain survival analysis benchmark with improved framework')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate (default: 5e-2)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--rel-factor', type=float, default=0.5, help='Relative factor for horizon loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for loss scaling (default: 1.0)')
    parser.add_argument('--loss-types', nargs='+', 
                        default=['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw', 'cphl_exp', 'hybrid_exp'],
                        help='Loss types to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"COMPREHENSIVE SURVIVAL ANALYSIS COMPARISON - {dataset_config.name.upper()} DATASET")
    print("Using Improved Framework with Horizon-Weighted Losses")
    print("=" * 80)
    print(f"Parameters: horizon_kind='exp', hetero_tau=True, use_uncertainty_weighting=True")
    print(f"Loss types: {args.loss_types}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    print(f"Rel factor: {args.rel_factor}, Temperature: {args.temperature}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create data loader
    data_loader = FLChainDataLoader()
    dataset_config = DATASET_CONFIGS['flchain']
    
    # Create runner with command line arguments and improved parameters
    runner = BenchmarkRunnerImproved(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=64,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        horizon_kind="exp",
        hetero_tau=True,
        rel_factor=args.rel_factor,
        temperature=args.temperature,
        use_uncertainty_weighting=True
    )
    
    # Run the comparison
    results = runner.run_comparison(loss_types=args.loss_types)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Print comprehensive summary table
    print(f"\n{'Method':<25} {'Harrell C':<10} {'Uno C':<10} {'Cum AUC':<10} {'Inc AUC':<10} {'Brier':<10}")
    print("-" * 85)
    
    # Method name mapping for display [[memory:8493814]]
    method_names = {
        'nll': 'NLL',
        'pairwise': 'CPL',
        'pairwise_ipcw': 'CPL (ipcw)',
        'normalized_combination': 'NLL+CPL',
        'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
        'cphl_exp': 'CPL (horizon)',
        'hybrid_exp': 'Hybrid (exp)'
    }
    
    for loss_type, result in results.items():
        eval_result = result['evaluation']
        method_name = method_names.get(loss_type, loss_type.upper())
        
        harrell = eval_result['harrell_cindex']
        uno = eval_result['uno_cindex']
        cumulative_auc = eval_result['cumulative_auc']
        incident_auc = eval_result['incident_auc']
        brier = eval_result['brier_score']
        
        print(f"{method_name:<25} {harrell:<10.4f} {uno:<10.4f} {cumulative_auc:<10.4f} {incident_auc:<10.4f} {brier:<10.4f}")
    
    print(f"\n{'='*80}")
    print("FLCHAIN IMPROVED BENCHMARK COMPLETE!")
    print("Used horizon_kind='exp', hetero_tau=True, use_uncertainty_weighting=True")
    print(f"{'='*80}")
