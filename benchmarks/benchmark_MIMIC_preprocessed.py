#!/usr/bin/env python3
"""
MIMIC-IV Chest X-ray Survival Analysis Benchmark (Preprocessed Images)

This benchmark compares core loss functions on preprocessed MIMIC-IV chest X-ray dataset:
1. NLL (Negative Log-Likelihood)
2. CPL (Concordance Pairwise Loss)
3. CPL (ipcw) (CPL with IPCW weighting computed per batch)
4. CPL (ipcw batch) (CPL with IPCW weighting precomputed from full training set)

Uses EfficientNet-B0 backbone with preprocessed RGB images (224x224).
Preprocessed images eliminate expensive on-the-fly transforms for faster training.

Usage:
    # First, preprocess images (one-time setup)
    python src/mimic/preprocess_images.py --limit 1000  # Test with 1000 images
    python src/mimic/preprocess_images.py              # Process all images
    
    # Then run benchmark
    conda activate concordance-pairwise-loss
    python benchmarks/benchmark_MIMIC_preprocessed.py --epochs 50 --batch-size 64
    
    # With data fraction for testing
    python benchmarks/benchmark_MIMIC_preprocessed.py --epochs 5 --batch-size 32 --data-fraction 0.01
"""

import argparse
import os
import sys
import torch

# Ensure we can import the original framework
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

# Import MIMIC components from preprocessed modules
from mimic.util import MIMICBenchmarkRunner
from mimic.preprocessed_data_loader import PreprocessedMIMICDataLoader, OptimizedPreprocessedMIMICDataLoader


class PreprocessedMIMICBenchmarkRunner(MIMICBenchmarkRunner):
    """Benchmark runner for preprocessed MIMIC images."""
    
    def __init__(
        self,
        data_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
        csv_path: str = "data/mimic/mimic_cxr_splits_preprocessed.csv",
        data_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initialize preprocessed MIMIC benchmark runner.
        
        Args:
            data_dir: Directory containing preprocessed images
            csv_path: Path to preprocessed CSV file
            data_fraction: Fraction of data to use (1.0 = all, 0.01 = 1%)
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(data_dir=data_dir, csv_path=csv_path, **kwargs)
        self.data_fraction = data_fraction
    
    def run_comparison(self, args=None) -> dict:
        """Run comparison with preprocessed images."""
        print("=" * 80)
        print("MIMIC-IV CHEST X-RAY SURVIVAL ANALYSIS COMPARISON (PREPROCESSED)")
        print("Core Loss Functions: NLL, CPL, CPL(ipcw), CPL(ipcw batch)")
        print(f"Multiple Runs: {self.num_runs} independent runs per configuration")
        print(f"Data fraction: {self.data_fraction * 100:.1f}%")
        print("=" * 80)
        
        # Set random seed
        self._set_random_seed(self.random_seed)
        
        # Load preprocessed data
        use_augmentation = getattr(args, 'enable_augmentation', False)
        
        # Try MONAI optimized loader first, fallback to standard
        try:
            from monai.utils import set_determinism
            data_loader = OptimizedPreprocessedMIMICDataLoader(
                batch_size=self.batch_size,
                data_dir=self.data_dir,
                csv_path=self.csv_path,
                use_augmentation=use_augmentation,
                cache_rate=0.4,  # Cache 40% of preprocessed data (~50 GB RAM)
                num_workers=12,  # Conservative setting for i9-12900KF (24 threads)
                pin_memory=True,
                data_fraction=self.data_fraction
            )
            print("Using MONAI optimized data loader for preprocessed images")
        except ImportError:
            data_loader = PreprocessedMIMICDataLoader(
                batch_size=self.batch_size,
                data_dir=self.data_dir,
                csv_path=self.csv_path,
                use_augmentation=use_augmentation,
                num_workers=12,  # Conservative setting for i9-12900KF (24 threads)
                pin_memory=True,
                data_fraction=self.data_fraction
            )
            print("Using standard data loader for preprocessed images")
        
        dataloader_train, dataloader_val, dataloader_test, num_features = data_loader.load_data()
        
        # Update dataset config with actual values
        stats = data_loader.get_dataset_stats()
        self.dataset_stats = stats
        
        print(f"Full dataset available: {stats['total_samples']:,} preprocessed samples")
        print(f"Full dataset Train/Val/Test: {stats['train_samples']:,}/{stats['val_samples']:,}/{stats['test_samples']:,}")
        
        if 'used_total_samples' in stats:
            print(f"Using subset ({stats['data_fraction']*100:.1f}%): {stats['used_total_samples']:,} samples")
            print(f"Subset Train/Val/Test: {stats['used_train_samples']:,}/{stats['used_val_samples']:,}/{stats['used_test_samples']:,}")
        else:
            print(f"Using full dataset: {stats['total_samples']:,} samples")
        
        print(f"Event rate: {stats['overall_event_rate']:.3f}")
        print(f"Data augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
        print(f"Images: Preprocessed RGB 224x224 (no on-the-fly transforms)")
        
        # Store data loaders for parent class usage
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test
        
        # Test each loss type (matching the names in util.py)
        loss_types = ['nll', 'cpl', 'cpl_ipcw', 'cpl_ipcw_batch']
        
        results = {}
        
        from tqdm import tqdm
        loss_pbar = tqdm(enumerate(loss_types), total=len(loss_types), desc="Testing loss functions")
        
        for loss_idx, loss_type in loss_pbar:
            loss_pbar.set_description(f"Testing {loss_type.upper()}")
            
            print(f"\n{'=' * 50}")
            print(f"TESTING {loss_type.upper()} ({loss_idx + 1}/{len(loss_types)})")
            print(f"{'=' * 50}")
            
            # Run multiple times for this loss type
            val_uno_scores = []
            test_eval_results = []
            training_results_list = []
            
            run_pbar = tqdm(range(self.num_runs), desc=f"Runs for {loss_type.upper()}", leave=False)
            
            for run_idx in run_pbar:
                print(f"\n  Run {run_idx+1}/{self.num_runs}...")
                print(f"  {'='*50}")
                
                # Set different random seed for each run
                run_seed = (self.random_seed + run_idx) if self.random_seed is not None else (42 + run_idx)
                self._set_random_seed(run_seed)
                
                # Create and train model
                model = self.trainer.create_model()
                training_results = self.trainer.train_model(
                    model, dataloader_train, dataloader_val, loss_type, temperature=1.0
                )
                
                # Evaluate on validation and test sets
                print(f"  Evaluating model on validation set...")
                val_eval = self.evaluator.evaluate_model(model, dataloader_val)
                val_uno = float(val_eval['uno_cindex'])
                val_uno_scores.append(val_uno)
                
                print(f"  Evaluating model on test set...")
                test_eval = self.evaluator.evaluate_model(model, dataloader_test)
                test_eval_results.append(test_eval)
                training_results_list.append(training_results)
                
                print(f"  Results - Val C-index: {val_uno:.4f}, Test C-index: {test_eval['uno_cindex']:.4f}")
                
                # Save model weights if this is the best validation C-index so far for this loss type
                if val_uno == max(val_uno_scores):  # This is the best run so far
                    self._save_best_model(model, loss_type, run_idx, val_uno, test_eval['uno_cindex'], training_results)
                
                # Update run progress bar
                run_pbar.set_postfix({
                    'Val C': f'{val_uno:.4f}',
                    'Test C': f'{test_eval["uno_cindex"]:.4f}',
                    'Epochs': training_results['total_epochs']
                })
            
            # Store results for this loss type
            results[loss_type] = {
                'training_results': training_results_list,
                'val_evaluations': [{'uno_cindex': score} for score in val_uno_scores],
                'test_evaluations': test_eval_results,
                'best_val_uno': max(val_uno_scores),
                'mean_test_uno': sum([r['uno_cindex'] for r in test_eval_results]) / len(test_eval_results)
            }
            
            print(f"\n{loss_type.upper()} Summary:")
            print(f"  Best validation C-index: {max(val_uno_scores):.4f}")
            print(f"  Mean test C-index: {results[loss_type]['mean_test_uno']:.4f}")
        
        # Save results if requested (use parent class method)
        if self.save_results and hasattr(super(), '_save_results'):
            try:
                super()._save_results(results, args)
            except Exception as e:
                print(f"Note: Could not save additional results: {e}")
                print("Models were saved successfully though!")
        
        # Print final summary of saved models
        self._print_model_summary(results)
        
        return results
    
    def _save_best_model(self, model, loss_type, run_idx, val_cindex, test_cindex, training_results):
        """Save the best model weights based on validation C-index."""
        import os
        from datetime import datetime
        
        # Create results directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.output_dir, "models", f"{loss_type}_best")
        os.makedirs(model_dir, exist_ok=True)
        
        # Model filename with metrics
        model_filename = f"{loss_type}_best_run{run_idx+1}_val{val_cindex:.4f}_test{test_cindex:.4f}_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save model state dict and metadata
        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'loss_type': loss_type,
            'run_idx': run_idx,
            'val_cindex': val_cindex,
            'test_cindex': test_cindex,
            'training_results': training_results,
            'model_architecture': 'EfficientNet-B0',
            'data_fraction': self.data_fraction,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'timestamp': timestamp
        }
        
        torch.save(model_save_dict, model_path)
        
        # Also save a symlink/copy as the "latest best" for this loss type
        latest_path = os.path.join(model_dir, f"{loss_type}_latest_best.pth")
        torch.save(model_save_dict, latest_path)
        
        print(f"  ✅ Saved best model: {model_path}")
        print(f"     Val C-index: {val_cindex:.4f}, Test C-index: {test_cindex:.4f}")
        
        # Create a summary file for this loss type
        summary_path = os.path.join(model_dir, f"{loss_type}_best_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Best Model Summary for {loss_type.upper()}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Loss Type: {loss_type}\n")
            f.write(f"Run: {run_idx + 1}\n")
            f.write(f"Validation C-index: {val_cindex:.6f}\n")
            f.write(f"Test C-index: {test_cindex:.6f}\n")
            f.write(f"Architecture: EfficientNet-B0\n")
            f.write(f"Data Fraction: {self.data_fraction * 100:.1f}%\n")
            f.write(f"Training Epochs: {training_results.get('total_epochs', 'N/A')}\n")
            f.write(f"Final Training Loss: {training_results.get('train_losses', [0])[-1]:.6f}\n")
            f.write(f"Best Validation Loss: {training_results.get('best_val_loss', 'N/A'):.6f}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"\nHyperparameters:\n")
            f.write(f"  Epochs: {self.epochs}\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  Weight Decay: {self.weight_decay}\n")
    
    def _print_model_summary(self, results):
        """Print summary of all saved models."""
        print(f"\n{'=' * 80}")
        print("SAVED MODEL SUMMARY")
        print(f"{'=' * 80}")
        
        models_dir = os.path.join(self.output_dir, "models")
        if os.path.exists(models_dir):
            print(f"Models saved in: {models_dir}")
            print()
            
            for loss_type in results.keys():
                model_subdir = os.path.join(models_dir, f"{loss_type}_best")
                if os.path.exists(model_subdir):
                    latest_model = os.path.join(model_subdir, f"{loss_type}_latest_best.pth")
                    summary_file = os.path.join(model_subdir, f"{loss_type}_best_summary.txt")
                    
                    if os.path.exists(latest_model):
                        # Load model info
                        try:
                            model_info = torch.load(latest_model, map_location='cpu')
                            val_c = model_info.get('val_cindex', 0)
                            test_c = model_info.get('test_cindex', 0)
                            
                            print(f"📁 {loss_type.upper()}:")
                            print(f"   Best Model: {latest_model}")
                            print(f"   Validation C-index: {val_c:.6f}")
                            print(f"   Test C-index: {test_c:.6f}")
                            print(f"   Summary: {summary_file}")
                            print()
                        except Exception as e:
                            print(f"❌ Error loading {latest_model}: {e}")
        else:
            print("No models were saved.")
        
        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Chest X-ray Survival Analysis Benchmark (Preprocessed)")
    parser.add_argument('--data-dir', type=str, 
                       default='Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr',
                       help='Path to preprocessed MIMIC data directory')
    parser.add_argument('--csv-path', type=str, default='data/mimic/mimic_cxr_splits_preprocessed.csv',
                       help='Path to preprocessed CSV file')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                       help='Fraction of data to use (1.0 = all, 0.01 = 1%)')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of data loading workers')
    parser.add_argument('--enable-augmentation', action='store_true', help='Enable data augmentation')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of independent runs per loss type')
    
    args = parser.parse_args()
    
    # Check if preprocessed data exists
    if not os.path.exists(args.csv_path):
        print(f"❌ Preprocessed CSV file not found: {args.csv_path}")
        print("   Run preprocessing first:")
        print("   python src/mimic/preprocess_images.py --limit 1000  # Test with 1000 images")
        print("   python src/mimic/preprocess_images.py              # Process all images")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Preprocessed data directory not found: {args.data_dir}")
        print("   Run preprocessing first:")
        print("   python src/mimic/preprocess_images.py")
        return
    
    # Validate data fraction
    if not 0 < args.data_fraction <= 1.0:
        print(f"❌ Invalid data fraction: {args.data_fraction}")
        print("   Data fraction must be between 0 and 1.0")
        return
    
    # Run benchmark
    runner = PreprocessedMIMICBenchmarkRunner(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        data_fraction=args.data_fraction,
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
    
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETED")
    print(f"{'=' * 80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
