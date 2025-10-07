#!/usr/bin/env python3
"""
MIMIC-IV Chest X-ray Survival Analysis Benchmark V2 (with Comprehensive Evaluation)

This benchmark combines training and comprehensive evaluation in a single run:
1. Trains models with different loss functions (NLL, CPL, CPL(ipcw), CPL(ipcw batch))
2. Evaluates with comprehensive metrics during training (not just after saving)
3. Uses the same evaluation approach as evaluate_saved_models.py

Comprehensive metrics include:
- Harrell's C-index
- Uno's C-index
- Cumulative/Dynamic AUC
- Incident/Dynamic AUC
- Brier Score

Usage:
    conda activate concordance-pairwise-loss
    python benchmarks/benchmark_MIMIC_v2.py --epochs 50 --batch-size 64
    
    # Quick test with subset
    python benchmarks/benchmark_MIMIC_v2.py --epochs 5 --batch-size 32 --data-fraction 0.01
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure we can import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

# Import MIMIC components
from mimic.util import MIMICBenchmarkRunner
from mimic.preprocessed_data_loader import OptimizedPreprocessedMIMICDataLoader

# Import DatasetConfig
from dataset_configs import DatasetConfig

# Import evaluation components
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw

from tqdm import tqdm


class ComprehensiveMIMICEvaluator:
    """Comprehensive evaluator that matches evaluate_saved_models.py behavior."""
    
    def __init__(self, device: torch.device, dataset_config: DatasetConfig):
        """
        Initialize comprehensive evaluator.
        
        Args:
            device: Device to run evaluation on
            dataset_config: Dataset configuration with AUC time settings
        """
        self.device = device
        self.dataset_config = dataset_config
    
    def evaluate_model(self, model: torch.nn.Module, dataloader_test) -> Dict[str, float]:
        """
        Comprehensive evaluation matching evaluate_saved_models.py.
        
        This method replicates the _create_mimic_evaluator approach from evaluate_saved_models.py.
        """
        model.eval()
        all_log_hz = []
        all_events = []
        all_times = []
        
        print(f"    Collecting predictions from {len(dataloader_test)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_test):
                # Handle different data formats (MONAI vs standard)
                if isinstance(batch, dict):
                    # MONAI format: {"image": tensor, "event": tensor, "time": tensor}
                    x = batch["image"]
                    event = batch["event"]
                    time = batch["time"]
                else:
                    # Standard format: (image, (event, time))
                    x, (event, time) = batch
                
                # Convert MONAI MetaTensors to regular PyTorch tensors if needed
                if hasattr(x, 'as_tensor'):
                    x = x.as_tensor()
                if hasattr(event, 'as_tensor'):
                    event = event.as_tensor()
                if hasattr(time, 'as_tensor'):
                    time = time.as_tensor()
                
                x = x.to(self.device)
                
                # Get predictions
                log_hz = model(x)
                log_hz = log_hz.squeeze()  # Remove extra dimensions
                
                all_log_hz.append(log_hz.cpu())
                all_events.append(event.cpu())
                all_times.append(time.cpu())
        
        # Concatenate all batches
        log_hz = torch.cat(all_log_hz, dim=0)
        event = torch.cat(all_events, dim=0)
        time = torch.cat(all_times, dim=0)
        
        # Ensure correct data types for torchsurv requirements
        event = event.bool()  # torchsurv requires boolean events
        time = time.float()   # torchsurv requires float times
        
        print(f"    Data collected: {len(log_hz)} samples")
        print(f"    Event rate: {event.float().mean():.3f}")
        print(f"    Time range: {time.min():.1f} - {time.max():.1f} days")
        print(f"    Prediction range: {log_hz.min():.4f} - {log_hz.max():.4f}")
        
        # Get IPCW weights for Uno's C-index (same as evaluate_saved_models.py)
        if event.any():  # Only compute IPCW if there are events
            # torchsurv's get_ipcw has device issues, so we compute on CPU
            event_cpu = event.cpu()
            time_cpu = time.cpu()
            ipcw_weights = get_ipcw(event_cpu, time_cpu)
            print(f"    IPCW weights computed successfully")
        else:
            ipcw_weights = torch.ones_like(event, dtype=torch.float)
            print(f"    No events found, using unit weights")
        
        # Initialize metrics (same as evaluate_saved_models.py)
        cindex = ConcordanceIndex()
        auc = Auc()
        brier = BrierScore()
        
        # Move all tensors to CPU for torchsurv (same as evaluate_saved_models.py)
        log_hz_cpu = log_hz.cpu()
        event_cpu = event.cpu()
        time_cpu = time.cpu()
        ipcw_weights_cpu = ipcw_weights.cpu()
        
        results = {}
        
        # 1. Harrell's C-index (without IPCW weights)
        harrell_cindex = cindex(log_hz_cpu, event_cpu, time_cpu)
        results['harrell_cindex'] = harrell_cindex.item()
        print(f"    Harrell's C-index: {harrell_cindex.item():.6f}")
        
        # 2. Uno's C-index (with IPCW weights)
        uno_cindex = cindex(log_hz_cpu, event_cpu, time_cpu, weight=ipcw_weights_cpu)
        results['uno_cindex'] = uno_cindex.item()
        print(f"    Uno's C-index: {uno_cindex.item():.6f}")
        
        # 3. Cumulative AUC
        cumulative_auc_tensor = auc(log_hz_cpu, event_cpu, time_cpu)
        cumulative_auc = torch.mean(cumulative_auc_tensor)
        results['cumulative_auc'] = cumulative_auc.item()
        print(f"    Cumulative AUC: {cumulative_auc.item():.6f}")
        
        # 4. Incident AUC at specified time point
        new_time_cpu = torch.tensor(self.dataset_config.auc_time)
        incident_auc_result = auc(log_hz_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        results['incident_auc'] = incident_auc_result.item()
        print(f"    Incident AUC ({self.dataset_config.auc_time:.0f} days): {results['incident_auc']:.6f}")
        
        # 5. Brier Score
        survival_probs_1d = torch.sigmoid(-log_hz_cpu) 
        survival_probs_cpu = survival_probs_1d.unsqueeze(1)  
        
        new_time_cpu = torch.tensor(self.dataset_config.auc_time)  # Scalar tensor
        brier_score_result = brier(survival_probs_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        results['brier_score'] = brier_score_result.item()
        print(f"    Brier Score: {results['brier_score']:.6f}")
        
        return results


class PreprocessedMIMICBenchmarkRunnerV2(MIMICBenchmarkRunner):
    """Enhanced benchmark runner with comprehensive evaluation metrics."""
    
    def __init__(
        self,
        data_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
        csv_path: str = "data/mimic/mimic_cxr_splits_preprocessed.csv",
        data_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initialize preprocessed MIMIC benchmark runner with comprehensive evaluation.
        
        Args:
            data_dir: Directory containing preprocessed images
            csv_path: Path to preprocessed CSV file
            data_fraction: Fraction of data to use (1.0 = all, 0.01 = 1%)
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(data_dir=data_dir, csv_path=csv_path, **kwargs)
        self.data_fraction = data_fraction
        
        # Initialize dataset config for comprehensive evaluation
        self.dataset_config = DatasetConfig(
            name="MIMIC-IV",
            auc_time=365.0,  # 1 year in days
            auc_time_unit="days"
        )
        self.comprehensive_evaluator = ComprehensiveMIMICEvaluator(
            self.trainer.device, 
            self.dataset_config
        )
    
    def run_comparison(self, args=None) -> dict:
        """Run comparison with comprehensive evaluation."""
        print("=" * 80)
        print("MIMIC-IV CHEST X-RAY SURVIVAL ANALYSIS BENCHMARK V2")
        print("Training + Comprehensive Evaluation (Single Run)")
        print("=" * 80)
        print(f"Core Loss Functions: NLL, CPL, CPL(ipcw), CPL(ipcw batch)")
        print(f"Multiple Runs: {self.num_runs} independent runs per configuration")
        print(f"Data fraction: {self.data_fraction * 100:.1f}%")
        print(f"Comprehensive Metrics: Harrell's C, Uno's C, Cumulative AUC, Incident AUC, Brier Score")
        print("=" * 80)
        
        # Set random seed
        self._set_random_seed(self.random_seed)
        
        # Load preprocessed data
        use_augmentation = getattr(args, 'enable_augmentation', False)
        
        # Use MONAI optimized data loader for preprocessed images
        data_loader = OptimizedPreprocessedMIMICDataLoader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            use_augmentation=use_augmentation,
            cache_rate=0.4,  # Cache 40% of preprocessed data
            num_workers=12,
            pin_memory=True,
            data_fraction=self.data_fraction
        )
        print("Using MONAI optimized data loader for preprocessed images")
        
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
        
        # Test each loss type
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
            val_comprehensive_results = []
            test_comprehensive_results = []
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
                
                # Comprehensive evaluation on validation set
                print(f"  Comprehensive evaluation on validation set...")
                val_eval = self.comprehensive_evaluator.evaluate_model(model, dataloader_val)
                val_comprehensive_results.append(val_eval)
                
                # Comprehensive evaluation on test set
                print(f"  Comprehensive evaluation on test set...")
                test_eval = self.comprehensive_evaluator.evaluate_model(model, dataloader_test)
                test_comprehensive_results.append(test_eval)
                training_results_list.append(training_results)
                
                print(f"  Results Summary:")
                print(f"    Val Uno C: {val_eval.get('uno_cindex', 0):.4f}, Test Uno C: {test_eval.get('uno_cindex', 0):.4f}")
                print(f"    Val Harrell C: {val_eval.get('harrell_cindex', 0):.4f}, Test Harrell C: {test_eval.get('harrell_cindex', 0):.4f}")
                print(f"    Test Incident AUC: {test_eval.get('incident_auc', 0):.4f}")
                print(f"    Test Brier Score: {test_eval.get('brier_score', 0):.4f}")
                
                # Save model weights if this is the best validation Uno C-index so far
                val_uno = val_eval.get('uno_cindex', 0)
                best_val_uno = max([r.get('uno_cindex', 0) for r in val_comprehensive_results])
                if val_uno == best_val_uno:
                    self._save_best_model(
                        model, loss_type, run_idx, 
                        val_eval, test_eval, training_results
                    )
                
                # Update run progress bar
                run_pbar.set_postfix({
                    'Val Uno C': f'{val_uno:.4f}',
                    'Test Uno C': f'{test_eval.get("uno_cindex", 0):.4f}',
                    'Epochs': training_results['total_epochs']
                })
            
            # Store results for this loss type
            results[loss_type] = {
                'training_results': training_results_list,
                'val_evaluations': val_comprehensive_results,
                'test_evaluations': test_comprehensive_results,
                'best_val_uno': max([r.get('uno_cindex', 0) for r in val_comprehensive_results]),
                'mean_test_uno': np.mean([r.get('uno_cindex', 0) for r in test_comprehensive_results]),
                'mean_test_harrell': np.mean([r.get('harrell_cindex', 0) for r in test_comprehensive_results]),
                'mean_test_incident_auc': np.mean([r.get('incident_auc', 0) for r in test_comprehensive_results]),
                'mean_test_brier': np.mean([r.get('brier_score', 0) for r in test_comprehensive_results])
            }
            
            print(f"\n{loss_type.upper()} Summary:")
            print(f"  Best validation Uno C-index: {results[loss_type]['best_val_uno']:.4f}")
            print(f"  Mean test Uno C-index: {results[loss_type]['mean_test_uno']:.4f}")
            print(f"  Mean test Harrell C-index: {results[loss_type]['mean_test_harrell']:.4f}")
            print(f"  Mean test Incident AUC: {results[loss_type]['mean_test_incident_auc']:.4f}")
            print(f"  Mean test Brier Score: {results[loss_type]['mean_test_brier']:.4f}")
        
        # Save comprehensive results
        self._save_comprehensive_results(results, args)
        
        # Print final summary
        self._print_comprehensive_summary(results)
        
        return results
    
    def _save_best_model(
        self, model, loss_type, run_idx, 
        val_eval: Dict, test_eval: Dict, training_results: Dict
    ):
        """Save the best model weights with comprehensive metrics."""
        import os
        from datetime import datetime
        
        # Create results directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.output_dir, "models", f"{loss_type}_best")
        os.makedirs(model_dir, exist_ok=True)
        
        # Extract key metrics
        val_uno = val_eval.get('uno_cindex', 0)
        test_uno = test_eval.get('uno_cindex', 0)
        
        # Model filename with metrics
        model_filename = f"{loss_type}_best_run{run_idx+1}_valUno{val_uno:.4f}_testUno{test_uno:.4f}_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save model state dict and metadata
        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'loss_type': loss_type,
            'run_idx': run_idx,
            'val_evaluation': val_eval,
            'test_evaluation': test_eval,
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
        
        # Also save as "latest best" for this loss type
        latest_path = os.path.join(model_dir, f"{loss_type}_latest_best.pth")
        torch.save(model_save_dict, latest_path)
        
        print(f"  ✅ Saved best model: {model_path}")
        print(f"     Val Uno C: {val_uno:.4f}, Test Uno C: {test_uno:.4f}")
        
        # Create a comprehensive summary file
        summary_path = os.path.join(model_dir, f"{loss_type}_best_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Best Model Summary for {loss_type.upper()}\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Loss Type: {loss_type}\n")
            f.write(f"Run: {run_idx + 1}\n\n")
            
            f.write(f"Validation Metrics:\n")
            f.write(f"  Uno's C-index: {val_eval.get('uno_cindex', 'N/A'):.6f}\n")
            f.write(f"  Harrell's C-index: {val_eval.get('harrell_cindex', 'N/A'):.6f}\n")
            f.write(f"  Cumulative AUC: {val_eval.get('cumulative_auc', 'N/A'):.6f}\n")
            f.write(f"  Incident AUC: {val_eval.get('incident_auc', 'N/A'):.6f}\n")
            f.write(f"  Brier Score: {val_eval.get('brier_score', 'N/A'):.6f}\n\n")
            
            f.write(f"Test Metrics:\n")
            f.write(f"  Uno's C-index: {test_eval.get('uno_cindex', 'N/A'):.6f}\n")
            f.write(f"  Harrell's C-index: {test_eval.get('harrell_cindex', 'N/A'):.6f}\n")
            f.write(f"  Cumulative AUC: {test_eval.get('cumulative_auc', 'N/A'):.6f}\n")
            f.write(f"  Incident AUC: {test_eval.get('incident_auc', 'N/A'):.6f}\n")
            f.write(f"  Brier Score: {test_eval.get('brier_score', 'N/A'):.6f}\n\n")
            
            f.write(f"Model Details:\n")
            f.write(f"  Architecture: EfficientNet-B0\n")
            f.write(f"  Data Fraction: {self.data_fraction * 100:.1f}%\n")
            f.write(f"  Training Epochs: {training_results.get('total_epochs', 'N/A')}\n")
            f.write(f"  Final Training Loss: {training_results.get('train_losses', [0])[-1]:.6f}\n")
            f.write(f"  Best Validation Loss: {training_results.get('best_val_loss', 'N/A'):.6f}\n")
            f.write(f"  Timestamp: {timestamp}\n\n")
            
            f.write(f"Hyperparameters:\n")
            f.write(f"  Epochs: {self.epochs}\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  Weight Decay: {self.weight_decay}\n")
    
    def _save_comprehensive_results(self, results: Dict, args):
        """Save comprehensive results to CSV and JSON files."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive evaluation directory
        eval_dir = os.path.join(self.output_dir, "comprehensive_evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Method name mapping (from benchmark_tabular_v2.py)
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        # 1. Save summary CSV with mean metrics
        summary_data = []
        for loss_type, result in results.items():
            method_name = method_names.get(loss_type, loss_type.upper())
            
            # Calculate means and stds across runs
            test_evals = result['test_evaluations']
            
            summary_data.append({
                'Method': method_name,
                'Loss Type': loss_type,
                'N Runs': len(test_evals),
                "Harrell's C (mean)": np.mean([e.get('harrell_cindex', np.nan) for e in test_evals]),
                "Harrell's C (std)": np.std([e.get('harrell_cindex', np.nan) for e in test_evals]),
                "Uno's C (mean)": np.mean([e.get('uno_cindex', np.nan) for e in test_evals]),
                "Uno's C (std)": np.std([e.get('uno_cindex', np.nan) for e in test_evals]),
                'Cumulative AUC (mean)': np.mean([e.get('cumulative_auc', np.nan) for e in test_evals]),
                'Cumulative AUC (std)': np.std([e.get('cumulative_auc', np.nan) for e in test_evals]),
                'Incident AUC (mean)': np.mean([e.get('incident_auc', np.nan) for e in test_evals]),
                'Incident AUC (std)': np.std([e.get('incident_auc', np.nan) for e in test_evals]),
                'Brier Score (mean)': np.mean([e.get('brier_score', np.nan) for e in test_evals]),
                'Brier Score (std)': np.std([e.get('brier_score', np.nan) for e in test_evals])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(eval_dir, f"benchmark_v2_summary_{timestamp}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n✅ Saved summary CSV: {summary_csv_path}")
        
        # 2. Save detailed results JSON
        json_results = {}
        for loss_type, result in results.items():
            # Convert numpy/torch types for JSON serialization
            json_result = {
                'test_evaluations': [],
                'val_evaluations': [],
                'best_val_uno': float(result['best_val_uno']),
                'mean_test_uno': float(result['mean_test_uno']),
                'mean_test_harrell': float(result['mean_test_harrell']),
                'mean_test_incident_auc': float(result['mean_test_incident_auc']),
                'mean_test_brier': float(result['mean_test_brier'])
            }
            
            # Convert evaluation results
            for eval_dict in result['test_evaluations']:
                json_eval = {}
                for k, v in eval_dict.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        json_eval[k] = v.tolist()
                    elif isinstance(v, (np.floating, np.integer)):
                        json_eval[k] = float(v)
                    else:
                        json_eval[k] = v
                json_result['test_evaluations'].append(json_eval)
            
            for eval_dict in result['val_evaluations']:
                json_eval = {}
                for k, v in eval_dict.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        json_eval[k] = v.tolist()
                    elif isinstance(v, (np.floating, np.integer)):
                        json_eval[k] = float(v)
                    else:
                        json_eval[k] = v
                json_result['val_evaluations'].append(json_eval)
            
            json_results[loss_type] = json_result
        
        json_path = os.path.join(eval_dir, f"benchmark_v2_detailed_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ Saved detailed JSON: {json_path}")
        
        # 3. Save per-run detailed CSV
        detailed_data = []
        for loss_type, result in results.items():
            method_name = method_names.get(loss_type, loss_type.upper())
            for run_idx, test_eval in enumerate(result['test_evaluations']):
                detailed_data.append({
                    'method': method_name,
                    'loss_type': loss_type,
                    'run': run_idx + 1,
                    'harrell_cindex': test_eval.get('harrell_cindex', np.nan),
                    'uno_cindex': test_eval.get('uno_cindex', np.nan),
                    'cumulative_auc': test_eval.get('cumulative_auc', np.nan),
                    'incident_auc': test_eval.get('incident_auc', np.nan),
                    'brier_score': test_eval.get('brier_score', np.nan)
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(eval_dir, f"benchmark_v2_per_run_{timestamp}.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"✅ Saved per-run CSV: {detailed_csv_path}")
    
    def _print_comprehensive_summary(self, results: Dict):
        """Print comprehensive summary of all results."""
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'=' * 80}")
        
        # Method name mapping
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        print(f"\nTest Set Results (Mean ± Std across {self.num_runs} runs):")
        print("-" * 120)
        print(f"{'Method':<20} {'Uno C':<18} {'Harrell C':<18} {'Incident AUC':<18} {'Brier Score':<18}")
        print("-" * 120)
        
        for loss_type, result in results.items():
            method_name = method_names.get(loss_type, loss_type.upper())
            test_evals = result['test_evaluations']
            
            uno_mean = np.mean([e.get('uno_cindex', np.nan) for e in test_evals])
            uno_std = np.std([e.get('uno_cindex', np.nan) for e in test_evals])
            
            harrell_mean = np.mean([e.get('harrell_cindex', np.nan) for e in test_evals])
            harrell_std = np.std([e.get('harrell_cindex', np.nan) for e in test_evals])
            
            incident_mean = np.mean([e.get('incident_auc', np.nan) for e in test_evals])
            incident_std = np.std([e.get('incident_auc', np.nan) for e in test_evals])
            
            brier_mean = np.mean([e.get('brier_score', np.nan) for e in test_evals])
            brier_std = np.std([e.get('brier_score', np.nan) for e in test_evals])
            
            print(f"{method_name:<20} {uno_mean:.4f}±{uno_std:.4f}    {harrell_mean:.4f}±{harrell_std:.4f}    {incident_mean:.4f}±{incident_std:.4f}    {brier_mean:.4f}±{brier_std:.4f}")
        
        print("-" * 120)
        
        # Find best methods for each metric
        print(f"\n🏆 Best Mean Performance by Metric:")
        
        # Uno's C-index (higher is better)
        best_uno_loss = max(results.keys(), key=lambda k: results[k]['mean_test_uno'])
        print(f"  Uno's C-index: {method_names[best_uno_loss]} ({results[best_uno_loss]['mean_test_uno']:.6f})")
        
        # Harrell's C-index (higher is better)
        best_harrell_loss = max(results.keys(), key=lambda k: results[k]['mean_test_harrell'])
        print(f"  Harrell's C-index: {method_names[best_harrell_loss]} ({results[best_harrell_loss]['mean_test_harrell']:.6f})")
        
        # Incident AUC (higher is better)
        best_auc_loss = max(results.keys(), key=lambda k: results[k]['mean_test_incident_auc'])
        print(f"  Incident AUC: {method_names[best_auc_loss]} ({results[best_auc_loss]['mean_test_incident_auc']:.6f})")
        
        # Brier Score (lower is better)
        best_brier_loss = min(results.keys(), key=lambda k: results[k]['mean_test_brier'])
        print(f"  Brier Score: {method_names[best_brier_loss]} ({results[best_brier_loss]['mean_test_brier']:.6f})")
        
        print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV Benchmark V2: Training + Comprehensive Evaluation"
    )
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
    parser.add_argument('--output-dir', type=str, default='results-batch-64', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of independent runs per loss type')
    
    args = parser.parse_args()
    
    # Check if preprocessed data exists
    if not os.path.exists(args.csv_path):
        print(f"❌ Preprocessed CSV file not found: {args.csv_path}")
        print("   Run preprocessing first:")
        print("   python src/mimic/preprocess_images.py --limit 1000  # Test with 1000 images")
        print("   python src/mimic/preprocess_images.py              # Process all images")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Preprocessed data directory not found: {args.data_dir}")
        print("   Run preprocessing first:")
        print("   python src/mimic/preprocess_images.py")
        return 1
    
    # Validate data fraction
    if not 0 < args.data_fraction <= 1.0:
        print(f"❌ Invalid data fraction: {args.data_fraction}")
        print("   Data fraction must be between 0 and 1.0")
        return 1
    
    # Run benchmark
    runner = PreprocessedMIMICBenchmarkRunnerV2(
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
    
    try:
        results = runner.run_comparison(args)
        
        print(f"\n{'=' * 80}")
        print("BENCHMARK V2 COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")
        print(f"Results saved to: {args.output_dir}")
        print(f"  - Models: {args.output_dir}/models/")
        print(f"  - Comprehensive evaluation: {args.output_dir}/comprehensive_evaluation/")
        
        return 0
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print("BENCHMARK FAILED")
        print(f"{'=' * 80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

