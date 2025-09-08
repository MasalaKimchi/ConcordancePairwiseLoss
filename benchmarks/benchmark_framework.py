#!/usr/bin/env python3
"""
Shared Benchmark Framework for ConcordancePairwiseLoss

This module provides reusable components for benchmarking different loss functions
across various survival analysis datasets.
"""

import warnings
warnings.filterwarnings("ignore")

import time as time_module
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import seaborn as sns
from abc import ABC, abstractmethod
import json
import csv
from datetime import datetime
import os
import sys

# TorchSurv imports
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw

# Local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from concordance_pairwise_loss import ConcordancePairwiseLoss, NormalizedLossCombination


class DatasetConfig:
    """Configuration for different datasets."""
    
    def __init__(self, 
                 name: str,
                 auc_time: float,
                 auc_time_unit: str = "days"):
        self.name = name
        self.auc_time = auc_time
        self.auc_time_unit = auc_time_unit


# Predefined dataset configurations
DATASET_CONFIGS = {
    'gbsg2': DatasetConfig('GBSG2', 1825.0, 'days'),  # 5 years
    'lung': DatasetConfig('Lung', 365.0, 'days'),     # 1 year
    'rossi': DatasetConfig('Rossi', 365.0, 'days'),   # 1 year
    'flchain': DatasetConfig('FLChain', 1825.0, 'days'),  # 5 years (similar to GBSG2)
    'whas500': DatasetConfig('WHAS500', 365.0, 'days'),   # 1 year (cardiac event)
    'veterans': DatasetConfig('Veterans', 365.0, 'days'), # 1 year (lung cancer)
    'breast_cancer': DatasetConfig('Breast Cancer', 1825.0, 'days'),  # 5 years
    'support2': DatasetConfig('SUPPORT2', 180.0, 'days'), # 6 months (critically ill patients)
    'cancer': DatasetConfig('Cancer', 365.0, 'days'),     # 1 year (general cancer dataset)
}


class AbstractDataLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load and return train, val, test dataloaders and number of features."""
        pass


class BenchmarkEvaluator:
    """Shared evaluation logic for all benchmarks."""
    
    def __init__(self, device: torch.device, dataset_config: DatasetConfig):
        self.device = device
        self.dataset_config = dataset_config
    
    def evaluate_model(self, model: torch.nn.Module, dataloader_test: DataLoader) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        model.eval()
        with torch.no_grad():
            x, (event, time) = next(iter(dataloader_test))
            x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
            log_hz = model(x)
            
            # Get IPCW weights for Uno's C-index
            try:
                if event.any():  # Only compute IPCW if there are events
                    ipcw_weights = get_ipcw(event, time)
                else:
                    ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
            except Exception:
                ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
            
            # Initialize metrics
            cindex = ConcordanceIndex()
            auc = Auc()
            brier = BrierScore()
            
            # 1. Harrell's C-index (without IPCW weights)
            harrell_cindex = cindex(log_hz, event, time)
            
            # 2. Uno's C-index (with IPCW weights)
            uno_cindex = cindex(log_hz, event, time, weight=ipcw_weights)
            
            # 3. AUC at specified time point
            new_time = torch.tensor(self.dataset_config.auc_time, device=self.device)
            auc_score = auc(log_hz, event, time, new_time=new_time)
            
            # 4. Brier Score at specified time point
            try:
                # Convert log hazards to survival probabilities
                # S(t) = exp(-H(t)) where H(t) = exp(log_hz) * t for exponential model
                # For simplicity, use sigmoid transformation to get probabilities
                survival_probs = torch.sigmoid(-log_hz)  # Higher risk -> lower survival probability
                brier_score = brier(survival_probs, event, time, new_time=new_time)
            except Exception as e:
                # If Brier score fails, set to NaN
                brier_score = torch.tensor(float('nan'))
            
            return {
                'harrell_cindex': harrell_cindex.item(),
                'uno_cindex': uno_cindex.item(),
                'auc': auc_score.item(),
                'brier_score': brier_score.item()
            }


class BenchmarkTrainer:
    """Shared training logic for all benchmarks."""
    
    def __init__(self, device: torch.device, epochs: int = 50, learning_rate: float = 5e-2):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def create_model(self, num_features: int) -> torch.nn.Module:
        """Create standard model architecture."""
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            # Modified to single layer
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(64, 1),
        ).to(self.device)
    
    def train_model(self, 
                   model: torch.nn.Module,
                   dataloader_train: DataLoader, 
                   dataloader_val: DataLoader,
                   loss_type: str) -> Dict[str, any]:
        """Train model with specified loss type."""
        print(f"\n=== Training {loss_type.upper()} ===")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        train_losses = []
        val_losses = []
        
        # Initialize normalized loss combination
        loss_combiner = None
        if loss_type in ['normalized_combination', 'normalized_combination_ipcw']:
            loss_combiner = NormalizedLossCombination(total_epochs=self.epochs)
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            epoch_total_loss = 0.0
            
            for batch in dataloader_train:
                x, (event, time) = batch
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                
                optimizer.zero_grad()
                log_hz = model(x)
                
                # Compute individual losses
                nll_loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
                
                # Pairwise loss without IPCW
                pairwise_loss_fn = ConcordancePairwiseLoss(
                    reduction="mean",
                    temp_scaling='linear',
                    pairwise_sampling='balanced',
                    use_ipcw=False
                )
                pairwise_loss = pairwise_loss_fn(log_hz, time, event)
                
                # Pairwise loss with IPCW
                pairwise_loss_ipcw_fn = ConcordancePairwiseLoss(
                    reduction="mean",
                    temp_scaling='linear',
                    pairwise_sampling='balanced',
                    use_ipcw=True
                )
                pairwise_loss_ipcw = pairwise_loss_ipcw_fn(log_hz, time, event)
                
                # Get total loss based on type
                if loss_type == "nll":
                    total_loss = nll_loss
                elif loss_type == "pairwise":
                    total_loss = pairwise_loss
                elif loss_type == "pairwise_ipcw":
                    total_loss = pairwise_loss_ipcw
                elif loss_type == "normalized_combination":
                    nll_w, pairwise_w = loss_combiner.get_weights_scale_balanced(
                        epoch, nll_loss.item(), pairwise_loss.item()
                    )
                    total_loss = nll_w * nll_loss + pairwise_w * pairwise_loss
                elif loss_type == "normalized_combination_ipcw":
                    nll_w, pairwise_w = loss_combiner.get_weights_scale_balanced(
                        epoch, nll_loss.item(), pairwise_loss_ipcw.item()
                    )
                    total_loss = nll_w * nll_loss + pairwise_w * pairwise_loss_ipcw
                else:
                    total_loss = nll_loss
                
                total_loss.backward()
                optimizer.step()
                epoch_total_loss += total_loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                x_val, (event_val, time_val) = next(iter(dataloader_val))
                x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                log_hz_val = model(x_val)
                
                nll_val = neg_partial_log_likelihood(log_hz_val, event_val, time_val, reduction="mean")
                
                pairwise_val_fn = ConcordancePairwiseLoss(
                    reduction="mean",
                    temp_scaling='linear',
                    pairwise_sampling='balanced',
                    use_ipcw=False
                )
                pairwise_val = pairwise_val_fn(log_hz_val, time_val, event_val)
                
                pairwise_val_ipcw_fn = ConcordancePairwiseLoss(
                    reduction="mean",
                    temp_scaling='linear',
                    pairwise_sampling='balanced',
                    use_ipcw=True
                )
                pairwise_val_ipcw = pairwise_val_ipcw_fn(log_hz_val, time_val, event_val)
                
                if loss_type == "normalized_combination" and loss_combiner:
                    nll_w_val, pairwise_w_val = loss_combiner.get_weights_scale_balanced(
                        epoch, nll_val.item(), pairwise_val.item()
                    )
                    val_loss = nll_w_val * nll_val + pairwise_w_val * pairwise_val
                elif loss_type == "normalized_combination_ipcw" and loss_combiner:
                    nll_w_val, pairwise_w_val = loss_combiner.get_weights_scale_balanced(
                        epoch, nll_val.item(), pairwise_val_ipcw.item()
                    )
                    val_loss = nll_w_val * nll_val + pairwise_w_val * pairwise_val_ipcw
                elif loss_type == "pairwise_ipcw":
                    val_loss = pairwise_val_ipcw
                else:
                    val_loss = nll_val + pairwise_val
            
            # Store results
            train_losses.append(epoch_total_loss)
            val_losses.append(val_loss.item())
            
            if self.epochs >= 10 and epoch % (self.epochs // 10) == 0:
                if loss_type in ["normalized_combination", "normalized_combination_ipcw"] and loss_combiner and loss_combiner.weight_history:
                    nll_w, pairwise_w = loss_combiner.weight_history[-1]
                    print(f"Epoch {epoch:03d}: Total={epoch_total_loss:.4f}, Weights=({nll_w:.3f}, {pairwise_w:.3f})")
                else:
                    print(f"Epoch {epoch:03d}: Total={epoch_total_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weight_evolution': loss_combiner.weight_history if loss_combiner else []
        }


class ResultsLogger:
    """Handles saving benchmark results to files with timestamps."""
    
    def __init__(self, dataset_name: str, output_dir: str = "results"):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Base filename with timestamp
        self.base_filename = f"{dataset_name}_benchmark_{self.timestamp}"
    
    def save_results(self, results: Dict[str, Dict], run_info: Dict = None, execution_time: float = None) -> Dict[str, str]:
        """Save streamlined results - only comprehensive JSON and summary CSV."""
        saved_files = {}
        
        # 1. Save single comprehensive JSON file with everything
        comprehensive_json = os.path.join(self.output_dir, f"{self.base_filename}_comprehensive.json")
        self._save_comprehensive_json(results, comprehensive_json, run_info, execution_time)
        saved_files['comprehensive'] = comprehensive_json
        
        # 2. Save summary CSV table (nice to have for quick analysis)
        csv_file = os.path.join(self.output_dir, f"{self.base_filename}_summary.csv")
        self._save_csv_summary(results, csv_file)
        saved_files['summary'] = csv_file
        
        return saved_files
    
    def _save_comprehensive_json(self, results: Dict[str, Dict], filepath: str, run_info: Dict = None, execution_time: float = None):
        """Save single comprehensive JSON file with all experiment data."""
        output_data = {
            'experiment_summary': {
                'dataset': self.dataset_name,
                'timestamp': self.timestamp,
                'datetime': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'execution_time_formatted': f"{execution_time/60:.1f} minutes" if execution_time else None,
                **(run_info or {})
            },
            'performance_summary': {},
            'improvement_analysis': {},
            'detailed_results': {},
            'hyperparameters': run_info or {},
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'device': run_info.get('device', 'unknown') if run_info else 'unknown'
            }
        }
        
        # Performance summary table
        method_names = {
            'nll': 'NLL',
            'pairwise': 'Pairwise',
            'pairwise_ipcw': 'Pairwise_IPCW',
            'normalized_combination': 'Normalized_Combo',
            'normalized_combination_ipcw': 'Normalized_IPCW'
        }
        
        for method, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(method, method)
            
            output_data['performance_summary'][method_name] = {
                'harrell_cindex': round(eval_result['harrell_cindex'], 4),
                'uno_cindex': round(eval_result['uno_cindex'], 4),
                'auc': round(eval_result['auc'], 4),
                'brier_score': round(eval_result['brier_score'], 4) if not np.isnan(eval_result['brier_score']) else None
            }
        
        # Improvement analysis
        if 'nll' in results:
            nll_eval = results['nll']['evaluation']
            for method, result in results.items():
                if method == 'nll':
                    continue
                
                eval_result = result['evaluation']
                method_name = method_names.get(method, method)
                
                harrell_imp = ((eval_result['harrell_cindex'] - nll_eval['harrell_cindex']) / nll_eval['harrell_cindex'] * 100)
                uno_imp = ((eval_result['uno_cindex'] - nll_eval['uno_cindex']) / nll_eval['uno_cindex'] * 100)
                auc_imp = ((eval_result['auc'] - nll_eval['auc']) / nll_eval['auc'] * 100)
                
                if not (np.isnan(nll_eval['brier_score']) or np.isnan(eval_result['brier_score'])):
                    brier_imp = ((nll_eval['brier_score'] - eval_result['brier_score']) / nll_eval['brier_score'] * 100)
                else:
                    brier_imp = None
                
                output_data['improvement_analysis'][method_name] = {
                    'harrell_improvement_percent': round(harrell_imp, 2),
                    'uno_improvement_percent': round(uno_imp, 2),
                    'auc_improvement_percent': round(auc_imp, 2),
                    'brier_improvement_percent': round(brier_imp, 2) if brier_imp is not None else None
                }
        
        # Detailed results (same as before)
        for method, result in results.items():
            output_data['detailed_results'][method] = {
                'evaluation': {k: float(v) if hasattr(v, 'item') else v 
                             for k, v in result['evaluation'].items()},
                'training': {
                    'train_losses': [float(x) for x in result['training']['train_losses']],
                    'val_losses': [float(x) for x in result['training']['val_losses']],
                    'weight_evolution': result['training']['weight_evolution']
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_json_results(self, results: Dict[str, Dict], filepath: str, run_info: Dict = None):
        """Save complete results as JSON."""
        output_data = {
            'experiment_info': {
                'dataset': self.dataset_name,
                'timestamp': self.timestamp,
                'datetime': datetime.now().isoformat(),
                **(run_info or {})
            },
            'results': {}
        }
        
        # Convert torch tensors to floats for JSON serialization
        for method, result in results.items():
            output_data['results'][method] = {
                'evaluation': {k: float(v) if hasattr(v, 'item') else v 
                             for k, v in result['evaluation'].items()},
                'training': {
                    'train_losses': [float(x) for x in result['training']['train_losses']],
                    'val_losses': [float(x) for x in result['training']['val_losses']],
                    'weight_evolution': result['training']['weight_evolution']
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_csv_summary(self, results: Dict[str, Dict], filepath: str):
        """Save summary metrics as CSV."""
        method_names = {
            'nll': 'NLL',
            'pairwise': 'Pairwise',
            'pairwise_ipcw': 'Pairwise_IPCW',
            'normalized_combination': 'Normalized_Combo',
            'normalized_combination_ipcw': 'Normalized_IPCW'
        }
        
        rows = []
        for method, result in results.items():
            eval_result = result['evaluation']
            row = {
                'Method': method_names.get(method, method),
                'Harrell_C_Index': f"{eval_result['harrell_cindex']:.4f}",
                'Uno_C_Index': f"{eval_result['uno_cindex']:.4f}",
                'AUC': f"{eval_result['auc']:.4f}",
                'Brier_Score': f"{eval_result['brier_score']:.4f}" if not np.isnan(eval_result['brier_score']) else 'NaN'
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def _save_improvement_analysis(self, results: Dict[str, Dict], filepath: str):
        """Save improvement analysis as CSV."""
        method_names = {
            'pairwise': 'Pairwise',
            'pairwise_ipcw': 'Pairwise_IPCW',
            'normalized_combination': 'Normalized_Combo',
            'normalized_combination_ipcw': 'Normalized_IPCW'
        }
        
        # Get NLL baseline
        nll_eval = results['nll']['evaluation']
        nll_harrell = nll_eval['harrell_cindex']
        nll_uno = nll_eval['uno_cindex']
        nll_auc = nll_eval['auc']
        nll_brier = nll_eval['brier_score']
        
        rows = []
        for method, result in results.items():
            if method == 'nll':
                continue
            
            eval_result = result['evaluation']
            
            harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
            uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
            auc_imp = ((eval_result['auc'] - nll_auc) / nll_auc * 100)
            
            # For Brier score, lower is better
            if not (np.isnan(nll_brier) or np.isnan(eval_result['brier_score'])):
                brier_imp = ((nll_brier - eval_result['brier_score']) / nll_brier * 100)
            else:
                brier_imp = float('nan')
            
            row = {
                'Method': method_names.get(method, method),
                'Harrell_Improvement_%': f"{harrell_imp:.2f}",
                'Uno_Improvement_%': f"{uno_imp:.2f}",
                'AUC_Improvement_%': f"{auc_imp:.2f}",
                'Brier_Improvement_%': f"{brier_imp:.2f}" if not np.isnan(brier_imp) else 'NaN'
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def _save_metadata(self, run_info: Dict, filepath: str):
        """Save experiment metadata."""
        metadata = {
            'experiment': {
                'dataset': self.dataset_name,
                'timestamp': self.timestamp,
                'datetime': datetime.now().isoformat(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            'run_parameters': run_info or {},
            'methods_compared': [
                'NLL (Negative Partial Log-Likelihood)',
                'Pairwise (ConcordancePairwiseLoss without IPCW)',
                'Pairwise_IPCW (ConcordancePairwiseLoss with IPCW)',
                'Normalized_Combo (NormalizedLossCombination without IPCW)',
                'Normalized_IPCW (NormalizedLossCombination with IPCW)'
            ],
            'metrics_evaluated': [
                'Harrell C-index (traditional concordance)',
                'Uno C-index (IPCW-weighted concordance)',
                'AUC (Area Under Curve at specified time)',
                'Brier Score (prediction accuracy at specified time)'
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


class BenchmarkVisualizer:
    """Shared visualization logic for all benchmarks."""
    
    def __init__(self, dataset_config: DatasetConfig, output_dir: str = None):
        self.dataset_config = dataset_config
        self.output_dir = output_dir
    
    def analyze_results(self, results: Dict[str, Dict]) -> None:
        """Analyze results and create comprehensive visualizations."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Print comprehensive summary table
        print(f"\n{'Method':<25} {'Harrell C':<10} {'Uno C':<10} {'AUC':<10} {'Brier':<10}")
        print("-" * 75)
        
        # Method name mapping for display
        method_names = {
            'nll': 'NLL',
            'pairwise': 'Pairwise',
            'pairwise_ipcw': 'Pairwise + IPCW',
            'normalized_combination': 'Normalized Combo',
            'normalized_combination_ipcw': 'Normalized + IPCW'
        }
        
        # Get NLL baseline for comparison
        nll_harrell = results['nll']['evaluation']['harrell_cindex']
        nll_uno = results['nll']['evaluation']['uno_cindex']
        nll_auc = results['nll']['evaluation']['auc']
        nll_brier = results['nll']['evaluation']['brier_score']
        
        for loss_type, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(loss_type, loss_type.upper())
            
            harrell = eval_result['harrell_cindex']
            uno = eval_result['uno_cindex']
            auc = eval_result['auc']
            brier = eval_result['brier_score']
            
            print(f"{method_name:<25} {harrell:<10.4f} {uno:<10.4f} {auc:<10.4f} {brier:<10.4f}")
        
        # Print improvement analysis
        print(f"\n{'='*80}")
        print("IMPROVEMENT OVER NLL BASELINE")
        print(f"{'='*80}")
        print(f"{'Method':<25} {'Harrell Δ%':<12} {'Uno Δ%':<12} {'AUC Δ%':<12} {'Brier Δ%':<12}")
        print("-" * 85)
        
        for loss_type, result in results.items():
            if loss_type == 'nll':
                continue
            
            eval_result = result['evaluation']
            method_name = method_names.get(loss_type, loss_type.upper())
            
            harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
            uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
            auc_imp = ((eval_result['auc'] - nll_auc) / nll_auc * 100)
            
            # For Brier score, lower is better, so improvement is negative change
            if not (np.isnan(nll_brier) or np.isnan(eval_result['brier_score'])):
                brier_imp = ((nll_brier - eval_result['brier_score']) / nll_brier * 100)
            else:
                brier_imp = float('nan')
            
            print(f"{method_name:<25} {harrell_imp:<12.2f} {uno_imp:<12.2f} {auc_imp:<12.2f} {brier_imp:<12.2f}")
        
        # Create comprehensive visualizations
        self.create_comprehensive_plots(results)
    
    def create_comprehensive_plots(self, results: Dict[str, Dict]) -> None:
        """Create comprehensive visualization plots with all metrics and methods."""
        print("\n=== Creating Comprehensive Analysis Plots ===")
        
        # Set up the plotting style with fixed dimensions
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 3x2 layout for comprehensive analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Comprehensive Survival Analysis Comparison - {self.dataset_config.name} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # 1. Concordance indices comparison (Harrell vs Uno)
        self._plot_concordance_comparison(axes[0, 0], results)
        
        # 2. All metrics performance comparison
        self._plot_all_metrics_comparison(axes[0, 1], results)
        
        # 3. IPCW effect analysis
        self._plot_ipcw_effect(axes[1, 0], results)
        
        # 4. Training loss evolution
        self._plot_training_evolution(axes[1, 1], results)
        
        # 5. Weight evolution for combination methods
        self._plot_weight_evolution(axes[2, 0], results)
        
        # 6. Performance improvement over baseline
        self._plot_comprehensive_improvement(axes[2, 1], results)
        
        plt.tight_layout()
        
        # Save figure to results folder
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figure_filename = f"{self.dataset_config.name}_comprehensive_analysis_{timestamp}.png"
            figure_path = os.path.join(self.output_dir, figure_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Comprehensive analysis figure saved: {figure_path}")
        
        plt.show()
    
    def create_plots(self, results: Dict[str, Dict]) -> None:
        """Legacy method - redirects to comprehensive plots."""
        self.create_comprehensive_plots(results)
    
    def _plot_concordance_comparison(self, ax, results):
        """Plot Harrell vs Uno C-index comparison."""
        method_names = ['NLL', 'Pairwise', 'Pairwise+IPCW', 'Combo', 'Combo+IPCW']
        methods = ['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw']
        
        harrell_scores = [results[method]['evaluation']['harrell_cindex'] for method in methods]
        uno_scores = [results[method]['evaluation']['uno_cindex'] for method in methods]
        
        x = np.arange(len(method_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, harrell_scores, width, label="Harrell's C-index", alpha=0.8)
        bars2 = ax.bar(x + width/2, uno_scores, width, label="Uno's C-index", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('C-index Score')
        ax.set_title('Concordance Index Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_all_metrics_comparison(self, ax, results):
        """Plot all metrics (Harrell, Uno, AUC, Brier) for all methods."""
        method_names = ['NLL', 'Pairwise', 'Pairwise+IPCW', 'Combo', 'Combo+IPCW']
        methods = ['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw']
        
        # Normalize all metrics to [0,1] for comparison
        harrell_scores = [results[method]['evaluation']['harrell_cindex'] for method in methods]
        uno_scores = [results[method]['evaluation']['uno_cindex'] for method in methods]
        auc_scores = [results[method]['evaluation']['auc'] for method in methods]
        
        # For Brier score, invert and normalize (lower is better)
        brier_scores = []
        for method in methods:
            brier = results[method]['evaluation']['brier_score']
            if not np.isnan(brier):
                # Convert to "higher is better" by taking 1 - normalized_brier
                brier_scores.append(max(0, 1 - brier))
            else:
                brier_scores.append(0)
        
        x = np.arange(len(method_names))
        width = 0.2
        
        ax.bar(x - 1.5*width, harrell_scores, width, label="Harrell's C", alpha=0.8)
        ax.bar(x - 0.5*width, uno_scores, width, label="Uno's C", alpha=0.8)
        ax.bar(x + 0.5*width, auc_scores, width, label='AUC', alpha=0.8)
        ax.bar(x + 1.5*width, brier_scores, width, label='1-Brier', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('All Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    def _plot_ipcw_effect(self, ax, results):
        """Plot the effect of IPCW on pairwise and combination methods."""
        methods_without_ipcw = ['pairwise', 'normalized_combination']
        methods_with_ipcw = ['pairwise_ipcw', 'normalized_combination_ipcw']
        method_labels = ['Pairwise', 'Normalized Combo']
        
        uno_without = [results[method]['evaluation']['uno_cindex'] for method in methods_without_ipcw]
        uno_with = [results[method]['evaluation']['uno_cindex'] for method in methods_with_ipcw]
        
        x = np.arange(len(method_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, uno_without, width, label='Without IPCW', alpha=0.8)
        bars2 = ax.bar(x + width/2, uno_with, width, label='With IPCW', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel("Uno's C-index")
        ax.set_title('Effect of IPCW Weighting')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels and improvement arrows
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            h1, h2 = bar1.get_height(), bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.01,
                   f'{h1:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.01,
                   f'{h2:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Draw improvement arrow
            improvement = ((h2 - h1) / h1 * 100)
            mid_x = x[i]
            ax.annotate(f'{improvement:+.1f}%', xy=(mid_x, max(h1, h2) + 0.03),
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    def _plot_training_evolution(self, ax, results):
        """Plot training loss evolution for all methods."""
        method_colors = {'nll': 'blue', 'pairwise': 'orange', 'pairwise_ipcw': 'red',
                        'normalized_combination': 'green', 'normalized_combination_ipcw': 'purple'}
        
        for loss_type, result in results.items():
            train_losses = result['training']['train_losses']
            color = method_colors.get(loss_type, 'gray')
            ax.plot(train_losses, label=loss_type.replace('_', ' ').title(), 
                   alpha=0.8, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_weight_evolution(self, ax, results):
        """Plot weight evolution for both combination methods."""
        has_weights = False
        
        for method_name, method_key in [('Without IPCW', 'normalized_combination'), 
                                       ('With IPCW', 'normalized_combination_ipcw')]:
            if method_key in results and results[method_key]['training']['weight_evolution']:
                weights = results[method_key]['training']['weight_evolution']
                nll_weights = [w[0] for w in weights]
                pairwise_weights = [w[1] for w in weights]
                
                linestyle = '-' if 'ipcw' not in method_key else '--'
                ax.plot(nll_weights, label=f'NLL Weight ({method_name})', 
                       alpha=0.8, linewidth=2, linestyle=linestyle, color='blue')
                ax.plot(pairwise_weights, label=f'Pairwise Weight ({method_name})', 
                       alpha=0.8, linewidth=2, linestyle=linestyle, color='orange')
                has_weights = True
        
        if has_weights:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight')
            ax.set_title('Weight Evolution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No weight evolution available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Evolution')
    
    def _plot_comprehensive_improvement(self, ax, results):
        """Plot comprehensive improvement over NLL baseline."""
        methods = ['pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw']
        method_names = ['Pairwise', 'Pairwise+IPCW', 'Combo', 'Combo+IPCW']
        
        # Get baseline
        nll_harrell = results['nll']['evaluation']['harrell_cindex']
        nll_uno = results['nll']['evaluation']['uno_cindex']
        
        harrell_improvements = []
        uno_improvements = []
        
        for method in methods:
            eval_result = results[method]['evaluation']
            harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
            uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
            
            harrell_improvements.append(harrell_imp)
            uno_improvements.append(uno_imp)
        
        x = np.arange(len(method_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, harrell_improvements, width, label="Harrell's C Improvement", alpha=0.8)
        bars2 = ax.bar(x + width/2, uno_improvements, width, label="Uno's C Improvement", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Improvement over NLL (%)')
        ax.set_title('Performance Improvement Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >= 0 else -0.3),
                       f'{height:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=8)


class BenchmarkRunner:
    """Main benchmark runner that orchestrates the entire process."""
    
    def __init__(self, 
                 data_loader: AbstractDataLoader,
                 dataset_config: DatasetConfig,
                 batch_size: int = 64,
                 epochs: int = 50,
                 learning_rate: float = 5e-2,
                 output_dir: str = "results",
                 save_results: bool = True,
                 random_seed: int = None):
        
        self.data_loader = data_loader
        self.dataset_config = dataset_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_results = save_results
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # Initialize components
        self.trainer = BenchmarkTrainer(self.device, epochs, learning_rate)
        self.evaluator = BenchmarkEvaluator(self.device, dataset_config)
        self.visualizer = BenchmarkVisualizer(dataset_config, output_dir if save_results else None)
        self.logger = ResultsLogger(dataset_config.name, output_dir) if save_results else None
        
        print(f"Using device: {self.device}")
        print(f"Dataset: {dataset_config.name}")
        print(f"Learning rate: {learning_rate}")
        if random_seed is not None:
            print(f"Random seed: {random_seed} (reproducible results)")
    
    def _set_random_seed(self, seed: int):
        """Set random seed for reproducible results."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For reproducible behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def run_comparison(self, num_runs: int = 1) -> Dict[str, Dict]:
        """Run complete benchmark comparison with optional multiple runs."""
        start_time = time_module.time()
        
        print("=" * 80)
        print(f"COMPREHENSIVE SURVIVAL ANALYSIS COMPARISON - {self.dataset_config.name.upper()} DATASET")
        if num_runs > 1:
            print(f"Running {num_runs} independent experiments for statistical robustness")
        print("=" * 80)
        
        # Store results across all runs
        all_runs_results = []
        
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"\n{'='*80}")
                print(f"RUN {run_idx + 1}/{num_runs}")
                print(f"{'='*80}")
            
            # Load data (fresh for each run if multiple runs)
            dataloader_train, dataloader_val, dataloader_test, num_features = self.data_loader.load_data()
            
            # Define loss types to compare (including IPCW variants)
            loss_types = [
                'nll',                              # NLL without IPCW
                'pairwise',                         # Pairwise without IPCW
                'pairwise_ipcw',                    # Pairwise with IPCW
                'normalized_combination',           # Normalized combination without IPCW
                'normalized_combination_ipcw'       # Normalized combination with IPCW
            ]
            
            run_results = {}
            
            for loss_type in loss_types:
                print(f"\n{'='*60}")
                print(f"TESTING {loss_type.upper()}")
                if num_runs > 1:
                    print(f"Run {run_idx + 1}/{num_runs}")
                print(f"{'='*60}")
                
                # Create fresh model
                model = self.trainer.create_model(num_features)
                
                # Train model
                training_results = self.trainer.train_model(model, dataloader_train, dataloader_val, loss_type)
                
                # Evaluate model
                eval_results = self.evaluator.evaluate_model(model, dataloader_test)
                
                # Store results
                run_results[loss_type] = {
                    'training': training_results,
                    'evaluation': eval_results
                }
                
                print(f"Results: Harrell C={eval_results['harrell_cindex']:.4f}, "
                      f"Uno C={eval_results['uno_cindex']:.4f}, "
                      f"AUC={eval_results['auc']:.4f}, "
                      f"Brier={eval_results['brier_score']:.4f}")
            
            all_runs_results.append(run_results)
        
        # Aggregate results across runs
        if num_runs == 1:
            final_results = all_runs_results[0]
        else:
            final_results = self._aggregate_multiple_runs(all_runs_results)
        
        # Analyze results
        self.visualizer.analyze_results(final_results)
        
        # Calculate execution time
        end_time = time_module.time()
        execution_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED IN {execution_time/60:.1f} MINUTES ({execution_time:.1f} seconds)")
        print(f"{'='*80}")
        
        # Save results to files
        if self.save_results and self.logger:
            run_info = {
                'num_runs': num_runs,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'dataset_auc_time': self.dataset_config.auc_time
            }
            
            saved_files = self.logger.save_results(final_results, run_info, execution_time)
            
            print(f"\n{'='*80}")
            print("RESULTS SAVED TO FILES")
            print(f"{'='*80}")
            for file_type, filepath in saved_files.items():
                print(f"{file_type.upper()}: {filepath}")
        
        return final_results
    
    def _aggregate_multiple_runs(self, all_runs: List[Dict[str, Dict]]) -> Dict[str, Dict]:
        """Aggregate results from multiple runs (mean and std)."""
        aggregated = {}
        loss_types = all_runs[0].keys()
        
        for loss_type in loss_types:
            # Collect metrics across all runs
            harrell_scores = [run[loss_type]['evaluation']['harrell_cindex'] for run in all_runs]
            uno_scores = [run[loss_type]['evaluation']['uno_cindex'] for run in all_runs]
            auc_scores = [run[loss_type]['evaluation']['auc'] for run in all_runs]
            brier_scores = [run[loss_type]['evaluation']['brier_score'] for run in all_runs if not np.isnan(run[loss_type]['evaluation']['brier_score'])]
            
            # Aggregate training losses (use first run as representative)
            representative_training = all_runs[0][loss_type]['training']
            
            aggregated[loss_type] = {
                'training': representative_training,  # Use first run for training curves
                'evaluation': {
                    'harrell_cindex': np.mean(harrell_scores),
                    'harrell_cindex_std': np.std(harrell_scores),
                    'uno_cindex': np.mean(uno_scores),
                    'uno_cindex_std': np.std(uno_scores),
                    'auc': np.mean(auc_scores),
                    'auc_std': np.std(auc_scores),
                    'brier_score': np.mean(brier_scores) if brier_scores else float('nan'),
                    'brier_score_std': np.std(brier_scores) if brier_scores else float('nan')
                },
                'run_details': {
                    'num_runs': len(all_runs),
                    'individual_runs': all_runs
                }
            }
        
        return aggregated
