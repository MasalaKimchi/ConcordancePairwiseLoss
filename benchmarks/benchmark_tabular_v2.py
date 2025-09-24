#!/usr/bin/env python3
"""
Focused Tabular Benchmark for Core Loss Comparison v2

This benchmark focuses on comparing 4 core loss functions across all tabular datasets:
1. NLL (Negative Log-Likelihood)
2. CPL (Concordance Pairwise Loss)
3. CPL (ipcw) (CPL with IPCW weighting computed per batch)
4. CPL (ipcw batch) (CPL with IPCW weighting precomputed from full training set)

Uses learning rates: [0.001, 0.005, 0.0005] and hidden dimensions: [64, 128]

Usage:
    conda activate concordance-pairwise-loss
    python benchmarks/benchmark_tabular_v2.py --dataset gbsg2 --epochs 50
    python benchmarks/benchmark_tabular_v2.py --dataset all --epochs 30  # Run on all datasets
"""

import argparse
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure we can import the original framework
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

from benchmark_framework import (
    BenchmarkEvaluator,
    BenchmarkVisualizer,
    DATASET_CONFIGS,
)
from benchmark_framework_improved import ResultsLogger
from data_loaders import DATA_LOADERS
from concordance_pairwise_loss import ConcordancePairwiseLoss
from concordance_pairwise_loss.dynamic_weighting import NormalizedLossCombination
from torchsurv.loss.cox import neg_partial_log_likelihood


class TabularBenchmarkTrainer:
    """Focused trainer for tabular datasets with core loss functions only."""
    
    def __init__(
        self, 
        device: torch.device, 
        epochs: int = 100, 
        learning_rate: float = 5e-2, 
        weight_decay: float = 1e-4,
        hidden_dim: int = 128,
        dataset_config = None
    ):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.dataset_config = dataset_config
        
        # Pre-initialize loss functions for efficiency
        self.cpl_loss = ConcordancePairwiseLoss(
            reduction="mean",
            temp_scaling='linear',
            pairwise_sampling='balanced',
            use_ipcw=False
        )
        self.cpl_ipcw_loss = ConcordancePairwiseLoss(
            reduction="mean",
            temp_scaling='linear',
            pairwise_sampling='balanced',
            use_ipcw=True
        )
        self.cpl_ipcw_batch_loss = ConcordancePairwiseLoss(
            reduction="mean",
            temp_scaling='linear',
            pairwise_sampling='balanced',
            use_ipcw=True
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Store precomputed IPCW weights for batch variant
        self.precomputed_ipcw_weights = None
    
    class _RiskModel(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module, risk_head: torch.nn.Module):
            super().__init__()
            self.backbone = backbone
            self.risk_head = risk_head
        
        def forward(self, x):
            feats = self.backbone(x)
            risk = self.risk_head(feats)
            return risk
    
    def create_model(self, num_features: int) -> torch.nn.Module:
        """Create model architecture optimized for tabular data."""
        backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(num_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        risk_head = torch.nn.Linear(self.hidden_dim, 1)
        model = self._RiskModel(backbone, risk_head)
        model = model.to(self.device)
        
        print(f"Using tabular model with {num_features} → {self.hidden_dim} → 1")
        return model
    
    def _precompute_ipcw_weights(self, dataloader_train: DataLoader) -> None:
        """Precompute IPCW weights from the full training set for batch variant."""
        from torchsurv.stats.ipcw import get_ipcw
        
        # Collect all training data
        all_times = []
        all_events = []
        
        for batch in dataloader_train:
            x, (event, time) = batch
            all_times.append(time)
            all_events.append(event)
        
        # Concatenate all data
        all_times = torch.cat(all_times, dim=0)
        all_events = torch.cat(all_events, dim=0)
        
        # Compute IPCW weights on full training set
        if not all_events.any():
            # If all censored, use unit weights
            self.precomputed_ipcw_weights = torch.ones(len(all_events), device=self.device)
        else:
            # Compute IPCW on CPU then move to GPU
            events_cpu = all_events.cpu()
            times_cpu = all_times.cpu()
            ipcw = get_ipcw(events_cpu, times_cpu)
            self.precomputed_ipcw_weights = ipcw.to(self.device)
        
        print(f"Precomputed IPCW weights for {len(all_times)} training samples")

    def train_model(
        self,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        loss_type: str,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Train model with specified core loss type."""
        print(f"\n=== Training {loss_type.upper()} ===")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        train_losses = []
        val_losses = []
        
        # No combination strategy needed for this simplified comparison
        loss_combiner = None
        use_gradnorm = False
        
        # Precompute IPCW weights for batch variant
        if loss_type == "cpl_ipcw_batch":
            self._precompute_ipcw_weights(dataloader_train)
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            epoch_total_loss = 0.0
            
            for batch in dataloader_train:
                x, (event, time) = batch
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                
                optimizer.zero_grad()
                log_hz = model(x)
                
                # Use mixed precision for faster training
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        total_loss = self._compute_loss(
                            log_hz, event, time, loss_type, epoch, 
                            loss_combiner, use_gradnorm, model, temperature
                        )
                else:
                    total_loss = self._compute_loss(
                        log_hz, event, time, loss_type, epoch, 
                        loss_combiner, use_gradnorm, model, temperature
                    )
                
                # Backward pass with error handling
                if self.device.type == 'cuda' and self.scaler is not None:
                    # Check for NaN or infinite values before scaling
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss value {total_loss.item()}, skipping this batch")
                        continue
                    
                    self.scaler.scale(total_loss).backward()
                    
                    # Check gradients: presence and NaN/Inf
                    has_any_grad = False
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            has_any_grad = True
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                    
                    if not has_any_grad:
                        # Nothing to step; skip update to avoid GradScaler assertion
                        print("Warning: No gradients found, skipping optimizer step")
                        # Do not call scaler.update() without a corresponding step()
                    elif has_nan_grad:
                        print("Warning: NaN/Inf gradients detected, skipping optimizer step")
                        # Do not call scaler.update() without a corresponding step()
                    else:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    # Check for NaN or infinite values
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss value {total_loss.item()}, skipping this batch")
                        continue
                    
                    total_loss.backward()
                    optimizer.step()
                
                epoch_total_loss += total_loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                x_val, (event_val, time_val) = next(iter(dataloader_val))
                x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                log_hz_val = model(x_val)
                
                val_loss = self._compute_loss(
                    log_hz_val, event_val, time_val, loss_type, epoch,
                    loss_combiner, False, model, temperature
                )
            
            # Store results
            train_losses.append(epoch_total_loss)
            val_losses.append(val_loss.item())
            
            if epoch % 10 == 0:
                if loss_combiner and loss_combiner.weight_history:
                    nll_w, cpl_w = loss_combiner.weight_history[-1]
                    print(f"Epoch {epoch:03d}: Total={epoch_total_loss:.4f}, Weights=({nll_w:.3f}, {cpl_w:.3f})")
                else:
                    print(f"Epoch {epoch:03d}: Total={epoch_total_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weight_evolution': loss_combiner.weight_history if loss_combiner else []
        }
    
    def _compute_loss(
        self, 
        log_hz: torch.Tensor, 
        event: torch.Tensor, 
        time: torch.Tensor,
        loss_type: str,
        epoch: int,
        loss_combiner: Optional[NormalizedLossCombination],
        use_gradnorm: bool,
        model: torch.nn.Module,
        temperature: float
    ) -> torch.Tensor:
        """Compute loss based on the specified type."""
        # Ensure proper shape for NLL
        log_hz_2d = log_hz if log_hz.dim() == 2 else log_hz.unsqueeze(1)
        log_hz_1d = log_hz.squeeze(-1) if log_hz.dim() == 2 else log_hz
        
        # Clamp log_hz to prevent extreme values that could cause NaN
        log_hz_2d = torch.clamp(log_hz_2d, min=-10.0, max=10.0)
        log_hz_1d = torch.clamp(log_hz_1d, min=-10.0, max=10.0)
        
        if loss_type == "nll":
            loss = neg_partial_log_likelihood(log_hz_2d, event, time, reduction="mean")
        elif loss_type == "cpl":
            # Update temperature in CPL loss
            self.cpl_loss.temperature = temperature
            loss = self.cpl_loss(log_hz_1d, time, event)
        elif loss_type == "cpl_ipcw":
            # Update temperature in CPL IPCW loss
            self.cpl_ipcw_loss.temperature = temperature
            loss = self.cpl_ipcw_loss(log_hz_1d, time, event)
        elif loss_type == "cpl_ipcw_batch":
            # Update temperature in CPL IPCW batch loss
            self.cpl_ipcw_batch_loss.temperature = temperature
            # Use precomputed IPCW weights instead of computing per batch
            if self.precomputed_ipcw_weights is not None:
                # For batch variant, we need to create a new loss instance with precomputed weights
                # or modify the existing one. For now, we'll create a temporary one.
                temp_loss = ConcordancePairwiseLoss(
                    reduction="mean",
                    temp_scaling='linear',
                    pairwise_sampling='balanced',
                    use_ipcw=True,
                    ipcw_weights=self.precomputed_ipcw_weights
                )
                temp_loss.temperature = temperature
                loss = temp_loss(log_hz_1d, time, event)
            else:
                # Fallback to regular IPCW if not precomputed
                loss = self.cpl_ipcw_batch_loss(log_hz_1d, time, event)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Final safety check - ensure we return a valid tensor
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Computed loss is {loss.item()}, replacing with 1.0")
            return torch.tensor(1.0, device=log_hz.device, requires_grad=True)
        
        return loss


class TabularBenchmarkRunner:
    """Focused benchmark runner for tabular datasets and core loss comparison."""
    
    # Core loss functions to compare
    CORE_LOSS_TYPES = [
        'nll',
        'cpl', 
        'cpl_ipcw',
        'cpl_ipcw_batch'
    ]
    
    # Tabular datasets available
    TABULAR_DATASETS = ['gbsg2', 'flchain', 'whas500', 'support2', 'metabric']
    
    def __init__(
        self,
        dataset_name: str,
        epochs: int = 100,
        learning_rate: float = 5e-2,
        weight_decay: float = 1e-4,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        batch_size: int = None,
        output_dir: str = "results",
        save_results: bool = True,
        random_seed: int = None,
        num_features: Optional[int] = None
    ):
        if dataset_name not in self.TABULAR_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not in tabular datasets: {self.TABULAR_DATASETS}")
        
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.output_dir = output_dir
        self.save_results = save_results
        self.random_seed = random_seed
        self.num_features = num_features
        
        # Initialize dataset components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DATA_LOADERS[dataset_name]()
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        
        # Override batch size if provided
        if batch_size is not None:
            self.data_loader.batch_size = batch_size
        
        # Set random seed if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # Initialize components
        self.trainer = TabularBenchmarkTrainer(
            device=self.device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            
            dataset_config=self.dataset_config
        )
        self.evaluator = BenchmarkEvaluator(self.device, self.dataset_config)
        self.visualizer = BenchmarkVisualizer(self.dataset_config, output_dir if save_results else None)
        self.logger = ResultsLogger(self.dataset_config.name, output_dir) if save_results else None
        
        print(f"Tabular Benchmark Setup:")
        print(f"  Dataset: {self.dataset_config.name}")
        print(f"  Device: {self.device}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Learning rate: {learning_rate}")
        if self.device.type == 'cuda':
            print("  ✅ Mixed precision enabled")
        if random_seed is not None:
            print(f"  Random seed: {random_seed}")
    
    def _set_random_seed(self, seed: int):
        """Set random seed for reproducible results."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def run_comparison(self) -> Dict[str, Dict]:
        """Run comparison of core loss functions with single run and per-architecture reporting."""
        import time as time_module
        start_time = time_module.time()
        
        print("=" * 80)
        print(f"TABULAR SURVIVAL ANALYSIS COMPARISON - {self.dataset_config.name.upper()}")
        print("Core Loss Functions: NLL, CPL, CPL(ipcw), CPL(ipcw batch)")
        print("=" * 80)
        
        # Load data
        dataloader_train, dataloader_val, dataloader_test, loader_num_features = self.data_loader.load_data()
        effective_num_features = self.num_features if self.num_features is not None else loader_num_features
        
        results = {}
            
        # Grid search per loss and per hidden size (report per-architecture)
        for loss_type in self.CORE_LOSS_TYPES:
            print(f"\n{'='*50}")
            print(f"TESTING {loss_type.upper()}")
            print(f"{'='*50}")
            
            # Define grids
            lr_grid = [0.001, 0.005, 0.0005]
            hd_grid = [64, 128]
            temp_grid = [0.5, 1.0] if (loss_type != 'nll') else [1.0]
                
            for hd in hd_grid:
                best_val_uno = -1.0
                best = None
                best_training = None
                best_model = None
                
                for lr in lr_grid:
                        for temp in temp_grid:
                            # Update trainer settings
                            self.trainer.learning_rate = lr
                            self.trainer.hidden_dim = hd
                            
                            model = self.trainer.create_model(effective_num_features)
                            training_results = self.trainer.train_model(
                                model, dataloader_train, dataloader_val, loss_type, temp
                            )
                            # Evaluate on validation for selection
                            val_eval = self.evaluator.evaluate_model(model, dataloader_val)
                            val_uno = float(val_eval['uno_cindex'])
                            if val_uno > best_val_uno:
                                best_val_uno = val_uno
                                best = (lr, hd, temp)
                                best_training = training_results
                                best_model = model
                
                # Evaluate best for this hidden size on test
                eval_results = self.evaluator.evaluate_model(best_model, dataloader_test)
                loss_key = f"{loss_type}_hd{hd}"
                results[loss_key] = {
                    'training': best_training,
                    'evaluation': eval_results,
                    'best_hparams': {'lr': best[0], 'hidden_dim': best[1], 'temperature': best[2]}
                }
                
                print(f"[{loss_key}] Results: Harrell C={eval_results['harrell_cindex']:.4f}, "
                      f"Uno C={eval_results['uno_cindex']:.4f}, "
                      f"Cum AUC={eval_results['cumulative_auc']:.4f}, "
                      f"Inc AUC={eval_results['incident_auc']:.4f}, "
                      f"Brier={eval_results['brier_score']:.4f}")
            
        # Analyze results with visualization
        self._analyze_results(results)
        
        # Calculate execution time
        end_time = time_module.time()
        execution_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED IN {execution_time/60:.1f} MINUTES")
        print(f"{'='*80}")
        
        # Save results
        if self.save_results and self.logger:
            run_info = {
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'hidden_dim': self.hidden_dim,
                'temperature': self.temperature,
                'batch_size': self.data_loader.batch_size,
                'device': str(self.device),
                'mixed_precision': self.device.type == 'cuda',
                'core_loss_types': self.CORE_LOSS_TYPES,
                'execution_time': execution_time
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_config.name}_tabular_benchmark_{timestamp}"
            saved_files = self.logger.save_results(results, filename)
            
            # Save comprehensive CSV with all metrics and hyperparameters
            self._save_comprehensive_csv(results, execution_time)
            
            print(f"\n{'='*80}")
            print("RESULTS SAVED TO FILES")
            print(f"{'='*80}")
            for file_type, filepath in saved_files.items():
                print(f"{file_type.upper()}: {filepath}")
        
        return results
    
    def _analyze_results(self, results: Dict[str, Dict]) -> None:
        """Analyze results and create comprehensive visualizations."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Print comprehensive summary table
        print(f"\n{'Method':<30} {'Harrell C':<10} {'Uno C':<10} {'Cum AUC':<10} {'Inc AUC':<10} {'Brier':<10}")
        print("-" * 90)
        
        # Method name mapping for display
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        for loss_key, result in results.items():
            eval_result = result['evaluation']
            hparams = result['best_hparams']
            
            # Extract base method name and hidden dim
            if '_hd' in loss_key:
                base_method = loss_key.split('_hd')[0]
                hidden_dim = loss_key.split('_hd')[1]
                method_name = f"{method_names.get(base_method, base_method.upper())} (hd{hidden_dim})"
            else:
                method_name = method_names.get(loss_key, loss_key.upper())
            
            harrell = eval_result['harrell_cindex']
            uno = eval_result['uno_cindex']
            cumulative_auc = eval_result['cumulative_auc']
            incident_auc = eval_result['incident_auc']
            brier = eval_result['brier_score']
            
            print(f"{method_name:<30} {harrell:<10.4f} {uno:<10.4f} {cumulative_auc:<10.4f} {incident_auc:<10.4f} {brier:<10.4f}")
        
        # Create comprehensive visualizations
        self._create_comprehensive_plots(results)
    
    def _create_comprehensive_plots(self, results: Dict[str, Dict]) -> None:
        """Create comprehensive visualization plots with all metrics and methods."""
        print("\n=== Creating Comprehensive Analysis Plots ===")
        
        # Set up the plotting style with fixed dimensions
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 2x2 layout for comprehensive analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Tabular Survival Analysis Comparison - {self.dataset_config.name} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # 1. Concordance indices comparison (Harrell vs Uno)
        self._plot_concordance_comparison(axes[0, 0], results)
        
        # 2. All metrics performance comparison
        self._plot_all_metrics_comparison(axes[0, 1], results)
        
        # 3. Training loss evolution
        self._plot_training_evolution(axes[1, 0], results)
        
        # 4. Performance by hidden dimension
        self._plot_hidden_dim_analysis(axes[1, 1], results)
        
        plt.tight_layout()
        
        # Save figure to results folder
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figure_filename = f"{self.dataset_config.name}_tabular_analysis_{timestamp}.png"
            figure_path = os.path.join(self.output_dir, figure_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Comprehensive analysis figure saved: {figure_path}")
        
        plt.show()
    
    def _plot_concordance_comparison(self, ax, results):
        """Plot Harrell vs Uno C-index comparison."""
        # Group by base method and hidden dim
        base_methods = {}
        for loss_key, result in results.items():
            if '_hd' in loss_key:
                base_method = loss_key.split('_hd')[0]
                hidden_dim = loss_key.split('_hd')[1]
                if base_method not in base_methods:
                    base_methods[base_method] = {'hd64': None, 'hd128': None}
                base_methods[base_method][f'hd{hidden_dim}'] = result['evaluation']
        
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        methods = list(base_methods.keys())
        method_labels = [method_names.get(m, m.upper()) for m in methods]
        
        harrell_64 = [base_methods[m]['hd64']['harrell_cindex'] if base_methods[m]['hd64'] else 0 for m in methods]
        uno_64 = [base_methods[m]['hd64']['uno_cindex'] if base_methods[m]['hd64'] else 0 for m in methods]
        harrell_128 = [base_methods[m]['hd128']['harrell_cindex'] if base_methods[m]['hd128'] else 0 for m in methods]
        uno_128 = [base_methods[m]['hd128']['uno_cindex'] if base_methods[m]['hd128'] else 0 for m in methods]
        
        x = np.arange(len(methods))
        width = 0.2
        
        ax.bar(x - 1.5*width, harrell_64, width, label="Harrell's C (hd64)", alpha=0.8)
        ax.bar(x - 0.5*width, uno_64, width, label="Uno's C (hd64)", alpha=0.8)
        ax.bar(x + 0.5*width, harrell_128, width, label="Harrell's C (hd128)", alpha=0.8)
        ax.bar(x + 1.5*width, uno_128, width, label="Uno's C (hd128)", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('C-index Score')
        ax.set_title('Concordance Index Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    def _plot_all_metrics_comparison(self, ax, results):
        """Plot all metrics for all methods."""
        # Group by base method and hidden dim
        base_methods = {}
        for loss_key, result in results.items():
            if '_hd' in loss_key:
                base_method = loss_key.split('_hd')[0]
                hidden_dim = loss_key.split('_hd')[1]
                if base_method not in base_methods:
                    base_methods[base_method] = {'hd64': None, 'hd128': None}
                base_methods[base_method][f'hd{hidden_dim}'] = result['evaluation']
        
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        methods = list(base_methods.keys())
        method_labels = [method_names.get(m, m.upper()) for m in methods]
        
        # Use hd128 results for this plot
        harrell_scores = [base_methods[m]['hd128']['harrell_cindex'] if base_methods[m]['hd128'] else 0 for m in methods]
        uno_scores = [base_methods[m]['hd128']['uno_cindex'] if base_methods[m]['hd128'] else 0 for m in methods]
        cumulative_auc_scores = [base_methods[m]['hd128']['cumulative_auc'] if base_methods[m]['hd128'] else 0 for m in methods]
        incident_auc_scores = [base_methods[m]['hd128']['incident_auc'] if base_methods[m]['hd128'] else 0 for m in methods]
        
        # For Brier score, invert and normalize (lower is better)
        brier_scores = []
        for m in methods:
            brier = base_methods[m]['hd128']['brier_score'] if base_methods[m]['hd128'] else 0
            if not np.isnan(brier):
                brier_scores.append(max(0, 1 - brier))
            else:
                brier_scores.append(0)
        
        x = np.arange(len(methods))
        width = 0.15
        
        ax.bar(x - 2*width, harrell_scores, width, label="Harrell's C", alpha=0.8)
        ax.bar(x - width, uno_scores, width, label="Uno's C", alpha=0.8)
        ax.bar(x, cumulative_auc_scores, width, label='Cumulative AUC', alpha=0.8)
        ax.bar(x + width, incident_auc_scores, width, label='Incident AUC', alpha=0.8)
        ax.bar(x + 2*width, brier_scores, width, label='1-Brier', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('All Metrics Comparison (hd128)')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    def _plot_training_evolution(self, ax, results):
        """Plot training loss evolution for all methods."""
        method_colors = {'nll': 'blue', 'cpl': 'orange', 'cpl_ipcw': 'red',
                        'cpl_ipcw_batch': 'green'}
        
        for loss_key, result in results.items():
            train_losses = result['training']['train_losses']
            base_method = loss_key.split('_hd')[0] if '_hd' in loss_key else loss_key
            color = method_colors.get(base_method, 'gray')
            label = f"{loss_key.replace('_', ' ').title()}"
            ax.plot(train_losses, label=label, alpha=0.8, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_hidden_dim_analysis(self, ax, results):
        """Plot performance comparison between hidden dimensions."""
        # Group by base method
        base_methods = {}
        for loss_key, result in results.items():
            if '_hd' in loss_key:
                base_method = loss_key.split('_hd')[0]
                hidden_dim = loss_key.split('_hd')[1]
                if base_method not in base_methods:
                    base_methods[base_method] = {}
                base_methods[base_method][f'hd{hidden_dim}'] = result['evaluation']
        
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        methods = list(base_methods.keys())
        method_labels = [method_names.get(m, m.upper()) for m in methods]
        
        hd64_scores = []
        hd128_scores = []
        for m in methods:
            hd64_score = base_methods[m]['hd64']['uno_cindex'] if base_methods[m]['hd64'] else 0
            hd128_score = base_methods[m]['hd128']['uno_cindex'] if base_methods[m]['hd128'] else 0
            hd64_scores.append(hd64_score)
            hd128_scores.append(hd128_score)
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, hd64_scores, width, label="Hidden Dim 64", alpha=0.8)
        bars2 = ax.bar(x + width/2, hd128_scores, width, label="Hidden Dim 128", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel("Uno's C-index")
        ax.set_title('Performance by Hidden Dimension')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _save_comprehensive_csv(self, results: Dict[str, Dict], execution_time: float) -> None:
        """Save comprehensive CSV with all metrics and best hyperparameters."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.dataset_config.name}_tabular_results_{timestamp}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'loss_type', 'hidden_dim', 'best_lr', 'best_temperature',
                'harrell_cindex', 'uno_cindex', 'cumulative_auc', 'incident_auc', 'brier_score',
                'final_train_loss', 'final_val_loss', 'total_epochs'
            ])
            
            # Write data for each loss type and hidden dimension
            for loss_key, result in results.items():
                if '_hd' in loss_key:
                    base_method = loss_key.split('_hd')[0]
                    hidden_dim = loss_key.split('_hd')[1]
                else:
                    base_method = loss_key
                    hidden_dim = 'unknown'
                
                eval_result = result['evaluation']
                hparams = result['best_hparams']
                training = result['training']
                
                writer.writerow([
                    base_method,
                    hidden_dim,
                    hparams['lr'],
                    hparams['temperature'],
                    eval_result['harrell_cindex'],
                    eval_result['uno_cindex'],
                    eval_result['cumulative_auc'],
                    eval_result['incident_auc'],
                    eval_result['brier_score'],
                    training['train_losses'][-1] if training['train_losses'] else 0.0,
                    training['val_losses'][-1] if training['val_losses'] else 0.0,
                    len(training['train_losses'])
                ])
        
        print(f"📊 Comprehensive CSV saved: {csv_path}")


def run_single_dataset(
    dataset_name: str,
    epochs: int = 50,
    learning_rate: float = 5e-2,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    temperature: float = 1.0,
    output_dir: str = "results",
    seed: int = None,
    num_features: int = None
) -> Dict[str, Dict]:
    """Run benchmark on a single dataset."""
    runner = TabularBenchmarkRunner(
        dataset_name=dataset_name,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        temperature=temperature,
        output_dir=output_dir,
        save_results=True,
        random_seed=seed,
        num_features=num_features
    )
    return runner.run_comparison()


def run_all_datasets(
    epochs: int = 50,
    learning_rate: float = 5e-2,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    temperature: float = 1.0,
    output_dir: str = "results",
    seed: int = None
) -> Dict[str, Dict[str, Dict]]:
    """Run benchmark across all tabular datasets."""
    all_results = {}
    
    for dataset in TabularBenchmarkRunner.TABULAR_DATASETS:
        print(f"\n{'='*100}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*100}")
        
        try:
            results = run_single_dataset(
                dataset_name=dataset,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                temperature=temperature,
                output_dir=output_dir,
                seed=seed
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"❌ Failed to process {dataset}: {e}")
            all_results[dataset] = None
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Tabular Survival Analysis Benchmark")
    parser.add_argument('--dataset', choices=TabularBenchmarkRunner.TABULAR_DATASETS + ['all'], 
                       default='gbsg2', help='Dataset name or "all" for all datasets')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer width')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for CPL losses')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num-features', type=int, default=None, help='Input features (auto-detect if omitted)')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        results = run_all_datasets(
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        print(f"\n{'='*100}")
        print("SUMMARY ACROSS ALL DATASETS")
        print(f"{'='*100}")
        
        for dataset, result in results.items():
            if result is not None:
                print(f"\n{dataset.upper()}:")
                for loss_key in result.keys():
                    if '_hd' in loss_key:
                        uno_c = result[loss_key]['evaluation']['uno_cindex']
                        print(f"  {loss_key:<25}: Uno C = {uno_c:.4f}")
            else:
                print(f"\n{dataset.upper()}: FAILED")
    
    else:
        results = run_single_dataset(
            dataset_name=args.dataset,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            output_dir=args.output_dir,
            seed=args.seed,
            num_features=args.num_features
        )


if __name__ == "__main__":
    main()
