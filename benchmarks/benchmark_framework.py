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
from concordance_pairwise_loss.pairwise_horizon_loss import ConcordancePairwiseHorizonLoss
from concordance_pairwise_loss.uncertainty_combined_loss import UncertaintyWeightedCombination
from dataset_configs import DatasetConfig, load_dataset_configs
from abstract_data_loader import AbstractDataLoader


DATASET_CONFIGS = load_dataset_configs()

class BenchmarkEvaluator:
    """Shared evaluation logic for all benchmarks."""
    
    def __init__(self, device: torch.device, dataset_config: DatasetConfig):
        self.device = device
        self.dataset_config = dataset_config
    
    def evaluate_model(self, model: torch.nn.Module, dataloader_test: DataLoader) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics using full test set."""
        model.eval()
        all_log_hz = []
        all_events = []
        all_times = []
        
        # Collect all test data for comprehensive evaluation
        with torch.no_grad():
            for batch in dataloader_test:
                # Handle different data formats (MONAI vs standard)
                if isinstance(batch, dict):
                    # MONAI format: {"image": tensor, "event": tensor, "time": tensor}
                    x = batch["image"]
                    event = batch["event"]
                    time = batch["time"]
                else:
                    # Standard format: (image, (event, time))
                    x, (event, time) = batch
                
                # Convert MONAI MetaTensors to regular PyTorch tensors to avoid indexing issues
                if hasattr(x, 'as_tensor'):
                    x = x.as_tensor()
                if hasattr(event, 'as_tensor'):
                    event = event.as_tensor()
                if hasattr(time, 'as_tensor'):
                    time = time.as_tensor()
                
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                out = model(x)
                # Support models that optionally return (risk, log_tau)
                if isinstance(out, tuple):
                    log_hz = out[0]
                else:
                    log_hz = out
                
                all_log_hz.append(log_hz)
                all_events.append(event)
                all_times.append(time)
        
        # Concatenate all batches
        log_hz = torch.cat(all_log_hz, dim=0)
        event = torch.cat(all_events, dim=0)
        time = torch.cat(all_times, dim=0)
        
        # Ensure correct data types for torchsurv requirements
        event = event.bool()  # torchsurv requires boolean events
        time = time.float()   # torchsurv requires float times
        
        # Get IPCW weights for Uno's C-index
        try:
            if event.any():  # Only compute IPCW if there are events
                # torchsurv's get_ipcw has device issues, so we compute on CPU then move to GPU
                event_cpu = event.cpu()
                time_cpu = time.cpu()
                ipcw_weights = get_ipcw(event_cpu, time_cpu)
                ipcw_weights = ipcw_weights.to(self.device)
            else:
                ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
        except Exception:
            ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
        
        # Initialize metrics
        cindex = ConcordanceIndex()
        auc = Auc()
        brier = BrierScore()
        
        # torchsurv metrics have device issues, so we compute on CPU then move results back
        log_hz_cpu = log_hz.cpu()
        event_cpu = event.cpu()
        time_cpu = time.cpu()
        ipcw_weights_cpu = ipcw_weights.cpu()
        
        # 1. Harrell's C-index (without IPCW weights)
        harrell_cindex = cindex(log_hz_cpu, event_cpu, time_cpu)
        
        # 2. Uno's C-index (with IPCW weights)
        uno_cindex = cindex(log_hz_cpu, event_cpu, time_cpu, weight=ipcw_weights_cpu)
        
        # 3. AUC metrics (both cumulative and incident)
        # Cumulative AUC (over all observed times) - take mean of all time points
        cumulative_auc_tensor = auc(log_hz_cpu, event_cpu, time_cpu)
        cumulative_auc = torch.mean(cumulative_auc_tensor)
        
        # Incident AUC at specified time point
        new_time_cpu = torch.tensor(self.dataset_config.auc_time)
        incident_auc = auc(log_hz_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        
        # 4. Brier Score at specified time point
        try:
            # Convert log hazards to survival probabilities
            # S(t) = exp(-H(t)) where H(t) = exp(log_hz) * t for exponential model
            # For simplicity, use sigmoid transformation to get probabilities
            survival_probs_cpu = torch.sigmoid(-log_hz_cpu)  # Higher risk -> lower survival probability
            brier_score = brier(survival_probs_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        except Exception as e:
            # If Brier score fails, set to NaN
            brier_score = torch.tensor(float('nan'))
        
        return {
            'harrell_cindex': harrell_cindex.item(),
            'uno_cindex': uno_cindex.item(),
            'cumulative_auc': cumulative_auc.item(),
            'incident_auc': incident_auc.item(),
            'brier_score': brier_score.item()
        }


class BenchmarkTrainer:
    """Shared training logic for all benchmarks with GPU optimizations."""
    
    def __init__(self, device: torch.device, epochs: int = 50, learning_rate: float = 5e-2, weight_decay: float = 1e-4, use_mixed_precision: bool = True, dataset_config: DatasetConfig = None, horizon_kind: str = "exp", hetero_tau: bool = False, rel_factor: float = 0.5, temperature: float = 1.0, use_uncertainty_weighting: bool = True):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.dataset_config = dataset_config
        self.horizon_kind = horizon_kind
        self.hetero_tau = hetero_tau
        self.rel_factor = rel_factor
        self.temperature = temperature
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Pre-initialize loss functions for efficiency
        self.pairwise_loss_fn = ConcordancePairwiseLoss(
            reduction="mean",
            temp_scaling='linear',
            pairwise_sampling='balanced',
            use_ipcw=False
        )
        self.pairwise_loss_ipcw_fn = ConcordancePairwiseLoss(
            reduction="mean",
            temp_scaling='linear',
            pairwise_sampling='balanced',
            use_ipcw=True
        )
        # Horizon-weighted loss and hybrid combiner (initialized per-train with stats)
        self.rank_loss = None
        self.hybrid_combiner = None
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    class _RiskModel(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module, risk_head: torch.nn.Module, log_tau_head: Optional[torch.nn.Module] = None):
            super().__init__()
            self.backbone = backbone
            self.risk_head = risk_head
            self.log_tau_head = log_tau_head
        def forward(self, x):
            feats = self.backbone(x)
            risk = self.risk_head(feats)
            if self.log_tau_head is not None:
                log_tau = self.log_tau_head(feats)
                return risk, log_tau
            return risk
    
    def create_model(self, num_features: int) -> torch.nn.Module:
        """Create optimized model architecture with torch.compile for GPU acceleration."""
        backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(num_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        risk_head = torch.nn.Linear(128, 1)
        if self.hetero_tau:
            log_tau_head = torch.nn.Linear(128, 1)
            model = self._RiskModel(backbone, risk_head, log_tau_head)
        else:
            model = self._RiskModel(backbone, risk_head)
        model = model.to(self.device)
        
        # Note: torch.compile requires triton for optimal performance on Windows
        # For now, we'll use the other GPU optimizations (mixed precision, etc.)
        # torch.compile can be enabled later by installing triton
        print("ℹ️  Using standard model (torch.compile requires triton on Windows)")
        
        return model

    def _times_to_years(self, t: torch.Tensor) -> torch.Tensor:
        if self.dataset_config and getattr(self.dataset_config, 'auc_time_unit', None) == "days":
            return t / 365.25
        return t

    def _median_followup_years(self, dataloader: DataLoader) -> float:
        times = []
        for batch in dataloader:
            _, (_, t) = batch
            # extracting to CPU and converting units if necessary
            times.append(self._times_to_years(t.cpu()))
        return torch.cat(times).median().item()
    
    def train_model(self, 
                   model: torch.nn.Module,
                   dataloader_train: DataLoader, 
                   dataloader_val: DataLoader,
                   loss_type: str) -> Dict[str, any]:
        """Train model with specified loss type."""
        print(f"\n=== Training {loss_type.upper()} ===")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        train_losses = []
        val_losses = []
        
        # Initialize normalized loss combination
        loss_combiner = None
        if loss_type in ['normalized_combination', 'normalized_combination_ipcw']:
            loss_combiner = NormalizedLossCombination(total_epochs=self.epochs)
        
        # Initialize horizon-weighted rank loss / hybrid if requested
        loss_kind = self.horizon_kind
        if loss_type.startswith("cphl_") or loss_type.startswith("hybrid_"):
            loss_kind = loss_type.split("_")[1]
        if loss_type.startswith("cphl_") or loss_type.startswith("hybrid_"):
            median_follow = self._median_followup_years(dataloader_train)
            # Compute dataset-adaptive h bounds from training times (in years)
            # Use interquantile range scaled to provide a reasonable window per dataset
            times_years_all = []
            for batch in dataloader_train:
                _, (_, t) = batch
                times_years_all.append(self._times_to_years(t.cpu()))
            times_years_all = torch.cat(times_years_all)
            q10 = torch.quantile(times_years_all, 0.10).item()
            q90 = torch.quantile(times_years_all, 0.90).item()
            # Ensure small positive minimum and reasonable upper bound
            h_min_years = max(0.05, 0.25 * q10)
            h_max_years = max(h_min_years + 1e-6, 0.75 * q90)
            self.rank_loss = ConcordancePairwiseHorizonLoss(
                horizon_kind=loss_kind,
                rel_factor=self.rel_factor,
                temperature=self.temperature,
                hetero_tau=self.hetero_tau,
                reduction="mean",
                use_ipcw=True,  # Enable IPCW by default for CPHL
            )
            if loss_kind != "none":
                self.rank_loss.set_train_stats(median_follow, h_min_years=h_min_years, h_max_years=h_max_years)
            if loss_type.startswith("hybrid_") and self.use_uncertainty_weighting and not loss_type.endswith("_simple"):
                self.hybrid_combiner = UncertaintyWeightedCombination(
                    rank_loss=self.rank_loss,
                    disc_time_nll_fn=lambda scores, times, events, **_: neg_partial_log_likelihood(
                        scores.unsqueeze(1) if scores.dim() == 1 else scores, events, times, reduction="mean"
                    ),
                )
            else:
                self.hybrid_combiner = None
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            epoch_total_loss = 0.0
            
            for batch in dataloader_train:
                x, (event, time) = batch
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                
                optimizer.zero_grad()
                out = model(x)
                if self.hetero_tau:
                    if not isinstance(out, tuple):
                        raise RuntimeError("hetero_tau=True requires risk and log_tau outputs")
                    log_hz, log_tau = out
                    log_tau = log_tau.squeeze(-1)
                else:
                    log_hz = out
                    log_tau = None
                
                # Use mixed precision for faster training
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        # Compute individual losses
                        nll_loss = neg_partial_log_likelihood(
                            log_hz if (log_hz.dim() == 2) else log_hz.unsqueeze(1), event, time, reduction="mean"
                        )
                        pairwise_loss = self.pairwise_loss_fn(log_hz, time, event)
                        pairwise_loss_ipcw = self.pairwise_loss_ipcw_fn(log_hz, time, event)
                        # Times in years and boolean events for horizon losses
                        times_years = self._times_to_years(time)
                        events_bool = event.bool()
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
                        elif loss_type.startswith("cphl_"):
                            total_loss = self.rank_loss(log_hz.squeeze(-1), times_years, events_bool, log_tau)
                        elif loss_type.startswith("hybrid_"):
                            if self.hybrid_combiner is not None:
                                total_loss = self.hybrid_combiner(log_hz.squeeze(-1), times_years, events_bool, log_tau=log_tau)
                            else:
                                rank_component = self.rank_loss(log_hz.squeeze(-1), times_years, events_bool, log_tau)
                                total_loss = nll_loss + rank_component
                        else:
                            total_loss = nll_loss
                else:
                    # Compute individual losses
                    nll_loss = neg_partial_log_likelihood(
                        log_hz if (log_hz.dim() == 2) else log_hz.unsqueeze(1), event, time, reduction="mean"
                    )
                    pairwise_loss = self.pairwise_loss_fn(log_hz, time, event)
                    pairwise_loss_ipcw = self.pairwise_loss_ipcw_fn(log_hz, time, event)
                    times_years = self._times_to_years(time)
                    events_bool = event.bool()
                    
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
                    elif loss_type.startswith("cphl_"):
                        total_loss = self.rank_loss(log_hz.squeeze(-1), times_years, events_bool, log_tau)
                    elif loss_type.startswith("hybrid_"):
                        if self.hybrid_combiner is not None:
                            total_loss = self.hybrid_combiner(log_hz.squeeze(-1), times_years, events_bool, log_tau=log_tau)
                        else:
                            rank_component = self.rank_loss(log_hz.squeeze(-1), times_years, events_bool, log_tau)
                            total_loss = nll_loss + rank_component
                    else:
                        total_loss = nll_loss
                
                # Use mixed precision backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(total_loss).backward()
                    # Check if gradients exist before stepping
                    if any(p.grad is not None for p in model.parameters()):
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                
                epoch_total_loss += total_loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                x_val, (event_val, time_val) = next(iter(dataloader_val))
                x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                out_val = model(x_val)
                if self.hetero_tau:
                    log_hz_val, log_tau_val = out_val
                    log_tau_val = log_tau_val.squeeze(-1)
                else:
                    log_hz_val = out_val
                    log_tau_val = None
                # Use pre-initialized loss functions for validation
                nll_val = neg_partial_log_likelihood(
                    log_hz_val if (log_hz_val.dim() == 2) else log_hz_val.unsqueeze(1), event_val, time_val, reduction="mean"
                )
                pairwise_val = self.pairwise_loss_fn(log_hz_val, time_val, event_val)
                pairwise_val_ipcw = self.pairwise_loss_ipcw_fn(log_hz_val, time_val, event_val)
                times_val_years = self._times_to_years(time_val)
                events_val_bool = event_val.bool()
                
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
                elif loss_type.startswith("cphl_"):
                    val_loss = self.rank_loss(log_hz_val.squeeze(-1), times_val_years, events_val_bool, log_tau_val)
                elif loss_type.startswith("hybrid_"):
                    if self.hybrid_combiner is not None:
                        val_loss = self.hybrid_combiner(log_hz_val.squeeze(-1), times_val_years, events_val_bool, log_tau=log_tau_val)
                    else:
                        rank_component_val = self.rank_loss(log_hz_val.squeeze(-1), times_val_years, events_val_bool, log_tau_val)
                        val_loss = nll_val + rank_component_val
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
        """Save results with individual run metrics in JSON and separate CSV for losses."""
        saved_files = {}
        
        # 1. Save comprehensive JSON file with all individual run metrics
        comprehensive_json = os.path.join(self.output_dir, f"{self.base_filename}_comprehensive.json")
        self._save_comprehensive_json(results, comprehensive_json, run_info, execution_time)
        saved_files['comprehensive'] = comprehensive_json
        
        # 2. Save training and validation losses as separate CSV
        losses_csv = os.path.join(self.output_dir, f"{self.base_filename}_losses.csv")
        self._save_losses_csv(results, losses_csv)
        saved_files['losses'] = losses_csv
        
        return saved_files
    
    def _save_comprehensive_json(self, results: Dict[str, Dict], filepath: str, run_info: Dict = None, execution_time: float = None):
        """Save comprehensive JSON file with all individual run metrics for statistical analysis."""
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
            'individual_runs': {},
            'hyperparameters': run_info or {},
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'device': run_info.get('device', 'unknown') if run_info else 'unknown'
            }
        }
        
        # Performance summary table
        method_names = {
            'nll': 'NLL',
            'pairwise': 'CPL',
            'pairwise_ipcw': 'CPL (ipcw)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
            'cphl_none': 'CPHL (none)',
            'cphl_exp': 'CPHL (exp)',
            'hybrid_exp': 'Hybrid (exp)',
            'hybrid_exp_simple': 'Hybrid (exp, simple)',
            'hybrid_none': 'Hybrid (none)'
        }
        
        for method, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(method, method)
            
            output_data['performance_summary'][method_name] = {
                'harrell_cindex': round(eval_result['harrell_cindex'], 4),
                'uno_cindex': round(eval_result['uno_cindex'], 4),
                'cumulative_auc': round(eval_result['cumulative_auc'], 4),
                'incident_auc': round(eval_result['incident_auc'], 4),
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
                cumulative_auc_imp = ((eval_result['cumulative_auc'] - nll_eval['cumulative_auc']) / nll_eval['cumulative_auc'] * 100)
                incident_auc_imp = ((eval_result['incident_auc'] - nll_eval['incident_auc']) / nll_eval['incident_auc'] * 100)
                
                if not (np.isnan(nll_eval['brier_score']) or np.isnan(eval_result['brier_score'])):
                    brier_imp = ((nll_eval['brier_score'] - eval_result['brier_score']) / nll_eval['brier_score'] * 100)
                else:
                    brier_imp = None
                
                output_data['improvement_analysis'][method_name] = {
                    'harrell_improvement_percent': round(harrell_imp, 2),
                    'uno_improvement_percent': round(uno_imp, 2),
                    'cumulative_auc_improvement_percent': round(cumulative_auc_imp, 2),
                    'incident_auc_improvement_percent': round(incident_auc_imp, 2),
                    'brier_improvement_percent': round(brier_imp, 2) if brier_imp is not None else None
                }
        
        # Individual runs data for statistical analysis
        if 'run_details' in results.get('nll', {}).get('evaluation', {}):
            # Multiple runs case - extract individual run data
            num_runs = results['nll']['evaluation']['run_details']['num_runs']
            individual_runs = results['nll']['evaluation']['run_details']['individual_runs']
            
            for run_idx in range(num_runs):
                run_data = {}
                for method, result in individual_runs[run_idx].items():
                    # Only include evaluation metrics, not training data
                    run_data[method] = {
                        'evaluation': {k: float(v) if hasattr(v, 'item') else v 
                                     for k, v in result['evaluation'].items()}
                    }
                output_data['individual_runs'][f'run_{run_idx + 1}'] = run_data
        else:
            # Single run case - store as individual run
            run_data = {}
            for method, result in results.items():
                # Only include evaluation metrics, not training data
                run_data[method] = {
                    'evaluation': {k: float(v) if hasattr(v, 'item') else v 
                                 for k, v in result['evaluation'].items()}
                }
            output_data['individual_runs']['run_1'] = run_data
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_losses_csv(self, results: Dict[str, Dict], filepath: str):
        """Save training and validation losses as CSV for easy analysis."""
        method_names = {
            'nll': 'NLL',
            'pairwise': 'CPL',
            'pairwise_ipcw': 'CPL (ipcw)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
            'cphl_none': 'CPHL (none)',
            'cphl_exp': 'CPHL (exp)',
            'hybrid_exp': 'Hybrid (exp)',
            'hybrid_exp_simple': 'Hybrid (exp, simple)',
            'hybrid_none': 'Hybrid (none)'
        }
        
        rows = []
        
        # Check if we have multiple runs
        if 'run_details' in results.get('nll', {}).get('evaluation', {}):
            # Multiple runs case
            individual_runs = results['nll']['evaluation']['run_details']['individual_runs']
            for run_idx, run_data in enumerate(individual_runs):
                for method, result in run_data.items():
                    method_name = method_names.get(method, method)
                    train_losses = result['training']['train_losses']
                    val_losses = result['training']['val_losses']
                    
                    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                        rows.append({
                            'run': run_idx + 1,
                            'method': method_name,
                            'epoch': epoch + 1,
                            'train_loss': float(train_loss),
                            'val_loss': float(val_loss)
                        })
        else:
            # Single run case
            for method, result in results.items():
                method_name = method_names.get(method, method)
                train_losses = result['training']['train_losses']
                val_losses = result['training']['val_losses']
                
                for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                    rows.append({
                        'run': 1,
                        'method': method_name,
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'val_loss': float(val_loss)
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
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
            'pairwise': 'CPL',
            'pairwise_ipcw': 'CPL (ipcw)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)'
        }
        
        rows = []
        for method, result in results.items():
            eval_result = result['evaluation']
            row = {
                'Method': method_names.get(method, method),
                'Harrell_C_Index': f"{eval_result['harrell_cindex']:.4f}",
                'Uno_C_Index': f"{eval_result['uno_cindex']:.4f}",
                'Cumulative_AUC': f"{eval_result['cumulative_auc']:.4f}",
                'Incident_AUC': f"{eval_result['incident_auc']:.4f}",
                'Brier_Score': f"{eval_result['brier_score']:.4f}" if not np.isnan(eval_result['brier_score']) else 'NaN'
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def _save_improvement_analysis(self, results: Dict[str, Dict], filepath: str):
        """Save improvement analysis as CSV."""
        method_names = {
            'pairwise': 'CPL',
            'pairwise_ipcw': 'CPL (ipcw)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
            'cphl_none': 'CPHL (none)',
            'cphl_exp': 'CPHL (exp)',
            'cphl_gauss': 'CPHL (gauss)',
            'cphl_tri': 'CPHL (tri)',
            'hybrid_exp': 'Hybrid (exp)',
            'hybrid_gauss': 'Hybrid (gauss)',
            'hybrid_tri': 'Hybrid (tri)'
        }
        
        # Get NLL baseline
        nll_eval = results['nll']['evaluation']
        nll_harrell = nll_eval['harrell_cindex']
        nll_uno = nll_eval['uno_cindex']
        nll_cumulative_auc = nll_eval['cumulative_auc']
        nll_incident_auc = nll_eval['incident_auc']
        nll_brier = nll_eval['brier_score']
        
        rows = []
        for method, result in results.items():
            if method == 'nll':
                continue
            
            eval_result = result['evaluation']
            
            harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
            uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
            cumulative_auc_imp = ((eval_result['cumulative_auc'] - nll_cumulative_auc) / nll_cumulative_auc * 100)
            incident_auc_imp = ((eval_result['incident_auc'] - nll_incident_auc) / nll_incident_auc * 100)
            
            # For Brier score, lower is better
            if not (np.isnan(nll_brier) or np.isnan(eval_result['brier_score'])):
                brier_imp = ((nll_brier - eval_result['brier_score']) / nll_brier * 100)
            else:
                brier_imp = float('nan')
            
            row = {
                'Method': method_names.get(method, method),
                'Harrell_Improvement_%': f"{harrell_imp:.2f}",
                'Uno_Improvement_%': f"{uno_imp:.2f}",
                'Cumulative_AUC_Improvement_%': f"{cumulative_auc_imp:.2f}",
                'Incident_AUC_Improvement_%': f"{incident_auc_imp:.2f}",
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
                'CPL (ConcordancePairwiseLoss without IPCW)',
                'CPL (ipcw) (ConcordancePairwiseLoss with IPCW)',
                'NLL+CPL (NormalizedLossCombination without IPCW)',
                'NLL+CPL (ipcw) (NormalizedLossCombination with IPCW)'
            ],
            'metrics_evaluated': [
                'Harrell C-index (traditional concordance)',
                'Uno C-index (IPCW-weighted concordance)',
                'Cumulative AUC (over all observed times)',
                'Incident AUC (at specified time point)',
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
        print(f"\n{'Method':<25} {'Harrell C':<10} {'Uno C':<10} {'Cum AUC':<10} {'Inc AUC':<10} {'Brier':<10}")
        print("-" * 85)
        
        # Method name mapping for display
        method_names = {
            'nll': 'NLL',
            'pairwise': 'CPL',
            'pairwise_ipcw': 'CPL (ipcw)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
            'cphl_none': 'CPHL (none)',
            'cphl_exp': 'CPHL (exp)',
            'hybrid_exp': 'NLL + CPHL (exp)',
            'hybrid_exp_simple': 'NLL + CPHL (exp, simple)',
            'hybrid_none': 'NLL + CPHL (none)',
            # disabled: gauss/tri variants
        }
        
        # Get NLL baseline for comparison (if available)
        if 'nll' in results:
            nll_harrell = results['nll']['evaluation']['harrell_cindex']
            nll_uno = results['nll']['evaluation']['uno_cindex']
            nll_cumulative_auc = results['nll']['evaluation']['cumulative_auc']
            nll_incident_auc = results['nll']['evaluation']['incident_auc']
            nll_brier = results['nll']['evaluation']['brier_score']
        else:
            # No NLL baseline available (e.g., single loss type sweep)
            nll_harrell = nll_uno = nll_cumulative_auc = nll_incident_auc = nll_brier = None
        
        for loss_type, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(loss_type, loss_type.upper())
            
            harrell = eval_result['harrell_cindex']
            uno = eval_result['uno_cindex']
            cumulative_auc = eval_result['cumulative_auc']
            incident_auc = eval_result['incident_auc']
            brier = eval_result['brier_score']
            
            print(f"{method_name:<25} {harrell:<10.4f} {uno:<10.4f} {cumulative_auc:<10.4f} {incident_auc:<10.4f} {brier:<10.4f}")
        
        # Print improvement analysis (only if NLL baseline available)
        if nll_harrell is not None:
            print(f"\n{'='*80}")
            print("IMPROVEMENT OVER NLL BASELINE")
            print(f"{'='*80}")
            print(f"{'Method':<25} {'Harrell Δ%':<12} {'Uno Δ%':<12} {'Cum AUC Δ%':<12} {'Inc AUC Δ%':<12} {'Brier Δ%':<12}")
            print("-" * 105)
            
            for loss_type, result in results.items():
                if loss_type == 'nll':
                    continue
                
                eval_result = result['evaluation']
                method_name = method_names.get(loss_type, loss_type.upper())
                
                harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
                uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
                cumulative_auc_imp = ((eval_result['cumulative_auc'] - nll_cumulative_auc) / nll_cumulative_auc * 100)
                incident_auc_imp = ((eval_result['incident_auc'] - nll_incident_auc) / nll_incident_auc * 100)
                
                # For Brier score, lower is better, so improvement is negative change
                if not (np.isnan(nll_brier) or np.isnan(eval_result['brier_score'])):
                    brier_imp = ((nll_brier - eval_result['brier_score']) / nll_brier * 100)
                else:
                    brier_imp = float('nan')
                
                print(f"{method_name:<25} {harrell_imp:<12.2f} {uno_imp:<12.2f} {cumulative_auc_imp:<12.2f} {incident_auc_imp:<12.2f} {brier_imp:<12.2f}")
        else:
            print(f"\n{'='*80}")
            print("SINGLE METHOD EVALUATION (No NLL baseline available)")
            print(f"{'='*80}")
        
        # Create comprehensive visualizations
        self.create_comprehensive_plots(results)
    
    def create_comprehensive_plots(self, results: Dict[str, Dict]) -> None:
        """Create compact visualization with two key panels and captions."""
        print("\n=== Creating Compact Analysis Plots (2 panels) ===")
        
        # Set up the plotting style with fixed dimensions
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 1x2 layout (only first two graphs)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Survival Analysis Comparison - {self.dataset_config.name}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Concordance indices comparison (Harrell vs Uno)
        self._plot_concordance_comparison(axes[0], results)
        axes[0].set_title("Concordance (Harrell vs Uno)")
        
        # 2. All metrics performance comparison
        self._plot_all_metrics_comparison(axes[1], results)
        axes[1].set_title("All Metrics (higher is better; 1-Brier shown)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure to results folder
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figure_filename = f"{self.dataset_config.name}_compact_analysis_{timestamp}.png"
            figure_path = os.path.join(self.output_dir, figure_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Compact analysis figure saved: {figure_path}")
        
        plt.show()
    
    def create_plots(self, results: Dict[str, Dict]) -> None:
        """Legacy method - redirects to comprehensive plots."""
        self.create_comprehensive_plots(results)
    
    def _plot_concordance_comparison(self, ax, results):
        """Plot Harrell vs Uno C-index comparison."""
        # Dynamically include available methods
        all_methods = ['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw',
                       'cphl_none', 'cphl_exp', 'hybrid_exp', 'hybrid_exp_simple', 'hybrid_none']
        methods = [m for m in all_methods if m in results]
        method_names = [
            {
                'nll': 'NLL', 'pairwise': 'CPL', 'pairwise_ipcw': 'CPL (ipcw)',
                'normalized_combination': 'NLL+CPL', 'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
                'cphl_none': 'CPHL (none)', 'cphl_exp': 'CPHL (exp)',
                'hybrid_exp': 'Hybrid (exp)', 'hybrid_exp_simple': 'Hybrid (exp, simple)', 'hybrid_none': 'Hybrid (none)'
            }[m]
            for m in methods
        ]
        harrell_scores = [results[m]['evaluation']['harrell_cindex'] for m in methods]
        uno_scores = [results[m]['evaluation']['uno_cindex'] for m in methods]
        x = np.arange(len(methods))
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
        """Plot all metrics (Harrell, Uno, Cumulative AUC, Incident AUC, Brier) for all methods."""
        # Dynamically include available methods
        all_methods = ['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw',
                       'cphl_none', 'cphl_exp', 'hybrid_exp', 'hybrid_exp_simple', 'hybrid_none']
        methods = [m for m in all_methods if m in results]
        method_names = [
            {
                'nll': 'NLL', 'pairwise': 'CPL', 'pairwise_ipcw': 'CPL (ipcw)',
                'normalized_combination': 'NLL+CPL', 'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
                'cphl_none': 'CPHL (none)', 'cphl_exp': 'CPHL (exp)',
                'hybrid_exp': 'Hybrid (exp)', 'hybrid_exp_simple': 'Hybrid (exp, simple)', 'hybrid_none': 'Hybrid (none)'
            }[m]
            for m in methods
        ]
        
        # Normalize all metrics to [0,1] for comparison
        harrell_scores = [results[m]['evaluation']['harrell_cindex'] for m in methods]
        uno_scores = [results[m]['evaluation']['uno_cindex'] for m in methods]
        cumulative_auc_scores = [results[m]['evaluation']['cumulative_auc'] for m in methods]
        incident_auc_scores = [results[m]['evaluation']['incident_auc'] for m in methods]
        
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
        width = 0.15
        
        ax.bar(x - 2*width, harrell_scores, width, label="Harrell's C", alpha=0.8)
        ax.bar(x - width, uno_scores, width, label="Uno's C", alpha=0.8)
        ax.bar(x, cumulative_auc_scores, width, label='Cumulative AUC', alpha=0.8)
        ax.bar(x + width, incident_auc_scores, width, label='Incident AUC', alpha=0.8)
        ax.bar(x + 2*width, brier_scores, width, label='1-Brier', alpha=0.8)
        
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
        methods_without_ipcw = [m for m in ['pairwise', 'normalized_combination'] if m in results]
        methods_with_ipcw = [m for m in ['pairwise_ipcw', 'normalized_combination_ipcw'] if m in results]
        method_labels = [
            'CPL' if 'pairwise' in methods_without_ipcw else None,
            'NLL+CPL' if 'normalized_combination' in methods_without_ipcw else None
        ]
        method_labels = [lbl for lbl in method_labels if lbl is not None]
        
        uno_without = [results[m]['evaluation']['uno_cindex'] for m in methods_without_ipcw]
        uno_with = [results[m]['evaluation']['uno_cindex'] for m in methods_with_ipcw]
        
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
                ax.plot(pairwise_weights, label=f'CPL Weight ({method_name})', 
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
        method_names = ['CPL', 'CPL (ipcw)', 'NLL+CPL', 'NLL+CPL (ipcw)']
        
        # Get baseline (if available)
        if 'nll' in results:
            nll_harrell = results['nll']['evaluation']['harrell_cindex']
            nll_uno = results['nll']['evaluation']['uno_cindex']
            nll_cumulative_auc = results['nll']['evaluation']['cumulative_auc']
            nll_incident_auc = results['nll']['evaluation']['incident_auc']
        else:
            # No NLL baseline available, skip this plot
            return
        
        harrell_improvements = []
        uno_improvements = []
        cumulative_auc_improvements = []
        incident_auc_improvements = []
        
        for method in methods:
            eval_result = results[method]['evaluation']
            harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
            uno_imp = ((eval_result['uno_cindex'] - nll_uno) / nll_uno * 100)
            cumulative_auc_imp = ((eval_result['cumulative_auc'] - nll_cumulative_auc) / nll_cumulative_auc * 100)
            incident_auc_imp = ((eval_result['incident_auc'] - nll_incident_auc) / nll_incident_auc * 100)
            
            harrell_improvements.append(harrell_imp)
            uno_improvements.append(uno_imp)
            cumulative_auc_improvements.append(cumulative_auc_imp)
            incident_auc_improvements.append(incident_auc_imp)
        
        x = np.arange(len(method_names))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, harrell_improvements, width, label="Harrell's C", alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, uno_improvements, width, label="Uno's C", alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, cumulative_auc_improvements, width, label="Cumulative AUC", alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, incident_auc_improvements, width, label="Incident AUC", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Improvement over NLL (%)')
        ax.set_title('Performance Improvement Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3, bars4]:
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
                 random_seed: int = None,
                 use_mixed_precision: bool = True,
                 loss_types: List[str] = None,
                 horizon_kind: str = 'exp',
                 hetero_tau: bool = False,
                 rel_factor: float = 0.5,
                 temperature: float = 1.0,
                 use_uncertainty_weighting: bool = True,
                 num_features: Optional[int] = None):
        
        self.data_loader = data_loader
        self.dataset_config = dataset_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_results = save_results
        self.batch_size = self._optimize_batch_size(batch_size)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.use_mixed_precision = use_mixed_precision
        self.loss_types = loss_types
        self.num_features = num_features
        
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # Initialize components
        self.trainer = BenchmarkTrainer(
            self.device, epochs, learning_rate,
            use_mixed_precision=use_mixed_precision,
            dataset_config=dataset_config,
            horizon_kind=horizon_kind,
            hetero_tau=hetero_tau,
            rel_factor=rel_factor,
            temperature=temperature,
            use_uncertainty_weighting=use_uncertainty_weighting,
        )
        self.evaluator = BenchmarkEvaluator(self.device, dataset_config)
        self.visualizer = BenchmarkVisualizer(dataset_config, output_dir if save_results else None)
        self.logger = ResultsLogger(dataset_config.name, output_dir) if save_results else None
        
        print(f"Using device: {self.device}")
        print(f"Dataset: {dataset_config.name}")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimized batch size: {self.batch_size}")
        if self.use_mixed_precision and self.device.type == 'cuda':
            print("✅ Mixed precision training enabled")
        if random_seed is not None:
            print(f"Random seed: {random_seed} (reproducible results)")
    
    def _optimize_batch_size(self, initial_batch_size: int) -> int:
        """Optimize batch size for GPU memory usage."""
        if self.device.type != 'cuda':
            return initial_batch_size
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        
        # Estimate memory per sample (rough approximation)
        # This is a conservative estimate for survival analysis data
        estimated_memory_per_sample = 1024 * 4  # 4KB per sample (features + gradients)
        
        # Calculate optimal batch size based on available memory
        max_batch_size = int(free_memory * 0.8 / estimated_memory_per_sample)  # Use 80% of free memory
        
        # Use the smaller of initial batch size or calculated max
        optimal_batch_size = min(initial_batch_size, max_batch_size)
        
        # Ensure minimum batch size of 16
        optimal_batch_size = max(16, optimal_batch_size)
        
        if optimal_batch_size != initial_batch_size:
            print(f"🔄 Batch size optimized: {initial_batch_size} → {optimal_batch_size} (GPU memory: {free_memory/1024**3:.1f}GB free)")
        
        return optimal_batch_size
    
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
            dataloader_train, dataloader_val, dataloader_test, loader_num_features = self.data_loader.load_data()
            
            # Define loss types to compare (including new horizon-weighted variants)
            default_loss_types = [
                'nll',
                'pairwise',
                'pairwise_ipcw',
                'normalized_combination',
                'normalized_combination_ipcw',
                'cphl_none',
                'cphl_exp',
                'hybrid_none',
                'hybrid_exp',
                'hybrid_exp_simple',
            ]
            loss_types = default_loss_types if self.loss_types is None else self.loss_types
            
            run_results = {}
            
            for loss_type in loss_types:
                print(f"\n{'='*60}")
                print(f"TESTING {loss_type.upper()}")
                if num_runs > 1:
                    print(f"Run {run_idx + 1}/{num_runs}")
                print(f"{'='*60}")
                
                # Create fresh model
                effective_num_features = self.num_features if self.num_features is not None else loader_num_features
                model = self.trainer.create_model(effective_num_features)
                
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
                      f"Cum AUC={eval_results['cumulative_auc']:.4f}, "
                      f"Inc AUC={eval_results['incident_auc']:.4f}, "
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
                'dataset_auc_time': self.dataset_config.auc_time,
                'mixed_precision': self.use_mixed_precision,
                'gpu_optimizations': {
                    'torch_compile': hasattr(torch, 'compile') and self.device.type == 'cuda',
                    'mixed_precision': self.use_mixed_precision,
                    'optimized_batch_size': True,
                    'pre_initialized_losses': True,
                    'full_test_evaluation': True
                },
                'horizon_kind': self.trainer.horizon_kind,
                'hetero_tau': self.trainer.hetero_tau,
                'rel_factor': self.trainer.rel_factor,
                'temperature': self.trainer.temperature,
                'use_uncertainty_weighting': self.trainer.use_uncertainty_weighting
            }
            
            saved_files = self.logger.save_results(final_results, run_info, execution_time)
            
            print(f"\n{'='*80}")
            print("RESULTS SAVED TO FILES")
            print(f"{'='*80}")
            for file_type, filepath in saved_files.items():
                if file_type == 'comprehensive':
                    print(f"COMPREHENSIVE JSON (with individual run metrics): {filepath}")
                elif file_type == 'losses':
                    print(f"TRAINING/VALIDATION LOSSES CSV: {filepath}")
                else:
                    print(f"{file_type.upper()}: {filepath}")
        
        return final_results
    
    def _aggregate_multiple_runs(self, all_runs: List[Dict[str, Dict]]) -> Dict[str, Dict]:
        """Aggregate results from multiple runs (mean and std) while preserving individual run data."""
        aggregated = {}
        loss_types = all_runs[0].keys()
        
        for loss_type in loss_types:
            # Collect metrics across all runs
            harrell_scores = [run[loss_type]['evaluation']['harrell_cindex'] for run in all_runs]
            uno_scores = [run[loss_type]['evaluation']['uno_cindex'] for run in all_runs]
            cumulative_auc_scores = [run[loss_type]['evaluation']['cumulative_auc'] for run in all_runs]
            incident_auc_scores = [run[loss_type]['evaluation']['incident_auc'] for run in all_runs]
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
                    'cumulative_auc': np.mean(cumulative_auc_scores),
                    'cumulative_auc_std': np.std(cumulative_auc_scores),
                    'incident_auc': np.mean(incident_auc_scores),
                    'incident_auc_std': np.std(incident_auc_scores),
                    'brier_score': np.mean(brier_scores) if brier_scores else float('nan'),
                    'brier_score_std': np.std(brier_scores) if brier_scores else float('nan'),
                    'run_details': {
                        'num_runs': len(all_runs),
                        'individual_runs': all_runs
                    }
                }
            }
        
        return aggregated


if __name__ == "__main__":
    import argparse
    from data_loaders import DATA_LOADERS
    parser = argparse.ArgumentParser(description="Run survival analysis benchmark")
    parser.add_argument('--dataset', required=True, choices=DATA_LOADERS.keys(), help='Dataset name')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--loss-types', nargs='+', default=None, help='Loss types to evaluate')
    parser.add_argument('--horizon-kind', type=str, default='exp', help='Horizon kind for CPHL/Hybrid losses')
    parser.add_argument('--hetero-tau', action='store_true', help='Use heterogeneous tau head')
    parser.add_argument('--rel-factor', type=float, default=0.5, help='Relative factor for horizon loss scale')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for horizon loss')
    parser.add_argument('--no-uncertainty-weighting', action='store_true', help='Disable uncertainty weighting in Hybrid')
    parser.add_argument('--num-features', type=int, required=True, help='Number of input features for the model')
    args = parser.parse_args()
    data_loader_cls = DATA_LOADERS[args.dataset]
    data_loader = data_loader_cls()
    dataset_config = DATASET_CONFIGS[args.dataset]
    runner = BenchmarkRunner(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=data_loader.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        random_seed=args.seed,
        loss_types=args.loss_types,
        horizon_kind=args.horizon_kind,
        hetero_tau=args.hetero_tau,
        rel_factor=args.rel_factor,
        temperature=args.temperature,
        use_uncertainty_weighting=not args.no_uncertainty_weighting,
        num_features=args.num_features,
    )
    runner.run_comparison(num_runs=args.runs)
