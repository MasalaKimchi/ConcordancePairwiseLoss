#!/usr/bin/env python3
"""Improved benchmark framework with horizon-weighted losses."""

import warnings
warnings.filterwarnings("ignore")

import time as time_module
import json
import csv
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from concordance_pairwise_loss import ConcordancePairwiseLoss, NormalizedLossCombination
from dataset_configs import DatasetConfig, load_dataset_configs
from abstract_data_loader import AbstractDataLoader

from concordance_pairwise_loss.pairwise_horizon_loss import ConcordancePairwiseHorizonLoss
from concordance_pairwise_loss.uncertainty_combined_loss import UncertaintyWeightedCombination
DATASET_CONFIGS = load_dataset_configs()



class BenchmarkEvaluator:
    def __init__(self, device: torch.device, dataset_config: DatasetConfig):
        self.device = device
        self.dataset_config = dataset_config

    def evaluate_model(self, model: torch.nn.Module, dataloader_test: DataLoader) -> Dict[str, float]:
        model.eval()
        all_log_hz, all_events, all_times = [], [], []
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
                if isinstance(out, tuple):
                    out = out[0]
                all_log_hz.append(out)
                all_events.append(event)
                all_times.append(time)
        log_hz = torch.cat(all_log_hz, dim=0)
        event = torch.cat(all_events, dim=0)
        time = torch.cat(all_times, dim=0)
        
        # Ensure correct data types for torchsurv requirements
        event = event.bool()  # torchsurv requires boolean events
        time = time.float()   # torchsurv requires float times
        try:
            if event.any():
                ipcw_weights = get_ipcw(event.cpu(), time.cpu()).to(self.device)
            else:
                ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
        except Exception:
            ipcw_weights = torch.ones_like(event, dtype=torch.float, device=self.device)
        cindex = ConcordanceIndex()
        auc = Auc()
        brier = BrierScore()
        log_hz_cpu = log_hz.cpu()
        event_cpu = event.cpu()
        time_cpu = time.cpu()
        ipcw_weights_cpu = ipcw_weights.cpu()
        harrell_cindex = cindex(log_hz_cpu, event_cpu, time_cpu)
        uno_cindex = cindex(log_hz_cpu, event_cpu, time_cpu, weight=ipcw_weights_cpu)
        cumulative_auc = torch.mean(auc(log_hz_cpu, event_cpu, time_cpu))
        new_time_cpu = torch.tensor(self.dataset_config.auc_time)
        incident_auc = auc(log_hz_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        try:
            survival_probs_cpu = torch.sigmoid(-log_hz_cpu)
            brier_score = brier(survival_probs_cpu, event_cpu, time_cpu, new_time=new_time_cpu)
        except Exception:
            brier_score = torch.tensor(float('nan'))
        return {
            'harrell_cindex': harrell_cindex.item(),
            'uno_cindex': uno_cindex.item(),
            'cumulative_auc': cumulative_auc.item(),
            'incident_auc': incident_auc.item(),
            'brier_score': brier_score.item(),
        }


class BenchmarkTrainerImproved:
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
        self.pairwise_loss_fn = ConcordancePairwiseLoss(reduction="mean", temp_scaling='linear', pairwise_sampling='balanced', use_ipcw=False)
        self.pairwise_loss_ipcw_fn = ConcordancePairwiseLoss(reduction="mean", temp_scaling='linear', pairwise_sampling='balanced', use_ipcw=True)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def _disc_nll(self, scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor, **kwargs) -> torch.Tensor:
        if scores.dim() == 1:
            scores = scores.unsqueeze(1)
        return neg_partial_log_likelihood(scores, events, times, reduction="mean")

    def create_model(self, num_features: int) -> torch.nn.Module:
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
        return model.to(self.device)

    def _times_to_years(self, t: torch.Tensor) -> torch.Tensor:
        if self.dataset_config and self.dataset_config.auc_time_unit == "days":
            return t / 365.25
        return t

    def _median_followup_years(self, dataloader: DataLoader) -> float:
        times = []
        for batch in dataloader:
            _, (_, t) = batch
            times.append(self._times_to_years(t.cpu()))
        return torch.cat(times).median().item()

    def train_model(self, model: torch.nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader, loss_type: str) -> Dict[str, any]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        train_losses: List[float] = []
        val_losses: List[float] = []
        loss_kind = self.horizon_kind
        if loss_type.startswith("cphl_") or loss_type.startswith("hybrid_"):
            loss_kind = loss_type.split("_")[1]
        median_follow = self._median_followup_years(dataloader_train)
        rank_loss = ConcordancePairwiseHorizonLoss(horizon_kind=loss_kind, rel_factor=self.rel_factor, temperature=self.temperature, hetero_tau=self.hetero_tau, reduction="mean")
        if loss_kind != "none":
            rank_loss.set_train_stats(median_follow)
        combiner = None
        if loss_type.startswith("hybrid_"):
            combiner = UncertaintyWeightedCombination(rank_loss=rank_loss, disc_time_nll_fn=self._disc_nll)
        print(f"loss_type={loss_type}, horizon_kind={loss_kind}, hetero_tau={self.hetero_tau}, horizon_scale={rank_loss.derived_h.item():.2f}, uncertainty_weighting={self.use_uncertainty_weighting if combiner else False}")
        loss_combiner = None
        if loss_type in ['normalized_combination', 'normalized_combination_ipcw']:
            loss_combiner = NormalizedLossCombination(total_epochs=self.epochs)
        for epoch in range(self.epochs):
            model.train()
            epoch_total = 0.0
            for batch in dataloader_train:
                x, (event, time) = batch
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                if self.hetero_tau:
                    if not isinstance(out, tuple):
                        raise RuntimeError("hetero_tau=True requires risk and log_tau outputs")
                    risk, log_tau = out
                    log_tau = log_tau.squeeze(-1)
                else:
                    risk = out
                    log_tau = None
                risk_flat = risk.squeeze(-1)
                times_years = self._times_to_years(time)
                events_bool = event.bool()
                if loss_type in ['nll', 'pairwise', 'pairwise_ipcw', 'normalized_combination', 'normalized_combination_ipcw']:
                    nll_loss = neg_partial_log_likelihood(risk, event, time, reduction="mean")
                    pair_loss = self.pairwise_loss_fn(risk, time, event)
                    pair_loss_ipcw = self.pairwise_loss_ipcw_fn(risk, time, event)
                    if loss_type == 'nll':
                        total_loss = nll_loss
                    elif loss_type == 'pairwise':
                        total_loss = pair_loss
                    elif loss_type == 'pairwise_ipcw':
                        total_loss = pair_loss_ipcw
                    elif loss_type == 'normalized_combination':
                        nll_w, pair_w = loss_combiner.get_weights_scale_balanced(epoch, nll_loss.item(), pair_loss.item())
                        total_loss = nll_w * nll_loss + pair_w * pair_loss
                    elif loss_type == 'normalized_combination_ipcw':
                        nll_w, pair_w = loss_combiner.get_weights_scale_balanced(epoch, nll_loss.item(), pair_loss_ipcw.item())
                        total_loss = nll_w * nll_loss + pair_w * pair_loss_ipcw
                elif loss_type.startswith('cphl_'):
                    total_loss = rank_loss(risk_flat, times_years, events_bool, log_tau)
                elif loss_type.startswith('hybrid_'):
                    total_loss = combiner(risk_flat, times_years, events_bool, log_tau=log_tau)
                else:
                    total_loss = neg_partial_log_likelihood(risk, event, time, reduction="mean")
                if self.use_mixed_precision:
                    self.scaler.scale(total_loss).backward()
                    if any(p.grad is not None for p in model.parameters()):
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                epoch_total += total_loss.item()
            model.eval()
            with torch.no_grad():
                x_val, (event_val, time_val) = next(iter(dataloader_val))
                x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                out = model(x_val)
                if self.hetero_tau:
                    risk_val, log_tau_val = out
                    log_tau_val = log_tau_val.squeeze(-1)
                else:
                    risk_val = out
                    log_tau_val = None
                risk_val_flat = risk_val.squeeze(-1)
                times_val_years = self._times_to_years(time_val)
                events_val_bool = event_val.bool()
                nll_val = neg_partial_log_likelihood(risk_val, event_val, time_val, reduction="mean")
                pair_val = self.pairwise_loss_fn(risk_val, time_val, event_val)
                pair_val_ipcw = self.pairwise_loss_ipcw_fn(risk_val, time_val, event_val)
                if loss_type == 'normalized_combination' and loss_combiner:
                    nw, pw = loss_combiner.get_weights_scale_balanced(epoch, nll_val.item(), pair_val.item())
                    val_loss = nw * nll_val + pw * pair_val
                elif loss_type == 'normalized_combination_ipcw' and loss_combiner:
                    nw, pw = loss_combiner.get_weights_scale_balanced(epoch, nll_val.item(), pair_val_ipcw.item())
                    val_loss = nw * nll_val + pw * pair_val_ipcw
                elif loss_type == 'pairwise_ipcw':
                    val_loss = pair_val_ipcw
                elif loss_type.startswith('cphl_'):
                    val_loss = rank_loss(risk_val_flat, times_val_years, events_val_bool, log_tau_val)
                elif loss_type.startswith('hybrid_'):
                    val_loss = combiner(risk_val_flat, times_val_years, events_val_bool, log_tau=log_tau_val)
                else:
                    val_loss = nll_val + pair_val
            train_losses.append(epoch_total)
            val_losses.append(val_loss.item())
            if self.epochs >= 10 and epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch:03d}: Total={epoch_total:.4f}")
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weight_evolution': loss_combiner.weight_history if loss_combiner else []
        }


def _sanity_check_rank_loss() -> None:
    scores = torch.tensor([0.1, -0.2, 0.3])
    times = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1, 1, 0], dtype=torch.bool)
    mean_loss_fn = ConcordancePairwiseHorizonLoss(horizon_kind="none", reduction="mean")
    base = mean_loss_fn(scores, times, events)
    improved = mean_loss_fn(torch.tensor([2.0, 1.0, -1.0]), times, events)
    assert improved < base
    sum_loss_fn = ConcordancePairwiseHorizonLoss(horizon_kind="none", reduction="sum")
    numerator = sum_loss_fn(scores, times, events)
    weights = numerator / base
    assert torch.allclose(base, numerator / weights, atol=1e-6)


_sanity_check_rank_loss()


class ResultsLogger:
    def __init__(self, dataset_name: str, output_dir: str = "results"):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: Dict[str, Dict], filename: str) -> None:
        json_path = os.path.join(self.output_dir, f"{filename}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        csv_path = os.path.join(self.output_dir, f"{filename}_losses.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["loss_type", "train_loss", "val_loss"])
            for loss_type, vals in results.items():
                writer.writerow([loss_type, sum(vals['training']['train_losses']), sum(vals['training']['val_losses'])])


class BenchmarkRunnerImproved:
    def __init__(self, data_loader: AbstractDataLoader, dataset_config: DatasetConfig, batch_size: int = 64, epochs: int = 50, learning_rate: float = 5e-2, output_dir: str = "results", save_results: bool = True, random_seed: int = None, use_mixed_precision: bool = True, loss_type: str = 'nll', horizon_kind: str = 'exp', hetero_tau: bool = False, rel_factor: float = 0.5, temperature: float = 1.0, use_uncertainty_weighting: bool = True):
        self.data_loader = data_loader
        self.dataset_config = dataset_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.save_results = save_results
        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        self.trainer = BenchmarkTrainerImproved(self.device, epochs, learning_rate, use_mixed_precision=use_mixed_precision, dataset_config=dataset_config, horizon_kind=horizon_kind, hetero_tau=hetero_tau, rel_factor=rel_factor, temperature=temperature, use_uncertainty_weighting=use_uncertainty_weighting)
        self.evaluator = BenchmarkEvaluator(self.device, dataset_config)
        self.logger = ResultsLogger(dataset_config.name, output_dir) if save_results else None
        self.loss_type = loss_type
        self.default_experiments = ['nll', 'pairwise', 'cphl_none', 'cphl_exp', 'hybrid_exp']

    def run_comparison(self, loss_types: Optional[List[str]] = None, num_runs: int = 1) -> Dict[str, Dict]:
        """Run complete benchmark comparison with optional multiple runs."""
        start_time = time_module.time()
        
        print("=" * 80)
        print(f"IMPROVED SURVIVAL ANALYSIS COMPARISON - {self.dataset_config.name.upper()} DATASET")
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
            loss_types = loss_types or self.default_experiments
            run_results: Dict[str, Dict] = {}
            
            for lt in loss_types:
                if num_runs > 1:
                    print(f"\n{'='*60}")
                    print(f"TESTING {lt.upper()}")
                    print(f"Run {run_idx + 1}/{num_runs}")
                    print(f"{'='*60}")
                
                model = self.trainer.create_model(num_features)
                training = self.trainer.train_model(model, dataloader_train, dataloader_val, lt)
                evaluation = self.evaluator.evaluate_model(model, dataloader_test)
                run_results[lt] = {'training': training, 'evaluation': evaluation}
                
                # Print results for each loss type
                print(f"Results: Harrell C={evaluation['harrell_cindex']:.4f}, "
                      f"Uno C={evaluation['uno_cindex']:.4f}, "
                      f"Cum AUC={evaluation['cumulative_auc']:.4f}, "
                      f"Inc AUC={evaluation['incident_auc']:.4f}, "
                      f"Brier={evaluation['brier_score']:.4f}")
            
            all_runs_results.append(run_results)
        
        # Aggregate results across runs
        if num_runs == 1:
            final_results = all_runs_results[0]
        else:
            final_results = self._aggregate_multiple_runs(all_runs_results)
        
        # Calculate execution time
        end_time = time_module.time()
        execution_time = end_time - start_time
        
        # Analyze and visualize results
        self._analyze_results(final_results)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED IN {execution_time/60:.1f} MINUTES ({execution_time:.1f} seconds)")
        print(f"{'='*80}")
        
        if self.save_results and self.logger:
            run_info = {
                'num_runs': num_runs,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'dataset_auc_time': self.dataset_config.auc_time,
                'mixed_precision': self.trainer.use_mixed_precision,
                'horizon_kind': self.trainer.horizon_kind,
                'hetero_tau': self.trainer.hetero_tau,
                'rel_factor': self.trainer.rel_factor,
                'temperature': self.trainer.temperature,
                'use_uncertainty_weighting': self.trainer.use_uncertainty_weighting
            }
            self.logger.save_results(final_results, f"{self.dataset_config.name}_improved")
        
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
    
    def _analyze_results(self, results: Dict[str, Dict]) -> None:
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
            'cphl_none': 'CPHL (none)',
            'cphl_exp': 'CPHL (exp)',
            'hybrid_exp': 'Hybrid (exp)',
            'normalized_combination': 'NLL+CPL',
            'normalized_combination_ipcw': 'NLL+CPL (ipcw)'
        }
        
        # Get NLL baseline for comparison
        nll_harrell = results['nll']['evaluation']['harrell_cindex']
        nll_uno = results['nll']['evaluation']['uno_cindex']
        nll_cumulative_auc = results['nll']['evaluation']['cumulative_auc']
        nll_incident_auc = results['nll']['evaluation']['incident_auc']
        nll_brier = results['nll']['evaluation']['brier_score']
        
        for loss_type, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(loss_type, loss_type.upper())
            
            harrell = eval_result['harrell_cindex']
            uno = eval_result['uno_cindex']
            cumulative_auc = eval_result['cumulative_auc']
            incident_auc = eval_result['incident_auc']
            brier = eval_result['brier_score']
            
            print(f"{method_name:<25} {harrell:<10.4f} {uno:<10.4f} {cumulative_auc:<10.4f} {incident_auc:<10.4f} {brier:<10.4f}")
        
        # Print improvement analysis
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
        
        # Create comprehensive visualizations
        self._create_comprehensive_plots(results)
    
    def _create_comprehensive_plots(self, results: Dict[str, Dict]) -> None:
        """Create comprehensive visualization plots with all metrics and methods."""
        print("\n=== Creating Comprehensive Analysis Plots ===")
        
        # Set up the plotting style with fixed dimensions
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 3x2 layout for comprehensive analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Improved Survival Analysis Comparison - {self.dataset_config.name} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # 1. Concordance indices comparison (Harrell vs Uno)
        self._plot_concordance_comparison(axes[0, 0], results)
        
        # 2. All metrics performance comparison
        self._plot_all_metrics_comparison(axes[0, 1], results)
        
        # 3. New vs Standard methods comparison
        self._plot_new_vs_standard(axes[1, 0], results)
        
        # 4. Training loss evolution
        self._plot_training_evolution(axes[1, 1], results)
        
        # 5. Performance improvement over baseline
        self._plot_improvement_analysis(axes[2, 0], results)
        
        # 6. Horizon scale analysis
        self._plot_horizon_analysis(axes[2, 1], results)
        
        plt.tight_layout()
        
        # Save figure to results folder
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figure_filename = f"{self.dataset_config.name}_improved_analysis_{timestamp}.png"
            figure_path = os.path.join(self.output_dir, figure_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Comprehensive analysis figure saved: {figure_path}")
        
        plt.show()
    
    def _plot_concordance_comparison(self, ax, results):
        """Plot Harrell vs Uno C-index comparison."""
        method_names = ['NLL', 'CPL', 'CPL (ipcw)', 'CPHL (none)', 'CPHL (exp)', 'Hybrid (exp)']
        methods = ['nll', 'pairwise', 'pairwise_ipcw', 'cphl_none', 'cphl_exp', 'hybrid_exp']
        
        harrell_scores = [results[method]['evaluation']['harrell_cindex'] for method in methods if method in results]
        uno_scores = [results[method]['evaluation']['uno_cindex'] for method in methods if method in results]
        
        x = np.arange(len(harrell_scores))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, harrell_scores, width, label="Harrell's C-index", alpha=0.8)
        bars2 = ax.bar(x + width/2, uno_scores, width, label="Uno's C-index", alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('C-index Score')
        ax.set_title('Concordance Index Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names[:len(harrell_scores)], rotation=45, ha='right')
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
        """Plot all metrics for all methods."""
        method_names = ['NLL', 'CPL', 'CPL (ipcw)', 'CPHL (none)', 'CPHL (exp)', 'Hybrid (exp)']
        methods = ['nll', 'pairwise', 'pairwise_ipcw', 'cphl_none', 'cphl_exp', 'hybrid_exp']
        
        # Filter methods that exist in results
        existing_methods = [m for m in methods if m in results]
        existing_names = [method_names[i] for i, m in enumerate(methods) if m in results]
        
        harrell_scores = [results[method]['evaluation']['harrell_cindex'] for method in existing_methods]
        uno_scores = [results[method]['evaluation']['uno_cindex'] for method in existing_methods]
        cumulative_auc_scores = [results[method]['evaluation']['cumulative_auc'] for method in existing_methods]
        incident_auc_scores = [results[method]['evaluation']['incident_auc'] for method in existing_methods]
        
        # For Brier score, invert and normalize (lower is better)
        brier_scores = []
        for method in existing_methods:
            brier = results[method]['evaluation']['brier_score']
            if not np.isnan(brier):
                brier_scores.append(max(0, 1 - brier))
            else:
                brier_scores.append(0)
        
        x = np.arange(len(existing_methods))
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
        ax.set_xticklabels(existing_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    def _plot_new_vs_standard(self, ax, results):
        """Plot comparison between new and standard methods."""
        standard_methods = ['nll', 'pairwise', 'pairwise_ipcw']
        new_methods = ['cphl_none', 'cphl_exp', 'hybrid_exp']
        
        standard_scores = [results[method]['evaluation']['harrell_cindex'] for method in standard_methods if method in results]
        new_scores = [results[method]['evaluation']['harrell_cindex'] for method in new_methods if method in results]
        
        standard_names = ['NLL', 'CPL', 'CPL (ipcw)']
        new_names = ['CPHL (none)', 'CPHL (exp)', 'Hybrid (exp)']
        
        # Filter names to match existing methods
        existing_standard = [name for i, method in enumerate(standard_methods) if method in results for name in [standard_names[i]]]
        existing_new = [name for i, method in enumerate(new_methods) if method in results for name in [new_names[i]]]
        
        all_scores = standard_scores + new_scores
        all_names = existing_standard + existing_new
        
        x = np.arange(len(all_scores))
        colors = ['blue'] * len(standard_scores) + ['red'] * len(new_scores)
        
        ax.bar(x, all_scores, color=colors, alpha=0.7)
        ax.set_xlabel('Method')
        ax.set_ylabel("Harrell's C-index")
        ax.set_title('Standard vs New Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Standard'),
                          Patch(facecolor='red', alpha=0.7, label='New')]
        ax.legend(handles=legend_elements)
    
    def _plot_training_evolution(self, ax, results):
        """Plot training loss evolution for all methods."""
        method_colors = {'nll': 'blue', 'pairwise': 'orange', 'pairwise_ipcw': 'red',
                        'cphl_none': 'green', 'cphl_exp': 'purple', 'hybrid_exp': 'brown'}
        
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
    
    def _plot_improvement_analysis(self, ax, results):
        """Plot improvement over NLL baseline."""
        methods = ['pairwise', 'pairwise_ipcw', 'cphl_none', 'cphl_exp', 'hybrid_exp']
        method_names = ['CPL', 'CPL (ipcw)', 'CPHL (none)', 'CPHL (exp)', 'Hybrid (exp)']
        
        # Get baseline
        nll_harrell = results['nll']['evaluation']['harrell_cindex']
        
        harrell_improvements = []
        existing_methods = []
        existing_names = []
        
        for method, name in zip(methods, method_names):
            if method in results:
                eval_result = results[method]['evaluation']
                harrell_imp = ((eval_result['harrell_cindex'] - nll_harrell) / nll_harrell * 100)
                harrell_improvements.append(harrell_imp)
                existing_methods.append(method)
                existing_names.append(name)
        
        x = np.arange(len(existing_methods))
        colors = ['green' if imp > 0 else 'red' for imp in harrell_improvements]
        
        bars = ax.bar(x, harrell_improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Method')
        ax.set_ylabel('Improvement over NLL (%)')
        ax.set_title('Performance Improvement Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(existing_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, harrell_improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >= 0 else -0.3),
                   f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontweight='bold', fontsize=8)
    
    def _plot_horizon_analysis(self, ax, results):
        """Plot horizon scale analysis for new methods."""
        horizon_methods = ['cphl_none', 'cphl_exp', 'hybrid_exp']
        method_names = ['CPHL (none)', 'CPHL (exp)', 'Hybrid (exp)']
        
        # Get horizon scales (this would need to be stored during training)
        # For now, we'll show a placeholder
        ax.text(0.5, 0.5, 'Horizon Scale Analysis\n(Feature to be implemented)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Horizon Scale Analysis')
        ax.set_xlabel('Method')
        ax.set_ylabel('Horizon Scale')


if __name__ == "__main__":
    import argparse
    from data_loaders import DATA_LOADERS
    parser = argparse.ArgumentParser(description="Run improved survival analysis benchmark")
    parser.add_argument('--dataset', required=True, choices=DATA_LOADERS.keys(), help='Dataset name')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--loss-types', nargs='+', default=None, help='Loss types to evaluate')
    parser.add_argument('--horizon-kind', type=str, default='exp', help='Horizon kind for CPLH losses')
    parser.add_argument('--hetero-tau', action='store_true', help='Use heterogeneous tau')
    parser.add_argument('--rel-factor', type=float, default=0.5, help='Relative factor for horizon loss')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for horizon loss')
    parser.add_argument('--no-uncertainty-weighting', action='store_true', help='Disable uncertainty weighting')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()
    data_loader_cls = DATA_LOADERS[args.dataset]
    data_loader = data_loader_cls()
    dataset_config = DATASET_CONFIGS[args.dataset]
    runner = BenchmarkRunnerImproved(
        data_loader=data_loader,
        dataset_config=dataset_config,
        batch_size=data_loader.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        random_seed=args.seed,
        horizon_kind=args.horizon_kind,
        hetero_tau=args.hetero_tau,
        rel_factor=args.rel_factor,
        temperature=args.temperature,
        use_uncertainty_weighting=not args.no_uncertainty_weighting,
    )
    runner.run_comparison(loss_types=args.loss_types, num_runs=args.runs)
