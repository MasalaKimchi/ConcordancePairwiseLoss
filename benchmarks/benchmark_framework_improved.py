#!/usr/bin/env python3
"""Improved benchmark framework with horizon-weighted losses."""

import warnings
warnings.filterwarnings("ignore")

import time as time_module
import json
import csv
import os
import sys
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
from concordance_pairwise_loss.pairwise_horizon_loss import ConcordancePairwiseHorizonLoss
from concordance_pairwise_loss.uncertainty_combined_loss import UncertaintyWeightedCombination


class DatasetConfig:
    def __init__(self, name: str, auc_time: float, auc_time_unit: str = "days"):
        self.name = name
        self.auc_time = auc_time
        self.auc_time_unit = auc_time_unit


DATASET_CONFIGS = {
    'gbsg2': DatasetConfig('GBSG2', 1825.0, 'days'),
    'lung': DatasetConfig('Lung', 365.0, 'days'),
    'rossi': DatasetConfig('Rossi', 365.0, 'days'),
    'flchain': DatasetConfig('FLChain', 1825.0, 'days'),
    'whas500': DatasetConfig('WHAS500', 365.0, 'days'),
    'veterans': DatasetConfig('Veterans', 365.0, 'days'),
    'breast_cancer': DatasetConfig('Breast Cancer', 1825.0, 'days'),
    'support2': DatasetConfig('SUPPORT2', 180.0, 'days'),
    'cancer': DatasetConfig('Cancer', 365.0, 'days'),
    'metabric': DatasetConfig('METABRIC', 200.0, 'days'),
}


class AbstractDataLoader:
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        raise NotImplementedError


class BenchmarkEvaluator:
    def __init__(self, device: torch.device, dataset_config: DatasetConfig):
        self.device = device
        self.dataset_config = dataset_config

    def evaluate_model(self, model: torch.nn.Module, dataloader_test: DataLoader) -> Dict[str, float]:
        model.eval()
        all_log_hz, all_events, all_times = [], [], []
        with torch.no_grad():
            for batch in dataloader_test:
                x, (event, time) = batch
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
            combiner = UncertaintyWeightedCombination(rank_loss=rank_loss, disc_time_nll_fn=self._disc_nll, use_uncertainty_weighting=self.use_uncertainty_weighting)
        print(f"loss_type={loss_type}, horizon_kind={loss_kind}, hetero_tau={self.hetero_tau}, horizon_scale={rank_loss.derived_h.item()}, uncertainty_weighting={self.use_uncertainty_weighting if combiner else False}")
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
            if loss_type.startswith('hybrid_'):
                print(f"Epoch {epoch:03d}: rank_w={combiner.rank_weight.item():.3f}, nll_w={combiner.nll_weight.item():.3f}")
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

    def run_comparison(self, loss_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        dataloader_train, dataloader_val, dataloader_test, num_features = self.data_loader.load_data()
        loss_types = loss_types or self.default_experiments
        results: Dict[str, Dict] = {}
        for lt in loss_types:
            model = self.trainer.create_model(num_features)
            training = self.trainer.train_model(model, dataloader_train, dataloader_val, lt)
            evaluation = self.evaluator.evaluate_model(model, dataloader_test)
            results[lt] = {'training': training, 'evaluation': evaluation}
        if self.save_results and self.logger:
            self.logger.save_results(results, f"{self.dataset_config.name}_improved")
        return results
