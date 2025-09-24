#!/usr/bin/env python3
"""
Extended Benchmark Runner that adds --hidden-dim without modifying benchmark_framework.py.

This thin wrapper reuses all logic from benchmark_framework but swaps in a
trainer that honors a user-specified hidden layer width for the MLP.
"""

import argparse
import os
import sys
from typing import Optional

import torch

# Ensure we can import the original framework
sys.path.append(os.path.dirname(__file__))
from benchmark_framework import (
    BenchmarkRunner as BaseRunner,
    BenchmarkTrainer as BaseTrainer,
    DATASET_CONFIGS,
)
from data_loaders import DATA_LOADERS


class BenchmarkTrainerExt(BaseTrainer):
    def __init__(self, *args, hidden_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

    def create_model(self, num_features: int) -> torch.nn.Module:
        """Override to support configurable hidden width."""
        backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(num_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        risk_head = torch.nn.Linear(self.hidden_dim, 1)
        if self.hetero_tau:
            log_tau_head = torch.nn.Linear(self.hidden_dim, 1)
            model = self._RiskModel(backbone, risk_head, log_tau_head)
        else:
            model = self._RiskModel(backbone, risk_head)
        model = model.to(self.device)

        print(f"Using extended model with hidden_dim={self.hidden_dim}")
        return model


class BenchmarkRunnerExt(BaseRunner):
    def __init__(
        self,
        *args,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Replace trainer with extended version preserving existing settings
        self.trainer = BenchmarkTrainerExt(
            self.device,
            self.epochs,
            self.learning_rate,
            use_mixed_precision=self.use_mixed_precision,
            dataset_config=self.dataset_config,
            horizon_kind=self.trainer.horizon_kind,
            hetero_tau=self.trainer.hetero_tau,
            rel_factor=self.trainer.rel_factor,
            temperature=self.trainer.temperature,
            use_uncertainty_weighting=self.trainer.use_uncertainty_weighting,
            hidden_dim=hidden_dim,
        )


def main():
    parser = argparse.ArgumentParser(description="Extended runner with --hidden-dim")
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
    parser.add_argument('--num-features', type=int, default=None, help='Number of input features; if omitted, auto-detect from loader')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer width for MLP')

    args = parser.parse_args()
    data_loader_cls = DATA_LOADERS[args.dataset]
    data_loader = data_loader_cls()
    dataset_config = DATASET_CONFIGS[args.dataset]

    runner = BenchmarkRunnerExt(
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
        hidden_dim=args.hidden_dim,
    )
    runner.run_comparison(num_runs=args.runs)


if __name__ == "__main__":
    main()


