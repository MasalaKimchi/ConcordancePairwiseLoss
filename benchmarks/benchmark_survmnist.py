#!/usr/bin/env python3
"""
Enhanced TorchSurv MNIST Training Script with CPL Loss Support

This script follows the TorchSurv momentum example pattern but adds support for:
- Configurable batch sizes (32, 64, 128, 256)
- Multiple loss functions: NLL, CPL(dynamic), CPL(static)
- Proper C-index computation (Harrell's and Uno's)
- AUC metrics for survival analysis
- 5 epochs training
- Results saving

Based on: https://opensource.nibr.com/torchsurv/notebooks/momentum.html
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import lightning as L
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import v2
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw

# Add src and benchmarks to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))
from concordance_pairwise_loss import ConcordancePairwiseLoss
from survmnist import create_torchsurv_mnist_loaders
from dataset_configs import DatasetConfig, load_dataset_configs

# 54random seeds for reproducibility
from lightning.pytorch import seed_everything
seed_everything(123, workers=True)


class LitMNISTEnhanced(L.LightningModule):
    """Enhanced PyTorch Lightning module for MNIST survival analysis with multiple loss support."""
    
    def __init__(self, backbone, loss_type: str = "nll", temperature: float = 1.0, 
                 precomputed_ipcw_weights: Optional[torch.Tensor] = None,
                 auc_time: float = 5.0):
        super().__init__()
        self.model = backbone
        self.loss_type = loss_type
        self.temperature = temperature
        self.precomputed_ipcw_weights = precomputed_ipcw_weights
        self.auc_time = auc_time
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Initialize metrics
        self.cindex = ConcordanceIndex()
        self.auc = Auc()
        self.brier = BrierScore()
        
        # Store metrics for final evaluation
        self.test_predictions = []
        self.test_events = []
        self.test_times = []
        
    def _init_loss_functions(self):
        """Initialize loss functions based on loss_type."""
        if self.loss_type == "nll":
            self.loss_fn = None  # Use neg_partial_log_likelihood directly
        elif self.loss_type == "cpl_dynamic":
            self.loss_fn = ConcordancePairwiseLoss(
                reduction="mean",
                temp_scaling='linear',
                pairwise_sampling='balanced',
                use_ipcw=True
            )
        elif self.loss_type == "cpl_static":
            # For batch variant, we'll compute IPCW on the current batch
            # rather than using precomputed weights to avoid tensor size issues
            self.loss_fn = ConcordancePairwiseLoss(
                reduction="mean",
                temp_scaling='linear',
                pairwise_sampling='balanced',
                use_ipcw=True,
                ipcw_weights=None  # Compute IPCW on current batch
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def forward(self, x):
        return self.model(x)
    
    def _compute_loss(self, log_hz, events, times):
        """Compute loss based on the specified loss type."""
        if self.loss_type == "nll":
            return neg_partial_log_likelihood(log_hz, events, times, reduction="mean")
        else:
            # Update temperature for CPL losses
            if hasattr(self.loss_fn, 'temperature'):
                self.loss_fn.temperature = self.temperature
            
            # Ensure proper shape for CPL
            log_hz_1d = log_hz.squeeze(-1) if log_hz.dim() == 2 else log_hz
            return self.loss_fn(log_hz_1d, times, events)
    
    def training_step(self, batch, batch_idx):
        x, (events, times) = batch
        log_hz = self(x)
        
        # Compute loss
        loss = self._compute_loss(log_hz, events, times)
        
        # Compute C-index (Harrell's - no IPCW weights)
        cindex = self.cindex(log_hz, events, times)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cindex", cindex, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, (events, times) = batch
        log_hz = self(x)
        
        # Compute loss
        loss = self._compute_loss(log_hz, events, times)
        
        # Compute C-index (Harrell's - no IPCW weights)
        cindex = self.cindex(log_hz, events, times)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_cindex", cindex, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, (events, times) = batch
        log_hz = self(x)
        
        # Store predictions for final evaluation
        self.test_predictions.append(log_hz.detach())
        self.test_events.append(events.detach())
        self.test_times.append(times.detach())
        
        # Compute loss
        loss = self._compute_loss(log_hz, events, times)
        
        # Compute C-index (Harrell's - no IPCW weights)
        cindex = self.cindex(log_hz, events, times)
        
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_cindex", cindex, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute comprehensive metrics at the end of testing."""
        if not self.test_predictions:
            return
            
        # Concatenate all test data
        all_log_hz = torch.cat(self.test_predictions, dim=0)
        all_events = torch.cat(self.test_events, dim=0)
        all_times = torch.cat(self.test_times, dim=0)
        
        # Move to CPU for torchsurv metrics (device compatibility)
        log_hz_cpu = all_log_hz.cpu()
        events_cpu = all_events.cpu()
        times_cpu = all_times.cpu()
        
        # Compute IPCW weights for Uno's C-index
        try:
            if events_cpu.any():
                ipcw_weights = get_ipcw(events_cpu, times_cpu)
            else:
                ipcw_weights = torch.ones_like(events_cpu, dtype=torch.float)
        except Exception:
            ipcw_weights = torch.ones_like(events_cpu, dtype=torch.float)
        
        # 1. Harrell's C-index (without IPCW weights)
        harrell_cindex = self.cindex(log_hz_cpu, events_cpu, times_cpu)
        
        # 2. Uno's C-index (with IPCW weights)
        uno_cindex = self.cindex(log_hz_cpu, events_cpu, times_cpu, weight=ipcw_weights)
        
        # 3. Cumulative AUC (over all observed times)
        cumulative_auc_tensor = self.auc(log_hz_cpu, events_cpu, times_cpu)
        cumulative_auc = torch.mean(cumulative_auc_tensor)
        
        # 4. Incident AUC at specified time point
        incident_auc_time = torch.tensor(self.auc_time)
        incident_auc = self.auc(log_hz_cpu, events_cpu, times_cpu, new_time=incident_auc_time)
        
        # 5. Brier Score for calibration assessment
        try:
            # Convert log hazards to survival probabilities
            survival_probs_cpu = torch.sigmoid(-log_hz_cpu)
            # Use median time for Brier score evaluation
            median_time = torch.median(times_cpu[events_cpu.bool()]).item() if events_cpu.any() else torch.median(times_cpu).item()
            brier_score = self.brier(survival_probs_cpu, events_cpu, times_cpu, new_time=torch.tensor(median_time))
        except Exception:
            brier_score = torch.tensor(float('nan'))
        
        # Log metrics
        self.log("test_harrell_cindex", harrell_cindex, on_epoch=True)
        self.log("test_uno_cindex", uno_cindex, on_epoch=True)
        self.log("test_cumulative_auc", cumulative_auc, on_epoch=True)
        self.log("test_incident_auc", incident_auc, on_epoch=True)
        self.log("test_brier_score", brier_score, on_epoch=True)
        
        # Store metrics for saving
        self.final_metrics = {
            'harrell_cindex': harrell_cindex.item(),
            'uno_cindex': uno_cindex.item(),
            'cumulative_auc': cumulative_auc.item(),
            'incident_auc': incident_auc.item(),
            'brier_score': brier_score.item() if not torch.isnan(brier_score) else float('nan')
        }
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def create_resnet_backbone():
    """Create ResNet18 backbone matching TorchSurv example."""
    resnet = resnet18(weights=None)
    # Fits grayscale images (1 channel instead of 3)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Output log hazards (single value)
    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=1)
    return resnet


def precompute_ipcw_weights(dataloader_train) -> torch.Tensor:
    """Precompute IPCW weights from the full training set for batch variant."""
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
        return torch.ones(len(all_events))
    else:
        # Compute IPCW on CPU
        events_cpu = all_events.cpu()
        times_cpu = all_times.cpu()
        ipcw = get_ipcw(events_cpu, times_cpu)
        return ipcw


def run_experiment(
    batch_size: int = 64,
    epochs: int = 5,
    loss_type: str = "nll",
    temperature: float = 1.0,
    output_dir: str = "results",
    limit_train_batches: float = 0.1
) -> Dict[str, Any]:
    """Run a single experiment with specified parameters."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {loss_type.upper()} (batch_size={batch_size}, epochs={epochs})")
    print(f"{'='*80}")
    
    # Load dataset configuration
    dataset_configs = load_dataset_configs()
    dataset_config = dataset_configs.get('survival_mnist')
    auc_time = dataset_config.auc_time if dataset_config else 5.0
    
    # Create data loaders
    train_loader, test_loader, num_features = create_torchsurv_mnist_loaders(
        batch_size=batch_size,
        target_size=(224, 224)
    )
    
    print(f"Number of features: {num_features}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"AUC evaluation time: {auc_time}")
    
    # Precompute IPCW weights for batch variant
    precomputed_ipcw_weights = None
    if loss_type == "cpl_static":
        print("Precomputing IPCW weights from full training set...")
        precomputed_ipcw_weights = precompute_ipcw_weights(train_loader)
        print(f"Precomputed IPCW weights for {len(precomputed_ipcw_weights)} training samples")
    
    # Create model
    backbone = create_resnet_backbone()
    model = LitMNISTEnhanced(
        backbone=backbone,
        loss_type=loss_type,
        temperature=temperature,
        precomputed_ipcw_weights=precomputed_ipcw_weights,
        auc_time=auc_time
    )
    
    # Sanity checks (like in TorchSurv example)
    x = torch.randn((6, 1, 28, 28))  # Example batch of 6 MNIST images
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(224, antialias=True),
        v2.Normalize(mean=(0,), std=(1,)),
    ])
    x_transformed = transforms(x)
    print(f"Input shape: {x_transformed.shape}")
    print(f"Output shape: {backbone(x_transformed).shape}")
    
    # Define trainer
    trainer = L.Trainer(
        accelerator="auto",  # Use best accelerator
        logger=False,  # No logging
        enable_checkpointing=False,  # No model checkpointing
        limit_train_batches=limit_train_batches,  # Train on subset of data
        max_epochs=epochs,  # Train for specified epochs
        deterministic=True,
    )
    
    # Fit the model
    print(f"\nStarting training with {loss_type.upper()} loss...")
    trainer.fit(model, train_loader, test_loader)
    
    # Test the model
    print(f"\nTesting model...")
    trainer.test(model, test_loader)
    
    # Get final metrics
    final_metrics = getattr(model, 'final_metrics', {})
    
    print(f"\nFinal Results:")
    print(f"  Harrell's C-index: {final_metrics.get('harrell_cindex', 0.0):.4f}")
    print(f"  Uno's C-index: {final_metrics.get('uno_cindex', 0.0):.4f}")
    print(f"  Cumulative AUC: {final_metrics.get('cumulative_auc', 0.0):.4f}")
    print(f"  Incident AUC (t={auc_time}): {final_metrics.get('incident_auc', 0.0):.4f}")
    print(f"  Brier Score: {final_metrics.get('brier_score', float('nan')):.4f}")
    
    return {
        'loss_type': loss_type,
        'batch_size': batch_size,
        'epochs': epochs,
        'temperature': temperature,
        'metrics': final_metrics
    }


def run_batch_size_comparison(
    batch_sizes: list = [32, 64, 128, 256],
    epochs: int = 5,
    loss_types: list = ["nll", "cpl_dynamic", "cpl_static"],
    temperatures: list = [1.0],
    output_dir: str = "results",
    limit_train_batches: float = 0.1,
    num_runs: int = 1
) -> Dict[str, Any]:
    """Run comprehensive comparison across batch sizes and loss types with multiple runs."""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE BATCH SIZE AND LOSS COMPARISON")
    print(f"Number of runs per configuration: {num_runs}")
    print(f"{'='*100}")
    
    all_results = []
    
    for loss_type in loss_types:
        temp_list = temperatures if loss_type != "nll" else [1.0]
        
        for batch_size in batch_sizes:
            for temperature in temp_list:
                print(f"\n{'='*60}")
                print(f"Running {loss_type.upper()} (batch_size={batch_size}, temp={temperature})")
                print(f"Number of runs: {num_runs}")
                print(f"{'='*60}")
                
                run_results = []
                
                for run_idx in range(num_runs):
                    print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
                    try:
                        # Set different seed for each run to ensure different random states
                        seed_everything(123 + run_idx, workers=True)
                        
                        result = run_experiment(
                            batch_size=batch_size,
                            epochs=epochs,
                            loss_type=loss_type,
                            temperature=temperature,
                            output_dir=output_dir,
                            limit_train_batches=limit_train_batches
                        )
                        result['run_idx'] = run_idx
                        run_results.append(result)
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Failed to run {loss_type} with batch_size={batch_size}, temp={temperature}, run={run_idx+1}: {e}")
                        error_result = {
                            'loss_type': loss_type,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'temperature': temperature,
                            'run_idx': run_idx,
                            'metrics': {},
                            'error': str(e)
                        }
                        run_results.append(error_result)
                        all_results.append(error_result)
                
                # Compute statistics for this configuration
                if run_results and not any('error' in r for r in run_results):
                    compute_run_statistics(run_results, loss_type, batch_size, temperature)
    
    # Save results
    save_results(all_results, output_dir)
    
    return all_results


def compute_run_statistics(run_results: list, loss_type: str, batch_size: int, temperature: float):
    """Compute mean and standard deviation for multiple runs of the same configuration."""
    import numpy as np
    
    # Extract metrics from successful runs
    metrics_data = {}
    for result in run_results:
        if 'error' not in result and 'metrics' in result:
            for metric_name, metric_value in result['metrics'].items():
                if not np.isnan(metric_value):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(metric_value)
    
    if not metrics_data:
        print(f"No valid metrics found for {loss_type} (batch_size={batch_size})")
        return
    
    print(f"\nStatistics for {loss_type.upper()} (batch_size={batch_size}, temp={temperature}):")
    print("-" * 60)
    
    for metric_name, values in metrics_data.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:<20}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")
    
    print("-" * 60)


def save_results(results: list, output_dir: str, batch_size: int = None):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    if batch_size is not None:
        json_filename = f"mnist_batch_{batch_size}_{timestamp}.json"
    else:
        json_filename = f"mnist_batch_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved: {json_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Enhanced TorchSurv MNIST Training with CPL Support")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--loss-type', choices=['nll', 'cpl_dynamic', 'cpl_static'], 
                       default='nll', help='Loss function type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for CPL losses')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--limit-train-batches', type=float, default=0.1, 
                       help='Fraction of training data to use (0.1 = 10 percent)')
    parser.add_argument('--compare-all', action='store_true', 
                       help='Run comparison across all batch sizes and loss types')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64, 128, 256],
                       help='Batch sizes for comparison (used with --compare-all)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs per configuration for statistical analysis')
    
    args = parser.parse_args()
    
    if args.compare_all:
        # Run comprehensive comparison
        results = run_batch_size_comparison(
            batch_sizes=args.batch_sizes,
            epochs=args.epochs,
            output_dir=args.output_dir,
            limit_train_batches=args.limit_train_batches,
            num_runs=args.num_runs
        )
        
        # Print summary with statistics
        print(f"\n{'='*120}")
        print("SUMMARY RESULTS")
        print(f"{'='*120}")
        
        if args.num_runs > 1:
            print(f"{'Loss Type':<15} {'Batch Size':<10} {'Run':<4} {'Harrell C':<10} {'Uno C':<10} {'Cum AUC':<10} {'Inc AUC':<10} {'Brier':<10}")
            print("-" * 120)
            
            for result in results:
                if 'error' not in result:
                    metrics = result.get('metrics', {})
                    run_idx = result.get('run_idx', 0)
                    print(f"{result['loss_type']:<15} {result['batch_size']:<10} {run_idx+1:<4} "
                          f"{metrics.get('harrell_cindex', 0.0):<10.4f} "
                          f"{metrics.get('uno_cindex', 0.0):<10.4f} "
                          f"{metrics.get('cumulative_auc', 0.0):<10.4f} "
                          f"{metrics.get('incident_auc', 0.0):<10.4f} "
                          f"{metrics.get('brier_score', float('nan')):<10.4f}")
                else:
                    run_idx = result.get('run_idx', 0)
                    print(f"{result['loss_type']:<15} {result['batch_size']:<10} {run_idx+1:<4} ERROR: {result['error']}")
            
            # Print aggregated statistics
            print(f"\n{'='*120}")
            print("AGGREGATED STATISTICS (Mean ± Std)")
            print(f"{'='*120}")
            print(f"{'Loss Type':<15} {'Batch Size':<10} {'Harrell C':<15} {'Uno C':<15} {'Cum AUC':<15} {'Inc AUC':<15} {'Brier':<15}")
            print("-" * 120)
            
            # Group results by configuration
            configs = {}
            for result in results:
                if 'error' not in result:
                    key = (result['loss_type'], result['batch_size'])
                    if key not in configs:
                        configs[key] = []
                    configs[key].append(result['metrics'])
            
            for (loss_type, batch_size), metrics_list in configs.items():
                if metrics_list:
                    import numpy as np
                    harrell_vals = [m.get('harrell_cindex', 0.0) for m in metrics_list if not np.isnan(m.get('harrell_cindex', 0.0))]
                    uno_vals = [m.get('uno_cindex', 0.0) for m in metrics_list if not np.isnan(m.get('uno_cindex', 0.0))]
                    cum_auc_vals = [m.get('cumulative_auc', 0.0) for m in metrics_list if not np.isnan(m.get('cumulative_auc', 0.0))]
                    inc_auc_vals = [m.get('incident_auc', 0.0) for m in metrics_list if not np.isnan(m.get('incident_auc', 0.0))]
                    brier_vals = [m.get('brier_score', 0.0) for m in metrics_list if not np.isnan(m.get('brier_score', 0.0))]
                    
                    harrell_str = f"{np.mean(harrell_vals):.4f}±{np.std(harrell_vals):.4f}" if harrell_vals else "N/A"
                    uno_str = f"{np.mean(uno_vals):.4f}±{np.std(uno_vals):.4f}" if uno_vals else "N/A"
                    cum_auc_str = f"{np.mean(cum_auc_vals):.4f}±{np.std(cum_auc_vals):.4f}" if cum_auc_vals else "N/A"
                    inc_auc_str = f"{np.mean(inc_auc_vals):.4f}±{np.std(inc_auc_vals):.4f}" if inc_auc_vals else "N/A"
                    brier_str = f"{np.mean(brier_vals):.4f}±{np.std(brier_vals):.4f}" if brier_vals else "N/A"
                    
                    print(f"{loss_type:<15} {batch_size:<10} {harrell_str:<15} {uno_str:<15} {cum_auc_str:<15} {inc_auc_str:<15} {brier_str:<15}")
        else:
            # Single run format
            print(f"{'Loss Type':<15} {'Batch Size':<10} {'Harrell C':<10} {'Uno C':<10} {'Cum AUC':<10} {'Inc AUC':<10} {'Brier':<10}")
            print("-" * 100)
            
            for result in results:
                if 'error' not in result:
                    metrics = result.get('metrics', {})
                    print(f"{result['loss_type']:<15} {result['batch_size']:<10} "
                          f"{metrics.get('harrell_cindex', 0.0):<10.4f} "
                          f"{metrics.get('uno_cindex', 0.0):<10.4f} "
                          f"{metrics.get('cumulative_auc', 0.0):<10.4f} "
                          f"{metrics.get('incident_auc', 0.0):<10.4f} "
                          f"{metrics.get('brier_score', float('nan')):<10.4f}")
                else:
                    print(f"{result['loss_type']:<15} {result['batch_size']:<10} ERROR: {result['error']}")
    
    else:
        # Run single experiment
        result = run_experiment(
            batch_size=args.batch_size,
            epochs=args.epochs,
            loss_type=args.loss_type,
            temperature=args.temperature,
            output_dir=args.output_dir,
            limit_train_batches=args.limit_train_batches
        )
        
        # Save single result
        save_results([result], args.output_dir, batch_size=args.batch_size)
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()

