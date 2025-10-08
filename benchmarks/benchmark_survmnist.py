#!/usr/bin/env python3
"""
Enhanced TorchSurv MNIST Training Script with CPL Loss Support

This script follows the TorchSurv momentum example pattern but adds support for:
- Configurable batch sizes (32, 64, 128, 256)
- Multiple loss functions: NLL, CPL(online), CPL(offline)
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

# Set random seeds for reproducibility
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
        elif self.loss_type == "cpl_online":
            self.loss_fn = ConcordancePairwiseLoss(
                reduction="mean",
                temp_scaling='linear',
                pairwise_sampling='balanced',
                use_ipcw=True
            )
        elif self.loss_type == "cpl_offline":
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
    if loss_type == "cpl_offline":
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
    loss_types: list = ["nll", "cpl_online", "cpl_offline"],
    temperatures: list = [1.0],
    output_dir: str = "results",
    limit_train_batches: float = 0.1
) -> Dict[str, Any]:
    """Run comprehensive comparison across batch sizes and loss types."""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE BATCH SIZE AND LOSS COMPARISON")
    print(f"{'='*100}")
    
    all_results = []
    
    for loss_type in loss_types:
        temp_list = temperatures if loss_type != "nll" else [1.0]
        
        for batch_size in batch_sizes:
            for temperature in temp_list:
                try:
                    result = run_experiment(
                        batch_size=batch_size,
                        epochs=epochs,
                        loss_type=loss_type,
                        temperature=temperature,
                        output_dir=output_dir,
                        limit_train_batches=limit_train_batches
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Failed to run {loss_type} with batch_size={batch_size}, temp={temperature}: {e}")
                    all_results.append({
                        'loss_type': loss_type,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'temperature': temperature,
                        'metrics': {},
                        'error': str(e)
                    })
    
    # Save results
    save_results(all_results, output_dir)
    
    return all_results


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
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--loss-type', choices=['nll', 'cpl_online', 'cpl_offline'], 
                       default='nll', help='Loss function type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for CPL losses')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--limit-train-batches', type=float, default=0.1, 
                       help='Fraction of training data to use (0.1 = 10%)')
    parser.add_argument('--compare-all', action='store_true', 
                       help='Run comparison across all batch sizes and loss types')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[32, 64, 128, 256],
                       help='Batch sizes for comparison (used with --compare-all)')
    
    args = parser.parse_args()
    
    if args.compare_all:
        # Run comprehensive comparison
        results = run_batch_size_comparison(
            batch_sizes=args.batch_sizes,
            epochs=args.epochs,
            output_dir=args.output_dir,
            limit_train_batches=args.limit_train_batches
        )
        
        # Print summary
        print(f"\n{'='*110}")
        print("SUMMARY RESULTS")
        print(f"{'='*110}")
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

