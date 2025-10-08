"""
MIMIC-IV Utility Classes and Functions

This module contains utility classes and functions for MIMIC-IV chest X-ray
survival analysis, including optimized data loaders and benchmark components.
"""

import os
import time
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Suppress MONAI deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# MONAI imports for optimized data loading
from monai.data import CacheDataset, ThreadDataLoader, DataLoader as MonaiDataLoader
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityD, RandFlipD, RandRotateD, RandZoomD
from monai.utils import set_determinism

set_determinism(seed=42, use_deterministic_algorithms=False)

# Import MIMIC components
from .mimic_data_loader import MIMICDataLoader
from .transforms import get_efficientnet_transforms


class OptimizedMIMICDataLoader:
    """MONAI-optimized MIMIC data loader for maximum performance."""
    
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "Y:/mimic-cxr-jpg-2.1.0.physionet.org/",
        csv_path: str = "data/mimic/mimic_cxr_splits.csv",
        target_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = True,
        cache_rate: float = 0.1,  # Cache 10% of data in memory
        num_workers: int = 8,
        pin_memory: bool = True
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def _create_monai_transforms(self, is_training: bool = True):
        """Create MONAI-optimized transforms."""
        # MONAI transforms for better performance
        from monai.data import PILReader
        
        def convert_to_rgb(image):
            """Convert image to RGB format for EfficientNet compatibility."""
            return image.convert("RGB")
        
        transforms = [
            LoadImageD(keys=["image"], reader=PILReader(converter=convert_to_rgb)),
            EnsureChannelFirstD(keys=["image"]),
            ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        # No augmentation by default for large datasets
        # if is_training and self.use_augmentation:
        #     transforms.extend([
        #         RandFlipD(keys=["image"], prob=0.5, spatial_axis=1),
        #         RandRotateD(keys=["image"], range_x=15, prob=0.5),
        #         RandZoomD(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        #     ])
        
        # Resize to target size
        from monai.transforms import ResizeD
        transforms.append(ResizeD(keys=["image"], spatial_size=self.target_size))
        
        # Create Compose transform
        try:
            return Compose(transforms)
        except Exception as e:
            print(f"Warning: MONAI Compose error: {e}")
            # Fallback to standard transforms
            from .transforms import get_efficientnet_transforms
            return get_efficientnet_transforms(self.target_size, is_training)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load data with MONAI optimizations."""
        # Load CSV data
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        df = df[df['exists'] == True].copy()
        
        # Create splits
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        print(f"Dataset sizes - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        
        # Create MONAI data dictionaries
        train_data = []
        for _, row in df_train.iterrows():
            train_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        val_data = []
        for _, row in df_val.iterrows():
            val_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        test_data = []
        for _, row in df_test.iterrows():
            test_data.append({
                "image": os.path.join(self.data_dir, row['path']),
                "event": row['event'],
                "time": row['tte']
            })
        
        # Create transforms
        train_transforms = self._create_monai_transforms(is_training=True)
        val_test_transforms = self._create_monai_transforms(is_training=False)
        
        # Create MONAI datasets with caching
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        val_dataset = CacheDataset(
            data=val_data,
            transform=val_test_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        test_dataset = CacheDataset(
            data=test_data,
            transform=val_test_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        
        # Create MONAI data loaders
        train_loader = ThreadDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = ThreadDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = ThreadDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        print(f"MONAI datasets created with cache_rate={self.cache_rate}")
        
        return train_loader, val_loader, test_loader, 3  # 3 channels for RGB
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics."""
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        df = df[df['exists'] == True].copy()
        
        return {
            'total_samples': len(df),
            'train_samples': len(df[df['split'] == 'train']),
            'val_samples': len(df[df['split'] == 'val']),
            'test_samples': len(df[df['split'] == 'test']),
            'overall_event_rate': df['event'].mean(),
            'train_event_rate': df[df['split'] == 'train']['event'].mean(),
            'val_event_rate': df[df['split'] == 'val']['event'].mean(),
            'test_event_rate': df[df['split'] == 'test']['event'].mean(),
            'mean_survival_time': df['tte'].mean(),
            'median_survival_time': df['tte'].median()
        }


class MIMICBenchmarkTrainer:
    """Trainer for MIMIC dataset with EfficientNet-B0 backbone."""
    
    def __init__(
        self, 
        device: torch.device, 
        epochs: int = 100, 
        learning_rate: float = 1e-4, 
        weight_decay: float = 1e-5,
        patience: int = 20,
        max_steps: int = 100000
    ):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_steps = max_steps
        
        # Pre-initialize loss functions for efficiency
        from concordance_pairwise_loss import ConcordancePairwiseLoss
        from concordance_pairwise_loss.dynamic_weighting import NormalizedLossCombination
        
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
    
    class _EfficientNetSurvivalModel(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module, risk_head: torch.nn.Module):
            super().__init__()
            self.backbone = backbone
            self.risk_head = risk_head
        
        def forward(self, x):
            # Extract features from EfficientNet backbone
            features = self.backbone.features(x)
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            # Risk prediction head
            risk = self.risk_head(features)
            return risk
    
    def create_model(self) -> torch.nn.Module:
        """Create EfficientNet-B0 based survival model with torch.compile optimization."""
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for EfficientNet models. Install with: pip install torchvision")
        
        # Load EfficientNet-B0 backbone
        backbone = models.efficientnet_b0(pretrained=True)
        # Remove the classifier head
        backbone.classifier = torch.nn.Identity()
        
        # Freeze early layers for transfer learning
        for param in backbone.features[:5].parameters():
            param.requires_grad = False
        
        # Risk prediction head (single layer)
        risk_head = torch.nn.Linear(1280, 1)  # EfficientNet-B0 output features -> single output
        
        model = self._EfficientNetSurvivalModel(backbone, risk_head)
        model = model.to(self.device)
        
        return model
    
    def _precompute_ipcw_weights(self, dataloader_train: DataLoader) -> None:
        """Precompute IPCW weights from the full training set for batch variant."""
        try:
            from torchsurv.stats.ipcw import get_ipcw
        except ImportError:
            raise ImportError("torchsurv is required for IPCW weights. Install with: pip install torchsurv")
        
        # Collect all training data
        all_times = []
        all_events = []
        
        for batch in dataloader_train:
            # Handle different data formats (MONAI vs standard)
            if isinstance(batch, dict):
                # MONAI format: {"image": tensor, "event": tensor, "time": tensor}
                event = batch["event"]
                time = batch["time"]
            else:
                # Standard format: (image, (event, time))
                x, (event, time) = batch
            
            # Convert MONAI MetaTensors to regular PyTorch tensors to avoid indexing issues
            if hasattr(event, 'as_tensor'):
                event = event.as_tensor()
            if hasattr(time, 'as_tensor'):
                time = time.as_tensor()
            
            all_times.append(time)
            all_events.append(event)
        
        # Concatenate all data
        all_times = torch.cat(all_times, dim=0)
        all_events = torch.cat(all_events, dim=0)
        
        # Ensure correct data types for torchsurv compatibility
        all_events = all_events.bool()  # torchsurv requires boolean events
        all_times = all_times.float()   # torchsurv requires float times
        
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
        """Train model with specified loss type and early stopping."""
        print(f"\n=== Training {loss_type.upper()} ===")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        step_count = 0
        
        # Precompute IPCW weights for batch variant
        if loss_type == "cpl_ipcw_batch":
            self._precompute_ipcw_weights(dataloader_train)
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc=f"Training {loss_type.upper()}", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for epoch in epoch_pbar:
            # Training
            model.train()
            epoch_total_loss = 0.0
            epoch_steps = 0
            
            # Create progress bar for batches
            batch_pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}", 
                             leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for batch_idx, batch in enumerate(batch_pbar):
                if step_count >= self.max_steps:
                    print(f"Reached maximum steps ({self.max_steps}), stopping training")
                    break
                
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
                
                # Ensure correct data types for torchsurv compatibility
                event = event.bool()  # torchsurv requires boolean events
                time = time.float()   # torchsurv requires float times
                
                optimizer.zero_grad()
                log_hz = model(x)
                
                # Use mixed precision for faster training
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        total_loss = self._compute_loss(
                            log_hz, event, time, loss_type, epoch, 
                            None, False, model, temperature
                        )
                else:
                    total_loss = self._compute_loss(
                        log_hz, event, time, loss_type, epoch, 
                        None, False, model, temperature
                    )
                
                # Backward pass with error handling
                if self.device.type == 'cuda' and self.scaler is not None:
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss value {total_loss.item()}, skipping this batch")
                        continue
                    
                    self.scaler.scale(total_loss).backward()
                    
                    # Check gradients
                    has_any_grad = False
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            has_any_grad = True
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                    
                    if not has_any_grad:
                        print("Warning: No gradients found, skipping optimizer step")
                    elif has_nan_grad:
                        print("Warning: NaN/Inf gradients detected, skipping optimizer step")
                    else:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss value {total_loss.item()}, skipping this batch")
                        continue
                    
                    total_loss.backward()
                    optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_steps += 1
                step_count += 1
                
                # Update batch progress bar with GPU info
                gpu_info = 'CPU'
                if self.device.type == 'cuda':
                    gpu_info = f'{torch.cuda.memory_allocated()/1e9:.1f}GB/{torch.cuda.max_memory_allocated()/1e9:.1f}GB'
                
                batch_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Steps': step_count,
                    'GPU': gpu_info
                })
                
                if step_count >= self.max_steps:
                    batch_pbar.close()
                    break
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in dataloader_val:
                    # Handle different data formats (MONAI vs standard)
                    if isinstance(batch, dict):
                        # MONAI format: {"image": tensor, "event": tensor, "time": tensor}
                        x_val = batch["image"]
                        event_val = batch["event"]
                        time_val = batch["time"]
                    else:
                        # Standard format: (image, (event, time))
                        x_val, (event_val, time_val) = batch
                    
                    # Convert MONAI MetaTensors to regular PyTorch tensors to avoid indexing issues
                    if hasattr(x_val, 'as_tensor'):
                        x_val = x_val.as_tensor()
                    if hasattr(event_val, 'as_tensor'):
                        event_val = event_val.as_tensor()
                    if hasattr(time_val, 'as_tensor'):
                        time_val = time_val.as_tensor()
                    
                    x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                    
                    # Ensure correct data types for torchsurv compatibility
                    event_val = event_val.bool()  # torchsurv requires boolean events
                    time_val = time_val.float()   # torchsurv requires float times
                    
                    log_hz_val = model(x_val)
                    
                    batch_val_loss = self._compute_loss(
                        log_hz_val, event_val, time_val, loss_type, epoch,
                        None, False, model, temperature
                    )
                    val_loss += batch_val_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
            avg_train_loss = epoch_total_loss / epoch_steps if epoch_steps > 0 else 0.0
            
            # Store results
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train': f'{avg_train_loss:.4f}',
                'Val': f'{avg_val_loss:.4f}',
                'Patience': f'{patience_counter}/{self.patience}',
                'Best': f'{best_val_loss:.4f}'
            })
            
            # Close batch progress bar
            batch_pbar.close()
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break
            
            if step_count >= self.max_steps:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model (val_loss={best_val_loss:.4f})")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses),
            'total_steps': step_count
        }
    
    def _compute_loss(
        self, 
        log_hz: torch.Tensor, 
        event: torch.Tensor, 
        time: torch.Tensor,
        loss_type: str,
        epoch: int,
        loss_combiner: Optional[Any],
        use_gradnorm: bool,
        model: torch.nn.Module,
        temperature: float
    ) -> torch.Tensor:
        """Compute loss based on the specified type."""
        try:
            from torchsurv.loss.cox import neg_partial_log_likelihood
        except ImportError:
            raise ImportError("torchsurv is required for NLL loss. Install with: pip install torchsurv")
        
        try:
            from concordance_pairwise_loss import ConcordancePairwiseLoss
        except ImportError:
            raise ImportError("concordance_pairwise_loss is required. Install the package first.")
        
        # Ensure proper shape for NLL
        log_hz_2d = log_hz if log_hz.dim() == 2 else log_hz.unsqueeze(1)
        log_hz_1d = log_hz.squeeze(-1) if log_hz.dim() == 2 else log_hz
        
        # Clamp log_hz to prevent extreme values
        log_hz_2d = torch.clamp(log_hz_2d, min=-10.0, max=10.0)
        log_hz_1d = torch.clamp(log_hz_1d, min=-10.0, max=10.0)
        
        if loss_type == "nll":
            loss = neg_partial_log_likelihood(log_hz_2d, event, time, reduction="mean")
        elif loss_type == "cpl":
            self.cpl_loss.temperature = temperature
            loss = self.cpl_loss(log_hz_1d, time, event)
        elif loss_type == "cpl_ipcw":
            self.cpl_ipcw_loss.temperature = temperature
            loss = self.cpl_ipcw_loss(log_hz_1d, time, event)
        elif loss_type == "cpl_ipcw_batch":
            self.cpl_ipcw_batch_loss.temperature = temperature
            if self.precomputed_ipcw_weights is not None:
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
                loss = self.cpl_ipcw_batch_loss(log_hz_1d, time, event)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Computed loss is {loss.item()}, replacing with 1.0")
            return torch.tensor(1.0, device=log_hz.device, requires_grad=True)
        
        return loss


class MIMICBenchmarkRunner:
    """Benchmark runner for MIMIC-IV chest X-ray dataset."""
    
    # Core loss functions to compare
    CORE_LOSS_TYPES = [
        'nll',
        'cpl', 
        'cpl_ipcw',
        'cpl_ipcw_batch'
    ]
    
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 20,
        max_steps: int = 100000,
        batch_size: int = 64,  # Increased default batch size
        target_size: Tuple[int, int] = (224, 224),
        output_dir: str = "results",
        save_results: bool = True,
        random_seed: int = None,
        num_runs: int = 1,
    ):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.target_size = target_size
        self.output_dir = output_dir
        self.save_results = save_results
        self.random_seed = random_seed
        self.num_runs = num_runs
        
        # Initialize dataset components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verify GPU usage
        self._verify_gpu_usage()
        
        # Set random seed if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # Initialize components
        self.trainer = MIMICBenchmarkTrainer(
            device=self.device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            max_steps=max_steps
        )
        
        # Create dataset config for MIMIC
        try:
            from benchmark_framework import DatasetConfig, BenchmarkEvaluator, BenchmarkVisualizer
            from benchmark_framework_improved import ResultsLogger
            
            self.dataset_config = DatasetConfig(
                name="MIMIC-IV",
                auc_time=365.0,  # 1 year in days
                auc_time_unit="days"
            )
            
            self.evaluator = BenchmarkEvaluator(self.device, self.dataset_config)
            self.visualizer = BenchmarkVisualizer(self.dataset_config, output_dir if save_results else None)
            self.logger = ResultsLogger(self.dataset_config.name, output_dir) if save_results else None
        except ImportError as e:
            # Allow subclasses to work without benchmark_framework if they override run_comparison
            print(f"Warning: benchmark_framework not available ({e}). Some features may be limited.")
            self.dataset_config = None
            self.evaluator = None
            self.visualizer = None
            self.logger = None
        
        print(f"MIMIC Benchmark Setup:")
        print(f"  Dataset: MIMIC-IV Chest X-ray")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Patience: {patience}")
        print(f"  Max steps: {max_steps}")
        print(f"  Number of runs: {num_runs}")
        print(f"  Batch size: {self.batch_size}")
        if self.device.type == 'cuda':
            print("  ✅ Mixed precision enabled")
        if random_seed is not None:
            print(f"  Random seed: {random_seed}")
    
    def _verify_gpu_usage(self):
        """Verify GPU is available and being used properly."""
        if self.device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    
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
    
    def run_comparison(self, args=None) -> Dict[str, Dict]:
        """Run comparison of core loss functions with multiple runs and statistical analysis."""
        # Check if evaluator is available (required for this method)
        if self.evaluator is None:
            raise RuntimeError(
                "BenchmarkEvaluator is not available. This method requires benchmark_framework to be installed. "
                "Use PreprocessedMIMICBenchmarkRunnerV2 from benchmark_MIMIC_v2.py instead, which doesn't require it."
            )
        
        import time as time_module
        start_time = time_module.time()
        
        print("=" * 80)
        print("MIMIC-IV CHEST X-RAY SURVIVAL ANALYSIS COMPARISON")
        print("Core Loss Functions: NLL, CPL, CPL(ipcw), CPL(ipcw batch)")
        print(f"Multiple Runs: {self.num_runs} independent runs per configuration")
        print("=" * 80)
        
        # Load data with MONAI optimizations
        use_augmentation = getattr(args, 'enable_augmentation', False)
        
        data_loader = OptimizedMIMICDataLoader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            target_size=self.target_size,
            use_augmentation=use_augmentation,
            cache_rate=0.1,  # Cache 10% of data in memory
            num_workers=8,  # Optimized for faster data loading
            pin_memory=True  # Enable pin_memory for faster GPU transfer
        )
        
        dataloader_train, dataloader_val, dataloader_test, num_features = data_loader.load_data()
        
        # Update dataset config with actual values
        stats = data_loader.get_dataset_stats()
        # Store stats for later use
        self.dataset_stats = stats
        
        print(f"Dataset loaded: {stats['total_samples']:,} samples")
        print(f"Event rate: {stats['overall_event_rate']:.3f}")
        print(f"Train/Val/Test: {stats['train_samples']:,}/{stats['val_samples']:,}/{stats['test_samples']:,}")
        print(f"Data augmentation: {'Enabled' if use_augmentation else 'Disabled (large dataset)'}")
        
        results = {}
        
        # Test each loss type
        loss_pbar = tqdm(enumerate(self.CORE_LOSS_TYPES), total=len(self.CORE_LOSS_TYPES), 
                        desc="Loss Types", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for loss_idx, loss_type in loss_pbar:
            loss_pbar.set_description(f"Testing {loss_type.upper()}")
            print(f"\n{'='*50}")
            print(f"TESTING {loss_type.upper()} ({loss_idx+1}/{len(self.CORE_LOSS_TYPES)})")
            print(f"{'='*50}")
            
            # Run multiple times for this loss type
            val_uno_scores = []
            test_eval_results = []
            test_training_results = []
            
            run_pbar = tqdm(range(self.num_runs), desc=f"Runs for {loss_type.upper()}", 
                           leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
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
                
                # Evaluate on validation for selection
                val_eval = self.evaluator.evaluate_model(model, dataloader_val)
                val_uno = float(val_eval['uno_cindex'])
                val_uno_scores.append(val_uno)
                
                # Also evaluate on test set for final results
                test_eval = self.evaluator.evaluate_model(model, dataloader_test)
                test_eval_results.append(test_eval)
                test_training_results.append(training_results)
                
                # Update run progress bar
                run_pbar.set_postfix({
                    'Val C': f'{val_uno:.4f}',
                    'Test C': f'{test_eval["uno_cindex"]:.4f}',
                    'Epochs': training_results['total_epochs']
                })
                
                print(f"    ✅ Run {run_idx+1} completed: Val Uno C={val_uno:.4f}, Test Uno C={test_eval['uno_cindex']:.4f}")
                print(f"    📊 Epochs: {training_results['total_epochs']}, Steps: {training_results['total_steps']}")
                print(f"    {'='*50}")
            
            # Calculate mean performance
            mean_val_uno = np.mean(val_uno_scores)
            std_val_uno = np.std(val_uno_scores)
            print(f"\n  Mean val Uno C: {mean_val_uno:.4f} ± {std_val_uno:.4f}")
            
            # Aggregate results from all runs
            aggregated_results = self._aggregate_multiple_runs(test_eval_results, test_training_results)
            
            results[loss_type] = {
                'training': aggregated_results['training'],
                'evaluation': aggregated_results['evaluation'],
                'individual_runs': test_eval_results,
                'num_runs': self.num_runs
            }
            
            # Print aggregated results
            eval_stats = aggregated_results['evaluation']
            print(f"\n🎯 [{loss_type.upper()}] Final Results (n={self.num_runs}):")
            print(f"  Harrell C: {eval_stats['harrell_cindex_mean']:.4f} ± {eval_stats['harrell_cindex_std']:.4f}")
            print(f"  Uno C: {eval_stats['uno_cindex_mean']:.4f} ± {eval_stats['uno_cindex_std']:.4f}")
            print(f"  Cum AUC: {eval_stats['cumulative_auc_mean']:.4f} ± {eval_stats['cumulative_auc_std']:.4f}")
            print(f"  Inc AUC: {eval_stats['incident_auc_mean']:.4f} ± {eval_stats['incident_auc_std']:.4f}")
            print(f"  Brier: {eval_stats['brier_score_mean']:.4f} ± {eval_stats['brier_score_std']:.4f}")
            print(f"✅ {loss_type.upper()} completed! ({loss_idx+1}/{len(self.CORE_LOSS_TYPES)} loss types done)")
            
            # Close run progress bar
            run_pbar.close()
        
        # Close loss type progress bar
        loss_pbar.close()
        
        # Perform statistical analysis
        self._perform_statistical_analysis(results)
        
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
                'patience': self.patience,
                'max_steps': self.max_steps,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'mixed_precision': self.device.type == 'cuda',
                'core_loss_types': self.CORE_LOSS_TYPES,
                'num_runs': self.num_runs,
                'execution_time': execution_time
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MIMIC_benchmark_{timestamp}"
            saved_files = self.logger.save_results(results, filename)
            
            # Save comprehensive CSV
            self._save_comprehensive_csv(results, execution_time)
            
            print(f"\n{'='*80}")
            print("RESULTS SAVED TO FILES")
            print(f"{'='*80}")
            if saved_files:
                for file_type, filepath in saved_files.items():
                    print(f"{file_type.upper()}: {filepath}")
            else:
                print("No additional files saved by logger")
        
        return results
    
    def _aggregate_multiple_runs(self, eval_results: List[Dict], training_results: List[Dict]) -> Dict[str, Dict]:
        """Aggregate results from multiple runs with mean and standard deviation."""
        # Aggregate evaluation metrics
        eval_metrics = ['harrell_cindex', 'uno_cindex', 'cumulative_auc', 'incident_auc', 'brier_score']
        aggregated_eval = {}
        
        for metric in eval_metrics:
            values = [result[metric] for result in eval_results if not np.isnan(result[metric])]
            if values:
                aggregated_eval[f'{metric}_mean'] = np.mean(values)
                aggregated_eval[f'{metric}_std'] = np.std(values)
                aggregated_eval[f'{metric}_values'] = values
            else:
                aggregated_eval[f'{metric}_mean'] = 0.0
                aggregated_eval[f'{metric}_std'] = 0.0
                aggregated_eval[f'{metric}_values'] = []
        
        # Aggregate training metrics
        train_losses = [result['train_losses'] for result in training_results]
        val_losses = [result['val_losses'] for result in training_results]
        
        # Average training curves
        max_epochs = max(len(losses) for losses in train_losses)
        avg_train_losses = []
        avg_val_losses = []
        
        for epoch in range(max_epochs):
            epoch_train_losses = [losses[epoch] for losses in train_losses if epoch < len(losses)]
            epoch_val_losses = [losses[epoch] for losses in val_losses if epoch < len(losses)]
            
            if epoch_train_losses:
                avg_train_losses.append(np.mean(epoch_train_losses))
            if epoch_val_losses:
                avg_val_losses.append(np.mean(epoch_val_losses))
        
        aggregated_training = {
            'train_losses': avg_train_losses,
            'val_losses': avg_val_losses,
            'best_val_loss': np.mean([result['best_val_loss'] for result in training_results]),
            'total_epochs': np.mean([result['total_epochs'] for result in training_results]),
            'total_steps': np.mean([result['total_steps'] for result in training_results])
        }
        
        return {
            'evaluation': aggregated_eval,
            'training': aggregated_training
        }
    
    def _perform_statistical_analysis(self, results: Dict[str, Dict]) -> None:
        """Perform statistical significance testing between methods."""
        from scipy import stats
        
        print(f"\n{'='*80}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Extract Uno C-index values for statistical testing
        method_data = {}
        for loss_key, result in results.items():
            if 'individual_runs' in result:
                uno_values = []
                for run_result in result['individual_runs']:
                    if not np.isnan(run_result['uno_cindex']):
                        uno_values.append(run_result['uno_cindex'])
                if uno_values:
                    method_data[loss_key] = uno_values
        
        if len(method_data) < 2:
            print("Insufficient data for statistical analysis")
            return
        
        # Perform pairwise comparisons
        methods = list(method_data.keys())
        print(f"\nPairwise t-tests (Uno C-index):")
        print(f"{'Method 1':<20} {'Method 2':<20} {'t-statistic':<12} {'p-value':<10} {'Significant'}")
        print("-" * 80)
        
        significant_pairs = []
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                data1, data2 = method_data[method1], method_data[method2]
                
                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    significant = "Yes" if p_value < 0.05 else "No"
                    if p_value < 0.05:
                        significant_pairs.append((method1, method2, p_value))
                    
                    print(f"{method1:<20} {method2:<20} {t_stat:<12.4f} {p_value:<10.4f} {significant}")
        
        if significant_pairs:
            print(f"\nSignificant differences found in {len(significant_pairs)} pairs (p < 0.05)")
        else:
            print("\nNo significant differences found between methods (p >= 0.05)")
    
    def _analyze_results(self, results: Dict[str, Dict]) -> None:
        """Analyze results and create comprehensive visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Print comprehensive summary table
        print(f"\n{'Method':<20} {'Harrell C':<15} {'Uno C':<15} {'Cum AUC':<15} {'Inc AUC':<15} {'Brier':<15}")
        print("-" * 100)
        
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        for loss_key, result in results.items():
            eval_result = result['evaluation']
            method_name = method_names.get(loss_key, loss_key.upper())
            
            # Use mean ± std format
            harrell = f"{eval_result['harrell_cindex_mean']:.4f}±{eval_result['harrell_cindex_std']:.4f}"
            uno = f"{eval_result['uno_cindex_mean']:.4f}±{eval_result['uno_cindex_std']:.4f}"
            cumulative_auc = f"{eval_result['cumulative_auc_mean']:.4f}±{eval_result['cumulative_auc_std']:.4f}"
            incident_auc = f"{eval_result['incident_auc_mean']:.4f}±{eval_result['incident_auc_std']:.4f}"
            brier = f"{eval_result['brier_score_mean']:.4f}±{eval_result['brier_score_std']:.4f}"
            
            print(f"{method_name:<20} {harrell:<15} {uno:<15} {cumulative_auc:<15} {incident_auc:<15} {brier:<15}")
        
        # Create comprehensive visualizations
        self._create_comprehensive_plots(results)
    
    def _create_comprehensive_plots(self, results: Dict[str, Dict]) -> None:
        """Create comprehensive visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n=== Creating Comprehensive Analysis Plots ===")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MIMIC-IV Chest X-ray Survival Analysis Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Concordance indices comparison
        self._plot_concordance_comparison(axes[0, 0], results)
        
        # 2. All metrics performance comparison
        self._plot_all_metrics_comparison(axes[0, 1], results)
        
        # 3. Training loss evolution
        self._plot_training_evolution(axes[1, 0], results)
        
        # 4. Performance distribution
        self._plot_performance_distribution(axes[1, 1], results)
        
        plt.tight_layout()
        
        # Save figure
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            figure_filename = f"MIMIC_analysis_{timestamp}.png"
            figure_path = os.path.join(self.output_dir, figure_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Comprehensive analysis figure saved: {figure_path}")
        
        plt.show()
    
    def _plot_concordance_comparison(self, ax, results):
        """Plot Harrell vs Uno C-index comparison."""
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        methods = list(results.keys())
        method_labels = [method_names.get(m, m.upper()) for m in methods]
        
        harrell_scores = []
        uno_scores = []
        
        for method in methods:
            eval_data = results[method]['evaluation']
            harrell_scores.append(eval_data['harrell_cindex_mean'])
            uno_scores.append(eval_data['uno_cindex_mean'])
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, harrell_scores, width, label="Harrell's C", alpha=0.8)
        ax.bar(x + width/2, uno_scores, width, label="Uno's C", alpha=0.8)
        
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
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        methods = list(results.keys())
        method_labels = [method_names.get(m, m.upper()) for m in methods]
        
        harrell_scores = []
        uno_scores = []
        cumulative_auc_scores = []
        incident_auc_scores = []
        brier_scores = []
        
        for method in methods:
            eval_data = results[method]['evaluation']
            harrell_scores.append(eval_data['harrell_cindex_mean'])
            uno_scores.append(eval_data['uno_cindex_mean'])
            cumulative_auc_scores.append(eval_data['cumulative_auc_mean'])
            incident_auc_scores.append(eval_data['incident_auc_mean'])
            # Invert Brier score (lower is better)
            brier = eval_data['brier_score_mean']
            brier_scores.append(max(0, 1 - brier) if not np.isnan(brier) else 0)
        
        x = np.arange(len(methods))
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
            color = method_colors.get(loss_key, 'gray')
            label = loss_key.upper()
            ax.plot(train_losses, label=label, alpha=0.8, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_distribution(self, ax, results):
        """Plot performance distribution across runs."""
        method_names = {
            'nll': 'NLL',
            'cpl': 'CPL',
            'cpl_ipcw': 'CPL (ipcw)',
            'cpl_ipcw_batch': 'CPL (ipcw batch)'
        }
        
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for method, result in results.items():
            if 'individual_runs' in result:
                uno_values = []
                for run_result in result['individual_runs']:
                    if not np.isnan(run_result['uno_cindex']):
                        uno_values.append(run_result['uno_cindex'])
                if uno_values:
                    data_for_boxplot.append(uno_values)
                    labels_for_boxplot.append(method_names.get(method, method.upper()))
        
        if data_for_boxplot:
            ax.boxplot(data_for_boxplot, labels=labels_for_boxplot)
            ax.set_ylabel("Uno's C-index")
            ax.set_title('Performance Distribution Across Runs')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
    
    def _save_comprehensive_csv(self, results: Dict[str, Dict], execution_time: float) -> None:
        """Save comprehensive CSV with all metrics."""
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"MIMIC_results_{timestamp}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'loss_type', 'harrell_cindex', 'uno_cindex', 'cumulative_auc', 'incident_auc', 'brier_score',
                'final_train_loss', 'final_val_loss', 'best_val_loss', 'total_epochs', 'total_steps'
            ])
            
            # Write data for each loss type
            for loss_key, result in results.items():
                eval_result = result['evaluation']
                training = result['training']
                
                # Use mean values
                harrell_c = eval_result.get('harrell_cindex_mean', 0.0)
                uno_c = eval_result.get('uno_cindex_mean', 0.0)
                cum_auc = eval_result.get('cumulative_auc_mean', 0.0)
                inc_auc = eval_result.get('incident_auc_mean', 0.0)
                brier = eval_result.get('brier_score_mean', 0.0)
                
                writer.writerow([
                    loss_key,
                    harrell_c,
                    uno_c,
                    cum_auc,
                    inc_auc,
                    brier,
                    training['train_losses'][-1] if training['train_losses'] else 0.0,
                    training['val_losses'][-1] if training['val_losses'] else 0.0,
                    training['best_val_loss'],
                    training['total_epochs'],
                    training['total_steps']
                ])
        
        print(f"📊 Comprehensive CSV saved: {csv_path}")
