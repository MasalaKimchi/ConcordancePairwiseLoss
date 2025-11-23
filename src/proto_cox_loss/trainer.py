"""
Custom training loop for ProtoCoxLoss models.

This module provides training functionality specifically designed for:
1. ProtoCoxLoss (PCL) - Standard geometric contrastive Cox loss
2. ProtoCoxLoss with MoCo (PCL+MoCo) - ProtoCoxLoss with momentum contrast queue

The MoCo version requires special handling because the MoCo wrapper's forward
method computes the loss internally, so we need a custom training loop.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from torchsurv.stats.ipcw import get_ipcw
except ImportError:
    get_ipcw = None


class ProtoCoxLossTrainer:
    """
    Custom trainer for ProtoCoxLoss models.
    
    This trainer handles both standard ProtoCoxLoss and ProtoCoxLoss with MoCo,
    which require different training loops due to how MoCo computes loss internally.
    """
    
    def __init__(
        self,
        device: torch.device,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 20,
        max_steps: int = 100000,
        use_mixed_precision: bool = True
    ):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_steps = max_steps
        self.use_mixed_precision = use_mixed_precision
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and use_mixed_precision) else None
    
    def train_model(
        self,
        model: nn.Module,
        loss_fn: nn.Module,  # ProtoCoxLoss instance
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        loss_type: str,  # 'pcl' or 'pcl_moco'
        temperature: float = 0.07
    ) -> Dict[str, Any]:
        """
        Train model with ProtoCoxLoss.
        
        Args:
            model: The model to train (may be wrapped with MoCo)
            loss_fn: ProtoCoxLoss instance (contains learnable prototype)
            dataloader_train: Training data loader
            dataloader_val: Validation data loader
            loss_type: 'pcl' or 'pcl_moco'
            temperature: Temperature parameter (usually 0.07)
        
        Returns:
            Dictionary with training results
        """
        print(f"\n=== Training {loss_type.upper()} ===")
        
        # Set temperature if loss_fn has it
        if hasattr(loss_fn, 'temperature'):
            loss_fn.temperature = temperature
        
        # Optimizer includes both model parameters and loss_fn parameters (prototype)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_loss_fn_state = None  # Save prototype state too
        
        train_losses = []
        val_losses = []
        step_count = 0
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc=f"Training {loss_type.upper()}", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for epoch in epoch_pbar:
            # Training
            model.train()
            loss_fn.train()
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
                    x = batch["image"]
                    event = batch["event"]
                    time = batch["time"]
                else:
                    x, (event, time) = batch
                
                # Convert MONAI MetaTensors to regular PyTorch tensors
                if hasattr(x, 'as_tensor'):
                    x = x.as_tensor()
                if hasattr(event, 'as_tensor'):
                    event = event.as_tensor()
                if hasattr(time, 'as_tensor'):
                    time = time.as_tensor()
                
                x, event, time = x.to(self.device), event.to(self.device), time.to(self.device)
                
                # Ensure correct data types
                event = event.bool()
                time = time.float()
                
                optimizer.zero_grad()
                
                # Different handling for MoCo vs standard
                if loss_type == "pcl_moco":
                    # MoCo wrapper computes loss internally
                    # Its forward method expects (x, times, events, loss_fn)
                    total_loss = model(x, time, event, loss_fn)
                else:
                    # Standard ProtoCoxLoss
                    # Extract features from model
                    if hasattr(model, 'get_features'):
                        # Model stores features during forward
                        _ = model(x)  # Forward pass to store features
                        features = model.get_features()
                    elif hasattr(model, 'last_features'):
                        _ = model(x)
                        features = model.last_features
                    else:
                        # Extract features manually
                        if hasattr(model, 'base_model'):
                            features = model.base_model.backbone.features(x)
                        elif hasattr(model, 'backbone'):
                            features = model.backbone.features(x)
                        else:
                            raise ValueError("Cannot extract features from model")
                        
                        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = torch.flatten(features, 1)
                    
                    # Get IPCW weights if needed
                    ipcw_weights = None
                    if loss_fn.use_ipcw and get_ipcw is not None:
                        event_cpu = event.cpu()
                        time_cpu = time.cpu()
                        ipcw_weights = get_ipcw(event_cpu, time_cpu).to(self.device)
                    
                    # Compute loss
                    total_loss = loss_fn(features, time, event, ipcw_weights)
                
                # Backward pass with error handling
                if self.device.type == 'cuda' and self.scaler is not None:
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss value {total_loss.item()}, skipping this batch")
                        continue
                    
                    self.scaler.scale(total_loss).backward()
                    
                    # Check gradients
                    has_any_grad = False
                    has_nan_grad = False
                    for param in list(model.parameters()) + list(loss_fn.parameters()):
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
                
                # Update batch progress bar
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
            loss_fn.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in dataloader_val:
                    # Handle different data formats
                    if isinstance(batch, dict):
                        x_val = batch["image"]
                        event_val = batch["event"]
                        time_val = batch["time"]
                    else:
                        x_val, (event_val, time_val) = batch
                    
                    # Convert MONAI MetaTensors
                    if hasattr(x_val, 'as_tensor'):
                        x_val = x_val.as_tensor()
                    if hasattr(event_val, 'as_tensor'):
                        event_val = event_val.as_tensor()
                    if hasattr(time_val, 'as_tensor'):
                        time_val = time_val.as_tensor()
                    
                    x_val, event_val, time_val = x_val.to(self.device), event_val.to(self.device), time_val.to(self.device)
                    
                    # Ensure correct data types
                    event_val = event_val.bool()
                    time_val = time_val.float()
                    
                    # Compute validation loss
                    if loss_type == "pcl_moco":
                        batch_val_loss = model(x_val, time_val, event_val, loss_fn)
                    else:
                        # Extract features
                        if hasattr(model, 'get_features'):
                            _ = model(x_val)
                            features_val = model.get_features()
                        elif hasattr(model, 'last_features'):
                            _ = model(x_val)
                            features_val = model.last_features
                        else:
                            if hasattr(model, 'base_model'):
                                features_val = model.base_model.backbone.features(x_val)
                            elif hasattr(model, 'backbone'):
                                features_val = model.backbone.features(x_val)
                            else:
                                raise ValueError("Cannot extract features from model")
                            
                            features_val = torch.nn.functional.adaptive_avg_pool2d(features_val, (1, 1))
                            features_val = torch.flatten(features_val, 1)
                        
                        # Get IPCW weights if needed
                        ipcw_weights_val = None
                        if loss_fn.use_ipcw and get_ipcw is not None:
                            event_val_cpu = event_val.cpu()
                            time_val_cpu = time_val.cpu()
                            ipcw_weights_val = get_ipcw(event_val_cpu, time_val_cpu).to(self.device)
                        
                        batch_val_loss = loss_fn(features_val, time_val, event_val, ipcw_weights_val)
                    
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
                best_loss_fn_state = loss_fn.state_dict().copy()  # Save prototype
            else:
                patience_counter += 1
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train': f'{avg_train_loss:.4f}',
                'Val': f'{avg_val_loss:.4f}',
                'Best': f'{best_val_loss:.4f}',
                'Patience': patience_counter
            })
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience: {self.patience})")
                break
            
            if step_count >= self.max_steps:
                epoch_pbar.close()
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            loss_fn.load_state_dict(best_loss_fn_state)  # Load prototype
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses),
            'total_steps': step_count
        }


def save_protocox_model(
    model: nn.Module,
    loss_fn: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save ProtoCoxLoss model with prototype.
    
    The prototype (risk_prototype) is a learnable parameter in ProtoCoxLoss
    that represents the "direction of high risk" in the embedding space.
    This must be saved along with the model so that during evaluation,
    we can compute risk scores using the same prototype that was learned
    during training.
    
    Args:
        model: The trained model
        loss_fn: ProtoCoxLoss instance containing the prototype
        save_path: Path to save the model
        metadata: Optional metadata to save (e.g., metrics, hyperparameters)
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),  # Contains risk_prototype
        'prototype': loss_fn.risk_prototype.data.clone(),  # Also save separately for easy access
        'metadata': metadata or {}
    }
    torch.save(save_dict, save_path)
    print(f"Saved ProtoCoxLoss model with prototype to: {save_path}")


def load_protocox_model(
    model: nn.Module,
    loss_fn: nn.Module,
    load_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load ProtoCoxLoss model with prototype.
    
    Args:
        model: Model to load weights into
        loss_fn: ProtoCoxLoss instance to load prototype into
        load_path: Path to saved model
        device: Device to load on
    
    Returns:
        Dictionary containing metadata
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
    
    # Verify prototype was loaded correctly
    if 'prototype' in checkpoint:
        assert torch.allclose(
            loss_fn.risk_prototype.data,
            checkpoint['prototype'].to(device),
            atol=1e-6
        ), "Prototype mismatch after loading"
    
    print(f"Loaded ProtoCoxLoss model with prototype from: {load_path}")
    return checkpoint.get('metadata', {})

