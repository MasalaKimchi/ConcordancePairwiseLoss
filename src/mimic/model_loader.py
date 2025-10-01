#!/usr/bin/env python3
"""
MIMIC Model Loader Utility

This utility provides functions to load saved MIMIC models from benchmark results.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import torchvision.models as models


def load_saved_model(model_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a saved MIMIC model from disk.
    
    Args:
        model_path: Path to the saved .pth file
        device: Device to load the model on (None for CPU)
        
    Returns:
        Dictionary containing the loaded model and metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    # Load the saved model dictionary
    saved_dict = torch.load(model_path, map_location=device)
    
    # Create the model architecture (EfficientNet-B0)
    model = create_efficientnet_survival_model(device)
    
    # Load the state dict
    model.load_state_dict(saved_dict['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return {
        'model': model,
        'metadata': {
            'loss_type': saved_dict.get('loss_type'),
            'val_cindex': saved_dict.get('val_cindex'),
            'test_cindex': saved_dict.get('test_cindex'),
            'training_results': saved_dict.get('training_results'),
            'model_architecture': saved_dict.get('model_architecture'),
            'data_fraction': saved_dict.get('data_fraction'),
            'epochs': saved_dict.get('epochs'),
            'batch_size': saved_dict.get('batch_size'),
            'learning_rate': saved_dict.get('learning_rate'),
            'weight_decay': saved_dict.get('weight_decay'),
            'timestamp': saved_dict.get('timestamp'),
        }
    }


def create_efficientnet_survival_model(device: torch.device) -> nn.Module:
    """
    Create EfficientNet-B0 based survival model (same as in training).
    
    Args:
        device: Device to create the model on
        
    Returns:
        EfficientNet survival model
    """
    class _EfficientNetSurvivalModel(nn.Module):
        def __init__(self, backbone: nn.Module, risk_head: nn.Module):
            super().__init__()
            self.backbone = backbone
            self.risk_head = risk_head
        
        def forward(self, x):
            # Extract features from EfficientNet backbone
            features = self.backbone.features(x)
            # Global average pooling
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            # Risk prediction head
            risk = self.risk_head(features)
            return risk
    
    # Load EfficientNet-B0 backbone
    backbone = models.efficientnet_b0(pretrained=False)  # Don't need pretrained weights when loading
    # Remove the classifier head
    backbone.classifier = nn.Identity()
    
    # Risk prediction head (single layer)
    risk_head = nn.Linear(1280, 1)  # EfficientNet-B0 output features -> single output
    
    model = _EfficientNetSurvivalModel(backbone, risk_head)
    model = model.to(device)
    
    return model


def load_best_models(results_dir: str = "results", device: Optional[torch.device] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all best models from a results directory.
    
    Args:
        results_dir: Directory containing saved models
        device: Device to load models on (None for CPU)
        
    Returns:
        Dictionary mapping loss types to loaded models and metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    models_dir = os.path.join(results_dir, "models")
    loaded_models = {}
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return loaded_models
    
    # Look for each loss type
    loss_types = ['nll', 'cpl', 'cpl_ipcw', 'cpl_ipcw_batch']
    
    for loss_type in loss_types:
        model_subdir = os.path.join(models_dir, f"{loss_type}_best")
        latest_model_path = os.path.join(model_subdir, f"{loss_type}_latest_best.pth")
        
        if os.path.exists(latest_model_path):
            try:
                loaded_model_info = load_saved_model(latest_model_path, device)
                loaded_models[loss_type] = loaded_model_info
                print(f"✅ Loaded {loss_type.upper()} model (Val C-index: {loaded_model_info['metadata']['val_cindex']:.4f})")
            except Exception as e:
                print(f"❌ Failed to load {loss_type} model: {e}")
        else:
            print(f"⚠️  {loss_type.upper()} model not found at {latest_model_path}")
    
    return loaded_models


def compare_models(models_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Compare loaded models and print a summary.
    
    Args:
        models_dict: Dictionary of loaded models from load_best_models()
    """
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    if not models_dict:
        print("No models to compare.")
        return
    
    # Sort by validation C-index
    sorted_models = sorted(
        models_dict.items(), 
        key=lambda x: x[1]['metadata']['val_cindex'], 
        reverse=True
    )
    
    print(f"{'Rank':<4} {'Loss Type':<15} {'Val C-index':<12} {'Test C-index':<12} {'Epochs':<8}")
    print("-" * 60)
    
    for rank, (loss_type, model_info) in enumerate(sorted_models, 1):
        metadata = model_info['metadata']
        val_c = metadata['val_cindex']
        test_c = metadata['test_cindex']
        epochs = metadata.get('training_results', {}).get('total_epochs', 'N/A')
        
        print(f"{rank:<4} {loss_type.upper():<15} {val_c:<12.6f} {test_c:<12.6f} {epochs:<8}")
    
    print(f"\n🏆 Best model: {sorted_models[0][0].upper()} (Val C-index: {sorted_models[0][1]['metadata']['val_cindex']:.6f})")
    print(f"{'=' * 80}")


def predict_with_model(model_path: str, dataloader, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Make predictions using a saved model.
    
    Args:
        model_path: Path to the saved model
        dataloader: DataLoader containing the data to predict on
        device: Device to run predictions on
        
    Returns:
        Tensor of risk predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model_info = load_saved_model(model_path, device)
    model = model_info['model']
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different data formats (MONAI vs standard)
            if isinstance(batch, dict):
                x = batch["image"]
            else:
                x, _ = batch
            
            # Convert MONAI MetaTensors to regular PyTorch tensors if needed
            if hasattr(x, 'as_tensor'):
                x = x.as_tensor()
            
            x = x.to(device)
            
            # Get predictions
            log_hz = model(x)
            predictions.append(log_hz.cpu())
    
    return torch.cat(predictions, dim=0)


if __name__ == "__main__":
    # Example usage
    print("MIMIC Model Loader Utility")
    print("=" * 40)
    
    # Load all best models
    models = load_best_models("results")
    
    if models:
        # Compare models
        compare_models(models)
        
        # Example: Load specific model
        if 'cpl' in models:
            print(f"\nExample: CPL model metadata:")
            cpl_metadata = models['cpl']['metadata']
            for key, value in cpl_metadata.items():
                if key != 'training_results':  # Skip large training results
                    print(f"  {key}: {value}")
    else:
        print("No saved models found. Run the benchmark first:")
        print("python benchmarks/benchmark_MIMIC_preprocessed.py --epochs 5 --data-fraction 0.01")
