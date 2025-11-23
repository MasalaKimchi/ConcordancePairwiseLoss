# ProtoCoxLoss: Geometric Contrastive Formulation of Cox Partial Likelihood

## Overview

ProtoCoxLoss is a geometric contrastive formulation of Cox partial likelihood that uses cosine similarity between patient embeddings and a learnable "risk prototype" vector instead of a linear risk score.

## Key Concepts

### The Risk Prototype

The **risk prototype** (`risk_prototype`) is a learnable parameter in ProtoCoxLoss that represents the "direction of high risk" in the embedding space. 

- **What it is**: A vector of the same dimension as patient embeddings (e.g., 1280 for EfficientNet-B0)
- **What it does**: Acts as a reference point - patients with embeddings closer to the prototype (higher cosine similarity) have higher risk
- **How it's learned**: During training, the prototype rotates in the hypersphere to find the direction that best separates high-risk from low-risk patients

### Why Save and Load the Prototype?

**The prototype must be saved and loaded because:**

1. **It's a learnable parameter**: The prototype is learned during training, just like model weights. It's not a fixed value.

2. **Evaluation requires it**: To compute risk scores during evaluation, we need to:
   - Extract patient embeddings from the model
   - Compute cosine similarity with the prototype: `risk_score = cos(embedding, prototype) / temperature`
   - Without the prototype, we cannot compute accurate risk scores

3. **It's part of the model**: The prototype is as important as the model weights - both are needed for inference.

### Example

```python
# During training
loss_fn = ProtoCoxLoss(input_dim=1280)
# ... training ...
# The prototype is learned: loss_fn.risk_prototype

# During evaluation
features = model.extract_features(patient_image)  # (1, 1280)
features_norm = F.normalize(features, p=2, dim=1)
proto_norm = F.normalize(loss_fn.risk_prototype, p=2, dim=0)
risk_score = torch.matmul(features_norm, proto_norm) / temperature
```

## Saving and Loading

### Saving

```python
from proto_cox_loss import save_protocox_model

save_protocox_model(
    model=model,
    loss_fn=loss_fn,  # Contains the prototype
    save_path="model.pth",
    metadata={"val_uno_cindex": 0.75, "epoch": 50}
)
```

This saves:
- Model state dict
- Loss function state dict (includes `risk_prototype`)
- Prototype separately (for easy access)
- Metadata

### Loading

```python
from proto_cox_loss import load_protocox_model, ProtoCoxLoss

# Create model and loss function
model = create_model()
loss_fn = ProtoCoxLoss(input_dim=1280)

# Load
metadata = load_protocox_model(
    model=model,
    loss_fn=loss_fn,
    load_path="model.pth",
    device=device
)

# Now loss_fn.risk_prototype contains the loaded prototype
# You can use it for evaluation
```

## Training

### Standard ProtoCoxLoss (PCL)

```python
from proto_cox_loss import ProtoCoxLoss, ProtoCoxLossTrainer

# Create model and loss
model = create_model()
loss_fn = ProtoCoxLoss(input_dim=1280, temperature=0.07, use_ipcw=True)

# Train
trainer = ProtoCoxLossTrainer(device=device, epochs=50)
results = trainer.train_model(
    model=model,
    loss_fn=loss_fn,
    dataloader_train=train_loader,
    dataloader_val=val_loader,
    loss_type="pcl"
)
```

### ProtoCoxLoss with MoCo (PCL+MoCo)

```python
from proto_cox_loss import ProtoCoxLoss, MoCoCoxWrapper, ProtoCoxLossTrainer

# Create model and loss
model = create_model()
loss_fn = ProtoCoxLoss(input_dim=1280, temperature=0.07, use_ipcw=True)

# Wrap with MoCo
moco_model = MoCoCoxWrapper(
    backbone=model,
    dim=1280,
    queue_size=1024,
    m=0.999,
    temperature=0.07
)

# Train
trainer = ProtoCoxLossTrainer(device=device, epochs=50)
results = trainer.train_model(
    model=moco_model,
    loss_fn=loss_fn,
    dataloader_train=train_loader,
    dataloader_val=val_loader,
    loss_type="pcl_moco"
)
```

## Files

- `loss.py`: ProtoCoxLoss implementation
- `moco_wrapper.py`: MoCo wrapper for ProtoCoxLoss
- `trainer.py`: Custom training loop for ProtoCoxLoss models
- `__init__.py`: Package exports

