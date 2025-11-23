# Understanding the ProtoCoxLoss Prototype

## What is the Prototype?

The **prototype** in ProtoCoxLoss is a learnable vector (parameter) that represents the "direction of high risk" in the embedding space. Think of it as a reference point that the model learns during training.

### Mathematical Definition

- **Dimension**: Same as patient embeddings (e.g., 1280 for EfficientNet-B0)
- **Location**: On the unit hypersphere (normalized to length 1)
- **Purpose**: Risk score = cosine similarity between patient embedding and prototype

```
risk_score = cos(patient_embedding, prototype) / temperature
```

### Visual Analogy

Imagine a 2D space:
- Each patient is a point on a circle (unit hypersphere)
- The prototype is another point on the circle
- Patients closer to the prototype have higher risk
- During training, the prototype "rotates" to find the best position

## Why Save and Load?

### The Problem

The prototype is **learned during training**, just like model weights. It's not a fixed value that can be recomputed.

### Why It Matters

1. **During Training**: The prototype is updated via backpropagation
2. **During Evaluation**: We need the same prototype to compute risk scores
3. **Without It**: We cannot compute accurate risk scores for new patients

### Example Scenario

```python
# Training
loss_fn = ProtoCoxLoss(input_dim=1280)
# ... train for 50 epochs ...
# Now loss_fn.risk_prototype has learned the optimal direction

# Evaluation (WRONG - prototype is lost!)
model = load_model("model.pth")  # Only loads model weights
features = model.extract_features(patient)
# How do we compute risk? We need the prototype!

# Evaluation (CORRECT - prototype is saved!)
model, loss_fn = load_protocox_model("model.pth")
features = model.extract_features(patient)
features_norm = F.normalize(features, p=2, dim=1)
proto_norm = F.normalize(loss_fn.risk_prototype, p=2, dim=0)
risk_score = torch.matmul(features_norm, proto_norm) / loss_fn.temperature
```

## Implementation Details

### Saving

The prototype is saved in three ways:

1. **In loss function state dict**: `loss_fn.state_dict()` contains `risk_prototype`
2. **Separately**: As `'prototype'` key for easy access
3. **In model metadata**: For reference

```python
save_dict = {
    'model_state_dict': model.state_dict(),
    'loss_fn_state_dict': loss_fn.state_dict(),  # Contains prototype
    'prototype': loss_fn.risk_prototype.data.clone(),  # Easy access
    'metadata': {...}
}
```

### Loading

When loading, we restore both the model and the prototype:

```python
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])  # Restores prototype
```

### Verification

After loading, we verify the prototype was restored correctly:

```python
assert torch.allclose(
    loss_fn.risk_prototype.data,
    checkpoint['prototype'].to(device),
    atol=1e-6
), "Prototype mismatch!"
```

## Best Practices

1. **Always save the prototype** when saving ProtoCoxLoss models
2. **Always load the prototype** when loading ProtoCoxLoss models
3. **Use the provided functions**: `save_protocox_model()` and `load_protocox_model()`
4. **Verify after loading**: Check that the prototype matches
5. **Include in evaluation**: Pass `loss_fn` to evaluator so it can use the prototype

## Common Mistakes

### Mistake 1: Only Saving Model Weights

```python
# WRONG
torch.save(model.state_dict(), "model.pth")
# Prototype is lost!
```

### Mistake 2: Not Using Prototype During Evaluation

```python
# WRONG
features = model.extract_features(patient)
risk = model.risk_head(features)  # Uses linear head, not prototype!
```

### Mistake 3: Using Dummy Prototype

```python
# WRONG
dummy_proto = torch.ones(1280)  # Not the learned prototype!
risk = cos(features, dummy_proto)  # Inaccurate!
```

## Summary

- **Prototype = Learnable parameter** that represents risk direction
- **Must be saved** along with model weights
- **Must be loaded** for accurate evaluation
- **Use provided functions** to ensure correctness
- **Think of it as part of the model** - both are needed for inference

