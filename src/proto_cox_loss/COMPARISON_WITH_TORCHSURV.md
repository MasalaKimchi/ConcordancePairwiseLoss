# Comparison: ProtoCoxLoss+MoCo vs Torchsurv Momentum Loss

## Overview

Both approaches use momentum/queue concepts inspired by MoCo (Momentum Contrast), but they apply them **differently** and for **different purposes** in survival analysis.

## Torchsurv's Momentum Loss

### What It Does
- **Purpose**: Increases effective sample size for loss computation
- **Mechanism**: Maintains a queue of past batch outputs (features/logits)
- **Application**: Works with **any loss function** (Cox NLL, etc.)
- **Goal**: Better training with small batch sizes by using historical batches

### How It Works
```python
# Torchsurv Momentum (conceptual)
# 1. Compute loss on current batch
current_loss = loss_fn(current_features, current_times, current_events)

# 2. Maintain queue of past batches
queue = [batch1_features, batch2_features, ..., batchN_features]

# 3. Combine current + queue for larger effective batch
all_features = concat([current_features, queue])
all_times = concat([current_times, queue_times])
all_events = concat([current_events, queue_events])

# 4. Compute loss on combined data (larger sample size)
combined_loss = loss_fn(all_features, all_times, all_events)
```

### Key Characteristics
- **Loss-agnostic**: Works with any survival loss function
- **Simple queue**: Just stores past batch outputs
- **No momentum encoder**: Uses current model to encode queue entries
- **Direct extension**: Extends batch size for loss computation

## ProtoCoxLoss + MoCo (Our Approach)

### What It Does
- **Purpose**: Expands **risk set denominator** in Cox partial likelihood
- **Mechanism**: Maintains a queue of **stable features** from momentum encoder
- **Application**: Specifically designed for **ProtoCoxLoss** (geometric Cox)
- **Goal**: More accurate risk set computation with historical context

### How It Works
```python
# ProtoCoxLoss + MoCo
# 1. Query encoder (main model, receives gradients)
q = backbone_q(x)  # Current batch features

# 2. Key encoder (momentum copy, no gradients)
k = backbone_k(x)  # Stable features for queue

# 3. Update momentum encoder
backbone_k = m * backbone_k + (1-m) * backbone_q

# 4. Compute risk scores using prototype
logits_q = cos(q, prototype) / temperature
logits_queue = cos(queue_features, prototype) / temperature

# 5. Expand risk set: [current batch] + [queue]
all_logits = concat([logits_q, logits_queue])
all_times = concat([current_times, queue_times])

# 6. Compute Cox loss with expanded risk set
risk_mask = (all_times >= current_times)  # Risk set for each event
denom = sum(exp(all_logits) * risk_mask)  # Expanded denominator
loss = -[logits_q - log(denom)]  # Cox partial likelihood

# 7. Update queue with stable keys (k), not queries (q)
queue.enqueue(k, times, events)
```

### Key Characteristics
- **Loss-specific**: Designed specifically for ProtoCoxLoss
- **Momentum encoder**: Maintains EMA copy of backbone for stability
- **Risk set expansion**: Expands denominator, not just batch size
- **Geometric**: Uses cosine similarity with learnable prototype

## Key Differences

| Aspect | Torchsurv Momentum | ProtoCoxLoss + MoCo |
|--------|-------------------|---------------------|
| **Purpose** | Increase effective batch size | Expand risk set denominator |
| **Loss Function** | Works with any loss | Specific to ProtoCoxLoss |
| **Encoder** | Single model (current) | Dual encoders (query + momentum key) |
| **Queue Content** | Past batch outputs | Stable features from momentum encoder |
| **What's Expanded** | Entire loss computation | Risk set in Cox denominator |
| **Gradient Flow** | Through current model | Through query encoder only |
| **Stability** | Queue may have inconsistent features | Queue has stable features (momentum) |
| **Prototype** | Not used | Central to risk computation |

## Detailed Comparison

### 1. **Architecture**

**Torchsurv Momentum:**
```
Model → Features → Queue → Loss (larger batch)
```

**ProtoCoxLoss + MoCo:**
```
Query Encoder → Features → Prototype → Risk Scores
Key Encoder (momentum) → Stable Features → Queue → Expanded Risk Set
```

### 2. **Queue Management**

**Torchsurv Momentum:**
- Stores outputs from **current model** at time of enqueue
- Features may become inconsistent as model updates
- Simple FIFO queue

**ProtoCoxLoss + MoCo:**
- Stores outputs from **momentum encoder** (EMA copy)
- Features remain stable even as query encoder updates
- Queue updated with stable keys (k), not queries (q)

### 3. **Loss Computation**

**Torchsurv Momentum:**
```python
# Computes loss on combined batch
loss = loss_fn(all_features, all_times, all_events)
# Risk set computed from all_features
```

**ProtoCoxLoss + MoCo:**
```python
# Computes Cox loss with expanded risk set
# Current batch: query features (q)
# Risk set: [current batch] + [queue]
# Denominator includes historical patients in risk set
loss = -[logits_q - log(sum(exp(logits_all) * risk_mask))]
```

### 4. **Mathematical Formulation**

**Torchsurv Momentum:**
```
L = Loss(features_current ∪ features_queue, times, events)
```

**ProtoCoxLoss + MoCo:**
```
L = -Σ_{i: Event} [ sim(z_i, w)/τ - log( Σ_{j in RiskSet(i)} exp(sim(z_j, w)/τ) ) ]

Where RiskSet(i) = {j in (current_batch ∪ queue) | t_j ≥ t_i}
```

## When to Use Which?

### Use Torchsurv Momentum When:
- ✅ You want to use it with **any loss function** (Cox NLL, etc.)
- ✅ You want **simple implementation** (just queue past batches)
- ✅ You want to **increase effective batch size** for any loss
- ✅ You don't need momentum encoder stability

### Use ProtoCoxLoss + MoCo When:
- ✅ You're using **ProtoCoxLoss** (geometric Cox)
- ✅ You want **stable features** in the queue (momentum encoder)
- ✅ You want to expand **risk set denominator** specifically
- ✅ You need **consistent queue features** as model updates
- ✅ You want **geometric contrastive** formulation

## Summary

**Torchsurv Momentum** is a **general-purpose** technique to increase effective batch size for any survival loss function. It's simple and loss-agnostic.

**ProtoCoxLoss + MoCo** is a **specialized** technique for ProtoCoxLoss that:
1. Uses momentum encoder for stable features
2. Expands the risk set denominator (not just batch size)
3. Integrates with the geometric contrastive formulation
4. Maintains consistency as the model updates

The key insight: **Torchsurv expands the batch, we expand the risk set denominator** - these are different operations in survival analysis!

