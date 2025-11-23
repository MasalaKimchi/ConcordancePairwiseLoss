# Debugging ProtoCoxLoss on MIMIC Dataset

## Quick Debug Command

For quick debugging to verify the code works:

```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --num-runs 1 \
    --patience 2 \
    --max-steps 100 \
    --output-dir results-pcl-debug
```

## Recommended Debugging Configurations

### Minimal Test (Fastest - ~5 minutes)
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --num-runs 1 \
    --patience 2 \
    --max-steps 50 \
    --output-dir results-pcl-debug
```

**What this does:**
- Uses only 1% of data (~100-200 samples)
- Trains for 2 epochs max
- Small batch size (32)
- Stops after 50 steps or 2 epochs
- Single run (no averaging)

### Standard Debug (Medium - ~15 minutes)
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.05 \
    --epochs 5 \
    --batch-size 64 \
    --num-runs 1 \
    --patience 3 \
    --max-steps 200 \
    --output-dir results-pcl-debug
```

**What this does:**
- Uses 5% of data (~500-1000 samples)
- Trains for 5 epochs max
- Medium batch size (64)
- Stops after 200 steps or 3 epochs without improvement
- Single run

### Full Debug (Slower - ~30 minutes)
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.1 \
    --epochs 10 \
    --batch-size 128 \
    --num-runs 1 \
    --patience 5 \
    --max-steps 500 \
    --output-dir results-pcl-debug
```

## Testing Both ProtoCoxLoss Variants

### Test Standard ProtoCoxLoss Only
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --output-dir results-pcl-debug
```

### Test ProtoCoxLoss with MoCo Only
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl_moco \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --output-dir results-pcl-debug
```

### Test Both (Using Shortcut)
```bash
python benchmarks/benchmark_MIMIC_pcl.py \
    --pcl-only \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --output-dir results-pcl-debug
```

## Key Arguments for Debugging

| Argument | Debug Value | Purpose |
|----------|-------------|---------|
| `--loss-types` | `pcl` | Test only ProtoCoxLoss (not MoCo) |
| `--data-fraction` | `0.01` to `0.1` | Use small subset for speed |
| `--epochs` | `2` to `5` | Few epochs for quick test |
| `--batch-size` | `32` to `64` | Smaller batches for debugging |
| `--num-runs` | `1` | Single run (no averaging) |
| `--patience` | `2` to `3` | Early stopping patience |
| `--max-steps` | `50` to `200` | Limit training steps |
| `--output-dir` | `results-pcl-debug` | Separate output directory |

## What to Check During Debugging

### 1. **Loss Values**
- Should be finite and positive
- Should decrease over epochs
- Check for NaN or Inf values

### 2. **Prototype**
- Should be learnable (changes during training)
- Should be normalized (check `loss_fn.risk_prototype.norm()` ≈ 1.0)
- Should be saved in model checkpoint

### 3. **Gradients**
- Check that gradients flow to both model and prototype
- No gradient explosion or vanishing

### 4. **Risk Set Computation**
- Verify risk sets are computed correctly
- Check that queue (for MoCo) is being updated

### 5. **Evaluation Metrics**
- C-index should be between 0.5 and 1.0
- Check that evaluation uses prototype correctly

## Common Issues and Solutions

### Issue: "Prototype not found"
**Solution**: Make sure `loss_fn` is passed to evaluator:
```python
evaluator.loss_fn = trainer.pcl_loss
```

### Issue: "Loss is NaN"
**Solution**: 
- Check temperature (should be > 0.01)
- Check that features are normalized
- Check that risk sets are not empty

### Issue: "No gradients"
**Solution**:
- Verify model and loss_fn are both in training mode
- Check that optimizer includes both model and loss_fn parameters

### Issue: "Queue not updating"
**Solution** (for MoCo):
- Verify `_dequeue_and_enqueue` is called
- Check queue pointer is updating
- Verify keys (k) are being used, not queries (q)

## Debugging Checklist

- [ ] Code runs without errors
- [ ] Loss values are finite
- [ ] Prototype is learnable (changes during training)
- [ ] Gradients flow to model and prototype
- [ ] Model saves correctly (with prototype)
- [ ] Model loads correctly (prototype restored)
- [ ] Evaluation uses prototype correctly
- [ ] Metrics are reasonable (C-index > 0.5)
- [ ] Risk sets computed correctly
- [ ] Queue updates (for MoCo version)

## Example Debug Session

```bash
# 1. Quick test - verify code runs
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.01 \
    --epochs 1 \
    --batch-size 16 \
    --max-steps 10 \
    --output-dir results-pcl-debug

# 2. If successful, test with more data
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl \
    --data-fraction 0.05 \
    --epochs 3 \
    --batch-size 32 \
    --output-dir results-pcl-debug

# 3. Test MoCo version
python benchmarks/benchmark_MIMIC_pcl.py \
    --loss-types pcl_moco \
    --data-fraction 0.01 \
    --epochs 2 \
    --batch-size 32 \
    --output-dir results-pcl-debug
```

## Expected Output

You should see:
- Training progress bars
- Loss values decreasing
- Validation metrics (C-index, AUC, etc.)
- Model saved with prototype
- Evaluation results

If everything works, you'll see:
```
✅ Saved ProtoCoxLoss model with prototype to: results-pcl-debug/models/pcl_best/...
```

