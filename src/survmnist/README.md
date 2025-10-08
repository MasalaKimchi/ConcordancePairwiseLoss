# Enhanced TorchSurv MNIST Training Script

This enhanced version of the TorchSurv MNIST training script adds support for multiple loss functions, configurable batch sizes, and comprehensive evaluation metrics.

## Dataset Configuration

**Default Censoring Rate**: 30% (0.3)
- Censoring is applied reproducibly using sample index as seed
- For censored samples, censoring time is randomly drawn between 1 and the true survival time
- This creates realistic survival data where IPCW is meaningful

## Usage

### Single Experiment
```bash
# Run with default parameters (NLL, batch_size=64, epochs=2)
python benchmarks/benchmark_survmnist.py

# Run with CPL (online) - IPCW computed per batch
python benchmarks/benchmark_survmnist.py --loss-type cpl_online --batch-size 64 --epochs 5

# Run with CPL (offline) - IPCW precomputed from full training set
python benchmarks/benchmark_survmnist.py --loss-type cpl_offline --batch-size 64 --epochs 5
```

### Comprehensive Comparison
```bash
# Run comparison across all loss types and batch sizes
python benchmarks/benchmark_survmnist.py --compare-all

# Run comparison with custom batch sizes
python benchmarks/benchmark_survmnist.py --compare-all --batch-sizes 32 64 128 256

# Run comparison with more training data
python benchmarks/benchmark_survmnist.py --compare-all --limit-train-batches 0.2
```


## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 64 | Batch size for training |
| `--epochs` | int | 2 | Number of training epochs |
| `--loss-type` | str | nll | Loss function: nll, cpl_online, cpl_offline |
| `--temperature` | float | 1.0 | Temperature for CPL losses |
| `--output-dir` | str | results | Output directory for results |
| `--limit-train-batches` | float | 0.1 | Fraction of training data to use |
| `--compare-all` | flag | False | Run comparison across all configurations |
| `--batch-sizes` | list | [32,64,128,256] | Batch sizes for comparison |

## Output Files

The script generates JSON output files with all metrics and metadata:

### JSON Results (`mnist_batch_<batch_size>_YYYYMMDD_HHMMSS.json`)
Complete results with all metrics:
```json
[
  {
    "loss_type": "cpl_online",
    "batch_size": 64,
    "epochs": 5,
    "temperature": 1.0,
    "metrics": {
      "harrell_cindex": 0.7234,
      "uno_cindex": 0.7456,
      "cumulative_auc": 0.7123,
      "brier_score": 0.2345
    }
  }
]
```

For comparison mode (--compare-all), the filename is `mnist_batch_YYYYMMDD_HHMMSS.json` (without specific batch size).

## Key Differences from Original TorchSurv Example

### 1. **Loss Function Support**
- Original: Only NLL (Cox loss)
- Enhanced: NLL + 2 CPL variants (online/offline)

### 2. **Batch Size Configuration**
- Original: Fixed 500 (GPU) / 50 (CPU)
- Enhanced: Configurable 32, 64, 128, 256

### 3. **Evaluation Metrics**
- Original: Basic C-index only
- Enhanced: Harrell's C, Uno's C, Cumulative AUC, Brier Score

### 4. **IPCW Handling**
- Original: Not applicable
- Enhanced: Per-batch and precomputed IPCW weights

### 5. **Results Saving**
- Original: No persistent results
- Enhanced: JSON output with comprehensive metrics
