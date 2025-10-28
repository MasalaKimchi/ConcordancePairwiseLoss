# Enhanced TorchSurv MNIST Training Script

This enhanced version of the TorchSurv MNIST training script adds support for multiple loss functions, configurable batch sizes, and comprehensive evaluation metrics.

#### Dataset Creation Methodology

The SurvMNIST dataset converts MNIST handwritten digits into a survival analysis problem using a direct mapping approach:

**Survival Time Assignment:**
- Each MNIST digit (0-9) is mapped to a survival time
- Digit 0 → survival time = 10 (to avoid log(0) issues in Cox models)
- Digits 1-9 → survival time = digit value (1, 2, 3, ..., 9)
- **Time range**: All survival times fall between 1 and 10 units

**Censoring Mechanism (30% rate):**
1. For each sample at index `i`, uses `random.seed(i)` for reproducibility
2. Determines censoring: `is_censored = random.random() < 0.3`
3. If censored (30% of samples):
   - Event indicator: `event = False`
   - Observed time: `random.uniform(1.0, true_survival_time)` 
   - Censoring occurs before the event
4. If not censored (70% of samples):
   - Event indicator: `event = True`
   - Observed time: `true_survival_time`

**Why This Design?**
- **Simplicity**: Direct digit-to-time mapping makes the problem interpretable
- **Reproducibility**: Index-based seeding ensures consistent train/test splits
- **IPCW Relevance**: 30% censoring rate creates sufficient censored data for IPCW weighting
- **Model Testing**: Known ground truth allows validation of survival model predictions
- **Quick Iteration**: MNIST's fast loading enables rapid algorithm development

**Dataset Statistics:**
- Training samples: 60,000 (42,000 events / 18,000 censored)
- Test samples: 10,000 (7,000 events / 3,000 censored)
- Survival time distribution: Depends on digit frequency in MNIST
- Image size: 28×28 grayscale, resized to 224×224 for ResNet

**For detailed usage and configuration**, see [src/survmnist/README.md](src/survmnist/README.md) which includes:
- Dataset configuration and censoring methodology
- Loss function variants explanation
- Command-line arguments reference
- Output format specifications

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

# Run with CPL (dynamic) - IPCW computed per batch
python benchmarks/benchmark_survmnist.py --loss-type cpl_dynamic --batch-size 64 --epochs 5

# Run with CPL (static) - IPCW precomputed from full training set
python benchmarks/benchmark_survmnist.py --loss-type cpl_static --batch-size 64 --epochs 5
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
| `--loss-type` | str | nll | Loss function: nll, cpl_dynamic, cpl_static |
| `--temperature` | float | 1.0 | Temperature for CPL losses |
| `--output-dir` | str | results | Output directory for results |
| `--limit-train-batches` | float | 0.1 | Fraction of training data to use |
| `--compare-all` | flag | False | Run comparison across all configurations |
| `--batch-sizes` | list | [32,64,128,256] | Batch sizes for comparison |


## Key Differences from Original TorchSurv Example

### 1. **Loss Function Support**
- Original: Only NLL (Cox loss)
- Enhanced: NLL + 2 CPL variants (dynamic/static)

### 2. **Batch Size Configuration**
- Original: Fixed 500 (GPU) / 50 (CPU)
- Enhanced: Configurable 32, 64, 128, 256

### 3. **Evaluation Metrics**
- Original: Basic C-index only
- Enhanced: Harrell's C, Uno's C, Cumulative AUC, Incident AUC (t=5.0), Brier Score

### 4. **IPCW Handling**
- Original: Not applicable
- Enhanced: Per-batch and precomputed IPCW weights

### 5. **Results Saving**
- Original: No persistent results
- Enhanced: JSON output with comprehensive metrics
