# Enhanced TorchSurv MNIST Training Script

This enhanced version of the TorchSurv MNIST training script adds support for multiple loss functions, configurable batch sizes, and comprehensive evaluation metrics.

## Features

### Loss Functions
- **NLL**: Negative Log-Likelihood (Cox loss) - baseline
- **CPL**: Concordance Pairwise Loss - direct concordance optimization
- **CPL (dynamic)**: CPL with Inverse Probability of Censoring Weights (IPCW) computed dynamically per batch
- **CPL (static)**: CPL with IPCW weights precomputed once from the full training set and reused

### Configurable Parameters
- **Batch sizes**: 32, 64, 128, 256 (configurable)
- **Epochs**: 5 (configurable)
- **Temperature**: For CPL losses (0.5, 1.0)
- **Training data fraction**: 10% by default (configurable)

### Evaluation Metrics
- **Harrell's C-index**: Standard concordance index without IPCW weights
- **Uno's C-index**: Concordance index with IPCW weights (more appropriate for censored data)
- **Cumulative AUC**: Area under the curve over all observed times (useful for discrimination assessment)
- **Brier Score**: Calibration metric at median time point

## Usage

### Single Experiment
```bash
# Run with default parameters (NLL, batch_size=64, epochs=5)
python train_torchsurv_mnist_enhanced.py

# Run with specific parameters (CPL)
python train_torchsurv_mnist_enhanced.py --loss-type cpl --batch-size 128 --epochs 5 --temperature 1.0

# Run with CPL (dynamic) - IPCW computed per batch
python train_torchsurv_mnist_enhanced.py --loss-type cpl_ipcw --batch-size 64 --epochs 5

# Run with CPL (static) - IPCW precomputed from full training set
python train_torchsurv_mnist_enhanced.py --loss-type cpl_ipcw_batch --batch-size 64 --epochs 5
```

### Comprehensive Comparison
```bash
# Run comparison across all loss types and batch sizes
python train_torchsurv_mnist_enhanced.py --compare-all

# Run comparison with custom batch sizes
python train_torchsurv_mnist_enhanced.py --compare-all --batch-sizes 32 64 128 256

# Run comparison with more training data
python train_torchsurv_mnist_enhanced.py --compare-all --limit-train-batches 0.2
```


## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 64 | Batch size for training |
| `--epochs` | int | 5 | Number of training epochs |
| `--loss-type` | str | nll | Loss function: nll, cpl, cpl_ipcw, cpl_ipcw_batch |
| `--temperature` | float | 1.0 | Temperature for CPL losses |
| `--output-dir` | str | results | Output directory for results |
| `--limit-train-batches` | float | 0.1 | Fraction of training data to use |
| `--compare-all` | flag | False | Run comparison across all configurations |
| `--batch-sizes` | list | [32,64,128,256] | Batch sizes for comparison |

## Output Files

The script generates two types of output files:

### JSON Results (`mnist_survival_results_YYYYMMDD_HHMMSS.json`)
Complete results with all metrics and metadata:
```json
[
  {
    "loss_type": "cpl",
    "batch_size": 64,
    "epochs": 5,
    "temperature": 1.0,
    "metrics": {
      "harrell_cindex": 0.7234,
      "uno_cindex": 0.7456,
      "cumulative_auc": 0.7123,
      "incident_auc": 0.7890,
      "brier_score": 0.2345
    }
  }
]
```

### CSV Results (`mnist_survival_results_YYYYMMDD_HHMMSS.csv`)
Tabular format for easy analysis:
```csv
loss_type,batch_size,epochs,temperature,harrell_cindex,uno_cindex,cumulative_auc,incident_auc,brier_score
cpl,64,5,1.0,0.7234,0.7456,0.7123,0.7890,0.2345
```

## Key Differences from Original TorchSurv Example

### 1. **Loss Function Support**
- Original: Only NLL (Cox loss)
- Enhanced: NLL + 3 CPL variants

### 2. **Batch Size Configuration**
- Original: Fixed 500 (GPU) / 50 (CPU)
- Enhanced: Configurable 32, 64, 128, 256

### 3. **Evaluation Metrics**
- Original: Basic C-index only
- Enhanced: Harrell's C, Uno's C, Cumulative AUC, Incident AUC, Brier Score

### 4. **IPCW Handling**
- Original: Not applicable
- Enhanced: Per-batch and precomputed IPCW weights

### 5. **Results Saving**
- Original: No persistent results
- Enhanced: JSON and CSV output with comprehensive metrics

## C-index Explanation

### Harrell's C-index
- Standard concordance index without IPCW weights
- Computed as: `ConcordanceIndex(log_hz, events, times)`
- Good for general ranking assessment

### Uno's C-index
- Concordance index with IPCW weights to handle censoring
- Computed as: `ConcordanceIndex(log_hz, events, times, weight=ipcw_weights)`
- More appropriate for survival analysis with censored data
- **This is what you should focus on for your ISBI paper**

## AUC Metrics for Survival MNIST

### Cumulative AUC
- Measures discrimination ability over all observed time points
- Appropriate for survival analysis as it considers the entire time course
- **Recommended for your analysis**

### Incident AUC
- Measures discrimination at a specific time point (median time)
- Useful for understanding performance at a particular horizon
- **Also recommended for comprehensive evaluation**

## Batch Size Analysis for ISBI Paper

This script is designed to support your ISBI paper claim that CPL has advantages for medical imaging at various batch sizes:

1. **Run comprehensive comparison**: `--compare-all`
2. **Analyze results** to show CPL performance across batch sizes
3. **Compare against NLL** to demonstrate robustness
4. **Focus on Uno's C-index** as the primary metric

## Dependencies

Make sure you have the conda environment activated:
```bash
conda activate concordance-pairwise-loss
```

Required packages:
- torch
- lightning
- torchvision
- torchsurv
- matplotlib
- seaborn
- numpy
- pandas

## Example Workflow for ISBI Paper

1. **Test the setup**:
   ```bash
   python test_enhanced_mnist.py
   ```

2. **Run comprehensive comparison**:
   ```bash
   python train_torchsurv_mnist_enhanced.py --compare-all --epochs 5
   ```

3. **Analyze results** in the generated CSV file to show:
   - CPL performance across batch sizes
   - Comparison with NLL baseline
   - Focus on Uno's C-index as primary metric

4. **Create visualizations** from the results to demonstrate batch size robustness

This enhanced script provides everything you need to validate your claim about CPL's advantages for medical imaging at various batch sizes!
