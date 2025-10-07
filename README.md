# ConcordancePairwiseLoss

A pairwise loss function for survival analysis that improves concordance between predicted and actual survival times by comparing pairs of samples.

## Quick Start

### Installation
```bash
pip install torch torchsurv lifelines scikit-survival SurvSet pycox scikit-learn pandas numpy matplotlib seaborn
```

### Basic Usage
```python
from src.concordance_pairwise_loss import ConcordancePairwiseLoss

# Create loss function
loss_fn = ConcordancePairwiseLoss(
    temperature=1.0,
    temp_scaling='linear',
    pairwise_sampling='balanced',
    use_ipcw=False
)

# Use in training
loss = loss_fn(log_risks, times, events)
```

### Normalized Loss Combination
```python
from src.concordance_pairwise_loss import NormalizedLossCombination

# Create normalized loss combination
loss_combiner = NormalizedLossCombination(total_epochs=100)

# Get weights during training
nll_w, pairwise_w = loss_combiner.get_weights_scale_balanced(
    epoch, nll_loss, pairwise_loss
)
```

## Methods

### Loss Function Variants

- **NLL (Negative Partial Log-Likelihood)**: Traditional Cox proportional hazards partial log-likelihood loss
- **CPL (Concordance Pairwise Loss)**: Directly optimizes concordance via pairwise ranking
- **CPL (dynamic)**: CPL with Inverse Probability of Censoring Weights (IPCW) computed dynamically per batch
- **CPL (static)**: CPL with IPCW weights precomputed from the full training set and reused across batches

### Terminology Mapping

For clarity in documentation and code:

| **Documentation Name** | **Code Name** | **Description** |
|------------------------|---------------|-----------------|
| NLL | `nll` | Cox proportional hazards loss |
| CPL | `cpl` | Base concordance pairwise loss |
| CPL (dynamic) | `cpl_ipcw` | IPCW computed per batch |
| CPL (static) | `cpl_ipcw_batch` | IPCW precomputed once |

**Why "dynamic" vs "static"?**
- **Dynamic**: IPCW weights are recalculated for each batch during training, adapting to the specific samples
- **Static**: IPCW weights are computed once from the entire training set and remain fixed throughout training

## Running Benchmarks

Use the V2 benchmark framework to evaluate models across available datasets with comprehensive metrics.

### Tabular Datasets

```bash
# Run on a single dataset (recommended for tabular datasets)
conda activate concordance-pairwise-loss
python benchmarks/benchmark_tabular_v2.py --dataset gbsg2 --epochs 30 --num-runs 10

# Run on all tabular datasets
python benchmarks/benchmark_tabular_v2.py --dataset all --epochs 30 --num-runs 10
```

**Key Features**:
- Tests 4 core loss functions: NLL, CPL, CPL (dynamic), CPL (static)
- Hyperparameter grid search across learning rates and hidden dimensions
- Multiple independent runs for statistical robustness
- Comprehensive metrics: Harrell's C, Uno's C, Cumulative AUC, Incident AUC, Brier Score

### Medical Imaging (MIMIC-IV)

```bash
# Experimentation run with full dataset
python benchmarks/benchmark_MIMIC_v2.py --epochs 25 --batch-size 64
```

**See [src/mimic/README.md](src/mimic/README.md) for complete MIMIC-IV setup and usage guide.**

### Available Datasets
- **Large tabular**: FLChain (7,874), SUPPORT2 (8,873)
- **Medium tabular**: GBSG2 (686), WHAS500 (500), METABRIC (1,903)
- **Medical Imaging**: MIMIC-IV Chest X-ray (~300k images)

## Project Structure

```
ConcordancePairwiseLoss/
├── src/                          # Source code
│   ├── concordance_pairwise_loss/
│   │   ├── __init__.py
│   │   └── loss.py               # Main ConcordancePairwiseLoss implementation
│   ├── abstract_data_loader.py   # Base class for dataset loaders
│   ├── data_loaders.py           # Implementations of supported datasets
│   ├── mimic/                    # MIMIC-IV medical imaging module
│   ├── dataset_configs.py        # Utility for loading dataset metadata
│   └── flexible_dataset.py       # Dataset utilities
├── benchmarks/
│   ├── benchmark_tabular_v2.py   # ⭐ Tabular datasets benchmark
│   ├── benchmark_MIMIC_v2.py     # ⭐ MIMIC-IV imaging benchmark
│   └── dataset_configs.json      # Dataset-specific evaluation settings
├── README.md                     # This file
└── src/mimic/README.md          # MIMIC-IV specific documentation
```


## License

See LICENSE file for details.
