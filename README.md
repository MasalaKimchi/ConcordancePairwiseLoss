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

- **Negative Partial Log-Likelihood (NLL)**: Traditional Cox proportional hazards partial log-likelihood loss
- **Pairwise Ranking Loss**: ConcordancePairwiseLoss that directly optimizes concordance
- **Normalized Loss Combination**: Scale-balanced combination using fixed normalization factors (25.0 for NLL, 2.0 for Pairwise)

## Running Benchmarks

Use the unified benchmark framework to evaluate models across available datasets.

### Quick Start
```bash
python benchmarks/benchmark_framework_improved.py --dataset gbsg2 --loss-type nll --epochs 3
```

### Adding a New Dataset
1. Implement a loader inheriting from `AbstractDataLoader` in `src/data_loaders.py` and register it in `DATA_LOADERS`.
2. Add the dataset's AUC horizon and time units to `benchmarks/dataset_configs.json`.
3. Run the benchmark with `--dataset <name>`.

### Available Datasets
- **Large datasets**: FLChain (7,874), SUPPORT2 (8,873)
- **Medium datasets**: GBSG2 (686), WHAS500 (500), METABRIC (1,903)
- **Medical Imaging**: MIMIC-IV Chest X-ray (see [src/mimic/README.md](src/mimic/README.md) for details)

### Statistical Analysis
Each benchmark provides professional statistical analysis:
- **Multi-run support**: Mean ± standard deviation across independent runs
- **Comprehensive metrics**: Harrell C-index, Uno C-index, AUC, Brier Score
- **Rich visualizations**: 6-panel analysis plots with fixed dimensions
- **Professional output**: JSON + CSV + PNG files with timestamps


## Project Structure

```
ConcordancePairwiseLoss/
├── src/                          # Source code
│   ├── concordance_pairwise_loss/
│   │   ├── __init__.py
│   │   ├── loss.py               # Main ConcordancePairwiseLoss implementation
│   │   └── dynamic_weighting.py  # Dynamic weighting strategies
│   ├── abstract_data_loader.py   # Base class for dataset loaders
│   ├── data_loaders.py           # Implementations of supported datasets
│   ├── mimic/                    # MIMIC-IV medical imaging module
│   ├── dataset_configs.py        # Utility for loading dataset metadata
│   └── flexible_dataset.py       # Dataset utilities
├── benchmarks/
│   ├── benchmark_framework.py      # Legacy benchmark framework
│   ├── benchmark_framework_improved.py  # Current benchmark framework
│   └── dataset_configs.json        # Dataset-specific evaluation settings
├── src/mimic/preprocess.py       # MIMIC-IV preprocessing script
├── README.md                     # This file
└── src/mimic/README.md          # MIMIC-IV specific documentation
```


## License

See LICENSE file for details.
