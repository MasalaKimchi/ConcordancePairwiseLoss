# ConcordancePairwiseLoss

A pairwise loss function for survival analysis that improves concordance between predicted and actual survival times by comparing pairs of samples.

## Quick Start

### Installation
```bash
pip install torch torchsurv lifelines scikit-learn pandas numpy matplotlib seaborn
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

The comprehensive benchmark suite includes **9 diverse survival analysis datasets** for robust evaluation. See `benchmarks/README.md` for complete documentation.

### Quick Start
```bash
# Quick test (3 epochs, no saving)
python benchmarks/flchain_benchmark.py --epochs 3 --no-save

# Standard evaluation (50 epochs, save results)
python benchmarks/gbsg2_benchmark.py

# Multiple runs for statistical robustness
python benchmarks/whas500_benchmark.py --runs 5 --epochs 100
```

### Available Datasets
- **Large datasets**: FLChain (7,874), SUPPORT2 (9,105) - Best for robust evaluation
- **Medium datasets**: GBSG2 (686), WHAS500 (500), Rossi (432), Lung (228), Cancer (228)
- **Small datasets**: Breast Cancer (198), Veterans (137) - Quick testing

### Statistical Analysis
Each benchmark provides professional statistical analysis:
- **Multi-run support**: Mean ± standard deviation across independent runs
- **Comprehensive metrics**: Harrell C-index, Uno C-index, AUC, Brier Score
- **Rich visualizations**: 6-panel analysis plots with fixed dimensions
- **Professional output**: JSON + CSV + PNG files with timestamps

## Key Findings

- **Pairwise Ranking Loss** often provides the best performance across different survival analysis datasets
- **Normalized Loss Combination** can provide improvements when combining NLL and Pairwise Ranking Loss
- **Adaptive temperature scaling** and **balanced sampling** are key improvements

## Project Structure

```
ConcordancePairwiseLoss/
├── src/                          # Source code
│   ├── concordance_pairwise_loss/
│   │   ├── __init__.py
│   │   ├── loss.py               # Main ConcordancePairwiseLoss implementation
│   │   └── dynamic_weighting.py  # Dynamic weighting strategies
│   └── flexible_dataset.py       # Dataset utilities
├── benchmarks/                   # Comprehensive benchmark suite
│   ├── benchmark_framework.py      # Shared framework and components
│   ├── gbsg2_benchmark.py          # GBSG2 dataset benchmark
│   ├── lung_benchmark.py           # Lung cancer dataset benchmark
│   ├── rossi_benchmark.py          # Rossi recidivism dataset benchmark
│   ├── flchain_benchmark.py        # FLChain dataset benchmark
│   ├── whas500_benchmark.py        # WHAS500 cardiac dataset benchmark
│   ├── veterans_benchmark.py       # Veterans lung cancer benchmark
│   ├── breast_cancer_benchmark.py  # Breast cancer dataset benchmark
│   ├── support2_benchmark.py       # SUPPORT2 critical care benchmark
│   ├── cancer_benchmark.py         # General cancer dataset benchmark
│   └── README.md                   # Comprehensive benchmark documentation
└── README.md                      # This file
```


## License

See LICENSE file for details.
