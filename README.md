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

### Individual Dataset Benchmarks
```bash
# GBSG2 dataset
python benchmarks/gbsg2_benchmark.py

# Lung dataset
python benchmarks/lung_benchmark.py

# Rossi dataset
python benchmarks/rossi_benchmark.py
```

### Statistical Analysis
Each benchmark uses different random seeds for proper statistical analysis:
- **Seed variation**: Different seeds (42, 1042, 2042, etc.) for each run
- **Confidence intervals**: 95% CI calculated using percentiles
- **Fixed figure dimensions**: 15x12 inches, 300 DPI for consistent plots
- **Results saved**: JSON files with statistics and individual run data

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

## Key Improvements

### 1. **Code Consolidation**
- **Before**: 3 copies of `NormalizedLossCombination` class (~120 lines total)
- **After**: 1 copy in `src/concordance_pairwise_loss/dynamic_weighting.py`
- **Benefit**: Single source of truth, easier maintenance

### 2. **Proper Statistical Analysis**
- **Before**: Fixed seed (42) → identical results across runs
- **After**: Different seeds per run → proper statistical variation
- **Benefit**: Real confidence intervals and statistical significance

### 3. **Clean Documentation**
- **Before**: 4+ markdown files with redundant information
- **After**: Single, concise README with essential information
- **Benefit**: Easy to read, maintain, and understand

### 4. **GitHub Best Practices**
- **Removed**: Unnecessary directories and files
- **Organized**: Clear separation of concerns
- **Maintained**: Essential functionality and examples
- **Result**: Professional, maintainable project structure

## License

See LICENSE file for details.