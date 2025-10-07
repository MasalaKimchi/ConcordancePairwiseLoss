# ConcordancePairwiseLoss

A pairwise loss function for survival analysis that improves concordance between predicted and actual survival times by comparing pairs of samples.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ConcordancePairwiseLoss.git
cd ConcordancePairwiseLoss

# Create and activate conda environment (recommended)
conda create -n concordance-pairwise-loss python=3.9
conda activate concordance-pairwise-loss

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch>=2.0.0` - PyTorch for deep learning
- `torchsurv>=0.1.0` - Survival analysis metrics and losses
- `monai>=1.3.0` - Medical imaging optimizations (required for MIMIC-IV)
- `lifelines>=0.27.0`, `scikit-survival>=0.22.0` - Survival analysis tools
- See `requirements.txt` for complete list

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
# Production run with full preprocessed dataset
python benchmarks/benchmark_MIMIC_v2.py \
    --epochs 50 \
    --batch-size 64 \
    --num-runs 3 \
    --output-dir results
```

**Prerequisites**: MIMIC-IV requires data preprocessing before training. See the complete workflow:
- 📋 **Step 1**: Create CSV with survival data: `python -m src.mimic.preprocess`
- 🖼️ **Step 2**: Preprocess images to RGB 224×224: `python src/mimic/preprocess_images.py`
- 🎯 **Step 3**: Run benchmark (commands above)

**For detailed setup and usage**, see [src/mimic/README.md](src/mimic/README.md) which includes:
- Complete preprocessing workflow (6-8 hours one-time setup)
- MONAI-optimized data loading configuration
- Performance optimization guidelines
- Comprehensive evaluation metrics explanation
- Troubleshooting and best practices

### Available Datasets
- **Large tabular**: FLChain (7,874), SUPPORT2 (8,873)
- **Medium tabular**: GBSG2 (686), WHAS500 (500), METABRIC (1,903)
- **Medical Imaging**: MIMIC-IV Chest X-ray (~300k images)

### Statistical Analysis
Each benchmark provides comprehensive statistical analysis:
- **Multi-run support**: Mean ± standard deviation across 10 independent runs
- **Comprehensive metrics**: Harrell's C-index, Uno's C-index, Cumulative AUC, Incident AUC, Brier Score
- **Statistical testing**: Pairwise t-tests for significance analysis
- **Rich visualizations**: Multi-panel analysis plots with fixed dimensions
- **Professional output**: JSON + CSV + PNG files with timestamps

## Project Structure

```
ConcordancePairwiseLoss/
├── src/                                    # Source code
│   ├── concordance_pairwise_loss/          # Core loss implementation
│   │   ├── __init__.py
│   │   ├── loss.py                         # ConcordancePairwiseLoss main implementation
│   │   └── dynamic_weighting.py            # Dynamic weighting strategies
│   ├── mimic/                              # MIMIC-IV medical imaging module
│   │   ├── __init__.py                     # Module interface
│   │   ├── preprocessor.py                 # Initial data preprocessing (Step 1)
│   │   ├── preprocess_images.py            # Image preprocessing script (Step 2)
│   │   ├── preprocessed_data_loader.py     # MONAI-optimized data loader
│   │   ├── dataset.py                      # MIMICImageDataset class
│   │   ├── transforms.py                   # Image transform utilities
│   │   ├── model_loader.py                 # Model loading utilities
│   │   ├── util.py                         # Benchmark utilities and trainers
│   │   └── README.md                       # ⭐ MIMIC-IV comprehensive guide
│   ├── abstract_data_loader.py             # Base class for dataset loaders
│   ├── data_loaders.py                     # Tabular dataset implementations
│   ├── dataset_configs.py                  # Dataset metadata utilities
│   └── flexible_dataset.py                 # Dataset utilities
├── benchmarks/                             # Benchmark scripts (V2 - Current)
│   ├── benchmark_tabular_v2.py             # ⭐ Tabular datasets benchmark
│   ├── benchmark_MIMIC_v2.py               # ⭐ MIMIC-IV imaging benchmark
│   └── dataset_configs.json                # Dataset-specific configurations
├── data/                                   # Data storage
│   └── mimic/                              # MIMIC-IV CSV files
├── results/                                # Benchmark results and trained models
│   ├── models/                             # Saved model checkpoints
│   └── comprehensive_evaluation/           # Evaluation metrics and reports
├── requirements.txt                        # Python dependencies
└── README.md                               # ⭐ This file (main documentation)
```

## Documentation Hierarchy

- **README.md** (This file): Main project overview, installation, quick start, benchmarks, and references
- **src/mimic/README.md**: Comprehensive MIMIC-IV chest X-ray integration guide
  - Complete 3-step preprocessing workflow
  - MONAI-optimized data loading configuration
  - Performance optimization tips
  - Comprehensive evaluation metrics
  - Troubleshooting and best practices

## References and Acknowledgements

### Medical Imaging Preprocessing
Our MIMIC-IV preprocessing methodology follows the established approach from:
- **DiffSurv**: [Andre Vauvelle's DiffSurv preprocessing](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py)
- Ensures consistency with established medical imaging research practices

### Datasets
- **MIMIC-IV**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **MIMIC-CXR**: [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/)
- **Tabular Datasets**: FLChain, SUPPORT2, GBSG2, WHAS500, METABRIC

### Frameworks and Libraries
- **PyTorch**: Deep learning framework
- **TorchSurv**: Survival analysis metrics and losses
- **MONAI**: Medical Open Network for AI (optimized medical image processing)
- **lifelines**: Survival analysis in Python
- **scikit-survival**: Machine learning for survival analysis

### Key Publications
- **EfficientNet**: [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- **Medical Image Analysis**: [Best Practices for Medical Imaging](https://www.nature.com/articles/s41591-018-0316-z)
- **Survival Analysis with Neural Networks**: [Time-to-event prediction with neural networks](https://jmlr.org/papers/v20/18-424.html) (Kvamme et al., 2019)

## License

See LICENSE file for details.
