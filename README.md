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

# Create loss function (uses default settings from all experiments)
loss_fn = ConcordancePairwiseLoss(
    temperature=1.0,
    temp_scaling='linear',      # Default: linear scaling
    pairwise_sampling='balanced',  # Default: balanced sampling
    use_ipcw=False
)

# Use in training
loss = loss_fn(log_risks, times, events)
```

### Usage with IPCW (Inverse Probability of Censoring Weights)

**Note**: All experiments use `temp_scaling='linear'` and `pairwise_sampling='balanced'` as defaults.

```python
from src.concordance_pairwise_loss import ConcordancePairwiseLoss

# CPL with dynamic IPCW (computed per batch)
loss_fn_dynamic = ConcordancePairwiseLoss(
    temperature=1.0,
    use_ipcw=True
    # temp_scaling='linear' (default)
    # pairwise_sampling='balanced' (default)
)

# CPL with static IPCW (precomputed from training set)
loss_fn_static = ConcordancePairwiseLoss(
    temperature=1.0,
    use_ipcw=True,
    ipcw_weights=precomputed_weights  # Pass precomputed weights
    # temp_scaling='linear' (default)
    # pairwise_sampling='balanced' (default)
)
```

## Methods

### Loss Function Variants

- **NLL (Negative Partial Log-Likelihood)**: Traditional Cox proportional hazards partial log-likelihood loss
- **CPL (Concordance Pairwise Loss)**: Directly optimizes concordance via pairwise ranking
- **CPL (dynamic)**: CPL with Inverse Probability of Censoring Weights (IPCW) computed dynamically per batch
- **CPL (static)**: CPL with IPCW weights precomputed from the full training set and reused across batches

**Configuration for All Experiments**:
- Temperature scaling: `temp_scaling='linear'`
- Pairwise sampling: `pairwise_sampling='balanced'` (higher weight for event-event pairs)
- Reduction: `reduction='mean'` (weight-normalized)

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

### SurvMNIST (Synthetic Survival Dataset)

```bash
# Run single experiment with default parameters (NLL, batch_size=64, epochs=2)
python benchmarks/benchmark_survmnist.py

# Run comprehensive comparison across all loss types and batch sizes
python benchmarks/benchmark_survmnist.py --compare-all
```

**Key Features**:
- **Synthetic survival data** from MNIST digits with 30% censoring rate
- **Quick benchmarking**: Ideal for rapid prototyping and algorithm validation
- **Multiple loss comparisons**: NLL, CPL(online), CPL(offline)
- **Configurable batch sizes**: 32, 64, 128, 256
- **Comprehensive metrics**: Harrell's C, Uno's C, Cumulative AUC, Incident AUC, Brier Score

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

### Available Datasets
- **Large tabular**: FLChain (7,874), SUPPORT2 (8,873)
- **Medium tabular**: GBSG2 (686), WHAS500 (500), METABRIC (1,903)
- **Synthetic imaging**: SurvMNIST (MNIST-based survival, 60k train / 10k test, 30% censoring)
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
│   ├── survmnist/                          # SurvMNIST synthetic survival module
│   │   ├── __init__.py                     # Module interface
│   │   ├── survival_mnist_dataset.py       # MNIST-based survival dataset
│   │   └── README.md                       # ⭐ SurvMNIST usage guide
│   ├── data_loaders.py                     # Dataset loaders (includes AbstractDataLoader base class)
│   ├── dataset_configs.py                  # Dataset metadata utilities
│   └── flexible_dataset.py                 # Dataset utilities
├── benchmarks/                             # Benchmark scripts (V2 - Current)
│   ├── benchmark_tabular_v2.py             # ⭐ Tabular datasets benchmark
│   ├── benchmark_MIMIC_v2.py               # ⭐ MIMIC-IV imaging benchmark
│   ├── benchmark_survmnist.py              # ⭐ SurvMNIST synthetic benchmark
│   └── dataset_configs.json                # Dataset-specific configurations
├── data/                                   # Data storage
│   ├── MNIST/                              # MNIST raw data (auto-downloaded)
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
- **src/survmnist/README.md**: SurvMNIST synthetic survival benchmark guide
  - Dataset configuration and censoring methodology
  - Loss function variants (NLL, CPL online/offline)
  - Command-line arguments and usage examples
  - Output format specifications

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
