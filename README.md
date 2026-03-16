# HOW TO OPTIMIZE CONCORDANCE IN MEDICAL IMAGING FOR DEEP SURVIVAL ANALYSIS? CONCORDANCE PAIRWISE LOGISTIC LOSS

## Abstract (Accepted for Oral Presentation ISBI 2026)

Deep survival analysis in medical imaging traditionally relies on Cox proportional hazards with negative partial log-likelihood (NLL), which optimizes likelihood rather than di-rectly targeting concordance, the fundamental ranking metric for risk prediction. This disconnect is amplified in medical imag-ing, where high-resolution images force memory-constrained small batches, destabilizing NLL’s risk-set-based gradients. Here, we introduce Concordance Pairwise Logistic Loss (CPL), a novel objective that directly optimizes pairwise ranking con-cordance by penalizing incorrect risk orderings between compa-rable sample pairs. CPL integrates inverse probability of cen-soring weights (IPCW) through dynamic (batch-computed) and static (precomputed) strategies to handle informative censor-ing. We evaluated CPL and NLL on the largest chest X-ray da-taset MIMIC-CXR. Results show that CPL with dynamic IPCW consistently outperformed NLL in discrimination, especially in memory-constrained small-batch scenarios (batch size=32), achieving Harrell's C-index of 0.737 and Uno's C-index of 0.712 compared to NLL (Harrell's C=0.730, Uno's C=0.705). Importantly, CPL demonstrates improved stability across batch sizes, with its performance varying 2.4 times less than NLL (average CV 0.185% vs. 0.453%). By directly aligning the train-ing objective with rank-based evaluation, CPL offers a more robust and effective solution for developing deep survival models where accurate patient risk ranking is critical.

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

### Basic Usage of CPL with IPCW 

**Note**: All experiments use `temp_scaling='linear'` and `pairwise_sampling='balanced'` as defaults.

```python
from src.concordance_pairwise_loss import ConcordancePairwiseLoss

# CPL (dynamic): CPL with IPCW computed dynamically per batch
loss_fn_dynamic = ConcordancePairwiseLoss(
    temperature=1.0,
    use_ipcw=True
    # temp_scaling='linear' (default)
    # pairwise_sampling='balanced' (default)
)

# CPL (static): CPL with IPCW weights precomputed from the full training set and reused across batches
loss_fn_static = ConcordancePairwiseLoss(
    temperature=1.0,
    use_ipcw=True,
    ipcw_weights=precomputed_weights  # Pass precomputed weights
    # temp_scaling='linear' (default)
    # pairwise_sampling='balanced' (default)
)
```

### Terminology Mapping

For clarity in documentation and code:

| **Documentation Name** | **Code Name** | **Description** |
|------------------------|---------------|-----------------|
| NLL | `nll` | Cox proportional hazards loss |
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


### Medical Imaging (MIMIC-IV)

```bash
# Production run with full preprocessed dataset
python benchmarks/benchmark_MIMIC_v2.py \
    --epochs 50 \
    --batch-size 64 \
    --num-runs 1 \
    --output-dir results
```

**Prerequisites**: MIMIC-IV requires data preprocessing before training. See the complete workflow:
- 📋 **Step 1**: Create CSV with survival data: `python -m src.mimic.preprocess`
- 🖼️ **Step 2**: Preprocess images to RGB 224×224: `python src/mimic/preprocess_images.py`
- 🎯 **Step 3**: Run benchmark (commands above)

**For detailed setup and usage**, see [src/mimic/README.md](src/mimic/README.md)

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

## References and Acknowledgements

Our MIMIC-IV preprocessing methodology follows the established approach from:
- **DiffSurv**: [Andre Vauvelle's DiffSurv preprocessing](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py)
- **MIMIC-IV**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **MIMIC-CXR-JPG**: [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- **Tabular Datasets**: FLChain, SUPPORT2, GBSG2, METABRIC
- **PyTorch**: Deep learning framework
- **TorchSurv**: [Survival analysis metrics and losses](https://github.com/Novartis/torchsurv)
- **MONAI**: [Medical Open Network for AI](https://github.com/Project-MONAI/MONAI)
- **lifelines**: Survival analysis in Python
- **scikit-survival**: Machine learning for survival analysis

## License

See LICENSE file for details.
