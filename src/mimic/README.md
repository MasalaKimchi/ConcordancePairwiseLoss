# MIMIC-IV Chest X-ray Dataset Integration

This document provides a comprehensive guide for using the MIMIC-IV Chest X-ray dataset with the ConcordancePairwiseLoss framework for survival analysis with medical images.

## Overview

The MIMIC-IV Chest X-ray integration provides a streamlined workflow for survival analysis using chest X-ray images with **comprehensive evaluation metrics**:

## Complete Workflow

### Three-Step Process

```
STEP 1: Create CSV with Survival Data (One-time, ~30 min)
   Command: python -m src.mimic.preprocess
   Output: data/mimic/mimic_cxr_splits.csv (~300k records)

STEP 2: Preprocess Images (One-time, ~8-20 hours)
   Command: python src/mimic/preprocess_images.py --batch-size 2000 --num-workers 12
   Process: Convert grayscale → RGB 224×224 (~300k images)
   Output: data/mimic/mimic_cxr_splits_preprocessed.csv + preprocessed_files/

STEP 3: Train + Evaluate (Run as needed)
   Command: python benchmarks/benchmark_MIMIC_v2.py --epochs 50 --batch-size 64 --num-runs 3
   Process: Train 3 losses [NLL, CPL (dynamic), CPL (static)] + Comprehensive evaluation
   Output: results/models/ + results/comprehensive_evaluation/

NOTE: Steps 1 & 2 are done ONCE. Step 3 can be run many times with
      different --data-fraction values (0.01 for testing, 1.0 for final)
```

### Workflow Diagram

```mermaid
graph LR
    A[Raw MIMIC Data] --> B[Step 1: Create CSV]
    B --> C[Step 2: Preprocess Images]
    C --> D[Step 3: Train + Evaluate]
    D --> E[Results + Models]
```

## Quick Start Guide

### Step 1: Initial Data Preprocessing (One-time, ~30 min)

```bash
conda activate concordance-pairwise-loss
python -m src.mimic.preprocess
```

**Output**: `data/mimic/mimic_cxr_splits.csv` (~300k records)

### Step 2: Image Preprocessing (One-time, ~8-20 hours)

```bash
python src/mimic/preprocess_images.py --batch-size 128 --num-workers 12 --verify
```

**Note**: Processes ALL ~300k images. Requires ~250GB storage.

### Step 3: Training + Evaluation

```bash
# Quick test with 1% of data
python benchmarks/benchmark_MIMIC_v2.py --epochs 5 --batch-size 32 --data-fraction 0.01 --num-runs 2

# Production run with full dataset
python benchmarks/benchmark_MIMIC_v2.py --epochs 50 --batch-size 64
```

**Note**: `CPL (dynamic)` = `CPL (ipcw)`, `CPL (static)` = `CPL (ipcw batch)` in code

---

## Understanding Data Fraction vs Preprocessing

- **Preprocessing**: Processes ALL ~300k images once (one-time conversion to RGB 224×224)
- **Training**: Use `--data-fraction` to experiment with data subsets (0.01 = 1%, 1.0 = 100%)

## Setup Requirements

**Prerequisites**: MIMIC-IV access from PhysioNet and required packages installed

### Data Directory Structure

```
Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/
├── mimic-cxr-2.0.0-split.csv.gz
├── mimic-cxr-2.0.0-metadata.csv.gz
├── files/
│   ├── p10/p10000032/s50414267/02aa8044-*.jpg
│   ├── p11/p11000033/s50414268/03bb9055-*.jpg
│   └── ...
├── admissions.csv.gz
├── patients.csv.gz
└── preprocessed_mimic_cxr/           # Created by preprocessing
    └── preprocessed_files/
        ├── p10/p10000032/s50414267/02aa8044-*.jpg
        └── ...
```

## Image Preprocessing Details

Converts grayscale X-rays to RGB 224×224 images with maximum quality (JPEG quality=100) for EfficientNet compatibility.

**Storage**: ~250GB for preprocessed images (original: ~500GB)

**Options**:
```bash
# Basic preprocessing
python src/mimic/preprocess_images.py

# With custom settings
python src/mimic/preprocess_images.py --batch-size 128 --num-workers 12 --verify
```

## Module Structure

### Source Files
```
src/mimic/
├── __init__.py                      # Main module interface
├── preprocessor.py                  # Initial data preprocessing (Step 1)
├── preprocess_images.py             # Image preprocessing script (Step 2)
├── dataset.py                       # MIMICImageDataset class
├── transforms.py                    # Image transform utilities
├── mimic_data_loader.py             # Standard data loader (on-the-fly transforms)
├── preprocessed_data_loader.py      # Preprocessed data loader (fast loading)
├── model_loader.py                  # Model loading utilities
├── util.py                          # Benchmark utilities and trainers
└── README.md                        # This documentation
```

### Benchmark Scripts
```
benchmarks/
├── benchmark_MIMIC_v2.py            # RECOMMENDED: Training + Comprehensive Evaluation
├── benchmark_MIMIC_preprocessed.py  # Training + Basic Evaluation (legacy)
└── benchmark_MIMIC.py               # On-the-fly transforms (legacy, slow)
```

**Key Files**:
- **benchmark_MIMIC_v2.py**: Use this for all new experiments (recommended)
- **preprocess_images.py**: Run once to prepare images
- **preprocessed_data_loader.py**: Fast MONAI-optimized data loading

## Benchmark Configuration

### Command-Line Arguments

The `benchmark_MIMIC_v2.py` script accepts the following arguments:

```bash
python benchmarks/benchmark_MIMIC_v2.py \
    --data-dir <path>              # Default: Y:/MIMIC-CXR-JPG/.../preprocessed_mimic_cxr
    --csv-path <path>              # Default: data/mimic/mimic_cxr_splits_preprocessed.csv
    --data-fraction <float>        # Fraction of data to use (0.01 to 1.0)
    --epochs <int>                 # Maximum training epochs (default: 25)
    --lr <float>                   # Learning rate (default: 1e-4)
    --weight-decay <float>         # Weight decay (default: 1e-5)
    --patience <int>               # Early stopping patience (default: 10)
    --max-steps <int>              # Maximum training steps (default: 100000)
    --batch-size <int>             # Batch size (default: 64)
    --num-workers <int>            # Data loading workers (default: 12)
    --enable-augmentation          # Enable data augmentation
    --output-dir <path>            # Output directory (default: results)
    --seed <int>                   # Random seed (default: 42)
    --num-runs <int>               # Independent runs per loss (default: 1)
```

## Output Files

After running `benchmark_MIMIC_v2.py`:

**Model Checkpoints** (`results/models/`):
- `nll_best/`, `cpl_ipcw_best/` (CPL dynamic), `cpl_ipcw_batch_best/` (CPL static)
- Each contains best model weights and summary metrics

**Evaluation Results** (`results/comprehensive_evaluation/`):
- `benchmark_v2_summary_*.csv`: Mean ± Std across runs (for reporting)
- `benchmark_v2_per_run_*.csv`: Individual run metrics (for statistical analysis)
- `benchmark_v2_detailed_*.json`: Complete training history (for debugging)

## References

- **MIMIC-IV Dataset**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **MIMIC-CXR**: [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- **MONAI Framework**: [Medical Open Network for AI](https://monai.io/)
