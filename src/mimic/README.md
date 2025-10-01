# MIMIC-IV Chest X-ray Dataset Integration

This document provides a comprehensive guide for using the MIMIC-IV Chest X-ray dataset with the ConcordancePairwiseLoss framework for survival analysis with medical images.

## Overview

The MIMIC-IV Chest X-ray integration provides two approaches for survival analysis using chest X-ray images:

1. **Standard Pipeline**: On-the-fly image transforms (slower but less storage)
2. **Preprocessed Pipeline**: Pre-converted RGB images (faster training, more storage)

The preprocessing methodology follows established medical imaging research practices for handling grayscale medical images with pre-trained models.

## Quick Start Guide

### Option 1: Preprocessed Images (Recommended for Training)

#### Step 1: Initial Data Preprocessing
```bash
# First, create the initial CSV file with survival data
conda activate concordance-pairwise-loss
python -m src.mimic.preprocess  # Creates data/mimic/mimic_cxr_splits.csv
```

#### Step 2: Image Preprocessing (One-time Setup)
```bash
# Test with small subset first (recommended for testing the pipeline)
python src/mimic/preprocess_images.py --limit 1000 --verify

# Process ALL qualifying images from CSV (takes several hours, ~300k images)
# This processes every image that exists in the CSV file
python src/mimic/preprocess_images.py --batch-size 2000 --num-workers 12 --verify
```

**Important**: The preprocessing step processes **ALL** qualifying images from the CSV file. The `--limit` parameter is only for testing the pipeline, not for production use.

### Data Processing vs Training Distinction

- **Image Preprocessing**: Processes ALL ~300k images from `mimic_cxr_splits.csv` and creates `mimic_cxr_splits_preprocessed.csv`
- **Training Data Fraction**: Uses `--data-fraction` parameter during training to use a subset (e.g., 1% = ~3k images for testing)
- **Result**: You preprocess once (all images) but can train on different fractions for experimentation

#### Step 3: Run Training with Preprocessed Images
```bash
# Test run with 1% of data
python benchmarks/benchmark_MIMIC_preprocessed.py --epochs 5 --batch-size 32 --data-fraction 0.01

# Full training
python benchmarks/benchmark_MIMIC_preprocessed.py --epochs 50 --batch-size 64
```

### Option 2: Standard Pipeline (On-the-fly Transforms)

```bash
# Run with original pipeline (slower but no preprocessing needed)
python benchmarks/benchmark_MIMIC.py --epochs 5 --batch-size 32 --data-fraction 0.01
```

## Detailed Setup Instructions

### Prerequisites

1. **MIMIC-IV Access**: Obtain access to MIMIC-IV dataset from PhysioNet
2. **Data Directory**: Ensure MIMIC data is accessible at specified path
3. **Environment**: Install required packages

```bash
conda activate concordance-pairwise-loss
pip install pillow tqdm pandas numpy torch torchvision
pip install monai[all]  # Optional but recommended for optimization
```

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

### What the Preprocessing Does

1. **Grayscale → RGB Conversion**: Converts single-channel grayscale chest X-rays to 3-channel RGB
2. **Resizing**: Resizes all images to 224×224 for EfficientNet compatibility
3. **Maximum Quality**: Saves as maximum quality JPEG (quality=100) to preserve all diagnostic information
4. **Path Management**: Maintains directory structure with `preprocessed_` prefix
5. **Verification**: Checks processed images for correctness

### Preprocessing Options

```bash
# Basic preprocessing
python src/mimic/preprocess_images.py

# Advanced options
python src/mimic/preprocess_images.py \
    --input-csv data/mimic/mimic_cxr_splits.csv \
    --output-dir "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr" \
    --target-size 224 224 \
    --quality 100 \
    --batch-size 2000 \
    --num-workers 12 \
    --verify

# Test with limited images
python src/mimic/preprocess_images.py --limit 5000

# Verification only
python src/mimic/preprocess_images.py --verify-only
```

### Storage Requirements

- **Original images**: ~500GB (JPG, various sizes)
- **Preprocessed images**: ~250GB (RGB 224×224, quality=100)
- **CSV files**: <100MB each
- **Temporary processing**: ~50GB during preprocessing

## Module Structure

```
src/mimic/
├── __init__.py                      # Main module interface
├── preprocessor.py                  # Initial data preprocessing
├── preprocess_images.py             # Image preprocessing script
├── dataset.py                       # MIMICImageDataset class
├── transforms.py                    # Image transform utilities
├── mimic_data_loader.py             # Standard data loader (on-the-fly transforms)
├── preprocessed_data_loader.py      # Preprocessed data loader (fast loading)
├── util.py                          # Benchmark utilities and trainers
└── README.md                        # This documentation

benchmarks/
├── benchmark_MIMIC.py               # Standard benchmark (on-the-fly transforms)
└── benchmark_MIMIC_preprocessed.py  # Preprocessed benchmark (fast training)
```

## Configuration Options

### Standard Data Loader
```python
from src.mimic.mimic_data_loader import MIMICDataLoader

loader = MIMICDataLoader(
    batch_size=128,                    # Batch size for data loaders
    data_dir="Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org",
    csv_path="data/mimic/mimic_cxr_splits.csv",
    target_size=(224, 224),            # Target image size for EfficientNet-B0
    use_augmentation=True,             # Whether to use data augmentation
    num_workers=4,                     # Number of worker processes
    pin_memory=True                    # Pin memory for GPU transfer
)
```

### Preprocessed Data Loader
```python
from src.mimic.preprocessed_data_loader import PreprocessedMIMICDataLoader

loader = PreprocessedMIMICDataLoader(
    batch_size=128,                    # Batch size for data loaders
    data_dir="Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
    csv_path="data/mimic/mimic_cxr_splits_preprocessed.csv",
    use_augmentation=True,             # Light augmentation (rotation, flip, color)
    num_workers=8,                     # More workers (images load faster)
    pin_memory=True,
    data_fraction=1.0                  # Use full dataset (or fraction for testing)
)
```

## Benchmark Usage

### Testing with Small Data Fraction

```bash
# Test with 1% of data (fast iteration)
python benchmarks/benchmark_MIMIC_preprocessed.py \
    --epochs 3 \
    --batch-size 16 \
    --data-fraction 0.01 \
    --enable-augmentation

# Test with 10% of data
python benchmarks/benchmark_MIMIC_preprocessed.py \
    --epochs 10 \
    --batch-size 32 \
    --data-fraction 0.1
```

### Full Training

```bash
# Full training with preprocessed images
python benchmarks/benchmark_MIMIC_preprocessed.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --patience 20 \
    --enable-augmentation \
    --num-runs 3

# Multiple runs for statistical significance
python benchmarks/benchmark_MIMIC_preprocessed.py \
    --epochs 30 \
    --batch-size 128 \
    --num-runs 5
```

## Performance Optimization

### For Preprocessing
- **Use SSD storage** for input and output directories
- **Increase workers**: `--num-workers 16` (up to CPU cores)
- **Increase batch size**: `--batch-size 5000` (more memory usage)
- **Use high-end CPU**: Preprocessing is CPU-intensive

### For Training
- **Use preprocessed images**: 3-5x faster than on-the-fly transforms
- **Increase batch size**: Limited by GPU memory
- **Enable MONAI optimizations**: Automatic with preprocessed loader
- **Use mixed precision**: Enabled automatically on GPU
- **Pin memory**: `pin_memory=True` for GPU training

### Memory Management
```python
# For limited GPU memory
loader = PreprocessedMIMICDataLoader(
    batch_size=32,          # Smaller batch size
    num_workers=4,          # Fewer workers
    data_fraction=0.1       # Use subset of data
)

# For high-end GPU
loader = PreprocessedMIMICDataLoader(
    batch_size=128,         # Larger batch size
    num_workers=12,         # More workers
    cache_rate=0.2          # Cache more data in memory
)
```

## Troubleshooting

### Common Issues

1. **"Preprocessed CSV not found"**
   ```bash
   # Run image preprocessing first
   python src/mimic/preprocess_images.py
   ```

2. **"Can't pickle local object"**
   - This was fixed in the current version
   - Ensure you're using the latest code with proper functions (not lambdas)

3. **"Out of memory during preprocessing"**
   ```bash
   # Reduce batch size and workers
   python src/mimic/preprocess_images.py --batch-size 500 --num-workers 4
   ```

4. **Slow training with standard loader**
   - Use preprocessed images for faster training
   - Reduce image transforms
   - Use more workers: `--num-workers 12`

5. **MONAI import errors**
   ```bash
   # Install MONAI for optimizations
   pip install monai[all]
   ```

### Data Validation

```bash
# Check if preprocessing was successful
python src/mimic/preprocess_images.py --verify-only

# Test data loading
python -c "
from src.mimic.preprocessed_data_loader import PreprocessedMIMICDataLoader
loader = PreprocessedMIMICDataLoader(data_fraction=0.001)
train_loader, val_loader, test_loader, num_features = loader.load_data()
print(f'Successfully loaded data with {num_features} channels')
"
```

## Data Format

### Input Images
- **Original**: Grayscale JPEG, various sizes (typically ~2500×3000)
- **Preprocessed**: RGB JPEG, 224×224, quality=95

### Output Format
- **Images**: `torch.Tensor` of shape `(batch_size, 3, 224, 224)`
- **Events**: `torch.Tensor` of shape `(batch_size,)` - boolean event indicators
- **Times**: `torch.Tensor` of shape `(batch_size,)` - float time to event values

### CSV Structure

#### Original CSV (`mimic_cxr_splits.csv`)
```
subject_id,study_id,path,exists,split,tte,event
17242689,50893862,files/p17/p17242689/s50893862/75f3b604-*.jpg,True,train,2152,0
```

#### Preprocessed CSV (`mimic_cxr_splits_preprocessed.csv`)
```
subject_id,study_id,path,exists,split,tte,event,preprocessed_path,preprocessed_exists
17242689,50893862,files/p17/p17242689/s50893862/75f3b604-*.jpg,True,train,2152,0,preprocessed_files/p17/p17242689/s50893862/75f3b604-*.jpg,True
```

## Integration with Existing Framework

```python
# Use in benchmark framework
from src.mimic.preprocessed_data_loader import PreprocessedMIMICDataLoader

# Register with framework
DATA_LOADERS['mimic_preprocessed'] = PreprocessedMIMICDataLoader

# Use in experiments
loader_class = DATA_LOADERS['mimic_preprocessed']
loader = loader_class(batch_size=64, data_fraction=0.1)
```

## References

- **MIMIC-IV Dataset**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **EfficientNet**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **Medical Image Preprocessing**: [Best Practices for Medical Image Analysis](https://www.nature.com/articles/s41591-018-0316-z)
- **MONAI Framework**: [Medical Open Network for AI](https://monai.io/)