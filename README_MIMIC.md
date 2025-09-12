# MIMIC-IV Chest X-ray Dataset Integration

This document explains how to use the MIMIC-IV Chest X-ray dataset with the ConcordancePairwiseLoss framework for survival analysis with medical images.

## Overview

The MIMIC-IV Chest X-ray integration provides a complete pipeline for survival analysis using chest X-ray images. The preprocessing follows the same methodology as [DiffSurv](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py), ensuring consistency with established medical imaging research practices.

## Quick Start

### 2. Preprocess Your Data

```bash
python preprocess_mimic.py
```

**Note**: Update the `data_dir` path in `preprocess_mimic.py` to point to your MIMIC-IV data location.

### 3. Use in Your Code

```python
from src.data_loaders import MIMICDataLoader

# Initialize the data loader
loader = MIMICDataLoader(
    batch_size=128,
    data_dir="Z:/mimic-cxr-jpg-2.1.0.physionet.org/",
    csv_path="data/mimic/mimic_cxr_splits.csv"
)

# Load data
train_loader, val_loader, test_loader, num_features = loader.load_data()
```

## Data Structure

### Required Directory Structure
```
Z:/mimic-cxr-jpg-2.1.0.physionet.org/
├── mimic-cxr-2.0.0-split.csv.gz
├── mimic-cxr-2.0.0-metadata.csv.gz
├── files/
│   └── p00/p00001/s00001/00000001.jpg
├── admissions.csv.gz
└── patients.csv.gz
```

### Output CSV Structure
The preprocessing creates a CSV with the following columns:
- `subject_id`: Patient identifier
- `study_id`: Study identifier  
- `path`: Relative path to image file
- `exists`: Boolean indicating if image file exists
- `split`: Train/validation/test split
- `tte`: Time to event in days
- `event`: Event indicator (1=death, 0=censored)

## Preprocessing Methodology

Our preprocessing follows the same approach as [DiffSurv](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py):

1. **Data Merging**: Combines patient demographics, admission records, and CXR metadata
2. **Survival Time Calculation**: 
   - For deceased patients: Time from CXR to death
   - For censored patients: Time from CXR to discharge + 1 year
3. **Patient-Level Splitting**: Ensures no data leakage by splitting at patient level
4. **Image Path Generation**: Creates standardized paths for image loading

## Configuration Options

### MIMICDataLoader Parameters
```python
MIMICDataLoader(
    batch_size=128,                    # Batch size for data loaders
    data_dir="path/to/mimic/data",     # Base directory containing MIMIC data
    csv_path="data/mimic/splits.csv",  # Path to preprocessed CSV file
    target_size=(224, 224),            # Target image size for EfficientNet-B0
    use_augmentation=True              # Whether to use data augmentation for training
)
```

## Integration with Existing Framework

The MIMIC data loader is automatically registered in the `DATA_LOADERS` dictionary:

```python
from src.data_loaders import DATA_LOADERS

# Use in benchmark framework
loader_class = DATA_LOADERS['mimic']
loader = loader_class(batch_size=128)
```

## Data Format

The data loader returns:
- **Images**: `torch.Tensor` of shape `(batch_size, 3, height, width)`
- **Events**: `torch.Tensor` of shape `(batch_size,)` - boolean event indicators
- **Times**: `torch.Tensor` of shape `(batch_size,)` - float time to event values

### Performance Tips

1. Use SSD storage for faster image loading
2. Increase `num_workers` for faster data loading
3. Use `pin_memory=True` for GPU training
4. Consider using smaller batch sizes if memory is limited

## References

- **DiffSurv Preprocessing**: [andre-vauvelle/diffsurv](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py)
- **MIMIC-IV Dataset**: [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- **EfficientNet**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Files

- `preprocess_mimic.py`: Preprocessing script
- `src/data_loaders.py`: MIMICDataLoader class
- `src/image_dataset.py`: Image dataset and transform utilities
- `README_MIMIC.md`: This documentation file
