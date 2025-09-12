# Documentation Structure

## Overview
The documentation has been reorganized to provide clear, user-focused guidance for both tabular and medical imaging datasets.

## Documentation Files

### Main Documentation
- **`README.md`**: Main project documentation with overview of all features
- **`README_MIMIC.md`**: Comprehensive MIMIC-IV Chest X-ray integration guide
- **`USAGE_MIMIC.md`**: Quick start guide for MIMIC-IV usage

### Code Files
- **`preprocess_mimic.py`**: MIMIC-IV preprocessing script (references DiffSurv approach)
- **`src/data_loaders.py`**: Contains MIMICDataLoader class
- **`src/image_dataset.py`**: Image dataset utilities and transforms
- **`benchmarks/dataset_configs.json`**: Updated with MIMIC configuration

## Key Features

### 1. **User-Focused Organization**
- Clear separation between general project docs and MIMIC-specific docs
- Quick start guides for immediate usage
- Comprehensive technical documentation for advanced users

### 2. **DiffSurv Integration**
- Preprocessing methodology follows [DiffSurv](https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py)
- Proper citation and reference to established medical imaging research
- Consistent with existing medical imaging preprocessing practices

### 3. **Clear Usage Paths**
- **New Users**: Start with `README.md` → `USAGE_MIMIC.md`
- **Advanced Users**: Go directly to `README_MIMIC.md`
- **Developers**: Reference code files and inline documentation

## Quick Reference

### For Tabular Datasets
- Use existing `README.md` and benchmark framework
- All existing functionality remains unchanged

### For Medical Imaging (MIMIC-IV)
1. **Setup**: `conda activate concordance-pairwise-loss`
2. **Preprocess**: `python preprocess_mimic.py`
3. **Run**: `python benchmarks/benchmark_framework_improved.py --dataset mimic`

### Documentation Hierarchy
```
README.md (Main project overview)
├── README_MIMIC.md (Detailed MIMIC guide)
├── USAGE_MIMIC.md (Quick MIMIC start)
└── Code files with inline documentation
```
