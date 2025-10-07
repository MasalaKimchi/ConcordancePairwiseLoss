# Documentation Structure

## Overview
The documentation has been reorganized to provide clear, user-focused guidance for both tabular and medical imaging datasets. All documentation uses consistent terminology for Concordance Pairwise Loss (CPL) variants:
- **CPL**: Base concordance pairwise loss
- **CPL (dynamic)**: CPL with IPCW computed dynamically per batch (code: `cpl_ipcw`)
- **CPL (static)**: CPL with IPCW precomputed from full training set (code: `cpl_ipcw_batch`)

## Documentation Files

### Main Documentation
- **`README.md`**: Main project documentation with overview of all features
- **`src/mimic/README.md`**: Comprehensive MIMIC-IV Chest X-ray integration guide
- **`src/mimic/USAGE.md`**: Quick start guide for MIMIC-IV usage

### Benchmark Scripts (V2 - Current)
- **`benchmarks/benchmark_tabular_v2.py`**: Tabular datasets benchmark with comprehensive metrics
- **`benchmarks/benchmark_MIMIC_v2.py`**: MIMIC-IV imaging benchmark with integrated evaluation
- **`benchmarks/dataset_configs.json`**: Dataset-specific configurations

### Code Files
- **`src/mimic/preprocess.py`**: MIMIC-IV preprocessing script (references DiffSurv approach)
- **`src/mimic/preprocess_images.py`**: Image preprocessing for fast training
- **`src/mimic/preprocessed_data_loader.py`**: Optimized data loader for preprocessed images
- **`src/mimic/mimic_data_loader.py`**: Standard MIMICDataLoader class
- **`src/mimic/dataset.py`**: MIMICImageDataset class
- **`src/mimic/transforms.py`**: Image transform utilities

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
- **New Users**: Start with `README.md` → `src/mimic/USAGE.md`
- **Advanced Users**: Go directly to `src/mimic/README.md`
- **Developers**: Reference code files and inline documentation

## Quick Reference

### For Tabular Datasets
1. **Setup**: `conda activate concordance-pairwise-loss`
2. **Run single dataset**: `python benchmarks/benchmark_tabular_v2.py --dataset gbsg2 --epochs 30 --num-runs 10`
3. **Run all datasets**: `python benchmarks/benchmark_tabular_v2.py --dataset all --epochs 30 --num-runs 10`

**Loss Variants**: NLL, CPL, CPL (dynamic), CPL (static)

### For Medical Imaging (MIMIC-IV)
1. **Setup**: `conda activate concordance-pairwise-loss`
2. **Preprocess CSV**: `python -m src.mimic.preprocess`
3. **Preprocess Images**: `python src/mimic/preprocess_images.py --batch-size 2000 --num-workers 12`
4. **Run Training**: `python benchmarks/benchmark_MIMIC_v2.py --epochs 50 --batch-size 64 --num-runs 3`

**Loss Variants**: NLL, CPL, CPL (dynamic), CPL (static)

### Documentation Hierarchy
```
README.md (Main project overview)
├── src/mimic/README.md (Detailed MIMIC guide)
├── src/mimic/USAGE.md (Quick MIMIC start)
└── Code files with inline documentation
```
