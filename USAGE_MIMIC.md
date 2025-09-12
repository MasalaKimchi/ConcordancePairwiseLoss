# MIMIC-IV Usage Guide

## Quick Start

### 1. Setup Environment
```bash
conda activate concordance-pairwise-loss
```

### 2. Preprocess Data
```bash
python preprocess_mimic.py
```

### 3. Run Benchmark
```bash
python benchmarks/benchmark_framework_improved.py --dataset mimic --loss-type nll --epochs 10
```

## Configuration

### Data Directory
Update the path in `preprocess_mimic.py`:
```python
data_dir = "Z:/mimic-cxr-jpg-2.1.0.physionet.org/"
```

### Batch Size
Adjust in your code:
```python
loader = MIMICDataLoader(batch_size=64)  # Reduce if memory limited
```

For detailed documentation, see [README_MIMIC.md](README_MIMIC.md).
