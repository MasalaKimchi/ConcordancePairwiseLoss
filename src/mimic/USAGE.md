# MIMIC-IV Usage Guide

## Quick Start

### 1. Setup Environment
```bash
conda activate concordance-pairwise-loss
```

### 2. Preprocess Data
```bash
python -m src.mimic.preprocess
```

### 3. Run Benchmark
```bash
python benchmarks/benchmark_framework_improved.py --dataset mimic --loss-type nll --epochs 10
```

## Configuration

### Data Directory
Update the path in `src/mimic/preprocess.py`:
```python
data_dir = "Y:/mimic-cxr-jpg-2.1.0.physionet.org/"
```

### Batch Size
Adjust in your code:
```python
from src.mimic.mimic_data_loader import MIMICDataLoader
loader = MIMICDataLoader(batch_size=64)  # Reduce if memory limited
```

For detailed documentation, see [README.md](README.md).
