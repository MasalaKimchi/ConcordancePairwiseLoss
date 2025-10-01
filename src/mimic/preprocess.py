#!/usr/bin/env python3
"""
MIMIC-IV Data Preprocessing Script

This script preprocesses MIMIC-IV chest X-ray data for survival analysis using
the organized MIMIC module structure.

Usage:
    python -m src.mimic.preprocess

The script will:
1. Load and merge MIMIC-IV data files
2. Calculate survival times and event indicators
3. Perform stratified patient-level splitting
4. Save the processed data to CSV
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .preprocessor import preprocess_mimic_data


def main():
    """Main preprocessing function."""
    # Configuration
    data_dir = "Y:\\MIMIC-CXR-JPG\\mimic-cxr-jpg-2.1.0.physionet.org\\"
    output_dir = "data/mimic/"
    
    print("MIMIC-IV Chest X-ray Data Preprocessing")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please update the data_dir path to point to your MIMIC-IV data location.")
        return 1
    
    try:
        # Run preprocessing
        csv_path = preprocess_mimic_data(
            data_dir=data_dir,
            output_dir=output_dir,
            train_split=0.8,
            val_split=0.1,
            random_seed=42
        )
        
        print(f"\n✓ Preprocessing completed successfully!")
        print(f"✓ Processed data saved to: {csv_path}")
        
        # Verify the output
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"✓ Final dataset size: {len(df)}")
            print(f"✓ Event rate: {df['event'].mean():.3f}")
            print(f"✓ Split distribution:")
            print(f"  - Train: {len(df[df['split'] == 'train'])}")
            print(f"  - Val: {len(df[df['split'] == 'val'])}")
            print(f"  - Test: {len(df[df['split'] == 'test'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
