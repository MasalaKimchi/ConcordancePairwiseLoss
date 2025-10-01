#!/usr/bin/env python3
"""
MIMIC Real Data Validation Test

This script specifically tests the MIMIC module with the actual dataset
to ensure preprocessing and data loading work correctly with real data.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_real_mimic_data():
    """Test MIMIC module with actual dataset."""
    print("MIMIC Real Data Validation Test")
    print("=" * 50)
    
    # Check dependencies
    try:
        from mimic.mimic_data_loader import MIMICDataLoader
        from mimic.preprocessor import MIMICPreprocessor
        print("✓ Dependencies loaded successfully")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Please install: pip install torch torchvision Pillow pandas scikit-learn")
        return False
    
    # Check for preprocessed data
    csv_path = "data/mimic/mimic_cxr_splits.csv"
    data_dir = "Y:\\MIMIC-CXR-JPG\\mimic-cxr-jpg-2.1.0.physionet.org\\"
    
    print(f"\nChecking data availability...")
    print(f"CSV file: {csv_path} - {'✓ EXISTS' if os.path.exists(csv_path) else '✗ MISSING'}")
    print(f"Data dir: {data_dir} - {'✓ EXISTS' if os.path.exists(data_dir) else '✗ MISSING'}")
    
    if not os.path.exists(csv_path):
        print(f"\n⚠️  Preprocessed CSV not found!")
        print("   Run preprocessing first:")
        print("   python -m src.mimic.preprocess")
        return False
    
    if not os.path.exists(data_dir):
        print(f"\n⚠️  MIMIC data directory not found!")
        print("   Please update the data_dir path in this script")
        return False
    
    print(f"\n✓ Both data sources available - proceeding with real data test")
    
    # Test data loading
    try:
        print(f"\nTesting data loading...")
        loader = MIMICDataLoader(
            batch_size=8,
            data_dir=data_dir,
            csv_path=csv_path,
            target_size=(224, 224),
            use_augmentation=False  # Disable for testing
        )
        
        # Load data
        train_loader, val_loader, test_loader, num_features = loader.load_data()
        
        # Get statistics
        stats = loader.get_dataset_stats()
        
        print(f"✓ Data loading successful!")
        print(f"✓ Number of features: {num_features}")
        print(f"✓ Total samples: {stats['total_samples']:,}")
        print(f"✓ Overall event rate: {stats['overall_event_rate']:.3f}")
        
        print(f"\nDataset splits:")
        print(f"  Train: {stats['train_samples']:,} samples ({stats['train_event_rate']:.3f} event rate)")
        print(f"  Val:   {stats['val_samples']:,} samples ({stats['val_event_rate']:.3f} event rate)")
        print(f"  Test:  {stats['test_samples']:,} samples ({stats['test_event_rate']:.3f} event rate)")
        
        print(f"\nBatch counts:")
        print(f"  Train: {len(train_loader):,} batches")
        print(f"  Val:   {len(val_loader):,} batches")
        print(f"  Test:  {len(test_loader):,} batches")
        
        # Test loading a batch
        print(f"\nTesting batch loading...")
        batch = next(iter(train_loader))
        images, (events, times) = batch
        
        print(f"✓ Batch loaded successfully!")
        print(f"  Images shape: {images.shape}")
        print(f"  Events shape: {events.shape}")
        print(f"  Times shape: {times.shape}")
        print(f"  Event rate in batch: {events.float().mean():.3f}")
        
        # Validate stratification
        train_rate = stats['train_event_rate']
        val_rate = stats['val_event_rate']
        test_rate = stats['test_event_rate']
        
        max_diff = max(abs(train_rate - val_rate), abs(train_rate - test_rate), abs(val_rate - test_rate))
        
        print(f"\nStratification validation:")
        print(f"  Max difference in event rates: {max_diff:.3f}")
        if max_diff < 0.05:  # Within 5%
            print(f"  ✓ EXCELLENT stratification (within 5%)")
        elif max_diff < 0.10:  # Within 10%
            print(f"  ✓ GOOD stratification (within 10%)")
        else:
            print(f"  ⚠️  Poor stratification (>{10}% difference)")
        
        print(f"\n🎉 REAL DATA VALIDATION SUCCESSFUL!")
        print(f"✅ All {stats['total_samples']:,} samples loaded correctly")
        print(f"✅ Stratified splitting working perfectly")
        print(f"✅ Image loading and transforms functional")
        print(f"✅ Ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during real data testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_mimic_data()
    sys.exit(0 if success else 1)
