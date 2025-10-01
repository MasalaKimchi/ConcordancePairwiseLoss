#!/usr/bin/env python3
"""
Test script for MIMIC image preprocessing pipeline.

This script tests the preprocessing pipeline with a small subset of images
to ensure everything works correctly before processing the full dataset.
"""

import os
import sys

# Add src to path
sys.path.append('src')

def test_preprocessing():
    """Test the preprocessing pipeline with a small subset."""
    print("=" * 60)
    print("TESTING MIMIC PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Check if original CSV exists
    original_csv = "data/mimic/mimic_cxr_splits.csv"
    if not os.path.exists(original_csv):
        print(f"❌ Original CSV not found: {original_csv}")
        print("   Run preprocessing first: python -m src.mimic.preprocess")
        return False
    
    print(f"✅ Original CSV found: {original_csv}")
    
    # Test preprocessing with small subset
    print("\nTesting image preprocessing with 10 images...")
    
    try:
        from mimic.preprocess_images import MIMICImagePreprocessor
        
        preprocessor = MIMICImagePreprocessor(
            input_csv_path=original_csv,
            input_data_dir="Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org",
            output_dir="Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
            num_workers=4
        )
        
        # Process small subset for testing (use limit parameter for testing only)
        output_csv = preprocessor.preprocess_images(limit=10, batch_size=10)
        
        print(f"✅ Preprocessing test completed")
        print(f"   Output CSV: {output_csv}")
        
        # Test verification
        print("\nTesting verification...")
        preprocessor.verify_preprocessing(output_csv, sample_size=10)
        
        # Test data loading
        print("\nTesting preprocessed data loading...")
        from mimic.preprocessed_data_loader import PreprocessedMIMICDataLoader
        
        loader = PreprocessedMIMICDataLoader(
            csv_path=output_csv,
            data_fraction=1.0,  # Use all test images
            batch_size=2
        )
        
        train_loader, val_loader, test_loader, num_features = loader.load_data()
        
        print(f"✅ Data loading test completed")
        print(f"   Number of features (channels): {num_features}")
        
        # Test one batch
        print("\nTesting batch loading...")
        for batch in train_loader:
            if isinstance(batch, dict):
                # MONAI format
                images = batch['image']
                events = batch['event']
                times = batch['time']
            else:
                # Standard format
                images, (events, times) = batch
            
            print(f"✅ Batch loaded successfully")
            print(f"   Image shape: {images.shape}")
            print(f"   Events shape: {events.shape}")
            print(f"   Times shape: {times.shape}")
            break
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("You can now run full preprocessing:")
        print("python src/mimic/preprocess_images.py")
        print("\nOr run training with preprocessed images:")
        print("python benchmarks/benchmark_MIMIC_preprocessed.py --data-fraction 0.01")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)
