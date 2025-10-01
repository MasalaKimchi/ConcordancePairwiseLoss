#!/usr/bin/env python3
"""
MIMIC-IV Chest X-ray Image Preprocessing Script

This script preprocesses MIMIC-IV chest X-ray images by:
1. Converting grayscale images to RGB format
2. Resizing images to 224x224 for EfficientNet compatibility
3. Saving processed images to a new directory
4. Creating a new CSV file with updated paths

Usage:
    python src/mimic/preprocess_images.py --input-csv data/mimic/mimic_cxr_splits.csv --output-dir "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr" --batch-size 1000
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import time
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MIMICImagePreprocessor:
    """Preprocessor for MIMIC-IV chest X-ray images."""
    
    def __init__(
        self,
        input_csv_path: str,
        input_data_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org",
        output_dir: str = "Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr",
        target_size: tuple = (224, 224),
        quality: int = 100,
        num_workers: int = 8
    ):
        """
        Initialize the preprocessor.
        
        Args:
            input_csv_path: Path to original mimic_cxr_splits.csv
            input_data_dir: Directory containing original MIMIC images
            output_dir: Directory to save preprocessed images
            target_size: Target size for resized images (height, width)
            quality: JPEG quality for saved images (1-100)
            num_workers: Number of parallel workers for processing
        """
        self.input_csv_path = input_csv_path
        self.input_data_dir = input_data_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.quality = quality
        self.num_workers = num_workers
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"MIMIC Image Preprocessor Initialized:")
        print(f"  Input CSV: {self.input_csv_path}")
        print(f"  Input data directory: {self.input_data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Target size: {self.target_size}")
        print(f"  JPEG quality: {self.quality}")
        print(f"  Workers: {self.num_workers}")
    
    def process_single_image(self, args):
        """
        Process a single image.
        
        Args:
            args: Tuple of (input_path, output_path, target_size, quality)
            
        Returns:
            Tuple of (success, input_path, output_path, error_message)
        """
        input_path, output_path, target_size, quality = args
        
        try:
            # Create output directory for this image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load and process image
            with Image.open(input_path) as img:
                # Convert to RGB (handles grayscale automatically)
                img_rgb = img.convert('RGB')
                
                # Resize with high-quality resampling
                img_resized = img_rgb.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save with specified quality
                img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
                
            return (True, input_path, output_path, None)
            
        except Exception as e:
            return (False, input_path, output_path, str(e))
    
    def preprocess_images(self, batch_size: int = 1000, limit: int = None):
        """
        Preprocess all images in the CSV file.
        
        Args:
            batch_size: Number of images to process in each batch
            limit: Maximum number of images to process (None for all)
            
        Returns:
            Path to the new CSV file with preprocessed image paths
        """
        print("=" * 60)
        print("STARTING MIMIC IMAGE PREPROCESSING")
        print("=" * 60)
        
        # Load CSV file
        print("Loading CSV file...")
        df = pd.read_csv(self.input_csv_path)
        
        # Filter only existing images
        df = df[df['exists'] == True].copy()
        
        if limit:
            print(f"Limiting to first {limit} images for testing...")
            df = df.head(limit)
        
        print(f"Total images to process: {len(df):,}")
        
        # Prepare processing arguments
        processing_args = []
        new_paths = []
        
        for idx, row in df.iterrows():
            # Original path
            original_path = os.path.join(self.input_data_dir, row['path'])
            
            # New path (maintain directory structure)
            relative_path = row['path']  # e.g., "files/p17/p17242689/s50893862/image.jpg"
            new_relative_path = f"preprocessed_{relative_path}"  # e.g., "preprocessed_files/p17/p17242689/s50893862/image.jpg"
            new_full_path = os.path.join(self.output_dir, new_relative_path)
            
            processing_args.append((original_path, new_full_path, self.target_size, self.quality))
            new_paths.append(new_relative_path)
        
        # Add new paths to dataframe
        df['preprocessed_path'] = new_paths
        
        # Process images in batches
        successful_indices = []
        failed_count = 0
        
        print(f"Processing {len(processing_args):,} images with {self.num_workers} workers...")
        
        # Process in batches to manage memory
        for batch_start in range(0, len(processing_args), batch_size):
            batch_end = min(batch_start + batch_size, len(processing_args))
            batch_args = processing_args[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(processing_args)-1)//batch_size + 1}")
            print(f"Images {batch_start+1:,} to {batch_end:,}")
            
            # Process batch with multiprocessing
            with mp.Pool(self.num_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(self.process_single_image, batch_args),
                    total=len(batch_args),
                    desc="Processing images"
                ))
            
            # Collect results
            for i, (success, input_path, output_path, error) in enumerate(batch_results):
                original_idx = batch_indices[i]
                if success:
                    successful_indices.append(original_idx)
                else:
                    failed_count += 1
                    print(f"Failed to process {input_path}: {error}")
        
        # Filter dataframe to only successful images
        df_successful = df.iloc[successful_indices].copy()
        df_successful['preprocessed_exists'] = True
        
        print(f"\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Successfully processed: {len(successful_indices):,} images")
        print(f"Failed: {failed_count:,} images")
        print(f"Success rate: {len(successful_indices)/(len(successful_indices)+failed_count)*100:.1f}%")
        
        # Save new CSV file
        output_csv_path = self.input_csv_path.replace('.csv', '_preprocessed.csv')
        df_successful.to_csv(output_csv_path, index=False)
        
        print(f"New CSV saved to: {output_csv_path}")
        print(f"Preprocessed images saved to: {self.output_dir}")
        
        return output_csv_path
    
    def verify_preprocessing(self, csv_path: str, sample_size: int = 100):
        """
        Verify that preprocessing was successful by checking a sample of images.
        
        Args:
            csv_path: Path to the preprocessed CSV file
            sample_size: Number of images to verify
        """
        print(f"\n" + "=" * 60)
        print("VERIFYING PREPROCESSING")
        print("=" * 60)
        
        df = pd.read_csv(csv_path)
        
        # Sample images to verify
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        verification_results = []
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Verifying images"):
            preprocessed_path = os.path.join(self.output_dir, row['preprocessed_path'])
            
            try:
                with Image.open(preprocessed_path) as img:
                    # Check if image is RGB and correct size
                    is_rgb = img.mode == 'RGB'
                    is_correct_size = img.size == self.target_size
                    verification_results.append((True, is_rgb, is_correct_size, None))
            except Exception as e:
                verification_results.append((False, False, False, str(e)))
        
        # Summary
        total_checked = len(verification_results)
        successful_loads = sum(1 for r in verification_results if r[0])
        correct_rgb = sum(1 for r in verification_results if r[1])
        correct_size = sum(1 for r in verification_results if r[2])
        
        print(f"Verification Results (sample of {total_checked} images):")
        print(f"  Successfully loaded: {successful_loads}/{total_checked} ({successful_loads/total_checked*100:.1f}%)")
        print(f"  Correct RGB format: {correct_rgb}/{total_checked} ({correct_rgb/total_checked*100:.1f}%)")
        print(f"  Correct size {self.target_size}: {correct_size}/{total_checked} ({correct_size/total_checked*100:.1f}%)")
        
        if successful_loads == total_checked and correct_rgb == total_checked and correct_size == total_checked:
            print("✅ All verification checks passed!")
        else:
            print("❌ Some verification checks failed. Please review the preprocessing.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV chest X-ray images")
    parser.add_argument('--input-csv', type=str, default='data/mimic/mimic_cxr_splits.csv',
                       help='Path to input CSV file')
    parser.add_argument('--input-data-dir', type=str, 
                       default='Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org',
                       help='Directory containing original MIMIC images')
    parser.add_argument('--output-dir', type=str,
                       default='Y:/MIMIC-CXR-JPG/mimic-cxr-jpg-2.1.0.physionet.org/preprocessed_mimic_cxr',
                       help='Directory to save preprocessed images')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                       help='Target size for images (height width)')
    parser.add_argument('--quality', type=int, default=100,
                       help='JPEG quality (1-100, default=100 for maximum quality)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of images to process in each batch')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of images for testing (default: process all)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify preprocessing results')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only run verification (skip preprocessing)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MIMICImagePreprocessor(
        input_csv_path=args.input_csv,
        input_data_dir=args.input_data_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        quality=args.quality,
        num_workers=args.num_workers
    )
    
    if args.verify_only:
        # Only run verification
        preprocessed_csv = args.input_csv.replace('.csv', '_preprocessed.csv')
        if os.path.exists(preprocessed_csv):
            preprocessor.verify_preprocessing(preprocessed_csv)
        else:
            print(f"Preprocessed CSV not found: {preprocessed_csv}")
    else:
        # Run preprocessing
        start_time = time.time()
        output_csv = preprocessor.preprocess_images(
            batch_size=args.batch_size,
            limit=args.limit
        )
        end_time = time.time()
        
        print(f"\nTotal preprocessing time: {end_time - start_time:.1f} seconds")
        
        # Run verification if requested
        if args.verify:
            preprocessor.verify_preprocessing(output_csv)


if __name__ == "__main__":
    main()
