"""
MIMIC-IV Data Preprocessing Module

This module handles the preprocessing of MIMIC-IV chest X-ray data for survival analysis,
including data merging, survival time calculation, and stratified patient-level splitting.
"""

import os
import warnings
import random
from typing import Optional, Set, Dict, Tuple
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MIMICPreprocessor:
    """
    MIMIC-IV data preprocessor for survival analysis.
    
    This class handles the complete preprocessing pipeline for MIMIC-IV chest X-ray data,
    following the same methodology as DiffSurv for consistency with established practices.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/mimic/",
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize MIMIC preprocessor.
        
        Args:
            data_dir: Path to the MIMIC-IV data directory
            output_dir: Directory to save the processed CSV file
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_split = train_split
        self.val_split = val_split
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # MIMIC-IV file names
        self.file_paths = {
            'cxr_split': "mimic-cxr-2.0.0-split.csv.gz",
            'admissions': "admissions.csv.gz", 
            'patients': "patients.csv.gz",
            'metadata': "mimic-cxr-2.0.0-metadata.csv.gz"
        }
    
    def _get_available_images_efficiently(self) -> Set[str]:
        """
        Efficiently scan directories to find all available image files.
        This is much faster than checking individual files.
        
        Returns:
            Set of relative image paths that exist
        """
        print("Scanning image directories efficiently...")
        
        # Get all patient directories (p00, p01, etc.)
        files_dir = os.path.join(self.data_dir, "files")
        if not os.path.exists(files_dir):
            print(f"Warning: Files directory not found: {files_dir}")
            return set()
        
        available_images = set()
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(files_dir) 
                       if d.startswith('p') and os.path.isdir(os.path.join(files_dir, d))]
        print(f"Found {len(patient_dirs)} patient directories")
        
        for i, patient_dir in enumerate(patient_dirs):
            if i % 100 == 0:
                print(f"Processing patient directory {i+1}/{len(patient_dirs)}: {patient_dir}")
            
            patient_path = os.path.join(files_dir, patient_dir)
            
            # Get all subdirectories for this patient
            try:
                subdirs = [d for d in os.listdir(patient_path) 
                          if os.path.isdir(os.path.join(patient_path, d))]
                
                for subdir in subdirs:
                    subdir_path = os.path.join(patient_path, subdir)
                    
                    # Get all study directories
                    try:
                        study_dirs = [d for d in os.listdir(subdir_path) 
                                    if os.path.isdir(os.path.join(subdir_path, d))]
                        
                        for study_dir in study_dirs:
                            study_path = os.path.join(subdir_path, study_dir)
                            
                            # Get all jpg files in this study directory
                            try:
                                jpg_files = glob.glob(os.path.join(study_path, "*.jpg"))
                                for jpg_file in jpg_files:
                                    # Convert to relative path format
                                    rel_path = os.path.relpath(jpg_file, self.data_dir).replace("\\", "/")
                                    available_images.add(rel_path)
                            except (OSError, PermissionError):
                                continue
                                
                    except (OSError, PermissionError):
                        continue
                        
            except (OSError, PermissionError):
                continue
        
        print(f"Found {len(available_images)} available image files")
        return available_images
    
    def _load_and_merge_data(self) -> pd.DataFrame:
        """
        Load and merge all MIMIC-IV data files.
        
        Returns:
            Merged DataFrame with all necessary columns
        """
        print("=" * 60)
        print("STAGE 1: Data Loading and Initial Filtering")
        print("=" * 60)
        
        # Load data files
        splits = pd.read_csv(os.path.join(self.data_dir, self.file_paths['cxr_split']))
        cxr_metadata = pd.read_csv(os.path.join(self.data_dir, self.file_paths['metadata']))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.file_paths['admissions']))
        patients = pd.read_csv(os.path.join(self.data_dir, self.file_paths['patients']))
        
        print(f"Total starting images from CXR splits: {splits.shape[0]}")
        
        # Merge with patients data for death information
        splits = splits.merge(patients.loc[:, ["subject_id", "dod"]], on="subject_id", how="left")
        print(f"Total after merging with patients from MIMICIV (for DOD): {splits.shape[0]}")
        
        # Merge with CXR metadata for study date
        splits = splits.merge(
            cxr_metadata.loc[:, ["subject_id", "study_id", "StudyDate", "dicom_id"]],
            on=["subject_id", "study_id", "dicom_id"],
            how="left",
        )
        print(f"Total after merging with CXR Metadata for study date: {splits.shape[0]}")
        
        # Merge with admissions for discharge time (censoring time)
        last_discharge = admissions.groupby("subject_id").dischtime.max().reset_index()
        splits = splits.merge(
            last_discharge.loc[:, ["subject_id", "dischtime"]], on="subject_id", how="left"
        )
        print(f"Total after merging with admissions for dischtime (censoring time): {splits.shape[0]}")
        
        return splits
    
    def _calculate_survival_times(self, splits: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate survival times and event indicators.
        
        Args:
            splits: DataFrame with merged data
            
        Returns:
            DataFrame with survival times and events calculated
        """
        # Convert dates
        splits.StudyDate = pd.to_datetime(splits.StudyDate.astype(str))
        splits.dod = pd.to_datetime(splits.dod)
        
        # Create event indicator (death)
        splits.loc[:, "event"] = splits.dod.notnull().astype(int)
        
        # Calculate time to event for those who died
        splits.loc[splits.dod.notnull(), "tte"] = (
            splits.loc[splits.dod.notnull(), "dod"] - splits.loc[splits.dod.notnull(), "StudyDate"]
        )
        
        # Remove samples without death or discharge time
        missing_mask = splits.tte.isnull() & splits.dischtime.isnull()
        n_removed_missing = missing_mask.sum()
        splits = splits.loc[~missing_mask, :].copy()
        if n_removed_missing > 0:
            print(f"Removed {n_removed_missing} images that do not have a dod or dischtime")
        
        # Calculate time to event for censored patients (discharge + 1 year)
        splits.dischtime = pd.to_datetime(splits.dischtime.astype(str))
        splits.loc[splits.dod.isnull(), "tte"] = (
            splits.loc[splits.dod.isnull(), "dischtime"]
            + np.timedelta64(365, "D")
            - splits.loc[splits.dod.isnull(), "StudyDate"]
        )
        
        # Remove negative time to event cases
        negative_tte_cen = (splits.tte.dt.days < 365) & (splits.event == 0)
        n_removed_negative_cen = negative_tte_cen.sum()
        if n_removed_negative_cen > 0:
            print(f"Removed {n_removed_negative_cen} images taken after discharge without death")
        splits = splits[~negative_tte_cen].copy()
        
        negative_tte_death = (splits.tte.dt.days < 0) & (splits.event == 1)
        n_removed_negative_death = negative_tte_death.sum()
        if n_removed_negative_death > 0:
            print(f"Removed {n_removed_negative_death} images taken after death")
        splits = splits[~negative_tte_death].copy()
        
        # Convert time to days
        splits.tte = splits.tte.dt.days
        
        print(f"After initial filtering: {splits.shape[0]} images remaining")
        
        return splits, n_removed_missing, n_removed_negative_cen, n_removed_negative_death
    
    def _create_image_paths(self, splits: pd.DataFrame) -> pd.DataFrame:
        """
        Create standardized image paths for MIMIC data.
        
        Args:
            splits: DataFrame with survival data
            
        Returns:
            DataFrame with image paths added
        """
        # Create image paths
        p_folders = "p" + splits.subject_id.astype(str).str[:2]
        p_subfolders = "p" + splits.subject_id.astype(str)
        s_folders = "s" + splits.study_id.astype(str)
        image_name = splits.dicom_id + ".jpg"
        
        splits.loc[:, "path"] = (
            "files/"
            + p_folders
            + "/"
            + p_subfolders
            + "/"
            + s_folders
            + "/"
            + image_name
        )
        
        return splits
    
    def _filter_existing_images(self, splits: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Filter out images that don't exist on disk.
        
        Args:
            splits: DataFrame with image paths
            
        Returns:
            Tuple of (filtered DataFrame, number of missing images)
        """
        print("=" * 60)
        print("STAGE 2: Efficient Image Existence Checking")
        print("=" * 60)
        
        # Get all available images efficiently
        available_images = self._get_available_images_efficiently()
        
        # Check which images exist
        print("Checking image existence against available files...")
        splits.loc[:, "exists"] = splits.path.isin(available_images)
        
        n_missing_images = (~splits.exists).sum()
        print(f"Found {n_missing_images} missing images out of {splits.shape[0]} total")
        
        # Filter out missing images
        splits = splits[splits.exists].copy()
        print(f"After removing missing images: {splits.shape[0]} images remaining")
        
        return splits, n_missing_images
    
    def _create_stratified_splits(self, splits: pd.DataFrame) -> pd.DataFrame:
        """
        Create stratified patient-level train/validation/test splits.
        
        Args:
            splits: DataFrame with survival data
            
        Returns:
            DataFrame with split assignments
        """
        print("=" * 60)
        print("STAGE 3: Stratified Patient-Level Train/Val/Test Splitting")
        print("=" * 60)
        
        # Create patient-level summary for stratified splitting
        patient_summary = splits.groupby('subject_id').agg({
            'event': 'max',  # Patient event status (1 if any death, 0 if all censored)
            'tte': 'mean'    # Average survival time for this patient
        }).reset_index()
        
        print(f"Patient-level event rate: {patient_summary['event'].mean():.3f}")
        print(f"Total patients: {len(patient_summary)}")
        
        # Stratified splitting by event rate
        # First split: train vs (val + test)
        train_patients, temp_patients = train_test_split(
            patient_summary['subject_id'].values,
            test_size=(1 - self.train_split),
            random_state=self.random_seed,
            stratify=patient_summary['event'].values
        )
        
        # Second split: val vs test from remaining patients
        val_ratio = self.val_split / (self.val_split + (1 - self.train_split - self.val_split))
        temp_patient_summary = patient_summary[patient_summary['subject_id'].isin(temp_patients)]
        
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=(1 - val_ratio),
            random_state=self.random_seed,
            stratify=temp_patient_summary['event'].values
        )
        
        # Convert to lists for consistency
        train_patients = list(train_patients)
        val_patients = list(val_patients)
        test_patients = list(test_patients)
        
        # Assign splits
        splits.loc[splits.subject_id.isin(train_patients), "split"] = "train"
        splits.loc[splits.subject_id.isin(val_patients), "split"] = "val"
        splits.loc[splits.subject_id.isin(test_patients), "split"] = "test"
        
        print(f"Final patient counts:")
        print(f"  Train patients: {len(train_patients)}")
        print(f"  Val patients: {len(val_patients)}")
        print(f"  Test patients: {len(test_patients)}")
        
        # Verify stratification by checking event rates in each split
        train_event_rate = splits[splits.split == 'train']['event'].mean()
        val_event_rate = splits[splits.split == 'val']['event'].mean()
        test_event_rate = splits[splits.split == 'test']['event'].mean()
        
        print(f"\nEvent rates by split:")
        print(f"  Train: {train_event_rate:.3f}")
        print(f"  Val: {val_event_rate:.3f}")
        print(f"  Test: {test_event_rate:.3f}")
        print(f"  Overall: {splits['event'].mean():.3f}")
        
        return splits
    
    def preprocess(self) -> str:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Path to the saved CSV file
        """
        # Load and merge data
        splits = self._load_and_merge_data()
        
        # Calculate survival times
        splits, n_removed_missing, n_removed_negative_cen, n_removed_negative_death = \
            self._calculate_survival_times(splits)
        
        # Create image paths
        splits = self._create_image_paths(splits)
        
        # Filter existing images
        splits, n_missing_images = self._filter_existing_images(splits)
        
        # Create stratified splits
        splits = self._create_stratified_splits(splits)
        
        # Save processed data
        output_file = os.path.join(self.output_dir, "mimic_cxr_splits.csv")
        splits.loc[:, ["subject_id", "study_id", "path", "exists", "split", "tte", "event"]].to_csv(
            output_file, index=False
        )
        
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Processed data saved to: {output_file}")
        print(f"Final dataset size: {splits.shape[0]}")
        print(f"Train: {splits[splits.split == 'train'].shape[0]}")
        print(f"Validation: {splits[splits.split == 'val'].shape[0]}")
        print(f"Test: {splits[splits.split == 'test'].shape[0]}")
        print(f"Event rate: {splits.event.mean():.3f}")
        
        print(f"\nExclusion Summary:")
        print(f"  Removed due to missing dod/dischtime: {n_removed_missing}")
        print(f"  Removed due to negative TTE (censored): {n_removed_negative_cen}")
        print(f"  Removed due to negative TTE (death): {n_removed_negative_death}")
        print(f"  Removed due to missing images: {n_missing_images}")
        print(f"  Total exclusions: {n_removed_missing + n_removed_negative_cen + n_removed_negative_death + n_missing_images}")
        
        return output_file


def preprocess_mimic_data(
    data_dir: str,
    output_dir: str = "data/mimic/",
    train_split: float = 0.8,
    val_split: float = 0.1,
    random_seed: int = 42
) -> str:
    """
    Convenience function to preprocess MIMIC-IV data.
    
    This preprocessing follows the same methodology as DiffSurv:
    https://github.com/andre-vauvelle/diffsurv/blob/main/src/data/preprocess/preprocess_mimic_cxr.py
    
    Args:
        data_dir: Path to the MIMIC-IV data directory
        output_dir: Directory to save the processed CSV file
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to the saved CSV file
    """
    preprocessor = MIMICPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        random_seed=random_seed
    )
    
    return preprocessor.preprocess()
