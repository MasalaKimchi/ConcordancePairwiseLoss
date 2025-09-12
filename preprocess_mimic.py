import os
import warnings
import random
import multiprocessing as mp
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_mimic_data(
    data_dir: str = "Z:/mimic-cxr-jpg-2.1.0.physionet.org/",
    output_dir: str = "data/mimic/",
    train_split: float = 0.8,
    val_split: float = 0.1,
    random_seed: int = 42
) -> str:
    """
    Preprocess MIMIC-IV Chest X-ray dataset for survival analysis.
    
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
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    cxr_split_path = "physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz"
    admission_path = "physionet.org/files/mimiciv/2.0/hosp/admissions.csv.gz"
    patients_path = "physionet.org/files/mimiciv/2.0/hosp/patients.csv.gz"
    metadata_path = "physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
    
    print("Loading MIMIC-IV data files...")
    
    # Load data files
    splits = pd.read_csv(os.path.join(data_dir, cxr_split_path))
    cxr_metadata = pd.read_csv(os.path.join(data_dir, metadata_path))
    admissions = pd.read_csv(os.path.join(data_dir, admission_path))
    patients = pd.read_csv(os.path.join(data_dir, patients_path))
    
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
    splits = splits.loc[~missing_mask, :].copy()
    if missing_mask.sum() > 0:
        warnings.warn(f"Warning: removed {missing_mask.sum()} images that do not have a dod or dischtime")
    
    # Calculate time to event for censored patients (discharge + 1 year)
    splits.dischtime = pd.to_datetime(splits.dischtime.astype(str))
    splits.loc[splits.dod.isnull(), "tte"] = (
        splits.loc[splits.dod.isnull(), "dischtime"]
        + np.timedelta64(365, "D")
        - splits.loc[splits.dod.isnull(), "StudyDate"]
    )
    
    # Remove negative time to event cases
    negative_tte_cen = (splits.tte.dt.days < 365) & (splits.event == 0)
    if negative_tte_cen.sum() > 0:
        warnings.warn(f"Warning: removed {negative_tte_cen.sum()} images taken after discharge without death")
    splits = splits[~negative_tte_cen].copy()
    
    negative_tte_death = (splits.tte.dt.days < 0) & (splits.event == 1)
    if negative_tte_death.sum() > 0:
        warnings.warn(f"Warning: removed {negative_tte_death.sum()} images taken after death")
    splits = splits[~negative_tte_death].copy()
    
    # Convert time to days
    splits.tte = splits.tte.dt.days
    
    # Create image paths
    p_folders = "p" + splits.subject_id.astype(str).str[:2]
    p_subfolders = "p" + splits.subject_id.astype(str)
    s_folders = "s" + splits.study_id.astype(str)
    image_name = splits.dicom_id + ".jpg"
    
    splits.loc[:, "path"] = (
        "physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        + p_folders
        + "/"
        + p_subfolders
        + "/"
        + s_folders
        + "/"
        + image_name
    )
    
    # Check if images exist
    print("Checking if images exist...")
    with mp.Pool(mp.cpu_count() - 1) as pool:
        splits.loc[:, "exists"] = pool.map(os.path.exists, data_dir + splits.path)
    
    if splits.exists.sum() != splits.shape[0]:
        warnings.warn(f"Warning: only {splits.exists.sum()} images found out of {splits.shape[0]}")
    
    # Create train/val/test splits by patient
    splits = splits.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    patients = list(splits.subject_id.unique())
    n_patients = len(patients)
    random.shuffle(patients)
    
    n_train = int(n_patients * train_split)
    n_val = int(n_patients * val_split)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    # Assign splits
    splits.loc[splits.subject_id.isin(train_patients), "split"] = "train"
    splits.loc[splits.subject_id.isin(val_patients), "split"] = "val"
    splits.loc[splits.subject_id.isin(test_patients), "split"] = "test"
    
    # Save processed data
    output_file = os.path.join(output_dir, "mimic_cxr_splits.csv")
    splits.loc[:, ["subject_id", "study_id", "path", "exists", "split", "tte", "event"]].to_csv(
        output_file, index=False
    )
    
    print(f"Processed data saved to: {output_file}")
    print(f"Final dataset size: {splits.shape[0]}")
    print(f"Train: {splits[splits.split == 'train'].shape[0]}")
    print(f"Validation: {splits[splits.split == 'val'].shape[0]}")
    print(f"Test: {splits[splits.split == 'test'].shape[0]}")
    print(f"Event rate: {splits.event.mean():.3f}")
    
    return output_file


if __name__ == "__main__":
    # Example usage
    data_dir = "Z:/mimic-cxr-jpg-2.1.0.physionet.org/"
    output_dir = "data/mimic/"
    
    if os.path.exists(data_dir):
        csv_path = preprocess_mimic_data(data_dir, output_dir)
        print(f"Preprocessing complete. CSV saved to: {csv_path}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path to point to your MIMIC-IV data location.")
