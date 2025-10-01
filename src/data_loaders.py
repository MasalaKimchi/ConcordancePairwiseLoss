import warnings
warnings.filterwarnings("ignore")

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from flexible_dataset import FlexibleDataset, Custom_dataset
from abstract_data_loader import AbstractDataLoader
from mimic.mimic_data_loader import MIMICDataLoader


class GBSG2DataLoader(AbstractDataLoader):
    """GBSG2 dataset loader implementation using pycox."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def _preprocess_columns(self, df: pd.DataFrame, columns_to_scale: list, columns_to_one_hot: list, min_cat_prop: float = 0.01) -> pd.DataFrame:
        """
        Preprocess columns by scaling numerical features and one-hot encoding categorical features.
        Adapted from diffsurv preprocessing approach.
        """
        # Create a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Scale numerical columns
        if columns_to_scale:
            scaler = StandardScaler()
            df_processed[columns_to_scale] = scaler.fit_transform(df_processed[columns_to_scale])
        
        # One-hot encode categorical columns
        if columns_to_one_hot:
            for col in columns_to_one_hot:
                if col in df_processed.columns:
                    # Get unique values and filter by minimum category proportion
                    value_counts = df_processed[col].value_counts()
                    valid_categories = value_counts[value_counts / len(df_processed) >= min_cat_prop].index
                    
                    # Create one-hot encoded columns
                    for category in valid_categories:
                        new_col_name = f"{col}_{category}"
                        df_processed[new_col_name] = (df_processed[col] == category).astype(int)
                    
                    # Drop the original categorical column
                    df_processed = df_processed.drop(columns=[col])
        
        return df_processed

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from pycox.datasets import gbsg

        # Load GBSG2 dataset from pycox
        df = gbsg.read_df()
        
        # Rename duration column to time to match our standard
        if 'duration' in df.columns:
            df = df.rename(columns={'duration': 'time'})
        
        # Convert time from years to days (pycox GBSG2 dataset has time in years)
        # This ensures compatibility with the benchmark framework which expects days
        df['time'] = df['time'] * 365.25
        
        # Define columns for preprocessing based on diffsurv approach
        columns_to_scale = ["x3", "x5", "x4", "x6"]
        columns_to_one_hot = ["x1"]
        
        # Preprocess the dataset
        df_processed = self._preprocess_columns(df, columns_to_scale, columns_to_one_hot)
        
        # Ensure event column is binary (0/1)
        if df_processed['event'].min() < 0 or df_processed['event'].max() > 1:
            uniq = sorted(df_processed['event'].unique())
            if len(uniq) == 2:
                df_processed['event'] = (df_processed['event'] == uniq[1]).astype(int)
            else:
                df_processed['event'] = (df_processed['event'] > 0).astype(int)
        
        # Remove any remaining missing values
        df_processed = df_processed.dropna()
        
        # Split the data
        df_train, df_test = train_test_split(df_processed, test_size=0.15, random_state=42, stratify=df_processed['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.15/0.85, random_state=42, stratify=df_train['event'])
        
        # Create data loaders using FlexibleDataset
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        
        # Get number of features
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


class FLChainDataLoader(AbstractDataLoader):
    """FLChain dataset loader implementation."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from sksurv.datasets import load_flchain

        X, y = load_flchain()
        df = pd.DataFrame(X)
        df['time'] = y['futime']
        df['event'] = y['death'].astype(int)
        
        # Remove problematic columns as per literature recommendations
        # 'chapter' has 72.5% missing values and complicates analysis
        # 'sample.yr' is not relevant for survival prediction
        columns_to_remove = ['chapter', 'sample.yr']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['time', 'event']:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'unknown')
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        if feature_cols:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.15/0.85, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class WHAS500DataLoader(AbstractDataLoader):
    """WHAS500 dataset loader implementation."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from sksurv.datasets import load_whas500

        X, y = load_whas500()
        df = pd.DataFrame(X)
        df['time'] = y['lenfol']
        df['event'] = y['fstat'].astype(int)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.15/0.85, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class SUPPORT2DataLoader(AbstractDataLoader):
    """SUPPORT2 dataset loader implementation using pycox."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def _preprocess_pycox_support(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess pycox SUPPORT dataset with proper column mapping and encoding."""
        
        # Map pycox columns to meaningful names (based on the provided code)
        new_columns = [
            "age",
            "sex", 
            "race",
            "number_of_comorbidities",
            "presence_of_diabetes",
            "presence_of_dementia", 
            "presence_of_cancer",
            "mean_arterial_blood_pressure",
            "heart_rate",
            "respiration_rate",
            "temperature",
            "white_blood_cell_count",
            "serums_sodium",
            "serums_creatinine",
        ]
        
        # Get x_covar columns (x0, x1, x2, etc.)
        x_covar_columns = [c for c in df.columns if "x" == c[0]]
        col_map = dict(zip(x_covar_columns, new_columns))
        df = df.rename(columns=col_map)
        
        # Rename duration and event columns to match our standard
        df = df.rename(columns={'duration': 'time'})
        
        # The pycox SUPPORT dataset is already preprocessed and contains only numeric features
        # All columns are already float32/int32, so we just need to scale them
        
        # Define columns for scaling (all feature columns except time and event)
        feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        
        # Handle any missing values (though pycox dataset should be complete)
        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Scale all numerical features
        if feature_cols:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Ensure event column is binary (0/1) - should already be correct
        if df['event'].min() < 0 or df['event'].max() > 1:
            uniq = sorted(df['event'].unique())
            if len(uniq) == 2:
                df['event'] = (df['event'] == uniq[1]).astype(int)
            else:
                df['event'] = (df['event'] > 0).astype(int)
        
        # Remove any remaining missing values (shouldn't be any)
        df = df.dropna()
        
        return df

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from pycox.datasets import support

        # Load SUPPORT dataset from pycox
        df = support.read_df()
        
        # Preprocess the dataset
        df = self._preprocess_pycox_support(df)
        
        # Split the data
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.15/0.85, random_state=42, stratify=df_train['event'])
        
        # Create data loaders
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        
        # Get number of features
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        
        return dataloader_train, dataloader_val, dataloader_test, num_features


class METABRICDataLoader(AbstractDataLoader):
    """METABRIC dataset loader implementation using pycox."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from pycox.datasets import metabric

        df = metabric.read_df()
        if 'duration' in df.columns:
            df = df.rename(columns={'duration': 'time'})
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        continuous_cols = df.select_dtypes(include=[np.number]).columns
        feature_categorical_cols = [c for c in categorical_cols if c not in ['time', 'event']]
        feature_continuous_cols = [c for c in continuous_cols if c not in ['time', 'event']]

        # Fill missing values for categoricals first
        for col in feature_categorical_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'unknown')
        
        # One-hot encode non-binary categoricals; label-encode binary ones
        one_hot_cols = []
        binary_cols = []
        for col in feature_categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 2:
                binary_cols.append(col)
            else:
                one_hot_cols.append(col)
        
        # Apply encodings
        if binary_cols:
            for col in binary_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        if one_hot_cols:
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True).astype('float')
        
        # Continuous: impute median
        for col in feature_continuous_cols:
            df[col] = df[col].fillna(df[col].median())
        
        df = df.dropna()
        if df['event'].min() < 0 or df['event'].max() > 1:
            uniq = sorted(df['event'].unique())
            if len(uniq) == 2:
                df['event'] = (df['event'] == uniq[1]).astype(int)
            else:
                df['event'] = (df['event'] > 0).astype(int)
        df = df[df['time'] > 0]
        if feature_continuous_cols:
            scaler = StandardScaler()
            cont_cols_present = [c for c in feature_continuous_cols if c in df.columns]
            if cont_cols_present:
                df[cont_cols_present] = scaler.fit_transform(df[cont_cols_present])
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.15/0.85, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class SurvivalMNISTDataLoader(AbstractDataLoader):
    """SurvivalMNIST dataset loader for quick imaging experiments."""
    
    def __init__(
        self, 
        batch_size: int = 128,
        root: str = './data',
        target_size: Tuple[int, int] = (28, 28),
        max_survival_time: float = 100.0,
        min_survival_time: float = 1.0,
        event_rate: float = 0.7
    ):
        """
        Initialize SurvivalMNIST data loader.
        
        Args:
            batch_size: Batch size for data loaders
            root: Root directory for MNIST data
            target_size: Target image size
            max_survival_time: Maximum survival time
            min_survival_time: Minimum survival time
            event_rate: Probability of having an event
        """
        self.batch_size = batch_size
        self.root = root
        self.target_size = target_size
        self.max_survival_time = max_survival_time
        self.min_survival_time = min_survival_time
        self.event_rate = event_rate
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Load SurvivalMNIST dataset.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_features)
        """
        from .survival_mnist_dataset import create_survival_mnist_loaders
        
        # Create train and test loaders
        train_loader, test_loader, num_features = create_survival_mnist_loaders(
            batch_size=self.batch_size,
            root=self.root,
            target_size=self.target_size,
            max_survival_time=self.max_survival_time,
            min_survival_time=self.min_survival_time,
            event_rate=self.event_rate
        )
        
        # For simplicity, use test_loader as validation loader
        # In practice, you might want to split train into train/val
        val_loader = test_loader
        
        print(f"SurvivalMNIST loaded - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader, num_features


DATA_LOADERS = {
    'gbsg2': GBSG2DataLoader,
    'flchain': FLChainDataLoader,
    'whas500': WHAS500DataLoader,
    'survival_mnist': SurvivalMNISTDataLoader,
    'support2': SUPPORT2DataLoader,
    'metabric': METABRICDataLoader,
    'mimic': MIMICDataLoader,
}

