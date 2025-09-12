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
from image_dataset import MIMICImageDataset, get_efficientnet_transforms


class GBSG2DataLoader(AbstractDataLoader):
    """GBSG2 dataset loader implementation."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        import lifelines

        df = lifelines.datasets.load_gbsg2()
        categorical_cols = ["horTh", "menostat", "tgrade"]
        drop_cols = ["horTh_no", "menostat_Post", "tgrade_I"]
        df_onehot = pd.get_dummies(df, columns=categorical_cols).astype("float")
        for col in drop_cols:
            if col in df_onehot.columns:
                df_onehot.drop(col, axis=1, inplace=True)
        df_train, df_test = train_test_split(df_onehot, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)
        dataloader_train = DataLoader(Custom_dataset(df_train), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(Custom_dataset(df_val), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(Custom_dataset(df_test), batch_size=len(df_test), shuffle=False)
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
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class LungDataLoader(AbstractDataLoader):
    """Lung dataset loader implementation."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        import lifelines

        df = lifelines.datasets.load_lung().dropna()
        if 'sex' in df.columns:
            df['sex'] = df['sex'] - 1
        time_col = 'time'
        event_col = 'status'
        if df[event_col].max() > 1:
            df[event_col] = (df[event_col] == 2).astype(int)
        feature_cols = [c for c in df.columns if c not in [time_col, event_col]]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col=time_col, event_col=event_col), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col=time_col, event_col=event_col), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col=time_col, event_col=event_col), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class RossiDataLoader(AbstractDataLoader):
    """Rossi dataset loader implementation."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        import lifelines

        df = lifelines.datasets.load_rossi().dropna()
        time_col = 'week'
        event_col = 'arrest'
        categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in [time_col, event_col]]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        feature_cols = [c for c in df.columns if c not in [time_col, event_col]]
        df[feature_cols] = df[feature_cols].astype(float)
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col=time_col, event_col=event_col), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col=time_col, event_col=event_col), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col=time_col, event_col=event_col), batch_size=len(df_test), shuffle=False)
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
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class CancerDataLoader(AbstractDataLoader):
    """Cancer dataset loader implementation from SurvSet."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        from SurvSet.data import SurvLoader

        loader = SurvLoader()
        df = loader.load_pickle('cancer')
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['time', 'event']:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'unknown')
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        if df['event'].min() < 0 or df['event'].max() > 1:
            uniq = sorted(df['event'].unique())
            if len(uniq) == 2:
                df['event'] = (df['event'] == uniq[1]).astype(int)
            else:
                df['event'] = (df['event'] > 0).astype(int)
        numerical_feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        if numerical_feature_cols:
            scaler = StandardScaler()
            df[numerical_feature_cols] = scaler.fit_transform(df[numerical_feature_cols])
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class SUPPORT2DataLoader(AbstractDataLoader):
    """SUPPORT2 dataset loader implementation."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        import pickle
        import SurvSet

        survset_base = os.path.dirname(SurvSet.__file__)
        support2_path = os.path.join(survset_base, 'resources', 'pickles', 'support2.pickle')
        with open(support2_path, 'rb') as f:
            df = pickle.load(f)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['time', 'event']:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'unknown')
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        if df['event'].min() < 0 or df['event'].max() > 1:
            uniq = sorted(df['event'].unique())
            if len(uniq) == 2:
                df['event'] = (df['event'] == uniq[1]).astype(int)
            else:
                df['event'] = (df['event'] > 0).astype(int)
        numerical_feature_cols = [c for c in df.columns if c not in ['time', 'event']]
        if numerical_feature_cols:
            scaler = StandardScaler()
            df[numerical_feature_cols] = scaler.fit_transform(df[numerical_feature_cols])
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
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
        for col in feature_categorical_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'unknown')
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
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
            df[feature_continuous_cols] = scaler.fit_transform(df[feature_continuous_cols])
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])
        dataloader_train = DataLoader(FlexibleDataset(df_train, time_col='time', event_col='event'), batch_size=self.batch_size, shuffle=True)
        dataloader_val = DataLoader(FlexibleDataset(df_val, time_col='time', event_col='event'), batch_size=len(df_val), shuffle=False)
        dataloader_test = DataLoader(FlexibleDataset(df_test, time_col='time', event_col='event'), batch_size=len(df_test), shuffle=False)
        x, _ = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features


class MIMICDataLoader(AbstractDataLoader):
    """MIMIC-IV Chest X-ray dataset loader implementation."""

    def __init__(
        self, 
        batch_size: int = 128,
        data_dir: str = "Z:/mimic-cxr-jpg-2.1.0.physionet.org/",
        csv_path: str = "data/mimic/mimic_cxr_splits.csv",
        target_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = True
    ):
        """
        Initialize MIMIC data loader.
        
        Args:
            batch_size: Batch size for data loaders
            data_dir: Base directory containing MIMIC data
            csv_path: Path to preprocessed CSV file
            target_size: Target image size for EfficientNet-B0
            use_augmentation: Whether to use data augmentation for training
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_size = target_size
        self.use_augmentation = use_augmentation

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Load MIMIC-IV Chest X-ray dataset.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_features)
            For image data, num_features represents the number of channels (3 for RGB)
        """
        import pandas as pd
        
        # Load preprocessed data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Preprocessed CSV not found at {self.csv_path}. "
                "Please run preprocess_mimic.py first to generate the CSV file."
            )
        
        df = pd.read_csv(self.csv_path)
        
        # Filter out images that don't exist
        df = df[df['exists'] == True].copy()
        
        # Split data
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'val'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        print(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        print(f"Event rates - Train: {df_train['event'].mean():.3f}, Val: {df_val['event'].mean():.3f}, Test: {df_test['event'].mean():.3f}")
        
        # Create transforms
        train_transform = get_efficientnet_transforms(
            target_size=self.target_size,
            is_training=self.use_augmentation
        )
        val_test_transform = get_efficientnet_transforms(
            target_size=self.target_size,
            is_training=False
        )
        
        # Create datasets
        train_dataset = MIMICImageDataset(
            df_train, 
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=train_transform,
            target_size=self.target_size
        )
        
        val_dataset = MIMICImageDataset(
            df_val,
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=val_test_transform,
            target_size=self.target_size
        )
        
        test_dataset = MIMICImageDataset(
            df_test,
            self.data_dir,
            time_col='tte',
            event_col='event',
            path_col='path',
            transform=val_test_transform,
            target_size=self.target_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # For image data, num_features represents the number of channels
        num_features = 3  # RGB channels
        
        return train_loader, val_loader, test_loader, num_features


DATA_LOADERS = {
    'gbsg2': GBSG2DataLoader,
    'flchain': FLChainDataLoader,
    'lung': LungDataLoader,
    'rossi': RossiDataLoader,
    'whas500': WHAS500DataLoader,
    'cancer': CancerDataLoader,
    'support2': SUPPORT2DataLoader,
    'metabric': METABRICDataLoader,
    'mimic': MIMICDataLoader,
}

