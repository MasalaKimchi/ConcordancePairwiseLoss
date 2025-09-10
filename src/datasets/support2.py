import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
from typing import Tuple

from benchmark_framework import AbstractDataLoader
from flexible_dataset import FlexibleDataset


class SUPPORT2DataLoader(AbstractDataLoader):
    """SUPPORT2 dataset loader implementation."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load SUPPORT2 dataset and return dataloaders."""
        # Attempt to locate SurvSet installation
        survset_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'Anaconda3', 'envs',
            'concordance-pairwise-loss', 'lib', 'site-packages', 'SurvSet',
            'resources', 'pickles'
        )
        support2_path = os.path.join(survset_path, 'support2.pickle')
        try:
            import SurvSet
            survset_base = os.path.dirname(SurvSet.__file__)
            support2_path = os.path.join(survset_base, 'resources', 'pickles', 'support2.pickle')
        except Exception:
            pass

        with open(support2_path, 'rb') as f:
            df = pickle.load(f)

        if 'time' not in df.columns or 'event' not in df.columns:
            raise ValueError("Expected 'time' and 'event' columns not found in dataset")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}
        for col in categorical_cols:
            if col not in ['time', 'event']:
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna('unknown')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna()

        feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        if df['event'].min() < 0 or df['event'].max() > 1:
            unique_events = sorted(df['event'].unique())
            if len(unique_events) == 2:
                df['event'] = (df['event'] == unique_events[1]).astype(int)
            else:
                df['event'] = (df['event'] > 0).astype(int)
        df = df[df['time'] > 0]

        numerical_feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        if len(numerical_feature_cols) > 0:
            scaler = StandardScaler()
            df[numerical_feature_cols] = scaler.fit_transform(df[numerical_feature_cols])

        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['event'])
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42, stratify=df_train['event'])

        dataloader_train = DataLoader(
            FlexibleDataset(df_train, time_col='time', event_col='event'),
            batch_size=self.batch_size,
            shuffle=True,
        )
        dataloader_val = DataLoader(
            FlexibleDataset(df_val, time_col='time', event_col='event'),
            batch_size=len(df_val),
            shuffle=False,
        )
        dataloader_test = DataLoader(
            FlexibleDataset(df_test, time_col='time', event_col='event'),
            batch_size=len(df_test),
            shuffle=False,
        )

        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features
