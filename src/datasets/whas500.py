import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from typing import Tuple

from benchmark_framework import AbstractDataLoader
from flexible_dataset import FlexibleDataset


class WHAS500DataLoader(AbstractDataLoader):
    """WHAS500 dataset loader implementation."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load WHAS500 dataset and return dataloaders."""
        from sksurv.datasets import load_whas500

        X, y = load_whas500()
        df = pd.DataFrame(X)
        df['time'] = y['lenfol']
        df['event'] = y['fstat'].astype(int)

        numerical_cols = df.select_dtypes(include=[float, int]).columns
        for col in numerical_cols:
            if col not in ['time', 'event']:
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna()

        feature_cols = [col for col in df.columns if col not in ['time', 'event']]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

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
