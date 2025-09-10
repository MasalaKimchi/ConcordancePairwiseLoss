import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple

from benchmark_framework import AbstractDataLoader
from flexible_dataset import FlexibleDataset


class RossiDataLoader(AbstractDataLoader):
    """Rossi dataset loader implementation."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load Rossi dataset and return dataloaders."""
        import lifelines

        df = lifelines.datasets.load_rossi()
        df = df.dropna()

        time_col = 'week'
        event_col = 'arrest'

        categorical_cols = [
            col for col in df.columns
            if col not in [time_col, event_col] and df[col].dtype == 'object'
        ]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        feature_cols = [col for col in df.columns if col not in [time_col, event_col]]
        df[feature_cols] = df[feature_cols].astype(float)

        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=42)

        dataloader_train = DataLoader(
            FlexibleDataset(df_train, time_col=time_col, event_col=event_col),
            batch_size=self.batch_size,
            shuffle=True,
        )
        dataloader_val = DataLoader(
            FlexibleDataset(df_val, time_col=time_col, event_col=event_col),
            batch_size=len(df_val),
            shuffle=False,
        )
        dataloader_test = DataLoader(
            FlexibleDataset(df_test, time_col=time_col, event_col=event_col),
            batch_size=len(df_test),
            shuffle=False,
        )

        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)
        return dataloader_train, dataloader_val, dataloader_test, num_features
