import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple

from benchmark_framework import AbstractDataLoader
from flexible_dataset import Custom_dataset


class GBSG2DataLoader(AbstractDataLoader):
    """GBSG2 dataset loader implementation."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load GBSG2 dataset and return dataloaders."""
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

        x, (event, time) = next(iter(dataloader_train))
        num_features = x.size(1)

        return dataloader_train, dataloader_val, dataloader_test, num_features
