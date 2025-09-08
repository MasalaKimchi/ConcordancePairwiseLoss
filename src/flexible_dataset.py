import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset


def plot_losses(train_losses, val_losses, title: str = "Cox") -> None:
    train_losses = torch.stack(train_losses) / train_losses[0]
    val_losses = torch.stack(val_losses) / val_losses[0]

    plt.plot(train_losses, label="training")
    plt.plot(val_losses, label="validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Normalized loss")
    plt.title(title)
    plt.yscale("log")
    plt.show()


class FlexibleDataset(Dataset):
    """Flexible dataset that can handle different column names for event and time."""

    def __init__(self, df: pd.DataFrame, event_col: str = "cens", time_col: str = "time"):
        self.df = df
        self.event_col = event_col
        self.time_col = time_col
        
        # Verify columns exist
        if event_col not in df.columns:
            raise ValueError(f"Event column '{event_col}' not found in dataset. Available columns: {list(df.columns)}")
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset. Available columns: {list(df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Targets
        event = torch.tensor(sample[self.event_col]).bool()
        time = torch.tensor(sample[self.time_col]).float()
        # Predictors
        x = torch.tensor(sample.drop([self.event_col, self.time_col]).values).float()
        return x, (event, time)


# Keep the original class for backward compatibility
class Custom_dataset(Dataset):
    """Custom dataset for the GSBG2 brain cancer dataset"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Targets
        event = torch.tensor(sample["cens"]).bool()
        time = torch.tensor(sample["time"]).float()
        # Predictors
        x = torch.tensor(sample.drop(["cens", "time"]).values).float()
        return x, (event, time)
