from abc import ABC, abstractmethod
from typing import Tuple
from torch.utils.data import DataLoader


class AbstractDataLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Load and return train, validation, and test dataloaders plus feature count."""
        raise NotImplementedError
