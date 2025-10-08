"""
SurvMNIST: Survival MNIST Dataset Module

A synthetic survival analysis dataset based on MNIST digits for quick prototyping
and testing of imaging survival methods.
"""

from .survival_mnist_dataset import (
    SurvivalMNISTDataset,
    TorchSurvMNISTDataset,
    create_survival_mnist_loaders,
    create_torchsurv_mnist_loaders,
    get_survival_mnist_transforms
)

__all__ = [
    "SurvivalMNISTDataset",
    "TorchSurvMNISTDataset",
    "create_survival_mnist_loaders",
    "create_torchsurv_mnist_loaders",
    "get_survival_mnist_transforms"
]

