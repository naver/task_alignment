import torchvision

from .base import BaseDataset


class FashionMNIST(BaseDataset):
    dataset_class = torchvision.datasets.FashionMNIST
    data_subdir = "FashionMNIST"
    train_kwargs = {"train": True}
    test_kwargs = {"train": False}
