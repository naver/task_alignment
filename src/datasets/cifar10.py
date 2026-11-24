import torchvision

from .base import BaseDataset


class CIFAR10(BaseDataset):
    dataset_class = torchvision.datasets.CIFAR10
    data_subdir = None
    train_kwargs = {"train": True}
    test_kwargs = {"train": False}
    classnames_from = "test"
