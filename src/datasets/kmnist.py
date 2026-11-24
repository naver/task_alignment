import torchvision

from .base import BaseDataset


class KMNIST(BaseDataset):
    dataset_class = torchvision.datasets.KMNIST
    data_subdir = "KMNIST"
    train_kwargs = {"train": True}
    test_kwargs = {"train": False}
