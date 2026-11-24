import torchvision

from .base import BaseDataset


class MNIST(BaseDataset):
    dataset_class = torchvision.datasets.MNIST
    data_subdir = None
    train_kwargs = {"train": True}
    test_kwargs = {"train": False}
    classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
