import torchvision

from .base import BaseDataset


class SVHN(BaseDataset):
    dataset_class = torchvision.datasets.SVHN
    data_subdir = "svhn"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}
    classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
