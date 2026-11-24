import torchvision

from .base import BaseDataset


class STL10(BaseDataset):
    dataset_class = torchvision.datasets.STL10
    data_subdir = "stl10"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}
