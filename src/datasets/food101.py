import torchvision

from .base import BaseDataset


class Food101(BaseDataset):
    dataset_class = torchvision.datasets.Food101
    data_subdir = "food101"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}
