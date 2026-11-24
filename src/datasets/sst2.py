import torchvision

from .base import BaseDataset


class RenderedSST2(BaseDataset):
    dataset_class = torchvision.datasets.RenderedSST2
    data_subdir = "RenderedSST2"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}
