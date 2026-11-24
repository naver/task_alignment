import torchvision

from .base import BaseDataset


class OxfordIIITPet(BaseDataset):
    dataset_class = torchvision.datasets.OxfordIIITPet
    data_subdir = "OxfordIIITPet"
    train_kwargs = {"split": "trainval"}
    test_kwargs = {"split": "test"}
