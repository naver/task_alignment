import torchvision

from .base import BaseDataset


class CIFAR100(BaseDataset):
    dataset_class = torchvision.datasets.CIFAR100
    data_subdir = None
    train_kwargs = {"train": True}
    test_kwargs = {"train": False}
    classnames_from = "test"
    # Long-standing behaviour of this codebase: the CIFAR100 train loader does
    # not shuffle.  Downstream results depend on it, so it is preserved.
    shuffle_train = False
