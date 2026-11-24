import copy
import inspect
import random
import sys

import torch
from torch.utils.data.dataset import random_split

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.emnist import EMNIST
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.fashionmnist import FashionMNIST
from src.datasets.fer2013 import FER2013
from src.datasets.flowers102 import Flowers102
from src.datasets.food101 import Food101
from src.datasets.gtsrb import GTSRB
from src.datasets.openimages import OpenImages
from src.datasets.kmnist import KMNIST
from src.datasets.mnist import MNIST
from src.datasets.oxfordpets import OxfordIIITPet
from src.datasets.pcam import PCAM
from src.datasets.resisc45 import RESISC45
from src.datasets.sst2 import RenderedSST2
from src.datasets.stl10 import STL10
from src.datasets.sun397 import SUN397
from src.datasets.svhn import SVHN

registry = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class SplitView:
    """Lightweight container holding a train/test split and their loaders.

    Mirrors the attribute layout of the concrete dataset wrappers in this
    package, so that a derived split is interchangeable with a real dataset.
    """

    _FIELDS = (
        "train_dataset",
        "train_loader",
        "test_dataset",
        "test_loader",
        "classnames",
    )

    def __init__(self):
        for field in self._FIELDS:
            setattr(self, field, None)


def carve_val_split(
    source,
    split_name,
    batch_size,
    num_workers,
    val_frac,
    val_cap=None,
    seed=0,
):
    """Carve a validation split out of ``source``'s training set.

    The partition is drawn with a generator seeded by ``seed``, so the same
    ``seed`` always yields the same split, both within and across runs. The
    returned object is an instance of a class named ``split_name``;
    downstream feature caching keys off that name via ``type(...).__name__``.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must lie in (0, 1), got {val_frac!r}")

    n_total = len(source.train_dataset)
    n_val = int(n_total * val_frac)
    if val_cap is not None:
        n_val = min(n_val, val_cap)
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Cannot split {n_total} training examples into "
            f"{n_train} train / {n_val} val with val_frac={val_frac}."
        )

    train_split, val_split = random_split(
        source.train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # Guards that the seeded partition has not drifted; 36044 is the first
    # training index MNIST has always produced for this seed.
    if split_name == "MNISTVal":
        assert train_split.indices[0] == 36044

    def _build_loader(subset, shuffle):
        return torch.utils.data.DataLoader(
            subset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    split_dataset = type(split_name, (SplitView,), {})()
    split_dataset.train_dataset = train_split
    split_dataset.train_loader = _build_loader(train_split, shuffle=True)
    split_dataset.test_dataset = val_split
    split_dataset.test_loader = _build_loader(val_split, shuffle=False)
    split_dataset.classnames = copy.copy(source.classnames)
    return split_dataset


def dataset_by_name(
    name,
    preprocess,
    location,
    batch_size=128,
    num_workers=6,
    val_frac=0.1,
    val_cap=5000,
):
    """Instantiate a dataset by name.

    A name ending in ``Val`` is served by a dedicated class when the registry
    provides one (a few datasets ship their own validation split); otherwise the
    validation set is carved out of the base dataset's training split.
    """
    if name in registry:
        return registry[name](
            preprocess,
            location=location,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if not name.endswith("Val"):
        raise ValueError(
            f"Unsupported dataset: {name}. Supported: {sorted(registry)}"
        )

    base = dataset_by_name(
        name[: -len("Val")], preprocess, location, batch_size, num_workers
    )
    return carve_val_split(
        base,
        name,
        batch_size,
        num_workers,
        val_frac,
        val_cap,
    )
