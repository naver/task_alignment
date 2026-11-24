"""Shared base class for the torchvision-backed dataset wrappers.

Every wrapper in this package exposes the same five public attributes
(``train_dataset``, ``test_dataset``, ``train_loader``, ``test_loader`` and
``classnames``) and is built the same way, so the construction logic lives here
once.  A concrete dataset module only declares what differs from the default:
the backing torchvision class, the split keyword arguments, the sub-directory of
``location`` that holds its files, and where its class names come from.

Class attributes a subclass may set
-----------------------------------
``dataset_class``   torchvision dataset class used for both splits (required).
``data_subdir``     name of the sub-directory of ``location`` handed to the
                    backing class as ``root``.  ``None`` (the default) means
                    ``location`` is passed through unchanged.
``train_kwargs``    split selection kwargs for the training split.
``test_kwargs``     split selection kwargs for the test split.
``classnames``      an explicit list of class names, in label-index order.  When
                    left as ``None`` the names are taken from the backing
                    dataset's ``.classes`` attribute, from the split named by
                    ``classnames_from``.
``classnames_from`` ``"train"`` (default) or ``"test"`` -- which backing split
                    supplies ``.classes``.
``shuffle_train``   whether the train loader shuffles.  ``True`` everywhere
                    except the few datasets whose long-standing behaviour is an
                    unshuffled train loader; those set it to ``False``.
"""

import os

import torch


class BaseDataset:
    # --- what a subclass declares -------------------------------------------
    dataset_class = None
    data_subdir = None
    train_kwargs = {}
    test_kwargs = {}
    classnames = None
    classnames_from = "train"
    shuffle_train = True

    def __init__(self, preprocess,
                 location=os.path.expanduser("~/data"),
                 batch_size=128,
                 num_workers=6):
        root = self.get_root(location)

        self.train_dataset = self.build_dataset(root, preprocess, self.train_kwargs)
        self.test_dataset = self.build_dataset(root, preprocess, self.test_kwargs)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle_train,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.get_classnames()

    # --- hooks ---------------------------------------------------------------
    def get_root(self, location):
        """Directory handed to the backing dataset class as ``root``."""
        if self.data_subdir is None:
            return location
        return os.path.join(location, self.data_subdir)

    def build_dataset(self, root, preprocess, split_kwargs):
        return self.dataset_class(
            root=root,
            download=True,
            transform=preprocess,
            **split_kwargs,
        )

    def get_classnames(self):
        """Class names in label-index order."""
        if self.classnames is not None:
            # Copy so that a subclass' declared list cannot be mutated through
            # an instance attribute.
            return list(self.classnames)
        source = self.test_dataset if self.classnames_from == "test" else self.train_dataset
        return list(source.classes)


def classnames_from_folders(dataset, strip=0):
    """Readable class names taken from a dataset's ``class_to_idx`` mapping.

    Names come back in label-index order. ``strip`` drops that many leading
    characters from each folder name, for layouts that prefix them (SUN397 files
    its categories under ``a_abbey``, ``b_bakery`` and so on). Underscores become
    spaces, which reads better in a zero-shot prompt.
    """
    ordered = sorted(dataset.class_to_idx, key=dataset.class_to_idx.get)
    return [name[strip:].replace("_", " ") for name in ordered]
