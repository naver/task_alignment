import os

import torchvision

from .base import BaseDataset, classnames_from_folders


class SUN397(BaseDataset):
    """SUN397 scenes, read from `sun397/<split>/<class>/` image folders."""

    data_subdir = "sun397"
    train_kwargs = {"folder": "train"}
    test_kwargs = {"folder": "val"}

    def __init__(self, preprocess, location=os.path.expanduser("~/data"),
                 batch_size=32, num_workers=6):
        super().__init__(preprocess, location, batch_size, num_workers)

    def build_dataset(self, root, preprocess, split_kwargs):
        return torchvision.datasets.ImageFolder(
            os.path.join(root, split_kwargs["folder"]), transform=preprocess
        )

    def get_classnames(self):
        # Categories are filed under an alphabetical prefix, e.g. "a_abbey".
        return classnames_from_folders(self.train_dataset, strip=2)
