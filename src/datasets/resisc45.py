import os

from torchvision.datasets import ImageFolder

from .base import BaseDataset, classnames_from_folders


class RESISC45(BaseDataset):
    """RESISC45 aerial scene classification.

    The images sit in per-class folders under ``resisc45/NWPU-RESISC45/``. The
    published train/val/test partition is not encoded in the directory layout;
    it comes from ``resisc45/resisc45-<split>.txt``, each of which lists the
    file names belonging to that split. We therefore read the split file and
    let ``ImageFolder`` keep only the images it names.
    """

    data_subdir = "resisc45"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}

    def __init__(self, preprocess, location=os.path.expanduser("~/data"),
                 batch_size=32, num_workers=6):
        super().__init__(preprocess, location, batch_size, num_workers)

    def build_dataset(self, root, preprocess, split_kwargs):
        split = split_kwargs["split"]
        with open(os.path.join(root, f"resisc45-{split}.txt")) as handle:
            in_split = {line.strip() for line in handle if line.strip()}
        return ImageFolder(
            os.path.join(root, "NWPU-RESISC45"),
            transform=preprocess,
            is_valid_file=lambda path: os.path.basename(path) in in_split,
        )

    def get_classnames(self):
        # Folder names are underscore-separated; spaces read better in a prompt.
        return classnames_from_folders(self.train_dataset)
