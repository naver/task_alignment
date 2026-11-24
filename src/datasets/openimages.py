import os
from pathlib import Path
from typing import Iterable, List

from PIL import Image
import torch
from torch.utils.data import Dataset


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


class _FlatImageDataset(Dataset):
    """A flat-folder image dataset with dummy labels."""

    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        self.samples = _list_images(root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # Return a dummy int label to stay compatible with default collate.
        return image, 0


class OpenImages:
    """
    OpenImages wrapper for flat image directories.

    Expected directory layout under args.data_location:
    - openimages/images/train
    - openimages/images/validation
    """

    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=6,
    ):
        base = Path(location) / "openimages" / "images"
        train_dir = base / "validation"
        # val_dir = base / "validation"

        self.train_dataset = _FlatImageDataset(train_dir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # self.test_dataset = _FlatImageDataset(val_dir, transform=preprocess)
        # self.test_loader = torch.utils.data.DataLoader(
        #     self.test_dataset, batch_size=batch_size, num_workers=num_workers
        # )

        # Labels are ignored, but keep a placeholder to match other datasets.
        self.classnames: Iterable[str] = ["openimages_dummy"]

        print(
            f"OpenImages: Note I'm using val as train as it is already very large! "
            f"train={len(self.train_dataset)} "
            f"(loading images from: {train_dir})"
            f"BS={batch_size} "
        )
