# Class names as published by OpenAI for CLIP:
#   https://github.com/openai/CLIP/blob/main/data/prompts.md (MIT)
import torchvision

from .base import BaseDataset


# Published class labels for this dataset, in label-index order.
pcam_classnames = [
    "lymph node", "lymph node containing metastatic tumor tissue",
]


class PCAM(BaseDataset):
    dataset_class = torchvision.datasets.PCAM
    data_subdir = "PCAM"
    train_kwargs = {"split": "train"}
    test_kwargs = {"split": "test"}
    classnames = pcam_classnames
