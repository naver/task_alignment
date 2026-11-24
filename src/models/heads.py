# Our own implementation. The technique -- embed every prompt for a class,
# average the unit-normalised embeddings and renormalise -- is the standard CLIP
# zero-shot classifier construction, as implemented by the open_clip library:
#   https://github.com/mlfoundations/open_clip/blob/v2.10.1/src/training/zero_shot.py (MIT)
#     - zero_shot_classifier()

import argparse
import os
from typing import Callable, List

import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.registry import dataset_by_name
from src.datasets.templates import get_templates
from src.models.modeling import ClassificationHead, ImageEncoder
from src.utils.variables_and_paths import TQDM_BAR_FORMAT


def encode_zeroshot_head(
    text_model: nn.Module,
    task: str,
    prompts: List[Callable[[str], str]],
    data_root: str,
    device: torch.device,
) -> ClassificationHead:
    """Encode a zero-shot classifier for ``task``.

    Each class is described by every prompt in ``prompts``. The prompts are
    embedded by the text tower, each embedding is unit-normalised, and their mean
    -- itself re-normalised -- becomes that class's row. Rows are scaled by the
    model's learned logit scale so the head can be used as a plain linear layer.

    Returns a ``ClassificationHead`` whose weight matrix is (classes x features).
    """
    dataset = dataset_by_name(task, None, location=data_root)
    text_model.to(device).eval()

    print(f"Encoding zero-shot class prompts for {task}.")
    rows = []
    with torch.no_grad():
        for classname in tqdm(dataset.classnames, bar_format=TQDM_BAR_FORMAT):
            tokens = open_clip.tokenize([fill(classname) for fill in prompts])
            embedded = text_model.encode_text(tokens.to(device))
            embedded = embedded / embedded.norm(dim=-1, keepdim=True)
            averaged = embedded.mean(dim=0)
            rows.append(averaged / averaged.norm())

        weights = torch.stack(rows, dim=0).float().to(device)
        weights = weights * text_model.logit_scale.exp()

    print(f"zero-shot head for {task}: {tuple(weights.shape)}")
    return ClassificationHead(normalize=True, weights=weights)


def zeroshot_head_for(args: argparse.Namespace, task: str) -> nn.Module:
    """Return the zero-shot head for ``task``, encoding it if not already cached.

    Heads are cached per model and task under ``args.save_dir``; encoding one
    costs a full pass of the text tower over every class prompt, so it is done
    once and reused.
    """
    if not task.endswith("Val"):
        task += "Val"

    cached = os.path.join(args.save_dir, f"head_{task}.pt")
    if os.path.exists(cached):
        print(f"Loading classification head for {args.model} on {task} from {cached}")
        return ClassificationHead.load(cached)

    print(f"No cached head for {args.model}/{task} at {cached}; encoding a new one.")
    text_tower = ImageEncoder(args.model, keep_lang=True).model
    head = encode_zeroshot_head(
        text_tower, task, get_templates(task), args.data_location, args.device
    )
    os.makedirs(args.save_dir, exist_ok=True)
    head.save(cached)
    return head
