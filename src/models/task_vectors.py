# Adapted from:
#   https://github.com/nik-dim/tall_masks/blob/master/src/models/task_vectors.py (MIT)
#   https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/models/task_vectors.py (Apache-2.0)
import abc
from typing import OrderedDict, Union

import torch

from src.models.modeling import ImageEncoder
from src.utils.variables_and_paths import MODELS

_Checkpoint = Union[str, dict, torch.nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector(abc.ABC):
    def __init__(
        self,
        model_name,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
    ):
        """Build a task vector for ``model_name``.

        Supply ``vector`` to wrap an existing per-parameter delta directly, or
        supply both ``pretrained_checkpoint`` and ``finetuned_checkpoint`` to
        derive one as the difference between the two sets of weights.
        """
        assert (
            model_name in MODELS
        ), f"Invalid model name: {model_name}. Valid models are: {MODELS}"
        self.model_name = model_name
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                # parse pretrained_checkpoint
                pretrained_state_dict = self._safe_load(pretrained_checkpoint)

                # parse finetuned_checkpoint
                finetuned_state_dict = self._safe_load(finetuned_checkpoint)

                if "model_name" in finetuned_state_dict.keys():
                    finetuned_state_dict.pop("model_name")

                # the final task vector is the difference between finetuned and pretrained vectors.
                assert (
                    pretrained_state_dict.keys() == finetuned_state_dict.keys()
                ), f"State dicts have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}."
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return self._load_checkpoint(checkpoint).state_dict()
        elif isinstance(checkpoint, dict):
            return checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Invalid type for checkpoint: {type(checkpoint)}")

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint) -> torch.nn.Module:
        """Load a checkpoint into a model."""
        raise NotImplementedError

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Skipping key {key}: absent from the other task vector.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, args=None):
        """Return ``pretrained_checkpoint`` with this vector added, scaled by ``scaling_coef``."""
        with torch.no_grad():
            model = self._load_checkpoint(pretrained_checkpoint).to(args.device)
            base = model.state_dict()
            for key in base:
                if key not in self.vector:
                    print(f"Skipping key {key}: no matching entry in the task vector.")
            merged = {
                key: weights + scaling_coef * self.vector[key].to(args.device)
                for key, weights in base.items()
                if key in self.vector
            }
        model.load_state_dict(merged)
        return model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return ImageEncoder.load(self.model_name, checkpoint)

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)
