# Adapted from, in the order this file was built up:
#   https://github.com/mlfoundations/wise-ft/blob/master/src/models/utils.py (MIT)
#   https://github.com/nik-dim/tall_masks/blob/master/src/utils/utils.py (MIT)
#     - the task-vector and state-dict helpers
import copy
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: List[int] = (1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(model, save_path, save_state_dict=True):
    # TODO: hacky way to save state dict
    if save_state_dict and isinstance(model, torch.nn.Module):
        model = model.state_dict()
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier, dataset_name=None):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    # if dataset_name is not None:
    #    return classifier(inputs, dataset_name)
    return classifier(inputs)


def find_optimal_coef(
    results: Dict[str, Any],
    metric: str = "avg_normalized_top1",
    minimize: bool = False,
    control_metric: Optional[str] = None,
    control_metric_threshold: float = 0.0,
) -> float:
    """
    Finds the optimal coefficient based on the given results and metric.

    Args:
        results (Dict[str, Any]): A dictionary containing the results for different scaling coefficients.
        metric (str, optional): The metric to optimize. Defaults to "avg_normalized_top1".
        minimize (bool, optional): Whether to minimize the metric. Defaults to False.
        control_metric (str, optional): The control metric to check against. Defaults to None.
        control_metric_threshold (float, optional): The threshold value for the control metric. Defaults to 0.0.

    Returns:
        The optimal coefficient based on the given results and metric.
    """
    best_coef = None
    best_metric = float("inf") if minimize else float("-inf")
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
    if K == 100:
        # print("Not applying mask")
        if return_mask:
            return M, torch.ones_like(M), None
        else:
            return M, torch.ones_like(M)

    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if reshape_mask:
        final_mask = final_mask.reshape(M.shape)

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    else:
        return M * final_mask, final_mask.float().mean(dim=1)


