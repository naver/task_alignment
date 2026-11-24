# Adapted from, in the order this file was built up:
#   https://github.com/mlfoundations/wise-ft/blob/master/src/models/eval.py (MIT)
#     - the single-dataset evaluation loop
#   https://github.com/nik-dim/tall_masks/blob/master/src/eval/eval.py (MIT)
#     - multi-task evaluation and the mask/coefficient paths
#   https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/eval/eval.py (Apache-2.0)
#     - the merging-method evaluation entry points
# The task-alignment additions are ours.
import copy
import time

import numpy as np
import torch
import tqdm

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import dataset_by_name
from src.models.heads import zeroshot_head_for
from src.models.modeling import ImageClassifier
from src.models.task_vectors import _Checkpoint, _TaskVector
from src.utils import utils
from src.utils.tas_utils import tas_single_dataset, get_ft_encoder
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_confusion_matrix,
)


def LiNeS_scaling(task_vector, alpha, beta, num_blocks):
    """
    LiNeS: Progressively scales the task vector based on layer depth.
    """
    scaled_task_vector = copy.deepcopy(task_vector)
    key_blocks = list(f".{i}." for i in range(0, num_blocks))

    layer_scalings_dict = {}
    for k in scaled_task_vector.vector.keys():
        for layer, block in enumerate(key_blocks):
            if block in k:
                layer_scalings_dict[k] = alpha + beta * (layer / (num_blocks - 1))
                break

    print(f"LiNeS: The layers are scaled between {alpha} to {alpha + beta}")

    scaled_task_vector.vector = {
        k: scaled_task_vector.vector[k] * layer_scalings_dict.get(k, alpha)
        for k in scaled_task_vector.vector.keys()
    }

    return scaled_task_vector


def eval_single_dataset(image_encoder, dataset_name, args):
    """Top-1 accuracy of ``image_encoder`` on ``dataset_name``'s test split.

    The encoder is paired with that dataset's zero-shot head to form a
    classifier, which is then run over the whole split.
    """
    if args.skip_eval:
        return {"top1": float("nan")}

    started = time.time()
    model = ImageClassifier(image_encoder, zeroshot_head_for(args, dataset_name))
    model.eval()

    dataset = dataset_by_name(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    hits = seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            predicted = utils.get_logits(images, model).argmax(dim=1)
            hits += (predicted == labels).sum().item()
            seen += labels.numel()

    top1 = hits / seen
    elapsed = time.time() - started
    print(
        f"Done evaluating on {dataset_name}.\t Accuracy: {100*top1:.2f}%.\t Total time: {elapsed:.2f}s"
    )
    return {"top1": top1}


def evaluate(
    pretrained_checkpoint,
    task_vector,
    args,
    scaling_coef,
    eval_masks=None,
    svd_dict=None,
):
    per_dataset_results = {}

    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )

    if eval_masks != None:
        assert args.method.name in ["tall_mask", "mag_masking"]
    elif svd_dict is not None:
        assert args.method.name in ["TSVC"]
    else:
        if args.apply_lines:
            if args.method.name not in ["sum", "ties", "consensus", "single_task", "zeroshot"]:
                raise ValueError(
                    "apply_lines is only supported for methods "
                    "sum/ties/consensus/single_task/zeroshot "
                    f"(got {args.method.name})."
                )
            num_blocks = 12 if args.model != "ViT-L-14" else 24
            if args.method.name == "single_task":
                task_vector = LiNeS_scaling(
                    task_vector,
                    alpha=scaling_coef,
                    beta=1 - scaling_coef,
                    num_blocks=num_blocks,
                )
            else:
                alpha = (args.norm_summed_tvs / args.norm_mtv) * 1 / args.num_tasks
                task_vector = LiNeS_scaling(
                    task_vector,
                    alpha=alpha,
                    beta=scaling_coef,
                    num_blocks=num_blocks,
                )
            image_encoder = task_vector.apply_to(
                pretrained_checkpoint, scaling_coef=1.0, args=args
            )
        else:
            image_encoder = task_vector.apply_to(
                pretrained_checkpoint, scaling_coef=scaling_coef, args=args
            )

    for dataset_name in eval_datasets:

        if eval_masks != None:
            sparse_task_vector = copy.deepcopy(task_vector)
            # remove "Val" from dataset_name
            mask = (
                eval_masks[dataset_name[:-3]]
                if "Val" in dataset_name
                else eval_masks[dataset_name]
            )
            # apply mask to sparsify the task vectors with Hadamard product
            sparse_task_vector.vector = {
                k: sparse_task_vector.vector[k] * mask[k].bool().cpu()
                for k in mask.keys()
            }
            # reconstruct theta_t^
            image_encoder = sparse_task_vector.apply_to(
                pretrained_checkpoint, scaling_coef=1.0, args=args
            )
        elif svd_dict != None:
            new_task_vector = copy.deepcopy(task_vector)
            # remove "Val" from dataset_name
            dt_name = dataset_name if "Val" in dataset_name else dataset_name + "Val"
            new_task_vector.vector = {
                key: (
                    (
                        svd_dict[dt_name][key]["u"]
                        @ torch.diag_embed(svd_dict[dt_name][key]["s"])
                        @ svd_dict[dt_name][key]["v"]
                    ).to(args.device)
                    if "u" in svd_dict[dt_name][key]
                    else svd_dict[dt_name][key]["dim1"].to(args.device)
                )
                for key in svd_dict[dt_name].keys()
            }

            image_encoder = new_task_vector.apply_to(
                pretrained_checkpoint, scaling_coef=1.0, args=args
            )

        # evalute performance
        results = eval_single_dataset(image_encoder, dataset_name, args)
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

        if args.compute_tas and not args.skip_tas:
            ft_encoder = get_ft_encoder(dataset_name, args)
            tas = tas_single_dataset(image_encoder, ft_encoder, dataset_name, args)
            per_dataset_results[dataset_name + ":tas"] = tas["tas"]

    return per_dataset_results


def evaluate_image_encoder(image_encoder, args):
    per_dataset_results = {}

    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )

    for dataset_name in eval_datasets:
        results = eval_single_dataset(image_encoder, dataset_name, args)
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

        if args.compute_tas and not args.skip_tas:
            ft_encoder = get_ft_encoder(dataset_name, args)
            tas = tas_single_dataset(image_encoder, ft_encoder, dataset_name, args)
            per_dataset_results[dataset_name + ":tas"] = tas["tas"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector: _TaskVector,
    pretrained_checkpoint: _Checkpoint,
    args,
    scaling_coef: float,
    eval_masks=None,
    svd_dict=None,
):
    start_time = time.time()

    coef_info = evaluate(
        pretrained_checkpoint,
        task_vector,
        args,
        scaling_coef,
        eval_masks,
        svd_dict,
    )

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )
    if args.compute_tas and not args.skip_tas:
        coef_info["avg_tas"] = np.mean(
            [coef_info[dataset + ":tas"] for dataset in args.eval_datasets]
        )

    print(f"Total evaluation time: {time.time() - start_time:.2f}s")
    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, eval_masks=None, svd_dict=None
):
    info = {}

    if args.method.name == "tall_mask" or eval_masks is not None:
        scaling_coef_range = [1.0]
    elif args.method.name == "TSVC":
        scaling_coef_range = [1.0]
    elif args.method.name == "single_task":
        if args.apply_lines:
            scaling_coef_range = np.arange(0.0, 1.1, 0.1)[::-1]
        else:
            scaling_coef_range = [1.0]
    elif args.method.name == "zeroshot":
        scaling_coef_range = [0.0]
    elif args.method.name == "average":
        scaling_coef_range = [1 / args.num_tasks]
    elif args.specify_lambda != "None":
        scaling_coef_range = [args.specify_lambda]
    # elif args.method.name == "TSVM":
    #     # if args.num_tasks < 8:
    #     # scaling_coef_range = np.linspace(0.0, 2.0, args.n_eval_points)[1:]
    #     #    scaling_coef_range = np.linspace(0.0, 3.0, args.n_eval_points)[1:]
    #     # else:
    #     scaling_coef_range = np.linspace(0.0, 3.0, args.n_eval_points)[1:]
    # elif args.method.name == "TSVM_2":
    #     scaling_coef_range = np.linspace(0.0, 3.0, args.n_eval_points)[1:]
    # else:
        # scaling_coef_range = np.linspace(0.0, 1.0, args.n_eval_points)[1:]
    elif args.method.name in ["sum"]:
        scaling_coef_range = np.linspace(0.0, 1.0, args.n_eval_points)[1:]
    elif args.method.name in ["ties"]:
        args.n_eval_points = 8
        scaling_coef_range = np.linspace(0.6, 1.8, args.n_eval_points)[1:]
    elif args.method.name == "STAR":
        args.n_eval_points = 11
        scaling_coef_range = np.linspace(0.0, 2.0, args.n_eval_points)[1:]
    elif args.method.name in ["iso_c", "iso_cts"]:
        scaling_coef_range = np.linspace(0.0, 3.0, args.n_eval_points)[1:]
    else:
        scaling_coef_range = np.linspace(0.0, 2.0, args.n_eval_points)[1:]

    if args.method.name == "tall_mask":
        if args.method.load_mask:
            print("=" * 43, f"Evaluating the loaded TALL masks", "=" * 43)
            info["loaded_mask"] = evaluate_task_vector_at_coef(
                task_vector, pretrained_checkpoint, args, 1.0, eval_masks, svd_dict
            )
            print(
                "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                    round(info["loaded_mask"]["avg_normalized_top1"] * 100, 2),
                    round(info["loaded_mask"]["avg_top1"] * 100, 2),
                )
            )
        else:
            for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
                print("\n" * 2)
                print("=" * 43, f"tall_mask_lambda = {tall_mask_lambda:.2f}", "=" * 43)
                info[tall_mask_lambda] = evaluate_task_vector_at_coef(
                    task_vector,
                    pretrained_checkpoint,
                    args,
                    1.0,
                    eval_masks[tall_mask_lambda],
                    svd_dict,
                )
                print(
                    "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                        round(info[tall_mask_lambda]["avg_normalized_top1"] * 100, 2),
                        round(info[tall_mask_lambda]["avg_top1"] * 100, 2),
                    )
                )
    else:
        for scaling_coef in scaling_coef_range:
            print("\n" * 2)
            print("=" * 43, f"alpha = {scaling_coef:.2f}", "=" * 43)
            info[scaling_coef] = evaluate_task_vector_at_coef(
                task_vector,
                pretrained_checkpoint,
                args,
                scaling_coef,
                eval_masks,
                svd_dict,
            )
            print(
                "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                    round(info[scaling_coef]["avg_normalized_top1"] * 100, 2),
                    round(info[scaling_coef]["avg_top1"] * 100, 2),
                )
            )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results
