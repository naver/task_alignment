# Adapted from:
#   https://github.com/nik-dim/tall_masks/blob/master/src/eval/eval_utils.py (MIT)
#   https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/eval/eval_utils.py (Apache-2.0)
import json
import os
import numpy as np

import torch
from omegaconf import open_dict

from src.eval.eval import (
    evaluate_task_vector,
    evaluate_task_vector_at_coef,
    evaluate_image_encoder,
    add_normalized_accuracy,
)
from src.utils.adamerging import optimize_adamerging, optimize_tasmerging
from src.utils.tallmask_utils import find_optimal_mask
from src.utils.utils import find_optimal_coef
from src.utils.logging import log_results
from src.utils.variables_and_paths import (
    get_finetuned_path,
    get_zeroshot_path,
    get_single_task_accuracies_path,
)

OPTIMIZED_IMAGE_ENCODER_METHODS = [
    "adamerging_task",
    "adamerging_layer",
    "tasmerging_task",
    "tasmerging_layer",
]


def _summarize_eval_metrics(raw_metrics, args):
    raw_metrics = add_normalized_accuracy(raw_metrics, args)
    raw_metrics["avg_normalized_top1"] = np.mean(
        [raw_metrics[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    raw_metrics["avg_top1"] = np.mean(
        [raw_metrics[dataset + ":top1"] for dataset in args.eval_datasets]
    )
    compute_tas_now = args.compute_tas and not getattr(args, "skip_tas", False)
    if compute_tas_now:
        raw_metrics["avg_tas"] = np.mean(
            [raw_metrics[dataset + ":tas"] for dataset in args.eval_datasets]
        )
    return raw_metrics


def _tas_selection_enabled(args) -> bool:
    return bool(args.compute_tas and not getattr(args, "skip_tas", False))


def _select_best_result_entry(results, metric: str, minimize: bool):
    best_key = None
    best_value = None

    for key, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if best_value is None or (value < best_value if minimize else value > best_value):
            best_key = key
            best_value = value

    if best_key is None:
        return None, None
    return best_key, results[best_key]


def _select_tall_mask_by_metric(val_metrics, eval_masks, args, metric_suffix: str, minimize: bool):
    best_masks_for_test = {}
    best_val_metrics = {}
    selected_lambdas = {}
    selected_values = []

    for dataset in args.DATASETS:
        metric_name = f"{dataset}Val:{metric_suffix}"
        best_lambda = None
        best_value = None

        for lambda_key, metrics in val_metrics.items():
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(metric_name)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            if best_value is None or (value < best_value if minimize else value > best_value):
                best_lambda = lambda_key
                best_value = value

        if best_lambda is None:
            return None, None, None

        best_lambda = float(best_lambda)
        selected_lambdas[dataset] = best_lambda
        best_masks_for_test[dataset] = eval_masks[best_lambda][dataset]
        best_val_metrics[metric_name] = best_value
        selected_values.append(best_value)

    if selected_values:
        best_val_metrics[f"avg_{metric_suffix}"] = float(np.mean(selected_values))

    return best_masks_for_test, best_val_metrics, selected_lambdas


def _evaluate_test_metrics(
    args,
    task_vector,
    pretrained_checkpoint,
    *,
    test_model=None,
    scaling_coef=None,
    eval_masks=None,
    svd_dict=None,
):
    if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
        if args.skip_eval:
            return {"avg_normalized_top1": float("nan"), "avg_top1": float("nan")}
        test_metrics = evaluate_image_encoder(test_model, args)
        return _summarize_eval_metrics(test_metrics, args)

    if args.skip_eval:
        return {"avg_normalized_top1": float("nan"), "avg_top1": float("nan")}

    return evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(1.0 if scaling_coef is None else scaling_coef),
        eval_masks=eval_masks,
        svd_dict=svd_dict,
    )


def _build_test_tas_meta(selected_from_val_key):
    meta = {
        "selected_from_val_key": selected_from_val_key,
        "script": "main.py",
    }

    if isinstance(selected_from_val_key, dict):
        meta["selection_metric"] = "per_dataset_tas"
    else:
        meta["selection_metric"] = "avg_tas"
        try:
            meta["selected_coef"] = float(selected_from_val_key)
        except (TypeError, ValueError):
            pass

    return meta


def perform_eval_with_merged_vector(args, task_vector, eval_masks=None, svd_dict=None):
    if args.method.name not in OPTIMIZED_IMAGE_ENCODER_METHODS:
        assert task_vector is not None, "Task vector should not be None."
    if eval_masks is not None:
        assert args.method.name in ["tall_mask", "mag_masking"]
    if svd_dict is not None:
        assert args.method.name == "TSVC"
    with open_dict(args):
        args.save_dir = os.path.join(args.model_location, args.model)

    ft_accuracies_path = get_single_task_accuracies_path(args.model)
    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)

    with open_dict(args):
        with open(ft_accuracies_path) as f:
            args.finetuning_accuracies = json.load(f)
        args.eval_datasets = args.DATASETS_VAL
        args.control_dataset = None

    if args.method.name in ["adamerging_task", "adamerging_layer"]:
        layerwise = args.method.name == "adamerging_layer"
        model, task_vectors, lambdas_by_param, info = optimize_adamerging(
            args, layerwise=layerwise
        )
    elif args.method.name in ["tasmerging_task", "tasmerging_layer"]:
        layerwise = args.method.name == "tasmerging_layer"
        model, task_vectors, lambdas_by_param, info = optimize_tasmerging(
            args, layerwise=layerwise
        )

    if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
        # Even in `skip_eval` mode, we still run this loop so TAS debug can sample/save images.
        val_raw = evaluate_image_encoder(model.get_image_encoder(), args)
        val_raw = _summarize_eval_metrics(val_raw, args)

        val_metrics = {"final": val_raw}
        best_val_metrics = val_raw
        optimal_coef = None
    else:
        # Even in `skip_eval` mode, we still run this loop so TAS debug can sample/save images.
        val_metrics = evaluate_task_vector(
            task_vector,
            pretrained_checkpoint,
            args,
            eval_masks=eval_masks,
            svd_dict=svd_dict,
        )

    best_val_tas_metrics = None
    test_tas_selection = None
    best_masks_for_test_tas = None
    optimal_tas_coef = None

    if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
        best_val_metrics = val_metrics["final"]
        test_model = model.get_image_encoder()
    elif args.method.name == "tall_mask":
        if args.method.load_mask:
            best_masks_for_test = eval_masks
            best_val_metrics = val_metrics
        else:
            # find the best mask individually for each task based on validation accuracy
            best_masks_for_test, best_val_metrics = find_optimal_mask(
                val_metrics, eval_masks, args, save_masks=True
            )
    elif args.method.name == "mag_masking":
        best_masks_for_test = eval_masks
        best_val_metrics = val_metrics[1.0]
    elif args.method.name == "TSVC":
        best_val_metrics = val_metrics
        optimal_coef = 1.0
    elif args.method.name in ["dummy"]:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        if args.skip_eval:
            optimal_coef = 1.0 if 1.0 in val_metrics else sorted(val_metrics.keys())[0]
            best_val_metrics = val_metrics[optimal_coef]
        else:
            optimal_coef = find_optimal_coef(
                val_metrics, metric=f"{args.eval_datasets[0]}:top1", minimize=False
            )
            if optimal_coef is None:
                optimal_coef = 1.0 if 1.0 in val_metrics else sorted(val_metrics.keys())[0]
            best_val_metrics = val_metrics[optimal_coef]
    else:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        if args.skip_eval:
            optimal_coef = 1.0 if 1.0 in val_metrics else sorted(val_metrics.keys())[0]
            best_val_metrics = val_metrics[optimal_coef]
        else:
            optimal_coef = find_optimal_coef(
                val_metrics, metric="avg_normalized_top1", minimize=False
            )
            if optimal_coef is None:
                optimal_coef = 1.0 if 1.0 in val_metrics else sorted(val_metrics.keys())[0]
            best_val_metrics = val_metrics[optimal_coef]

    if _tas_selection_enabled(args):
        if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
            best_val_tas_metrics = val_metrics["final"]
            test_tas_selection = "final"
        elif args.method.name == "tall_mask":
            if args.method.load_mask:
                best_masks_for_test_tas = eval_masks
                best_val_tas_metrics = (
                    val_metrics["loaded_mask"] if isinstance(val_metrics, dict) and "loaded_mask" in val_metrics else val_metrics
                )
                test_tas_selection = "loaded_mask"
            else:
                (
                    best_masks_for_test_tas,
                    best_val_tas_metrics,
                    test_tas_selection,
                ) = _select_tall_mask_by_metric(
                    val_metrics,
                    eval_masks,
                    args,
                    metric_suffix="tas",
                    minimize=True,
                )
        elif args.method.name == "mag_masking":
            best_masks_for_test_tas = eval_masks
            best_val_tas_metrics = val_metrics[1.0]
            test_tas_selection = 1.0
        elif args.method.name == "TSVC":
            best_val_tas_metrics = val_metrics
            optimal_tas_coef = 1.0
            test_tas_selection = optimal_tas_coef
        else:
            optimal_tas_coef, best_val_tas_metrics = _select_best_result_entry(
                val_metrics, metric="avg_tas", minimize=True
            )
            test_tas_selection = optimal_tas_coef

    print("\n" * 2)

    # Evaluate on the test set with the optimal coefficients / masks
    with open_dict(args):
        args.eval_datasets = args.DATASETS

    if args.method.name in ["tall_mask", "mag_masking"]:
        test_metrics = _evaluate_test_metrics(
            args,
            task_vector,
            pretrained_checkpoint,
            eval_masks=best_masks_for_test,
            svd_dict=svd_dict,
        )
    else:
        test_metrics = _evaluate_test_metrics(
            args,
            task_vector,
            pretrained_checkpoint,
            test_model=test_model if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS else None,
            scaling_coef=optimal_coef,
            svd_dict=svd_dict,
        )

    test_tas_metrics = None
    if _tas_selection_enabled(args) and best_val_tas_metrics is not None:
        if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
            test_tas_metrics = test_metrics
        elif args.method.name == "tall_mask":
            test_tas_metrics = _evaluate_test_metrics(
                args,
                task_vector,
                pretrained_checkpoint,
                eval_masks=best_masks_for_test_tas,
                svd_dict=svd_dict,
            )
        elif args.method.name == "mag_masking":
            test_tas_metrics = test_metrics
        elif optimal_tas_coef == optimal_coef:
            test_tas_metrics = test_metrics
        else:
            test_tas_metrics = _evaluate_test_metrics(
                args,
                task_vector,
                pretrained_checkpoint,
                scaling_coef=optimal_tas_coef,
                svd_dict=svd_dict,
            )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    final_results = {
        "test": test_metrics,
        "val": val_metrics,
        "val_best": best_val_metrics,
    }
    if test_tas_metrics is not None:
        final_results["val_best_tas"] = best_val_tas_metrics
        final_results["test_tas"] = test_tas_metrics
        final_results["test_tas_meta"] = _build_test_tas_meta(test_tas_selection)
    if args.method.name in OPTIMIZED_IMAGE_ENCODER_METHODS:
        final_results["lambdas"] = lambdas_by_param
        final_results["optimization"] = info

    log_results(final_results, args)

    return final_results
