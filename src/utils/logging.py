# Adapted from:
#   https://github.com/nik-dim/tall_masks/blob/master/src/utils/logging.py (MIT)
#   https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/utils/logging.py (Apache-2.0)
# The optional-wandb handling and the result-naming scheme are ours.
import os
import json
import hydra
from argparse import Namespace
from pprint import pprint

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Weights & Biases logging: optional, opt-in, disabled by default.
#
# `wandb` is imported lazily inside initialize_wandb() and only when the user
# explicitly enables it (`wandb.enabled=true`), so the code runs fine without
# wandb installed. This module is the only place in the repo that touches wandb.
# ---------------------------------------------------------------------------

# Holds the imported `wandb` module once a run is live, and stays None
# otherwise -- which makes every helper below a cheap no-op.
_WANDB = None

_WANDB_DEFAULTS = {
    "enabled": False,
    "project": "task-vectors",
    "entity": None,
    "group": None,
    "mode": "online",
    "dir": "logs/",
}


def _is_rank_zero():
    """Rank-0 check that is also valid outside a DDP run (e.g. main.py)."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _wandb_settings(args):
    """
    Read the wandb settings from the Hydra config (`cfg.wandb.*`). Missing keys
    fall back to the defaults above, so a config without a `wandb:` section --
    or one that only overrides a key or two -- cannot raise here.
    """
    section = getattr(args, "wandb", None)
    if section is None:
        return dict(_WANDB_DEFAULTS)
    if OmegaConf.is_config(section):
        section = OmegaConf.to_container(section, resolve=True)
    return {**_WANDB_DEFAULTS, **dict(section)}


def initialize_wandb(args):
    """
    Start a wandb run if the user opted in, otherwise do nothing.

    Safe to call unconditionally: when wandb is disabled (the default) this is
    a no-op, and every later wandb_log()/wandb_finish() call is a no-op too.
    """
    global _WANDB

    settings = _wandb_settings(args)
    if not settings["enabled"] or not _is_rank_zero():
        return None

    try:
        import wandb  # imported lazily: wandb is an optional dependency
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested but the `wandb` package is not "
            "installed. Install it with `pip install wandb==0.16.6`, or turn "
            "logging off again with `wandb.enabled=false`."
        ) from exc

    if settings["dir"]:
        os.makedirs(settings["dir"], exist_ok=True)

    config = (
        OmegaConf.to_container(args, resolve=True)
        if OmegaConf.is_config(args)
        else vars(args)
    )
    wandb.init(
        config=config,
        project=settings["project"],
        entity=settings["entity"],
        group=settings["group"],
        mode=settings["mode"],
        dir=settings["dir"],
    )
    _WANDB = wandb
    print(f"[wandb] logging to project '{settings['project']}' (mode={settings['mode']}).")
    return wandb


def wandb_log(dictionary: dict) -> None:
    """Log a dict to wandb. No-op unless a run was started on the main process."""
    if _WANDB is None or not _is_rank_zero():
        return
    _WANDB.log(dictionary)


def wandb_finish(quiet: bool = True) -> None:
    """Close the current wandb run. No-op if wandb logging is disabled."""
    global _WANDB
    if _WANDB is None:
        return
    _WANDB.finish(quiet=quiet)
    _WANDB = None


def _sanitize(val):
    return str(val).replace("/", "_").replace(" ", "").replace(".", "p")


def _proxy_dataset_suffix(args, include_eval_tas_params: bool = True):
    mode = getattr(args, "proxy_dataset_mode", "per_task")
    mode_s = _sanitize(mode)
    if mode == "shared":
        ds = _sanitize(getattr(args, "proxy_shared_dataset", "na"))
        return f"proxy={mode_s}-{ds}"
    if mode == "per_task_map":
        mapping = getattr(args, "proxy_task_to_dataset", {}) or {}
        unique_vals = sorted({str(v) for v in mapping.values()})
        vals_s = "-".join(_sanitize(v) for v in unique_vals) if unique_vals else "none"
        return f"proxy={mode_s}-{vals_s}"
    return f"proxy={mode_s}"


def _method_suffix(args):
    if args.method.name in [
        "adamerging_task",
        "adamerging_layer",
        "tasmerging_task",
        "tasmerging_layer",
    ]:
        bs = _sanitize(args.method.batch_size)
        lr = _sanitize(args.method.lr)
        linit = _sanitize(args.method.lambda_init)
        seed = _sanitize(getattr(args.method, "data_seed", "na"))
        extra = f"bs={bs}_lr={lr}_linit={linit}_seed={seed}"
        proxy_ds = _proxy_dataset_suffix(args)
        if args.method.name in ["tasmerging_task", "tasmerging_layer"]:
            nb = _sanitize(getattr(args, "num_batches_tas", "na"))
            extra += f"_nb={nb}_{proxy_ds}"
        else:
            extra += f"_{proxy_ds}"
        return extra
    return ""


def log_results(final_results, args):
    if args.method.name == "tall_mask":
        mask_suffix = f"tall_mask_ties" if args.method.use_ties else f"tall_mask_ta"
    elif args.method.name == "mag_masking":
        mask_suffix = "mag_mask"
    elif args.method.name == "consensus":
        mask_suffix = (
            f"k_{args.method.prun_thre_k}_ties"
            if args.method.use_ties
            else f"k_{args.method.prun_thre_k}_ta"
        )
    elif args.method.name == "TSVM":
        mask_suffix = "svd"
    else:
        mask_suffix = ""

    results_root = f"results/{args.save_subfolder}"
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    method_suffix = _method_suffix(args)
    if args.compute_tas:
        drop_eval = args.method.name in [
            "adamerging_task",
            "adamerging_layer",
            "tasmerging_task",
            "tasmerging_layer",
        ]
        proxy_suffix = _proxy_dataset_suffix(args, include_eval_tas_params=not drop_eval)
        if not drop_eval:
            if args.tas_seed is not None:
                proxy_suffix = f"{proxy_suffix}_tasseed={_sanitize(args.tas_seed)}"
            if args.tas_batch_size is not None:
                proxy_suffix = f"{proxy_suffix}_tasbs={_sanitize(args.tas_batch_size)}"
        if args.num_batches_tas is not None and args.num_batches_tas != -1:
            proxy_suffix = f"{proxy_suffix}_tasnb={_sanitize(args.num_batches_tas)}"
        if getattr(args, "tas_metric", None):
            proxy_suffix = f"{proxy_suffix}_tasmetric={_sanitize(args.tas_metric)}"
        if method_suffix:
            if proxy_suffix not in method_suffix:
                method_suffix = f"{method_suffix}_{proxy_suffix}"
        else:
            method_suffix = proxy_suffix
    if args.apply_lines:
        if method_suffix:
            if "lines" not in method_suffix:
                method_suffix = f"{method_suffix}_lines"
        else:
            method_suffix = "lines"
    if args.num_tasks < 8:
        cleaned_datasets = (
            str(args.DATASETS)
            .translate(str.maketrans("", "", '[]",'))
            .replace(" ", "_")
            .replace("'", "")
        )
        save_file = f"{results_root}/{args.model}_{args.num_tasks}tasks_{args.method.full_name}"
        if method_suffix:
            save_file += f"_{method_suffix}"
        save_file += f"_nonlinear_additions_{mask_suffix}_{cleaned_datasets}.json"
    else:
        save_file = f"{results_root}/{args.model}_{args.num_tasks}tasks_{args.method.full_name}"
        if method_suffix:
            save_file += f"_{method_suffix}"
        save_file += f"_nonlinear_additions_{mask_suffix}.json"

    with open(save_file, "w") as f:
        json.dump(final_results, f, indent=4)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_save_file = f"{args.method.full_name}_nonlinear_additions.json"
    hydra_save_file = os.path.join(hydra_dir, hydra_save_file)
    json.dump(final_results, open(hydra_save_file, "w"), indent=4)

    print("saved results to: ", save_file)
    print("saved results to: ", hydra_save_file)
