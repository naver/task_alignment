# Adapted from:
#   https://github.com/nik-dim/tall_masks/blob/master/main.py (MIT)
import os
from pprint import pprint

import hydra

from omegaconf import DictConfig, OmegaConf

from src.eval.aggregation import create_task_vector
from src.eval.eval_utils import perform_eval_with_merged_vector
from src.utils.logging import initialize_wandb, wandb_finish, wandb_log
from src.utils.variables_and_paths import ALL_DATASETS


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[: cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    if cfg.apply_lines and cfg.method.name not in [
        "sum",
        "ties",
        "consensus",
        "single_task",
        "zeroshot",
    ]:
        raise ValueError(
            "apply_lines is only supported for methods "
            "sum/ties/consensus/single_task/zeroshot "
            f"(got {cfg.method.name})."
        )

    print(cfg.method.full_name)
    print()
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)

    # Optional wandb logging (opt-in via `wandb.enabled=true`); no-op otherwise.
    initialize_wandb(cfg)

    # create final task vector
    task_vector_dict, eval_masks, svd_dict = create_task_vector(cfg)
    print("*" * 100)
    print("*" * 37, "Created task vector dict", "*" * 37)
    print("*" * 100)
    print("\n" * 3)

    # perform evaluation and log results
    print("*" * 100)
    print("*" * 39, "Starting Evaluation.", "*" * 39)
    print("*" * 100)
    additive_accuracies = perform_eval_with_merged_vector(
        cfg, task_vector_dict, eval_masks, svd_dict
    )
    pprint(additive_accuracies, width=1)
    wandb_log(
        {
            f"test/{k}": v
            for k, v in additive_accuracies["test"].items()
            if isinstance(v, (int, float))
        }
    )
    wandb_finish()


if __name__ == "__main__":
    my_app()
