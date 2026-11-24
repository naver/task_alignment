for method in [
    "sum",
]:
    for model in [
        "ViT-B-32",
        # "ViT-B-16",
        # "ViT-L-14"
    ]:
        for num_tasks in [
            8,
            # 14,
            # 20
        ]:
            for tas_metric in [
                "l1",
                "l2",
                "cosine"
            ]:  # TAS distance ablation
                for BS in [128]:
                    for proxy_dataset_mode in [
                        # "shared",  # All use the same proxy dataset OpenImages default (not used in paper exps)
                        "per_task" # Each task uses its own validation dataset as proxy
                    ]:
                        for seed in [1, 2, 3, 4, 5]:
                            # for num_batches_tas in [-1]:  # -1 means use all proxy data
                            for num_batches_tas in [1]:
                                cmd = "export PYTHONUNBUFFERED=1; "
                                cmd += "export OMP_NUM_THREADS=1; "
                                cmd += "export MKL_NUM_THREADS=1; "
                                cmd += "export OPENBLAS_NUM_THREADS=1; "
                                cmd += "export NUMEXPR_NUM_THREADS=1; "
                                cmd += (
                                    f"python -u main.py model={model} method={method} "
                                    f"num_tasks={num_tasks} "
                                )
                                cmd += "compute_tas=True "
                                cmd += f"proxy_dataset_mode={proxy_dataset_mode} "
                                cmd += (
                                    f"num_batches_tas={num_batches_tas} tas_seed={seed} "
                                )
                                cmd += f"tas_batch_size={BS} tas_metric={tas_metric} "
                                cmd += "skip_eval=True"
                                print(cmd)
