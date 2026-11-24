for method in [
    # "sum",
    # "average",
    "TSVM",
    # "ties",
    # "consensus",
    # "consensus apply_lines=True",
    # "STAR"
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
            for BS in [4, 8, 16, 32, 64, 128, 256, 512, 1024]: # Batch size experiment
            # for BS in [64]:
                for proxy_dataset_mode in [
                    # "shared", # All use the same proxy dataset OpenImages default (not used in paper exps)
                    "per_task" #  Each task uses its own validation dataset as proxy
                ]:
                    for seed in [1, 2, 3, 4, 5]: # For BS experiment
                    # for seed in [0]:
                        # for num_batches_tas in [-1]: # -1 means use all proxy data
                        for num_batches_tas in [1]: # For BS experiment
                            cmd = "export PYTHONUNBUFFERED=1; "
                            cmd += "export OMP_NUM_THREADS=1; "
                            cmd += "export MKL_NUM_THREADS=1; "
                            cmd += "export OPENBLAS_NUM_THREADS=1; "
                            cmd += "export NUMEXPR_NUM_THREADS=1; "
                            cmd += f"python -u main.py model={model} method={method} num_tasks={num_tasks} "
                            cmd += "compute_tas=True "
                            cmd += f"proxy_dataset_mode={proxy_dataset_mode} "
                            cmd += f"num_batches_tas={num_batches_tas} tas_seed={seed} "
                            cmd += f"tas_batch_size={BS} skip_eval=True"
                            if method == "ties":
                                for k in [20, 40, 80]:
                                    print(cmd + f" method.k={k}")
                            elif method == "STAR":
                                for eta in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
                                    print(cmd + f" method.eta={eta}")
                            else:
                                print(cmd)
