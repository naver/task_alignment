for model in [
    "ViT-B-32",
    "ViT-B-16",
    "ViT-L-14"
]:    
    for num_tasks in [
        8,
        14,
        20
    ]:
        for method in [
            # "sum",
            # "average",
            # "TSVM",
            # "ties",
            # "consensus",
            # "consensus apply_lines=True",
            # "STAR",
            # "iso_cts",
            "iso_c"
        ]:
            for BS in [128]:
                for proxy_dataset_mode in [
                    "per_task", #  Each task uses its own validation dataset as proxy
                    # "shared", # All use the same proxy dataset OpenImages default (not used in paper exps)
                ]:
                    for seed in [0]:
                        for num_batches_tas in [1]:
                            cmd = "export PYTHONUNBUFFERED=1; "
                            cmd += "export OMP_NUM_THREADS=1; "
                            cmd += "export MKL_NUM_THREADS=1; "
                            cmd += "export OPENBLAS_NUM_THREADS=1; "
                            cmd += "export NUMEXPR_NUM_THREADS=1; "
                            cmd += f"python -u main.py model={model} method={method} num_tasks={num_tasks} "
                            cmd += "compute_tas=True "
                            cmd += f"proxy_dataset_mode={proxy_dataset_mode} "
                            cmd += f"num_batches_tas={num_batches_tas} tas_seed={seed} tas_batch_size={BS} "
                            if method == "ties":
                                for k in [20, 40, 80]:
                                    print(cmd + f" method.k={k}")
                            elif method == "STAR":
                                for eta in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
                                    print(cmd + f" method.eta={eta}")
                            else:
                                print(cmd)
