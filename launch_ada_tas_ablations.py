import itertools


def main():
    methods = [
        "adamerging_task",
        "adamerging_layer",
        # "tasmerging_task",
        # "tasmerging_layer"
    ]
    models = [
        # "ViT-B-32",
        # "ViT-B-16",
        "ViT-L-14"
    ]
    num_tasks_list = [
        # 8,
        14,
        20
    ]

    batch_sizes = [16]
    lrs = [1e-3]
    lambda_inits = [0.3]
    data_seeds = [123]
    num_batches_tas = [1000]
    tas_metric = ["cosine"]

    base = (
        "export PYTHONUNBUFFERED=1; "
        "export OMP_NUM_THREADS=1; "
        "export MKL_NUM_THREADS=1; "
        "export OPENBLAS_NUM_THREADS=1; "
        "export NUMEXPR_NUM_THREADS=1; "
    )

    for method, model, num_tasks, bs, lr, linit, seed, tas_m in itertools.product(
        methods, models, num_tasks_list, batch_sizes, lrs, lambda_inits, data_seeds, tas_metric
    ):
        cmd = base + f"python -u main.py model={model} method={method} num_tasks={num_tasks} "
        cmd += f" method.batch_size={bs} method.lr={lr} method.lambda_init={linit} method.data_seed={seed}"
        if method.startswith("tasmerging"):
            for nb in num_batches_tas:
                print(cmd + f" num_batches_tas={nb} tas_metric={tas_m} compute_tas=True")
            continue
        print(cmd)


if __name__ == "__main__":
    main()
