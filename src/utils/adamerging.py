import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import dataset_by_name
from src.eval.eval import eval_single_dataset
from src.models.heads import zeroshot_head_for
from src.models.modeling import ImageEncoder
from src.utils.tas_utils import resolve_proxy_dataset_for_task, tas_from_cached, get_ft_encoder
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path


def _del_attr(obj, names: List[str]) -> None:
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])


def _set_attr(obj, names: List[str], val) -> None:
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)


def _make_functional(mod) -> List[str]:
    names = []
    for name, _ in list(mod.named_parameters()):
        _del_attr(mod, name.split("."))
        names.append(name)
    return names


def _load_weights(mod, names: Iterable[str], params: Iterable[torch.Tensor]) -> None:
    for name, p in zip(names, params):
        _set_attr(mod, name.split("."), p)


def _softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def _get_method_cfg(args):
    return getattr(args, "method", args)


def _get_method_attr(method_cfg, name, default):
    return getattr(method_cfg, name, default)


def _build_task_vectors(
    args, datasets: List[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)
    ptm_state = torch.load(pretrained_checkpoint, map_location="cpu")

    task_vectors = {}
    for dataset in datasets:
        ft_path = get_finetuned_path(args.model_location, dataset, model=args.model)
        ft_state = torch.load(ft_path, map_location="cpu")
        tv = {}
        for key, ptm_val in ptm_state.items():
            if ptm_val.dtype in [torch.int64, torch.uint8]:
                continue
            tv[key] = ft_state[key] - ptm_val
        task_vectors[dataset] = tv

    return ptm_state, task_vectors


def _build_param_tensors(
    param_names: List[str],
    ptm_state: Dict[str, torch.Tensor],
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
    ptm_params = [ptm_state[name].detach().cpu() for name in param_names]
    task_params = []
    for dataset in task_vectors:
        tv = task_vectors[dataset]
        task_params.append([tv[name].detach().cpu() for name in param_names])
    return ptm_params, task_params


def _build_val_loaders(args, image_encoder, datasets: List[str], batch_size: int):
    loaders = {}
    for dataset_name in datasets:
        dataset = dataset_by_name(
            dataset_name,
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=batch_size,
        )
        base_loader = dataset.test_loader
        loader = torch.utils.data.DataLoader(
            base_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=True,
        )
        loaders[dataset_name] = loader
    return loaders


class AdaMergingModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        param_names: List[str],
        ptm_params: List[torch.Tensor],
        task_params: List[List[torch.Tensor]],
        datasets: List[str],
        args,
        layerwise: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_names = param_names
        self.ptm_params = ptm_params
        self.task_params = task_params
        self.datasets = datasets
        self.args = args
        self.layerwise = layerwise

        method_cfg = _get_method_cfg(args)
        init_val = float(_get_method_attr(method_cfg, "lambda_init", 0.3))
        self.lambda_min = float(_get_method_attr(method_cfg, "lambda_min", 0.0))
        self.lambda_max = float(_get_method_attr(method_cfg, "lambda_max", 1.0))

        if layerwise:
            init = torch.ones(len(ptm_params), len(task_params)) * init_val
        else:
            init = torch.ones(1, len(task_params)) * init_val
        self.lambdas_raw = torch.nn.Parameter(init)

        self.classifier = {}
        for dataset_name in datasets:
            head = zeroshot_head_for(args, dataset_name)
            self.add_module(f"classifier_{dataset_name}", head.to(args.device))
            self.classifier[dataset_name] = f"classifier_{dataset_name}"

    def lambdas(self) -> torch.Tensor:
        return torch.clamp(self.lambdas_raw, min=self.lambda_min, max=self.lambda_max)

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def zeroshot_head_for(self, dataset_name: str):
        return getattr(self, self.classifier[dataset_name])

    def get_image_encoder(self):
        params = self._merge_params()
        _load_weights(self.model, self.param_names, params)
        self.model = self.model.to(self.args.device)
        return self.model

    def _merge_params(self) -> List[torch.Tensor]:
        lambdas = self.lambdas().to(self.args.device)
        merged = []
        num_tasks = len(self.task_params)

        for param_idx, ptm in enumerate(self.ptm_params):
            merged_param = ptm.to(self.args.device)
            if self.layerwise:
                coeffs = lambdas[param_idx]
            else:
                coeffs = lambdas[0]

            for t in range(num_tasks):
                tv = self.task_params[t][param_idx].to(self.args.device)
                merged_param = merged_param + coeffs[t] * tv
            merged.append(merged_param)
        return merged

    def forward(self, images, dataset_name: str):
        params = self._merge_params()
        _load_weights(self.model, self.param_names, params)
        self.model = self.model.to(self.args.device)
        features = self.model(images)
        head = self.zeroshot_head_for(dataset_name)
        return head(features)


def _summarize_lambdas(param_names, lambdas, datasets, layerwise):
    lambdas_by_param = {}
    if layerwise:
        for param_idx, name in enumerate(param_names):
            lambdas_by_param[name] = {
                ds: float(lambdas[param_idx, t]) for t, ds in enumerate(datasets)
            }
    else:
        for name in param_names:
            lambdas_by_param[name] = {
                ds: float(lambdas[0, t]) for t, ds in enumerate(datasets)
            }
    return lambdas_by_param


def _summarize_lambdas_by_task(lambdas, datasets, layerwise):
    if layerwise:
        task_means = lambdas.mean(dim=0)
        return {ds: float(task_means[t]) for t, ds in enumerate(datasets)}
    return {ds: float(lambdas[0, t]) for t, ds in enumerate(datasets)}


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
    suffix = f"proxy={mode_s}"
    if include_eval_tas_params:
        if getattr(args, "tas_seed", None) is not None:
            suffix = f"{suffix}_tasseed={_sanitize(args.tas_seed)}"
        if getattr(args, "tas_batch_size", None) is not None:
            suffix = f"{suffix}_tasbs={_sanitize(args.tas_batch_size)}"
    if getattr(args, "num_batches_tas", None) not in (None, -1):
        suffix = f"{suffix}_tasnb={_sanitize(args.num_batches_tas)}"
    if getattr(args, "tas_metric", None):
        suffix = f"{suffix}_tasmetric={_sanitize(args.tas_metric)}"
    return suffix


def _get_optimization_log_path(args, method_name):
    subfolder = args.save_subfolder if args.save_subfolder != "" else "default"
    out_dir = os.path.join("results", subfolder, "optimization_logs")
    os.makedirs(out_dir, exist_ok=True)
    if method_name in [
        "adamerging_task",
        "adamerging_layer",
        "tasmerging_task",
        "tasmerging_layer",
    ]:
        bs = str(args.method.batch_size).replace(".", "p")
        lr = str(args.method.lr).replace(".", "p")
        linit = str(args.method.lambda_init).replace(".", "p")
        seed = str(getattr(args.method, "data_seed", "na")).replace(".", "p")
        method_name = f"{method_name}_bs={bs}_lr={lr}_linit={linit}_seed={seed}"
        proxy_ds = _proxy_dataset_suffix(args, include_eval_tas_params=False)
        if method_name.startswith("tasmerging"):
            nb = str(getattr(args, "num_batches_tas", "na")).replace(".", "p")
            method_name = f"{method_name}_nb={nb}_{proxy_ds}"
        else:
            method_name = f"{method_name}_{proxy_ds}"
    fname = f"{args.model}_{args.num_tasks}tasks_{method_name}_optimization.jsonl"
    return os.path.join(out_dir, fname)


def _run_adamerging_optimization(args, layerwise, loss_fn):
    method_cfg = _get_method_cfg(args)
    num_iters = int(_get_method_attr(method_cfg, "num_iters", 500))
    eval_every = int(_get_method_attr(method_cfg, "eval_every", 100))
    log_every = int(_get_method_attr(method_cfg, "log_every", 10))
    lr = float(_get_method_attr(method_cfg, "lr", 1e-3))
    batch_size = int(_get_method_attr(method_cfg, "batch_size", args.batch_size))
    data_seed = int(_get_method_attr(method_cfg, "data_seed", 123))

    print("Building task vectors...")
    ptm_state, task_vectors = _build_task_vectors(args, args.DATASETS_VAL)

    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)
    image_encoder = ImageEncoder.load(args.model, pretrained_checkpoint)
    image_encoder = image_encoder.to(args.device)

    param_names = _make_functional(image_encoder)
    ptm_params, task_params = _build_param_tensors(param_names, ptm_state, task_vectors)

    print("Building AdaMerging model...")
    model = AdaMergingModel(
        image_encoder,
        param_names,
        ptm_params,
        task_params,
        args.DATASETS_VAL,
        args,
        layerwise=layerwise,
    ).to(args.device)

    print("Building dataloaders...")
    proxy_dataset_for_task = {
        task_name: resolve_proxy_dataset_for_task(task_name, args)
        for task_name in args.DATASETS_VAL
    }

    proxy_dataset_cache = {}
    loader_fns = {}
    for task_name in args.DATASETS_VAL:
        proxy_dataset_name = proxy_dataset_for_task[task_name]
        if proxy_dataset_name not in proxy_dataset_cache:
            dataset = dataset_by_name(
                proxy_dataset_name,
                image_encoder.val_preprocess,
                location=args.data_location,
                batch_size=batch_size,
            )
            proxy_dataset_cache[proxy_dataset_name] = dataset.test_loader.dataset

        base_dataset = proxy_dataset_cache[proxy_dataset_name]
        loader_fns[task_name] = lambda ds=base_dataset: _make_deterministic_loader(
            ds, batch_size, data_seed, 0
        )
    loaders_iter = {name: iter(loader_fns[name]()) for name in args.DATASETS_VAL}

    optimizer = torch.optim.Adam(
        model.collect_trainable_params(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0
    )

    loss_history = []
    val_history = []
    log_path = _get_optimization_log_path(args, method_cfg.name)
    log_f = open(log_path, "w", encoding="utf-8")

    model.train()
    start_time = time.time()
    for iteration in range(num_iters):
        losses = defaultdict(float)
        time_iter_start = time.time()
        for dataset_name in args.DATASETS_VAL:
            try:
                batch = next(loaders_iter[dataset_name])
            except StopIteration:
                loaders_iter[dataset_name] = iter(loader_fns[dataset_name]())
                try:
                    batch = next(loaders_iter[dataset_name])
                except StopIteration as exc:
                    raise RuntimeError(
                        "No full batches available for AdaMerging optimization "
                        f"(dataset={dataset_name}, batch_size={batch_size}). "
                        "This likely means the proxy dataset has fewer samples than "
                        "batch_size after dropping the last incomplete batch."
                    ) from exc

            loss = loss_fn(model, batch, dataset_name, None)
            loss.backward()
            losses[dataset_name] += float(loss.detach().cpu())

        optimizer.step()
        optimizer.zero_grad()
        eta = time.time() - time_iter_start
        print(f"Iteration: {iteration}, Loss: {losses}, ETA: {eta:.2f} sec")

        loss_history.append({"iter": iteration, **losses})

        lambdas = model.lambdas().detach().cpu()
        lambda_by_task = _summarize_lambdas_by_task(
            lambdas, args.DATASETS_VAL, layerwise
        )
        avg_loss = sum(losses.values()) / max(len(losses), 1)
        log_f.write(
            json.dumps(
                {
                    "iter": iteration,
                    "avg_loss": avg_loss,
                    "losses": dict(losses),
                    "lambdas": lambda_by_task,
                    "proxy_datasets": proxy_dataset_for_task,
                }
            )
            + "\n"
        )
        log_f.flush()

        if log_every > 0 and (iteration + 1) % log_every == 0:
            loss_str = ", ".join(
                f"{ds}={losses[ds]:.4f}" for ds in args.DATASETS_VAL
            )
            lambda_str = ", ".join(
                f"{ds}={lambda_by_task[ds]:.4f}" for ds in args.DATASETS_VAL
            )
            print(
                f"[iter {iteration+1}] avg_loss={avg_loss:.4f} | losses: {loss_str} | lambdas: {lambda_str}"
            )

        if eval_every > 0 and (iteration + 1) % eval_every == 0:
            model.eval()
            val_metrics = {}
            for dataset_name in args.DATASETS_VAL:
                encoder = model.get_image_encoder()
                metrics = eval_single_dataset(encoder, dataset_name, args)
                val_metrics[dataset_name + ":top1"] = metrics["top1"]
            val_history.append({"iter": iteration, **val_metrics})

    elapsed = time.time() - start_time
    lambdas = model.lambdas().detach().cpu()
    lambdas_by_param = _summarize_lambdas(
        param_names, lambdas, args.DATASETS_VAL, layerwise
    )

    info = {
        "loss_history": loss_history,
        "val_history": val_history,
        "elapsed_sec": elapsed,
    }

    log_f.close()
    return model, task_vectors, lambdas_by_param, info


def _make_deterministic_loader(dataset, batch_size, seed, num_workers, *, drop_last: bool = True):
    if drop_last:
        try:
            dataset_len = len(dataset)
        except TypeError:
            dataset_len = None
        if dataset_len is not None and dataset_len < batch_size:
            raise ValueError(
                "Cannot drop the last (incomplete) batch when the dataset has fewer "
                f"samples than batch_size (len={dataset_len}, batch_size={batch_size}). "
                "Reduce batch_size or disable drop_last."
            )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def _run_tasmerging_optimization(args, layerwise):
    method_cfg = _get_method_cfg(args)
    num_iters = int(_get_method_attr(method_cfg, "num_iters", 500))
    eval_every = int(_get_method_attr(method_cfg, "eval_every", 100))
    log_every = int(_get_method_attr(method_cfg, "log_every", 10))
    lr = float(_get_method_attr(method_cfg, "lr", 1e-3))
    batch_size = int(_get_method_attr(method_cfg, "batch_size", args.batch_size))
    tas_num_batches = int(getattr(args, "num_batches_tas", 5))
    tas_seed = int(_get_method_attr(method_cfg, "data_seed", 123))
    tas_cache_on_gpu = bool(_get_method_attr(method_cfg, "tas_cache_on_gpu", True))
    nan_patience = int(_get_method_attr(method_cfg, "nan_patience", 5))

    print("Building task vectors...")
    ptm_state, task_vectors = _build_task_vectors(args, args.DATASETS_VAL)

    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)
    image_encoder = ImageEncoder.load(args.model, pretrained_checkpoint)
    image_encoder = image_encoder.to(args.device)

    param_names = _make_functional(image_encoder)
    ptm_params, task_params = _build_param_tensors(param_names, ptm_state, task_vectors)

    print("Building AdaMerging model...")
    model = AdaMergingModel(
        image_encoder,
        param_names,
        ptm_params,
        task_params,
        args.DATASETS_VAL,
        args,
        layerwise=layerwise,
    ).to(args.device)

    proxy_dataset_for_task = {
        task_name: resolve_proxy_dataset_for_task(task_name, args)
        for task_name in args.DATASETS_VAL
    }

    proxy_dataset_cache = {}
    loader_fns = {}
    for task_name in args.DATASETS_VAL:
        proxy_dataset_name = proxy_dataset_for_task[task_name]
        if proxy_dataset_name not in proxy_dataset_cache:
            dataset = dataset_by_name(
                proxy_dataset_name,
                image_encoder.val_preprocess,
                location=args.data_location,
                batch_size=batch_size,
            )
            proxy_dataset_cache[proxy_dataset_name] = dataset.test_loader.dataset

        base_dataset = proxy_dataset_cache[proxy_dataset_name]
        num_workers = 0
        try:
            print(
                f"TAS cache dataset size {task_name} -> {proxy_dataset_name}: {len(base_dataset)}"
            )
        except Exception:
            print(f"TAS cache dataset size {task_name} -> {proxy_dataset_name}: <unknown>")
        loader_fns[task_name] = lambda ds=base_dataset, nw=num_workers: _make_deterministic_loader(
            ds, batch_size, tas_seed, nw
        )

    print("Caching FT features...")
    cached_ft = {dataset: [] for dataset in args.DATASETS_VAL}
    for dataset_name in args.DATASETS_VAL:
        loader = loader_fns[dataset_name]()
        loader_iter = iter(loader)
        ft_encoder = get_ft_encoder(dataset_name, args).to(args.device)
        for _ in range(tas_num_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(args.device)
            with torch.no_grad():
                feat = ft_encoder(x).detach()
            if not tas_cache_on_gpu:
                feat = feat.cpu()
            cached_ft[dataset_name].append(feat)
            
    del ft_encoder
    torch.cuda.empty_cache()

    loaders_iter = {name: iter(loader_fns[name]()) for name in args.DATASETS_VAL}
    batch_idx = {name: 0 for name in args.DATASETS_VAL}
    cached_sizes = {name: len(cached_ft[name]) for name in args.DATASETS_VAL}

    optimizer = torch.optim.Adam(
        model.collect_trainable_params(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0
    )

    loss_history = []
    val_history = []
    log_path = _get_optimization_log_path(args, method_cfg.name)
    log_f = open(log_path, "w", encoding="utf-8")

    model.train()
    start_time = time.time()
    num_tasks = len(args.DATASETS_VAL)
    nan_iters_in_row = 0
    for iteration in range(num_iters):
        losses = defaultdict(float)
        time_iter_start = time.time()
        valid_tasks = 0
        any_nonfinite = False
        for dataset_name in args.DATASETS_VAL:
            max_batches = cached_sizes[dataset_name]
            if max_batches == 0:
                continue
            if batch_idx[dataset_name] >= max_batches:
                loaders_iter[dataset_name] = iter(loader_fns[dataset_name]())
                batch_idx[dataset_name] = 0

            try:
                batch = next(loaders_iter[dataset_name])
            except StopIteration:
                loaders_iter[dataset_name] = iter(loader_fns[dataset_name]())
                batch = next(loaders_iter[dataset_name])

            cached_feat = cached_ft[dataset_name][batch_idx[dataset_name]]
            if not tas_cache_on_gpu:
                cached_feat = cached_feat.to(args.device)

            merged_encoder = model.get_image_encoder().to(args.device)
            loss = tas_from_cached(merged_encoder, batch, cached_feat, args)
            if not torch.isfinite(loss.detach()):
                any_nonfinite = True
                losses[dataset_name] = float("nan")
                continue

            (loss / max(num_tasks, 1)).backward()
            losses[dataset_name] += float(loss.detach().cpu())
            valid_tasks += 1
            batch_idx[dataset_name] += 1

        if valid_tasks == 0:
            nan_iters_in_row += 1
            optimizer.zero_grad()
            print(
                f"[iter {iteration}] all TAS losses were non-finite; "
                f"nan_streak={nan_iters_in_row}/{nan_patience}"
            )
            if nan_iters_in_row >= nan_patience:
                log_f.close()
                raise RuntimeError(
                    "TAS merging encountered only non-finite losses for "
                    f"{nan_patience} iterations in a row; aborting."
                )
            continue

        optimizer.step()
        optimizer.zero_grad()

        if any_nonfinite:
            nan_iters_in_row += 1
            print(
                f"[iter {iteration}] encountered non-finite TAS loss; "
                f"nan_streak={nan_iters_in_row}/{nan_patience}"
            )
            if nan_iters_in_row >= nan_patience:
                log_f.close()
                raise RuntimeError(
                    "TAS merging encountered non-finite losses for "
                    f"{nan_patience} iterations in a row; aborting."
                )
        else:
            nan_iters_in_row = 0
        eta = time.time() - time_iter_start
        print(f"Iteration: {iteration}, Loss: {losses}, ETA: {eta:.2f} sec")

        loss_history.append({"iter": iteration, **losses})

        lambdas = model.lambdas().detach().cpu()
        lambda_by_task = _summarize_lambdas_by_task(
            lambdas, args.DATASETS_VAL, layerwise
        )
        finite_losses = [v for v in losses.values() if isinstance(v, (int, float)) and math.isfinite(v)]
        avg_loss = sum(finite_losses) / max(len(finite_losses), 1)
        log_f.write(
            json.dumps(
                {
                    "iter": iteration,
                    "avg_loss": avg_loss,
                    "losses": dict(losses),
                    "lambdas": lambda_by_task,
                    "proxy_datasets": proxy_dataset_for_task,
                }
            )
            + "\n"
        )
        log_f.flush()

        if log_every > 0 and (iteration + 1) % log_every == 0:
            loss_str = ", ".join(
                f"{ds}={losses[ds]:.4f}" for ds in args.DATASETS_VAL
            )
            lambda_str = ", ".join(
                f"{ds}={lambda_by_task[ds]:.4f}" for ds in args.DATASETS_VAL
            )
            print(
                f"[iter {iteration+1}] avg_loss={avg_loss:.4f} | losses: {loss_str} | lambdas: {lambda_str}"
            )

        if eval_every > 0 and (iteration + 1) % eval_every == 0:
            model.eval()
            val_metrics = {}
            for dataset_name in args.DATASETS_VAL:
                encoder = model.get_image_encoder()
                metrics = eval_single_dataset(encoder, dataset_name, args)
                val_metrics[dataset_name + ":top1"] = metrics["top1"]
            val_history.append({"iter": iteration, **val_metrics})

    elapsed = time.time() - start_time
    lambdas = model.lambdas().detach().cpu()
    lambdas_by_param = _summarize_lambdas(
        param_names, lambdas, args.DATASETS_VAL, layerwise
    )

    info = {
        "loss_history": loss_history,
        "val_history": val_history,
        "elapsed_sec": elapsed,
    }

    log_f.close()
    return model, task_vectors, lambdas_by_param, info


def optimize_adamerging(args, layerwise: bool = False):
    def _entropy_loss(model, batch, dataset_name, _ft_encoder):
        batch = maybe_dictionarize(batch)
        x = batch["images"].to(args.device)
        outputs = model(x, dataset_name)
        return _softmax_entropy(outputs).mean(0)

    return _run_adamerging_optimization(args, layerwise, loss_fn=_entropy_loss)


def optimize_tasmerging(args, layerwise: bool = False):
    return _run_tasmerging_optimization(args, layerwise)
