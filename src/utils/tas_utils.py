import torch
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import dataset_by_name
from src.utils.variables_and_paths import get_finetuned_path
from src.models.task_vectors import ImageEncoder


def resolve_proxy_dataset_for_task(task_name, args):
    """Resolve which dataset should be used as the proxy for a task."""
    mode = getattr(args, "proxy_dataset_mode", "per_task")

    if mode == "shared":
        return getattr(args, "proxy_shared_dataset", task_name)

    if mode == "per_task_map":
        mapping = getattr(args, "proxy_task_to_dataset", {}) or {}
        return mapping.get(task_name, task_name)

    return task_name


def tas_single_dataset(merged_encoder, ft_encoder, dataset_name, args):
    if args.skip_tas:
        return {
            "tas": float("nan"),
            "proxy_dataset": resolve_proxy_dataset_for_task(dataset_name, args),
        }

    merged_encoder.eval().to(args.device)
    ft_encoder.eval().to(args.device)

    proxy_dataset_name = resolve_proxy_dataset_for_task(dataset_name, args)
    dataset = dataset_by_name(
        proxy_dataset_name,
        merged_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.tas_batch_size,
    )
    
    generator = torch.Generator()
    generator.manual_seed(int(args.tas_seed))
    dataloader = torch.utils.data.DataLoader(
        dataset.test_dataset,
        batch_size=args.tas_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    with torch.no_grad():
        feat_merged = torch.tensor([]).to(args.device)
        feat_ft = torch.tensor([]).to(args.device)
        for ii, data in enumerate(dataloader):
            if args.num_batches_tas > 0 and ii >= args.num_batches_tas:
                break
            data = maybe_dictionarize(data)
            x = data["images"].to(args.device)
            out = merged_encoder(x)
            feat_merged = torch.cat((feat_merged, out), dim=0)
            out = ft_encoder(x)
            feat_ft = torch.cat((feat_ft, out), dim=0)

        tas = get_tas(
            feat_merged,
            feat_ft,
            normalize=True,
            metric=args.tas_metric,
        ).item()

    return {"tas": tas, "proxy_dataset": proxy_dataset_name}


def tas_batch(merged_encoder, ft_encoder, batch, args):
    merged_encoder.eval()
    ft_encoder.eval()

    batch = maybe_dictionarize(batch)
    x = batch["images"].to(args.device)

    out_merged = merged_encoder(x)
    out_ft = ft_encoder(x)

    return get_tas(out_merged, out_ft, normalize=True, metric=args.tas_metric)


def tas_from_cached(merged_encoder, batch, cached_ft_feat, args):
    merged_encoder.eval()
    batch = maybe_dictionarize(batch)
    x = batch["images"].to(args.device)
    out_merged = merged_encoder(x)
    return get_tas(out_merged, cached_ft_feat, normalize=True, metric=args.tas_metric)


def get_tas(feat1, feat2, normalize=True, metric="l2"):
    """Compute distance between two feature stacks."""
    feat1 = torch.nan_to_num(feat1, nan=0.0, posinf=1e4, neginf=-1e4)
    feat2 = torch.nan_to_num(feat2, nan=0.0, posinf=1e4, neginf=-1e4)

    if normalize:
        std1 = feat1.std(dim=0, keepdim=True).clamp_min(1e-6)
        std2 = feat2.std(dim=0, keepdim=True).clamp_min(1e-6)
        feat1 = (feat1 - feat1.mean(dim=0, keepdim=True)) / std1
        feat2 = (feat2 - feat2.mean(dim=0, keepdim=True)) / std2

    metric = str(metric).lower()

    if metric == "l1":
        dist = torch.norm(feat1 - feat2, p=1, dim=-1)
        return dist.mean()
    if metric == "l2":
        dist = torch.norm(feat1 - feat2, p=2, dim=-1)
        return dist.mean()
    if metric == "cosine":
        feat1_n = torch.nn.functional.normalize(feat1, p=2, dim=-1, eps=1e-6)
        feat2_n = torch.nn.functional.normalize(feat2, p=2, dim=-1, eps=1e-6)
        cos_sim = (feat1_n * feat2_n).sum(dim=-1)
        return (1.0 - cos_sim).mean()
    raise ValueError(f"Unsupported tas_metric: {metric}")


def get_ft_encoder(dataset_name, args):
    model_dir = args.model_location
    ft_state_dict = torch.load(
        get_finetuned_path(model_dir, dataset_name, model=args.model),
        map_location="cpu",
    )
    ft_encoder = ImageEncoder(args.model)
    msg = ft_encoder.load_state_dict(ft_state_dict, strict=True)
    print(f"Loaded ft weights with msg: {msg}")
    return ft_encoder
