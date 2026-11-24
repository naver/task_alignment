import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)

FILENAME_RE = re.compile(
    r"^(?P<arch>.+)_(?P<num_tasks>\d+)tasks_(?P<method>.+)_nonlinear_additions_(?P<suffix>.*)\.json$"
)
METHOD_KEYS_RE = re.compile(r"(?:^|_)(k|eta|proxy|tasseed|tasbs|tasnb|tasmetric)=")

OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]

FIGSIZE = (8, 6)
LINEWIDTH = 3.5
MARKERSIZE = 12
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 20
LEGEND_FRAMEALPHA = 0.9
METRIC_LABELS = {
    "l1": "l1 dist",
    "l2": "l2 dist",
    "cosine": "cosine dist",
}
def _parse_method_full(method_full: str):
    matches = list(METHOD_KEYS_RE.finditer(method_full))
    if not matches:
        return method_full, {}
    method_name = method_full[: matches[0].start()].rstrip("_") or method_full
    kv = {}
    for idx, match in enumerate(matches):
        key = match.group(1)
        val_start = match.end()
        val_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(method_full)
        value = method_full[val_start:val_end]
        if value.startswith("_"):
            value = value[1:]
        kv[key] = value
    return method_name, kv


def _parse_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def _same_number(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _display_num(x):
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return str(x)
    if fx.is_integer():
        return str(int(fx))
    return str(fx)


def parse_filename(path: Path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    arch = match.group("arch")
    num_tasks = int(match.group("num_tasks"))
    method_full = match.group("method")
    method_name, kv = _parse_method_full(method_full)

    proxy = kv.get("proxy", "per_task")
    tasbs = _parse_number(kv.get("tasbs"))
    tasnb = _parse_number(kv.get("tasnb"))
    tasseed = _parse_number(kv.get("tasseed"))
    tasmetric = kv.get("tasmetric", "l2")

    return arch, num_tasks, method_name, proxy, tasbs, tasnb, tasseed, tasmetric


def load_val_tas_by_hp(path: Path):
    with path.open("r") as handle:
        payload = json.load(handle)

    val = payload.get("val", {})
    if not isinstance(val, dict):
        return {}

    hp_to_tas = {}
    for hp_key, metrics in val.items():
        try:
            hp = float(hp_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        avg_tas = metrics.get("avg_tas")
        if avg_tas is None:
            continue
        hp_to_tas[hp] = float(avg_tas)
    return hp_to_tas


def collect_distance_curves(
    results_root: Path,
    *,
    method: str,
    model: str,
    num_tasks: int,
    proxy: str,
    tasbs,
    tasnb,
    min_seed: int,
    metrics,
):
    metrics = [str(m) for m in metrics]
    metric_set = set(metrics)

    metric_hp_values = defaultdict(lambda: defaultdict(list))
    seeds_by_metric = defaultdict(set)
    matched_files = []

    for path in results_root.rglob("*.json"):
        parsed = parse_filename(path)
        if parsed is None:
            continue

        arch, n_tasks, method_name, proxy_name, tas_bs, tas_nb, tas_seed, tas_metric = parsed
        if arch != model:
            continue
        if n_tasks != num_tasks:
            continue
        if method_name != method:
            continue
        if proxy_name != proxy:
            continue
        if tas_seed is None or int(tas_seed) < min_seed:
            continue
        if not _same_number(tas_bs, tasbs):
            continue
        if not _same_number(tas_nb, tasnb):
            continue
        if tas_metric not in metric_set:
            continue

        hp_to_tas = load_val_tas_by_hp(path)
        if not hp_to_tas:
            continue

        for hp, avg_tas in hp_to_tas.items():
            metric_hp_values[tas_metric][hp].append(avg_tas)
        seeds_by_metric[tas_metric].add(int(tas_seed))
        matched_files.append(path)

    return metric_hp_values, seeds_by_metric, matched_files


def _mean_std(values):
    if not values:
        return float("nan"), 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def _normalize_curve_0_1(means, stds):
    valid = [m for m in means if not math.isnan(m)]
    if not valid:
        return means, stds

    min_m = min(valid)
    max_m = max(valid)
    span = max_m - min_m
    if span <= 0.0:
        return [0.0 if not math.isnan(m) else float("nan") for m in means], [0.0] * len(stds)

    means_norm = [((m - min_m) / span) if not math.isnan(m) else float("nan") for m in means]
    stds_norm = [s / span for s in stds]
    return means_norm, stds_norm


def save_figure_png_pdf(fig, output_path: Path, **kwargs):
    suffix = output_path.suffix.lower()
    if suffix in {".png", ".pdf"}:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    fig.savefig(base_path.with_suffix(".png"), **kwargs)
    fig.savefig(base_path.with_suffix(".pdf"), **kwargs)


def plot_distance_ablation(
    out_path: Path,
    *,
    metric_hp_values,
    seeds_by_metric,
    metrics,
    model: str,
    num_tasks: int,
    method: str,
    proxy: str,
    tasbs,
    tasnb,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    plotted_any = False
    for idx, metric in enumerate(metrics):
        hp_map = metric_hp_values.get(metric, {})
        if not hp_map:
            continue

        hp_vals = sorted(hp_map.keys())
        means = []
        stds = []
        for hp in hp_vals:
            mean, std = _mean_std(hp_map[hp])
            means.append(mean)
            stds.append(std)

        means, stds = _normalize_curve_0_1(means, stds)
        lower = [max(0.0, m - s) for m, s in zip(means, stds)]
        upper = [min(1.0, m + s) for m, s in zip(means, stds)]

        color = OKABE_ITO[idx % len(OKABE_ITO)]
        legend_label = METRIC_LABELS.get(metric, f"{metric} dist")
        ax.plot(
            hp_vals,
            means,
            marker="o",
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            color=color,
            label=legend_label,
        )
        ax.fill_between(hp_vals, lower, upper, color=color, alpha=0.2)
        plotted_any = True

    if not plotted_any:
        raise SystemExit("No plottable data found after filtering.")

    ax.set_xlabel("TA lambda", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Normalized TAP", fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAMEALPHA)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png_pdf(fig, out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot TAP distance-metric ablation.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to search for result JSON files.",
    )
    parser.add_argument(
        "--method",
        default="sum",
        help="Merging method name (default: sum).",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="Model/architecture name in result filenames (default: ViT-B-32).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=8,
        help="Number of tasks to filter (default: 8).",
    )
    parser.add_argument(
        "--proxy",
        default="per_task",
        help="Proxy dataset setting encoded in filename (default: per_task).",
    )
    parser.add_argument(
        "--tasbs",
        type=float,
        default=128,
        help="TAP batch size to filter (default: 128).",
    )
    parser.add_argument(
        "--tasnb",
        type=int,
        default=1,
        help="Number of TAP batches to filter (default: 1).",
    )
    parser.add_argument(
        "--min-seed",
        type=int,
        default=1,
        help="Minimum TAP seed to include (default: 1).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["l1", "l2", "cosine"],
        help="Distance metrics to include (default: l1 l2 cosine).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/tas_dist_ablation"),
        help="Directory where the output figure will be saved.",
    )
    args = parser.parse_args()

    metric_hp_values, seeds_by_metric, matched_files = collect_distance_curves(
        args.results_root,
        method=args.method,
        model=args.model,
        num_tasks=args.num_tasks,
        proxy=args.proxy,
        tasbs=args.tasbs,
        tasnb=args.tasnb,
        min_seed=args.min_seed,
        metrics=args.metrics,
    )

    if not matched_files:
        raise SystemExit("No matching results files found for the requested TAP distance setting.")

    out_name = (
        f"tasdist_ablation_model={args.model}_tasks={args.num_tasks}_method={args.method}"
        f"_proxy={args.proxy}_tasbs={_display_num(args.tasbs)}_tasnb={_display_num(args.tasnb)}.png"
    )
    out_path = args.out_dir / out_name

    plot_distance_ablation(
        out_path,
        metric_hp_values=metric_hp_values,
        seeds_by_metric=seeds_by_metric,
        metrics=args.metrics,
        model=args.model,
        num_tasks=args.num_tasks,
        method=args.method,
        proxy=args.proxy,
        tasbs=args.tasbs,
        tasnb=args.tasnb,
    )

    print(f"Matched files: {len(matched_files)}")
    for metric in args.metrics:
        n_files = len(seeds_by_metric.get(metric, []))
        print(f"  metric={metric}: {n_files} seeds")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
