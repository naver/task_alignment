import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

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


def save_figure_png_pdf(fig, output_path: Path, **kwargs):
    suffix = output_path.suffix.lower()
    if suffix in {".png", ".pdf"}:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    fig.savefig(base_path.with_suffix(".png"), **kwargs)
    fig.savefig(base_path.with_suffix(".pdf"), **kwargs)


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


def _display_num(value):
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fval.is_integer():
        return str(int(fval))
    return str(fval)


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


def load_val_metrics(path: Path):
    with path.open("r") as handle:
        payload = json.load(handle)

    val = payload.get("val", {})
    if not isinstance(val, dict):
        return []

    entries = []
    for hp_key, metrics in val.items():
        try:
            hp = float(hp_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        avg_tas = metrics.get("avg_tas")
        avg_norm = metrics.get("avg_normalized_top1")
        if avg_tas is None or avg_norm is None:
            continue
        entries.append(
            {
                "hp": hp,
                "avg_tas": float(avg_tas),
                "avg_norm": float(avg_norm),
            }
        )
    return entries


def collect_results(results_root: Path, method_filter: str, min_seed: int, tas_metric: str):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    curve_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    # data[(arch, num_tasks)][proxy][tasbs] -> list of per-seed values (tas-selected)
    # curve_data[(arch, num_tasks)][proxy][tasbs][hp] -> list of avg_tas over seeds
    for path in results_root.rglob("*.json"):
        parsed = parse_filename(path)
        if not parsed:
            continue
        arch, num_tasks, method, proxy, tasbs, tasnb, tasseed, tasmetric = parsed
        if method != method_filter:
            continue
        if tasseed is None or tasbs is None:
            continue
        if tasseed < min_seed:
            continue
        if tasmetric != tas_metric:
            continue

        entries = load_val_metrics(path)
        if not entries:
            continue

        # choose hyperparameter by min TAP
        best = min(entries, key=lambda item: item["avg_tas"])
        best["seed"] = tasseed
        data[(arch, num_tasks)][proxy][tasbs].append(best)

        for entry in entries:
            hp = entry["hp"]
            curve_data[(arch, num_tasks)][proxy][tasbs][hp].append(entry["avg_tas"])

    return data, curve_data


def plot_for_setting(out_dir: Path, arch: str, num_tasks: int, method: str, proxy: str, tasbs_map):
    tasbs_values = sorted(tasbs_map.keys())
    if not tasbs_values:
        return

    # Plot per-seed traces
    seed_points = defaultdict(list)
    for tasbs in tasbs_values:
        for item in tasbs_map[tasbs]:
            seed = item.get("seed")
            if seed is None:
                continue
            seed_points[seed].append((tasbs, item["avg_norm"]))

    means = []
    stds = []
    for tasbs in tasbs_values:
        vals = [item["avg_norm"] for item in tasbs_map[tasbs]]
        if not vals:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(7, 5))
    color = OKABE_ITO[0]
    # per-seed traces (light)
    for seed, points in sorted(seed_points.items()):
        pts = sorted(points, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="gray", alpha=0.35, linewidth=1.0, marker="o", markersize=4)

    ax.plot(tasbs_values, means, marker="o", color=color, label=f"proxy={proxy}")
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(tasbs_values, lower, upper, color=color, alpha=0.2)

    ax.set_xlabel("tasbs")
    ax.set_ylabel("avg_normalized_top1 (TAP-selected)")
    ax.set_title(f"{arch} {num_tasks} tasks - {method} (proxy={proxy})")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_proxy = re.sub(r"[^A-Za-z0-9._-]+", "_", proxy)
    save_figure_png_pdf(fig, out_dir / f"tasbs_ablation_proxy={safe_proxy}", dpi=200)
    plt.close(fig)


def plot_tas_curves(out_dir: Path, arch: str, num_tasks: int, method: str, proxy: str, tasbs_hp_map):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    tasbs_pairs = []
    for key in tasbs_hp_map.keys():
        try:
            num = float(key)
        except (TypeError, ValueError):
            num = None
        tasbs_pairs.append((num, key))

    tasbs_pairs.sort(key=lambda p: (p[0] is None, p[0] if p[0] is not None else str(p[1])))
    tasbs_values = [p[1] for p in tasbs_pairs]
    if not tasbs_values:
        return

    key_to_num = {key: num for num, key in tasbs_pairs}
    all_numeric = all(num is not None for num, _ in tasbs_pairs)

    if all_numeric and len(tasbs_values) > 1:
        nums = [key_to_num[key] for key in tasbs_values]
        if min(nums) != max(nums):
            cmap = mpl.cm.get_cmap("inferno")
            vmin = min(nums)
            vmax = max(nums)
            if vmin > 0:
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            def color_for(key):
                return cmap(0.25 + 0.7 * norm(key_to_num[key]))

        else:
            all_numeric = False

    if not all_numeric:
        colors = OKABE_ITO

        def color_for(key):
            idx = tasbs_values.index(key)
            return colors[idx % len(colors)]

    for tasbs in tasbs_values:
        hp_map = tasbs_hp_map[tasbs]
        hp_vals = sorted(hp_map.keys())
        means = []
        stds = []
        for hp in hp_vals:
            vals = hp_map[hp]
            if not vals:
                means.append(float('nan'))
                stds.append(0.0)
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
            means.append(mean)
            stds.append(std)

        color = color_for(tasbs)
        ax.plot(
            hp_vals,
            means,
            marker="o",
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            color=color,
            label=f"BS={_display_num(tasbs)}",
        )
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(hp_vals, lower, upper, color=color, alpha=0.2)

    ax.set_xlabel("TA lambda", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("TAP", fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAMEALPHA)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_proxy = re.sub(r"[^A-Za-z0-9._-]+", "_", proxy)
    save_figure_png_pdf(fig, out_dir / f"tas_curve_proxy={safe_proxy}", dpi=200)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Plot TAP batch-size ablation.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to search for results JSON files.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Merging method to filter (e.g., ties, sum, STAR, tasmerging_task).",
    )
    parser.add_argument(
        "--min-seed",
        type=int,
        default=1,
        help="Minimum TAP seed to include (default: 1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/tasbs_ablation"),
        help="Directory to write plot images.",
    )
    parser.add_argument(
        "--tas-metric",
        default="l2",
        help="T distance metric to include (default: l2).",
    )
    args = parser.parse_args()

    results, curve_data = collect_results(
        args.results_root,
        args.method,
        args.min_seed,
        args.tas_metric,
    )
    if not results and not curve_data:
        raise SystemExit("No matching results JSON files found.")

    for (arch, num_tasks), proxy_map in sorted(curve_data.items()):
        for proxy, tasbs_hp_map in sorted(proxy_map.items()):
            setting_dir = args.out_dir / f"{arch}_{num_tasks}tasks" / args.method
            plot_tas_curves(setting_dir, arch, num_tasks, args.method, proxy, tasbs_hp_map)


if __name__ == "__main__":
    main()
