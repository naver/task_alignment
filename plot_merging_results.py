import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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
METHOD_KEYS_RE = re.compile(r"(?:^|_)(k|eta|csf|proxy|tasseed|tasbs|tasnb|tasmetric)=")

DEFAULT_TAP_BS = 128
DEFAULT_TAP_NB = 1
DEFAULT_TAP_SEED = 0

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
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]

# Keep method colors/markers aligned with `model_merging/plots/data.py::method_styles`.
METHOD_STYLE_MAP = {
    "TA": ("#E69F00", "o"),
    "TIES": ("#56B4E9", "s"),
    "TSV_Lossless": ("#009E73", "D"),
    "TSV_Lossy": ("#F0E442", "^"),
    "CART": ("#0072B2", "v"),
    "Breadcrumbs": ("#D55E00", "<"),
    "STAR": ("#CC79A7", ">"),
    "Iso-C": ("#AA4499", "h"),
    "Iso-CTS": ("#DDCC77", "p"),
    "Consensus": ("#999999", "P"),
    "Lines+Consensus": ("#117733", "X"),
    "Lines+TA": ("#882255", "*"),
    "MetaGPT_l": ("#56B4B4", "H"),
    "NormAvg_l": ("#E69F99", "d"),
}

METHOD_DISPLAY_MAP = {
    "sum": "TA",
    "ties": "TIES",
    "consensus": "Consensus",
    "TSVM": "TSV",
    "TSV_Lossless": "TSV",
    "iso_cts": "Iso-CTS",
}

METHOD_STYLE_ALIAS = {
    "sum": "TA",
    "ties": "TIES",
    "consensus": "Consensus",
    "TSVM": "TSV_Lossless",
    "iso_c": "Iso-C",
    "iso_cts": "Iso-CTS",
}

PREFERRED_METHOD_ORDER = ["sum", "ties", "TSVM", "STAR", "consensus", "iso_cts"]


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


def parse_filename(path: Path, tas_bs, tas_nb, tas_seed):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    arch = match.group("arch")
    num_tasks = int(match.group("num_tasks"))
    method_full = match.group("method")
    method_name, kv = _parse_method_full(method_full)

    method_hp_name = None
    method_hp_val = None
    if "k" in kv:
        method_hp_name = "k"
        method_hp_val = _parse_number(kv.get("k"))
    elif "eta" in kv:
        method_hp_name = "eta"
        method_hp_val = _parse_number(kv.get("eta"))
    elif "csf" in kv:
        method_hp_name = "csf"
        method_hp_val = _parse_number(kv.get("csf"))

    proxy = kv.get("proxy", "per_task")
    tasbs = _parse_number(kv.get("tasbs", tas_bs))
    tasnb = _parse_number(kv.get("tasnb", tas_nb))
    tasseed = _parse_number(kv.get("tasseed", tas_seed))
    # Older result files may not include the TAP metric in the filename; treat as the default (l2).
    tasmetric = kv.get("tasmetric", "l2")

    return arch, num_tasks, method_name, method_hp_name, method_hp_val, proxy, tasmetric, tasbs, tasnb, tasseed


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
        avg_norm = metrics.get("avg_top1")
        avg_tas = metrics.get("avg_tas")
        if avg_norm is None or avg_tas is None:
            continue
        entries.append(
            {
                "hp": hp,
                "avg_norm": float(avg_norm),
                "avg_tas": float(avg_tas),
            }
        )
    return entries


def collect_results(results_root: Path, tas_bs, tas_nb, tas_seed):
    data = defaultdict(lambda: defaultdict(list))
    for path in results_root.rglob("*.json"):
        parsed = parse_filename(path, tas_bs, tas_nb, tas_seed)
        if not parsed:
            continue
        arch, num_tasks, method, method_hp_name, method_hp_val, proxy, tasmetric, tasbs, tasnb, tasseed = parsed
        if tasbs != tas_bs or tasnb != tas_nb or tasseed != tas_seed:
            continue
        entries = load_val_metrics(path)
        if not entries:
            continue
        for entry in entries:
            entry["method_hp_name"] = method_hp_name
            entry["method_hp_val"] = method_hp_val
            data[(arch, num_tasks, proxy, tasmetric)][method].append(entry)
    return data


def format_hp_label(entry):
    hp_name = entry.get("method_hp_name")
    hp_val = entry.get("method_hp_val")
    hp = entry.get("hp")
    if hp_name is None:
        return f"h={hp:g}"
    if isinstance(hp_val, float):
        hp_label = f"{hp_name}={hp_val:g}"
    else:
        hp_label = f"{hp_name}={hp_val}"
    return f"{hp_label}, h={hp:g}"


def k_sort_key(k_val):
    if k_val is None:
        return (0, "")
    if isinstance(k_val, (int, float)):
        return (1, float(k_val))
    return (2, str(k_val))


def hyperparam_count(entries):
    return len(
        {
            (entry.get("method_hp_name"), entry.get("method_hp_val"), entry.get("hp"))
            for entry in entries
        }
    )


def filter_single_setting(method_entries):
    return {
        method: entries
        for method, entries in method_entries.items()
        if hyperparam_count(entries) > 1
    }


def assign_styles(methods):
    styles = {}
    fallback_idx = 0
    for method in methods:
        canonical = METHOD_STYLE_ALIAS.get(method, method)
        style = METHOD_STYLE_MAP.get(canonical)
        if style is not None:
            color, marker = style
        else:
            color = OKABE_ITO[fallback_idx % len(OKABE_ITO)]
            marker = MARKERS[fallback_idx % len(MARKERS)]
            fallback_idx += 1
        styles[method] = (color, marker)
    return styles


def order_methods(methods):
    preferred = [name for name in PREFERRED_METHOD_ORDER if name in methods]
    return preferred


def filter_preferred_methods(method_entries):
    return {
        method: entries
        for method, entries in method_entries.items()
        if method in PREFERRED_METHOD_ORDER
    }


def add_regression_line(ax, xs, ys):
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    var_y = sum((y - y_mean) ** 2 for y in ys)
    if var_x == 0:
        return None

    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean

    x_min, x_max = min(xs), max(xs)
    line_x = [x_min, x_max]
    line_y = [slope * x + intercept for x in line_x]
    ax.plot(line_x, line_y, linestyle="--", color="red", linewidth=1.5, label="Linear fit")

    if var_y > 0:
        r = cov_xy / math.sqrt(var_x * var_y)
        return r
    return None


def save_figure_png_pdf(fig, output_path: Path, **kwargs):
    suffix = output_path.suffix.lower()
    if suffix in {".png", ".pdf"}:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    fig.savefig(base_path.with_suffix(".png"), **kwargs)
    fig.savefig(base_path.with_suffix(".pdf"), **kwargs)


def plot_method_curves(out_dir: Path, arch: str, num_tasks: int, proxy: str, method: str, entries):
    entries = sorted(
        entries,
        key=lambda item: (
            item.get("method_hp_name") or "",
            k_sort_key(item.get("method_hp_val")),
            item["hp"],
        ),
    )
    x_vals = list(range(len(entries)))
    avg_norm = [item["avg_norm"] for item in entries]
    avg_tas = [item["avg_tas"] for item in entries]
    labels = [format_hp_label(item) for item in entries]

    fig, ax_left = plt.subplots(figsize=(8, 4.5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        x_vals,
        avg_norm,
        marker="o",
        color=OKABE_ITO[0],
        label="avg_top1",
    )
    ax_right.plot(
        x_vals,
        avg_tas,
        marker="s",
        color=OKABE_ITO[1],
        label="avg_tas",
    )

    ax_left.set_xlabel("hyperparameter setting (k, h)")
    ax_left.set_ylabel("avg_top1")
    ax_right.set_ylabel("avg_tas")
    # ax_left.set_title(f"{arch} {num_tasks} tasks - {method} (proxy={proxy})")

    ax_left.set_xticks(x_vals)
    ax_left.set_xticklabels(labels, rotation=45, ha="right")
    ax_left.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)

    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_method = re.sub(r"[^A-Za-z0-9._-]+", "_", method)
    save_figure_png_pdf(fig, out_dir / f"{safe_method}_val_curves", dpi=200)
    plt.close(fig)


def plot_summary_scatter(out_dir: Path, arch: str, num_tasks: int, proxy: str, method_entries):
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered_methods = order_methods(method_entries.keys())
    styles = assign_styles(ordered_methods)

    all_xs = []
    all_ys = []
    for method in ordered_methods:
        entries = method_entries[method]
        color, marker = styles[method]
        xs = [item["avg_tas"] for item in entries]
        ys = [item["avg_norm"] for item in entries]
        label = METHOD_DISPLAY_MAP.get(method, method)
        ax.scatter(
            xs,
            ys,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            s=230,
            alpha=0.8,
            label=label,
        )
        all_xs.extend(xs)
        all_ys.extend(ys)

    r = add_regression_line(ax, all_xs, all_ys)
    if r is not None:
        ax.text(
            0.05,
            0.15,
            f"Pearson r = {r:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            color="red",
        )

    ax.set_xlabel("TAP", fontsize=25)
    ax.set_ylabel("Performance (CLIP)", fontsize=25)
    # ax.set_title(f"{arch} {num_tasks} tasks (proxy={proxy})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.legend(loc="best", fontsize=16, framealpha=0.9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure_png_pdf(fig, out_dir / "summary_scatter_all", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method_tradeoff(out_dir: Path, arch: str, num_tasks: int, proxy: str, method_entries):
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered_methods = order_methods(method_entries.keys())
    styles = assign_styles(ordered_methods)

    xs = []
    ys = []
    for method in ordered_methods:
        entries = method_entries[method]
        color, marker = styles[method]
        best_norm = max(item["avg_norm"] for item in entries)
        best_tas_entry = min(entries, key=lambda item: item["avg_tas"])
        norm_at_best_tas = best_tas_entry["avg_norm"]
        label = METHOD_DISPLAY_MAP.get(method, method)
        ax.scatter(
            best_norm,
            norm_at_best_tas,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            s=230,
            alpha=0.8,
            label=label,
        )
        xs.append(best_norm)
        ys.append(norm_at_best_tas)

    if xs and ys:
        min_val = min(min(xs), min(ys))
        max_val = max(max(xs), max(ys))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="k",
            alpha=0.5,
            label="y=x (oracle)",
            linewidth=2,
        )

    ax.set_xlabel("Perf w/ optimal Hyps", fontsize=25)
    ax.set_ylabel("Perf w/ TAP Hyp. selection", fontsize=25)
    # ax.set_title(f"{arch} {num_tasks} tasks (proxy={proxy})")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax.legend(loc="best", fontsize=16, framealpha=0.9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure_png_pdf(fig, out_dir / "summary_scatter_methods", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot model merging results from JSON files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to search for results JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/merging_results"),
        help="Directory to write plot images.",
    )
    parser.add_argument(
        "--tasbs",
        type=int,
        default=DEFAULT_TAP_BS,
        help="Filter results by TAP batch size (tasbs).",
    )
    parser.add_argument(
        "--tasnb",
        type=int,
        default=DEFAULT_TAP_NB,
        help="Filter results by TAP num batches (tasnb).",
    )
    parser.add_argument(
        "--tasseed",
        type=int,
        default=DEFAULT_TAP_SEED,
        help="Filter results by TAP seed (tasseed).",
    )
    args = parser.parse_args()

    results = collect_results(args.results_root, args.tasbs, args.tasnb, args.tasseed)
    if not results:
        raise SystemExit("No matching results JSON files found.")

    for (arch, num_tasks, proxy, tasmetric), method_entries in sorted(results.items()):
        filtered_entries = filter_single_setting(method_entries)
        filtered_entries = filter_preferred_methods(filtered_entries)
        if not filtered_entries:
            continue
        setting_dir = args.out_dir / f"{arch}_{num_tasks}tasks" / f"proxy={proxy}"
        if tasmetric:
            setting_dir = setting_dir / f"tasmetric={tasmetric}"
        for method in order_methods(filtered_entries.keys()):
            plot_method_curves(setting_dir, arch, num_tasks, proxy, method, filtered_entries[method])
        plot_summary_scatter(setting_dir, arch, num_tasks, proxy, filtered_entries)
        plot_method_tradeoff(setting_dir, arch, num_tasks, proxy, filtered_entries)


if __name__ == "__main__":
    main()
