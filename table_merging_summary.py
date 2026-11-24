import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HEADER_RE = re.compile(r"^(?P<arch>.+)_(?P<num_tasks>\d+)tasks_(?P<rest>.+)$")
Metric = Tuple[float, float]
ColumnKey = Tuple[str, int]
RowData = Tuple[str, str, List[Optional[Metric]]]

MODEL_ORDER = [
    "ViT-B-32",
    "ViT-B-16",
    "ViT-L-14"
]
TASK_ORDER = [8, 14, 20]

METHOD_PREFIXES = {
    "average": "average",
    "sum": "sum",
    "ties": "ties",
    "TSVM": "TSVM",
    "iso_cts": "iso_cts",
    "iso_c": "iso_c",
    "STAR": "STAR",
    "consensus": "consensus",
    "adamerging_layer": "adamerging_layer",
    "tasmerging_layer": "tasmerging_layer",
}

BASE_METHOD_ROWS = [
    ("average", "Average"),
    ("sum", "TA"),
    ("ties", "TIES"),
    ("TSVM", "TSV"),
    ("iso_c", "Iso-C"),
    ("iso_cts", "Iso-CTS"),
    ("STAR", "STAR"),
    ("consensus", "Consensus"),
    ("adamerging_layer", "Adamerge"),
    ("tasmerging_layer", "Adamerge$_{{\\text{{+\\talign}}}}$"),
]

PLOT_METHODS = {"sum", "ties", "TSVM", "iso_c", "iso_cts", "STAR", "consensus"}


def parse_file_metadata(path: Path) -> Optional[Tuple[str, int, str]]:
    name = path.name
    if not name.endswith(".json"):
        return None
    stem = path.stem
    if "_nonlinear_additions_" not in stem:
        return None
    stem = stem.split("_nonlinear_additions_", 1)[0]
    match = HEADER_RE.match(stem)
    if not match:
        return None
    arch = match.group("arch")
    num_tasks = int(match.group("num_tasks"))
    rest = match.group("rest")

    method = None
    for key, prefix in METHOD_PREFIXES.items():
        if rest.startswith(prefix):
            method = key
            break
    if method is None:
        return None
    return arch, num_tasks, method


def _read_metrics(payload: Dict, split: str) -> Optional[Tuple[float, float]]:
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        return None
    top1 = split_payload.get("avg_top1")
    norm = split_payload.get("avg_normalized_top1")
    if top1 is None or norm is None:
        return None
    try:
        top1 = float(top1)
        norm = float(norm)
        if not math.isfinite(top1) or not math.isfinite(norm):
            return None
        return top1, norm
    except (TypeError, ValueError):
        return None


def _update_best(
    store: Dict[str, Dict[ColumnKey, Metric]],
    row_key: str,
    col_key: ColumnKey,
    metric: Metric,
) -> None:
    current = store[row_key].get(col_key)
    if current is None or metric[0] > current[0]:
        store[row_key][col_key] = metric


def _sort_columns(
    columns: List[ColumnKey],
    model_order: List[str],
    task_order: List[int],
) -> List[ColumnKey]:
    model_rank = {m: i for i, m in enumerate(model_order)}
    task_rank = {t: i for i, t in enumerate(task_order)}
    return sorted(
        columns,
        key=lambda x: (
            model_rank.get(x[0], len(model_rank)),
            x[0],
            task_rank.get(x[1], len(task_rank)),
            x[1],
        ),
    )


def _format_cell(
    metric: Optional[Metric],
    with_normalized_subscript: bool,
    bold_top1: bool = False,
) -> str:
    if metric is None:
        return "--"
    top1, norm = metric
    top1 = top1 * 100.0
    top1_str = f"{top1:.2f}"
    if bold_top1:
        top1_str = f"\\textbf{{{top1_str}}}"
    if not with_normalized_subscript:
        return f"{{\\scriptsize {top1_str}}}"
    norm = norm * 100.0
    return f"{top1_str}$_{{\\scriptscriptstyle({norm:.2f})}}$"


def _average_metric(metrics: List[Optional[Metric]]) -> Optional[Metric]:
    valid = [m for m in metrics if m is not None]
    if not valid:
        return None
    avg_top1 = sum(m[0] for m in valid) / len(valid)
    avg_norm = sum(m[1] for m in valid) / len(valid)
    return avg_top1, avg_norm


def _format_col_header(col: Tuple[str, int]) -> str:
    arch, num_tasks = col
    return f"{arch} ({num_tasks})"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def build_table(
    results_root: Path,
    model_order: List[str],
    task_order: List[int],
) -> Tuple[List[ColumnKey], List[RowData]]:
    row_metrics: Dict[str, Dict[ColumnKey, Metric]] = defaultdict(dict)
    columns = set()
    allowed_models = set(model_order)
    allowed_tasks = set(task_order)

    for path in results_root.glob("*.json"):
        parsed = parse_file_metadata(path)
        if parsed is None:
            continue
        arch, num_tasks, method = parsed
        if arch not in allowed_models:
            continue
        if num_tasks not in allowed_tasks:
            continue
        col_key = (arch, num_tasks)
        columns.add(col_key)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        base_metric = _read_metrics(payload, "test")
        if base_metric is not None:
            _update_best(row_metrics, method, col_key, base_metric)

        if method in PLOT_METHODS:
            talign_metric = _read_metrics(payload, "test_tas")
            if talign_metric is not None:
                _update_best(row_metrics, f"{method}_talign", col_key, talign_metric)

    sorted_cols = _sort_columns(
        list(columns),
        model_order=model_order,
        task_order=task_order,
    )
    rows: List[RowData] = []

    for method_key, label in BASE_METHOD_ROWS:
        metrics = [row_metrics[method_key].get(col) for col in sorted_cols]
        rows.append((method_key, label, metrics))
        if method_key in PLOT_METHODS:
            talign_key = f"{method_key}_talign"
            talign_label = f"{label}$_{{\\text{{+\\talign}}}}$"
            talign_metrics = [row_metrics[talign_key].get(col) for col in sorted_cols]
            rows.append((talign_key, talign_label, talign_metrics))

    return sorted_cols, rows


def render_latex(
    columns: List[ColumnKey],
    rows: List[RowData],
    model_order: List[str],
    task_order: List[int],
    with_normalized_subscript: bool,
) -> str:
    groups: Dict[str, List[ColumnKey]] = defaultdict(list)
    for col in columns:
        groups[col[0]].append(col)
    task_rank = {t: i for i, t in enumerate(task_order)}
    for arch in groups:
        groups[arch] = sorted(groups[arch], key=lambda x: (task_rank.get(x[1], len(task_rank)), x[1]))

    ordered_arches = [arch for arch in model_order if arch in groups] + [
        arch for arch in sorted(groups.keys()) if arch not in model_order
    ]

    col_spec_parts = ["l|"]
    ordered_cols: List[ColumnKey] = []
    for arch in ordered_arches:
        cols = groups[arch]
        col_spec_parts.append("c" * len(cols) + "|")
        ordered_cols.extend(cols)
    col_spec_parts.append("c")
    col_spec = "".join(col_spec_parts)

    lines = []
    lines.append("\\begingroup")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    top_header = ["\\multirow{2}{\\textbf{Merging}}{\\textbf{Method}}"]
    for arch in ordered_arches:
        top_header.append(f"\\multicolumn{{{len(groups[arch])}}}{{c}}{{\\texttt{{{arch}}}}}")
    top_header.append("\\multirow{2}{\\textbf{Avg}}{\\textbf{Perf}}")
    lines.append(" & ".join(top_header) + " \\\\")
    col_start = 2
    for arch in ordered_arches:
        col_end = col_start + len(groups[arch]) - 1
        lines.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1

    second_header = [""]
    for arch in ordered_arches:
        second_header.extend([f"T={num_tasks}" for _, num_tasks in groups[arch]])
    second_header.append("")
    lines.append(" & ".join(second_header) + " \\\\")
    lines.append("\\midrule")

    ordered_metrics_by_key: Dict[str, List[Optional[Metric]]] = {}
    avg_metric_by_key: Dict[str, Optional[Metric]] = {}
    for row_key, _, metrics in rows:
        metric_map = {col: metric for col, metric in zip(columns, metrics)}
        ordered_metrics = [metric_map.get(col) for col in ordered_cols]
        ordered_metrics_by_key[row_key] = ordered_metrics
        avg_metric_by_key[row_key] = _average_metric(ordered_metrics)

    bold_mask_by_key: Dict[str, List[bool]] = {}
    compare_len = len(ordered_cols) + 1
    compare_pairs = [(method_key, f"{method_key}_talign") for method_key in PLOT_METHODS]
    compare_pairs.append(("adamerging_layer", "tasmerging_layer"))
    for left_key, right_key in compare_pairs:
        if left_key not in ordered_metrics_by_key or right_key not in ordered_metrics_by_key:
            continue
        left_series = ordered_metrics_by_key[left_key] + [avg_metric_by_key[left_key]]
        right_series = ordered_metrics_by_key[right_key] + [avg_metric_by_key[right_key]]
        left_mask = [False] * compare_len
        right_mask = [False] * compare_len
        for index, (left_metric, right_metric) in enumerate(zip(left_series, right_series)):
            if left_metric is None or right_metric is None:
                continue
            left_display = round(left_metric[0] * 100.0, 2)
            right_display = round(right_metric[0] * 100.0, 2)
            if left_display > right_display:
                left_mask[index] = True
            elif right_display > left_display:
                right_mask[index] = True
            else:
                left_mask[index] = True
                right_mask[index] = True
        bold_mask_by_key[left_key] = left_mask
        bold_mask_by_key[right_key] = right_mask

    for row_key, row_label, _ in rows:
        row_metrics = ordered_metrics_by_key[row_key]
        row_avg_metric = avg_metric_by_key[row_key]
        row_mask = bold_mask_by_key.get(row_key, [False] * compare_len)
        cells = [
            _format_cell(metric, with_normalized_subscript, bold_top1=row_mask[idx])
            for idx, metric in enumerate(row_metrics)
        ]
        cells.append(_format_cell(row_avg_metric, with_normalized_subscript, bold_top1=row_mask[-1]))
        label_out = row_label if "$" in row_label else _latex_escape(row_label)
        lines.append(" & ".join([label_out] + cells) + " \\\\")
        if row_key == "average" or row_key.endswith("_talign"):
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\endgroup")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build LaTeX summary table for merging results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/tables/merging_summary_talign.tex"),
        help="Output LaTeX file path.",
    )
    parser.add_argument(
        "--model-order",
        type=str,
        default=",".join(MODEL_ORDER),
        help="Comma-separated model order and inclusion filter, e.g. ViT-B-32,ViT-L-14",
    )
    parser.add_argument(
        "--task-order",
        type=str,
        default=",".join(str(x) for x in TASK_ORDER),
        help="Comma-separated task-count order and inclusion filter, e.g. 8,20",
    )
    parser.add_argument(
        "--with-normalized-subscript",
        action="store_true",
        help="Show avg_normalized_top1 as a smaller subscript in each table cell.",
    )
    args = parser.parse_args()

    model_order = [x.strip() for x in args.model_order.split(",") if x.strip()]
    task_order = [int(x.strip()) for x in args.task_order.split(",") if x.strip()]
    if not model_order:
        raise SystemExit("Empty --model-order is not allowed.")
    if not task_order:
        raise SystemExit("Empty --task-order is not allowed.")

    columns, rows = build_table(
        args.results_root,
        model_order=model_order,
        task_order=task_order,
    )
    if not columns:
        raise SystemExit(
            "No matching result files after applying --model-order and --task-order filters."
        )
    latex = render_latex(
        columns,
        rows,
        model_order=model_order,
        task_order=task_order,
        with_normalized_subscript=args.with_normalized_subscript,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table: {args.out}")


if __name__ == "__main__":
    main()
