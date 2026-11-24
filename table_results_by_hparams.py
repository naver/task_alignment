import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<method>(?:tasmerging|adamerging)_(?:task|layer))_"
    r"bs=(?P<bs>\d+)_lr=(?P<lr>[^_]+)_linit=(?P<linit>[^_]+)"
    r"(?:_(?P<proxy>proxy=[^_]+))?.*\.json$"
)


def _parse_float_token(token: str) -> float:
    # filenames use 0p001 instead of 0.001
    return float(token.replace("p", "."))


def _read_metric(path: Path, split: str, metric: str) -> Optional[float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get(split, {}).get(metric)


def _fmt(val: Optional[float]) -> str:
    if val is None or not math.isfinite(val):
        return "NA"
    return f"{val:.4f}"


def build_tables(results_root: Path, method_filter: str, split: str, metric: str):
    records = []
    for path in results_root.glob("*.json"):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        method = m.group("method")
        if method_filter and method != method_filter:
            continue
        proxy = m.group("proxy") or "proxy=per_task"
        bs = int(m.group("bs"))
        lr = _parse_float_token(m.group("lr"))
        linit = _parse_float_token(m.group("linit"))
        value = _read_metric(path, split=split, metric=metric)
        records.append((method, proxy, bs, lr, linit, value, path))

    # method -> proxy -> bs -> (lr, linit) -> value
    tables = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    lrs_by_key = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    linits_by_key = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for method, proxy, bs, lr, linit, value, _ in records:
        tables[method][proxy][bs][(lr, linit)] = value
        lrs_by_key[method][proxy][bs].add(lr)
        linits_by_key[method][proxy][bs].add(linit)

    return tables, lrs_by_key, linits_by_key


def render_markdown(tables, lrs_by_key, linits_by_key):
    chunks: list[str] = []
    for method in sorted(tables.keys()):
        chunks.append(f"## {method}")
        for proxy in sorted(tables[method].keys()):
            chunks.append(f"\n### {proxy}")
            for bs in sorted(tables[method][proxy].keys()):
                lrs = sorted(lrs_by_key[method][proxy][bs])
                linits = sorted(linits_by_key[method][proxy][bs])
                chunks.append(f"\n#### bs={bs}")

                header = ["linit \\ lr", *[f"{lr:g}" for lr in lrs]]
                rows = ["| " + " | ".join(header) + " |"]
                rows.append("| " + " | ".join(["---"] * len(header)) + " |")

                for linit in linits:
                    cells = [f"{linit:g}"]
                    for lr in lrs:
                        val = tables[method][proxy][bs].get((lr, linit))
                        cells.append(_fmt(val))
                    rows.append("| " + " | ".join(cells) + " |")

                chunks.extend(rows)
                chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Create Markdown tables of avg top1 vs (lr, linit) grouped by batch size."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing results .json files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        choices=[
            "",
            "tasmerging_task",
            "tasmerging_layer",
            "adamerging_task",
            "adamerging_layer",
        ],
        help="Optional method filter.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Metric split to read (e.g., test, val_best).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_top1",
        help="Metric key to read (e.g., avg_top1, avg_normalized_top1).",
    )
    args = parser.parse_args()

    tables, lrs_by_method_bs, linits_by_method_bs = build_tables(
        args.results_root, args.method, args.split, args.metric
    )
    md = render_markdown(tables, lrs_by_method_bs, linits_by_method_bs)
    print(md, end="")


if __name__ == "__main__":
    main()
