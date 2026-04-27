#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from scripts.mixed_v6_local_utils import bucket_label, parse_bucket_edges


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_training_table.csv"
DEFAULT_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v6_bucket_ablation.csv"
DEFAULT_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v6_bucket_ablation.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bucket ablation for mixed_v6 local correction policies.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-csv", default=DEFAULT_CSV)
    p.add_argument("--output-json", default=DEFAULT_JSON)
    p.add_argument("--bucket-edges", default="5e-4,1e-3,2e-3,5e-3")
    p.add_argument("--logprob-delta-threshold", type=float, default=0.0)
    p.add_argument("--selfcert-delta-threshold", type=float, default=0.0)
    p.add_argument("--cache-keys", default="", help="Optional comma-separated cache keys.")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def to_int01(v: str) -> int:
    s = str(v).strip().lower()
    if s in {"1", "true", "t"}:
        return 1
    return 0


def safe_acc(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(correct / total)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    edges = parse_bucket_edges(args.bucket_edges)
    filter_keys = {x.strip() for x in args.cache_keys.split(",") if x.strip()}

    by_bucket = defaultdict(lambda: {
        "total": 0,
        "baseline_correct": 0,
        "noflip_correct": 0,
        "logprob_correct": 0,
        "selfcert_correct": 0,
        "logprob_flip_count": 0,
        "selfcert_flip_count": 0,
    })

    skipped_unlabeled = 0
    for row in rows:
        ck = row.get("cache_key", "")
        if filter_keys and ck not in filter_keys:
            continue

        label = row.get("label_should_flip", "")
        if label == "":
            skipped_unlabeled += 1
            continue

        y1 = to_int01(row.get("y_top1", "0"))
        y2 = to_int01(row.get("y_top2", "0"))
        gap = to_float(row.get("gap", "nan"))
        b = bucket_label(gap, edges)

        lp_gap = to_float(row.get("lp_gap_top2_minus_top1", "nan"))
        sc_gap = to_float(row.get("sc_gap_top2_minus_top1", "nan"))

        do_logprob_flip = (lp_gap > float(args.logprob_delta_threshold))
        do_selfcert_flip = (sc_gap > float(args.selfcert_delta_threshold))

        st = by_bucket[b]
        st["total"] += 1
        st["baseline_correct"] += y1
        st["noflip_correct"] += y1
        st["logprob_correct"] += (y2 if do_logprob_flip else y1)
        st["selfcert_correct"] += (y2 if do_selfcert_flip else y1)
        st["logprob_flip_count"] += int(do_logprob_flip)
        st["selfcert_flip_count"] += int(do_selfcert_flip)

    bucket_names = ["lt_5e4", "btw_5e4_1e3", "btw_1e3_2e3", "ge_2e3"]
    if not by_bucket:
        bucket_names = []
    out_rows = []

    overall = {
        "bucket": "overall",
        "total": 0,
        "baseline_acc": 0.0,
        "noflip_acc": 0.0,
        "logprob_acc": 0.0,
        "selfcert_acc": 0.0,
        "delta_logprob_vs_baseline": 0.0,
        "delta_selfcert_vs_baseline": 0.0,
        "logprob_flip_rate": 0.0,
        "selfcert_flip_rate": 0.0,
    }

    for b in bucket_names:
        st = by_bucket.get(b)
        if not st:
            continue
        total = int(st["total"])
        if total <= 0:
            continue
        baseline_acc = safe_acc(int(st["baseline_correct"]), total)
        noflip_acc = safe_acc(int(st["noflip_correct"]), total)
        logprob_acc = safe_acc(int(st["logprob_correct"]), total)
        selfcert_acc = safe_acc(int(st["selfcert_correct"]), total)

        row = {
            "bucket": b,
            "total": total,
            "baseline_acc": baseline_acc,
            "noflip_acc": noflip_acc,
            "logprob_acc": logprob_acc,
            "selfcert_acc": selfcert_acc,
            "delta_logprob_vs_baseline": logprob_acc - baseline_acc,
            "delta_selfcert_vs_baseline": selfcert_acc - baseline_acc,
            "logprob_flip_rate": float(st["logprob_flip_count"] / total),
            "selfcert_flip_rate": float(st["selfcert_flip_count"] / total),
        }
        out_rows.append(row)

        overall["total"] += total
        overall["baseline_acc"] += baseline_acc * total
        overall["noflip_acc"] += noflip_acc * total
        overall["logprob_acc"] += logprob_acc * total
        overall["selfcert_acc"] += selfcert_acc * total
        overall["logprob_flip_rate"] += row["logprob_flip_rate"] * total
        overall["selfcert_flip_rate"] += row["selfcert_flip_rate"] * total

    if overall["total"] > 0:
        t = overall["total"]
        for k in ["baseline_acc", "noflip_acc", "logprob_acc", "selfcert_acc", "logprob_flip_rate", "selfcert_flip_rate"]:
            overall[k] = float(overall[k] / t)
        overall["delta_logprob_vs_baseline"] = overall["logprob_acc"] - overall["baseline_acc"]
        overall["delta_selfcert_vs_baseline"] = overall["selfcert_acc"] - overall["baseline_acc"]
        out_rows.append(overall)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

    summary = {
        "input": args.input,
        "bucket_edges": edges,
        "logprob_delta_threshold": args.logprob_delta_threshold,
        "selfcert_delta_threshold": args.selfcert_delta_threshold,
        "cache_key_filter": sorted(filter_keys),
        "skipped_unlabeled": skipped_unlabeled,
        "rows": out_rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"rows={len(out_rows)} skipped_unlabeled={skipped_unlabeled}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
