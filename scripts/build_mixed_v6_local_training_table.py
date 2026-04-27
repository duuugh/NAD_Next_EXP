#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP
from scripts.mixed_v6_local_utils import (
    bucket_label,
    build_local_pair_features,
    is_aime_cache_key,
    load_eval_run_info,
    parse_bucket_edges,
    ranked_items,
)


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json"
DEFAULT_OUTPUT = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_training_table.csv"
DEFAULT_META = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_training_table_meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AIME-aware local top1/top2 correction training table.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--meta-output", default=DEFAULT_META)
    p.add_argument("--target-cache-keys", default="", help="Comma-separated cache keys; empty means auto AIME keys.")
    p.add_argument("--aime-only", action="store_true", help="Only keep AIME24/AIME25 cache keys.")
    p.add_argument("--min-gap", type=float, default=0.0, help="Keep rows with gap >= min-gap.")
    p.add_argument("--max-gap", type=float, default=0.002, help="Keep rows with gap < max-gap; <=0 means no upper bound.")
    p.add_argument("--bucket-edges", default="5e-4,1e-3,2e-3,5e-3", help="Comma-separated gap bucket edges.")
    p.add_argument("--include-unlabeled", action="store_true", help="Keep rows without ground-truth labels.")
    return p.parse_args()


def default_target_cache_keys(score_map: Dict[str, Dict]) -> List[str]:
    keys = list(score_map.keys())
    return [k for k in keys if is_aime_cache_key(k)]


def keep_by_gap(gap: float, min_gap: float, max_gap: float) -> bool:
    if gap < min_gap:
        return False
    if max_gap > 0 and gap >= max_gap:
        return False
    return True


def to_float01(x: Optional[bool]) -> float:
    if x is None:
        return float("nan")
    return float(1.0 if bool(x) else 0.0)


def main() -> None:
    args = parse_args()
    inp = json.loads(Path(args.input).read_text())
    scores = inp["scores"]
    edges = parse_bucket_edges(args.bucket_edges)

    if args.target_cache_keys.strip():
        target_cache_keys = [x.strip() for x in args.target_cache_keys.split(",") if x.strip()]
    else:
        target_cache_keys = default_target_cache_keys(scores)

    if args.aime_only:
        target_cache_keys = [k for k in target_cache_keys if is_aime_cache_key(k)]

    readers: Dict[str, CacheReader] = {}
    corr_maps: Dict[str, Optional[Dict[int, bool]]] = {}
    eval_infos: Dict[str, Dict[int, Dict[str, float]]] = {}
    missing_cache_keys: List[str] = []

    for ck in target_cache_keys:
        root = DEFAULT_CACHE_MAP.get(ck)
        if root is None:
            missing_cache_keys.append(ck)
            continue
        readers[ck] = CacheReader(root)
        eval_infos[ck] = load_eval_run_info(root)
        try:
            corr_maps[ck] = load_correctness_map(root)
        except Exception:
            corr_maps[ck] = None

    rows: List[Dict[str, object]] = []
    bucket_counter = defaultdict(int)
    bucket_positive = defaultdict(int)
    cache_counter = defaultdict(int)

    stats = {
        "input": args.input,
        "target_cache_keys": target_cache_keys,
        "missing_cache_keys": missing_cache_keys,
        "aime_only": args.aime_only,
        "min_gap": args.min_gap,
        "max_gap": args.max_gap,
        "bucket_edges": edges,
        "include_unlabeled": args.include_unlabeled,
        "total_problem_rows": 0,
        "kept_rows": 0,
        "dropped_gap": 0,
        "dropped_unlabeled": 0,
    }

    for ck in target_cache_keys:
        problem_map = scores.get(ck)
        if ck not in readers or problem_map is None:
            continue
        reader = readers[ck]
        corr_map = corr_maps.get(ck)
        eval_info = eval_infos.get(ck, {})

        for pid, sid_scores in problem_map.items():
            stats["total_problem_rows"] += 1

            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue
            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            top3_score = float(ranked[2][1]) if len(ranked) >= 3 else float("nan")
            gap = float(top1_score - top2_score)

            if not keep_by_gap(gap, min_gap=float(args.min_gap), max_gap=float(args.max_gap)):
                stats["dropped_gap"] += 1
                continue

            y_top1 = None
            y_top2 = None
            label_should_flip = None
            if corr_map is not None:
                y_top1 = bool(corr_map.get(int(top1_sid), False))
                y_top2 = bool(corr_map.get(int(top2_sid), False))
                label_should_flip = int((not y_top1) and y_top2)
            elif not args.include_unlabeled:
                stats["dropped_unlabeled"] += 1
                continue

            feat = build_local_pair_features(
                reader=reader,
                eval_info=eval_info,
                top1_sid=top1_sid,
                top2_sid=top2_sid,
                top1_score=float(top1_score),
                top2_score=float(top2_score),
                top3_score=float(top3_score),
            )
            gap_bucket = bucket_label(gap, edges)
            row = {
                "cache_key": ck,
                "problem_id": str(pid),
                "top1_sid": top1_sid,
                "top2_sid": top2_sid,
                "gap_bucket": gap_bucket,
                "y_top1": y_top1,
                "y_top2": y_top2,
                "label_should_flip": label_should_flip,
                "y_top1_float": to_float01(y_top1),
                "y_top2_float": to_float01(y_top2),
            }
            row.update(feat)

            rows.append(row)
            stats["kept_rows"] += 1
            cache_counter[ck] += 1
            bucket_counter[gap_bucket] += 1
            if label_should_flip == 1:
                bucket_positive[gap_bucket] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    stats["rows_per_cache_key"] = dict(cache_counter)
    stats["rows_per_bucket"] = dict(bucket_counter)
    stats["positive_per_bucket"] = dict(bucket_positive)
    stats["positive_rate_per_bucket"] = {
        b: float(bucket_positive.get(b, 0) / max(bucket_counter.get(b, 1), 1))
        for b in bucket_counter
    }

    Path(args.meta_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"rows={len(rows)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.meta_output}")


if __name__ == "__main__":
    main()
