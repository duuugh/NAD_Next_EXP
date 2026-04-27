#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP
from scripts.mixed_v6_local_utils import is_aime_cache_key, ranked_items, safe_mean


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json"
DEFAULT_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v2_final_rule_scan.csv"
DEFAULT_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v2_final_rule_scan.json"
DEFAULT_TOP3_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v2_top3_conservative_scan.csv"
DEFAULT_TOP3_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v2_top3_conservative_scan.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final mixed_v2 rule scan: gap x tie-break + conservative top3 baseline.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-csv", default=DEFAULT_CSV)
    p.add_argument("--output-json", default=DEFAULT_JSON)
    p.add_argument("--top3-output-csv", default=DEFAULT_TOP3_CSV)
    p.add_argument("--top3-output-json", default=DEFAULT_TOP3_JSON)
    p.add_argument("--gap-thresholds", default="5e-4,1e-3,2e-3,3e-3")
    p.add_argument("--lp-deltas", default="0")
    p.add_argument("--sc-deltas", default="0")
    p.add_argument("--top3-gap23-thresholds", default="5e-4,1e-3")
    p.add_argument("--top3-margin-deltas", default="0,0.02")
    return p.parse_args()


def parse_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted(set(vals))


def metric_means(reader: CacheReader, sid: int) -> Tuple[float, float]:
    tv = reader.get_token_view(int(sid))
    lp = safe_mean(getattr(tv, "tok_logprob", None))
    sc = safe_mean(getattr(tv, "tok_selfcert", None))
    return float(lp), float(sc)


def get_metrics(cache_key: str, reader: CacheReader, sid: str, memo: Dict[Tuple[str, str], Tuple[float, float]]) -> Tuple[float, float]:
    k = (cache_key, str(sid))
    if k not in memo:
        memo[k] = metric_means(reader, int(sid))
    return memo[k]


def evaluate_top2_policy(
    scores: Dict[str, Dict],
    readers: Dict[str, CacheReader],
    corr_maps: Dict[str, Dict[int, bool]],
    policy: str,
    gap_thr: float,
    lp_delta: float,
    sc_delta: float,
    metric_memo: Dict[Tuple[str, str], Tuple[float, float]],
) -> Dict[str, float]:
    total = 0
    baseline_correct = 0
    policy_correct = 0
    changed = 0

    for ck, problem_map in scores.items():
        if ck not in readers or ck not in corr_maps or not is_aime_cache_key(ck):
            continue
        reader = readers[ck]
        corr = corr_maps[ck]

        for pid, sid_scores in problem_map.items():
            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue
            top1_sid, s1 = ranked[0]
            top2_sid, s2 = ranked[1]
            gap = float(s1 - s2)

            y1 = bool(corr.get(int(top1_sid), False))
            y2 = bool(corr.get(int(top2_sid), False))

            baseline_correct += int(y1)
            total += 1

            chosen_sid = top1_sid
            if gap <= gap_thr:
                lp1, sc1 = get_metrics(ck, reader, top1_sid, metric_memo)
                lp2, sc2 = get_metrics(ck, reader, top2_sid, metric_memo)
                lp_gap = lp2 - lp1
                sc_gap = sc2 - sc1

                if policy == "logprob" and lp_gap > lp_delta:
                    chosen_sid = top2_sid
                elif policy == "selfcert" and sc_gap > sc_delta:
                    chosen_sid = top2_sid
                elif policy == "lp_and_sc" and lp_gap > lp_delta and sc_gap > sc_delta:
                    chosen_sid = top2_sid

            if chosen_sid != top1_sid:
                changed += 1
                policy_correct += int(y2)
            else:
                policy_correct += int(y1)

    return {
        "total": int(total),
        "baseline_correct": int(baseline_correct),
        "policy_correct": int(policy_correct),
        "baseline_acc": float(baseline_correct / max(total, 1)),
        "policy_acc": float(policy_correct / max(total, 1)),
        "delta_acc": float((policy_correct - baseline_correct) / max(total, 1)),
        "changed_count": int(changed),
        "changed_rate": float(changed / max(total, 1)),
    }


def evaluate_top3_conservative(
    scores: Dict[str, Dict],
    readers: Dict[str, CacheReader],
    corr_maps: Dict[str, Dict[int, bool]],
    gap12_thr: float,
    gap23_thr: float,
    margin_delta: float,
    metric: str,
    metric_memo: Dict[Tuple[str, str], Tuple[float, float]],
) -> Dict[str, float]:
    total = 0
    baseline_correct = 0
    policy_correct = 0
    changed = 0

    for ck, problem_map in scores.items():
        if ck not in readers or ck not in corr_maps or not is_aime_cache_key(ck):
            continue
        reader = readers[ck]
        corr = corr_maps[ck]

        for pid, sid_scores in problem_map.items():
            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 3:
                continue

            top1_sid, s1 = ranked[0]
            top2_sid, s2 = ranked[1]
            top3_sid, s3 = ranked[2]
            gap12 = float(s1 - s2)
            gap23 = float(s2 - s3)

            y1 = bool(corr.get(int(top1_sid), False))
            baseline_correct += int(y1)
            total += 1

            chosen_sid = top1_sid
            if gap12 <= gap12_thr and gap23 <= gap23_thr:
                lp1, sc1 = get_metrics(ck, reader, top1_sid, metric_memo)
                lp2, sc2 = get_metrics(ck, reader, top2_sid, metric_memo)
                lp3, sc3 = get_metrics(ck, reader, top3_sid, metric_memo)

                if metric == "logprob":
                    m1, m2, m3 = lp1, lp2, lp3
                else:
                    m1, m2, m3 = sc1, sc2, sc3

                if m2 >= m3:
                    best_sid, best_metric = top2_sid, m2
                else:
                    best_sid, best_metric = top3_sid, m3

                if (best_metric - m1) > margin_delta:
                    chosen_sid = best_sid

            if chosen_sid != top1_sid:
                changed += 1
            policy_correct += int(bool(corr.get(int(chosen_sid), False)))

    return {
        "total": int(total),
        "baseline_correct": int(baseline_correct),
        "policy_correct": int(policy_correct),
        "baseline_acc": float(baseline_correct / max(total, 1)),
        "policy_acc": float(policy_correct / max(total, 1)),
        "delta_acc": float((policy_correct - baseline_correct) / max(total, 1)),
        "changed_count": int(changed),
        "changed_rate": float(changed / max(total, 1)),
    }


def main() -> None:
    args = parse_args()
    gap_thresholds = parse_floats(args.gap_thresholds)
    lp_deltas = parse_floats(args.lp_deltas)
    sc_deltas = parse_floats(args.sc_deltas)
    gap23_thresholds = parse_floats(args.top3_gap23_thresholds)
    margin_deltas = parse_floats(args.top3_margin_deltas)

    inp = json.loads(Path(args.input).read_text())
    scores = inp["scores"]

    readers: Dict[str, CacheReader] = {}
    corr_maps: Dict[str, Dict[int, bool]] = {}
    for ck in scores.keys():
        if not is_aime_cache_key(ck):
            continue
        root = DEFAULT_CACHE_MAP.get(ck)
        if root is None:
            continue
        readers[ck] = CacheReader(root)
        try:
            corr_maps[ck] = load_correctness_map(root)
        except Exception:
            pass

    metric_memo: Dict[Tuple[str, str], Tuple[float, float]] = {}

    top2_rows: List[Dict[str, object]] = []

    # no flip baseline rows by gap-threshold for side-by-side table
    for gap_thr in gap_thresholds:
        m = evaluate_top2_policy(
            scores=scores,
            readers=readers,
            corr_maps=corr_maps,
            policy="no_flip",
            gap_thr=gap_thr,
            lp_delta=0.0,
            sc_delta=0.0,
            metric_memo=metric_memo,
        )
        top2_rows.append({
            "policy": "no_flip",
            "gap_thr": gap_thr,
            "lp_delta": "",
            "sc_delta": "",
            **m,
        })

    for gap_thr in gap_thresholds:
        for lp_delta in lp_deltas:
            m = evaluate_top2_policy(
                scores=scores,
                readers=readers,
                corr_maps=corr_maps,
                policy="logprob",
                gap_thr=gap_thr,
                lp_delta=lp_delta,
                sc_delta=0.0,
                metric_memo=metric_memo,
            )
            top2_rows.append({
                "policy": "logprob",
                "gap_thr": gap_thr,
                "lp_delta": lp_delta,
                "sc_delta": "",
                **m,
            })

    for gap_thr in gap_thresholds:
        for sc_delta in sc_deltas:
            m = evaluate_top2_policy(
                scores=scores,
                readers=readers,
                corr_maps=corr_maps,
                policy="selfcert",
                gap_thr=gap_thr,
                lp_delta=0.0,
                sc_delta=sc_delta,
                metric_memo=metric_memo,
            )
            top2_rows.append({
                "policy": "selfcert",
                "gap_thr": gap_thr,
                "lp_delta": "",
                "sc_delta": sc_delta,
                **m,
            })

    for gap_thr in gap_thresholds:
        for lp_delta in lp_deltas:
            for sc_delta in sc_deltas:
                m = evaluate_top2_policy(
                    scores=scores,
                    readers=readers,
                    corr_maps=corr_maps,
                    policy="lp_and_sc",
                    gap_thr=gap_thr,
                    lp_delta=lp_delta,
                    sc_delta=sc_delta,
                    metric_memo=metric_memo,
                )
                top2_rows.append({
                    "policy": "lp_and_sc",
                    "gap_thr": gap_thr,
                    "lp_delta": lp_delta,
                    "sc_delta": sc_delta,
                    **m,
                })

    top2_rows_sorted = sorted(top2_rows, key=lambda r: (float(r["delta_acc"]), -float(r["changed_rate"])), reverse=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top2_rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(top2_rows_sorted)

    top2_json = {
        "input": args.input,
        "gap_thresholds": gap_thresholds,
        "lp_deltas": lp_deltas,
        "sc_deltas": sc_deltas,
        "best_row": top2_rows_sorted[0],
        "rows": top2_rows_sorted,
    }
    Path(args.output_json).write_text(json.dumps(top2_json, ensure_ascii=False, indent=2))

    # B) very conservative top3 rule baseline
    top3_rows: List[Dict[str, object]] = []
    for metric in ["logprob", "selfcert"]:
        for gap12_thr in gap_thresholds:
            for gap23_thr in gap23_thresholds:
                for margin_delta in margin_deltas:
                    m = evaluate_top3_conservative(
                        scores=scores,
                        readers=readers,
                        corr_maps=corr_maps,
                        gap12_thr=gap12_thr,
                        gap23_thr=gap23_thr,
                        margin_delta=margin_delta,
                        metric=metric,
                        metric_memo=metric_memo,
                    )
                    top3_rows.append({
                        "policy": f"top3_conservative_{metric}",
                        "gap12_thr": gap12_thr,
                        "gap23_thr": gap23_thr,
                        "margin_delta": margin_delta,
                        **m,
                    })

    top3_rows_sorted = sorted(top3_rows, key=lambda r: (float(r["delta_acc"]), -float(r["changed_rate"])), reverse=True)

    with Path(args.top3_output_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top3_rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(top3_rows_sorted)

    top3_json = {
        "input": args.input,
        "gap12_thresholds": gap_thresholds,
        "gap23_thresholds": gap23_thresholds,
        "margin_deltas": margin_deltas,
        "best_row": top3_rows_sorted[0],
        "rows": top3_rows_sorted,
    }
    Path(args.top3_output_json).write_text(json.dumps(top3_json, ensure_ascii=False, indent=2))

    print(f"top2_rows={len(top2_rows_sorted)} best={top2_rows_sorted[0]['policy']} gap={top2_rows_sorted[0]['gap_thr']} delta_acc={top2_rows_sorted[0]['delta_acc']:.6f} changed={top2_rows_sorted[0]['changed_count']}")
    print(f"top3_rows={len(top3_rows_sorted)} best={top3_rows_sorted[0]['policy']} gap12={top3_rows_sorted[0]['gap12_thr']} gap23={top3_rows_sorted[0]['gap23_thr']} margin={top3_rows_sorted[0]['margin_delta']} delta_acc={top3_rows_sorted[0]['delta_acc']:.6f} changed={top3_rows_sorted[0]['changed_count']}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.top3_output_csv}")
    print(f"wrote {args.top3_output_json}")


if __name__ == "__main__":
    main()
