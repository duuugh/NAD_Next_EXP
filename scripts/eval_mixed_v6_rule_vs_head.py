#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_OOF_PRED = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_oof_predictions.csv"
DEFAULT_SELECTED = "/home/jovyan/work/NAD_Next/result/mixed_v6_selected_thresholds.json"
DEFAULT_OUT_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v6_rule_vs_head_comparison.csv"
DEFAULT_OUT_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v6_rule_vs_head_comparison.json"
DEFAULT_AUDIT_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v6_flip_audit.csv"
FIXED_BUCKETS = ["lt_5e4", "btw_5e4_1e3", "btw_1e3_2e3", "ge_2e3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pure rules vs learned head on mixed_v6 OOF predictions.")
    p.add_argument("--oof-predictions", default=DEFAULT_OOF_PRED)
    p.add_argument("--selected-thresholds", default=DEFAULT_SELECTED)
    p.add_argument("--lambda-fp", type=float, default=2.0)
    p.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    p.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV)
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def to_int01(v: str) -> int:
    s = str(v).strip().lower()
    return 1 if s in {"1", "true", "t"} else 0


def compute_metrics(y: np.ndarray, pred: np.ndarray, lambda_fp: float) -> Dict[str, float]:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    flip_count = int(tp + fp)
    flip_rate = float(flip_count / max(y.size, 1))
    net_gain = float(tp - float(lambda_fp) * fp)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "predicted_flip_count": flip_count,
        "flip_rate": flip_rate,
        "net_gain": net_gain,
    }


def sweep_rule_threshold(values: np.ndarray, y: np.ndarray, lambda_fp: float, n_steps: int = 101) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    v = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(v)
    if int(np.sum(mask)) == 0:
        pred = np.zeros_like(y)
        m = compute_metrics(y, pred, lambda_fp=lambda_fp)
        return float("inf"), m, [{"threshold": float("inf"), **m}]

    finite_vals = v[mask]
    uniq = np.unique(finite_vals)
    if uniq.size <= n_steps:
        candidates = uniq
    else:
        candidates = np.linspace(float(np.min(finite_vals)), float(np.max(finite_vals)), num=n_steps)
    candidates = np.unique(np.concatenate([candidates, np.asarray([0.0], dtype=np.float64)]))

    rows = []
    for t in candidates:
        pred = (v > float(t)).astype(np.int64)
        m = compute_metrics(y, pred, lambda_fp=lambda_fp)
        m["threshold"] = float(t)
        rows.append(m)

    best = sorted(rows, key=lambda r: (r["net_gain"], r["precision"], -r["FP"], r["threshold"]), reverse=True)[0]
    return float(best["threshold"]), best, rows


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.oof_predictions))
    selected = json.loads(Path(args.selected_thresholds).read_text())

    rows = [r for r in rows if r.get("label_should_flip", "") != ""]
    y = np.asarray([to_int01(r.get("label_should_flip", "0")) for r in rows], dtype=np.int64)

    p_head = np.asarray([to_float(r.get("p_flip_oof", "nan")) for r in rows], dtype=np.float64)
    lp_gap = np.asarray([to_float(r.get("lp_gap_top2_minus_top1", "nan")) for r in rows], dtype=np.float64)
    sc_gap = np.asarray([to_float(r.get("sc_gap_top2_minus_top1", "nan")) for r in rows], dtype=np.float64)
    buckets = [str(r.get("gap_bucket", "")) for r in rows]

    global_thr = float(selected.get("best_global_threshold", 0.99))
    bucket_thr_map = {str(k): float(v) for k, v in selected.get("best_per_bucket_thresholds", {}).items()}

    lp_thr, lp_best, lp_sweep = sweep_rule_threshold(lp_gap, y, lambda_fp=args.lambda_fp)
    sc_thr, sc_best, sc_sweep = sweep_rule_threshold(sc_gap, y, lambda_fp=args.lambda_fp)

    pred_no = np.zeros_like(y)
    pred_lp = (lp_gap > lp_thr).astype(np.int64)
    pred_sc = (sc_gap > sc_thr).astype(np.int64)
    pred_head_global = (p_head >= global_thr).astype(np.int64)

    pred_head_bucket = np.zeros_like(y)
    for i, b in enumerate(buckets):
        thr = bucket_thr_map.get(b, global_thr)
        pred_head_bucket[i] = int(p_head[i] >= thr)

    comparison = []
    for name, pred in [
        ("no_flip", pred_no),
        ("pure_logprob_rule", pred_lp),
        ("pure_selfcert_rule", pred_sc),
        ("head_global_threshold", pred_head_global),
        ("head_per_bucket_threshold", pred_head_bucket),
    ]:
        m = compute_metrics(y, pred, lambda_fp=args.lambda_fp)
        m["strategy_name"] = name
        comparison.append(m)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["strategy_name", "net_gain", "TP", "FP", "FN", "TN", "precision", "recall", "predicted_flip_count", "flip_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(comparison)

    summary = {
        "lambda_fp": float(args.lambda_fp),
        "selected_thresholds": {
            "head_global": global_thr,
            "head_per_bucket": bucket_thr_map,
            "pure_logprob_rule": lp_thr,
            "pure_selfcert_rule": sc_thr,
        },
        "rule_threshold_sweeps": {
            "pure_logprob_rule": lp_sweep,
            "pure_selfcert_rule": sc_sweep,
        },
        "comparison": comparison,
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    audit_rows = []
    for i, r in enumerate(rows):
        row = {
            "dataset_key": str(r.get("dataset_key", r.get("cache_key", ""))),
            "problem_id": str(r.get("problem_id", "")),
            "gap_bucket": str(r.get("gap_bucket", "")),
            "label_should_flip": int(y[i]),
            "top1_id": str(r.get("top1_sid", "")),
            "top2_id": str(r.get("top2_sid", "")),
            "s1": to_float(r.get("s1", "nan")),
            "s2": to_float(r.get("s2", "nan")),
            "gap": to_float(r.get("gap", "nan")),
            "lp1": to_float(r.get("lp1", "nan")),
            "lp2": to_float(r.get("lp2", "nan")),
            "lp_gap": lp_gap[i],
            "sc1": to_float(r.get("sc1", "nan")),
            "sc2": to_float(r.get("sc2", "nan")),
            "sc_gap": sc_gap[i],
            "len1": to_float(r.get("len1", "nan")),
            "len2": to_float(r.get("len2", "nan")),
            "len_gap": to_float(r.get("len_gap_top2_minus_top1", "nan")),
            "parse_ok1": to_float(r.get("parse_ok1", "nan")),
            "parse_ok2": to_float(r.get("parse_ok2", "nan")),
            "parse_ok_gap": to_float(r.get("parse_ok_gap_top2_minus_top1", "nan")),
            "is_int1": to_float(r.get("is_int1", "nan")),
            "is_int2": to_float(r.get("is_int2", "nan")),
            "is_int_gap": to_float(r.get("is_int_gap_top2_minus_top1", "nan")),
            "p_flip_oof": float(p_head[i]),
            "pure_lp_rule_decision": int(pred_lp[i]),
            "pure_sc_rule_decision": int(pred_sc[i]),
            "head_global_decision": int(pred_head_global[i]),
            "head_bucket_decision": int(pred_head_bucket[i]),
        }
        audit_rows.append(row)

    with Path(args.audit_csv).open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(audit_rows[0].keys()) if audit_rows else []
        if fieldnames:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(audit_rows)

    print(f"rows={len(rows)}")
    print(f"pure_logprob_threshold={lp_thr:.6g} pure_selfcert_threshold={sc_thr:.6g}")
    print(f"head_global_threshold={global_thr:.4f}")
    print(f"head_bucket_thresholds={json.dumps(bucket_thr_map, ensure_ascii=False)}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.audit_csv}")


if __name__ == "__main__":
    main()
