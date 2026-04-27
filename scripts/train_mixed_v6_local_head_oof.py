#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from scripts.mixed_v6_local_utils import canonical_gap_bucket
from scripts.train_mixed_v6_local_head import LOCAL_FEATURES


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_training_table.csv"
DEFAULT_OOF_PRED = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_oof_predictions.csv"
DEFAULT_OOF_SUMMARY = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_oof_summary.json"
DEFAULT_FULL_MODEL = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_head_full.pkl"
DEFAULT_SWEEP_GLOBAL_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v6_threshold_sweep_global.csv"
DEFAULT_SWEEP_GLOBAL_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v6_threshold_sweep_global.json"
DEFAULT_SWEEP_BUCKET_CSV = "/home/jovyan/work/NAD_Next/result/mixed_v6_threshold_sweep_per_bucket.csv"
DEFAULT_SWEEP_BUCKET_JSON = "/home/jovyan/work/NAD_Next/result/mixed_v6_threshold_sweep_per_bucket.json"
DEFAULT_SELECTED = "/home/jovyan/work/NAD_Next/result/mixed_v6_selected_thresholds.json"

FIXED_BUCKETS = ["lt_5e4", "btw_5e4_1e3", "btw_1e3_2e3", "ge_2e3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mixed_v6.1 local head with K-fold OOF threshold selection.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--features", default=",".join(LOCAL_FEATURES))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-fp", type=float, default=2.0)
    p.add_argument("--min-threshold", type=float, default=0.50)
    p.add_argument("--max-threshold", type=float, default=0.99)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--oof-pred-output", default=DEFAULT_OOF_PRED)
    p.add_argument("--oof-summary-output", default=DEFAULT_OOF_SUMMARY)
    p.add_argument("--full-model-output", default=DEFAULT_FULL_MODEL)
    p.add_argument("--sweep-global-csv", default=DEFAULT_SWEEP_GLOBAL_CSV)
    p.add_argument("--sweep-global-json", default=DEFAULT_SWEEP_GLOBAL_JSON)
    p.add_argument("--sweep-bucket-csv", default=DEFAULT_SWEEP_BUCKET_CSV)
    p.add_argument("--sweep-bucket-json", default=DEFAULT_SWEEP_BUCKET_JSON)
    p.add_argument("--selected-thresholds-output", default=DEFAULT_SELECTED)
    return p.parse_args()


def parse_feature_names(raw: str) -> List[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No features provided.")
    return out


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


def ensure_bucket_name(row: Dict[str, str]) -> str:
    raw = str(row.get("gap_bucket", "")).strip()
    if raw in FIXED_BUCKETS:
        return raw
    return canonical_gap_bucket(to_float(row.get("gap", "nan")))


def build_matrix(rows: List[Dict[str, str]], features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for r in rows:
        y_raw = r.get("label_should_flip", "")
        if y_raw == "":
            continue
        xs.append([to_float(r.get(col, "")) for col in features])
        ys.append(to_int01(y_raw))
    if not xs:
        raise ValueError("No labeled rows found in training table.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def fill_nan_with_median(x_train: np.ndarray, x_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train_f = np.where(np.isfinite(x_train), x_train, med)
    x_eval_f = np.where(np.isfinite(x_eval), x_eval, med)
    return x_train_f, x_eval_f, med


def get_threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold-step must be > 0")
    n = int(np.floor((end - start) / step + 1e-9)) + 1
    vals = start + np.arange(n, dtype=np.float64) * step
    vals = np.clip(vals, start, end)
    return np.unique(np.round(vals, 10))


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


def sweep_thresholds(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, lambda_fp: float) -> List[Dict[str, float]]:
    out = []
    for t in thresholds:
        pred = (p >= float(t)).astype(np.int64)
        m = compute_metrics(y, pred, lambda_fp=lambda_fp)
        m["threshold"] = float(t)
        out.append(m)
    return out


def pick_best(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return sorted(rows, key=lambda r: (r["net_gain"], r["precision"], -r["FP"], r["threshold"]), reverse=True)[0]


def metrics_from_threshold(y: np.ndarray, p: np.ndarray, thr: float, lambda_fp: float) -> Dict[str, float]:
    pred = (p >= float(thr)).astype(np.int64)
    out = compute_metrics(y, pred, lambda_fp=lambda_fp)
    out["threshold"] = float(thr)
    return out


def train_fold(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int) -> np.ndarray:
    x_train_f, x_val_f, _ = fill_nan_with_median(x_train, x_val)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_f)
    x_val_s = scaler.transform(x_val_f)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(x_train_s, y_train)
    return model.predict_proba(x_val_s)[:, 1]


def train_full_model(x: np.ndarray, y: np.ndarray, features: List[str], seed: int) -> Dict[str, object]:
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_f = np.where(np.isfinite(x), x, med)
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x_f)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(x_s, y)
    return {
        "features": features,
        "nan_medians": med.astype(np.float64),
        "scaler": scaler,
        "model": model,
        "seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    features = parse_feature_names(args.features)
    thresholds = get_threshold_grid(args.min_threshold, args.max_threshold, args.threshold_step)

    all_rows_raw = load_rows(Path(args.input))
    rows = [r for r in all_rows_raw if r.get("label_should_flip", "") != ""]
    for r in rows:
        r["gap_bucket"] = ensure_bucket_name(r)

    x, y = build_matrix(rows, features)
    n = len(rows)
    if n != x.shape[0]:
        raise RuntimeError("Row/feature length mismatch.")
    if n < 2:
        raise ValueError("Need at least 2 labeled rows for OOF.")

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))

    split_logs: List[Dict[str, int]] = []
    if pos_count >= args.n_splits and neg_count >= args.n_splits and len(np.unique(y)) > 1:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(x, y)
        splitter_name = "StratifiedKFold"
    else:
        splitter = KFold(n_splits=min(args.n_splits, n), shuffle=True, random_state=args.seed)
        split_iter = splitter.split(x)
        splitter_name = "KFold"

    oof = np.full((n,), np.nan, dtype=np.float64)
    for fold_id, idx in enumerate(split_iter):
        if splitter_name == "StratifiedKFold":
            train_idx, val_idx = idx
        else:
            train_idx, val_idx = idx
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        if len(np.unique(y_train)) < 2:
            p_val = np.full((len(val_idx),), float(np.mean(y_train)), dtype=np.float64)
        else:
            p_val = train_fold(x_train, y_train, x_val, seed=args.seed + fold_id)

        oof[val_idx] = p_val
        split_logs.append(
            {
                "fold": int(fold_id),
                "train_size": int(len(train_idx)),
                "valid_size": int(len(val_idx)),
                "train_pos": int(np.sum(y_train == 1)),
                "train_neg": int(np.sum(y_train == 0)),
                "valid_pos": int(np.sum(y_val == 1)),
                "valid_neg": int(np.sum(y_val == 0)),
            }
        )

    if not np.all(np.isfinite(oof)):
        raise RuntimeError("OOF prediction contains NaN. Check split logic.")

    global_rows = sweep_thresholds(y, oof, thresholds=thresholds, lambda_fp=args.lambda_fp)
    best_global = pick_best(global_rows)

    per_bucket_rows: List[Dict[str, float]] = []
    best_per_bucket: Dict[str, Dict[str, float]] = {}
    per_bucket_thresholds: Dict[str, float] = {}

    gap_buckets = np.asarray([r["gap_bucket"] for r in rows])

    for b in FIXED_BUCKETS:
        mask = gap_buckets == b
        if int(np.sum(mask)) == 0:
            continue
        ys = y[mask]
        ps = oof[mask]
        sweep = sweep_thresholds(ys, ps, thresholds=thresholds, lambda_fp=args.lambda_fp)
        for row_metric in sweep:
            per_bucket_rows.append({"gap_bucket": b, **row_metric})
        best_b = pick_best(sweep)
        best_per_bucket[b] = best_b
        per_bucket_thresholds[b] = float(best_b["threshold"])

    pred_bucket = np.zeros_like(y)
    for i, b in enumerate(gap_buckets):
        thr_b = per_bucket_thresholds.get(str(b), float(best_global["threshold"]))
        pred_bucket[i] = int(oof[i] >= thr_b)
    per_bucket_overall = compute_metrics(y, pred_bucket, lambda_fp=args.lambda_fp)

    best_global_thr = float(best_global["threshold"])
    pred_global = (oof >= best_global_thr).astype(np.int64)

    full_bundle = train_full_model(x, y, features=features, seed=args.seed)
    full_bundle["threshold_global_oof"] = best_global_thr
    full_bundle["threshold_per_bucket_oof"] = per_bucket_thresholds

    out_oof_rows: List[Dict[str, object]] = []
    for i, r in enumerate(rows):
        row = dict(r)
        row["dataset_key"] = str(r.get("cache_key", ""))
        row["gap_bucket"] = str(gap_buckets[i])
        row["lp_gap"] = float(r.get("lp_gap_top2_minus_top1", float("nan")))
        row["sc_gap"] = float(r.get("sc_gap_top2_minus_top1", float("nan")))
        row["len_gap"] = float(r.get("len_gap_top2_minus_top1", float("nan")))
        row["top1_id"] = str(r.get("top1_sid", ""))
        row["top2_id"] = str(r.get("top2_sid", ""))
        row["p_flip_oof"] = float(oof[i])
        row["head_global_decision_oof"] = int(pred_global[i])
        row["head_bucket_decision_oof"] = int(pred_bucket[i])
        out_oof_rows.append(row)

    Path(args.oof_pred_output).parent.mkdir(parents=True, exist_ok=True)

    if out_oof_rows:
        fieldnames = list(out_oof_rows[0].keys())
        with Path(args.oof_pred_output).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_oof_rows)

    with Path(args.sweep_global_csv).open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(global_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(global_rows)

    with Path(args.sweep_bucket_csv).open("w", newline="", encoding="utf-8") as f:
        if per_bucket_rows:
            fieldnames = list(per_bucket_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_bucket_rows)

    sweep_global_json = {
        "mode": "global",
        "lambda_fp": args.lambda_fp,
        "threshold_grid": [float(x) for x in thresholds.tolist()],
        "rows": global_rows,
        "best": best_global,
    }
    Path(args.sweep_global_json).write_text(json.dumps(sweep_global_json, ensure_ascii=False, indent=2))

    sweep_bucket_json = {
        "mode": "per_bucket",
        "lambda_fp": args.lambda_fp,
        "buckets": FIXED_BUCKETS,
        "best_per_bucket": best_per_bucket,
        "overall_when_using_bucket_thresholds": per_bucket_overall,
        "rows": per_bucket_rows,
    }
    Path(args.sweep_bucket_json).write_text(json.dumps(sweep_bucket_json, ensure_ascii=False, indent=2))

    selected = {
        "mode": "both",
        "lambda_fp": float(args.lambda_fp),
        "best_global_threshold": best_global_thr,
        "best_global_metrics": metrics_from_threshold(y, oof, best_global_thr, lambda_fp=args.lambda_fp),
        "best_per_bucket_thresholds": per_bucket_thresholds,
        "best_per_bucket_metrics": per_bucket_overall,
        "threshold_grid": [float(x) for x in thresholds.tolist()],
    }
    Path(args.selected_thresholds_output).write_text(json.dumps(selected, ensure_ascii=False, indent=2))

    with Path(args.full_model_output).open("wb") as f:
        pickle.dump(full_bundle, f)

    oof_summary = {
        "input": args.input,
        "n_rows": int(n),
        "positive_count": int(pos_count),
        "negative_count": int(neg_count),
        "splitter": splitter_name,
        "n_splits": int(args.n_splits),
        "features": features,
        "split_logs": split_logs,
        "outputs": {
            "oof_predictions": args.oof_pred_output,
            "oof_model_full": args.full_model_output,
            "sweep_global_csv": args.sweep_global_csv,
            "sweep_bucket_csv": args.sweep_bucket_csv,
            "selected_thresholds": args.selected_thresholds_output,
        },
    }
    Path(args.oof_summary_output).write_text(json.dumps(oof_summary, ensure_ascii=False, indent=2))

    print(f"rows={n} pos={pos_count} neg={neg_count}")
    print(f"splitter={splitter_name} n_splits={args.n_splits}")
    print(f"best_global_threshold={best_global_thr:.4f} net_gain={best_global['net_gain']:.4f} flips={best_global['predicted_flip_count']}")
    print(f"per_bucket_thresholds={json.dumps(per_bucket_thresholds, ensure_ascii=False)}")
    print(f"wrote {args.oof_pred_output}")
    print(f"wrote {args.oof_summary_output}")
    print(f"wrote {args.full_model_output}")
    print(f"wrote {args.selected_thresholds_output}")


if __name__ == "__main__":
    main()
