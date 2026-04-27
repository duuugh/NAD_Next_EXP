#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/top2_training_table.csv'
DEFAULT_MODEL = '/home/jovyan/work/NAD_Next/result/top2_flipper_model.pkl'
DEFAULT_METRICS = '/home/jovyan/work/NAD_Next/result/top2_flipper_metrics.json'

FEATURES = [
    'gap',
    'logprob_delta_top2_minus_top1',
    'selfcert_delta_top2_minus_top1',
    'conf_delta_top2_minus_top1',
    'neg_entropy_delta_top2_minus_top1',
    'gini_delta_top2_minus_top1',
    'tail_delta_top2_minus_top1',
    'plateau_delta_top2_minus_top1',
    'final_count_delta_top2_minus_top1',
    'top1_tok_logprob_mean',
    'top2_tok_logprob_mean',
    'top1_tok_selfcert_mean',
    'top2_tok_selfcert_mean',
    'top1_tail_new_ratio',
    'top2_tail_new_ratio',
    'top1_plateau_progress',
    'top2_plateau_progress',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train a conservative top2 flip logistic model.')
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--model-output', default=DEFAULT_MODEL)
    p.add_argument('--metrics-output', default=DEFAULT_METRICS)
    p.add_argument('--valid-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target-max-flip-rate', type=float, default=0.03)
    p.add_argument('--min-threshold', type=float, default=0.5)
    p.add_argument('--max-threshold', type=float, default=0.98)
    p.add_argument('--threshold-steps', type=int, default=49)
    return p.parse_args()


def stable_split_key(cache_key: str, problem_id: str, seed: int) -> float:
    h = hashlib.md5(f'{cache_key}::{problem_id}::{seed}'.encode('utf-8')).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def to_float(x: str) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float('nan')


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def build_xy(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feats = []
    labels = []
    cache_keys = []
    problem_ids = []
    for r in rows:
        y = r.get('label_should_flip')
        if y is None or y == '':
            continue
        x = [to_float(r.get(col, '')) for col in FEATURES]
        feats.append(x)
        labels.append(int(y))
        cache_keys.append(r['cache_key'])
        problem_ids.append(r['problem_id'])
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.int64), cache_keys, problem_ids


def fill_nan_with_train_median(x_train: np.ndarray, x_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train_f = np.where(np.isfinite(x_train), x_train, med)
    x_valid_f = np.where(np.isfinite(x_valid), x_valid, med)
    return x_train_f, x_valid_f, med


def compute_threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    flip_rate = (tp + fp) / max(y.size, 1)
    return {
        'threshold': float(thr),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'flip_rate': float(flip_rate),
    }


def pick_threshold(metrics: List[Dict[str, float]], target_max_flip_rate: float) -> Dict[str, float]:
    feasible = [m for m in metrics if m['flip_rate'] <= target_max_flip_rate and (m['tp'] + m['fp']) > 0]
    if feasible:
        feasible.sort(key=lambda m: (m['precision'], m['f1'], -m['flip_rate']), reverse=True)
        return feasible[0]
    any_pred = [m for m in metrics if (m['tp'] + m['fp']) > 0]
    if any_pred:
        any_pred.sort(key=lambda m: (m['precision'], m['f1']), reverse=True)
        return any_pred[0]
    return metrics[-1]


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    x, y, cks, pids = build_xy(rows)
    if x.size == 0:
        raise ValueError('No labeled rows found. Build table with labeled data first.')

    split_vals = np.array([stable_split_key(ck, pid, args.seed) for ck, pid in zip(cks, pids)], dtype=np.float64)
    valid_mask = split_vals < float(args.valid_ratio)
    train_mask = ~valid_mask

    if np.sum(valid_mask) == 0 or np.sum(train_mask) == 0:
        raise ValueError('Train/valid split failed. Adjust valid-ratio.')

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[valid_mask], y[valid_mask]

    x_train, x_valid, med = fill_nan_with_train_median(x_train, x_valid)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_valid_s = scaler.transform(x_valid)

    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=args.seed)
    model.fit(x_train_s, y_train)

    p_valid = model.predict_proba(x_valid_s)[:, 1]
    auc = float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else float('nan')

    thresholds = np.linspace(args.min_threshold, args.max_threshold, num=max(2, args.threshold_steps))
    all_metrics = [compute_threshold_metrics(y_valid, p_valid, float(t)) for t in thresholds]
    chosen = pick_threshold(all_metrics, args.target_max_flip_rate)

    bundle = {
        'features': FEATURES,
        'nan_medians': med.astype(np.float64),
        'scaler': scaler,
        'model': model,
        'threshold': float(chosen['threshold']),
        'seed': int(args.seed),
    }

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.model_output).open('wb') as f:
        pickle.dump(bundle, f)

    metrics = {
        'input': args.input,
        'model_output': args.model_output,
        'features': FEATURES,
        'train_size': int(y_train.size),
        'valid_size': int(y_valid.size),
        'train_positive_rate': float(np.mean(y_train)),
        'valid_positive_rate': float(np.mean(y_valid)),
        'valid_auc': auc,
        'target_max_flip_rate': float(args.target_max_flip_rate),
        'chosen_threshold': chosen,
        'threshold_grid_metrics': all_metrics,
    }
    Path(args.metrics_output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f'train={y_train.size} valid={y_valid.size}')
    print(f'valid_auc={auc:.4f}')
    print(f"chosen_threshold={chosen['threshold']:.4f} precision={chosen['precision']:.4f} recall={chosen['recall']:.4f} flip_rate={chosen['flip_rate']:.4f}")
    print(f'wrote {args.model_output}')
    print(f'wrote {args.metrics_output}')


if __name__ == '__main__':
    main()
