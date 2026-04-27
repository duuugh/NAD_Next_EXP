#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader, ViewSpec, CutSpec, Agg, CutType, Order
from nad.core.distance.engine import DistanceEngine, DistanceSpec

DEFAULT_CACHE_MAP = {
    'DS-R1/aime24': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610',
    'DS-R1/aime25': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548',
    'DS-R1/brumo25': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142',
    'DS-R1/gpqa': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853',
    'DS-R1/hmmt25': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151',
    'DS-R1/lcb_v5': '/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808',
    'Qwen3-4B/aime24': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/aime24/cache_neuron_output_1_act_no_rms_20250917_070758',
    'Qwen3-4B/aime25': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/aime25/cache_neuron_output_1_act_no_rms_20250917_091548',
    'Qwen3-4B/brumo25': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/brumo25/cache_neuron_output_1_act_no_rms_20251125_024402',
    'Qwen3-4B/gpqa': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/gpqa/cache_neuron_output_1_act_no_rms_20250917_143310',
    'Qwen3-4B/hmmt25': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/hmmt25/cache_neuron_output_1_act_no_rms_20251125_054608',
    'Qwen3-4B/lcb_v5': '/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/livecodebench_v5/cache_neuron_output_1_act_no_rms_20250920_094942',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build hybrid best-of-n submission using KNN + token confidence signals.')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/best_of_n_hybrid_v1_wrapped.json')
    parser.add_argument('--method-name', default='best_of_n_hybrid_v1')
    parser.add_argument('--knn-weight', type=float, default=0.6)
    parser.add_argument('--metric-weight', type=float, default=0.4)
    parser.add_argument('--metric', choices=['tok_selfcert', 'tok_conf', 'tok_neg_entropy', 'tok_logprob', 'tok_gini'], default='tok_selfcert')
    parser.add_argument('--reduction', choices=['mean', 'min_group'], default='mean')
    parser.add_argument('--group-size', type=int, default=20)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--limit-cache-keys', default=None, help='Comma-separated subset of cache keys to run')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/best_of_n_hybrid_v1_notes.json')
    return parser.parse_args()


def grouped_run_ids(meta: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for sample_idx, sample in enumerate(meta['samples']):
        groups[str(sample['problem_id'])].append(sample_idx)
    return groups


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - lo) / (hi - lo)


def least_grouped_strict(x: np.ndarray, w: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float('-inf')
    if w is None or w <= 1 or n < w:
        return float(np.mean(x))
    c = np.cumsum(x)
    sums = c[w - 1:] - np.concatenate(([0.0], c[:-w]))
    means = sums / float(w)
    return float(np.min(means))


def get_metric_array(token_view, metric: str) -> np.ndarray:
    arr = getattr(token_view, metric)
    if arr is None:
        raise ValueError(f'metric {metric} not available in cache')
    return arr


def metric_quality(arr: np.ndarray, metric: str, reduction: str, group_size: int) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if reduction == 'mean':
        base = float(np.mean(arr))
    else:
        base = least_grouped_strict(arr, group_size)
    if metric == 'tok_conf':
        return -base
    return base


def compute_problem_scores(cache_root: str, topk: int, metric: str, reduction: str, group_size: int,
                           knn_weight: float, metric_weight: float) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    reader = CacheReader(cache_root)
    meta = json.loads((Path(cache_root) / 'meta.json').read_text())
    groups = grouped_run_ids(meta)
    rp = reader.row_ptr

    all_scores: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, int] = {}

    for problem_id, run_ids in groups.items():
        kmax_lengths = np.asarray([int(rp[rid + 1] - rp[rid]) for rid in run_ids], dtype=np.int32)
        common_k = int(kmax_lengths.max())
        views = []
        metric_scores = []

        for rid, kmax in zip(run_ids, kmax_lengths):
            k = min(common_k, int(kmax))
            view_spec = ViewSpec(agg=Agg.MAX, cut=CutSpec(CutType.TOPK, float(k)), order=Order.BY_KEY)
            view = reader.get_run_view(rid, view_spec, normalize_l1=True)
            safe_weights = np.nan_to_num(view.weights, nan=0.0, posinf=0.0, neginf=0.0)
            views.append(type(view)(keys=view.keys, weights=safe_weights))
            token_view = reader.get_token_view(int(rid))
            metric_arr = get_metric_array(token_view, metric)
            metric_scores.append(metric_quality(metric_arr, metric, reduction, group_size))

        if len(run_ids) == 1:
            knn_scores = np.array([1.0], dtype=np.float64)
        else:
            D = DistanceEngine(DistanceSpec('wj')).dense_matrix(views)
            S = 1.0 - D
            k_eff = min(topk, max(1, len(run_ids) - 1))
            knn_scores = np.zeros(len(run_ids), dtype=np.float64)
            for i in range(len(run_ids)):
                sims = np.delete(S[i], i)
                top = np.partition(sims, -k_eff)[-k_eff:]
                knn_scores[i] = float(top.mean())

        metric_scores = np.asarray(metric_scores, dtype=np.float64)
        knn_scores = np.nan_to_num(knn_scores, nan=0.0, posinf=0.0, neginf=0.0)
        metric_scores = np.nan_to_num(metric_scores, nan=0.0, posinf=0.0, neginf=0.0)
        knn_norm = normalize_minmax(knn_scores)
        metric_norm = normalize_minmax(metric_scores)
        final_scores = knn_weight * knn_norm + metric_weight * metric_norm

        problem_scores: Dict[str, float] = {}
        for rid, score in zip(run_ids, final_scores.tolist()):
            problem_scores[str(rid)] = float(score)
        all_scores[problem_id] = problem_scores

        best_sid, _ = max(problem_scores.items(), key=lambda kv: kv[1])
        try:
            predictions[problem_id] = int(best_sid)
        except ValueError:
            predictions[problem_id] = best_sid

    return all_scores, predictions


def main() -> None:
    args = parse_args()
    cache_map = dict(DEFAULT_CACHE_MAP)
    if args.limit_cache_keys:
        wanted = [x.strip() for x in args.limit_cache_keys.split(',') if x.strip()]
        cache_map = {k: v for k, v in cache_map.items() if k in wanted}

    wrapped_scores = {}
    for cache_key, cache_root in cache_map.items():
        scores, predictions = compute_problem_scores(
            cache_root=cache_root,
            topk=args.topk,
            metric=args.metric,
            reduction=args.reduction,
            group_size=args.group_size,
            knn_weight=args.knn_weight,
            metric_weight=args.metric_weight,
        )
        wrapped_scores[cache_key] = {
            'predictions': predictions,
            'scores': scores,
        }
        print(f'finished {cache_key}: {len(scores)} problems')

    output = {
        'task': 'best_of_n',
        'method_name': args.method_name,
        'scores': wrapped_scores,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps({
        'task': 'best_of_n',
        'method_name': args.method_name,
        'output': args.output,
        'notes_output': args.notes_output,
        'cache_keys': list(cache_map.keys()),
        'weights': {'knn': args.knn_weight, 'metric': args.metric_weight},
        'metric': args.metric,
        'reduction': args.reduction,
        'group_size': args.group_size,
        'topk': args.topk,
    }, ensure_ascii=False, indent=2))
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')


if __name__ == '__main__':
    main()
