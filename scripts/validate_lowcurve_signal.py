#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.uniques import extract_tokenwise_counts


METRICS = [
    'mid_abs_mean',
    'mid_rel_mean',
    'final_count',
    'auc_abs',
    'auc_rel',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate the low-curve hypothesis: correct runs tend to stay lower than incorrect runs '
                    'in token-wise cumulative unique-neuron curves.'
    )
    parser.add_argument('--cache-root', required=True, help='Path to a NAD cache directory containing meta.json')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/lowcurve_signal_validation.json')
    parser.add_argument('--progress-start', type=float, default=0.40,
                        help='Start of normalized progress window used for the primary low-curve score')
    parser.add_argument('--progress-end', type=float, default=0.80,
                        help='End of normalized progress window used for the primary low-curve score')
    parser.add_argument('--window-points', type=int, default=9,
                        help='Number of interpolation points inside [progress-start, progress-end]')
    parser.add_argument('--auc-points', type=int, default=101,
                        help='Number of interpolation points for AUC-style metrics')
    parser.add_argument('--token-axis', choices=['row', 'tokens'], default='row')
    parser.add_argument('--max-problems', type=int, default=None,
                        help='Optional cap on number of problems for smoke tests')
    parser.add_argument('--problem-ids', default=None,
                        help='Comma-separated subset of problem IDs to analyze')
    parser.add_argument('--notes', default='')
    return parser.parse_args()


def grouped_run_ids(meta: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for sample_idx, sample in enumerate(meta.get('samples', [])):
        groups[str(sample['problem_id'])].append(sample_idx)
    return dict(groups)


def _safe_progress(tokens: np.ndarray) -> np.ndarray:
    if tokens.size <= 1:
        return np.array([0.0], dtype=np.float64) if tokens.size == 1 else np.array([], dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.float64)
    span = float(tokens[-1] - tokens[0])
    if span <= 1e-12:
        return np.linspace(0.0, 1.0, num=tokens.size, dtype=np.float64)
    return (tokens - tokens[0]) / span


def _interp_curve(progress: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    if values.size == 1:
        return np.full_like(grid, float(values[0]), dtype=np.float64)
    return np.interp(grid, progress, values.astype(np.float64, copy=False))



def compute_curve_metrics(
    reader: CacheReader,
    run_id: int,
    progress_grid: np.ndarray,
    auc_grid: np.ndarray,
    token_axis: str,
) -> Dict[str, float] | None:
    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    rows_keys = reader.rows_keys
    rows_slice_ids = reader.rows_slice_ids
    rows_trp = reader.rows_token_row_ptr

    if rows_srp is None or rows_rp is None or rows_keys is None:
        raise RuntimeError('rows/ bank not available in cache; cannot compute token-wise cumulative curves')

    tokens, counts = extract_tokenwise_counts(
        run_id=run_id,
        rows_srp=rows_srp,
        rows_rp=rows_rp,
        rows_keys=rows_keys,
        rows_slice_ids=rows_slice_ids,
        rows_trp=rows_trp,
        token_axis=token_axis,
    )
    if counts.size == 0:
        return None

    progress = _safe_progress(tokens)
    counts = counts.astype(np.float64, copy=False)
    final_count = float(counts[-1])
    denom = max(final_count, 1.0)
    rel_counts = counts / denom

    mid_abs = _interp_curve(progress, counts, progress_grid)
    mid_rel = _interp_curve(progress, rel_counts, progress_grid)
    auc_abs_curve = _interp_curve(progress, counts, auc_grid)
    auc_rel_curve = _interp_curve(progress, rel_counts, auc_grid)

    return {
        'mid_abs_mean': float(np.mean(mid_abs)),
        'mid_rel_mean': float(np.mean(mid_rel)),
        'final_count': final_count,
        'auc_abs': float(np.trapezoid(auc_abs_curve, auc_grid)),
        'auc_rel': float(np.trapezoid(auc_rel_curve, auc_grid)),
        'num_points': int(counts.size),
        'final_progress': float(progress[-1]) if progress.size else 0.0,
    }



def pairwise_preference(correct_values: Iterable[float], incorrect_values: Iterable[float]) -> Tuple[float, int]:
    correct = np.asarray(list(correct_values), dtype=np.float64)
    incorrect = np.asarray(list(incorrect_values), dtype=np.float64)
    if correct.size == 0 or incorrect.size == 0:
        return 0.0, 0
    comp = correct[:, None] - incorrect[None, :]
    wins = float((comp < 0).sum()) + 0.5 * float((comp == 0).sum())
    total = int(comp.size)
    return wins, total



def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    meta = json.loads((cache_root / 'meta.json').read_text())
    groups = grouped_run_ids(meta)
    correctness_map = load_correctness_map(str(cache_root))
    reader = CacheReader(str(cache_root))

    wanted_problem_ids = None
    if args.problem_ids:
        wanted_problem_ids = {pid.strip() for pid in args.problem_ids.split(',') if pid.strip()}

    progress_grid = np.linspace(args.progress_start, args.progress_end, num=max(2, args.window_points), dtype=np.float64)
    auc_grid = np.linspace(0.0, 1.0, num=max(5, args.auc_points), dtype=np.float64)

    problem_ids = sorted(groups.keys(), key=lambda x: (len(x), x))
    if wanted_problem_ids is not None:
        problem_ids = [pid for pid in problem_ids if pid in wanted_problem_ids]
    if args.max_problems is not None:
        problem_ids = problem_ids[:args.max_problems]

    per_problem: Dict[str, Dict[str, object]] = {}
    aggregate_metric_values = {
        metric: {
            'correct': [],
            'incorrect': [],
            'pairwise_wins': 0.0,
            'pairwise_total': 0,
            'top1_correct': 0,
            'top1_total': 0,
            'mean_pref_correct': 0,
            'mean_pref_total': 0,
            'strict_pref_correct': 0,
            'strict_pref_total': 0,
        }
        for metric in METRICS
    }

    total_runs = 0
    skipped_no_curve = 0
    skipped_single_class = 0

    for problem_id in problem_ids:
        run_ids = groups[problem_id]
        problem_runs = []

        for run_id in run_ids:
            metrics = compute_curve_metrics(
                reader=reader,
                run_id=run_id,
                progress_grid=progress_grid,
                auc_grid=auc_grid,
                token_axis=args.token_axis,
            )
            if metrics is None:
                skipped_no_curve += 1
                continue
            is_correct = bool(correctness_map.get(int(run_id), False))
            row = {
                'sample_id': int(run_id),
                'is_correct': is_correct,
                **metrics,
            }
            problem_runs.append(row)
            total_runs += 1

        correct_runs = [r for r in problem_runs if r['is_correct']]
        incorrect_runs = [r for r in problem_runs if not r['is_correct']]

        if not correct_runs or not incorrect_runs:
            skipped_single_class += 1
            per_problem[problem_id] = {
                'status': 'skipped_single_class',
                'num_runs': len(problem_runs),
                'num_correct': len(correct_runs),
                'num_incorrect': len(incorrect_runs),
            }
            continue

        per_problem_metrics: Dict[str, object] = {}
        for metric in METRICS:
            correct_values = [float(r[metric]) for r in correct_runs]
            incorrect_values = [float(r[metric]) for r in incorrect_runs]
            wins, total = pairwise_preference(correct_values, incorrect_values)
            agg = aggregate_metric_values[metric]
            agg['correct'].extend(correct_values)
            agg['incorrect'].extend(incorrect_values)
            agg['pairwise_wins'] += wins
            agg['pairwise_total'] += total
            agg['mean_pref_total'] += 1
            if np.mean(correct_values) < np.mean(incorrect_values):
                agg['mean_pref_correct'] += 1
            agg['strict_pref_total'] += 1
            if min(correct_values) < min(incorrect_values):
                agg['strict_pref_correct'] += 1

            best_run = min(problem_runs, key=lambda row: (float(row[metric]), int(row['sample_id'])))
            agg['top1_total'] += 1
            if bool(best_run['is_correct']):
                agg['top1_correct'] += 1

            per_problem_metrics[metric] = {
                'correct_mean': float(np.mean(correct_values)),
                'incorrect_mean': float(np.mean(incorrect_values)),
                'correct_min': float(np.min(correct_values)),
                'incorrect_min': float(np.min(incorrect_values)),
                'gap_incorrect_minus_correct_mean': float(np.mean(incorrect_values) - np.mean(correct_values)),
                'pairwise_pref_correct': float(wins / total) if total else None,
                'top1_selected_sample_id': int(best_run['sample_id']),
                'top1_selected_is_correct': bool(best_run['is_correct']),
            }

        per_problem[problem_id] = {
            'status': 'ok',
            'num_runs': len(problem_runs),
            'num_correct': len(correct_runs),
            'num_incorrect': len(incorrect_runs),
            'metrics': per_problem_metrics,
        }

    summary_metrics: Dict[str, Dict[str, object]] = {}
    for metric in METRICS:
        agg = aggregate_metric_values[metric]
        correct_values = np.asarray(agg['correct'], dtype=np.float64)
        incorrect_values = np.asarray(agg['incorrect'], dtype=np.float64)
        summary_metrics[metric] = {
            'overall_correct_mean': float(np.mean(correct_values)) if correct_values.size else None,
            'overall_incorrect_mean': float(np.mean(incorrect_values)) if incorrect_values.size else None,
            'overall_gap_incorrect_minus_correct': (
                float(np.mean(incorrect_values) - np.mean(correct_values))
                if correct_values.size and incorrect_values.size else None
            ),
            'pairwise_pref_correct': (
                float(agg['pairwise_wins'] / agg['pairwise_total']) if agg['pairwise_total'] else None
            ),
            'problem_mean_pref_correct': (
                float(agg['mean_pref_correct'] / agg['mean_pref_total']) if agg['mean_pref_total'] else None
            ),
            'problem_strict_pref_correct': (
                float(agg['strict_pref_correct'] / agg['strict_pref_total']) if agg['strict_pref_total'] else None
            ),
            'top1_accuracy_if_pick_min_score': (
                float(agg['top1_correct'] / agg['top1_total']) if agg['top1_total'] else None
            ),
            'num_evaluable_problems': int(agg['top1_total']),
        }

    summary = {
        'cache_root': str(cache_root),
        'notes': args.notes,
        'primary_metric': 'mid_abs_mean',
        'parameters': {
            'progress_start': args.progress_start,
            'progress_end': args.progress_end,
            'window_points': args.window_points,
            'auc_points': args.auc_points,
            'token_axis': args.token_axis,
            'max_problems': args.max_problems,
            'problem_ids': sorted(wanted_problem_ids) if wanted_problem_ids else None,
        },
        'counts': {
            'requested_problem_count': len(problem_ids),
            'total_runs_with_curve': total_runs,
            'skipped_no_curve_runs': skipped_no_curve,
            'skipped_single_class_problems': skipped_single_class,
        },
        'metrics': summary_metrics,
    }

    output = {
        'summary': summary,
        'per_problem': per_problem,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f'wrote {out_path}')
    print(f"primary metric mid_abs_mean: pairwise_pref_correct={summary_metrics['mid_abs_mean']['pairwise_pref_correct']}")
    print(f"primary metric mid_abs_mean: top1_accuracy_if_pick_min_score={summary_metrics['mid_abs_mean']['top1_accuracy_if_pick_min_score']}")


if __name__ == '__main__':
    main()
