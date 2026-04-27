#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader
from nad.core.selectors.base import SelectorContext
from nad.ops.accuracy import load_correctness_map
from plugins.medoid_tail_warning import MedoidTailWarningSelector
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json'
DEFAULT_OUTPUT = '/home/jovyan/work/NAD_Next/result/top2_training_table.csv'
DEFAULT_META = '/home/jovyan/work/NAD_Next/result/top2_training_table_meta.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build top2 flip training table from a submission score map.')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--meta-output', default=DEFAULT_META)
    parser.add_argument('--target-cache-keys', default='', help='Comma-separated cache keys; empty means all in input.')
    parser.add_argument('--max-gap', type=float, default=0.002, help='Keep rows with top1-top2 <= max_gap; <=0 means no filter.')
    parser.add_argument('--include-unlabeled', action='store_true', help='Keep rows even when ground truth is unavailable.')
    return parser.parse_args()


def ranked_items(sid_scores: Dict[str, float]) -> List[tuple[str, float]]:
    return sorted(((str(s), float(v)) for s, v in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))


def safe_mean(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float('nan')
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return float(np.mean(a))


def metric_means(reader: CacheReader, sid: int) -> Dict[str, float]:
    tv = reader.get_token_view(int(sid))
    return {
        'tok_logprob_mean': safe_mean(getattr(tv, 'tok_logprob', None)),
        'tok_selfcert_mean': safe_mean(getattr(tv, 'tok_selfcert', None)),
        'tok_conf_mean': safe_mean(getattr(tv, 'tok_conf', None)),
        'tok_neg_entropy_mean': safe_mean(getattr(tv, 'tok_neg_entropy', None)),
        'tok_gini_mean': safe_mean(getattr(tv, 'tok_gini', None)),
    }


def main() -> None:
    args = parse_args()
    inp = json.loads(Path(args.input).read_text())
    scores = inp['scores']

    if args.target_cache_keys.strip():
        target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]
    else:
        target_cache_keys = list(scores.keys())

    readers: Dict[str, CacheReader] = {}
    plugins: Dict[str, MedoidTailWarningSelector] = {}
    correctness_maps: Dict[str, Optional[Dict[int, bool]]] = {}

    for ck in target_cache_keys:
        if ck not in DEFAULT_CACHE_MAP:
            continue
        readers[ck] = CacheReader(DEFAULT_CACHE_MAP[ck])
        plugins[ck] = MedoidTailWarningSelector(
            gap_abs=0.01,
            tail_start=0.85,
            plateau_fraction=0.98,
            tail_new_ratio_warn=0.015,
            plateau_progress_warn=0.85,
            min_tail_ratio_advantage=0.005,
            min_plateau_progress_advantage=0.03,
        )
        try:
            correctness_maps[ck] = load_correctness_map(DEFAULT_CACHE_MAP[ck])
        except Exception:
            correctness_maps[ck] = None

    rows: List[Dict[str, object]] = []
    stats = {
        'input': args.input,
        'target_cache_keys': target_cache_keys,
        'max_gap': args.max_gap,
        'include_unlabeled': args.include_unlabeled,
        'total_candidates': 0,
        'kept_rows': 0,
        'dropped_gap': 0,
        'dropped_missing_cache': 0,
        'dropped_unlabeled': 0,
        'cache_stats': {},
    }

    for ck in target_cache_keys:
        problem_map = scores.get(ck)
        if problem_map is None or ck not in readers:
            continue

        reader = readers[ck]
        plugin = plugins[ck]
        corr_map = correctness_maps.get(ck)

        cache_total = 0
        cache_kept = 0

        for pid, sid_scores in problem_map.items():
            cache_total += 1
            stats['total_candidates'] += 1

            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue

            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            gap = float(top1_score - top2_score)

            if args.max_gap > 0 and gap > args.max_gap:
                stats['dropped_gap'] += 1
                continue

            y_top1 = None
            y_top2 = None
            label_should_flip = None
            if corr_map is not None:
                y_top1 = bool(corr_map.get(int(top1_sid), False))
                y_top2 = bool(corr_map.get(int(top2_sid), False))
                label_should_flip = int((not y_top1) and y_top2)
            elif not args.include_unlabeled:
                stats['dropped_unlabeled'] += 1
                continue

            m1 = metric_means(reader, int(top1_sid))
            m2 = metric_means(reader, int(top2_sid))

            ctx = SelectorContext(cache=reader, problem_id=str(pid), run_ids=[int(top1_sid), int(top2_sid)], views=[])
            plugin.bind(ctx)
            t1 = plugin._tail_metrics(int(top1_sid)) or {}
            t2 = plugin._tail_metrics(int(top2_sid)) or {}

            t1_tail = float(t1.get('tail_new_ratio', np.nan))
            t2_tail = float(t2.get('tail_new_ratio', np.nan))
            t1_plateau = float(t1.get('plateau_progress', np.nan))
            t2_plateau = float(t2.get('plateau_progress', np.nan))
            t1_final = float(t1.get('final_count', np.nan))
            t2_final = float(t2.get('final_count', np.nan))

            row = {
                'cache_key': ck,
                'problem_id': str(pid),
                'top1_sid': top1_sid,
                'top2_sid': top2_sid,
                'top1_score': float(top1_score),
                'top2_score': float(top2_score),
                'gap': gap,
                'y_top1': y_top1,
                'y_top2': y_top2,
                'label_should_flip': label_should_flip,

                'top1_tok_logprob_mean': m1['tok_logprob_mean'],
                'top2_tok_logprob_mean': m2['tok_logprob_mean'],
                'logprob_delta_top2_minus_top1': m2['tok_logprob_mean'] - m1['tok_logprob_mean'],

                'top1_tok_selfcert_mean': m1['tok_selfcert_mean'],
                'top2_tok_selfcert_mean': m2['tok_selfcert_mean'],
                'selfcert_delta_top2_minus_top1': m2['tok_selfcert_mean'] - m1['tok_selfcert_mean'],

                'top1_tok_conf_mean': m1['tok_conf_mean'],
                'top2_tok_conf_mean': m2['tok_conf_mean'],
                'conf_delta_top2_minus_top1': m2['tok_conf_mean'] - m1['tok_conf_mean'],

                'top1_tok_neg_entropy_mean': m1['tok_neg_entropy_mean'],
                'top2_tok_neg_entropy_mean': m2['tok_neg_entropy_mean'],
                'neg_entropy_delta_top2_minus_top1': m2['tok_neg_entropy_mean'] - m1['tok_neg_entropy_mean'],

                'top1_tok_gini_mean': m1['tok_gini_mean'],
                'top2_tok_gini_mean': m2['tok_gini_mean'],
                'gini_delta_top2_minus_top1': m2['tok_gini_mean'] - m1['tok_gini_mean'],

                'top1_tail_new_ratio': t1_tail,
                'top2_tail_new_ratio': t2_tail,
                'tail_delta_top2_minus_top1': t2_tail - t1_tail,

                'top1_plateau_progress': t1_plateau,
                'top2_plateau_progress': t2_plateau,
                'plateau_delta_top2_minus_top1': t2_plateau - t1_plateau,

                'top1_final_count': t1_final,
                'top2_final_count': t2_final,
                'final_count_delta_top2_minus_top1': t2_final - t1_final,
            }
            rows.append(row)
            cache_kept += 1
            stats['kept_rows'] += 1

        stats['cache_stats'][ck] = {
            'total_candidates': cache_total,
            'kept_rows': cache_kept,
            'has_ground_truth': correctness_maps.get(ck) is not None,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    Path(args.meta_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'rows={len(rows)}')
    print(f'wrote {args.output}')
    print(f'wrote {args.meta_output}')


if __name__ == '__main__':
    main()
