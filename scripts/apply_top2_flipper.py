#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.selectors.base import SelectorContext
from nad.core.views.reader import CacheReader
from plugins.medoid_tail_warning import MedoidTailWarningSelector
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json'
DEFAULT_MODEL = '/home/jovyan/work/NAD_Next/result/top2_flipper_model_all.pkl'
DEFAULT_OUTPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_flipper_submit.json'
DEFAULT_NOTES = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_flipper_submit_notes.json'

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
    p = argparse.ArgumentParser(description='Apply trained top2 flipper to a submission score map.')
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--notes-output', default=DEFAULT_NOTES)
    p.add_argument('--method-name', default='nad_mixed_v2_top2_flipper')
    p.add_argument('--target-cache-keys', default='', help='Comma-separated cache keys; empty means all.')
    p.add_argument('--threshold', type=float, default=None, help='Override model threshold.')
    p.add_argument('--max-gap', type=float, default=0.002, help='Only consider top2 if gap <= max-gap. <=0 means no filter.')
    p.add_argument('--max-flips-total', type=int, default=5)
    p.add_argument('--max-flips-per-cache', type=int, default=2)
    p.add_argument('--score-bump', type=float, default=1e-9)
    return p.parse_args()


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


def build_feature_dict(
    reader: CacheReader,
    plugin: MedoidTailWarningSelector,
    problem_id: str,
    top1_sid: str,
    top2_sid: str,
    gap: float,
) -> Dict[str, float]:
    m1 = metric_means(reader, int(top1_sid))
    m2 = metric_means(reader, int(top2_sid))

    ctx = SelectorContext(cache=reader, problem_id=str(problem_id), run_ids=[int(top1_sid), int(top2_sid)], views=[])
    plugin.bind(ctx)
    t1 = plugin._tail_metrics(int(top1_sid)) or {}
    t2 = plugin._tail_metrics(int(top2_sid)) or {}

    t1_tail = float(t1.get('tail_new_ratio', np.nan))
    t2_tail = float(t2.get('tail_new_ratio', np.nan))
    t1_plateau = float(t1.get('plateau_progress', np.nan))
    t2_plateau = float(t2.get('plateau_progress', np.nan))
    t1_final = float(t1.get('final_count', np.nan))
    t2_final = float(t2.get('final_count', np.nan))

    return {
        'gap': float(gap),
        'logprob_delta_top2_minus_top1': m2['tok_logprob_mean'] - m1['tok_logprob_mean'],
        'selfcert_delta_top2_minus_top1': m2['tok_selfcert_mean'] - m1['tok_selfcert_mean'],
        'conf_delta_top2_minus_top1': m2['tok_conf_mean'] - m1['tok_conf_mean'],
        'neg_entropy_delta_top2_minus_top1': m2['tok_neg_entropy_mean'] - m1['tok_neg_entropy_mean'],
        'gini_delta_top2_minus_top1': m2['tok_gini_mean'] - m1['tok_gini_mean'],
        'tail_delta_top2_minus_top1': t2_tail - t1_tail,
        'plateau_delta_top2_minus_top1': t2_plateau - t1_plateau,
        'final_count_delta_top2_minus_top1': t2_final - t1_final,
        'top1_tok_logprob_mean': m1['tok_logprob_mean'],
        'top2_tok_logprob_mean': m2['tok_logprob_mean'],
        'top1_tok_selfcert_mean': m1['tok_selfcert_mean'],
        'top2_tok_selfcert_mean': m2['tok_selfcert_mean'],
        'top1_tail_new_ratio': t1_tail,
        'top2_tail_new_ratio': t2_tail,
        'top1_plateau_progress': t1_plateau,
        'top2_plateau_progress': t2_plateau,
    }


def main() -> None:
    args = parse_args()

    with Path(args.model).open('rb') as f:
        bundle = pickle.load(f)

    feat_names = list(bundle['features'])
    if feat_names != FEATURES:
        raise ValueError('Feature mismatch between script and model bundle.')

    med = np.asarray(bundle['nan_medians'], dtype=np.float64)
    scaler = bundle['scaler']
    model = bundle['model']
    thr = float(args.threshold) if args.threshold is not None else float(bundle['threshold'])

    inp = json.loads(Path(args.input).read_text())
    out_scores = copy.deepcopy(inp['scores'])

    if args.target_cache_keys.strip():
        target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]
    else:
        target_cache_keys = list(out_scores.keys())

    readers = {}
    plugins = {}
    for ck in target_cache_keys:
        if ck in DEFAULT_CACHE_MAP:
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

    candidates = []
    notes = {
        'task': inp.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'input': args.input,
        'output': args.output,
        'model': args.model,
        'threshold': thr,
        'max_gap': args.max_gap,
        'max_flips_total': args.max_flips_total,
        'max_flips_per_cache': args.max_flips_per_cache,
        'target_cache_keys': target_cache_keys,
        'considered_count': 0,
        'candidate_count': 0,
        'applied_count': 0,
        'applied_by_cache': {},
        'applied': [],
    }

    for ck, problem_map in out_scores.items():
        if ck not in target_cache_keys or ck not in readers:
            continue
        reader = readers[ck]
        plugin = plugins[ck]

        for pid, sid_scores in problem_map.items():
            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue
            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            gap = float(top1_score - top2_score)

            if args.max_gap > 0 and gap > args.max_gap:
                continue

            notes['considered_count'] += 1
            fd = build_feature_dict(reader, plugin, str(pid), top1_sid, top2_sid, gap)
            x = np.asarray([fd[k] for k in FEATURES], dtype=np.float64)
            x = np.where(np.isfinite(x), x, med)
            p_flip = float(model.predict_proba(scaler.transform(x.reshape(1, -1)))[0, 1])

            if p_flip >= thr:
                candidates.append({
                    'cache_key': ck,
                    'problem_id': str(pid),
                    'top1_sid': top1_sid,
                    'top2_sid': top2_sid,
                    'gap': gap,
                    'p_flip': p_flip,
                })

    notes['candidate_count'] = len(candidates)
    candidates.sort(key=lambda x: x['p_flip'], reverse=True)

    applied_by_cache: Dict[str, int] = {}
    applied_set = set()

    for c in candidates:
        if notes['applied_count'] >= args.max_flips_total:
            break
        ck = c['cache_key']
        if applied_by_cache.get(ck, 0) >= args.max_flips_per_cache:
            continue
        key = (ck, c['problem_id'])
        if key in applied_set:
            continue

        sid_scores = out_scores[ck][c['problem_id']]
        adjusted = {str(s): float(v) for s, v in sid_scores.items()}
        adjusted[c['top2_sid']] = max(adjusted.values()) + float(args.score_bump)
        out_scores[ck][c['problem_id']] = adjusted

        applied_set.add(key)
        applied_by_cache[ck] = applied_by_cache.get(ck, 0) + 1
        notes['applied_count'] += 1
        notes['applied'].append(c)

    notes['applied_by_cache'] = applied_by_cache

    out = {
        'task': inp.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'scores': out_scores,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps(notes, ensure_ascii=False, indent=2))

    print(f"considered={notes['considered_count']} candidates={notes['candidate_count']} applied={notes['applied_count']}")
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')


if __name__ == '__main__':
    main()
