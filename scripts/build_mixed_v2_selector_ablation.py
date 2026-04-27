#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader


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
    parser = argparse.ArgumentParser(
        description='Build mixed_v2 by reselecting inside baseline top-k for a small subset of cache keys.'
    )
    parser.add_argument('--input', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top3_selfcert.json')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top3_selfcert_notes.json')
    parser.add_argument('--method-name', default='nad_mixed_v2_aime_top3_selfcert')
    parser.add_argument('--target-cache-keys', default='DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--metric', choices=['tok_selfcert', 'tok_conf', 'tok_neg_entropy', 'tok_logprob', 'tok_gini'], default='tok_selfcert')
    parser.add_argument('--reduction', choices=['mean', 'min_group'], default='mean')
    parser.add_argument('--group-size', type=int, default=20)
    parser.add_argument('--max-gap', type=float, default=None, help='Only reselect when baseline top1 - top2 <= max_gap.')
    parser.add_argument('--drop-other-sids', action='store_true', help='Output only the selected sid per problem instead of full submission-safe score maps.')
    return parser.parse_args()


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


def metric_quality(arr: np.ndarray, metric: str, reduction: str, group_size: int) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('-inf')
    if reduction == 'mean':
        base = float(np.mean(arr))
    else:
        base = least_grouped_strict(arr, group_size)
    if metric == 'tok_conf':
        return -base
    return base


def get_metric_array(token_view, metric: str) -> np.ndarray:
    arr = getattr(token_view, metric)
    if arr is None:
        raise ValueError(f'metric {metric} not available in cache')
    return arr


def load_problem_run_ids(cache_root: str) -> Dict[str, set[str]]:
    meta = json.loads((Path(cache_root) / 'meta.json').read_text())
    grouped: Dict[str, set[str]] = defaultdict(set)
    for sample_idx, sample in enumerate(meta['samples']):
        grouped[str(sample['problem_id'])].add(str(sample_idx))
    return grouped


def choose_sid(
    sid_scores: Dict[str, float],
    reader: CacheReader,
    metric: str,
    reduction: str,
    group_size: int,
    topk: int,
    max_gap: float | None,
) -> Tuple[str, Dict[str, object]]:
    ranked = sorted(((str(sid), float(score)) for sid, score in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))
    original_sid, original_score = ranked[0]

    if len(ranked) == 1:
        return original_sid, {
            'original_sid': original_sid,
            'new_sid': original_sid,
            'reason': 'single_candidate',
        }

    gap = original_score - ranked[1][1]
    if max_gap is not None and gap > max_gap:
        return original_sid, {
            'original_sid': original_sid,
            'new_sid': original_sid,
            'reason': 'gap_too_large',
            'gap': gap,
        }

    candidates = ranked[:max(1, min(topk, len(ranked)))]
    scored_candidates = []
    for sid, baseline_score in candidates:
        token_view = reader.get_token_view(int(sid))
        metric_arr = get_metric_array(token_view, metric)
        quality = metric_quality(metric_arr, metric, reduction, group_size)
        scored_candidates.append((sid, baseline_score, quality))

    scored_candidates.sort(key=lambda row: (-row[2], -row[1], row[0]))
    new_sid = scored_candidates[0][0]
    return new_sid, {
        'original_sid': original_sid,
        'new_sid': new_sid,
        'reason': 'topk_reselect',
        'gap': gap,
        'candidates': [
            {
                'sid': sid,
                'baseline_score': baseline_score,
                'metric_quality': quality,
            }
            for sid, baseline_score, quality in scored_candidates
        ],
    }


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.input).read_text())
    target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]

    out_scores = {}
    notes = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'input': args.input,
        'output': args.output,
        'target_cache_keys': target_cache_keys,
        'metric': args.metric,
        'reduction': args.reduction,
        'group_size': args.group_size,
        'topk': args.topk,
        'max_gap': args.max_gap,
        'submission_safe_full_scores': not args.drop_other_sids,
        'cache_keys': {},
    }

    problem_run_ids_by_cache = {
        cache_key: load_problem_run_ids(DEFAULT_CACHE_MAP[cache_key])
        for cache_key in target_cache_keys
    }
    readers = {
        cache_key: CacheReader(DEFAULT_CACHE_MAP[cache_key])
        for cache_key in target_cache_keys
    }

    for cache_key, problem_map in base['scores'].items():
        cache_out = {}
        changed = 0
        skipped_missing = 0
        cache_notes = {
            'problem_count': len(problem_map),
            'changed_problem_ids': [],
            'unchanged_problem_ids': [],
            'details': {},
        }
        reader = readers.get(cache_key)
        valid_problem_runs = problem_run_ids_by_cache.get(cache_key)

        for problem_id, sid_scores in problem_map.items():
            sid_scores = {str(sid): float(score) for sid, score in sid_scores.items()}

            if cache_key not in target_cache_keys:
                cache_out[problem_id] = sid_scores
                continue

            valid_sids = valid_problem_runs.get(str(problem_id), set()) if valid_problem_runs else set()
            missing_sids = [sid for sid in sid_scores if sid not in valid_sids]
            if missing_sids:
                cache_out[problem_id] = sid_scores
                skipped_missing += 1
                cache_notes['details'][problem_id] = {
                    'reason': 'sid_mismatch_with_cache_meta',
                    'missing_sids_example': missing_sids[:5],
                }
                cache_notes['unchanged_problem_ids'].append(problem_id)
                continue

            new_sid, detail = choose_sid(
                sid_scores=sid_scores,
                reader=reader,
                metric=args.metric,
                reduction=args.reduction,
                group_size=args.group_size,
                topk=args.topk,
                max_gap=args.max_gap,
            )
            old_sid = detail['original_sid']
            if new_sid != old_sid:
                changed += 1
                cache_notes['changed_problem_ids'].append(problem_id)
            else:
                cache_notes['unchanged_problem_ids'].append(problem_id)
            cache_notes['details'][problem_id] = detail

            if args.drop_other_sids:
                cache_out[problem_id] = {new_sid: sid_scores[new_sid]}
            else:
                adjusted = dict(sid_scores)
                max_other = max(adjusted.values()) if adjusted else 0.0
                adjusted[new_sid] = max_other + 1e-9
                cache_out[problem_id] = adjusted

        out_scores[cache_key] = cache_out
        cache_notes['changed_count'] = changed
        cache_notes['skipped_missing_count'] = skipped_missing
        notes['cache_keys'][cache_key] = cache_notes
        print(f'finished {cache_key}: changed {changed}/{len(problem_map)}')

    out = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'scores': out_scores,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')


if __name__ == '__main__':
    main()
