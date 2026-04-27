#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader
from nad.core.selectors.base import SelectorContext
from plugins.medoid_tail_warning import MedoidTailWarningSelector
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json'
DEFAULT_OUTPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json'
DEFAULT_NOTES = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit_notes.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a submit-safe mixed_v3 by applying a tiny activation tail-warning veto on top of the strongest logprob mixed_v2 submission.'
    )
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--notes-output', default=DEFAULT_NOTES)
    parser.add_argument('--method-name', default='nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto')
    parser.add_argument('--target-cache-keys', default='DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25')
    parser.add_argument('--max-gap', type=float, default=0.001)
    parser.add_argument('--score-bump', type=float, default=1e-9)
    parser.add_argument('--tail-new-ratio-warn', type=float, default=0.015)
    parser.add_argument('--plateau-progress-warn', type=float, default=0.85)
    parser.add_argument('--min-tail-ratio-advantage', type=float, default=0.005)
    parser.add_argument('--min-plateau-progress-advantage', type=float, default=0.03)
    parser.add_argument('--tail-start', type=float, default=0.85)
    parser.add_argument('--plateau-fraction', type=float, default=0.98)
    return parser.parse_args()


def ranked_items(sid_scores: Dict[str, float]) -> List[tuple[str, float]]:
    return sorted(((str(sid), float(score)) for sid, score in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.input).read_text())
    target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]

    out_scores = copy.deepcopy(base['scores'])
    notes = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'input': args.input,
        'output': args.output,
        'target_cache_keys': target_cache_keys,
        'max_gap': args.max_gap,
        'score_bump': args.score_bump,
        'tail_warning_params': {
            'tail_new_ratio_warn': args.tail_new_ratio_warn,
            'plateau_progress_warn': args.plateau_progress_warn,
            'min_tail_ratio_advantage': args.min_tail_ratio_advantage,
            'min_plateau_progress_advantage': args.min_plateau_progress_advantage,
            'tail_start': args.tail_start,
            'plateau_fraction': args.plateau_fraction,
        },
        'cache_keys': {},
    }

    readers = {cache_key: CacheReader(DEFAULT_CACHE_MAP[cache_key]) for cache_key in target_cache_keys}

    for cache_key, problem_map in out_scores.items():
        if cache_key not in target_cache_keys:
            notes['cache_keys'][cache_key] = {
                'problem_count': len(problem_map),
                'changed_count': 0,
                'skipped_reason': 'not_target_cache_key',
            }
            continue

        reader = readers[cache_key]
        plugin = MedoidTailWarningSelector(
            gap_abs=0.01,
            tail_start=args.tail_start,
            plateau_fraction=args.plateau_fraction,
            tail_new_ratio_warn=args.tail_new_ratio_warn,
            plateau_progress_warn=args.plateau_progress_warn,
            min_tail_ratio_advantage=args.min_tail_ratio_advantage,
            min_plateau_progress_advantage=args.min_plateau_progress_advantage,
        )

        changed_problem_ids: List[str] = []
        unchanged_problem_ids: List[str] = []
        details: Dict[str, object] = {}

        for problem_id, sid_scores in problem_map.items():
            sid_scores = {str(sid): float(score) for sid, score in sid_scores.items()}
            ranked = ranked_items(sid_scores)
            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            gap = float(top1_score - top2_score)

            if gap > args.max_gap:
                unchanged_problem_ids.append(problem_id)
                details[problem_id] = {
                    'reason': 'gap_too_large',
                    'top1_sid': top1_sid,
                    'top2_sid': top2_sid,
                    'gap': gap,
                }
                continue

            ctx = SelectorContext(cache=reader, problem_id=str(problem_id), run_ids=[int(top1_sid), int(top2_sid)], views=[])
            plugin.bind(ctx)
            top1_metrics = plugin._tail_metrics(int(top1_sid))
            top2_metrics = plugin._tail_metrics(int(top2_sid))

            top1_warn = bool(top1_metrics and plugin._is_warned(top1_metrics))
            top2_warn = bool(top2_metrics and plugin._is_warned(top2_metrics))
            top2_healthier = bool(top1_metrics and top2_metrics and plugin._runner_is_healthier(top1_metrics, top2_metrics))

            if top1_warn and (not top2_warn) and top2_healthier:
                adjusted = dict(sid_scores)
                adjusted[top2_sid] = max(adjusted.values()) + float(args.score_bump)
                out_scores[cache_key][problem_id] = adjusted
                changed_problem_ids.append(problem_id)
                details[problem_id] = {
                    'reason': 'tail_warning_veto',
                    'old_sid': top1_sid,
                    'new_sid': top2_sid,
                    'gap': gap,
                    'top1_metrics': top1_metrics,
                    'top2_metrics': top2_metrics,
                }
            else:
                unchanged_problem_ids.append(problem_id)
                details[problem_id] = {
                    'reason': 'no_veto',
                    'top1_sid': top1_sid,
                    'top2_sid': top2_sid,
                    'gap': gap,
                    'top1_warn': top1_warn,
                    'top2_warn': top2_warn,
                    'top2_healthier': top2_healthier,
                    'top1_metrics': top1_metrics,
                    'top2_metrics': top2_metrics,
                }

        notes['cache_keys'][cache_key] = {
            'problem_count': len(problem_map),
            'changed_count': len(changed_problem_ids),
            'changed_problem_ids': changed_problem_ids,
            'unchanged_problem_ids': unchanged_problem_ids,
            'details': details,
        }
        print(f'finished {cache_key}: changed {len(changed_problem_ids)}/{len(problem_map)}')

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
