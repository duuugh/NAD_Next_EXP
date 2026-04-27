#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict


DEFAULT_SELECTOR_NAME = 'file:./plugins/medoid_tail_warning.py:MedoidTailWarningSelector'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Wrap a single-cache Version A selector output into a full submit-safe best_of_n submission.'
    )
    parser.add_argument('--input', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--selection', default='/home/jovyan/work/NAD_Next/result/versionA_medoid_tail_warning_aime24_loose.json')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/best_of_n_versionA_loose_aime24_submit.json')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/best_of_n_versionA_loose_aime24_submit_notes.json')
    parser.add_argument('--method-name', default='best_of_n_versionA_loose_aime24')
    parser.add_argument('--target-cache-key', default='DS-R1/aime24')
    parser.add_argument('--selector-name', default=DEFAULT_SELECTOR_NAME)
    parser.add_argument('--score-bump', type=float, default=1e-6)
    return parser.parse_args()


def top_sid(sid_scores: Dict[str, float]) -> str:
    return max(((str(sid), float(score)) for sid, score in sid_scores.items()), key=lambda kv: (kv[1], kv[0]))[0]


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.input).read_text())
    selection = json.loads(Path(args.selection).read_text())

    if base.get('task') != 'best_of_n':
        raise ValueError(f"Expected best_of_n base submission, got task={base.get('task')}")
    if 'scores' not in base or args.target_cache_key not in base['scores']:
        raise KeyError(f"Missing target cache key {args.target_cache_key} in base submission")
    if 'problems' not in selection:
        raise ValueError('Selection JSON must contain problems')

    out_scores = copy.deepcopy(base['scores'])
    changed = []
    unchanged = []

    for problem_id, pdata in selection['problems'].items():
        selectors = pdata.get('selectors', {})
        if args.selector_name not in selectors:
            raise KeyError(f"Selector {args.selector_name} missing from selection for problem {problem_id}")
        chosen_sid = str(selectors[args.selector_name])
        sid_scores = {str(sid): float(score) for sid, score in out_scores[args.target_cache_key][problem_id].items()}
        if chosen_sid not in sid_scores:
            raise KeyError(f"Chosen sid {chosen_sid} missing from base score map for {args.target_cache_key}/{problem_id}")

        old_sid = top_sid(sid_scores)
        adjusted = dict(sid_scores)
        adjusted[chosen_sid] = max(adjusted.values()) + float(args.score_bump)
        out_scores[args.target_cache_key][problem_id] = adjusted

        row = {
            'problem_id': str(problem_id),
            'old_sid': old_sid,
            'new_sid': chosen_sid,
            'changed': chosen_sid != old_sid,
            'old_score': float(sid_scores[old_sid]),
            'new_score_before_bump': float(sid_scores[chosen_sid]),
            'new_score_after_bump': float(adjusted[chosen_sid]),
        }
        if chosen_sid != old_sid:
            changed.append(row)
        else:
            unchanged.append(row)

    out = {
        'task': 'best_of_n',
        'method_name': args.method_name,
        'scores': out_scores,
    }
    notes = {
        'task': 'best_of_n',
        'method_name': args.method_name,
        'input': args.input,
        'selection': args.selection,
        'target_cache_key': args.target_cache_key,
        'selector_name': args.selector_name,
        'score_bump': args.score_bump,
        'changed_count': len(changed),
        'unchanged_count': len(unchanged),
        'changed_problems': changed,
    }

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')
    print(f'changed {len(changed)}/{len(selection["problems"])} problems in {args.target_cache_key}')


if __name__ == '__main__':
    main()
