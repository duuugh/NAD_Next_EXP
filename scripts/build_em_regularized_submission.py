#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Apply E_M-regularized Best-of-N to an existing best_of_n submission score map.')
    parser.add_argument('--input', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/best_of_n_em_regularized_m4_seed42.json')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/best_of_n_em_regularized_m4_seed42_notes.json')
    parser.add_argument('--method-name', default='best_of_n_em_regularized_m4_seed42')
    parser.add_argument('--M', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep-all-sids', action='store_true', help='Keep all sid scores but force the selected sid to rank first.')
    return parser.parse_args()


def em_select_sid(score_map: Dict[str, float], M: float, rng: random.Random) -> Tuple[str, float, int]:
    if M < 1:
        raise ValueError('M must be >= 1')
    items = [(str(sid), float(score)) for sid, score in score_map.items()]
    N = len(items)
    if N == 0:
        raise ValueError('score_map must not be empty')
    k = int(math.ceil(N / float(M)))
    decorated = [(score, rng.random(), sid) for sid, score in items]
    decorated.sort(key=lambda x: (-x[0], x[1]))
    top_k = decorated[:k]
    chosen_score, _, chosen_sid = rng.choice(top_k)
    return chosen_sid, float(chosen_score), k


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.input).read_text())
    rng = random.Random(args.seed)

    out_scores = {}
    summary = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'input': args.input,
        'output': args.output,
        'M': args.M,
        'seed': args.seed,
        'keep_all_sids': args.keep_all_sids,
        'cache_keys': {},
    }

    for cache_key, problem_map in base['scores'].items():
        cache_out = {}
        k_values = []
        for problem_id, sid_scores in problem_map.items():
            chosen_sid, chosen_score, k = em_select_sid(sid_scores, args.M, rng)
            k_values.append(k)
            if args.keep_all_sids:
                # Preserve the candidate set, but make the selected answer strictly best.
                adjusted = {str(sid): float(score) for sid, score in sid_scores.items()}
                max_other = max(adjusted.values()) if adjusted else 0.0
                adjusted[chosen_sid] = max(max_other, chosen_score) + 1e-9
                cache_out[problem_id] = adjusted
            else:
                cache_out[problem_id] = {chosen_sid: chosen_score}
        out_scores[cache_key] = cache_out
        summary['cache_keys'][cache_key] = {
            'problem_count': len(cache_out),
            'k_min': min(k_values) if k_values else None,
            'k_max': max(k_values) if k_values else None,
            'k_example': k_values[0] if k_values else None,
        }

    out = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'scores': out_scores,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')


if __name__ == '__main__':
    main()
