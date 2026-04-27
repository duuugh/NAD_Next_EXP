#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, '/home/jovyan/work/NAD_Next')
from plugins.em_regularized_bon_selector import select_from_answer_score_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Select one answer using E_M-regularized Best-of-N from a list of (answer, score).'
    )
    parser.add_argument('input_json', help='Path to JSON list: [[answer, score], ...] or [{"answer": ..., "score": ...}, ...]')
    parser.add_argument('--M', type=float, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output-json', default=None, help='Optional output file to save the selected answer')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text())
    rng = random.Random(args.seed) if args.seed is not None else None
    selected = select_from_answer_score_list(data, M=args.M, rng=rng)

    payload = {
        'method': 'E_M-regularized Best-of-N',
        'M': args.M,
        'seed': args.seed,
        'selected_answer': selected,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
