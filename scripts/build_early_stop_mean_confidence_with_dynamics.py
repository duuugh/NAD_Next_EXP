#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from scripts.build_early_stop_dynamics_v2_local import (
    build_problem_dynamics_signals,
    clip01,
    dyn_correction,
    load_dynamics_map,
)


BUDGETS = [i / 10.0 for i in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fuse mean-confidence backbone with conservative dynamics plugin.')
    parser.add_argument('--backbone', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence.json')
    parser.add_argument('--reference', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--policy', default='/home/jovyan/work/NAD_Next/result/dynamics_policy_conservative.json')
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_plus_dyn_conservative.json')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_plus_dyn_conservative_notes.json')
    parser.add_argument('--report-output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_plus_dyn_conservative_report.json')
    parser.add_argument('--method-name', default='early_stop_mean_confidence_plus_dyn_conservative')
    parser.add_argument('--alpha', type=float, default=0.18, help='Small additive plugin weight.')
    parser.add_argument('--dyn-strength', type=float, default=1.0, help='Extra multiplier after plugin normalization.')
    parser.add_argument('--round-digits', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--dynamics-stats',
        action='append',
        default=[
            'DS-R1/aime24=/home/jovyan/work/NAD_Next/result/dynamics_full/dynamics_statistics.csv',
            'DS-R1/aime25=/home/jovyan/work/NAD_Next/result/dynamics_full/dynamics_statistics_DS-R1_aime25.csv',
            'DS-R1/hmmt25=/home/jovyan/work/NAD_Next/result/dynamics_full/dynamics_statistics_DS-R1_hmmt25.csv',
        ],
        help='Repeatable: cache_key=/path/to/dynamics_statistics.csv',
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or 'scores' not in obj or not isinstance(obj['scores'], dict):
        raise ValueError(f'Invalid submission schema in {path}')
    return obj


def sample_sort_key(sample_id: str) -> Tuple[int, str]:
    text = str(sample_id)
    if text.isdigit():
        return (0, f'{int(text):012d}')
    return (1, text)


def rank_transform(values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(((str(k), float(v)) for k, v in values.items()), key=lambda x: (x[1], x[0]))
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}
    denom = float(n - 1)
    return {sid: idx / denom for idx, (sid, _) in enumerate(items)}


def ensure_same_structure(reference_scores: Mapping[str, Any], scores: Mapping[str, Any]) -> None:
    if set(reference_scores.keys()) != set(scores.keys()):
        raise ValueError('字段格式不匹配: backbone cache_key 覆盖与 reference 不一致')
    for cache_key, ref_problem_map in reference_scores.items():
        cur_problem_map = scores.get(cache_key)
        if set(ref_problem_map.keys()) != set(cur_problem_map.keys()):
            raise ValueError(f'字段格式不匹配: {cache_key} problem_id 覆盖不一致')
        for problem_id, ref_sample_map in ref_problem_map.items():
            cur_sample_map = cur_problem_map.get(problem_id)
            if set(ref_sample_map.keys()) != set(cur_sample_map.keys()):
                raise ValueError(f'字段格式不匹配: {cache_key}/{problem_id} sample_id 覆盖不一致')
            for sample_id, curve in cur_sample_map.items():
                if not isinstance(curve, list) or len(curve) != 10:
                    raise ValueError(f'字段格式不匹配: {cache_key}/{problem_id}/{sample_id} 不是 list[10]')


def main() -> None:
    args = parse_args()
    for path in [Path(args.output), Path(args.notes_output), Path(args.report_output)]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists, set --overwrite to replace: {path}')

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError('--alpha must be in [0,1]')

    backbone = load_json(Path(args.backbone))
    reference = load_json(Path(args.reference))
    ensure_same_structure(reference['scores'], backbone['scores'])

    policy_obj = json.loads(Path(args.policy).read_text(encoding='utf-8'))
    if not isinstance(policy_obj, dict):
        raise ValueError('policy must be a JSON object cache_key -> mode/disable')
    policy: Dict[str, str] = {str(k): str(v) for k, v in policy_obj.items()}

    dynamics_map = load_dynamics_map(args.dynamics_stats)

    out_scores = json.loads(json.dumps(backbone['scores']))

    enabled_caches: List[str] = []
    disabled_caches: List[str] = []
    cache_deltas: Dict[str, Dict[str, float]] = {}

    for cache_key, problem_map in out_scores.items():
        route = policy.get(cache_key, 'disable')
        if route == 'disable':
            disabled_caches.append(cache_key)
            cache_deltas[cache_key] = {
                'mean_abs_delta': 0.0,
                'max_abs_delta': 0.0,
                'changed_samples': 0,
                'total_samples': int(sum(len(smap) for smap in problem_map.values())),
            }
            continue

        if route not in {'rho_tail_only', 'neg_A_accel_only', 'rho_tail_plus_neg_A_accel'}:
            raise ValueError(f'Unsupported route in policy: {cache_key} -> {route}')

        if cache_key not in dynamics_map:
            raise ValueError(f'字段格式不匹配: conservative 启用了 {cache_key}，但缺 dynamics statistics')

        enabled_caches.append(cache_key)
        df_cache = dynamics_map[cache_key]

        abs_deltas: List[float] = []
        changed_samples = 0
        total_samples = 0

        for problem_id, sample_map in problem_map.items():
            sample_ids = sorted([str(sid) for sid in sample_map.keys()], key=sample_sort_key)
            dyn_signals = build_problem_dynamics_signals(df_cache, str(problem_id), sample_ids)
            if dyn_signals is None:
                raise ValueError(
                    f'字段格式不匹配: conservative 启用缓存 {cache_key} 在 problem_id={problem_id} 缺 dynamics 对齐数据'
                )

            for bidx, budget in enumerate(BUDGETS):
                plugin_raw = {
                    sid: dyn_correction(
                        budget=budget,
                        mode=route,
                        stop_boost_signal=float(dyn_signals['stop_boost_signal'][sid]),
                        rank_guard_signal=float(dyn_signals['rank_guard_signal'][sid]),
                    )
                    for sid in sample_ids
                }
                plugin_rank = rank_transform(plugin_raw)
                centered = {sid: float(plugin_rank[sid]) - 0.5 for sid in sample_ids}

                for sid in sample_ids:
                    base_score = float(sample_map[sid][bidx])
                    delta = float(args.alpha) * float(args.dyn_strength) * float(centered[sid])
                    new_score = clip01(base_score + delta)
                    sample_map[sid][bidx] = new_score
                    abs_deltas.append(abs(delta))

            for sid in sample_ids:
                total_samples += 1
                if any(abs(float(problem_map[problem_id][sid][i]) - float(backbone['scores'][cache_key][problem_id][sid][i])) > 1e-12 for i in range(10)):
                    changed_samples += 1

        if args.round_digits >= 0:
            for problem_id, sample_map in problem_map.items():
                for sid, curve in sample_map.items():
                    sample_map[sid] = [round(float(v), int(args.round_digits)) for v in curve]

        cache_deltas[cache_key] = {
            'mean_abs_delta': float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            'max_abs_delta': float(np.max(abs_deltas)) if abs_deltas else 0.0,
            'changed_samples': int(changed_samples),
            'total_samples': int(total_samples),
        }

    # strict: policy-disabled caches must remain unchanged
    unchanged_disabled_ok = True
    unchanged_disabled_details: Dict[str, bool] = {}
    for cache_key in disabled_caches:
        same = out_scores[cache_key] == backbone['scores'][cache_key]
        unchanged_disabled_details[cache_key] = bool(same)
        if not same:
            unchanged_disabled_ok = False

    out = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'scores': out_scores,
    }

    notes = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'settings': {
            'backbone': args.backbone,
            'reference': args.reference,
            'policy': args.policy,
            'dynamics_stats': args.dynamics_stats,
            'alpha': args.alpha,
            'dyn_strength': args.dyn_strength,
            'plugin_normalization': 'per-problem per-budget rank normalize to [0,1], center to [-0.5,0.5]',
            'plugin_formula': 'final = clip01(backbone + alpha * dyn_strength * centered_plugin)',
            'round_digits': args.round_digits,
        },
        'stats': {
            'enabled_caches': enabled_caches,
            'disabled_caches': disabled_caches,
            'enabled_cache_count': len(enabled_caches),
            'disabled_cache_count': len(disabled_caches),
            'unchanged_disabled_ok': unchanged_disabled_ok,
            'unchanged_disabled_details': unchanged_disabled_details,
            'cache_deltas': cache_deltas,
        },
    }

    report = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'summary': notes['stats'],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.notes_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.notes_output).write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.report_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')
    print(f'wrote {args.report_output}')
    print(f'enabled_cache_count={len(enabled_caches)} disabled_cache_count={len(disabled_caches)}')


if __name__ == '__main__':
    main()
