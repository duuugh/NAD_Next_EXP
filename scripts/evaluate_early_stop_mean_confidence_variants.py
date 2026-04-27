#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.ops.accuracy import load_correctness_map
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


BUDGETS = [i / 10.0 for i in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate confidence-only vs confidence+dynamics variants.')
    parser.add_argument('--confidence-only', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence.json')
    parser.add_argument('--confidence-plus-dyn', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_plus_dyn_conservative.json')
    parser.add_argument('--dynamics-only', default='/home/jovyan/work/NAD_Next/result/early_stop_dynamics_router_conservative_submit.json')
    parser.add_argument('--policy', default='/home/jovyan/work/NAD_Next/result/dynamics_policy_conservative.json')
    parser.add_argument('--output-json', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_variants_eval.json')
    parser.add_argument('--output-markdown', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_variants_eval.md')
    parser.add_argument('--overwrite', action='store_true')
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


def pick_top1(sample_map: Mapping[str, Any], budget_idx: int) -> Tuple[str, float]:
    best_sid: Optional[str] = None
    best_score: Optional[float] = None
    for sid, value in sample_map.items():
        if isinstance(value, list):
            score = float(value[budget_idx])
        else:
            score = float(value)
        sid_s = str(sid)
        if best_sid is None:
            best_sid, best_score = sid_s, score
            continue
        if score > float(best_score):
            best_sid, best_score = sid_s, score
        elif abs(score - float(best_score)) <= 1e-12 and sample_sort_key(sid_s) < sample_sort_key(best_sid):
            best_sid, best_score = sid_s, score
    if best_sid is None or best_score is None:
        raise ValueError('empty sample map')
    return best_sid, float(best_score)


def auc_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def evaluate_cache_curve(scores_cache: Mapping[str, Any], correctness_map: Optional[Mapping[int, bool]]) -> Dict[str, Any]:
    by_budget: List[Dict[str, Any]] = []

    if correctness_map is None:
        for budget in BUDGETS:
            by_budget.append({'budget': budget, 'SelAcc': None, 'AUROC': None, 'selected_count': 0})
        return {
            'labeled': False,
            'by_budget': by_budget,
            'AUC-AUROC': None,
            'AUC-SelAcc': None,
            'AUROC@10%': None,
            'AUROC@50%': None,
            'AUROC@100%': None,
            'Stop@100%': None,
            'Earliest>0.6': None,
        }

    for bidx, budget in enumerate(BUDGETS):
        ys: List[int] = []
        ss: List[float] = []

        for problem_id, sample_map in scores_cache.items():
            sid, score = pick_top1(sample_map, bidx)
            corr = bool(correctness_map.get(int(sid), False))
            ys.append(1 if corr else 0)
            ss.append(float(score))

        selacc = float(np.mean(ys)) if ys else None
        auroc = None
        if ys and len(set(ys)) >= 2:
            auroc = float(roc_auc_score(np.array(ys, dtype=np.int64), np.array(ss, dtype=np.float64)))

        by_budget.append({'budget': budget, 'SelAcc': selacc, 'AUROC': auroc, 'selected_count': len(ys)})

    auc_auroc = auc_mean([x['AUROC'] for x in by_budget])
    auc_selacc = auc_mean([x['SelAcc'] for x in by_budget])

    earliest = None
    for x in by_budget:
        if x['SelAcc'] is not None and float(x['SelAcc']) > 0.6:
            earliest = float(x['budget'])
            break

    def at_budget(b: float, key: str) -> Optional[float]:
        idx = int(round(b * 10)) - 1
        if idx < 0 or idx >= len(by_budget):
            return None
        return by_budget[idx][key]

    return {
        'labeled': True,
        'by_budget': by_budget,
        'AUC-AUROC': auc_auroc,
        'AUC-SelAcc': auc_selacc,
        'AUROC@10%': at_budget(0.1, 'AUROC'),
        'AUROC@50%': at_budget(0.5, 'AUROC'),
        'AUROC@100%': at_budget(1.0, 'AUROC'),
        'Stop@100%': at_budget(1.0, 'SelAcc'),
        'Earliest>0.6': earliest,
    }


def evaluate_submission(sub_obj: Mapping[str, Any], correctness_by_cache: Mapping[str, Optional[Mapping[int, bool]]]) -> Dict[str, Any]:
    per_cache: Dict[str, Any] = {}
    for cache_key, cache_scores in sub_obj['scores'].items():
        per_cache[cache_key] = evaluate_cache_curve(cache_scores, correctness_by_cache.get(cache_key))

    labeled_caches = [k for k, v in per_cache.items() if v['labeled']]
    overall = {
        'labeled_cache_count': len(labeled_caches),
        'cache_count': len(per_cache),
        'AUC-AUROC': auc_mean([per_cache[k]['AUC-AUROC'] for k in labeled_caches]),
        'AUC-SelAcc': auc_mean([per_cache[k]['AUC-SelAcc'] for k in labeled_caches]),
        'AUROC@100%': auc_mean([per_cache[k]['AUROC@100%'] for k in labeled_caches]),
        'Stop@100%': auc_mean([per_cache[k]['Stop@100%'] for k in labeled_caches]),
        'Earliest>0.6': auc_mean([per_cache[k]['Earliest>0.6'] for k in labeled_caches]),
    }

    return {
        'overall': overall,
        'per_cache': per_cache,
    }


def safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def compute_delta(eval_a: Mapping[str, Any], eval_b: Mapping[str, Any]) -> Dict[str, Any]:
    delta_overall = {
        k: safe_delta(eval_a['overall'].get(k), eval_b['overall'].get(k))
        for k in ['AUC-AUROC', 'AUC-SelAcc', 'AUROC@100%', 'Stop@100%', 'Earliest>0.6']
    }

    per_cache: Dict[str, Any] = {}
    for cache_key, m_a in eval_a['per_cache'].items():
        m_b = eval_b['per_cache'][cache_key]
        per_cache[cache_key] = {
            'AUC-AUROC': safe_delta(m_a.get('AUC-AUROC'), m_b.get('AUC-AUROC')),
            'AUC-SelAcc': safe_delta(m_a.get('AUC-SelAcc'), m_b.get('AUC-SelAcc')),
            'AUROC@10%': safe_delta(m_a.get('AUROC@10%'), m_b.get('AUROC@10%')),
            'AUROC@50%': safe_delta(m_a.get('AUROC@50%'), m_b.get('AUROC@50%')),
            'AUROC@100%': safe_delta(m_a.get('AUROC@100%'), m_b.get('AUROC@100%')),
            'Stop@100%': safe_delta(m_a.get('Stop@100%'), m_b.get('Stop@100%')),
        }
    return {'overall': delta_overall, 'per_cache': per_cache}


def plugin_effect_checks(
    conf_only_obj: Mapping[str, Any],
    conf_dyn_obj: Mapping[str, Any],
    policy: Mapping[str, str],
    delta_per_cache: Mapping[str, Any],
) -> Dict[str, Any]:
    enabled = sorted([k for k, v in policy.items() if v != 'disable'])
    disabled = sorted([k for k, v in policy.items() if v == 'disable'])

    changed_caches: List[str] = []
    unchanged_caches: List[str] = []
    for cache_key in conf_only_obj['scores'].keys():
        if conf_only_obj['scores'][cache_key] == conf_dyn_obj['scores'][cache_key]:
            unchanged_caches.append(cache_key)
        else:
            changed_caches.append(cache_key)

    disabled_unchanged = {k: (k in unchanged_caches) for k in disabled}

    enabled_gain = {}
    for k in enabled:
        delta = delta_per_cache.get(k, {})
        enabled_gain[k] = {
            'delta_AUC-AUROC': delta.get('AUC-AUROC'),
            'delta_AUC-SelAcc': delta.get('AUC-SelAcc'),
        }

    return {
        'policy_enabled_caches': enabled,
        'policy_disabled_caches': disabled,
        'changed_caches': sorted(changed_caches),
        'unchanged_caches': sorted(unchanged_caches),
        'disabled_unchanged': disabled_unchanged,
        'enabled_gain': enabled_gain,
        'only_expected_changed': sorted(changed_caches) == sorted(enabled),
        'expected_enabled_set': enabled,
    }


def fmt(v: Optional[float]) -> str:
    if v is None:
        return 'N/A'
    return f'{float(v):+.6f}'


def build_markdown(report: Mapping[str, Any]) -> str:
    eval_conf = report['evaluations']['confidence_only']
    eval_dyn = report['evaluations']['confidence_plus_dynamics_conservative']
    delta = report['delta']['confidence_plus_minus_confidence_only']
    checks = report['plugin_checks']

    dyn_only_eval = report['evaluations'].get('dynamics_only_conservative')

    lines: List[str] = []
    lines.append('# Mean-Confidence Early-Stop Variants Evaluation')
    lines.append('')
    lines.append('## Overall Metrics')
    lines.append('')
    lines.append('| Variant | AUC-AUROC | AUC-SelAcc | Earliest>0.6 | AUROC@100% | Stop@100% |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name, ev in [
        ('confidence_only', eval_conf),
        ('confidence_plus_dynamics_conservative', eval_dyn),
    ]:
        ov = ev['overall']
        lines.append(
            f"| {name} | {fmt(ov['AUC-AUROC'])} | {fmt(ov['AUC-SelAcc'])} | {fmt(ov['Earliest>0.6'])} | {fmt(ov['AUROC@100%'])} | {fmt(ov['Stop@100%'])} |"
        )

    if dyn_only_eval is not None:
        ov = dyn_only_eval['overall']
        lines.append(
            f"| dynamics_only_conservative | {fmt(ov['AUC-AUROC'])} | {fmt(ov['AUC-SelAcc'])} | {fmt(ov['Earliest>0.6'])} | {fmt(ov['AUROC@100%'])} | {fmt(ov['Stop@100%'])} |"
        )

    lines.append('')
    lines.append('## Per-cache Breakdown')
    lines.append('')
    lines.append('| Cache | conf AUC-AUROC | conf AUC-SelAcc | conf AUROC@10 | conf AUROC@50 | conf AUROC@100 | conf Stop@100 | plus AUC-AUROC | plus AUC-SelAcc | plus AUROC@10 | plus AUROC@50 | plus AUROC@100 | plus Stop@100 | delta AUC-AUROC | delta AUC-SelAcc | delta AUROC@100 | delta Stop@100 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for cache_key in sorted(eval_conf['per_cache'].keys()):
        c = eval_conf['per_cache'][cache_key]
        p = eval_dyn['per_cache'][cache_key]
        d = delta['per_cache'][cache_key]
        lines.append(
            f"| {cache_key} | {fmt(c['AUC-AUROC'])} | {fmt(c['AUC-SelAcc'])} | {fmt(c['AUROC@10%'])} | {fmt(c['AUROC@50%'])} | {fmt(c['AUROC@100%'])} | {fmt(c['Stop@100%'])} | {fmt(p['AUC-AUROC'])} | {fmt(p['AUC-SelAcc'])} | {fmt(p['AUROC@10%'])} | {fmt(p['AUROC@50%'])} | {fmt(p['AUROC@100%'])} | {fmt(p['Stop@100%'])} | {fmt(d['AUC-AUROC'])} | {fmt(d['AUC-SelAcc'])} | {fmt(d['AUROC@100%'])} | {fmt(d['Stop@100%'])} |"
        )

    lines.append('')
    lines.append('## Plugin Checks')
    lines.append('')
    lines.append(f"- Conservative enabled caches: {', '.join(checks['policy_enabled_caches'])}")
    lines.append(f"- Changed caches (score content actually changed): {', '.join(checks['changed_caches'])}")
    lines.append(f"- Disabled caches unchanged: {all(checks['disabled_unchanged'].values())}")
    lines.append(f"- Only expected caches changed: {checks['only_expected_changed']}")

    lines.append('')
    lines.append('## Required Conclusions')
    lines.append('')
    lines.append(
        f"- mean-confidence backbone 是否明显超过 conservative dynamics-only: "
        f"AUC-AUROC delta={fmt(None if dyn_only_eval is None else eval_conf['overall']['AUC-AUROC'] - dyn_only_eval['overall']['AUC-AUROC'])}, "
        f"AUC-SelAcc delta={fmt(None if dyn_only_eval is None else eval_conf['overall']['AUC-SelAcc'] - dyn_only_eval['overall']['AUC-SelAcc'])}."
    )
    lines.append(
        f"- confidence_plus_dynamics_conservative 是否比 confidence_only 更强: "
        f"AUC-AUROC delta={fmt(delta['overall']['AUC-AUROC'])}, "
        f"AUC-SelAcc delta={fmt(delta['overall']['AUC-SelAcc'])}, "
        f"Stop@100 delta={fmt(delta['overall']['Stop@100%'])}."
    )
    lines.append(
        f"- dynamics plugin 是否只在少数 DS-R1 上局部增益: only_expected_changed={checks['only_expected_changed']} "
        f"(expected: DS-R1/aime24, DS-R1/aime25, DS-R1/hmmt25)."
    )

    recommend = 'confidence_only'
    if (delta['overall']['AUC-AUROC'] is not None and delta['overall']['AUC-SelAcc'] is not None and
        delta['overall']['AUC-AUROC'] > 0 and delta['overall']['AUC-SelAcc'] >= 0):
        recommend = 'confidence_plus_dynamics_conservative'
    lines.append(f'- 下一次 leaderboard 主提交建议: {recommend}.')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()

    for path in [Path(args.output_json), Path(args.output_markdown)]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists, set --overwrite to replace: {path}')

    conf_only = load_json(Path(args.confidence_only))
    conf_dyn = load_json(Path(args.confidence_plus_dyn))

    dyn_only = None
    if Path(args.dynamics_only).exists():
        dyn_only = load_json(Path(args.dynamics_only))

    cache_keys = sorted(conf_only['scores'].keys())
    if set(cache_keys) != set(conf_dyn['scores'].keys()):
        raise ValueError('confidence_only 与 confidence_plus_dynamics_conservative cache 覆盖不一致')

    correctness_by_cache: Dict[str, Optional[Mapping[int, bool]]] = {}
    for cache_key in cache_keys:
        cache_root = DEFAULT_CACHE_MAP.get(cache_key)
        if cache_root is None:
            raise KeyError(f'DEFAULT_CACHE_MAP 缺 cache_key={cache_key}')
        try:
            correctness_by_cache[cache_key] = load_correctness_map(cache_root)
        except Exception:
            correctness_by_cache[cache_key] = None

    eval_conf = evaluate_submission(conf_only, correctness_by_cache)
    eval_dyn = evaluate_submission(conf_dyn, correctness_by_cache)
    delta_conf = compute_delta(eval_dyn, eval_conf)

    evaluations: Dict[str, Any] = {
        'confidence_only': eval_conf,
        'confidence_plus_dynamics_conservative': eval_dyn,
    }

    if dyn_only is not None:
        evaluations['dynamics_only_conservative'] = evaluate_submission(dyn_only, correctness_by_cache)

    policy_obj = json.loads(Path(args.policy).read_text(encoding='utf-8'))
    policy = {str(k): str(v) for k, v in policy_obj.items()}
    checks = plugin_effect_checks(conf_only, conf_dyn, policy, delta_conf['per_cache'])

    report = {
        'inputs': {
            'confidence_only': args.confidence_only,
            'confidence_plus_dynamics_conservative': args.confidence_plus_dyn,
            'dynamics_only': args.dynamics_only if dyn_only is not None else None,
            'policy': args.policy,
        },
        'evaluations': evaluations,
        'delta': {
            'confidence_plus_minus_confidence_only': delta_conf,
        },
        'plugin_checks': checks,
    }

    md = build_markdown(report)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.output_markdown).write_text(md, encoding='utf-8')

    print(f'wrote {args.output_json}')
    print(f'wrote {args.output_markdown}')


if __name__ == '__main__':
    main()
