#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP
from nad.ops.accuracy import load_correctness_map


MODES = [
    'rho_tail_only',
    'neg_A_accel_only',
    'rho_tail_plus_neg_A_accel',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate dynamics_v2_local on labeled/unlabeled slices.')
    parser.add_argument('--builder-script', default='/home/jovyan/work/NAD_Next/scripts/build_early_stop_dynamics_v2_local.py')
    parser.add_argument('--reference', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--baseline', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--dynamics-stats', action='append', default=['DS-R1/aime24=result/dynamics_full/dynamics_statistics.csv'])

    parser.add_argument('--value-mode', default='rank_problem', choices=['rank_problem', 'minmax_problem', 'raw'])
    parser.add_argument('--raw-min', type=float, default=0.0)
    parser.add_argument('--raw-max', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.6)
    parser.add_argument('--dyn-strength', type=float, default=0.60)

    parser.add_argument('--output-main', default='/home/jovyan/work/NAD_Next/result/early_stop_dynamics_v2_local.json')
    parser.add_argument('--output-notes', default='/home/jovyan/work/NAD_Next/result/early_stop_dynamics_v2_local_notes.json')
    parser.add_argument('--output-report', default='/home/jovyan/work/NAD_Next/result/early_stop_dynamics_v2_local_report.json')
    parser.add_argument('--method-name', default='early_stop_dynamics_v2_local')
    return parser.parse_args()


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Not found: {p}')
    return json.loads(p.read_text())


def parse_kv(text: str, sep: str = '=') -> Tuple[str, str]:
    if sep not in text:
        raise ValueError(f'Expected key{sep}value: {text}')
    key, value = text.split(sep, 1)
    return key.strip(), value.strip()


def sid_tie_key(sid: str) -> Tuple[int, str]:
    t = str(sid)
    if t.isdigit():
        return (0, f'{int(t):012d}')
    return (1, t)


def score_at_budget(value: Any, budget_index: int) -> float:
    if isinstance(value, list):
        if len(value) != 10:
            raise ValueError(f'Expected length=10 score list, got {len(value)}')
        return float(value[budget_index])
    return float(value)


def pick_top1(problem_scores: Mapping[str, Any], budget_index: int) -> Tuple[str, float]:
    best_sid: Optional[str] = None
    best_score: Optional[float] = None
    for sid, value in problem_scores.items():
        score = score_at_budget(value, budget_index)
        sid_str = str(sid)
        if best_sid is None:
            best_sid = sid_str
            best_score = score
            continue
        if score > float(best_score):
            best_sid = sid_str
            best_score = score
        elif abs(score - float(best_score)) <= 1e-12 and sid_tie_key(sid_str) < sid_tie_key(str(best_sid)):
            best_sid = sid_str
            best_score = score
    if best_sid is None or best_score is None:
        raise ValueError('Empty problem score map')
    return best_sid, float(best_score)


def compute_correctness_maps() -> Tuple[Dict[str, Dict[int, bool]], List[str]]:
    labeled: Dict[str, Dict[int, bool]] = {}
    unlabeled: List[str] = []
    for cache_key, cache_root in DEFAULT_CACHE_MAP.items():
        try:
            labeled[cache_key] = load_correctness_map(cache_root)
        except Exception:
            unlabeled.append(cache_key)
    return labeled, unlabeled


def metric_from_selected(selected_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    y = np.array([1 if row['is_correct'] else 0 for row in selected_rows], dtype=np.int64)
    s = np.array([float(row['score']) for row in selected_rows], dtype=np.float64)
    selacc = float(np.mean(y)) if y.size > 0 else float('nan')
    auroc = float('nan')
    if y.size > 0 and np.unique(y).size >= 2:
        auroc = float(roc_auc_score(y, s))
    return {'SelAcc': selacc, 'AUROC': auroc}


def aggregate_budget_metrics(curve: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        'AUC-SelAcc': float(np.nanmean([x['SelAcc'] for x in curve])),
        'AUC-AUROC': float(np.nanmean([x['AUROC'] for x in curve])),
        'Stop@100': float(curve[-1]['SelAcc']),
    }


def build_submission_for_mode(args: argparse.Namespace, mode: str, out_json: Path, out_notes: Path) -> None:
    cmd = [
        sys.executable,
        str(args.builder_script),
        '--reference', str(args.reference),
        '--final-scores', str(args.reference),
        '--output', str(out_json),
        '--notes-output', str(out_notes),
        '--method-name', (args.method_name if mode == 'rho_tail_plus_neg_A_accel' else f'{args.method_name}_{mode}'),
        '--value-mode', str(args.value_mode),
        '--raw-min', str(args.raw_min),
        '--raw-max', str(args.raw_max),
        '--gamma', str(args.gamma),
        '--dyn-strength', str(args.dyn_strength),
        '--mode', str(mode),
    ]
    for item in args.dynamics_stats:
        cmd.extend(['--dynamics-stats', str(item)])
    subprocess.run(cmd, check=True)


def evaluate_submission(
    submission: Dict[str, Any],
    baseline: Dict[str, Any],
    correctness_map_by_cache: Dict[str, Dict[int, bool]],
) -> Dict[str, Any]:
    budgets = [i / 10.0 for i in range(1, 11)]
    sub_scores = submission['scores']
    base_scores = baseline['scores']

    labeled_curve: List[Dict[str, float]] = []
    baseline_labeled_curve: List[Dict[str, float]] = []

    per_mode_selected: Dict[int, List[Dict[str, Any]]] = {}
    per_baseline_selected: Dict[int, List[Dict[str, Any]]] = {}

    for bidx, budget in enumerate(budgets):
        selected_rows: List[Dict[str, Any]] = []
        baseline_rows: List[Dict[str, Any]] = []

        for cache_key, problem_map in sub_scores.items():
            if cache_key not in correctness_map_by_cache:
                continue
            corr_map = correctness_map_by_cache[cache_key]
            for problem_id, sample_map in problem_map.items():
                sid, score = pick_top1(sample_map, bidx)
                selected_rows.append({
                    'cache_key': cache_key,
                    'problem_id': str(problem_id),
                    'sid': sid,
                    'score': score,
                    'is_correct': bool(corr_map.get(int(sid), False)),
                })

                b_sid, b_score = pick_top1(base_scores[cache_key][str(problem_id)], bidx)
                baseline_rows.append({
                    'cache_key': cache_key,
                    'problem_id': str(problem_id),
                    'sid': b_sid,
                    'score': b_score,
                    'is_correct': bool(corr_map.get(int(b_sid), False)),
                })

        met = metric_from_selected(selected_rows)
        b_met = metric_from_selected(baseline_rows)
        labeled_curve.append({'budget': budget, **met})
        baseline_labeled_curve.append({'budget': budget, **b_met})
        per_mode_selected[bidx] = selected_rows
        per_baseline_selected[bidx] = baseline_rows

    agg = aggregate_budget_metrics(labeled_curve)
    agg_base = aggregate_budget_metrics(baseline_labeled_curve)

    delta = {
        'delta AUC-AUROC': float(agg['AUC-AUROC'] - agg_base['AUC-AUROC']),
        'delta AUC-SelAcc': float(agg['AUC-SelAcc'] - agg_base['AUC-SelAcc']),
        'delta Stop@100': float(agg['Stop@100'] - agg_base['Stop@100']),
    }

    return {
        'labeled_metrics': agg,
        'labeled_budget_curve': labeled_curve,
        'baseline_labeled_metrics': agg_base,
        'baseline_labeled_budget_curve': baseline_labeled_curve,
        'delta_vs_baseline': delta,
        'selected_by_budget': per_mode_selected,
        'baseline_selected_by_budget': per_baseline_selected,
    }


def evaluate_unlabeled_change_rates(
    submission: Dict[str, Any],
    baseline: Dict[str, Any],
    unlabeled_cache_keys: List[str],
) -> Dict[str, Any]:
    budgets = [i / 10.0 for i in range(1, 11)]
    sub_scores = submission['scores']
    base_scores = baseline['scores']

    by_budget: List[Dict[str, float]] = []
    for bidx, budget in enumerate(budgets):
        total = 0
        changed = 0
        for cache_key in unlabeled_cache_keys:
            if cache_key not in sub_scores or cache_key not in base_scores:
                continue
            for problem_id, sample_map in sub_scores[cache_key].items():
                sid, _ = pick_top1(sample_map, bidx)
                b_sid, _ = pick_top1(base_scores[cache_key][str(problem_id)], bidx)
                total += 1
                if str(sid) != str(b_sid):
                    changed += 1
        rate = float(changed / total) if total > 0 else 0.0
        by_budget.append({
            'budget': budget,
            'selection_change_rate': rate,
            'rank_change_rate': rate,
        })

    return {
        'mean selection change rate': float(np.mean([x['selection_change_rate'] for x in by_budget])) if by_budget else 0.0,
        'mean rank change rate': float(np.mean([x['rank_change_rate'] for x in by_budget])) if by_budget else 0.0,
        'by_budget': by_budget,
    }


def compute_coverage_audit(reference: Dict[str, Any], dynamics_stats: List[str]) -> Dict[str, Any]:
    score_map = reference['scores']
    total_problems = sum(len(problem_map) for problem_map in score_map.values())

    real_cov: Dict[str, List[str]] = {}
    local_exp_cov: Dict[str, List[str]] = {}

    dyn_problem_ids: Dict[str, set[str]] = {}
    for item in dynamics_stats:
        cache_key, csv_path = parse_kv(item)
        df = np.genfromtxt(str(Path(csv_path)), delimiter=',', names=True, dtype=None, encoding='utf-8')
        if isinstance(df, np.ndarray) and df.size > 0:
            if df.ndim == 0:
                pids = {str(df['problem_id'])}
            else:
                pids = {str(x) for x in df['problem_id']}
        else:
            pids = set()
        dyn_problem_ids[cache_key] = pids

    for cache_key, problem_map in score_map.items():
        covered = sorted([str(pid) for pid in problem_map.keys() if str(pid) in dyn_problem_ids.get(cache_key, set())], key=lambda x: int(x) if x.isdigit() else x)
        if covered:
            real_cov[cache_key] = covered

    real_problem_count = sum(len(v) for v in real_cov.values())
    local_exp_problem_count = sum(len(v) for v in local_exp_cov.values())
    total_covered = real_problem_count + local_exp_problem_count

    return {
        'real_coverage': {
            'cache_problem_ids': real_cov,
            'problem_count': real_problem_count,
        },
        'local_expansion_coverage': {
            'cache_problem_ids': local_exp_cov,
            'problem_count': local_exp_problem_count,
        },
        'total_coverage': {
            'covered_problem_count': total_covered,
            'total_problem_count': total_problems,
            'coverage_rate': float(total_covered / total_problems) if total_problems > 0 else 0.0,
        },
    }


def validate_submission(input_path: Path, reference_path: Path) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        sys.executable,
        '/home/jovyan/work/NAD_Next/scripts/validate_early_stop_submission.py',
        '--input', str(input_path),
        '--reference', str(reference_path),
        '--check-range',
        '--report-json', str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    obj = json.loads(tmp_path.read_text())
    tmp_path.unlink(missing_ok=True)
    return obj


def main() -> None:
    args = parse_args()

    reference = load_json(args.reference)
    baseline = load_json(args.baseline)

    correctness_map_by_cache, unlabeled_cache_keys = compute_correctness_maps()

    built_outputs: Dict[str, Dict[str, Path]] = {}
    with tempfile.TemporaryDirectory() as tdir:
        tmp_dir = Path(tdir)
        for mode in MODES:
            if mode == 'rho_tail_plus_neg_A_accel':
                out_json = Path(args.output_main)
                out_notes = Path(args.output_notes)
            else:
                out_json = tmp_dir / f'early_stop_dynamics_v2_local_{mode}.json'
                out_notes = tmp_dir / f'early_stop_dynamics_v2_local_{mode}_notes.json'

            build_submission_for_mode(args=args, mode=mode, out_json=out_json, out_notes=out_notes)
            built_outputs[mode] = {'json': out_json, 'notes': out_notes}

        results_by_mode: Dict[str, Any] = {}
        for mode in MODES:
            sub_obj = load_json(built_outputs[mode]['json'])
            eval_obj = evaluate_submission(
                submission=sub_obj,
                baseline=baseline,
                correctness_map_by_cache=correctness_map_by_cache,
            )
            unlabeled_obj = evaluate_unlabeled_change_rates(
                submission=sub_obj,
                baseline=baseline,
                unlabeled_cache_keys=unlabeled_cache_keys,
            )
            results_by_mode[mode] = {
                'labeled slices': {
                    'AUC-AUROC': eval_obj['labeled_metrics']['AUC-AUROC'],
                    'AUC-SelAcc': eval_obj['labeled_metrics']['AUC-SelAcc'],
                    'Stop@100': eval_obj['labeled_metrics']['Stop@100'],
                },
                'budget curves': eval_obj['labeled_budget_curve'],
                'delta vs baseline': eval_obj['delta_vs_baseline'],
                'unlabeled slices': unlabeled_obj,
            }

        coverage = compute_coverage_audit(reference=reference, dynamics_stats=args.dynamics_stats)
        validation = validate_submission(Path(args.output_main), Path(args.reference))

        combo = results_by_mode['rho_tail_plus_neg_A_accel']['labeled slices']
        rho_only = results_by_mode['rho_tail_only']['labeled slices']
        stability = {
            'combo_vs_rho_tail_only_AUROC_delta': float(combo['AUC-AUROC'] - rho_only['AUC-AUROC']),
            'combo_vs_rho_tail_only_SelAcc_delta': float(combo['AUC-SelAcc'] - rho_only['AUC-SelAcc']),
            'combo_vs_rho_tail_only_Stop_delta': float(combo['Stop@100'] - rho_only['Stop@100']),
            'is_combo_more_stable_than_rho_only': bool(combo['AUC-AUROC'] >= rho_only['AUC-AUROC']),
        }

        recommendation = {
            'worth_next_leaderboard_candidate': bool(
                stability['is_combo_more_stable_than_rho_only']
                and combo['Stop@100'] >= rho_only['Stop@100'] - 1e-12
            ),
            'reason': (
                'combo keeps stop performance while improving AUROC stability'
                if (stability['is_combo_more_stable_than_rho_only'] and combo['Stop@100'] >= rho_only['Stop@100'] - 1e-12)
                else 'combo does not beat rho_tail_only on stability+stop jointly in current labeled slices'
            ),
        }

        report = {
            'task': 'early_stop',
            'method_name': args.method_name,
            'analysis_requirements': {
                '1_labeled_slices': True,
                '2_budget_curves': True,
                '3_delta_vs_baseline': True,
                '4_unlabeled_change_rates': True,
                '5_coverage_audit': True,
            },
            'modes': results_by_mode,
            'coverage_audit': coverage,
            'labeled_cache_keys': sorted(correctness_map_by_cache.keys()),
            'unlabeled_cache_keys': sorted(unlabeled_cache_keys),
            'stability_check': stability,
            'candidate_assessment': recommendation,
            'validation': validation,
        }

        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

        notes_path = Path(args.output_notes)
        notes = load_json(notes_path)
        notes['explanation'] = {
            'feature_policy': '主方案仅使用 rho_tail 与 -A_accel；psi_mid 和 T_p_norm 已移出主通路。',
            'plugin_role': 'dynamics 作为局部插件，不是主排序干线。',
            'budget_policy': '仅在中后段 budget 进行增强，早段保持保守。',
            'coverage_policy': '默认 strict_local，不做全局补全，不做 30->970 强外推。',
            'evidence_scope': '当前主要证据来自 DS-R1/aime24 的真实 dynamics 覆盖。',
            'objective': '尽量保留 SelAcc/Stop 增益，同时降低 AUROC 被拖垮风险。',
        }
        notes['comparison_summary'] = {
            'modes': {
                mode: results_by_mode[mode]['labeled slices'] for mode in MODES
            },
            'combo_vs_rho_tail_only': stability,
            'benefit_slices': {
                'real_dynamics_nonzero_cache_problem': coverage['real_coverage']['cache_problem_ids'],
                'unlabeled_cache_change': {
                    mode: results_by_mode[mode]['unlabeled slices']['mean selection change rate'] for mode in MODES
                },
            },
            'leaderboard_candidate': recommendation,
        }
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2))

        print(f'wrote {report_path}')
        print(f'updated {notes_path}')


if __name__ == '__main__':
    main()
