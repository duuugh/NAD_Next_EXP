#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/jovyan/work/NAD_Next')
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP
from nad.ops.accuracy import load_correctness_map


MODES = ['rho_tail_only', 'neg_A_accel_only', 'rho_tail_plus_neg_A_accel']
DEFAULT_BENCHMARKS = ['aime25', 'hmmt25', 'gpqa', 'lcb_v5']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-run v2_local evaluations on DS-R1 benchmarks.')
    parser.add_argument('--benchmarks', default=','.join(DEFAULT_BENCHMARKS))
    parser.add_argument('--include-brumo', action='store_true')
    parser.add_argument('--result-dir', default='/home/jovyan/work/NAD_Next/result')
    parser.add_argument('--dynamics-dir', default='/home/jovyan/work/NAD_Next/result/dynamics_full')
    parser.add_argument('--reference', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--build-script', default='/home/jovyan/work/NAD_Next/scripts/build_early_stop_dynamics_v2_local.py')
    parser.add_argument('--ensure-dynamics', action='store_true', help='Build missing per-benchmark dynamics CSVs before eval.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dyn-strength', type=float, default=0.60)
    parser.add_argument('--gamma', type=float, default=1.6)
    return parser.parse_args()


def score_at_budget(value: Any, idx: int) -> float:
    if isinstance(value, list):
        return float(value[idx])
    return float(value)


def sid_key(sid: str) -> Tuple[int, str]:
    s = str(sid)
    if s.isdigit():
        return (0, f'{int(s):012d}')
    return (1, s)


def pick_top1(sample_map: Mapping[str, Any], idx: int) -> Tuple[str, float]:
    best_sid = None
    best_score = None
    for sid, value in sample_map.items():
        score = score_at_budget(value, idx)
        sid_str = str(sid)
        if best_sid is None:
            best_sid = sid_str
            best_score = score
            continue
        if score > float(best_score):
            best_sid = sid_str
            best_score = score
        elif abs(score - float(best_score)) <= 1e-12 and sid_key(sid_str) < sid_key(str(best_sid)):
            best_sid = sid_str
            best_score = score
    return str(best_sid), float(best_score)


def compute_metrics_for_cache(sub_obj: Dict[str, Any], cache_key: str, corr_map: Dict[int, bool]) -> Dict[str, Any]:
    budgets = [i / 10.0 for i in range(1, 11)]
    curve = []
    score_map = sub_obj['scores'][cache_key]

    for bidx, budget in enumerate(budgets):
        y = []
        s = []
        for problem_id, sample_map in score_map.items():
            sid, score = pick_top1(sample_map, bidx)
            y.append(1 if corr_map.get(int(sid), False) else 0)
            s.append(float(score))
        y_arr = np.asarray(y, dtype=np.int64)
        s_arr = np.asarray(s, dtype=np.float64)
        selacc = float(np.mean(y_arr)) if len(y_arr) else float('nan')
        auroc = float('nan')
        if len(np.unique(y_arr)) >= 2:
            auroc = float(roc_auc_score(y_arr, s_arr))
        curve.append({'budget': budget, 'SelAcc': selacc, 'AUROC': auroc})

    agg = {
        'AUC-AUROC': float(np.nanmean([x['AUROC'] for x in curve])),
        'AUC-SelAcc': float(np.nanmean([x['SelAcc'] for x in curve])),
        'Stop@100': float(curve[-1]['SelAcc']),
    }
    return {'metrics': agg, 'budget_curves': curve}


def build_submission(build_script: str, reference: str, out_json: Path, out_notes: Path, mode: str | None, dynamics_stats: List[str], dyn_strength: float, gamma: float) -> None:
    cmd = [
        sys.executable,
        str(build_script),
        '--reference', str(reference),
        '--final-scores', str(reference),
        '--output', str(out_json),
        '--notes-output', str(out_notes),
        '--gamma', str(gamma),
        '--dyn-strength', str(dyn_strength),
    ]
    if mode is not None:
        cmd.extend(['--mode', str(mode)])
    for item in dynamics_stats:
        cmd.extend(['--dynamics-stats', str(item)])
    subprocess.run(cmd, check=True)


def maybe_build_dynamics(benchmarks: List[str], dynamics_dir: Path) -> None:
    cmd = [
        sys.executable,
        '/home/jovyan/work/NAD_Next/scripts/build_dynamics_stats_batch_ds_r1.py',
        '--benchmarks', ','.join(benchmarks),
        '--out-dir', str(dynamics_dir),
    ]
    subprocess.run(cmd, check=True)


def benchmark_recommendation(row: Dict[str, Any]) -> Tuple[bool, str]:
    deltas = row['delta_vs_baseline']

    rho_eff = (deltas['rho_tail_only']['delta AUC-SelAcc'] > 0) or (deltas['rho_tail_only']['delta Stop@100'] > 0)
    neg = deltas['neg_A_accel_only']
    combo = deltas['rho_tail_plus_neg_A_accel']
    rho = deltas['rho_tail_only']

    stable_neg = (
        neg['delta AUC-AUROC'] >= max(rho['delta AUC-AUROC'], combo['delta AUC-AUROC'])
        and neg['delta AUC-SelAcc'] > -0.002
        and neg['delta Stop@100'] > -0.002
    )

    combo_best = (
        combo['delta AUC-SelAcc'] >= max(rho['delta AUC-SelAcc'], neg['delta AUC-SelAcc']) - 1e-12
        and combo['delta Stop@100'] >= max(rho['delta Stop@100'], neg['delta Stop@100']) - 1e-12
        and combo['delta AUC-AUROC'] >= rho['delta AUC-AUROC'] - 0.005
    )

    all_nonpos = all(
        (v['delta AUC-SelAcc'] <= 0 and v['delta Stop@100'] <= 0 and v['delta AUC-AUROC'] <= 0)
        for v in deltas.values()
    )

    if all_nonpos:
        return False, 'disable'
    if combo_best:
        return True, 'rho_tail_plus_neg_A_accel'
    if rho_eff and rho['delta AUC-SelAcc'] >= neg['delta AUC-SelAcc'] - 1e-12:
        return True, 'rho_tail_only'
    if stable_neg:
        return True, 'neg_A_accel_only'
    return False, 'disable'


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dynamics_dir = Path(args.dynamics_dir)
    dynamics_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [x.strip() for x in str(args.benchmarks).split(',') if x.strip()]
    if args.include_brumo and 'brumo25' not in benchmarks:
        benchmarks.append('brumo25')

    for b in benchmarks:
        cache_key = f'DS-R1/{b}'
        if cache_key not in DEFAULT_CACHE_MAP:
            raise ValueError(f'unsupported benchmark/cache: {cache_key}')
        cache_root = Path(DEFAULT_CACHE_MAP[cache_key])
        if not cache_root.exists():
            raise FileNotFoundError(f'missing cache root for {cache_key}: {cache_root}')

    if args.ensure_dynamics:
        maybe_build_dynamics(benchmarks, dynamics_dir)

    reference = json.loads(Path(args.reference).read_text())

    batch_rows = {}
    recommendation_map = {}

    for benchmark in benchmarks:
        cache_key = f'DS-R1/{benchmark}'
        corr_map = load_correctness_map(DEFAULT_CACHE_MAP[cache_key])

        dyn_csv = dynamics_dir / f'dynamics_statistics_DS-R1_{benchmark}.csv'
        if not dyn_csv.exists():
            raise FileNotFoundError(
                f'missing dynamics CSV for {cache_key}: {dyn_csv}. '
                f'Run scripts/build_dynamics_stats_batch_ds_r1.py first or pass --ensure-dynamics.'
            )

        problem_total = len(reference['scores'][cache_key])
        dyn_df = json.loads(Path(dyn_csv).read_text().replace('NaN', 'null')) if False else None
        import pandas as pd
        _df = pd.read_csv(dyn_csv)
        real_problem_ids = sorted(set(str(x) for x in _df['problem_id'].astype(str).unique()), key=lambda x: int(x) if x.isdigit() else x)
        real_cov_count = len([pid for pid in reference['scores'][cache_key].keys() if str(pid) in set(real_problem_ids)])

        with tempfile.TemporaryDirectory() as tdir:
            tdirp = Path(tdir)

            baseline_json = tdirp / f'baseline_{benchmark}.json'
            baseline_notes = tdirp / f'baseline_{benchmark}_notes.json'
            build_submission(
                build_script=args.build_script,
                reference=args.reference,
                out_json=baseline_json,
                out_notes=baseline_notes,
                mode='rho_tail_plus_neg_A_accel',
                dynamics_stats=[],
                dyn_strength=args.dyn_strength,
                gamma=args.gamma,
            )
            base_obj = json.loads(baseline_json.read_text())
            baseline_eval = compute_metrics_for_cache(base_obj, cache_key, corr_map)

            per_mode_eval = {}
            delta_vs_baseline = {}
            for mode in MODES:
                out_json = tdirp / f'{mode}_{benchmark}.json'
                out_notes = tdirp / f'{mode}_{benchmark}_notes.json'
                build_submission(
                    build_script=args.build_script,
                    reference=args.reference,
                    out_json=out_json,
                    out_notes=out_notes,
                    mode=mode,
                    dynamics_stats=[f'{cache_key}={dyn_csv}'],
                    dyn_strength=args.dyn_strength,
                    gamma=args.gamma,
                )
                sub_obj = json.loads(out_json.read_text())
                ev = compute_metrics_for_cache(sub_obj, cache_key, corr_map)
                per_mode_eval[mode] = ev

                delta_vs_baseline[mode] = {
                    'delta AUC-AUROC': float(ev['metrics']['AUC-AUROC'] - baseline_eval['metrics']['AUC-AUROC']),
                    'delta AUC-SelAcc': float(ev['metrics']['AUC-SelAcc'] - baseline_eval['metrics']['AUC-SelAcc']),
                    'delta Stop@100': float(ev['metrics']['Stop@100'] - baseline_eval['metrics']['Stop@100']),
                }

        per_bench_out = {
            'cache_key': cache_key,
            'baseline': baseline_eval,
            'modes': per_mode_eval,
            'delta_vs_baseline': delta_vs_baseline,
            'coverage_audit': {
                'dynamics_problem_count': int(real_cov_count),
                'is_real_coverage': True,
                'used_local_expansion': False,
                'local_expansion_problem_count': 0,
                'coverage_rate': float(real_cov_count / problem_total) if problem_total > 0 else 0.0,
                'problem_total': int(problem_total),
                'real_problem_ids': real_problem_ids,
            },
        }

        enable, mode_rec = benchmark_recommendation(per_bench_out)
        per_bench_out['decision'] = {
            'should_enable_plugin': bool(enable),
            'recommended_mode': mode_rec,
        }

        per_out_path = result_dir / f'dynamics_v2_local_eval_{cache_key.replace("/", "_")}.json'
        per_out_path.write_text(json.dumps(per_bench_out, ensure_ascii=False, indent=2))
        print(f'wrote {per_out_path}')

        batch_rows[cache_key] = per_bench_out
        recommendation_map[cache_key] = mode_rec

    # answer required summary questions
    effective_rho = {
        ck: ((row['delta_vs_baseline']['rho_tail_only']['delta AUC-SelAcc'] > 0)
             or (row['delta_vs_baseline']['rho_tail_only']['delta Stop@100'] > 0))
        for ck, row in batch_rows.items()
    }

    stable_neg = {}
    combo_best = {}
    disable_map = {}
    for ck, row in batch_rows.items():
        deltas = row['delta_vs_baseline']
        neg = deltas['neg_A_accel_only']
        rho = deltas['rho_tail_only']
        combo = deltas['rho_tail_plus_neg_A_accel']

        stable_neg[ck] = bool(
            neg['delta AUC-AUROC'] >= max(rho['delta AUC-AUROC'], combo['delta AUC-AUROC'])
            and neg['delta AUC-SelAcc'] > -0.002
            and neg['delta Stop@100'] > -0.002
        )
        combo_best[ck] = bool(
            row['decision']['recommended_mode'] == 'rho_tail_plus_neg_A_accel'
            and row['decision']['should_enable_plugin']
        )
        disable_map[ck] = bool(not row['decision']['should_enable_plugin'])

    summary = {
        'task': 'dynamics_v2_local_batch_eval',
        'benchmarks': benchmarks,
        'per_benchmark': batch_rows,
        'qa_answers': {
            'which_benchmarks_rho_tail_only_effective': effective_rho,
            'which_benchmarks_neg_A_accel_only_more_stable': stable_neg,
            'which_benchmarks_combo_best': combo_best,
            'which_benchmarks_should_disable_plugin': disable_map,
            'final_recommended_enable_strategy': recommendation_map,
        },
        'final_recommendation_text': (
            'Use benchmark-selective plugin enabling. Do not globally enable across all DS-R1 benchmarks unless all per-benchmark deltas are consistently positive.'
        ),
    }

    summary_json = result_dir / 'dynamics_v2_local_batch_summary.json'
    summary_md = result_dir / 'dynamics_v2_local_batch_summary.md'
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = []
    lines.append('# Dynamics v2 local batch summary (DS-R1)')
    lines.append('')
    lines.append('## Per-benchmark verdict')
    for ck in sorted(batch_rows.keys()):
        row = batch_rows[ck]
        dec = row['decision']
        lines.append(
            f'- {ck}: recommended={dec["recommended_mode"]}, should_enable_plugin={dec["should_enable_plugin"]}, '
            f'coverage={row["coverage_audit"]["dynamics_problem_count"]}/{row["coverage_audit"]["problem_total"]} '
            f'({row["coverage_audit"]["coverage_rate"]:.3f})'
        )
    lines.append('')
    lines.append('## Required answers')
    lines.append(f'- rho_tail_only effective: {summary["qa_answers"]["which_benchmarks_rho_tail_only_effective"]}')
    lines.append(f'- neg_A_accel_only more stable: {summary["qa_answers"]["which_benchmarks_neg_A_accel_only_more_stable"]}')
    lines.append(f'- combo best: {summary["qa_answers"]["which_benchmarks_combo_best"]}')
    lines.append(f'- should disable: {summary["qa_answers"]["which_benchmarks_should_disable_plugin"]}')
    lines.append(f'- final strategy: {summary["qa_answers"]["final_recommended_enable_strategy"]}')
    lines.append('')
    lines.append('## Final policy')
    lines.append(f'- {summary["final_recommendation_text"]}')

    summary_md.write_text('\n'.join(lines), encoding='utf-8')

    print(f'wrote {summary_json}')
    print(f'wrote {summary_md}')


if __name__ == '__main__':
    main()
