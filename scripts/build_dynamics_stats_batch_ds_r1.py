#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, '/home/jovyan/work/NAD_Next')
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_BENCHMARKS = ['aime25', 'hmmt25', 'gpqa', 'lcb_v5']
OPTIONAL_BENCHMARKS = ['brumo25']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-generate DS-R1 dynamics statistics CSVs.')
    parser.add_argument('--benchmarks', default=','.join(DEFAULT_BENCHMARKS), help='Comma-separated DS-R1 benchmarks.')
    parser.add_argument('--include-brumo', action='store_true', help='Also include brumo25.')
    parser.add_argument('--out-dir', default='/home/jovyan/work/NAD_Next/result/dynamics_full')
    parser.add_argument('--merged-output', default='/home/jovyan/work/NAD_Next/result/dynamics_full/dynamics_statistics_DS-R1_merged.csv')
    parser.add_argument('--max-runs', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def ensure_cache(cache_key: str) -> str:
    if cache_key not in DEFAULT_CACHE_MAP:
        raise KeyError(f'cache_key not in DEFAULT_CACHE_MAP: {cache_key}')
    cache_root = DEFAULT_CACHE_MAP[cache_key]
    if not Path(cache_root).exists():
        raise FileNotFoundError(f'cache root missing for {cache_key}: {cache_root}')
    return cache_root


def run_single(cache_key: str, cache_root: str, out_csv: Path, max_runs: int | None) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as tdir:
        out_dir = Path(tdir)
        cmd = [
            sys.executable,
            '/home/jovyan/work/NAD_Next/scripts/validate_dynamics_model.py',
            '--cache', str(cache_root),
            '--out', str(out_dir),
        ]
        if max_runs is not None:
            cmd.extend(['--max-runs', str(max_runs)])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr.strip()[-1000:]
            stdout = proc.stdout.strip()[-1000:]
            raise RuntimeError(
                f'validate_dynamics_model failed for {cache_key}\n'
                f'cache_root={cache_root}\n'
                f'stdout_tail={stdout}\n'
                f'stderr_tail={stderr}'
            )

        src = out_dir / 'dynamics_statistics.csv'
        if not src.exists():
            raise FileNotFoundError(f'{cache_key}: missing generated dynamics_statistics.csv in {out_dir}')

        df = pd.read_csv(src)
        if 'problem_id' not in df.columns:
            raise ValueError(f'{cache_key}: dynamics_statistics.csv missing problem_id')
        df['problem_id'] = df['problem_id'].astype(str)
        df.to_csv(out_csv, index=False)

        return {
            'cache_key': cache_key,
            'cache_root': cache_root,
            'out_csv': str(out_csv),
            'rows': int(len(df)),
            'problem_count': int(df['problem_id'].nunique()),
            'run_count_per_problem_min': int(df.groupby('problem_id').size().min()) if len(df) else 0,
            'run_count_per_problem_max': int(df.groupby('problem_id').size().max()) if len(df) else 0,
        }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [x.strip() for x in str(args.benchmarks).split(',') if x.strip()]
    if args.include_brumo and 'brumo25' not in benchmarks:
        benchmarks.append('brumo25')

    benchmark_set = list(dict.fromkeys(benchmarks))
    for b in benchmark_set:
        if b not in set(DEFAULT_BENCHMARKS + OPTIONAL_BENCHMARKS):
            raise ValueError(f'unsupported benchmark `{b}`; allowed: {DEFAULT_BENCHMARKS + OPTIONAL_BENCHMARKS}')

    records: List[Dict[str, object]] = []
    merged_frames: List[pd.DataFrame] = []

    for benchmark in benchmark_set:
        cache_key = f'DS-R1/{benchmark}'
        cache_root = ensure_cache(cache_key)
        out_csv = out_dir / f'dynamics_statistics_DS-R1_{benchmark}.csv'

        if out_csv.exists() and not args.overwrite:
            df = pd.read_csv(out_csv)
            df['cache_key'] = cache_key
            merged_frames.append(df)
            records.append({
                'cache_key': cache_key,
                'cache_root': cache_root,
                'out_csv': str(out_csv),
                'rows': int(len(df)),
                'problem_count': int(df['problem_id'].astype(str).nunique()) if 'problem_id' in df.columns else 0,
                'run_count_per_problem_min': int(df.groupby('problem_id').size().min()) if 'problem_id' in df.columns and len(df) else 0,
                'run_count_per_problem_max': int(df.groupby('problem_id').size().max()) if 'problem_id' in df.columns and len(df) else 0,
                'reused_existing': True,
            })
            print(f'[reuse] {cache_key} -> {out_csv}')
            continue

        print(f'[run] {cache_key}')
        rec = run_single(cache_key=cache_key, cache_root=cache_root, out_csv=out_csv, max_runs=args.max_runs)
        rec['reused_existing'] = False
        records.append(rec)

        df_new = pd.read_csv(out_csv)
        df_new['cache_key'] = cache_key
        merged_frames.append(df_new)

    merged_out = Path(args.merged_output)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    if merged_frames:
        pd.concat(merged_frames, ignore_index=True).to_csv(merged_out, index=False)
    else:
        pd.DataFrame().to_csv(merged_out, index=False)

    summary = {
        'benchmarks': benchmark_set,
        'outputs': records,
        'merged_csv': str(merged_out),
        'total_rows': int(sum(int(x['rows']) for x in records)),
        'total_problem_count': int(sum(int(x['problem_count']) for x in records)),
    }
    summary_path = out_dir / 'dynamics_statistics_DS-R1_batch_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f'wrote {merged_out}')
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
