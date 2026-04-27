#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.selectors.base import SelectorContext
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from plugins.medoid_tail_warning import MedoidTailWarningSelector
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


TARGET_CACHE_KEYS = [
    'DS-R1/brumo25',
    'DS-R1/gpqa',
    'DS-R1/hmmt25',
    'Qwen3-4B/brumo25',
    'Qwen3-4B/gpqa',
    'Qwen3-4B/hmmt25',
]

DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json'
DEFAULT_OUTPUT_CSV = '/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science.csv'
DEFAULT_OUTPUT_META = '/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science_meta.json'
DEFAULT_OUTPUT_PARQUET = '/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science.parquet'

NUMERIC_DATASETS = {'aime24', 'aime25', 'hmmt25'}
MC_DATASETS = {'gpqa', 'brumo25'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build candidate-level ML table (science subset only).')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='best_of_n score map JSON (mixed_v1 recommended).')
    parser.add_argument('--output-csv', default=DEFAULT_OUTPUT_CSV)
    parser.add_argument('--output-meta', default=DEFAULT_OUTPUT_META)
    parser.add_argument('--output-parquet', default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument(
        '--target-cache-keys',
        default=','.join(TARGET_CACHE_KEYS),
        help='Comma-separated cache keys. Defaults to the 6 science caches.',
    )
    parser.add_argument('--write-parquet', action='store_true', help='Also write parquet when pandas+pyarrow are available.')
    return parser.parse_args()


def safe_mean(arr: object) -> float:
    if arr is None:
        return float('nan')
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    return float(np.mean(x))


def token_length(token_view: object) -> float:
    for field in ('token_ids', 'tok_logprob', 'tok_selfcert', 'tok_conf', 'tok_neg_entropy', 'tok_gini'):
        arr = getattr(token_view, field, None)
        if arr is not None:
            return float(len(arr))
    return float('nan')


def ranked_items(sid_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(((str(s), float(v)) for s, v in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))


def _unwrap_literal_answer(text: object) -> str:
    raw = '' if text is None else str(text).strip()
    if not raw:
        return ''
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return raw
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return ''
        return str(parsed[0]).strip()
    if parsed is None:
        return ''
    return str(parsed).strip()


def _strip_boxed(text: str) -> str:
    s = str(text)
    while True:
        m = re.search(r'\\boxed\s*\{([^{}]*)\}', s)
        if not m:
            break
        s = (s[:m.start()] + m.group(1) + s[m.end():]).strip()
    return s


def canonical_multiple_choice(text: object) -> str:
    x = _unwrap_literal_answer(text).upper().strip()
    x = re.sub(r'[\$\s`\'\"]+', '', x)
    if len(x) == 1 and x in {'A', 'B', 'C', 'D'}:
        return x
    m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', x)
    if m:
        return m.group(1)
    m = re.search(r'(?:OPTION|ANS(?:WER)?)[:：]?\s*([ABCD])', x)
    if m:
        return m.group(1)
    return x


def _canonical_decimal(text: str) -> Optional[str]:
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return None
    normalized = dec.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    out = format(normalized, 'f').rstrip('0').rstrip('.')
    return out if out else '0'


def canonical_numeric(text: object, apply_mod_1000: bool = False) -> str:
    x = _unwrap_literal_answer(text)
    x = _strip_boxed(x)
    x = x.replace('\\left', '').replace('\\right', '')
    x = x.replace('$', '').replace(',', '').strip()
    x = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'\1/\2', x)
    x = re.sub(r'\s+', '', x)
    if not x:
        return ''

    if re.fullmatch(r'[+-]?\d+', x):
        val = int(x)
        if apply_mod_1000:
            val %= 1000
        return str(val)

    frac_match = re.fullmatch(r'([+-]?\d+)/([+-]?\d+)', x)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den == 0:
            return x
        frac = Fraction(num, den)
        if frac.denominator == 1:
            val = frac.numerator
            if apply_mod_1000:
                val %= 1000
            return str(val)
        return f'{frac.numerator}/{frac.denominator}'

    dec_out = _canonical_decimal(x)
    if dec_out is not None:
        if re.fullmatch(r'[+-]?\d+', dec_out):
            val = int(dec_out)
            if apply_mod_1000:
                val %= 1000
            return str(val)
        return dec_out

    return x


def normalize_answer_for_dataset(extracted_answer: object, dataset: str) -> str:
    ds = str(dataset).strip().lower()
    if ds in NUMERIC_DATASETS:
        return f'NUM::{canonical_numeric(extracted_answer, apply_mod_1000=False)}'
    if ds in MC_DATASETS:
        return f'MC::{canonical_multiple_choice(extracted_answer)}'
    return f'TXT::{_unwrap_literal_answer(extracted_answer)}'


def is_integer_text(text: object) -> bool:
    s = str(text).strip()
    if not s:
        return False
    return re.fullmatch(r'[+-]?\d+', s) is not None


def parse_dataset(cache_key: str) -> str:
    return str(cache_key).split('/', 1)[1] if '/' in str(cache_key) else str(cache_key)


def parse_model_family(cache_key: str) -> str:
    return str(cache_key).split('/', 1)[0] if '/' in str(cache_key) else str(cache_key)


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float('nan')
    try:
        out = float(value)
    except Exception:
        return float('nan')
    return out if np.isfinite(out) else float('nan')


def load_eval_run_info(cache_root: str) -> Tuple[Dict[int, Dict[str, object]], List[str], Optional[str]]:
    root = Path(cache_root)
    meta_path = root / 'meta.json'
    if not meta_path.exists():
        return {}, [str(meta_path)], None

    deps: List[str] = [str(meta_path)]
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    sample_index: Dict[Tuple[str, int], int] = {}
    for sid, sample in enumerate(meta.get('samples', [])):
        pid = str(sample.get('problem_id'))
        run_index = int(sample.get('run_index', 0))
        sample_index[(pid, run_index)] = int(sid)

    report_path: Optional[Path] = None
    report = None
    for name in ('evaluation_report_compact.json', 'evaluation_report.json'):
        p = root / name
        if p.exists():
            report_path = p
            report = json.loads(p.read_text(encoding='utf-8'))
            deps.append(str(p))
            break

    if report is None or 'results' not in report:
        return {}, deps, None

    info: Dict[int, Dict[str, object]] = {}
    for result in report.get('results', []):
        pid = str(result.get('problem_id'))
        for run in result.get('runs', []):
            run_idx = int(run.get('run_index', run.get('index', 0)))
            sid = sample_index.get((pid, run_idx))
            if sid is None:
                continue
            extracted = run.get('extracted_answer', '')
            extracted_unwrapped = _unwrap_literal_answer(extracted)
            output_tokens = _to_float_or_nan(run.get('output_tokens'))
            is_correct = run.get('is_correct', None)
            info[sid] = {
                'extracted_answer': '' if extracted is None else str(extracted),
                'extracted_unwrapped': extracted_unwrapped,
                'parse_success': bool(extracted_unwrapped),
                'is_integer_answer': bool(is_integer_text(extracted_unwrapped)),
                'output_tokens': output_tokens,
                'is_correct_from_report': (None if is_correct is None else bool(is_correct)),
            }

    return info, deps, (str(report_path) if report_path is not None else None)


def build_cache_rows(
    cache_key: str,
    problem_scores: Dict[str, Dict[str, float]],
    cache_root: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[str]]:
    rows: List[Dict[str, object]] = []
    deps: List[str] = [cache_root]

    reader = CacheReader(cache_root)
    eval_info, eval_deps, report_path = load_eval_run_info(cache_root)
    deps.extend(eval_deps)

    try:
        corr_map = load_correctness_map(cache_root)
        has_correctness_map = True
    except Exception:
        corr_map = {}
        has_correctness_map = False

    medoid = MedoidTailWarningSelector()
    medoid.bind(SelectorContext(cache=reader, problem_id='__all__', run_ids=[], views=[]))

    cache_problem_count = 0
    cache_candidate_count = 0
    label_non_null = 0

    model_family = parse_model_family(cache_key)
    benchmark_name = parse_dataset(cache_key)

    for problem_id, sid_score_map in problem_scores.items():
        cache_problem_count += 1

        sid_scores = {str(sid): float(score) for sid, score in sid_score_map.items()}
        ranked = ranked_items(sid_scores)
        if not ranked:
            continue

        rank_map = {sid: rank + 1 for rank, (sid, _) in enumerate(ranked)}
        top1_score = float(ranked[0][1])
        top2_score = float(ranked[1][1]) if len(ranked) > 1 else float('nan')

        num_candidates = len(ranked)
        answer_norm_map: Dict[str, str] = {}
        support_key_map: Dict[str, str] = {}

        for sid, _ in ranked:
            sid_int = int(sid)
            extracted = eval_info.get(sid_int, {}).get('extracted_answer', '')
            answer_norm = normalize_answer_for_dataset(extracted, benchmark_name)
            answer_norm_map[sid] = answer_norm
            support_key_map[sid] = answer_norm if answer_norm not in {'NUM::', 'MC::', 'TXT::'} else ''

        support_counter = Counter(v for v in support_key_map.values() if v)
        unique_answer_count = len(support_counter)

        for sid, score in ranked:
            sid_int = int(sid)
            cache_candidate_count += 1

            token_view = reader.get_token_view(sid_int)
            tok_logprob_mean = safe_mean(getattr(token_view, 'tok_logprob', None))
            tok_selfcert_mean = safe_mean(getattr(token_view, 'tok_selfcert', None))
            tok_conf_mean = safe_mean(getattr(token_view, 'tok_conf', None))
            tok_neg_entropy_mean = safe_mean(getattr(token_view, 'tok_neg_entropy', None))
            tok_gini_mean = safe_mean(getattr(token_view, 'tok_gini', None))

            metrics = medoid._tail_metrics(sid_int)
            if metrics is None:
                tail_warning = None
                tail_new_ratio = float('nan')
                plateau_progress = float('nan')
                cumulative_unique_neurons_end = float('nan')
            else:
                tail_warning = bool(medoid._is_warned(metrics))
                tail_new_ratio = float(metrics.get('tail_new_ratio', float('nan')))
                plateau_progress = float(metrics.get('plateau_progress', float('nan')))
                cumulative_unique_neurons_end = float(metrics.get('final_count', float('nan')))

            eval_rec = eval_info.get(sid_int, {})
            extracted_answer = str(eval_rec.get('extracted_answer', ''))
            answer_unwrapped = str(eval_rec.get('extracted_unwrapped', ''))
            parse_success = bool(eval_rec.get('parse_success', False))
            is_integer_answer = bool(eval_rec.get('is_integer_answer', False))

            answer_length_tokens = _to_float_or_nan(eval_rec.get('output_tokens'))
            if not np.isfinite(answer_length_tokens):
                answer_length_tokens = token_length(token_view)

            is_correct: Optional[bool]
            if has_correctness_map:
                is_correct = bool(corr_map.get(sid_int, False))
            else:
                from_report = eval_rec.get('is_correct_from_report', None)
                is_correct = None if from_report is None else bool(from_report)

            if is_correct is not None:
                label_non_null += 1

            support_key = support_key_map.get(sid, '')
            support_count_same_answer = support_counter.get(support_key, 0) if support_key else float('nan')

            row = {
                'cache_key': cache_key,
                'model_family': model_family,
                'benchmark_name': benchmark_name,
                'problem_id': str(problem_id),
                'sid': sid,
                'is_correct': is_correct,
                'mixed_v1_score': float(score),
                'rank_by_mixed_v1': int(rank_map[sid]),
                'top1_gap_under_mixed_v1': float(top1_score - score),
                'top2_gap_under_mixed_v1': (float(top2_score - score) if np.isfinite(top2_score) else float('nan')),
                'tok_logprob_mean': tok_logprob_mean,
                'tok_selfcert_mean': tok_selfcert_mean,
                'tok_conf_mean': tok_conf_mean,
                'tok_neg_entropy_mean': tok_neg_entropy_mean,
                'tok_gini_mean': tok_gini_mean,
                'answer_length_tokens': float(answer_length_tokens),
                'parse_success': bool(parse_success),
                'is_integer_answer': bool(is_integer_answer),
                'extracted_answer': extracted_answer,
                'answer_norm': answer_norm_map.get(sid, ''),
                'tail_warning': tail_warning,
                'tail_new_ratio': tail_new_ratio,
                'plateau_progress': plateau_progress,
                'cumulative_unique_neurons_end': cumulative_unique_neurons_end,
                'num_candidates_for_problem': int(num_candidates),
                'support_count_same_answer': support_count_same_answer,
                'unique_answer_count_in_problem': int(unique_answer_count),
            }
            rows.append(row)

    stats = {
        'cache_key': cache_key,
        'cache_root': cache_root,
        'report_path': report_path,
        'problem_count': cache_problem_count,
        'candidate_count': cache_candidate_count,
        'has_correctness_map': has_correctness_map,
        'is_correct_non_null_count': label_non_null,
    }
    return rows, stats, deps


def is_missing(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        return math.isnan(v)
    if isinstance(v, str):
        return v == ''
    return False


def non_null_rate(rows: List[Dict[str, object]], col: str) -> float:
    if not rows:
        return 0.0
    nn = 0
    for r in rows:
        if not is_missing(r.get(col)):
            nn += 1
    return float(nn / len(rows))


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text('')
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_parquet(rows: List[Dict[str, object]], output_parquet: Path, enabled: bool) -> Dict[str, object]:
    result = {
        'requested': bool(enabled),
        'written': False,
        'path': str(output_parquet),
        'reason': 'not_requested',
    }
    if not enabled:
        return result

    try:
        import pandas as pd
    except Exception as e:
        result['reason'] = f'pandas_import_failed: {e}'
        return result

    try:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(output_parquet, index=False)
        result['written'] = True
        result['reason'] = 'ok'
        return result
    except Exception as e:
        result['reason'] = f'parquet_write_failed: {e}'
        return result


def list_feature_columns(rows: List[Dict[str, object]]) -> List[str]:
    if not rows:
        return []
    base = {'cache_key', 'model_family', 'benchmark_name', 'problem_id', 'sid', 'is_correct'}
    return [c for c in rows[0].keys() if c not in base]


def main() -> None:
    args = parse_args()

    target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]
    base = json.loads(Path(args.input).read_text(encoding='utf-8'))
    score_map_all = base.get('scores', {})

    all_rows: List[Dict[str, object]] = []
    per_cache_stats: Dict[str, Dict[str, object]] = {}
    dep_paths: List[str] = [str(Path(args.input).resolve())]

    missing_target_cache_keys: List[str] = []

    for cache_key in target_cache_keys:
        if cache_key not in score_map_all:
            missing_target_cache_keys.append(cache_key)
            continue
        cache_root = DEFAULT_CACHE_MAP.get(cache_key)
        if cache_root is None:
            missing_target_cache_keys.append(cache_key)
            continue

        rows, stats, deps = build_cache_rows(
            cache_key=cache_key,
            problem_scores=score_map_all[cache_key],
            cache_root=cache_root,
        )
        all_rows.extend(rows)
        per_cache_stats[cache_key] = stats
        dep_paths.extend(deps)

    output_csv = Path(args.output_csv)
    output_meta = Path(args.output_meta)
    output_parquet = Path(args.output_parquet)

    write_csv(all_rows, output_csv)
    parquet_status = maybe_write_parquet(all_rows, output_parquet, enabled=bool(args.write_parquet))

    key_columns = [
        'is_correct',
        'mixed_v1_score',
        'rank_by_mixed_v1',
        'top1_gap_under_mixed_v1',
        'top2_gap_under_mixed_v1',
        'tok_logprob_mean',
        'tok_selfcert_mean',
        'tok_conf_mean',
        'tok_neg_entropy_mean',
        'tok_gini_mean',
        'answer_length_tokens',
        'parse_success',
        'is_integer_answer',
        'extracted_answer',
        'answer_norm',
        'tail_warning',
        'tail_new_ratio',
        'plateau_progress',
        'cumulative_unique_neurons_end',
        'num_candidates_for_problem',
        'support_count_same_answer',
        'unique_answer_count_in_problem',
    ]

    non_null_rates = {col: non_null_rate(all_rows, col) for col in key_columns}

    fully_available = sorted([k for k, v in non_null_rates.items() if v >= 0.999999])
    partially_available = sorted([k for k, v in non_null_rates.items() if (v > 0.0 and v < 0.999999)])
    unavailable = sorted([k for k, v in non_null_rates.items() if v <= 0.0])

    total_rows = len(all_rows)
    labeled_rows = sum(1 for row in all_rows if row.get('is_correct') is not None)

    rows_per_cache = {
        ck: int(st.get('candidate_count', 0))
        for ck, st in per_cache_stats.items()
    }
    is_correct_coverage_per_cache = {
        ck: (
            float(st.get('is_correct_non_null_count', 0) / max(int(st.get('candidate_count', 0)), 1))
            if int(st.get('candidate_count', 0)) > 0 else 0.0
        )
        for ck, st in per_cache_stats.items()
    }

    meta = {
        'task': 'ml_candidate_table_v1_science',
        'input_scores': args.input,
        'method_name_from_input': base.get('method_name'),
        'target_cache_keys': target_cache_keys,
        'missing_target_cache_keys': missing_target_cache_keys,
        'output_csv': str(output_csv),
        'output_parquet': parquet_status,
        'total_rows': total_rows,
        'rows_per_cache_key': rows_per_cache,
        'cache_key_coverage': {
            'requested': target_cache_keys,
            'covered': sorted(rows_per_cache.keys()),
            'missing': missing_target_cache_keys,
        },
        'is_correct_coverage': {
            'non_null_rows': labeled_rows,
            'total_rows': total_rows,
            'coverage': (float(labeled_rows / total_rows) if total_rows > 0 else 0.0),
            'per_cache': is_correct_coverage_per_cache,
        },
        'key_feature_non_null_rate': non_null_rates,
        'feature_availability': {
            'fully_available': fully_available,
            'partially_available': partially_available,
            'unavailable': unavailable,
        },
        'columns': (list(all_rows[0].keys()) if all_rows else []),
        'feature_columns': list_feature_columns(all_rows),
        'per_cache_details': per_cache_stats,
        'dependencies': sorted(set(dep_paths)),
    }

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'rows={total_rows}')
    print(f'wrote {output_csv}')
    print(f'wrote {output_meta}')
    if parquet_status.get('requested'):
        print(f"parquet_written={parquet_status.get('written')} reason={parquet_status.get('reason')}")


if __name__ == '__main__':
    main()
