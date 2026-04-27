#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


BUDGETS = [i / 10.0 for i in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build early-stop submission from mean-confidence backbone.')
    parser.add_argument('--reference', default='/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json')
    parser.add_argument('--cache-map-json', default=None, help='Optional cache map JSON; default uses DEFAULT_CACHE_MAP.')
    parser.add_argument('--confidence-source', choices=['auto', 'tok_logprob', 'tok_conf', 'tok_neg_entropy'], default='auto')
    parser.add_argument(
        '--normalize-mode',
        choices=['none', 'cache_budget_quantile', 'rank_problem_per_budget'],
        default='cache_budget_quantile',
    )
    parser.add_argument(
        '--aggregate-mode',
        choices=['prefix_mean_logprob', 'tail_mean_logprob', 'trimmed_mean_logprob'],
        default='prefix_mean_logprob',
    )
    parser.add_argument('--tail-ratio', type=float, default=0.4, help='Used by tail_mean_logprob.')
    parser.add_argument('--trim-worst-ratio', type=float, default=0.2, help='Used by trimmed_mean_logprob.')
    parser.add_argument('--quantile-low', type=float, default=0.05)
    parser.add_argument('--quantile-high', type=float, default=0.95)
    parser.add_argument('--output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence.json')
    parser.add_argument('--notes-output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_notes.json')
    parser.add_argument('--report-output', default='/home/jovyan/work/NAD_Next/result/early_stop_mean_confidence_report.json')
    parser.add_argument('--method-name', default='early_stop_mean_confidence')
    parser.add_argument('--round-digits', type=int, default=8)
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


def rank_transform(values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(((str(k), float(v)) for k, v in values.items()), key=lambda x: (x[1], x[0]))
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}
    denom = float(n - 1)
    return {sid: idx / denom for idx, (sid, _) in enumerate(items)}


def resolve_cache_map(path: str | None) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_CACHE_MAP)
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError('--cache-map-json must be a JSON object')
    return {str(k): str(v) for k, v in obj.items()}


def detect_conf_source(reader: CacheReader, requested: str) -> str:
    has_logprob = reader.tok_logprob is not None
    has_conf = reader.tok_conf is not None
    has_negent = reader.tok_neg_entropy is not None

    if requested == 'tok_logprob':
        if not has_logprob:
            raise ValueError('缺 token probs/logprob: tok_logprob 不存在')
        return 'tok_logprob'
    if requested == 'tok_conf':
        if not has_conf:
            raise ValueError('缺 token confidence: tok_conf 不存在')
        return 'tok_conf'
    if requested == 'tok_neg_entropy':
        if not has_negent:
            raise ValueError('缺 token entropy: tok_neg_entropy 不存在')
        return 'tok_neg_entropy'

    if has_logprob:
        return 'tok_logprob'
    if has_conf:
        return 'tok_conf'
    if has_negent:
        return 'tok_neg_entropy'

    raise ValueError('字段格式不匹配: tok_logprob/tok_conf/tok_neg_entropy 都不存在，无法构建 confidence backbone')


def build_problem_to_runid(cache_root: Path) -> Dict[str, Dict[str, int]]:
    meta_path = cache_root / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'字段格式不匹配: 缺 meta.json ({meta_path})')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    samples = meta.get('samples')
    if not isinstance(samples, list):
        raise ValueError('字段格式不匹配: meta.json 缺 samples 列表')

    out: Dict[str, Dict[str, int]] = {}
    for run_id, sample in enumerate(samples):
        if 'problem_id' not in sample:
            raise ValueError('字段格式不匹配: samples 条目缺 problem_id')
        problem_id = str(sample['problem_id'])
        sample_id = str(int(run_id))
        if problem_id not in out:
            out[problem_id] = {}
        if sample_id in out[problem_id]:
            raise ValueError(f'字段格式不匹配: problem={problem_id} sample_id={sample_id} 重复')
        out[problem_id][sample_id] = int(run_id)
    return out


def token_series_from_run(reader: CacheReader, run_id: int, source: str) -> np.ndarray:
    token_view = reader.get_token_view(int(run_id))
    if source == 'tok_logprob':
        arr = token_view.tok_logprob
        if arr is None:
            raise ValueError(f'缺 token probs/logprob: run_id={run_id} tok_logprob 为空')
        if arr.size == 0:
            raise ValueError(f'字段格式不匹配: run_id={run_id} tok_logprob 长度为0')
        series = arr.astype(np.float64)
    elif source == 'tok_conf':
        arr = token_view.tok_conf
        if arr is None:
            raise ValueError(f'缺 token confidence: run_id={run_id} tok_conf 为空')
        if arr.size == 0:
            raise ValueError(f'字段格式不匹配: run_id={run_id} tok_conf 长度为0')
        series = arr.astype(np.float64)
    elif source == 'tok_neg_entropy':
        arr = token_view.tok_neg_entropy
        if arr is None:
            raise ValueError(f'缺 token entropy: run_id={run_id} tok_neg_entropy 为空')
        if arr.size == 0:
            raise ValueError(f'字段格式不匹配: run_id={run_id} tok_neg_entropy 长度为0')
        series = arr.astype(np.float64)
    else:
        raise ValueError(f'Unsupported source: {source}')

    if not np.isfinite(series).all():
        raise ValueError(f'字段格式不匹配: run_id={run_id} token 序列存在 NaN/Inf')
    return series


def aggregate_prefix(series: np.ndarray, end: int, mode: str, tail_ratio: float, trim_worst_ratio: float) -> float:
    prefix = series[:end]
    if prefix.size <= 0:
        raise ValueError('字段格式不匹配: prefix token 长度为0')

    if mode == 'prefix_mean_logprob':
        return float(np.mean(prefix))

    if mode == 'tail_mean_logprob':
        tail_n = max(1, int(math.ceil(prefix.size * float(tail_ratio))))
        return float(np.mean(prefix[-tail_n:]))

    if mode == 'trimmed_mean_logprob':
        keep_n = max(1, int(math.ceil(prefix.size * (1.0 - float(trim_worst_ratio)))))
        sorted_vals = np.sort(prefix)
        kept = sorted_vals[-keep_n:]
        return float(np.mean(kept))

    raise ValueError(f'Unsupported aggregate mode: {mode}')


def build_raw_curve(series: np.ndarray, aggregate_mode: str, tail_ratio: float, trim_worst_ratio: float) -> List[float]:
    n = int(series.size)
    if n <= 0:
        raise ValueError('字段格式不匹配: token 序列长度为0')

    curve: List[float] = []
    for budget in BUDGETS:
        end = max(1, int(math.ceil(n * budget)))
        end = min(end, n)
        value = aggregate_prefix(
            series=series,
            end=end,
            mode=aggregate_mode,
            tail_ratio=tail_ratio,
            trim_worst_ratio=trim_worst_ratio,
        )
        curve.append(float(value))
    return curve


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    notes_output = Path(args.notes_output)
    report_output = Path(args.report_output)

    if args.quantile_high <= args.quantile_low:
        raise ValueError('--quantile-high must be > --quantile-low')
    if not (0.0 <= args.quantile_low < 1.0 and 0.0 < args.quantile_high <= 1.0):
        raise ValueError('quantiles must be in (0,1) range')

    for path in [output, notes_output, report_output]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists, set --overwrite to replace: {path}')

    reference = load_json(Path(args.reference))
    cache_map = resolve_cache_map(args.cache_map_json)

    out_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cache_reports: Dict[str, Dict[str, Any]] = {}

    total_problem_count = 0
    total_sample_count = 0

    for cache_key, ref_problem_map in reference['scores'].items():
        if cache_key not in cache_map:
            raise KeyError(f'字段格式不匹配: cache_map 缺 cache_key={cache_key}')
        cache_root = Path(cache_map[cache_key])
        if not cache_root.exists():
            raise FileNotFoundError(f'字段格式不匹配: cache_root 不存在 {cache_root}')

        reader = CacheReader(str(cache_root))
        if reader.token_row_ptr is None:
            raise ValueError(f'缺 logits/token data: {cache_key} 的 token_row_ptr 不存在')

        source = detect_conf_source(reader, args.confidence_source)
        problem_to_runid = build_problem_to_runid(cache_root)

        raw_problem_map: Dict[str, Dict[str, List[float]]] = {}
        raw_stats: List[float] = []
        cache_problem_count = 0
        cache_sample_count = 0

        for problem_id, ref_sample_map in ref_problem_map.items():
            pid = str(problem_id)
            if pid not in problem_to_runid:
                raise KeyError(f'字段格式不匹配: {cache_key} 缺 problem_id={pid} 的 run 映射')

            sample_ids = sorted([str(sid) for sid in ref_sample_map.keys()], key=sample_sort_key)
            sid_to_runid = problem_to_runid[pid]

            raw_curves: Dict[str, List[float]] = {}
            for sample_id in sample_ids:
                if sample_id not in sid_to_runid:
                    raise KeyError(f'字段格式不匹配: {cache_key}/{pid} 缺 sample_id={sample_id} 的 run 映射')
                run_id = sid_to_runid[sample_id]
                token_series = token_series_from_run(reader, run_id=run_id, source=source)
                curve_raw = build_raw_curve(
                    series=token_series,
                    aggregate_mode=args.aggregate_mode,
                    tail_ratio=args.tail_ratio,
                    trim_worst_ratio=args.trim_worst_ratio,
                )
                raw_curves[sample_id] = curve_raw
                raw_stats.extend(curve_raw)

            raw_problem_map[pid] = raw_curves
            cache_problem_count += 1
            cache_sample_count += len(sample_ids)

        # normalization stage
        out_problem_map: Dict[str, Dict[str, List[float]]] = {}

        if args.normalize_mode == 'none':
            out_problem_map = json.loads(json.dumps(raw_problem_map))

        elif args.normalize_mode == 'rank_problem_per_budget':
            for pid, sample_map in raw_problem_map.items():
                sample_ids = sorted(sample_map.keys(), key=sample_sort_key)
                normalized: Dict[str, List[float]] = {sid: [0.0] * 10 for sid in sample_ids}
                for budget_idx in range(10):
                    values = {sid: sample_map[sid][budget_idx] for sid in sample_ids}
                    rank_vals = rank_transform(values)
                    for sid in sample_ids:
                        normalized[sid][budget_idx] = float(rank_vals[sid])
                out_problem_map[pid] = normalized

        elif args.normalize_mode == 'cache_budget_quantile':
            q_stats: Dict[int, Tuple[float, float]] = {}
            for budget_idx in range(10):
                values: List[float] = []
                for sample_map in raw_problem_map.values():
                    for curve in sample_map.values():
                        values.append(float(curve[budget_idx]))
                if not values:
                    raise ValueError(f'字段格式不匹配: {cache_key} budget_idx={budget_idx} 无 raw 分数')
                arr = np.array(values, dtype=np.float64)
                q_low = float(np.quantile(arr, args.quantile_low))
                q_high = float(np.quantile(arr, args.quantile_high))
                q_stats[budget_idx] = (q_low, q_high)

            for pid, sample_map in raw_problem_map.items():
                normalized: Dict[str, List[float]] = {}
                for sid, curve in sample_map.items():
                    new_curve: List[float] = []
                    for budget_idx, x in enumerate(curve):
                        q_low, q_high = q_stats[budget_idx]
                        if q_high <= q_low + 1e-12:
                            y = 0.5
                        else:
                            y = clip01((float(x) - q_low) / (q_high - q_low))
                        new_curve.append(float(y))
                    normalized[sid] = new_curve
                out_problem_map[pid] = normalized

        else:
            raise ValueError(f'Unsupported normalize mode: {args.normalize_mode}')

        if args.round_digits >= 0:
            for pid, sample_map in out_problem_map.items():
                for sid, curve in sample_map.items():
                    sample_map[sid] = [round(float(v), int(args.round_digits)) for v in curve]

        out_scores[cache_key] = out_problem_map
        total_problem_count += cache_problem_count
        total_sample_count += cache_sample_count

        raw_min = float(min(raw_stats)) if raw_stats else None
        raw_max = float(max(raw_stats)) if raw_stats else None
        raw_mean = float(np.mean(raw_stats)) if raw_stats else None

        cache_reports[cache_key] = {
            'cache_root': str(cache_root),
            'confidence_source_used': source,
            'aggregate_mode': args.aggregate_mode,
            'normalize_mode': args.normalize_mode,
            'problem_count': cache_problem_count,
            'sample_count': cache_sample_count,
            'raw_curve_min': raw_min,
            'raw_curve_max': raw_max,
            'raw_curve_mean': raw_mean,
        }

    out_obj = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'scores': out_scores,
    }

    notes = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'settings': {
            'reference': str(args.reference),
            'confidence_source_requested': args.confidence_source,
            'confidence_source_priority': ['tok_logprob', 'tok_conf', 'tok_neg_entropy'],
            'confidence_definition': 'tok_logprob: mean logprob over budget prefix (no exp transform)',
            'aggregate_mode': args.aggregate_mode,
            'tail_ratio': args.tail_ratio,
            'trim_worst_ratio': args.trim_worst_ratio,
            'normalize_mode': args.normalize_mode,
            'quantile_low': args.quantile_low,
            'quantile_high': args.quantile_high,
            'budgets': BUDGETS,
            'round_digits': args.round_digits,
            'cache_map_json': args.cache_map_json,
        },
        'stats': {
            'cache_count': len(out_scores),
            'problem_count': total_problem_count,
            'sample_count': total_sample_count,
        },
        'per_cache': cache_reports,
    }

    report = {
        'task': 'early_stop',
        'method_name': args.method_name,
        'summary': notes['stats'],
        'per_cache': cache_reports,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    notes_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
    notes_output.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'wrote {output}')
    print(f'wrote {notes_output}')
    print(f'wrote {report_output}')
    print(f"cache_count={len(out_scores)} problem_count={total_problem_count} sample_count={total_sample_count}")


if __name__ == '__main__':
    main()
