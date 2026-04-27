#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.views.reader import CacheReader
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json'
DEFAULT_OUTPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v5_cluster_router_submit.json'
DEFAULT_NOTES = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v5_cluster_router_submit_notes.json'

AIME_DATASETS = {'aime24', 'aime25'}
HMMT_DATASETS = {'hmmt25'}
MC_DATASETS = {'gpqa', 'brumo25'}


@dataclass
class ClusterStats:
    base_max: float
    base_mass: float
    aux_mean: float
    count: int
    top_sid: str
    top_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build mixed_v5 by applying answer-cluster-level local reranking on top of baseline mixed_v2 scores.'
    )
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--notes-output', default=DEFAULT_NOTES)
    parser.add_argument('--method-name', default='nad_mixed_v5_cluster_router')
    parser.add_argument('--score-bump', type=float, default=1e-9)

    parser.add_argument(
        '--target-cache-keys',
        default='DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25',
        help='Comma-separated cache keys to consider. Regime-gating still applies after this filter.',
    )
    parser.add_argument(
        '--enable-qwen-compressed',
        action='store_true',
        help='Also enable Qwen(gpqa/brumo25/hmmt25) compressed-score branches.',
    )

    parser.add_argument('--metric', choices=['tok_selfcert', 'tok_conf', 'tok_neg_entropy', 'tok_logprob', 'tok_gini'], default='tok_logprob')
    parser.add_argument('--reduction', choices=['mean', 'min_group'], default='mean')
    parser.add_argument('--group-size', type=int, default=20)
    parser.add_argument('--cluster-topr', type=int, default=3, help='Top-r members inside each cluster for base_mass/aux_mean.')

    parser.add_argument('--tau-max-gap', type=float, default=0.001, help='Trigger only when cluster top1-top2 base_max gap <= tau.')
    parser.add_argument('--tau-mass-gap', type=float, default=0.002, help='Trigger only when cluster top1-top2 base_mass gap <= tau.')
    parser.add_argument('--eps-mass', type=float, default=1e-6)
    parser.add_argument('--eps-aux', type=float, default=1e-7)
    parser.add_argument('--eps-count', type=float, default=1e-8)
    parser.add_argument('--disable-aux-rerank', action='store_true', help='Disable aux_mean rank in epsilon rerank.')
    parser.add_argument('--disable-count-rerank', action='store_true', help='Disable count rank in epsilon rerank.')

    parser.add_argument(
        '--mod1000-datasets',
        default='aime24,aime25',
        help='Datasets whose numeric canonical answer is reduced modulo 1000.',
    )
    parser.add_argument('--drop-other-sids', action='store_true', help='Output only selected sid per problem.')
    parser.add_argument('--keep-problem-details', action='store_true', help='Store per-problem details in notes (larger file).')
    return parser.parse_args()


def least_grouped_strict(x: np.ndarray, w: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float('-inf')
    if w is None or w <= 1 or n < w:
        return float(np.mean(x))
    c = np.cumsum(x)
    sums = c[w - 1:] - np.concatenate(([0.0], c[:-w]))
    means = sums / float(w)
    return float(np.min(means))


def metric_quality(arr: np.ndarray, metric: str, reduction: str, group_size: int) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('-inf')
    if reduction == 'mean':
        base = float(np.mean(arr))
    else:
        base = least_grouped_strict(arr, group_size)
    if metric == 'tok_conf':
        return -base
    return base


def get_metric_array(token_view, metric: str) -> np.ndarray:
    arr = getattr(token_view, metric)
    if arr is None:
        raise ValueError(f'metric {metric} not available in cache')
    return arr


def parse_dataset(cache_key: str) -> str:
    if '/' not in cache_key:
        return cache_key
    return cache_key.split('/', 1)[1]


def is_qwen(cache_key: str) -> bool:
    return cache_key.startswith('Qwen')


def allow_router_for_cache_key(cache_key: str, enable_qwen_compressed: bool) -> bool:
    dataset = parse_dataset(cache_key)
    if dataset == 'lcb_v5':
        return False
    if dataset in AIME_DATASETS:
        return True
    if enable_qwen_compressed and is_qwen(cache_key) and dataset in {'gpqa', 'brumo25', 'hmmt25'}:
        return True
    return False


def _unwrap_literal_answer(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return ''
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped
    if isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0])
    if parsed is None:
        return ''
    return str(parsed)


def _strip_boxed(text: str) -> str:
    s = text
    while True:
        m = re.search(r'\\boxed\s*\{([^{}]*)\}', s)
        if not m:
            break
        s = (s[:m.start()] + m.group(1) + s[m.end():]).strip()
    return s


def canonical_multiple_choice(text: str) -> str:
    x = _unwrap_literal_answer(text).strip().upper()
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


def canonical_numeric(text: str, apply_mod_1000: bool) -> str:
    x = _unwrap_literal_answer(text).strip()
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
            intval = frac.numerator
            if apply_mod_1000:
                intval %= 1000
            return str(intval)
        return f'{frac.numerator}/{frac.denominator}'

    dec_out = _canonical_decimal(x)
    if dec_out is not None:
        if re.fullmatch(r'[+-]?\d+', dec_out):
            intval = int(dec_out)
            if apply_mod_1000:
                intval %= 1000
            return str(intval)
        return dec_out

    return x


def normalize_answer_for_dataset(raw_answer: str, dataset: str, mod1000_datasets: set[str]) -> str:
    if dataset in (AIME_DATASETS | HMMT_DATASETS):
        return f'NUM::{canonical_numeric(raw_answer, apply_mod_1000=(dataset in mod1000_datasets))}'
    if dataset in MC_DATASETS:
        return f'MC::{canonical_multiple_choice(raw_answer)}'
    return f'TXT::{_unwrap_literal_answer(raw_answer).strip()}'


def load_problem_run_ids(cache_root: str) -> Dict[str, set[str]]:
    meta = json.loads((Path(cache_root) / 'meta.json').read_text())
    grouped: Dict[str, set[str]] = defaultdict(set)
    for sample_idx, sample in enumerate(meta['samples']):
        grouped[str(sample['problem_id'])].add(str(sample_idx))
    return grouped


def load_sid_answer_map(cache_root: str) -> Dict[str, Dict[str, str]]:
    root = Path(cache_root)
    meta = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
    samples = meta.get('samples', [])

    report = None
    for name in ('evaluation_report_compact.json', 'evaluation_report.json'):
        fp = root / name
        if fp.exists():
            report = json.loads(fp.read_text(encoding='utf-8'))
            break
    if report is None or 'results' not in report:
        raise FileNotFoundError(f'No evaluation report with results found under {cache_root}')

    run_answer_map: Dict[Tuple[str, int], str] = {}
    for result in report['results']:
        pid = str(result.get('problem_id'))
        for run in result.get('runs', []):
            run_index = int(run.get('run_index', run.get('index', 0)))
            answer = str(run.get('extracted_answer', ''))
            run_answer_map[(pid, run_index)] = answer

    sid_answer_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    for sid, sample in enumerate(samples):
        pid = str(sample.get('problem_id'))
        run_index = int(sample.get('run_index', 0))
        answer = run_answer_map.get((pid, run_index), '')
        sid_answer_map[pid][str(sid)] = answer
    return sid_answer_map


def descending_rank(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return {key: idx + 1 for idx, (key, _) in enumerate(ordered)}


def build_cluster_stats(
    sid_scores: Dict[str, float],
    sid_answers: Dict[str, str],
    sid_aux: Dict[str, float],
    dataset: str,
    mod1000_datasets: set[str],
    cluster_topr: int,
) -> Tuple[Dict[str, ClusterStats], Dict[str, List[str]]]:
    clusters: Dict[str, List[str]] = defaultdict(list)
    for sid in sid_scores:
        raw_answer = sid_answers.get(sid, '')
        answer_key = normalize_answer_for_dataset(raw_answer, dataset=dataset, mod1000_datasets=mod1000_datasets)
        clusters[answer_key].append(sid)

    stats: Dict[str, ClusterStats] = {}
    for answer_key, members in clusters.items():
        ranked_members = sorted(members, key=lambda sid: (-sid_scores[sid], sid))
        top_sid = ranked_members[0]
        top_score = float(sid_scores[top_sid])
        top_r_members = ranked_members[:max(1, min(cluster_topr, len(ranked_members)))]
        base_mass = float(sum(sid_scores[sid] for sid in top_r_members))
        aux_values = [float(sid_aux[sid]) for sid in top_r_members if np.isfinite(sid_aux.get(sid, float('nan')))]
        aux_mean = float(np.mean(aux_values)) if aux_values else float('-inf')
        stats[answer_key] = ClusterStats(
            base_max=top_score,
            base_mass=base_mass,
            aux_mean=aux_mean,
            count=len(members),
            top_sid=top_sid,
            top_score=top_score,
        )
    return stats, clusters


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.input).read_text())

    target_cache_keys = [x.strip() for x in args.target_cache_keys.split(',') if x.strip()]
    mod1000_datasets = {x.strip() for x in args.mod1000_datasets.split(',') if x.strip()}

    out_scores = {}
    notes = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'input': args.input,
        'output': args.output,
        'target_cache_keys': target_cache_keys,
        'enable_qwen_compressed': bool(args.enable_qwen_compressed),
        'metric': args.metric,
        'reduction': args.reduction,
        'group_size': args.group_size,
        'cluster_topr': args.cluster_topr,
        'tau_max_gap': args.tau_max_gap,
        'tau_mass_gap': args.tau_mass_gap,
        'eps_mass': args.eps_mass,
        'eps_aux': args.eps_aux,
        'eps_count': args.eps_count,
        'disable_aux_rerank': bool(args.disable_aux_rerank),
        'disable_count_rerank': bool(args.disable_count_rerank),
        'mod1000_datasets': sorted(mod1000_datasets),
        'submission_safe_full_scores': not args.drop_other_sids,
        'keep_problem_details': bool(args.keep_problem_details),
        'cache_keys': {},
    }

    target_cache_keys = [ck for ck in target_cache_keys if ck in base.get('scores', {})]
    problem_run_ids_by_cache = {
        cache_key: load_problem_run_ids(DEFAULT_CACHE_MAP[cache_key])
        for cache_key in target_cache_keys
        if cache_key in DEFAULT_CACHE_MAP
    }
    sid_answers_by_cache = {
        cache_key: load_sid_answer_map(DEFAULT_CACHE_MAP[cache_key])
        for cache_key in target_cache_keys
        if cache_key in DEFAULT_CACHE_MAP
    }
    readers = {
        cache_key: CacheReader(DEFAULT_CACHE_MAP[cache_key])
        for cache_key in target_cache_keys
        if cache_key in DEFAULT_CACHE_MAP
    }

    for cache_key, problem_map in base['scores'].items():
        cache_out = {}
        changed = 0
        skipped_missing = 0
        skipped_regime = 0
        skipped_single_cluster = 0
        triggered_count = 0
        changed_after_trigger = 0

        cache_notes = {
            'problem_count': len(problem_map),
            'changed_count': 0,
            'triggered_count': 0,
            'changed_after_trigger': 0,
            'skipped_missing_count': 0,
            'skipped_regime_count': 0,
            'single_cluster_count': 0,
            'cache_key_enabled': False,
            'details': {} if args.keep_problem_details else None,
        }

        enabled_by_target = cache_key in target_cache_keys
        enabled_by_regime = allow_router_for_cache_key(cache_key, enable_qwen_compressed=args.enable_qwen_compressed)
        cache_enabled = enabled_by_target and enabled_by_regime and (cache_key in readers) and (cache_key in sid_answers_by_cache)
        cache_notes['cache_key_enabled'] = bool(cache_enabled)
        cache_notes['cache_key_enable_reason'] = {
            'in_target': bool(enabled_by_target),
            'regime_passed': bool(enabled_by_regime),
            'reader_available': bool(cache_key in readers),
            'answer_map_available': bool(cache_key in sid_answers_by_cache),
        }

        reader = readers.get(cache_key)
        valid_problem_runs = problem_run_ids_by_cache.get(cache_key)
        sid_answer_map_all = sid_answers_by_cache.get(cache_key, {})
        dataset = parse_dataset(cache_key)
        sid_aux_cache: Dict[str, float] = {}

        for problem_id, sid_scores in problem_map.items():
            sid_scores = {str(sid): float(score) for sid, score in sid_scores.items()}
            ranked = sorted(sid_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            original_sid = ranked[0][0]

            if not cache_enabled:
                cache_out[problem_id] = sid_scores
                if enabled_by_target and not enabled_by_regime:
                    skipped_regime += 1
                continue

            valid_sids = valid_problem_runs.get(str(problem_id), set()) if valid_problem_runs else set()
            missing_sids = [sid for sid in sid_scores if sid not in valid_sids]
            if missing_sids:
                cache_out[problem_id] = sid_scores
                skipped_missing += 1
                if args.keep_problem_details:
                    cache_notes['details'][problem_id] = {
                        'reason': 'sid_mismatch_with_cache_meta',
                        'missing_sids_example': missing_sids[:5],
                    }
                continue

            sid_answers_for_problem = sid_answer_map_all.get(str(problem_id), {})
            if not sid_answers_for_problem:
                cache_out[problem_id] = sid_scores
                skipped_missing += 1
                if args.keep_problem_details:
                    cache_notes['details'][problem_id] = {'reason': 'missing_problem_answers'}
                continue

            for sid in sid_scores:
                if sid in sid_aux_cache:
                    continue
                token_view = reader.get_token_view(int(sid))
                metric_arr = get_metric_array(token_view, args.metric)
                sid_aux_cache[sid] = metric_quality(metric_arr, args.metric, args.reduction, args.group_size)

            stats, _clusters = build_cluster_stats(
                sid_scores=sid_scores,
                sid_answers=sid_answers_for_problem,
                sid_aux=sid_aux_cache,
                dataset=dataset,
                mod1000_datasets=mod1000_datasets,
                cluster_topr=args.cluster_topr,
            )
            cluster_rank = sorted(stats.items(), key=lambda kv: (-kv[1].base_max, kv[0]))
            if len(cluster_rank) <= 1:
                cache_out[problem_id] = sid_scores
                skipped_single_cluster += 1
                if args.keep_problem_details:
                    cache_notes['details'][problem_id] = {
                        'reason': 'single_cluster',
                        'cluster_count': len(cluster_rank),
                    }
                continue

            c1_key, c1 = cluster_rank[0]
            c2_key, c2 = cluster_rank[1]
            gap_max = float(c1.base_max - c2.base_max)
            gap_mass = float(c1.base_mass - c2.base_mass)
            triggered = (gap_max <= args.tau_max_gap) and (gap_mass <= args.tau_mass_gap)
            if triggered:
                triggered_count += 1

            selected_cluster_key = c1_key
            if triggered:
                mass_rank = descending_rank({k: st.base_mass for k, st in stats.items()})
                aux_rank = descending_rank({k: st.aux_mean for k, st in stats.items()})
                count_rank = descending_rank({k: float(st.count) for k, st in stats.items()})
                score_map = {}
                for k, st in stats.items():
                    score = float(st.base_max) - args.eps_mass * float(mass_rank[k])
                    if not args.disable_aux_rerank:
                        score -= args.eps_aux * float(aux_rank[k])
                    if not args.disable_count_rerank:
                        score -= args.eps_count * float(count_rank[k])
                    score_map[k] = score
                selected_cluster_key = max(score_map.items(), key=lambda kv: (kv[1], kv[0]))[0]
            selected_sid = stats[selected_cluster_key].top_sid

            if selected_sid != original_sid:
                changed += 1
                if triggered:
                    changed_after_trigger += 1

            if args.drop_other_sids:
                cache_out[problem_id] = {selected_sid: sid_scores[selected_sid]}
            else:
                adjusted = dict(sid_scores)
                adjusted[selected_sid] = max(adjusted.values()) + float(args.score_bump)
                cache_out[problem_id] = adjusted

            if args.keep_problem_details:
                cache_notes['details'][problem_id] = {
                    'reason': 'cluster_router',
                    'triggered': bool(triggered),
                    'original_sid': original_sid,
                    'new_sid': selected_sid,
                    'changed': bool(selected_sid != original_sid),
                    'cluster_count': len(stats),
                    'top2_cluster_keys': [c1_key, c2_key],
                    'top2_cluster_gap_max': gap_max,
                    'top2_cluster_gap_mass': gap_mass,
                    'top1_cluster_count': int(c1.count),
                    'top2_cluster_count': int(c2.count),
                    'top1_cluster_aux': float(c1.aux_mean),
                    'top2_cluster_aux': float(c2.aux_mean),
                }

        out_scores[cache_key] = cache_out
        cache_notes['changed_count'] = changed
        cache_notes['triggered_count'] = triggered_count
        cache_notes['changed_after_trigger'] = changed_after_trigger
        cache_notes['skipped_missing_count'] = skipped_missing
        cache_notes['skipped_regime_count'] = skipped_regime
        cache_notes['single_cluster_count'] = skipped_single_cluster
        notes['cache_keys'][cache_key] = cache_notes
        print(
            f'finished {cache_key}: changed {changed}/{len(problem_map)}, '
            f'triggered {triggered_count}, single_cluster {skipped_single_cluster}'
        )

    out = {
        'task': base.get('task', 'best_of_n'),
        'method_name': args.method_name,
        'scores': out_scores,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.notes_output).write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    print(f'wrote {args.output}')
    print(f'wrote {args.notes_output}')


if __name__ == '__main__':
    main()
