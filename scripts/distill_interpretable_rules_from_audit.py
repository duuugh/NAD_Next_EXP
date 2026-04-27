#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/home/jovyan/work/NAD_Next')

from nad.core.selectors.base import SelectorContext
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from plugins.medoid_tail_warning import MedoidTailWarningSelector
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP


DEFAULT_INPUT = '/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json'
DEFAULT_AUDIT_SUMMARY = '/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_audit_summary.json'
DEFAULT_OUTPUT_CSV = '/home/jovyan/work/NAD_Next/result/rule_distillation_candidates.csv'
DEFAULT_OUTPUT_SUMMARY = '/home/jovyan/work/NAD_Next/result/rule_distillation_summary.json'

PRIMARY_AIME_CACHES = [
    'DS-R1/aime24',
    'DS-R1/aime25',
    'Qwen3-4B/aime24',
    'Qwen3-4B/aime25',
]
SECONDARY_DS_CHECK_CACHES = [
    'DS-R1/gpqa',
    'DS-R1/hmmt25',
]


@dataclass
class PairRecord:
    cache_key: str
    problem_id: str
    top1_sid: str
    top2_sid: str
    top1_score: float
    top2_score: float
    gap: float
    len1: float
    len2: float
    parse1: float
    parse2: float
    int1: float
    int2: float
    unique_answer_count: int
    logprob_delta: float
    selfcert_delta: float
    conf_delta: float
    neg_entropy_delta: float
    gini_delta: float
    tail_warning1: int
    tail_warning2: int
    tail_delta: float
    plateau_delta: float


@dataclass
class RuleSpec:
    name: str
    family: str
    description: str
    max_gap: float
    max_flips_total: int
    max_flips_per_cache: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Distill small interpretable rule candidates from feature-audit insights.')
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--audit-summary', default=DEFAULT_AUDIT_SUMMARY)
    p.add_argument('--output-csv', default=DEFAULT_OUTPUT_CSV)
    p.add_argument('--output-summary', default=DEFAULT_OUTPUT_SUMMARY)
    p.add_argument('--score-bump', type=float, default=1e-9)
    p.add_argument('--write-submissions', action='store_true', default=True)
    return p.parse_args()


def ranked_items(sid_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(((str(s), float(v)) for s, v in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))


def safe_mean(arr: object) -> float:
    if arr is None:
        return float('nan')
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    return float(np.mean(x))


def parse_extracted_answer(raw: object) -> Tuple[str, bool]:
    if raw is None:
        return '', False
    text = str(raw).strip()
    if not text:
        return '', False
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return '', False
            ans = str(parsed[0]).strip()
            return ans, bool(ans)
        ans = str(parsed).strip()
        return ans, bool(ans)
    except Exception:
        return text, True


def canonical_answer(dataset: str, extracted_answer: object) -> str:
    ans, ok = parse_extracted_answer(extracted_answer)
    if not ok:
        return ''
    ds = dataset.lower()
    if ds in {'aime24', 'aime25', 'hmmt25'}:
        cleaned = re.sub(r'[^0-9+\-]', '', ans)
        if re.fullmatch(r'[+-]?\d+', cleaned or ''):
            return str(int(cleaned))
        return ans.strip()
    if ds in {'gpqa', 'brumo25'}:
        up = ans.strip().upper()
        m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', up)
        return m.group(1) if m else up
    return ans.strip()


def load_eval_info(cache_root: str) -> Dict[int, Dict[str, object]]:
    root = Path(cache_root)
    meta_path = root / 'meta.json'
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    sample_index: Dict[Tuple[str, int], int] = {}
    for sid, sample in enumerate(meta.get('samples', [])):
        sample_index[(str(sample.get('problem_id')), int(sample.get('run_index', 0)))] = int(sid)

    report = None
    for name in ('evaluation_report_compact.json', 'evaluation_report.json'):
        fp = root / name
        if fp.exists():
            report = json.loads(fp.read_text(encoding='utf-8'))
            break
    if report is None or 'results' not in report:
        return {}

    info: Dict[int, Dict[str, object]] = {}
    for result in report.get('results', []):
        pid = str(result.get('problem_id'))
        dataset = str(result.get('dataset', '')).lower()
        for run in result.get('runs', []):
            ri = int(run.get('run_index', run.get('index', 0)))
            sid = sample_index.get((pid, ri))
            if sid is None:
                continue
            extracted = run.get('extracted_answer', '')
            ans_text, parse_ok = parse_extracted_answer(extracted)
            out_toks = run.get('output_tokens', None)
            try:
                out_toks = float(out_toks) if out_toks is not None else float('nan')
            except Exception:
                out_toks = float('nan')
            info[sid] = {
                'dataset': dataset,
                'extracted_answer': '' if extracted is None else str(extracted),
                'answer_norm': canonical_answer(dataset, extracted),
                'parse_success': float(1.0 if parse_ok else 0.0),
                'is_integer_answer': float(1.0 if re.fullmatch(r'[+-]?\d+', ans_text or '') else 0.0),
                'output_tokens': out_toks,
            }
    return info


def token_length(reader: CacheReader, sid: int) -> float:
    tv = reader.get_token_view(int(sid))
    for field in ('token_ids', 'tok_logprob', 'tok_selfcert', 'tok_conf', 'tok_neg_entropy', 'tok_gini'):
        arr = getattr(tv, field, None)
        if arr is not None:
            return float(len(arr))
    return float('nan')


def sid_metrics(reader: CacheReader, sid: int) -> Dict[str, float]:
    tv = reader.get_token_view(int(sid))
    return {
        'tok_logprob_mean': safe_mean(getattr(tv, 'tok_logprob', None)),
        'tok_selfcert_mean': safe_mean(getattr(tv, 'tok_selfcert', None)),
        'tok_conf_mean': safe_mean(getattr(tv, 'tok_conf', None)),
        'tok_neg_entropy_mean': safe_mean(getattr(tv, 'tok_neg_entropy', None)),
        'tok_gini_mean': safe_mean(getattr(tv, 'tok_gini', None)),
    }


def build_pair_records(input_scores: Dict[str, Dict[str, Dict[str, float]]], target_caches: List[str]) -> Tuple[Dict[str, Dict[str, PairRecord]], Dict[str, Optional[Dict[int, bool]]]]:
    pair_by_cache: Dict[str, Dict[str, PairRecord]] = {}
    correctness_maps: Dict[str, Optional[Dict[int, bool]]] = {}

    for ck in target_caches:
        if ck not in input_scores or ck not in DEFAULT_CACHE_MAP:
            continue
        cache_root = DEFAULT_CACHE_MAP[ck]
        reader = CacheReader(cache_root)
        eval_info = load_eval_info(cache_root)
        try:
            correctness_maps[ck] = load_correctness_map(cache_root)
        except Exception:
            correctness_maps[ck] = None

        plugin = MedoidTailWarningSelector(
            gap_abs=0.01,
            tail_start=0.85,
            plateau_fraction=0.98,
            tail_new_ratio_warn=0.015,
            plateau_progress_warn=0.85,
            min_tail_ratio_advantage=0.005,
            min_plateau_progress_advantage=0.03,
        )
        plugin.bind(SelectorContext(cache=reader, problem_id='__all__', run_ids=[], views=[]))

        per_problem: Dict[str, PairRecord] = {}
        for pid, sid_scores in input_scores[ck].items():
            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue
            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            gap = float(top1_score - top2_score)

            # unique answers in problem
            dataset = ck.split('/')[-1].lower()
            answers = []
            for sid, _ in ranked:
                rec = eval_info.get(int(sid), {})
                ansn = str(rec.get('answer_norm', '')).strip()
                if ansn:
                    answers.append(ansn)
            unique_answer_count = len(set(answers)) if answers else 0

            m1 = sid_metrics(reader, int(top1_sid))
            m2 = sid_metrics(reader, int(top2_sid))

            e1 = eval_info.get(int(top1_sid), {})
            e2 = eval_info.get(int(top2_sid), {})

            len1 = float(e1.get('output_tokens', float('nan')))
            len2 = float(e2.get('output_tokens', float('nan')))
            if not np.isfinite(len1):
                len1 = token_length(reader, int(top1_sid))
            if not np.isfinite(len2):
                len2 = token_length(reader, int(top2_sid))

            t1 = plugin._tail_metrics(int(top1_sid)) or {}
            t2 = plugin._tail_metrics(int(top2_sid)) or {}
            tail_warning1 = int(plugin._is_warned(t1)) if t1 else 0
            tail_warning2 = int(plugin._is_warned(t2)) if t2 else 0

            per_problem[str(pid)] = PairRecord(
                cache_key=ck,
                problem_id=str(pid),
                top1_sid=str(top1_sid),
                top2_sid=str(top2_sid),
                top1_score=float(top1_score),
                top2_score=float(top2_score),
                gap=gap,
                len1=float(len1),
                len2=float(len2),
                parse1=float(e1.get('parse_success', 0.0)),
                parse2=float(e2.get('parse_success', 0.0)),
                int1=float(e1.get('is_integer_answer', 0.0)),
                int2=float(e2.get('is_integer_answer', 0.0)),
                unique_answer_count=int(unique_answer_count),
                logprob_delta=float(m2['tok_logprob_mean'] - m1['tok_logprob_mean']),
                selfcert_delta=float(m2['tok_selfcert_mean'] - m1['tok_selfcert_mean']),
                conf_delta=float(m2['tok_conf_mean'] - m1['tok_conf_mean']),
                neg_entropy_delta=float(m2['tok_neg_entropy_mean'] - m1['tok_neg_entropy_mean']),
                gini_delta=float(m2['tok_gini_mean'] - m1['tok_gini_mean']),
                tail_warning1=int(tail_warning1),
                tail_warning2=int(tail_warning2),
                tail_delta=float(t2.get('tail_new_ratio', float('nan')) - t1.get('tail_new_ratio', float('nan'))),
                plateau_delta=float(t2.get('plateau_progress', float('nan')) - t1.get('plateau_progress', float('nan'))),
            )

        pair_by_cache[ck] = per_problem

    return pair_by_cache, correctness_maps


def token_votes_conservative(r: PairRecord) -> int:
    v = 0
    v += int(r.conf_delta <= -0.12)
    v += int(r.selfcert_delta <= -0.04)
    v += int(r.gini_delta <= -0.02)
    v += int(r.neg_entropy_delta <= -0.04)
    return v


def token_votes_confident(r: PairRecord) -> int:
    v = 0
    v += int(r.conf_delta >= 0.12)
    v += int(r.selfcert_delta >= 0.04)
    v += int(r.gini_delta >= 0.02)
    v += int(r.neg_entropy_delta >= 0.04)
    return v


def structure_support(r: PairRecord) -> bool:
    len_ratio = (r.len2 / max(r.len1, 1.0)) if np.isfinite(r.len1) and np.isfinite(r.len2) else float('nan')
    shorter = np.isfinite(len_ratio) and (len_ratio <= 0.92)
    parse_better = (r.parse2 >= r.parse1)
    int_better = (r.int2 >= r.int1)
    diverse = (r.unique_answer_count >= 6)
    return bool(shorter and parse_better and int_better and diverse)


def should_flip(rule: RuleSpec, r: PairRecord) -> bool:
    if r.gap > rule.max_gap:
        return False

    if rule.name == 'token_only_conservative_v1':
        return token_votes_conservative(r) >= 3

    if rule.name == 'token_only_confident_v1':
        return token_votes_confident(r) >= 3

    if rule.name == 'structure_only_commitment_v1':
        len_ratio = (r.len2 / max(r.len1, 1.0)) if np.isfinite(r.len1) and np.isfinite(r.len2) else float('nan')
        return bool(np.isfinite(len_ratio) and len_ratio <= 0.88 and r.parse2 >= r.parse1 and r.int2 >= r.int1 and r.unique_answer_count >= 6)

    if rule.name == 'token_structure_main_v1':
        return bool(token_votes_conservative(r) >= 3 and structure_support(r))

    if rule.name == 'token_structure_with_tiny_activation_veto_v1':
        if not (token_votes_conservative(r) >= 3 and structure_support(r)):
            return False
        return bool((r.tail_warning1 == 1 and r.tail_warning2 == 0) or (np.isfinite(r.tail_delta) and r.tail_delta >= 0.02 and np.isfinite(r.plateau_delta) and r.plateau_delta >= 0.05))

    return False


def rule_priority(r: PairRecord) -> float:
    # smaller gap first, then stronger token vote
    return float(-r.gap + 1e-4 * (token_votes_conservative(r) - token_votes_confident(r)))


def apply_rule_and_collect(
    base_scores: Dict[str, Dict[str, Dict[str, float]]],
    pair_by_cache: Dict[str, Dict[str, PairRecord]],
    correctness_maps: Dict[str, Optional[Dict[int, bool]]],
    rule: RuleSpec,
    apply_caches: List[str],
    eval_caches: List[str],
    score_bump: float,
) -> Dict[str, object]:
    out_scores = copy.deepcopy(base_scores)

    candidates: List[PairRecord] = []
    for ck in apply_caches:
        for rec in pair_by_cache.get(ck, {}).values():
            if should_flip(rule, rec):
                candidates.append(rec)

    candidates.sort(key=lambda r: (-rule_priority(r), r.cache_key, r.problem_id))

    applied: List[PairRecord] = []
    applied_by_cache: Dict[str, int] = {}
    seen_problem: set[Tuple[str, str]] = set()
    for rec in candidates:
        if len(applied) >= rule.max_flips_total:
            break
        if applied_by_cache.get(rec.cache_key, 0) >= rule.max_flips_per_cache:
            continue
        key = (rec.cache_key, rec.problem_id)
        if key in seen_problem:
            continue

        sid_scores = {str(s): float(v) for s, v in out_scores[rec.cache_key][rec.problem_id].items()}
        sid_scores[rec.top2_sid] = max(sid_scores.values()) + float(score_bump)
        out_scores[rec.cache_key][rec.problem_id] = sid_scores

        seen_problem.add(key)
        applied_by_cache[rec.cache_key] = applied_by_cache.get(rec.cache_key, 0) + 1
        applied.append(rec)

    # evaluation
    def cache_metrics(ck: str) -> Dict[str, object]:
        corr = correctness_maps.get(ck)
        pmap = pair_by_cache.get(ck, {})

        considered = sum(int(rec.gap <= rule.max_gap) for rec in pmap.values())
        candidate_flips = sum(int(should_flip(rule, rec)) for rec in pmap.values())
        applied_here = [r for r in applied if r.cache_key == ck]

        out = {
            'considered': int(considered),
            'candidate_flips': int(candidate_flips),
            'applied_flips': int(len(applied_here)),
            'labeled_problem_count': 0,
            'top1_before_accuracy': None,
            'top1_after_accuracy': None,
            'delta_accuracy': None,
            'correct_flips': None,
            'wrong_flips': None,
            'net_gain_count': None,
        }

        if corr is None:
            return out

        before_correct = 0
        after_correct = 0
        n_prob = 0
        applied_keys = {(r.cache_key, r.problem_id): r for r in applied_here}
        correct_flips = 0
        wrong_flips = 0

        for pid, rec in pmap.items():
            n_prob += 1
            sid_before = int(rec.top1_sid)
            y_before = bool(corr.get(sid_before, False))
            before_correct += int(y_before)

            if (ck, pid) in applied_keys:
                sid_after = int(rec.top2_sid)
            else:
                sid_after = sid_before
            y_after = bool(corr.get(sid_after, False))
            after_correct += int(y_after)

            if (ck, pid) in applied_keys:
                if (not y_before) and y_after:
                    correct_flips += 1
                elif y_before and (not y_after):
                    wrong_flips += 1

        if n_prob > 0:
            out['labeled_problem_count'] = int(n_prob)
            out['top1_before_accuracy'] = float(before_correct / n_prob)
            out['top1_after_accuracy'] = float(after_correct / n_prob)
            out['delta_accuracy'] = float((after_correct - before_correct) / n_prob)
            out['correct_flips'] = int(correct_flips)
            out['wrong_flips'] = int(wrong_flips)
            out['net_gain_count'] = int(correct_flips - wrong_flips)

        return out

    per_cache = {ck: cache_metrics(ck) for ck in eval_caches}

    def aggregate(caches: List[str]) -> Dict[str, object]:
        labeled = [per_cache[c] for c in caches if per_cache[c]['top1_before_accuracy'] is not None]
        if not labeled:
            return {
                'cache_count': len(caches),
                'labeled_cache_count': 0,
                'applied_flips': int(sum(per_cache[c]['applied_flips'] for c in caches if c in per_cache)),
                'top1_before_accuracy': None,
                'top1_after_accuracy': None,
                'delta_accuracy': None,
                'correct_flips': None,
                'wrong_flips': None,
                'net_gain_count': None,
            }
        nprob = int(sum(int(x['labeled_problem_count']) for x in labeled))
        before_cnt = sum(float(x['top1_before_accuracy']) * int(x['labeled_problem_count']) for x in labeled)
        after_cnt = sum(float(x['top1_after_accuracy']) * int(x['labeled_problem_count']) for x in labeled)
        correct_flips = int(sum(int(x['correct_flips']) for x in labeled))
        wrong_flips = int(sum(int(x['wrong_flips']) for x in labeled))

        return {
            'cache_count': len(caches),
            'labeled_cache_count': len(labeled),
            'applied_flips': int(sum(per_cache[c]['applied_flips'] for c in caches if c in per_cache)),
            'top1_before_accuracy': float(before_cnt / nprob) if nprob > 0 else None,
            'top1_after_accuracy': float(after_cnt / nprob) if nprob > 0 else None,
            'delta_accuracy': float((after_cnt - before_cnt) / nprob) if nprob > 0 else None,
            'correct_flips': correct_flips,
            'wrong_flips': wrong_flips,
            'net_gain_count': int(correct_flips - wrong_flips),
        }

    primary = aggregate(PRIMARY_AIME_CACHES)
    secondary = aggregate(SECONDARY_DS_CHECK_CACHES)

    return {
        'rule': rule,
        'out_scores': out_scores,
        'applied': applied,
        'applied_by_cache': applied_by_cache,
        'per_cache': per_cache,
        'primary': primary,
        'secondary': secondary,
    }


def main() -> None:
    args = parse_args()

    inp = json.loads(Path(args.input).read_text(encoding='utf-8'))
    base_scores = inp.get('scores', {})

    target_caches = [
        c for c in (PRIMARY_AIME_CACHES + SECONDARY_DS_CHECK_CACHES)
        if c in base_scores and c in DEFAULT_CACHE_MAP
    ]

    pair_by_cache, correctness_maps = build_pair_records(base_scores, target_caches)

    audit_summary = None
    if Path(args.audit_summary).exists():
        audit_summary = json.loads(Path(args.audit_summary).read_text(encoding='utf-8'))

    rules = [
        RuleSpec(
            name='token_only_conservative_v1',
            family='token_only',
            description='Near-tie only; flip when top2 shows lower confidence on >=3 token uncertainty signals.',
            max_gap=1e-3,
            max_flips_total=8,
            max_flips_per_cache=3,
        ),
        RuleSpec(
            name='structure_only_commitment_v1',
            family='structure_only',
            description='Near-tie only; flip when top2 is shorter + parse/int no worse + high unique-answer ambiguity.',
            max_gap=1e-3,
            max_flips_total=8,
            max_flips_per_cache=3,
        ),
        RuleSpec(
            name='token_structure_main_v1',
            family='token_plus_structure',
            description='Main candidate: token conservative vote (>=3) AND structure support under near-tie.',
            max_gap=1e-3,
            max_flips_total=8,
            max_flips_per_cache=3,
        ),
        RuleSpec(
            name='token_structure_with_tiny_activation_veto_v1',
            family='token_plus_structure_tiny_activation_veto',
            description='Same as main but require tiny activation veto evidence (top1 warned / top2 healthier tail).',
            max_gap=1e-3,
            max_flips_total=6,
            max_flips_per_cache=2,
        ),
        RuleSpec(
            name='token_only_confident_v1',
            family='token_only',
            description='Control direction: near-tie only; flip when top2 is more confident on >=3 token signals.',
            max_gap=5e-4,
            max_flips_total=6,
            max_flips_per_cache=2,
        ),
    ]

    candidate_rows: List[Dict[str, object]] = []
    summary_rules: List[Dict[str, object]] = []

    for rule in rules:
        result = apply_rule_and_collect(
            base_scores=base_scores,
            pair_by_cache=pair_by_cache,
            correctness_maps=correctness_maps,
            rule=rule,
            apply_caches=PRIMARY_AIME_CACHES,
            eval_caches=target_caches,
            score_bump=float(args.score_bump),
        )

        primary = result['primary']
        secondary = result['secondary']

        output_submission = None
        output_notes = None
        method_name = f"nad_mixed_v7_{rule.name}"
        if args.write_submissions:
            output_submission = f"/home/jovyan/work/NAD_Next/result/best_of_n_{method_name}.json"
            output_notes = f"/home/jovyan/work/NAD_Next/result/best_of_n_{method_name}_notes.json"

            out_obj = {
                'task': inp.get('task', 'best_of_n'),
                'method_name': method_name,
                'scores': result['out_scores'],
            }
            notes = {
                'base_input': args.input,
                'method_name': method_name,
                'rule_family': rule.family,
                'rule_name': rule.name,
                'description': rule.description,
                'max_gap': rule.max_gap,
                'max_flips_total': rule.max_flips_total,
                'max_flips_per_cache': rule.max_flips_per_cache,
                'apply_scope': PRIMARY_AIME_CACHES,
                'secondary_eval_scope': SECONDARY_DS_CHECK_CACHES,
                'applied_count': len(result['applied']),
                'applied_by_cache': result['applied_by_cache'],
                'primary_metrics': primary,
                'secondary_metrics': secondary,
                'applied_examples': [
                    {
                        'cache_key': r.cache_key,
                        'problem_id': r.problem_id,
                        'top1_sid': r.top1_sid,
                        'top2_sid': r.top2_sid,
                        'gap': r.gap,
                        'token_votes_conservative': token_votes_conservative(r),
                        'token_votes_confident': token_votes_confident(r),
                        'len_ratio_top2_over_top1': (r.len2 / max(r.len1, 1.0)) if np.isfinite(r.len1) and np.isfinite(r.len2) else None,
                        'parse1': r.parse1,
                        'parse2': r.parse2,
                        'int1': r.int1,
                        'int2': r.int2,
                        'unique_answer_count': r.unique_answer_count,
                        'tail_warning1': r.tail_warning1,
                        'tail_warning2': r.tail_warning2,
                    }
                    for r in result['applied'][:20]
                ],
            }

            Path(output_submission).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
            Path(output_notes).write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')

        row = {
            'rule_name': rule.name,
            'family': rule.family,
            'description': rule.description,
            'max_gap': rule.max_gap,
            'max_flips_total': rule.max_flips_total,
            'max_flips_per_cache': rule.max_flips_per_cache,
            'applied_flips_primary_scope': int(primary['applied_flips']),
            'primary_labeled_cache_count': primary['labeled_cache_count'],
            'primary_top1_before_accuracy': primary['top1_before_accuracy'],
            'primary_top1_after_accuracy': primary['top1_after_accuracy'],
            'primary_delta_accuracy': primary['delta_accuracy'],
            'primary_correct_flips': primary['correct_flips'],
            'primary_wrong_flips': primary['wrong_flips'],
            'primary_net_gain_count': primary['net_gain_count'],
            'secondary_labeled_cache_count': secondary['labeled_cache_count'],
            'secondary_top1_before_accuracy': secondary['top1_before_accuracy'],
            'secondary_top1_after_accuracy': secondary['top1_after_accuracy'],
            'secondary_delta_accuracy': secondary['delta_accuracy'],
            'secondary_correct_flips': secondary['correct_flips'],
            'secondary_wrong_flips': secondary['wrong_flips'],
            'secondary_net_gain_count': secondary['net_gain_count'],
            'submission_output': output_submission,
            'notes_output': output_notes,
        }
        candidate_rows.append(row)
        summary_rules.append({
            'rule_name': rule.name,
            'family': rule.family,
            'primary': primary,
            'secondary': secondary,
            'submission_output': output_submission,
            'notes_output': output_notes,
        })

    cand_df = pd.DataFrame(candidate_rows).sort_values(
        ['primary_delta_accuracy', 'primary_net_gain_count', 'applied_flips_primary_scope'],
        ascending=[False, False, True],
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(args.output_csv, index=False)

    best = cand_df.iloc[0].to_dict() if len(cand_df) > 0 else None
    summary = {
        'task': 'rule_distillation_from_feature_audit',
        'base_input': args.input,
        'audit_summary_used': args.audit_summary if audit_summary is not None else None,
        'target_caches_primary': [c for c in PRIMARY_AIME_CACHES if c in target_caches],
        'target_caches_secondary': [c for c in SECONDARY_DS_CHECK_CACHES if c in target_caches],
        'rules_tested': summary_rules,
        'best_rule_by_primary_delta': best,
    }
    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'rules_tested={len(cand_df)}')
    print(f'wrote {args.output_csv}')
    print(f'wrote {args.output_summary}')


if __name__ == '__main__':
    main()
