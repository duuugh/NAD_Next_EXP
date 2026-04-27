#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nad.core.views.reader import CacheReader


AIME_DATASETS = {"aime24", "aime25"}


def is_aime_cache_key(cache_key: str) -> bool:
    dataset = str(cache_key).split("/")[-1].strip().lower()
    return dataset in AIME_DATASETS


def ranked_items(sid_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(((str(s), float(v)) for s, v in sid_scores.items()), key=lambda kv: (-kv[1], kv[0]))


def safe_mean(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float("nan")
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def token_length(reader: CacheReader, sid: int) -> float:
    tv = reader.get_token_view(int(sid))
    for field in ("token_ids", "tok_logprob", "tok_selfcert", "tok_conf", "tok_neg_entropy", "tok_gini"):
        arr = getattr(tv, field, None)
        if arr is not None:
            return float(len(arr))
    return float("nan")


def metric_means(reader: CacheReader, sid: int) -> Dict[str, float]:
    tv = reader.get_token_view(int(sid))
    return {
        "tok_logprob_mean": safe_mean(getattr(tv, "tok_logprob", None)),
        "tok_selfcert_mean": safe_mean(getattr(tv, "tok_selfcert", None)),
    }


def parse_extracted_answer(extracted_answer: object) -> Tuple[str, bool]:
    if extracted_answer is None:
        return "", False
    raw = str(extracted_answer).strip()
    if not raw:
        return "", False
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return "", False
            parsed_text = str(parsed[0]).strip()
            return parsed_text, bool(parsed_text)
        parsed_text = str(parsed).strip()
        return parsed_text, bool(parsed_text)
    except Exception:
        return raw, True


def is_integer_text(text: str) -> bool:
    s = str(text).strip()
    if not s:
        return False
    return re.fullmatch(r"[+-]?\d+", s) is not None


def load_eval_run_info(cache_root: str) -> Dict[int, Dict[str, float]]:
    root = Path(cache_root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sample_index: Dict[Tuple[str, int], int] = {}
    for sid, sample in enumerate(meta.get("samples", [])):
        pid = str(sample.get("problem_id"))
        run_idx = int(sample.get("run_index", 0))
        sample_index[(pid, run_idx)] = int(sid)

    report = None
    for p in (root / "evaluation_report_compact.json", root / "evaluation_report.json"):
        if p.exists():
            report = json.loads(p.read_text(encoding="utf-8"))
            break
    if report is None:
        return {}
    if "results" not in report:
        return {}

    info: Dict[int, Dict[str, float]] = {}
    for result in report.get("results", []):
        pid = str(result.get("problem_id"))
        for run in result.get("runs", []):
            run_idx = int(run.get("run_index", run.get("index", 0)))
            sid = sample_index.get((pid, run_idx))
            if sid is None:
                continue
            ans_text, parse_ok = parse_extracted_answer(run.get("extracted_answer", ""))
            out_tokens_raw = run.get("output_tokens", None)
            out_tokens = float(out_tokens_raw) if out_tokens_raw is not None else float("nan")
            if not np.isfinite(out_tokens):
                out_tokens = float("nan")
            info[sid] = {
                "answer_parse_ok": float(1.0 if parse_ok else 0.0),
                "answer_is_int": float(1.0 if is_integer_text(ans_text) else 0.0),
                "output_tokens": out_tokens,
            }
    return info


def get_run_aux_features(reader: CacheReader, eval_info: Dict[int, Dict[str, float]], sid: int) -> Dict[str, float]:
    m = metric_means(reader, sid)
    aux = eval_info.get(int(sid), {})
    length = float(aux.get("output_tokens", float("nan")))
    if not np.isfinite(length):
        length = token_length(reader, sid)
    return {
        "tok_logprob_mean": m["tok_logprob_mean"],
        "tok_selfcert_mean": m["tok_selfcert_mean"],
        "length": float(length),
        "answer_parse_ok": float(aux.get("answer_parse_ok", float("nan"))),
        "answer_is_int": float(aux.get("answer_is_int", float("nan"))),
    }


def build_local_pair_features(
    reader: CacheReader,
    eval_info: Dict[int, Dict[str, float]],
    top1_sid: str,
    top2_sid: str,
    top1_score: float,
    top2_score: float,
    top3_score: float,
) -> Dict[str, float]:
    s1 = float(top1_score)
    s2 = float(top2_score)
    s3 = float(top3_score)

    a1 = get_run_aux_features(reader, eval_info, int(top1_sid))
    a2 = get_run_aux_features(reader, eval_info, int(top2_sid))

    return {
        "s1": s1,
        "s2": s2,
        "gap": s1 - s2,
        "s1_minus_s3": s1 - s3 if np.isfinite(s3) else float("nan"),
        "s2_minus_s3": s2 - s3 if np.isfinite(s3) else float("nan"),
        "lp1": a1["tok_logprob_mean"],
        "lp2": a2["tok_logprob_mean"],
        "lp_gap_top2_minus_top1": a2["tok_logprob_mean"] - a1["tok_logprob_mean"],
        "sc1": a1["tok_selfcert_mean"],
        "sc2": a2["tok_selfcert_mean"],
        "sc_gap_top2_minus_top1": a2["tok_selfcert_mean"] - a1["tok_selfcert_mean"],
        "len1": a1["length"],
        "len2": a2["length"],
        "len_gap_top2_minus_top1": a2["length"] - a1["length"],
        "parse_ok1": a1["answer_parse_ok"],
        "parse_ok2": a2["answer_parse_ok"],
        "parse_ok_gap_top2_minus_top1": a2["answer_parse_ok"] - a1["answer_parse_ok"],
        "is_int1": a1["answer_is_int"],
        "is_int2": a2["answer_is_int"],
        "is_int_gap_top2_minus_top1": a2["answer_is_int"] - a1["answer_is_int"],
    }


def parse_bucket_edges(raw: str) -> List[float]:
    if not raw.strip():
        return []
    vals: List[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    vals = sorted(set(vals))
    return vals


def canonical_gap_bucket(gap: float) -> str:
    """Fixed bucket names for mixed_v6.1.

    Buckets:
    - lt_5e4: gap < 5e-4
    - btw_5e4_1e3: 5e-4 <= gap < 1e-3
    - btw_1e3_2e3: 1e-3 <= gap < 2e-3
    - ge_2e3: gap >= 2e-3
    """
    g = float(gap)
    if g < 5e-4:
        return "lt_5e4"
    if g < 1e-3:
        return "btw_5e4_1e3"
    if g < 2e-3:
        return "btw_1e3_2e3"
    return "ge_2e3"


def bucket_label(gap: float, edges: List[float]) -> str:
    # Keep API compatibility while standardizing bucket names.
    return canonical_gap_bucket(gap)
