#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from nad.core.views.reader import CacheReader
from scripts.build_mixed_v2_selector_ablation import DEFAULT_CACHE_MAP
from scripts.mixed_v6_local_utils import (
    build_local_pair_features,
    canonical_gap_bucket,
    is_aime_cache_key,
    load_eval_run_info,
    ranked_items,
)


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json"
DEFAULT_MODEL = "/home/jovyan/work/NAD_Next/result/mixed_v6_local_head_full.pkl"
DEFAULT_SELECTED = "/home/jovyan/work/NAD_Next/result/mixed_v6_selected_thresholds.json"

DEFAULT_OUT_GLOBAL = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v6_local_head_oof_global_submit.json"
DEFAULT_NOTES_GLOBAL = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v6_local_head_oof_global_submit_notes.json"
DEFAULT_OUT_BUCKET = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v6_local_head_oof_bucket_submit.json"
DEFAULT_NOTES_BUCKET = "/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v6_local_head_oof_bucket_submit_notes.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply mixed_v6.1 head using OOF-selected threshold(s).")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--selected-thresholds", default=DEFAULT_SELECTED)
    p.add_argument("--mode", choices=["global", "per_bucket"], default="global")
    p.add_argument("--output", default=None)
    p.add_argument("--notes-output", default=None)
    p.add_argument("--method-name", default="nad_mixed_v6_aime_local_binary_corrector_oof")
    p.add_argument("--target-cache-keys", default="", help="Comma-separated cache keys. Empty means auto AIME keys.")
    p.add_argument("--aime-only", action="store_true")
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--max-gap", type=float, default=0.005)
    p.add_argument("--max-flips-total", type=int, default=999999)
    p.add_argument("--max-flips-per-cache", type=int, default=999999)
    p.add_argument("--score-bump", type=float, default=1e-9)
    return p.parse_args()


def default_output_for_mode(mode: str) -> str:
    return DEFAULT_OUT_BUCKET if mode == "per_bucket" else DEFAULT_OUT_GLOBAL


def default_notes_for_mode(mode: str) -> str:
    return DEFAULT_NOTES_BUCKET if mode == "per_bucket" else DEFAULT_NOTES_GLOBAL


def default_target_cache_keys(score_map: Dict[str, Dict]) -> List[str]:
    return [k for k in score_map.keys() if is_aime_cache_key(k)]


def keep_by_gap(gap: float, min_gap: float, max_gap: float) -> bool:
    if gap < min_gap:
        return False
    if max_gap > 0 and gap >= max_gap:
        return False
    return True


def build_x(fd: Dict[str, float], features: List[str], med: np.ndarray) -> np.ndarray:
    x = np.asarray([float(fd.get(k, float("nan"))) for k in features], dtype=np.float64)
    return np.where(np.isfinite(x), x, med)


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_for_mode(args.mode)
    notes_path = args.notes_output or default_notes_for_mode(args.mode)

    with Path(args.model).open("rb") as f:
        bundle = pickle.load(f)

    selected = json.loads(Path(args.selected_thresholds).read_text())

    features = list(bundle["features"])
    med = np.asarray(bundle["nan_medians"], dtype=np.float64)
    scaler = bundle["scaler"]
    model = bundle["model"]

    global_thr = float(selected.get("best_global_threshold", bundle.get("threshold_global_oof", 0.99)))
    per_bucket_thr = {str(k): float(v) for k, v in selected.get("best_per_bucket_thresholds", {}).items()}

    inp = json.loads(Path(args.input).read_text())
    out_scores = copy.deepcopy(inp["scores"])

    if args.target_cache_keys.strip():
        target_cache_keys = [x.strip() for x in args.target_cache_keys.split(",") if x.strip()]
    else:
        target_cache_keys = default_target_cache_keys(out_scores)

    if args.aime_only:
        target_cache_keys = [k for k in target_cache_keys if is_aime_cache_key(k)]

    readers = {}
    eval_infos = {}
    for ck in target_cache_keys:
        root = DEFAULT_CACHE_MAP.get(ck)
        if root is None:
            continue
        readers[ck] = CacheReader(root)
        eval_infos[ck] = load_eval_run_info(root)

    candidates = []
    notes = {
        "task": inp.get("task", "best_of_n"),
        "method_name": args.method_name,
        "mode": args.mode,
        "input": args.input,
        "output": output_path,
        "model": args.model,
        "selected_thresholds": args.selected_thresholds,
        "threshold_global": global_thr,
        "threshold_per_bucket": per_bucket_thr,
        "min_gap": args.min_gap,
        "max_gap": args.max_gap,
        "max_flips_total": args.max_flips_total,
        "max_flips_per_cache": args.max_flips_per_cache,
        "target_cache_keys": target_cache_keys,
        "considered_count": 0,
        "candidate_count": 0,
        "applied_count": 0,
        "considered_by_bucket": {},
        "candidate_by_bucket": {},
        "applied_by_bucket": {},
        "applied_by_cache": {},
        "applied": [],
    }

    considered_by_bucket: Dict[str, int] = {}
    candidate_by_bucket: Dict[str, int] = {}

    for ck, problem_map in out_scores.items():
        if ck not in target_cache_keys or ck not in readers:
            continue
        reader = readers[ck]
        eval_info = eval_infos.get(ck, {})

        for pid, sid_scores in problem_map.items():
            ranked = ranked_items({str(s): float(v) for s, v in sid_scores.items()})
            if len(ranked) < 2:
                continue
            top1_sid, top1_score = ranked[0]
            top2_sid, top2_score = ranked[1]
            top3_score = float(ranked[2][1]) if len(ranked) >= 3 else float("nan")
            gap = float(top1_score - top2_score)

            if not keep_by_gap(gap, min_gap=float(args.min_gap), max_gap=float(args.max_gap)):
                continue

            gap_bucket = canonical_gap_bucket(gap)
            considered_by_bucket[gap_bucket] = considered_by_bucket.get(gap_bucket, 0) + 1
            notes["considered_count"] += 1

            fd = build_local_pair_features(
                reader=reader,
                eval_info=eval_info,
                top1_sid=top1_sid,
                top2_sid=top2_sid,
                top1_score=float(top1_score),
                top2_score=float(top2_score),
                top3_score=float(top3_score),
            )
            x = build_x(fd, features, med)
            p_flip = float(model.predict_proba(scaler.transform(x.reshape(1, -1)))[0, 1])

            if args.mode == "per_bucket":
                thr = float(per_bucket_thr.get(gap_bucket, global_thr))
            else:
                thr = global_thr

            if p_flip >= thr:
                candidate_by_bucket[gap_bucket] = candidate_by_bucket.get(gap_bucket, 0) + 1
                candidates.append(
                    {
                        "cache_key": ck,
                        "problem_id": str(pid),
                        "top1_sid": top1_sid,
                        "top2_sid": top2_sid,
                        "gap": gap,
                        "gap_bucket": gap_bucket,
                        "p_flip": p_flip,
                        "threshold": thr,
                    }
                )

    notes["candidate_count"] = len(candidates)
    notes["considered_by_bucket"] = dict(sorted(considered_by_bucket.items()))
    notes["candidate_by_bucket"] = dict(sorted(candidate_by_bucket.items()))

    candidates.sort(key=lambda x: x["p_flip"], reverse=True)

    applied_by_cache: Dict[str, int] = {}
    applied_by_bucket: Dict[str, int] = {}
    applied_set = set()

    for c in candidates:
        if notes["applied_count"] >= args.max_flips_total:
            break
        ck = c["cache_key"]
        if applied_by_cache.get(ck, 0) >= args.max_flips_per_cache:
            continue
        key = (ck, c["problem_id"])
        if key in applied_set:
            continue

        sid_scores = out_scores[ck][c["problem_id"]]
        adjusted = {str(s): float(v) for s, v in sid_scores.items()}
        adjusted[c["top2_sid"]] = max(adjusted.values()) + float(args.score_bump)
        out_scores[ck][c["problem_id"]] = adjusted

        applied_set.add(key)
        applied_by_cache[ck] = applied_by_cache.get(ck, 0) + 1
        gb = c["gap_bucket"]
        applied_by_bucket[gb] = applied_by_bucket.get(gb, 0) + 1
        notes["applied_count"] += 1
        notes["applied"].append(c)

    notes["applied_by_cache"] = dict(sorted(applied_by_cache.items()))
    notes["applied_by_bucket"] = dict(sorted(applied_by_bucket.items()))

    out = {
        "task": inp.get("task", "best_of_n"),
        "method_name": args.method_name,
        "scores": out_scores,
    }
    Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    Path(notes_path).write_text(json.dumps(notes, ensure_ascii=False, indent=2))

    print(f"mode={args.mode} considered={notes['considered_count']} candidates={notes['candidate_count']} applied={notes['applied_count']}")
    print(f"considered_by_bucket={json.dumps(notes['considered_by_bucket'], ensure_ascii=False)}")
    print(f"applied_by_bucket={json.dumps(notes['applied_by_bucket'], ensure_ascii=False)}")
    print(f"wrote {output_path}")
    print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
