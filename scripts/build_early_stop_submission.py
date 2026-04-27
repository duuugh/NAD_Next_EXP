#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd


def parse_kv(text: str, sep: str = "=") -> Tuple[str, str]:
    if sep not in text:
        raise ValueError(f"Expected '<key>{sep}<value>', got: {text}")
    key, value = text.split(sep, 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid mapping: {text}")
    return key, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an early-stop submission JSON with schema "
            "scores[cache_key][problem_id][sample_id] = [10 floats]."
        )
    )
    parser.add_argument(
        "--reference",
        default="/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json",
        help="Reference JSON providing the full expected cache/problem/sample structure.",
    )
    parser.add_argument(
        "--final-scores",
        default=None,
        help=(
            "Optional source JSON providing scalar sample scores under `scores`. "
            "If omitted, use `--reference` as the scalar source."
        ),
    )
    parser.add_argument(
        "--output",
        default="/home/jovyan/work/NAD_Next/result/early_stop_submission_baseline_v1.json",
    )
    parser.add_argument(
        "--notes-output",
        default="/home/jovyan/work/NAD_Next/result/early_stop_submission_baseline_v1_notes.json",
    )
    parser.add_argument("--method-name", default="early_stop_baseline_v1")
    parser.add_argument(
        "--value-mode",
        choices=["rank_problem", "minmax_problem", "raw"],
        default="rank_problem",
        help=(
            "How to transform source scalar scores to a [0,1] quality score per sample. "
            "rank_problem is usually the most robust."
        ),
    )

    parser.add_argument(
        "--curve",
        choices=["flat", "linear", "power"],
        default="power",
        help="Fallback curve when dynamics features are unavailable for a cache.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.6,
        help="Exponent for --curve power (ignored for other curves).",
    )
    parser.add_argument(
        "--raw-min",
        type=float,
        default=0.0,
        help="Min for raw normalization when --value-mode=raw.",
    )
    parser.add_argument(
        "--raw-max",
        type=float,
        default=1.0,
        help="Max for raw normalization when --value-mode=raw.",
    )
    parser.add_argument(
        "--require-finite",
        action="store_true",
        help="Fail if source scalar scores contain NaN/Inf.",
    )

    parser.add_argument(
        "--dynamics-stats",
        action="append",
        default=[],
        help=(
            "Cache dynamics stats mapping, repeatable: "
            "cache_key=/path/to/dynamics_statistics.csv"
        ),
    )

    parser.add_argument("--weight-early", type=float, default=0.45)
    parser.add_argument("--weight-tail", type=float, default=0.20)
    parser.add_argument("--weight-accel", type=float, default=0.15)
    parser.add_argument("--weight-quality", type=float, default=0.20)

    parser.add_argument("--early-decay", type=float, default=1.2)
    parser.add_argument("--tail-power", type=float, default=1.1)
    parser.add_argument("--accel-power", type=float, default=0.9)
    parser.add_argument("--quality-power", type=float, default=1.3)

    parser.add_argument(
        "--enforce-monotonic",
        action="store_true",
        help="Enforce score_10 <= ... <= score_100 per sample.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=8,
        help="Decimal rounding digits for each emitted score.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected top-level object in {path}")
    if "scores" not in obj or not isinstance(obj["scores"], dict):
        raise ValueError(f"Expected dict field `scores` in {path}")
    return obj


def scalar_from_value(value: Any) -> float:
    if isinstance(value, list):
        if len(value) != 10:
            raise ValueError(f"Expected list length 10 for early-stop value, got {len(value)}")
        return float(value[-1])
    return float(value)


def finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def to_unit_interval(raw: float, raw_min: float, raw_max: float) -> float:
    if raw_max <= raw_min:
        raise ValueError(f"Invalid raw range: raw_max ({raw_max}) must be > raw_min ({raw_min})")
    scaled = (raw - raw_min) / (raw_max - raw_min)
    return min(1.0, max(0.0, scaled))


def sample_sort_key(sample_id: str) -> Tuple[int, str]:
    text = str(sample_id)
    if text.isdigit():
        return (0, f"{int(text):012d}")
    return (1, text)


def rank_transform(values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(((str(sample_id), float(score)) for sample_id, score in values.items()), key=lambda x: (x[1], x[0]))
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}
    out: Dict[str, float] = {}
    denom = float(n - 1)
    for index, (sample_id, _) in enumerate(items):
        out[sample_id] = index / denom
    return out


def minmax_transform(values: Mapping[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    floats = [float(v) for v in values.values()]
    lo, hi = min(floats), max(floats)
    if hi <= lo:
        return {str(sample_id): 0.5 for sample_id in values}
    return {str(sample_id): (float(score) - lo) / (hi - lo) for sample_id, score in values.items()}


def build_fallback_curve(final_score: float, curve: str, gamma: float, enforce_monotonic: bool) -> List[float]:
    points: List[float] = []
    for idx in range(1, 11):
        p = idx / 10.0
        if curve == "flat":
            v = final_score
        elif curve == "linear":
            v = final_score * p
        elif curve == "power":
            v = final_score * (p ** gamma)
        else:
            raise ValueError(f"Unsupported curve: {curve}")
        points.append(min(1.0, max(0.0, float(v))))
    if enforce_monotonic:
        for i in range(1, len(points)):
            if points[i] < points[i - 1]:
                points[i] = points[i - 1]
    return points


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _median(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    arr = sorted(values)
    m = len(arr) // 2
    if len(arr) % 2 == 1:
        return float(arr[m])
    return float((arr[m - 1] + arr[m]) / 2.0)


def load_dynamics_map(dynamics_args: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for raw in dynamics_args:
        cache_key, path_text = parse_kv(raw)
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Dynamics CSV not found for {cache_key}: {path}")
        df = pd.read_csv(path)
        required = {"problem_id", "run_id", "psi_mid", "T_p_norm", "A_accel", "rho_tail"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dynamics CSV missing columns {missing}: {path}")
        out[cache_key] = df
    return out


def build_problem_dynamics_signals(
    df_cache: pd.DataFrame,
    problem_id: str,
    expected_sample_ids: List[str],
) -> Optional[Dict[str, Dict[str, float]]]:
    # Prefer explicit sample_id if present; otherwise align by run_id order within problem.
    sub = df_cache[df_cache["problem_id"].astype(str) == str(problem_id)].copy()
    if sub.empty:
        return None

    if "sample_id" in sub.columns:
        sub["sample_id_norm"] = sub["sample_id"].astype(str)
        sub = sub[sub["sample_id_norm"].isin(expected_sample_ids)]
    else:
        sub = sub.sort_values("run_id")
        ordered_ids = sorted(expected_sample_ids, key=sample_sort_key)
        if len(sub) != len(ordered_ids):
            # Unable to confidently align sample IDs.
            return None
        sub["sample_id_norm"] = ordered_ids

    if sub.empty:
        return None

    # Build raw per-sample features.
    raw: Dict[str, Dict[str, float]] = {}
    for _, row in sub.iterrows():
        sid = str(row["sample_id_norm"])
        raw[sid] = {
            "psi_mid": _safe_float(row.get("psi_mid"), 0.0),
            "T_p_norm": _safe_float(row.get("T_p_norm"), 0.0),
            "A_accel": _safe_float(row.get("A_accel"), 0.0),
            "rho_tail": _safe_float(row.get("rho_tail"), 0.0),
        }

    if not raw:
        return None

    # Fill missing expected sample ids with medians.
    psi_vals = [v["psi_mid"] for v in raw.values()]
    tp_vals = [v["T_p_norm"] for v in raw.values()]
    acc_vals = [v["A_accel"] for v in raw.values()]
    tail_vals = [v["rho_tail"] for v in raw.values()]

    psi_med = _median(psi_vals, 0.5)
    tp_med = _median(tp_vals, 0.5)
    acc_med = _median(acc_vals, 0.0)
    tail_med = _median(tail_vals, 0.0)

    for sid in expected_sample_ids:
        if sid not in raw:
            raw[sid] = {
                "psi_mid": psi_med,
                "T_p_norm": tp_med,
                "A_accel": acc_med,
                "rho_tail": tail_med,
            }

    early_raw = {
        sid: 0.65 * raw[sid]["psi_mid"] + 0.35 * raw[sid]["T_p_norm"]
        for sid in expected_sample_ids
    }
    tail_raw = {sid: raw[sid]["rho_tail"] for sid in expected_sample_ids}
    accel_inv_raw = {sid: -raw[sid]["A_accel"] for sid in expected_sample_ids}

    signals = {
        "early": rank_transform(early_raw),
        "tail": rank_transform(tail_raw),
        "accel": rank_transform(accel_inv_raw),
    }
    return signals


def budget_mix_score(
    p: float,
    quality: float,
    early: float,
    tail: float,
    accel: float,
    args: argparse.Namespace,
) -> float:
    w_early = float(args.weight_early) * ((1.0 - p) ** float(args.early_decay))
    w_tail = float(args.weight_tail) * (p ** float(args.tail_power))
    w_accel = float(args.weight_accel) * (p ** float(args.accel_power))
    w_quality = float(args.weight_quality) * (p ** float(args.quality_power))

    w_sum = w_early + w_tail + w_accel + w_quality
    if w_sum <= 1e-12:
        return float(quality)

    value = (w_early * early + w_tail * tail + w_accel * accel + w_quality * quality) / w_sum
    return min(1.0, max(0.0, float(value)))


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    source_path = Path(args.final_scores) if args.final_scores else reference_path

    ref_obj = load_json(reference_path)
    src_obj = load_json(source_path)
    dynamics_map = load_dynamics_map(args.dynamics_stats)

    ref_scores = ref_obj["scores"]
    src_scores = src_obj["scores"]

    out_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    stats = {
        "cache_count": 0,
        "problem_count": 0,
        "sample_count": 0,
        "source_file": str(source_path),
        "reference_file": str(reference_path),
        "dynamics_cache_count": len(dynamics_map),
        "dynamics_used_problems": 0,
        "dynamics_fallback_problems": 0,
    }

    for cache_key, ref_problem_map in ref_scores.items():
        if cache_key not in src_scores:
            raise KeyError(f"Missing cache_key in source scores: {cache_key}")

        src_problem_map = src_scores[cache_key]
        out_problem_map: Dict[str, Dict[str, List[float]]] = {}
        cache_dyn_df = dynamics_map.get(cache_key)

        for problem_id, ref_sample_map in ref_problem_map.items():
            if problem_id not in src_problem_map:
                raise KeyError(f"Missing problem_id in source: {cache_key}/{problem_id}")

            src_sample_map = src_problem_map[problem_id]
            sample_ids = [str(sid) for sid in ref_sample_map.keys()]
            raw_values: Dict[str, float] = {}
            for sample_id in sample_ids:
                if sample_id not in src_sample_map:
                    raise KeyError(f"Missing sample_id in source: {cache_key}/{problem_id}/{sample_id}")
                raw = scalar_from_value(src_sample_map[sample_id])
                if args.require_finite and not finite(raw):
                    raise ValueError(f"Non-finite source score at {cache_key}/{problem_id}/{sample_id}: {raw}")
                raw_values[sample_id] = raw

            if args.value_mode == "rank_problem":
                quality_values = rank_transform(raw_values)
            elif args.value_mode == "minmax_problem":
                quality_values = minmax_transform(raw_values)
            elif args.value_mode == "raw":
                quality_values = {
                    sid: to_unit_interval(raw, args.raw_min, args.raw_max)
                    for sid, raw in raw_values.items()
                }
            else:
                raise ValueError(f"Unsupported value mode: {args.value_mode}")

            dyn_signals = None
            if cache_dyn_df is not None:
                dyn_signals = build_problem_dynamics_signals(
                    cache_dyn_df,
                    str(problem_id),
                    sample_ids,
                )

            out_sample_map: Dict[str, List[float]] = {}
            for sample_id in sample_ids:
                quality = float(quality_values[sample_id])
                if dyn_signals is None:
                    curve = build_fallback_curve(quality, args.curve, args.gamma, args.enforce_monotonic)
                else:
                    curve = []
                    early = float(dyn_signals["early"][sample_id])
                    tail = float(dyn_signals["tail"][sample_id])
                    accel = float(dyn_signals["accel"][sample_id])
                    for i in range(1, 11):
                        p = i / 10.0
                        curve.append(budget_mix_score(p, quality, early, tail, accel, args))
                    if args.enforce_monotonic:
                        for i in range(1, len(curve)):
                            if curve[i] < curve[i - 1]:
                                curve[i] = curve[i - 1]

                if args.round_digits >= 0:
                    curve = [round(float(v), int(args.round_digits)) for v in curve]
                out_sample_map[sample_id] = curve

            out_problem_map[str(problem_id)] = out_sample_map
            stats["problem_count"] += 1
            stats["sample_count"] += len(out_sample_map)
            if dyn_signals is None:
                stats["dynamics_fallback_problems"] += 1
            else:
                stats["dynamics_used_problems"] += 1

        out_scores[str(cache_key)] = out_problem_map
        stats["cache_count"] += 1

    out = {
        "task": "early_stop",
        "method_name": args.method_name,
        "scores": out_scores,
    }

    notes = {
        "task": "early_stop",
        "method_name": args.method_name,
        "settings": {
            "reference": str(reference_path),
            "final_scores": str(source_path),
            "value_mode": args.value_mode,
            "curve": args.curve,
            "gamma": args.gamma,
            "raw_min": args.raw_min,
            "raw_max": args.raw_max,
            "require_finite": args.require_finite,
            "dynamics_stats": args.dynamics_stats,
            "weight_early": args.weight_early,
            "weight_tail": args.weight_tail,
            "weight_accel": args.weight_accel,
            "weight_quality": args.weight_quality,
            "early_decay": args.early_decay,
            "tail_power": args.tail_power,
            "accel_power": args.accel_power,
            "quality_power": args.quality_power,
            "enforce_monotonic": args.enforce_monotonic,
            "round_digits": args.round_digits,
        },
        "stats": stats,
    }

    output_path = Path(args.output)
    notes_path = Path(args.notes_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2))

    print(f"wrote {output_path}")
    print(f"wrote {notes_path}")
    print(
        "cache_count={cache_count} problem_count={problem_count} sample_count={sample_count} "
        "dynamics_used_problems={dynamics_used_problems} dynamics_fallback_problems={dynamics_fallback_problems}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
