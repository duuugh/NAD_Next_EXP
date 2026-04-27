#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd


DEFAULT_BENCHMARK_SCALE = {
    "aime24": 1.00,
    "aime25": 0.90,
    "brumo25": 0.90,
    "hmmt25": 0.75,
    "gpqa": 0.60,
    "lcb_v5": 0.55,
}

DEFAULT_MODEL_SCALE = {
    "DS-R1": 1.00,
    "Qwen3-4B": 0.85,
}

DEFAULT_W_STOP = {
    0.1: 0.00,
    0.2: 0.05,
    0.3: 0.10,
    0.4: 0.18,
    0.5: 0.22,
    0.6: 0.22,
    0.7: 0.18,
    0.8: 0.12,
    0.9: 0.08,
    1.0: 0.05,
}

DEFAULT_W_GUARD = {
    0.1: 0.10,
    0.2: 0.10,
    0.3: 0.09,
    0.4: 0.08,
    0.5: 0.07,
    0.6: 0.06,
    0.7: 0.05,
    0.8: 0.04,
    0.9: 0.03,
    1.0: 0.02,
}


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
            "Build early-stop submission with local dynamics v2 plugin: "
            "base_quality + local_gate * dyn_correction."
        )
    )
    parser.add_argument(
        "--reference",
        default="/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json",
    )
    parser.add_argument(
        "--final-scores",
        default=None,
        help="Scalar source JSON under `scores`; default uses --reference.",
    )
    parser.add_argument(
        "--output",
        default="/home/jovyan/work/NAD_Next/result/early_stop_dynamics_v2_local.json",
    )
    parser.add_argument(
        "--notes-output",
        default="/home/jovyan/work/NAD_Next/result/early_stop_dynamics_v2_local_notes.json",
    )
    parser.add_argument("--method-name", default="early_stop_dynamics_v2_local")

    parser.add_argument(
        "--value-mode",
        choices=["rank_problem", "minmax_problem", "raw"],
        default="rank_problem",
    )
    parser.add_argument("--raw-min", type=float, default=0.0)
    parser.add_argument("--raw-max", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.6, help="Base curve power exponent.")
    parser.add_argument(
        "--require-finite",
        action="store_true",
        help="Fail if source scalar score contains NaN/Inf.",
    )

    parser.add_argument(
        "--dynamics-stats",
        action="append",
        default=[],
        help="Repeatable: cache_key=/path/to/dynamics_statistics.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["rho_tail_only", "neg_A_accel_only", "rho_tail_plus_neg_A_accel"],
        default="rho_tail_plus_neg_A_accel",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["strict_local"],
        default="strict_local",
        help="Default conservative local coverage; no global imputation.",
    )
    parser.add_argument(
        "--allow-local-expansion",
        action="store_true",
        help="Reserved for future strict local expansion. Disabled by default.",
    )
    parser.add_argument(
        "--dyn-strength",
        type=float,
        default=0.60,
        help="Global dynamics correction amplitude multiplier.",
    )

    parser.add_argument(
        "--benchmark-scale",
        action="append",
        default=[],
        help="Override benchmark scale, repeatable: benchmark=value",
    )
    parser.add_argument(
        "--model-scale",
        action="append",
        default=[],
        help="Override model scale, repeatable: model=value",
    )

    parser.add_argument("--round-digits", type=int, default=8)
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
        required = {"problem_id", "run_id", "rho_tail", "A_accel"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dynamics CSV missing columns {missing}: {path}")
        out[cache_key] = df
    return out


def load_scale(default: Dict[str, float], overrides: List[str], name: str) -> Dict[str, float]:
    out = {str(k): float(v) for k, v in default.items()}
    for raw in overrides:
        key, value_text = parse_kv(raw)
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {name} override `{raw}`") from exc
        out[str(key)] = value
    return out


def parse_cache_key(cache_key: str) -> Tuple[str, str]:
    if "/" not in str(cache_key):
        return str(cache_key), ""
    model, bench = str(cache_key).split("/", 1)
    return model, bench


def build_problem_dynamics_signals(
    df_cache: pd.DataFrame,
    problem_id: str,
    expected_sample_ids: List[str],
) -> Optional[Dict[str, Dict[str, float]]]:
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
            return None
        sub["sample_id_norm"] = ordered_ids

    if sub.empty:
        return None

    raw: Dict[str, Dict[str, float]] = {}
    for _, row in sub.iterrows():
        sid = str(row["sample_id_norm"])
        raw[sid] = {
            "rho_tail": _safe_float(row.get("rho_tail"), 0.0),
            "A_accel": _safe_float(row.get("A_accel"), 0.0),
        }

    if not raw:
        return None

    tail_vals = [v["rho_tail"] for v in raw.values()]
    acc_vals = [v["A_accel"] for v in raw.values()]
    tail_med = _median(tail_vals, 0.0)
    acc_med = _median(acc_vals, 0.0)

    for sid in expected_sample_ids:
        if sid not in raw:
            raw[sid] = {
                "rho_tail": tail_med,
                "A_accel": acc_med,
            }

    stop_boost_signal = rank_transform({sid: raw[sid]["rho_tail"] for sid in expected_sample_ids})
    rank_guard_signal = rank_transform({sid: -raw[sid]["A_accel"] for sid in expected_sample_ids})

    return {
        "stop_boost_signal": stop_boost_signal,
        "rank_guard_signal": rank_guard_signal,
    }


def weight_stop(budget: float) -> float:
    return float(DEFAULT_W_STOP[round(float(budget), 1)])


def weight_guard(budget: float) -> float:
    return float(DEFAULT_W_GUARD[round(float(budget), 1)])


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def dyn_correction(
    budget: float,
    mode: str,
    stop_boost_signal: float,
    rank_guard_signal: float,
) -> float:
    stop_term = weight_stop(budget) * float(stop_boost_signal)
    guard_term = weight_guard(budget) * float(rank_guard_signal)
    if mode == "rho_tail_only":
        return stop_term
    if mode == "neg_A_accel_only":
        return guard_term
    if mode == "rho_tail_plus_neg_A_accel":
        return stop_term + guard_term
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    source_path = Path(args.final_scores) if args.final_scores else reference_path

    ref_obj = load_json(reference_path)
    src_obj = load_json(source_path)

    dynamics_map = load_dynamics_map(args.dynamics_stats)
    benchmark_scale = load_scale(DEFAULT_BENCHMARK_SCALE, args.benchmark_scale, "benchmark_scale")
    model_scale = load_scale(DEFAULT_MODEL_SCALE, args.model_scale, "model_scale")

    ref_scores = ref_obj["scores"]
    src_scores = src_obj["scores"]
    budgets = [i / 10.0 for i in range(1, 11)]

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
        "coverage_gate_nonzero_problems": 0,
    }

    for cache_key, ref_problem_map in ref_scores.items():
        if cache_key not in src_scores:
            raise KeyError(f"Missing cache_key in source scores: {cache_key}")

        src_problem_map = src_scores[cache_key]
        out_problem_map: Dict[str, Dict[str, List[float]]] = {}
        cache_dyn_df = dynamics_map.get(cache_key)
        model_name, benchmark_name = parse_cache_key(cache_key)
        model_gate = float(model_scale.get(model_name, 0.0))
        benchmark_gate = float(benchmark_scale.get(benchmark_name, 0.0))

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
                quality_values = {sid: to_unit_interval(raw, args.raw_min, args.raw_max) for sid, raw in raw_values.items()}
            else:
                raise ValueError(f"Unsupported value mode: {args.value_mode}")

            dyn_signals = None
            if cache_dyn_df is not None:
                dyn_signals = build_problem_dynamics_signals(
                    cache_dyn_df,
                    str(problem_id),
                    sample_ids,
                )

            coverage_gate = 1.0 if dyn_signals is not None else 0.0
            local_gate = float(args.dyn_strength) * coverage_gate * benchmark_gate * model_gate

            out_sample_map: Dict[str, List[float]] = {}
            for sample_id in sample_ids:
                quality = float(quality_values[sample_id])
                curve: List[float] = []
                for budget in budgets:
                    base_quality = clip01(quality * (budget ** float(args.gamma)))
                    if dyn_signals is None:
                        correction = 0.0
                    else:
                        stop_sig = float(dyn_signals["stop_boost_signal"][sample_id])
                        guard_sig = float(dyn_signals["rank_guard_signal"][sample_id])
                        correction = local_gate * dyn_correction(
                            budget=budget,
                            mode=str(args.mode),
                            stop_boost_signal=stop_sig,
                            rank_guard_signal=guard_sig,
                        )
                    score = clip01(base_quality + correction)
                    curve.append(score)

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
            if local_gate > 0.0:
                stats["coverage_gate_nonzero_problems"] += 1

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
            "raw_min": args.raw_min,
            "raw_max": args.raw_max,
            "gamma": args.gamma,
            "require_finite": args.require_finite,
            "dynamics_stats": args.dynamics_stats,
            "mode": args.mode,
            "coverage_mode": args.coverage_mode,
            "allow_local_expansion": args.allow_local_expansion,
            "dyn_strength": args.dyn_strength,
            "benchmark_scale": benchmark_scale,
            "model_scale": model_scale,
            "weights_stop": DEFAULT_W_STOP,
            "weights_guard": DEFAULT_W_GUARD,
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
