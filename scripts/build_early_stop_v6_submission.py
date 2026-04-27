#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# CLI and parsing
# -----------------------------

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
            "Build Early-Stop v6.1 submission with: "
            "(1) budget-conditioned message passing, "
            "(2) coverage augmentation, "
            "(3) cache alignment calibration."
        )
    )
    parser.add_argument(
        "--reference",
        default="/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json",
        help="Reference JSON for exact cache/problem/sample coverage.",
    )
    parser.add_argument(
        "--final-scores",
        default=None,
        help="Optional source JSON for base quality under `scores` (defaults to --reference).",
    )
    parser.add_argument("--output", default="/home/jovyan/work/NAD_Next/result/early_stop_v6_1_submission.json")
    parser.add_argument("--notes-output", default="/home/jovyan/work/NAD_Next/result/early_stop_v6_1_submission_notes.json")
    parser.add_argument("--method-name", default="early_stop_v6_1_calibrated")

    parser.add_argument("--preset", choices=["stable", "balanced", "aggressive"], default="balanced")

    parser.add_argument(
        "--value-mode",
        choices=["rank_problem", "minmax_problem", "raw"],
        default="rank_problem",
        help="How to map scalar score to base h^0 in [0,1].",
    )
    parser.add_argument("--raw-min", type=float, default=0.0)
    parser.add_argument("--raw-max", type=float, default=1.0)

    parser.add_argument(
        "--dynamics-stats",
        action="append",
        default=[],
        help="Repeatable: cache_key=/path/to/dynamics_statistics.csv",
    )

    # propagation
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--eta0", type=float, default=0.58)
    parser.add_argument("--eta-decay", type=float, default=0.92)

    # budget->depth remap: K(b) = 1 + (k_max-1) * sigmoid(alpha*(b-beta))
    parser.add_argument("--budget-alpha", type=float, default=8.0)
    parser.add_argument("--budget-beta", type=float, default=0.45)

    # gate coefficients
    parser.add_argument("--w-early", type=float, default=1.20)
    parser.add_argument("--w-tail", type=float, default=1.00)
    parser.add_argument("--w-instability", type=float, default=0.85)
    parser.add_argument("--w-cache", type=float, default=0.60)
    parser.add_argument("--w-layer", type=float, default=0.35)
    parser.add_argument("--w-early-dir", type=float, default=1.00)
    parser.add_argument("--w-tail-dir", type=float, default=0.90)
    parser.add_argument("--w-instability-dir", type=float, default=0.80)
    parser.add_argument("--gate-temperature", type=float, default=1.00)

    parser.add_argument(
        "--cache-bias",
        action="append",
        default=[],
        help="Optional overrides: cache_key=bias_value",
    )
    parser.add_argument(
        "--cache-scale",
        action="append",
        default=[],
        help="Optional overrides: cache_key=scale_value",
    )

    # v6.1 metric-aware reweighting
    parser.add_argument("--we-base", type=float, default=0.40)
    parser.add_argument("--wt-base", type=float, default=0.35)
    parser.add_argument("--ws-base", type=float, default=0.25)
    parser.add_argument("--ws-slope", type=float, default=0.10)
    parser.add_argument("--target-pull", type=float, default=0.28)

    # coverage augmentation
    parser.add_argument("--cluster-count", type=int, default=8)
    parser.add_argument("--cluster-iters", type=int, default=20)
    parser.add_argument("--aug-strength", type=float, default=0.35)
    parser.add_argument(
        "--disable-coverage-augmentation",
        action="store_true",
        help="Disable structure-preserving weak augmentation for non-anchor problems.",
    )

    # cache alignment head
    parser.add_argument("--alignment-lambda", type=float, default=0.20)
    parser.add_argument(
        "--alignment-group",
        choices=["dataset", "all"],
        default="dataset",
        help="Alignment groups for cross-cache consistency.",
    )

    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--round-digits", type=int, default=8)
    parser.add_argument("--strict-finite", action="store_true")
    return parser.parse_args()


# -----------------------------
# Utility
# -----------------------------

def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "stable":
        args.eta0 = 0.44
        args.eta_decay = 0.94
        args.alignment_lambda = 0.25
        args.aug_strength = 0.30
        args.budget_alpha = 7.0
        args.w_early_dir = 0.80
        args.w_tail_dir = 0.70
        args.w_instability_dir = 0.65
    elif args.preset == "aggressive":
        args.eta0 = 0.68
        args.eta_decay = 0.90
        args.alignment_lambda = 0.16
        args.aug_strength = 0.42
        args.budget_alpha = 10.0
        args.budget_beta = 0.43
        args.w_early_dir = 1.25
        args.w_tail_dir = 1.15
        args.w_instability_dir = 1.00
        args.target_pull = 0.36
    # balanced keeps parser defaults.


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
        if not value:
            raise ValueError("Expected non-empty score list")
        return float(value[-1])
    return float(value)


def finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def to_unit_interval(raw: float, raw_min: float, raw_max: float) -> float:
    if raw_max <= raw_min:
        raise ValueError(f"Invalid raw range: raw_max ({raw_max}) must be > raw_min ({raw_min})")
    return float(min(1.0, max(0.0, (raw - raw_min) / (raw_max - raw_min))))


def sample_sort_key(sample_id: str) -> Tuple[int, str]:
    text = str(sample_id)
    if text.isdigit():
        return (0, f"{int(text):012d}")
    return (1, text)


def rank_transform(values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(((str(sid), float(v)) for sid, v in values.items()), key=lambda x: (x[1], x[0]))
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}
    denom = float(n - 1)
    return {sid: idx / denom for idx, (sid, _) in enumerate(items)}


def minmax_transform(values: Mapping[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = [float(v) for v in values.values()]
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {str(sid): 0.5 for sid in values}
    return {str(sid): (float(v) - lo) / (hi - lo) for sid, v in values.items()}


def sigmoid_scalar(x: float) -> float:
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def parse_float_overrides(items: List[str], label: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in items:
        key, value = parse_kv(raw)
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float in {label}: {raw}") from exc
    return out


def cache_default_bias(cache_key: str) -> float:
    val = (sum(ord(c) for c in cache_key) % 101) / 100.0
    return (val - 0.5) * 0.5


# -----------------------------
# Dynamics loading and anchor prep
# -----------------------------

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


def align_problem_dynamics(
    cache_dyn_df: Optional[pd.DataFrame],
    problem_id: str,
    sample_ids: List[str],
) -> Optional[Dict[str, Dict[str, float]]]:
    if cache_dyn_df is None:
        return None
    sub = cache_dyn_df[cache_dyn_df["problem_id"].astype(str) == str(problem_id)].copy()
    if sub.empty:
        return None

    if "sample_id" in sub.columns:
        sub["sample_id_norm"] = sub["sample_id"].astype(str)
        sub = sub[sub["sample_id_norm"].isin(sample_ids)]
    else:
        sub = sub.sort_values("run_id")
        ordered = sorted(sample_ids, key=sample_sort_key)
        if len(sub) != len(ordered):
            return None
        sub["sample_id_norm"] = ordered

    if sub.empty:
        return None

    raw: Dict[str, Dict[str, float]] = {}
    for _, row in sub.iterrows():
        sid = str(row["sample_id_norm"])
        raw[sid] = {
            "psi_mid": _safe_float(row.get("psi_mid"), 0.5),
            "T_p_norm": _safe_float(row.get("T_p_norm"), 0.5),
            "A_accel": _safe_float(row.get("A_accel"), 0.0),
            "rho_tail": _safe_float(row.get("rho_tail"), 0.0),
        }

    if not raw:
        return None

    # fill missing sample ids with medians
    def median(xs: List[float], default: float) -> float:
        if not xs:
            return default
        arr = sorted(xs)
        m = len(arr) // 2
        return float(arr[m]) if len(arr) % 2 else float((arr[m - 1] + arr[m]) / 2.0)

    psi_med = median([v["psi_mid"] for v in raw.values()], 0.5)
    tp_med = median([v["T_p_norm"] for v in raw.values()], 0.5)
    acc_med = median([v["A_accel"] for v in raw.values()], 0.0)
    tail_med = median([v["rho_tail"] for v in raw.values()], 0.0)
    for sid in sample_ids:
        if sid not in raw:
            raw[sid] = {
                "psi_mid": psi_med,
                "T_p_norm": tp_med,
                "A_accel": acc_med,
                "rho_tail": tail_med,
            }
    return raw


# -----------------------------
# Weak coverage augmentation
# -----------------------------

@dataclass
class ProblemState:
    cache_key: str
    dataset_key: str
    problem_id: str
    sample_ids: List[str]
    quality: np.ndarray
    quality_rank: np.ndarray
    fp: np.ndarray
    has_anchor: bool
    early: np.ndarray
    tail: np.ndarray
    stability: np.ndarray


@dataclass
class LinearMap:
    slope: float
    intercept: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = self.slope * x + self.intercept
        return np.clip(y, 0.0, 1.0)


def problem_fingerprint(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q_sorted = np.sort(q)
    n = len(q_sorted)
    quant = lambda p: float(q_sorted[min(n - 1, max(0, int((n - 1) * p)))])
    return np.array([
        float(np.mean(q_sorted)),
        float(np.std(q_sorted)),
        quant(0.10),
        quant(0.25),
        quant(0.50),
        quant(0.75),
        quant(0.90),
        float(n) / 128.0,
    ], dtype=np.float64)


def kmeans_assign(features: np.ndarray, k: int, iters: int, rng_seed: int = 13) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = max(1, min(k, n))
    rng = np.random.default_rng(rng_seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = features[init_idx].copy()
    assign = np.zeros(n, dtype=np.int32)

    for _ in range(max(1, iters)):
        # assign
        d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assign = np.argmin(d, axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        # update
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = features[mask].mean(axis=0)
            else:
                centers[c] = features[rng.integers(0, n)]
    return assign


def fit_linear_map(x: np.ndarray, y: np.ndarray) -> LinearMap:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return LinearMap(1.0, 0.0)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var = float(np.mean((x - x_mean) ** 2))
    if var < 1e-9:
        return LinearMap(1.0, 0.0)
    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    slope = cov / var
    intercept = y_mean - slope * x_mean
    return LinearMap(float(slope), float(intercept))


def weak_augment_signal(
    q_rank: np.ndarray,
    mapping: LinearMap,
    aug_strength: float,
    template: Optional[np.ndarray] = None,
) -> np.ndarray:
    pred = mapping.apply(q_rank)
    if template is None:
        blend = pred
    else:
        blend = 0.5 * pred + 0.5 * np.asarray(template, dtype=np.float64)
    out = (1.0 - aug_strength) * q_rank + aug_strength * blend
    return np.clip(out, 0.0, 1.0)


def resample_template(values_sorted_by_rank: np.ndarray, target_len: int) -> np.ndarray:
    src = np.asarray(values_sorted_by_rank, dtype=np.float64)
    if len(src) == target_len:
        return src.copy()
    if len(src) <= 1:
        return np.full(target_len, float(src[0]) if len(src) else 0.5, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(src))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, src)


# -----------------------------
# Graph propagation core
# -----------------------------

def build_component_matrix(
    feature: np.ndarray,
    w_sim: float,
    w_dir: float,
    cache_bias: float,
    layer_bias: float,
    cache_scale: float,
    args: argparse.Namespace,
) -> np.ndarray:
    sim = 1.0 - np.abs(feature[:, None] - feature[None, :])
    direction = feature[None, :] - feature[:, None]
    z = (
        float(w_sim) * sim
        + float(w_dir) * direction
        + float(args.w_cache) * float(cache_bias)
        + float(args.w_layer) * float(layer_bias)
    ) * float(cache_scale) / max(float(args.gate_temperature), float(args.eps))

    m = sigmoid(z)
    np.fill_diagonal(m, 0.0)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > float(args.eps), row_sum, 1.0)
    return m / row_sum


def budget_to_depth(b: float, k_max: int, alpha: float, beta: float) -> int:
    s = sigmoid_scalar(alpha * (b - beta))
    val = 1.0 + (k_max - 1) * s
    k = int(round(val))
    return max(1, min(k_max, k))


def metric_weights(b: float, args: argparse.Namespace) -> Tuple[float, float, float]:
    we = float(args.we_base) * (1.0 - b)
    wt = float(args.wt_base) * b
    ws = float(args.ws_base) + float(args.ws_slope) * b
    z = max(1e-12, we + wt + ws)
    return we / z, wt / z, ws / z


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    apply_preset(args)

    reference_path = Path(args.reference)
    source_path = Path(args.final_scores) if args.final_scores else reference_path

    ref_obj = load_json(reference_path)
    src_obj = load_json(source_path)
    dyn_map = load_dynamics_map(args.dynamics_stats)

    bias_overrides = parse_float_overrides(args.cache_bias, "--cache-bias")
    scale_overrides = parse_float_overrides(args.cache_scale, "--cache-scale")

    ref_scores = ref_obj["scores"]
    src_scores = src_obj["scores"]

    budgets = [i / 10.0 for i in range(1, 11)]
    k_plan = {
        f"{b:.1f}": budget_to_depth(b, int(args.k_max), float(args.budget_alpha), float(args.budget_beta))
        for b in budgets
    }

    # Build problem states first (with anchors where available).
    states: Dict[Tuple[str, str], ProblemState] = {}

    anchor_pairs_global = {"early": ([], []), "tail": ([], []), "stability": ([], [])}
    anchor_pairs_dataset: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}

    cache_stats: Dict[str, Dict[str, Any]] = {}

    total_problems = 0
    total_samples = 0
    anchor_problem_count = 0

    for cache_key, ref_problem_map in ref_scores.items():
        if cache_key not in src_scores:
            raise KeyError(f"Missing cache_key in source scores: {cache_key}")

        dataset_key = cache_key.split("/", 1)[1] if "/" in cache_key else cache_key
        cache_dyn_df = dyn_map.get(cache_key)

        cache_stats[cache_key] = {
            "problems": 0,
            "samples": 0,
            "anchor_problems": 0,
            "augmented_problems": 0,
            "cache_bias": bias_overrides.get(cache_key, cache_default_bias(cache_key)),
            "cache_scale": scale_overrides.get(cache_key, 1.0),
            "max_abs_delta": 0.0,
        }

        for problem_id, ref_sample_map in ref_problem_map.items():
            src_problem_map = src_scores[cache_key]
            if problem_id not in src_problem_map:
                raise KeyError(f"Missing problem_id in source: {cache_key}/{problem_id}")

            sample_ids = [str(sid) for sid in ref_sample_map.keys()]
            raw_values: Dict[str, float] = {}
            for sid in sample_ids:
                if sid not in src_problem_map[problem_id]:
                    raise KeyError(f"Missing sample_id in source: {cache_key}/{problem_id}/{sid}")
                raw = scalar_from_value(src_problem_map[problem_id][sid])
                if args.strict_finite and not finite(raw):
                    raise ValueError(f"Non-finite source score at {cache_key}/{problem_id}/{sid}: {raw}")
                raw_values[sid] = raw

            if args.value_mode == "rank_problem":
                q_map = rank_transform(raw_values)
            elif args.value_mode == "minmax_problem":
                q_map = minmax_transform(raw_values)
            else:
                q_map = {sid: to_unit_interval(v, args.raw_min, args.raw_max) for sid, v in raw_values.items()}

            quality = np.array([float(q_map[sid]) for sid in sample_ids], dtype=np.float64)
            q_rank = np.array([rank_transform({sid: q for sid, q in zip(sample_ids, quality)})[sid] for sid in sample_ids], dtype=np.float64)

            raw_dyn = align_problem_dynamics(cache_dyn_df, str(problem_id), sample_ids)
            if raw_dyn is not None:
                early_raw = {sid: 0.65 * raw_dyn[sid]["psi_mid"] + 0.35 * raw_dyn[sid]["T_p_norm"] for sid in sample_ids}
                tail_raw = {sid: raw_dyn[sid]["rho_tail"] for sid in sample_ids}
                stability_raw = {sid: -raw_dyn[sid]["A_accel"] for sid in sample_ids}
                early_rank_map = rank_transform(early_raw)
                tail_rank_map = rank_transform(tail_raw)
                stability_rank_map = rank_transform(stability_raw)
                early = np.array([float(early_rank_map[sid]) for sid in sample_ids], dtype=np.float64)
                tail = np.array([float(tail_rank_map[sid]) for sid in sample_ids], dtype=np.float64)
                stability = np.array([float(stability_rank_map[sid]) for sid in sample_ids], dtype=np.float64)
                has_anchor = True
            else:
                early = q_rank.copy()
                tail = q_rank.copy()
                stability = q_rank.copy()
                has_anchor = False

            state = ProblemState(
                cache_key=cache_key,
                dataset_key=dataset_key,
                problem_id=str(problem_id),
                sample_ids=sample_ids,
                quality=quality,
                quality_rank=q_rank,
                fp=problem_fingerprint(quality),
                has_anchor=has_anchor,
                early=early,
                tail=tail,
                stability=stability,
            )
            states[(cache_key, str(problem_id))] = state

            total_problems += 1
            total_samples += len(sample_ids)
            cache_stats[cache_key]["problems"] += 1
            cache_stats[cache_key]["samples"] += len(sample_ids)

            if has_anchor:
                anchor_problem_count += 1
                cache_stats[cache_key]["anchor_problems"] += 1
                dmap = anchor_pairs_dataset.setdefault(dataset_key, {
                    "early": ([], []), "tail": ([], []), "stability": ([], [])
                })
                for key_name, y in [("early", early), ("tail", tail), ("stability", stability)]:
                    dmap[key_name][0].extend(q_rank.tolist())
                    dmap[key_name][1].extend(y.tolist())
                    anchor_pairs_global[key_name][0].extend(q_rank.tolist())
                    anchor_pairs_global[key_name][1].extend(y.tolist())

    # Build global/dataset linear maps for fallback augmentation.
    global_maps = {
        key_name: fit_linear_map(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
        for key_name, (x, y) in anchor_pairs_global.items()
    }
    dataset_maps: Dict[str, Dict[str, LinearMap]] = {}
    for dataset_key, dd in anchor_pairs_dataset.items():
        dataset_maps[dataset_key] = {
            key_name: fit_linear_map(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
            for key_name, (x, y) in dd.items()
        }

    # Cache-local cluster augmentation for non-anchor problems.
    if not args.disable_coverage_augmentation:
        for cache_key, ref_problem_map in ref_scores.items():
            problem_ids = [str(pid) for pid in ref_problem_map.keys()]
            cache_states = [states[(cache_key, pid)] for pid in problem_ids]
            features = np.stack([s.fp for s in cache_states], axis=0)
            assign = kmeans_assign(features, int(args.cluster_count), int(args.cluster_iters), rng_seed=13)

            # cluster maps + weak templates from anchors only
            cluster_maps: Dict[int, Dict[str, LinearMap]] = {}
            cluster_templates: Dict[int, Dict[str, np.ndarray]] = {}
            for c in np.unique(assign):
                idx = np.where(assign == c)[0]
                x_e, y_e, y_t, y_s = [], [], [], []
                tpl_e, tpl_t, tpl_s = [], [], []
                for i in idx:
                    s = cache_states[i]
                    if not s.has_anchor:
                        continue
                    x = s.quality_rank.tolist()
                    x_e.extend(x)
                    y_e.extend(s.early.tolist())
                    y_t.extend(s.tail.tolist())
                    y_s.extend(s.stability.tolist())

                    order = np.argsort(s.quality_rank)
                    tpl_e.append(s.early[order])
                    tpl_t.append(s.tail[order])
                    tpl_s.append(s.stability[order])
                if x_e:
                    cluster_maps[int(c)] = {
                        "early": fit_linear_map(np.array(x_e), np.array(y_e)),
                        "tail": fit_linear_map(np.array(x_e), np.array(y_t)),
                        "stability": fit_linear_map(np.array(x_e), np.array(y_s)),
                    }
                    max_len = max(len(x) for x in tpl_e)
                    te = np.mean(np.stack([resample_template(x, max_len) for x in tpl_e], axis=0), axis=0)
                    tt = np.mean(np.stack([resample_template(x, max_len) for x in tpl_t], axis=0), axis=0)
                    ts = np.mean(np.stack([resample_template(x, max_len) for x in tpl_s], axis=0), axis=0)
                    cluster_templates[int(c)] = {"early": te, "tail": tt, "stability": ts}

            dataset_key = cache_key.split("/", 1)[1] if "/" in cache_key else cache_key
            for i, pid in enumerate(problem_ids):
                s = states[(cache_key, pid)]
                if s.has_anchor:
                    continue
                c = int(assign[i])
                if c in cluster_maps:
                    maps = cluster_maps[c]
                    templates = cluster_templates.get(c)
                elif dataset_key in dataset_maps:
                    maps = dataset_maps[dataset_key]
                    templates = None
                else:
                    maps = global_maps
                    templates = None

                n = len(s.sample_ids)
                order = np.argsort(s.quality_rank)
                inv = np.empty_like(order)
                inv[order] = np.arange(n)

                def get_template(name: str) -> Optional[np.ndarray]:
                    if templates is None or name not in templates:
                        return None
                    base = resample_template(templates[name], n)
                    return base[inv]

                s.early = weak_augment_signal(s.quality_rank, maps["early"], float(args.aug_strength), get_template("early"))
                s.tail = weak_augment_signal(s.quality_rank, maps["tail"], float(args.aug_strength), get_template("tail"))
                s.stability = weak_augment_signal(s.quality_rank, maps["stability"], float(args.aug_strength), get_template("stability"))
                cache_stats[cache_key]["augmented_problems"] += 1

    # Propagation and budget scores.
    raw_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    max_delta_abs = 0.0
    nan_inf_count = 0

    for cache_key, ref_problem_map in ref_scores.items():
        out_problem_map: Dict[str, Dict[str, List[float]]] = {}
        cache_bias = cache_stats[cache_key]["cache_bias"]
        cache_scale = cache_stats[cache_key]["cache_scale"]

        for problem_id in ref_problem_map.keys():
            pid = str(problem_id)
            s = states[(cache_key, pid)]

            h = s.quality.copy()
            states_k = [h.copy()]
            for k in range(1, int(args.k_max) + 1):
                b_layer = float(k) / float(args.k_max)
                layer_bias = b_layer
                we, wt, ws = metric_weights(b_layer, args)

                p_e = build_component_matrix(
                    feature=s.early,
                    w_sim=float(args.w_early),
                    w_dir=float(args.w_early_dir),
                    cache_bias=cache_bias,
                    layer_bias=layer_bias,
                    cache_scale=cache_scale,
                    args=args,
                )
                p_t = build_component_matrix(
                    feature=s.tail,
                    w_sim=float(args.w_tail),
                    w_dir=float(args.w_tail_dir),
                    cache_bias=cache_bias,
                    layer_bias=layer_bias,
                    cache_scale=cache_scale,
                    args=args,
                )
                p_s = build_component_matrix(
                    feature=s.stability,
                    w_sim=float(args.w_instability),
                    w_dir=float(args.w_instability_dir),
                    cache_bias=cache_bias,
                    layer_bias=layer_bias,
                    cache_scale=cache_scale,
                    args=args,
                )

                comp_e = p_e @ h - h
                comp_t = p_t @ h - h
                comp_s = p_s @ h - h
                update = we * comp_e + wt * comp_t + ws * comp_s

                target = we * s.early + wt * s.tail + ws * s.stability
                pull = float(args.target_pull) * (0.6 + 0.4 * b_layer)
                update = update + pull * (target - h)

                eta_k = float(args.eta0) * (float(args.eta_decay) ** (k - 1))
                eta_k = max(0.0, min(1.0, eta_k))
                h_next = np.clip(h + eta_k * update, 0.0, 1.0)

                if not np.all(np.isfinite(h_next)):
                    nan_inf_count += int(np.size(h_next) - np.isfinite(h_next).sum())
                    h_next = np.nan_to_num(h_next, nan=0.5, posinf=1.0, neginf=0.0)

                delta = float(np.max(np.abs(h_next - h)))
                max_delta_abs = max(max_delta_abs, delta)
                cache_stats[cache_key]["max_abs_delta"] = max(cache_stats[cache_key]["max_abs_delta"], delta)

                h = h_next
                states_k.append(h.copy())

            out_sample_map: Dict[str, List[float]] = {}
            for idx, sid in enumerate(s.sample_ids):
                seq = []
                for b in budgets:
                    kk = budget_to_depth(float(b), int(args.k_max), float(args.budget_alpha), float(args.budget_beta))
                    v = float(states_k[kk][idx])
                    if args.round_digits >= 0:
                        v = round(v, int(args.round_digits))
                    seq.append(v)
                out_sample_map[sid] = seq

            out_problem_map[pid] = out_sample_map

        raw_scores[cache_key] = out_problem_map

    # Cache alignment head (cross-cache consistency calibration).
    aligned_scores = json.loads(json.dumps(raw_scores))

    if float(args.alignment_lambda) > 0.0:
        if args.alignment_group == "dataset":
            group_map: Dict[str, List[str]] = {}
            for cache_key in raw_scores.keys():
                dataset_key = cache_key.split("/", 1)[1] if "/" in cache_key else cache_key
                group_map.setdefault(dataset_key, []).append(cache_key)
        else:
            group_map = {"all": list(raw_scores.keys())}

        bias_values = []
        for _, cache_keys in group_map.items():
            if not cache_keys:
                continue
            # union problem ids in group
            problem_ids = sorted({pid for ck in cache_keys for pid in raw_scores[ck].keys()})
            for pid in problem_ids:
                for b_idx in range(10):
                    centers = {}
                    for ck in cache_keys:
                        if pid not in raw_scores[ck]:
                            continue
                        vals = [float(seq[b_idx]) for seq in raw_scores[ck][pid].values()]
                        if vals:
                            centers[ck] = float(np.mean(vals))
                    if len(centers) <= 1:
                        continue
                    global_center = float(np.mean(list(centers.values())))
                    for ck, c_center in centers.items():
                        bias = c_center - global_center
                        bias_values.append(bias)
                        correction = float(args.alignment_lambda) * bias
                        for sid, seq in aligned_scores[ck][pid].items():
                            seq[b_idx] = float(np.clip(seq[b_idx] - correction, 0.0, 1.0))

    # Sanity checks
    # 1) variance(cache_bias) > threshold
    bias_arr = np.array([cache_stats[ck]["cache_bias"] for ck in sorted(cache_stats)], dtype=np.float64)
    cache_bias_var = float(np.var(bias_arr)) if len(bias_arr) else 0.0

    # 2) rank_correlation(0.1,1.0) < 0.95 (average over problems)
    def rank_corr(v1: np.ndarray, v2: np.ndarray) -> float:
        r1 = np.argsort(np.argsort(v1))
        r2 = np.argsort(np.argsort(v2))
        if len(r1) < 2:
            return 1.0
        return float(np.corrcoef(r1, r2)[0, 1]) if np.std(r1) > 0 and np.std(r2) > 0 else 1.0

    cors = []
    for ck, pmap in aligned_scores.items():
        for _, smap in pmap.items():
            arr = np.array([seq for _, seq in sorted(smap.items(), key=lambda kv: sample_sort_key(kv[0]))], dtype=np.float64)
            if arr.shape[0] >= 2:
                cors.append(rank_corr(arr[:, 0], arr[:, 9]))
    avg_rank_corr_01_10 = float(np.mean(cors)) if cors else 1.0

    # 3) score range stable
    mn, mx = 1.0, 0.0
    for ck in aligned_scores:
        for pid in aligned_scores[ck]:
            for sid in aligned_scores[ck][pid]:
                seq = aligned_scores[ck][pid][sid]
                mn = min(mn, min(seq))
                mx = max(mx, max(seq))
    score_range_stable = (mn >= -1e-9 and mx <= 1.0 + 1e-9)

    out = {
        "task": "early_stop",
        "method_name": args.method_name,
        "scores": aligned_scores,
    }

    notes = {
        "task": "early_stop",
        "method_name": args.method_name,
        "settings": {
            "preset": args.preset,
            "reference": str(reference_path),
            "final_scores": str(source_path),
            "value_mode": args.value_mode,
            "raw_min": args.raw_min,
            "raw_max": args.raw_max,
            "k_max": args.k_max,
            "budget_to_k": k_plan,
            "budget_alpha": args.budget_alpha,
            "budget_beta": args.budget_beta,
            "eta0": args.eta0,
            "eta_decay": args.eta_decay,
            "w_early": args.w_early,
            "w_tail": args.w_tail,
            "w_instability": args.w_instability,
            "w_cache": args.w_cache,
            "w_layer": args.w_layer,
            "w_early_dir": args.w_early_dir,
            "w_tail_dir": args.w_tail_dir,
            "w_instability_dir": args.w_instability_dir,
            "gate_temperature": args.gate_temperature,
            "we_base": args.we_base,
            "wt_base": args.wt_base,
            "ws_base": args.ws_base,
            "ws_slope": args.ws_slope,
            "target_pull": args.target_pull,
            "dynamics_stats": args.dynamics_stats,
            "cluster_count": args.cluster_count,
            "cluster_iters": args.cluster_iters,
            "aug_strength": args.aug_strength,
            "coverage_augmentation_enabled": not args.disable_coverage_augmentation,
            "alignment_lambda": args.alignment_lambda,
            "alignment_group": args.alignment_group,
            "cache_bias_overrides": args.cache_bias,
            "cache_scale_overrides": args.cache_scale,
            "strict_finite": args.strict_finite,
            "round_digits": args.round_digits,
        },
        "summary": {
            "cache_count": len(aligned_scores),
            "problem_count": total_problems,
            "sample_count": total_samples,
            "anchor_problem_count": anchor_problem_count,
            "dynamics_cache_count": len(dyn_map),
            "global_max_abs_delta": max_delta_abs,
            "nan_inf_count": nan_inf_count,
            "cache_bias_variance": cache_bias_var,
            "avg_rank_corr_b01_b10": avg_rank_corr_01_10,
            "score_min": float(mn),
            "score_max": float(mx),
            "score_range_stable": bool(score_range_stable),
            "sanity_checks": {
                "cache_bias_variance_gt_1e-6": bool(cache_bias_var > 1e-6),
                "avg_rank_corr_b01_b10_lt_0_95": bool(avg_rank_corr_01_10 < 0.95),
                "score_range_stable": bool(score_range_stable),
            },
        },
        "cache_stats": cache_stats,
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
        "anchor_problem_count={anchor_problem_count} global_max_abs_delta={global_max_abs_delta:.6f} "
        "nan_inf_count={nan_inf_count} avg_rank_corr_b01_b10={avg_rank_corr_b01_b10:.4f}".format(
            **notes["summary"]
        )
    )


if __name__ == "__main__":
    main()
