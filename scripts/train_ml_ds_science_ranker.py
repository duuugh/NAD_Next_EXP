#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science.csv"
DEFAULT_OUTPUT_METRICS = "/home/jovyan/work/NAD_Next/result/ml_ds_science_ranker_lgbm_metrics.json"
DEFAULT_OUTPUT_VALID_PRED = "/home/jovyan/work/NAD_Next/result/ml_ds_science_ranker_valid_predictions.csv"
DEFAULT_OUTPUT_COMPARISON = "/home/jovyan/work/NAD_Next/result/ml_ds_science_ranker_comparison.csv"
DEFAULT_OUTPUT_MODEL = "/home/jovyan/work/NAD_Next/result/ml_ds_science_rker_lgbm.pkl"

DEFAULT_PREV_LOGREG_MODEL = "/home/jovyan/work/NAD_Next/result/ml_ds_science_logreg.pkl"
DEFAULT_PREV_LGBM_BINARY_MODEL = "/home/jovyan/work/NAD_Next/result/ml_ds_science_lgbm.pkl"

TARGET_CACHE_KEYS = [
    "DS-R1/brumo25",
    "DS-R1/gpqa",
    "DS-R1/hmmt25",
]

FEATURE_COLUMNS = [
    "mixed_v1_score",
    "rank_by_mixed_v1",
    "top1_gap_under_mixed_v1",
    "top2_gap_under_mixed_v1",
    "tok_logprob_mean",
    "tok_selfcert_mean",
    "tok_conf_mean",
    "tok_neg_entropy_mean",
    "tok_gini_mean",
    "answer_length_tokens",
    "parse_success",
    "is_integer_answer",
    "tail_warning",
    "tail_new_ratio",
    "plateau_progress",
    "cumulative_unique_neurons_end",
    "num_candidates_for_problem",
    "unique_answer_count_in_problem",
]

BOOL_COLUMNS = ["parse_success", "is_integer_answer", "tail_warning"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DS-science ranking baseline with LGBMRanker.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-metrics", default=DEFAULT_OUTPUT_METRICS)
    p.add_argument("--output-valid-predictions", default=DEFAULT_OUTPUT_VALID_PRED)
    p.add_argument("--output-comparison", default=DEFAULT_OUTPUT_COMPARISON)
    p.add_argument("--output-model", default=DEFAULT_OUTPUT_MODEL)
    p.add_argument("--prev-logreg-model", default=DEFAULT_PREV_LOGREG_MODEL)
    p.add_argument("--prev-lgbm-binary-model", default=DEFAULT_PREV_LGBM_BINARY_MODEL)
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def parse_bool_series(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    out: List[float] = []
    for x in series:
        if pd.isna(x):
            out.append(np.nan)
            continue
        if isinstance(x, bool):
            out.append(float(x))
            continue
        s = str(x).strip().lower()
        if s in truthy:
            out.append(1.0)
        elif s in falsy:
            out.append(0.0)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="float64")


def candidate_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def problem_top1_accuracy(df: pd.DataFrame, score_col: str) -> float:
    chosen = (
        df.sort_values(["group_id", score_col, "sid"], ascending=[True, False, True])
        .groupby("group_id", as_index=False)
        .head(1)
    )
    return float(chosen["is_correct"].astype(int).mean()) if not chosen.empty else float("nan")


def problem_topk_accuracy(df: pd.DataFrame, score_col: str, k: int) -> float:
    hits: List[int] = []
    for _, grp in df.groupby("group_id", sort=False):
        topk = grp.sort_values([score_col, "mixed_v1_score", "sid"], ascending=[False, False, True]).head(k)
        hits.append(int(topk["is_correct"].astype(int).max() > 0))
    return float(np.mean(hits)) if hits else float("nan")


def pairwise_accuracy(df: pd.DataFrame, score_col: str) -> float:
    wins = 0.0
    total = 0
    for _, grp in df.groupby("group_id", sort=False):
        pos = grp[grp["is_correct"] == 1][score_col].to_numpy(dtype=np.float64)
        neg = grp[grp["is_correct"] == 0][score_col].to_numpy(dtype=np.float64)
        if pos.size == 0 or neg.size == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        wins += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        total += int(diff.size)
    return float(wins / total) if total > 0 else float("nan")


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(np.sum((2.0 ** rel - 1.0) / denom))


def ndcg_at_k_for_group(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ideal_labels = np.sort(labels)[::-1]
    dcg = dcg_at_k(ranked_labels, k)
    idcg = dcg_at_k(ideal_labels, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int) -> float:
    vals: List[float] = []
    for _, grp in df.groupby("group_id", sort=False):
        y = grp["is_correct"].to_numpy(dtype=np.float64)
        s = grp[score_col].to_numpy(dtype=np.float64)
        vals.append(ndcg_at_k_for_group(y, s, k))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_problem_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    p1 = problem_top1_accuracy(df, score_col)
    base = problem_top1_accuracy(df, "mixed_v1_score")
    return {
        "valid_problem_count": int(df["group_id"].nunique()),
        "problem_top1_accuracy": p1,
        "mixed_v1_problem_top1_accuracy": base,
        "delta_vs_mixed_v1": float(p1 - base),
        "top3_accuracy": problem_topk_accuracy(df, score_col, 3),
        "pairwise_accuracy": pairwise_accuracy(df, score_col),
        "ndcg_at_1": ndcg_at_k(df, score_col, 1),
        "ndcg_at_3": ndcg_at_k(df, score_col, 3),
    }


def group_sizes(df_sorted: pd.DataFrame) -> List[int]:
    return df_sorted.groupby("group_id", sort=False).size().astype(int).tolist()


def score_with_prev_classifier(model_path: str, df: pd.DataFrame) -> Optional[np.ndarray]:
    p = Path(model_path)
    if not p.exists():
        return None
    obj = joblib.load(p)
    pipeline = obj.get("pipeline")
    feat_cols = obj.get("feature_columns")
    if pipeline is None or feat_cols is None:
        return None
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        return None
    return pipeline.predict_proba(df[feat_cols])[:, 1]


def add_rank_column(df: pd.DataFrame, score_col: str) -> pd.Series:
    ranked = (
        df.sort_values(["group_id", score_col, "sid"], ascending=[True, False, True])
        .groupby("group_id")
        .cumcount()
        .add(1)
    )
    return ranked.sort_index()


def main() -> None:
    args = parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    required = ["cache_key", "problem_id", "sid", "is_correct"] + FEATURE_COLUMNS
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[data["cache_key"].isin(TARGET_CACHE_KEYS)].copy()
    data["is_correct"] = parse_bool_series(data["is_correct"])
    data = data[~data["is_correct"].isna()].copy()
    data["is_correct"] = data["is_correct"].astype(int)

    data["group_id"] = data["cache_key"].astype(str) + "::" + data["problem_id"].astype(str)
    data["sid"] = data["sid"].astype(str)

    for col in FEATURE_COLUMNS:
        if col in BOOL_COLUMNS:
            data[col] = parse_bool_series(data[col])
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    X = data[FEATURE_COLUMNS]
    y = data["is_correct"].to_numpy(dtype=np.int32)
    groups = data["group_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.valid_size), random_state=int(args.random_state))
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    train_df = data.iloc[train_idx].copy()
    valid_df = data.iloc[valid_idx].copy()

    train_sorted = train_df.sort_values(["group_id", "sid"]).copy()
    valid_sorted = valid_df.sort_values(["group_id", "sid"]).copy()

    x_train = train_sorted[FEATURE_COLUMNS]
    y_train = train_sorted["is_correct"].to_numpy(dtype=np.int32)
    g_train = group_sizes(train_sorted)

    x_valid = valid_sorted[FEATURE_COLUMNS]
    y_valid = valid_sorted["is_correct"].to_numpy(dtype=np.int32)
    g_valid = group_sizes(valid_sorted)

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5],
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(args.random_state),
        n_jobs=4,
    )
    ranker.fit(
        x_train,
        y_train,
        group=g_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[g_valid],
        eval_at=[1, 3],
    )

    valid_df["ranker_score"] = ranker.predict(valid_df[FEATURE_COLUMNS])
    valid_df["rank_ranker_in_problem"] = add_rank_column(valid_df, "ranker_score")
    valid_df["rank_mixed_v1_in_problem"] = add_rank_column(valid_df, "mixed_v1_score")

    prev_logreg = score_with_prev_classifier(args.prev_logreg_model, valid_df)
    prev_lgbm_binary = score_with_prev_classifier(args.prev_lgbm_binary_model, valid_df)

    if prev_logreg is not None:
        valid_df["p_correct_logreg"] = prev_logreg
        valid_df["rank_logreg_in_problem"] = add_rank_column(valid_df, "p_correct_logreg")
    if prev_lgbm_binary is not None:
        valid_df["p_correct_lgbm_binary"] = prev_lgbm_binary
        valid_df["rank_lgbm_binary_in_problem"] = add_rank_column(valid_df, "p_correct_lgbm_binary")

    ranker_problem = evaluate_problem_metrics(valid_df, "ranker_score")
    ranker_auc = candidate_auc(valid_df["is_correct"].to_numpy(), valid_df["ranker_score"].to_numpy())

    comparison_rows: List[Dict[str, object]] = [
        {
            "model_name": "lightgbm_ranker",
            "valid_problem_count": ranker_problem["valid_problem_count"],
            "problem_top1_accuracy": ranker_problem["problem_top1_accuracy"],
            "mixed_v1_problem_top1_accuracy": ranker_problem["mixed_v1_problem_top1_accuracy"],
            "delta_vs_mixed_v1": ranker_problem["delta_vs_mixed_v1"],
            "candidate_auc": ranker_auc,
            "top3_accuracy": ranker_problem["top3_accuracy"],
            "pairwise_accuracy": ranker_problem["pairwise_accuracy"],
            "ndcg_at_1": ranker_problem["ndcg_at_1"],
            "ndcg_at_3": ranker_problem["ndcg_at_3"],
        }
    ]

    if prev_logreg is not None:
        m = evaluate_problem_metrics(valid_df, "p_correct_logreg")
        comparison_rows.append(
            {
                "model_name": "logistic_regression_baseline",
                "valid_problem_count": m["valid_problem_count"],
                "problem_top1_accuracy": m["problem_top1_accuracy"],
                "mixed_v1_problem_top1_accuracy": m["mixed_v1_problem_top1_accuracy"],
                "delta_vs_mixed_v1": m["delta_vs_mixed_v1"],
                "candidate_auc": candidate_auc(valid_df["is_correct"].to_numpy(), valid_df["p_correct_logreg"].to_numpy()),
                "top3_accuracy": m["top3_accuracy"],
                "pairwise_accuracy": m["pairwise_accuracy"],
                "ndcg_at_1": m["ndcg_at_1"],
                "ndcg_at_3": m["ndcg_at_3"],
            }
        )

    if prev_lgbm_binary is not None:
        m = evaluate_problem_metrics(valid_df, "p_correct_lgbm_binary")
        comparison_rows.append(
            {
                "model_name": "lightgbm_binary_baseline",
                "valid_problem_count": m["valid_problem_count"],
                "problem_top1_accuracy": m["problem_top1_accuracy"],
                "mixed_v1_problem_top1_accuracy": m["mixed_v1_problem_top1_accuracy"],
                "delta_vs_mixed_v1": m["delta_vs_mixed_v1"],
                "candidate_auc": candidate_auc(valid_df["is_correct"].to_numpy(), valid_df["p_correct_lgbm_binary"].to_numpy()),
                "top3_accuracy": m["top3_accuracy"],
                "pairwise_accuracy": m["pairwise_accuracy"],
                "ndcg_at_1": m["ndcg_at_1"],
                "ndcg_at_3": m["ndcg_at_3"],
            }
        )

    imp_gain = ranker.booster_.feature_importance(importance_type="gain")
    imp_split = ranker.booster_.feature_importance(importance_type="split")
    feature_importance = [
        {
            "feature": feat,
            "importance_gain": float(gain),
            "importance_split": int(split),
        }
        for feat, gain, split in zip(FEATURE_COLUMNS, imp_gain, imp_split)
    ]
    feature_importance.sort(key=lambda x: (-x["importance_gain"], x["feature"]))

    metrics = {
        "model_name": "lightgbm_ranker",
        "input": args.input,
        "target_cache_keys": TARGET_CACHE_KEYS,
        "features_used": FEATURE_COLUMNS,
        "split": {
            "method": "GroupShuffleSplit",
            "group_id": "cache_key::problem_id",
            "valid_size": float(args.valid_size),
            "random_state": int(args.random_state),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_problem_count": int(train_df["group_id"].nunique()),
            "valid_problem_count": int(valid_df["group_id"].nunique()),
        },
        "group_array": {
            "train_group_count": int(len(g_train)),
            "valid_group_count": int(len(g_valid)),
            "train_group_size_min": int(min(g_train)) if g_train else 0,
            "train_group_size_max": int(max(g_train)) if g_train else 0,
            "valid_group_size_min": int(min(g_valid)) if g_valid else 0,
            "valid_group_size_max": int(max(g_valid)) if g_valid else 0,
        },
        "candidate_metrics": {
            "candidate_auc": ranker_auc,
        },
        "problem_metrics": ranker_problem,
        "feature_importance_top20": feature_importance[:20],
        "classifier_baselines_loaded": {
            "logreg": bool(prev_logreg is not None),
            "lgbm_binary": bool(prev_lgbm_binary is not None),
        },
    }

    comp_df = pd.DataFrame(comparison_rows)

    out_paths = [args.output_metrics, args.output_valid_predictions, args.output_comparison, args.output_model]
    for p in out_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    Path(args.output_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    comp_df.to_csv(args.output_comparison, index=False)

    valid_cols = [
        "cache_key",
        "problem_id",
        "group_id",
        "sid",
        "is_correct",
        "mixed_v1_score",
        "ranker_score",
        "rank_mixed_v1_in_problem",
        "rank_ranker_in_problem",
    ]
    for c in ["p_correct_logreg", "rank_logreg_in_problem", "p_correct_lgbm_binary", "rank_lgbm_binary_in_problem"]:
        if c in valid_df.columns:
            valid_cols.append(c)
    valid_cols.extend(FEATURE_COLUMNS)
    valid_df[valid_cols].to_csv(args.output_valid_predictions, index=False)

    joblib.dump(
        {
            "model_name": "lightgbm_ranker",
            "feature_columns": FEATURE_COLUMNS,
            "model": ranker,
            "split": metrics["split"],
        },
        args.output_model,
    )

    print(f"train_rows={len(train_df)} valid_rows={len(valid_df)}")
    print(f"train_groups={len(g_train)} valid_groups={len(g_valid)}")
    print(f"ranker_problem_top1_accuracy={ranker_problem['problem_top1_accuracy']:.6f}")
    print(f"ranker_delta_vs_mixed_v1={ranker_problem['delta_vs_mixed_v1']:.6f}")
    print(f"wrote {args.output_metrics}")
    print(f"wrote {args.output_valid_predictions}")
    print(f"wrote {args.output_comparison}")
    print(f"wrote {args.output_model}")


if __name__ == "__main__":
    main()
