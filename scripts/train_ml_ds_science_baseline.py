#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science.csv"
DEFAULT_OUTPUT_LOGREG_METRICS = "/home/jovyan/work/NAD_Next/result/ml_ds_science_baseline_logreg_metrics.json"
DEFAULT_OUTPUT_LGBM_METRICS = "/home/jovyan/work/NAD_Next/result/ml_ds_science_baseline_lgbm_metrics.json"
DEFAULT_OUTPUT_COMPARISON = "/home/jovyan/work/NAD_Next/result/ml_ds_science_baseline_comparison.csv"
DEFAULT_OUTPUT_VALID_PRED = "/home/jovyan/work/NAD_Next/result/ml_ds_science_valid_predictions.csv"
DEFAULT_OUTPUT_LOGREG_MODEL = "/home/jovyan/work/NAD_Next/result/ml_ds_science_logreg.pkl"
DEFAULT_OUTPUT_LGBM_MODEL = "/home/jovyan/work/NAD_Next/result/ml_ds_science_lgbm.pkl"

TARGET_DS_LABELLED_CACHE_KEYS = [
    "DS-R1/brumo25",
    "DS-R1/gpqa",
    "DS-R1/hmmt25",
]

PREFERRED_FEATURES = [
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
    p = argparse.ArgumentParser(description="Train DS-science candidate-level binary baselines (logreg + lightgbm).")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-logreg-metrics", default=DEFAULT_OUTPUT_LOGREG_METRICS)
    p.add_argument("--output-lgbm-metrics", default=DEFAULT_OUTPUT_LGBM_METRICS)
    p.add_argument("--output-comparison", default=DEFAULT_OUTPUT_COMPARISON)
    p.add_argument("--output-valid-predictions", default=DEFAULT_OUTPUT_VALID_PRED)
    p.add_argument("--output-logreg-model", default=DEFAULT_OUTPUT_LOGREG_MODEL)
    p.add_argument("--output-lgbm-model", default=DEFAULT_OUTPUT_LGBM_MODEL)
    p.add_argument("--valid-size", type=float, default=0.2, help="Fraction of problem groups for validation.")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def parse_bool_series(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    out = []
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


def parse_label_series(series: pd.Series) -> pd.Series:
    return parse_bool_series(series).astype("float64")


def candidate_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(p, dtype=np.float64)
    pred = (probs >= 0.5).astype(np.int32)

    metrics: Dict[str, float] = {
        "candidate_accuracy": float(accuracy_score(y, pred)),
    }

    if len(np.unique(y)) >= 2:
        metrics["candidate_auc"] = float(roc_auc_score(y, probs))
        metrics["candidate_logloss"] = float(log_loss(y, probs, labels=[0, 1]))
    else:
        metrics["candidate_auc"] = float("nan")
        metrics["candidate_logloss"] = float("nan")
    return metrics


def topk_hit_accuracy(df: pd.DataFrame, score_col: str, k: int) -> float:
    if df.empty:
        return float("nan")
    hits = []
    for _, grp in df.groupby("group_id", sort=False):
        ranked = grp.sort_values([score_col, "mixed_v1_score", "sid"], ascending=[False, False, True])
        topk = ranked.head(k)
        hits.append(int(topk["is_correct"].astype(int).max() > 0))
    return float(np.mean(hits)) if hits else float("nan")


def problem_top1_accuracy(df: pd.DataFrame, score_col: str) -> float:
    if df.empty:
        return float("nan")
    chosen = (
        df.sort_values(["group_id", score_col, "sid"], ascending=[True, False, True])
        .groupby("group_id", as_index=False)
        .head(1)
    )
    return float(chosen["is_correct"].astype(int).mean())


def evaluate_problem_level(valid_df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    model_top1 = problem_top1_accuracy(valid_df, score_col)
    mixed_top1 = problem_top1_accuracy(valid_df, "mixed_v1_score")
    return {
        "valid_problem_count": int(valid_df["group_id"].nunique()),
        "problem_top1_accuracy": model_top1,
        "mixed_v1_problem_top1_accuracy": mixed_top1,
        "delta_vs_mixed_v1": float(model_top1 - mixed_top1),
        "problem_top3_accuracy": topk_hit_accuracy(valid_df, score_col, k=3),
        "mixed_v1_problem_top3_accuracy": topk_hit_accuracy(valid_df, "mixed_v1_score", k=3),
    }


def build_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rank_logreg = (
        out.sort_values(["group_id", "p_correct_logreg", "sid"], ascending=[True, False, True])
        .groupby("group_id")
        .cumcount()
        .add(1)
    )
    rank_lgbm = (
        out.sort_values(["group_id", "p_correct_lgbm", "sid"], ascending=[True, False, True])
        .groupby("group_id")
        .cumcount()
        .add(1)
    )
    rank_mixed = (
        out.sort_values(["group_id", "mixed_v1_score", "sid"], ascending=[True, False, True])
        .groupby("group_id")
        .cumcount()
        .add(1)
    )

    out["rank_logreg_in_problem"] = rank_logreg.sort_index()
    out["rank_lgbm_in_problem"] = rank_lgbm.sort_index()
    out["rank_mixed_v1_in_problem"] = rank_mixed.sort_index()
    out["is_logreg_top1"] = (out["rank_logreg_in_problem"] == 1).astype(int)
    out["is_lgbm_top1"] = (out["rank_lgbm_in_problem"] == 1).astype(int)
    out["is_mixed_v1_top1"] = (out["rank_mixed_v1_in_problem"] == 1).astype(int)
    return out


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input, low_memory=False)

    required_cols = ["cache_key", "problem_id", "sid", "is_correct", "mixed_v1_score"]
    missing_required = [c for c in required_cols if c not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    data = data[data["cache_key"].isin(TARGET_DS_LABELLED_CACHE_KEYS)].copy()
    data["is_correct"] = parse_label_series(data["is_correct"])
    data = data[~data["is_correct"].isna()].copy()
    data["is_correct"] = data["is_correct"].astype(int)

    if data.empty:
        raise ValueError("No labelled DS-science rows after filtering.")

    data["group_id"] = data["cache_key"].astype(str) + "::" + data["problem_id"].astype(str)
    data["sid"] = data["sid"].astype(str)

    available_features = [c for c in PREFERRED_FEATURES if c in data.columns]
    dropped_feature_columns = [c for c in PREFERRED_FEATURES if c not in data.columns]

    for col in available_features:
        if col in BOOL_COLUMNS:
            data[col] = parse_bool_series(data[col])
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    X = data[available_features].copy()
    y = data["is_correct"].astype(int).to_numpy()
    groups = data["group_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.valid_size), random_state=int(args.random_state))
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    train_df = data.iloc[train_idx].copy()
    valid_df = data.iloc[valid_idx].copy()

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=int(args.random_state),
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    logreg.fit(X_train, y_train)
    p_logreg = logreg.predict_proba(X_valid)[:, 1]

    lgbm = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    learning_rate=0.05,
                    n_estimators=300,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=int(args.random_state),
                    n_jobs=4,
                ),
            ),
        ]
    )
    lgbm.fit(X_train, y_train)
    p_lgbm = lgbm.predict_proba(X_valid)[:, 1]

    valid_df["p_correct_logreg"] = p_logreg
    valid_df["p_correct_lgbm"] = p_lgbm
    valid_df = build_rank_columns(valid_df)

    logreg_cand = candidate_metrics(y_valid, p_logreg)
    lgbm_cand = candidate_metrics(y_valid, p_lgbm)
    logreg_prob = evaluate_problem_level(valid_df, "p_correct_logreg")
    lgbm_prob = evaluate_problem_level(valid_df, "p_correct_lgbm")

    split_info = {
        "method": "GroupShuffleSplit",
        "valid_size": float(args.valid_size),
        "random_state": int(args.random_state),
        "group_key": "cache_key::problem_id",
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_problem_count": int(train_df["group_id"].nunique()),
        "valid_problem_count": int(valid_df["group_id"].nunique()),
    }

    logreg_metrics = {
        "model_name": "logistic_regression",
        "input": args.input,
        "target_cache_keys": TARGET_DS_LABELLED_CACHE_KEYS,
        "features_used": available_features,
        "dropped_feature_columns_not_found": dropped_feature_columns,
        "split": split_info,
        "candidate_metrics": logreg_cand,
        "problem_metrics": logreg_prob,
    }

    lgbm_metrics = {
        "model_name": "lightgbm_binary",
        "input": args.input,
        "target_cache_keys": TARGET_DS_LABELLED_CACHE_KEYS,
        "features_used": available_features,
        "dropped_feature_columns_not_found": dropped_feature_columns,
        "split": split_info,
        "candidate_metrics": lgbm_cand,
        "problem_metrics": lgbm_prob,
    }

    comparison_rows = [
        {
            "model_name": "logistic_regression",
            "train_rows": split_info["train_rows"],
            "valid_rows": split_info["valid_rows"],
            "valid_problem_count": split_info["valid_problem_count"],
            "candidate_auc": logreg_cand["candidate_auc"],
            "problem_top1_accuracy": logreg_prob["problem_top1_accuracy"],
            "mixed_v1_problem_top1_accuracy": logreg_prob["mixed_v1_problem_top1_accuracy"],
            "delta_vs_mixed_v1": logreg_prob["delta_vs_mixed_v1"],
        },
        {
            "model_name": "lightgbm_binary",
            "train_rows": split_info["train_rows"],
            "valid_rows": split_info["valid_rows"],
            "valid_problem_count": split_info["valid_problem_count"],
            "candidate_auc": lgbm_cand["candidate_auc"],
            "problem_top1_accuracy": lgbm_prob["problem_top1_accuracy"],
            "mixed_v1_problem_top1_accuracy": lgbm_prob["mixed_v1_problem_top1_accuracy"],
            "delta_vs_mixed_v1": lgbm_prob["delta_vs_mixed_v1"],
        },
    ]

    out_paths = [
        args.output_logreg_metrics,
        args.output_lgbm_metrics,
        args.output_comparison,
        args.output_valid_predictions,
        args.output_logreg_model,
        args.output_lgbm_model,
    ]
    for path in out_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    Path(args.output_logreg_metrics).write_text(json.dumps(logreg_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_lgbm_metrics).write_text(json.dumps(lgbm_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(args.output_comparison, index=False)

    valid_cols_base = [
        "cache_key",
        "problem_id",
        "group_id",
        "sid",
        "is_correct",
        "mixed_v1_score",
        "p_correct_logreg",
        "p_correct_lgbm",
        "rank_mixed_v1_in_problem",
        "rank_logreg_in_problem",
        "rank_lgbm_in_problem",
        "is_mixed_v1_top1",
        "is_logreg_top1",
        "is_lgbm_top1",
    ]
    valid_cols = valid_cols_base + [c for c in available_features if c not in set(valid_cols_base)]
    valid_df[valid_cols].to_csv(args.output_valid_predictions, index=False)

    joblib.dump(
        {
            "model_name": "logistic_regression",
            "feature_columns": available_features,
            "pipeline": logreg,
        },
        args.output_logreg_model,
    )
    joblib.dump(
        {
            "model_name": "lightgbm_binary",
            "feature_columns": available_features,
            "pipeline": lgbm,
        },
        args.output_lgbm_model,
    )

    print(f"train_rows={split_info['train_rows']} valid_rows={split_info['valid_rows']}")
    print(f"valid_problem_count={split_info['valid_problem_count']}")
    print(f"wrote {args.output_logreg_metrics}")
    print(f"wrote {args.output_lgbm_metrics}")
    print(f"wrote {args.output_comparison}")
    print(f"wrote {args.output_valid_predictions}")
    print(f"wrote {args.output_logreg_model}")
    print(f"wrote {args.output_lgbm_model}")


if __name__ == "__main__":
    main()
