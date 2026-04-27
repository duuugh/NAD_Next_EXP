#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = "/home/jovyan/work/NAD_Next/result/ml_candidate_table_v1_science.csv"
DEFAULT_FOLD_METRICS = "/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_audit_fold_metrics.csv"
DEFAULT_IMPORTANCE_SUMMARY = "/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_importance_summary.csv"
DEFAULT_FAMILY_ABLATION = "/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_family_ablation.csv"
DEFAULT_SUMMARY_JSON = "/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_audit_summary.json"
DEFAULT_IMPORTANCE_BY_FOLD = "/home/jovyan/work/NAD_Next/result/ml_ds_science_feature_importance_by_fold.csv"

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

FAMILIES = {
    "baseline": [
        "mixed_v1_score",
        "rank_by_mixed_v1",
        "top1_gap_under_mixed_v1",
        "top2_gap_under_mixed_v1",
    ],
    "token": [
        "tok_logprob_mean",
        "tok_selfcert_mean",
        "tok_conf_mean",
        "tok_neg_entropy_mean",
        "tok_gini_mean",
    ],
    "structure": [
        "answer_length_tokens",
        "parse_success",
        "is_integer_answer",
        "num_candidates_for_problem",
        "unique_answer_count_in_problem",
    ],
    "activation": [
        "tail_warning",
        "tail_new_ratio",
        "plateau_progress",
        "cumulative_unique_neurons_end",
    ],
}

ABLATION_CONFIGS = {
    "baseline_only": FAMILIES["baseline"],
    "baseline_plus_token": FAMILIES["baseline"] + FAMILIES["token"],
    "baseline_plus_structure": FAMILIES["baseline"] + FAMILIES["structure"],
    "baseline_plus_activation": FAMILIES["baseline"] + FAMILIES["activation"],
    "baseline_plus_token_structure": FAMILIES["baseline"] + FAMILIES["token"] + FAMILIES["structure"],
    "baseline_plus_all": FEATURE_COLUMNS,
    "activation_only": FAMILIES["activation"],
    "token_structure_only": FAMILIES["token"] + FAMILIES["structure"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit DS-science features with multi-fold stability and family ablations.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--fold-metrics-output", default=DEFAULT_FOLD_METRICS)
    p.add_argument("--importance-summary-output", default=DEFAULT_IMPORTANCE_SUMMARY)
    p.add_argument("--family-ablation-output", default=DEFAULT_FAMILY_ABLATION)
    p.add_argument("--summary-output", default=DEFAULT_SUMMARY_JSON)
    p.add_argument("--importance-by-fold-output", default=DEFAULT_IMPORTANCE_BY_FOLD)
    p.add_argument("--n-splits", type=int, default=5)
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


def safe_candidate_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= 0.5).astype(np.int32)

    out = {
        "candidate_accuracy": float(accuracy_score(y, pred)),
        "candidate_auc": float("nan"),
        "candidate_logloss": float("nan"),
    }
    if len(np.unique(y)) >= 2:
        out["candidate_auc"] = float(roc_auc_score(y, s))
        out["candidate_logloss"] = float(log_loss(y, s, labels=[0, 1]))
    return out


def problem_top1_accuracy(df: pd.DataFrame, score_col: str) -> float:
    top1 = (
        df.sort_values(["group_id", score_col, "sid"], ascending=[True, False, True])
        .groupby("group_id", as_index=False)
        .head(1)
    )
    return float(top1["is_correct"].astype(int).mean()) if not top1.empty else float("nan")


def evaluate_problem_level(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    model_acc = problem_top1_accuracy(df, score_col)
    baseline_acc = problem_top1_accuracy(df, "mixed_v1_score")
    return {
        "valid_problem_count": int(df["group_id"].nunique()),
        "problem_top1_accuracy": model_acc,
        "mixed_v1_problem_top1_accuracy": baseline_acc,
        "delta_vs_mixed_v1": float(model_acc - baseline_acc),
    }


def make_logreg(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=int(random_state),
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_lgbm_binary(random_state: int) -> Pipeline:
    return Pipeline(
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
                    random_state=int(random_state),
                    n_jobs=4,
                ),
            ),
        ]
    )


def get_logreg_importance(model: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    coef = model.named_steps["model"].coef_[0]
    return {f: float(abs(v)) for f, v in zip(feature_cols, coef)}


def get_lgbm_importance(model: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    booster = model.named_steps["model"].booster_
    gain = booster.feature_importance(importance_type="gain")
    return {f: float(v) for f, v in zip(feature_cols, gain)}


def aggregate_importance(by_fold: pd.DataFrame) -> pd.DataFrame:
    df = by_fold.copy()
    df["rank_in_fold"] = (
        df.groupby(["model_name", "fold"])["importance"].rank(method="average", ascending=False)
    )
    df["is_top5"] = (df["rank_in_fold"] <= 5).astype(float)

    out = (
        df.groupby(["model_name", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_rank=("rank_in_fold", "mean"),
            std_rank=("rank_in_fold", "std"),
            top5_freq=("is_top5", "mean"),
            folds_present=("fold", "nunique"),
        )
        .sort_values(["model_name", "mean_rank", "feature"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return out


def run_cv_baselines(
    data: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
    X = data[feature_cols]
    y = data["is_correct"].astype(int).to_numpy()
    groups = data["group_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        splits.append((train_idx, valid_idx))

        train_df = data.iloc[train_idx].copy()
        valid_df = data.iloc[valid_idx].copy()
        x_train = train_df[feature_cols]
        y_train = train_df["is_correct"].astype(int).to_numpy()
        x_valid = valid_df[feature_cols]
        y_valid = valid_df["is_correct"].astype(int).to_numpy()

        # Logistic regression
        logreg = make_logreg(random_state=random_state + fold_idx)
        logreg.fit(x_train, y_train)
        p_logreg = logreg.predict_proba(x_valid)[:, 1]
        valid_df["score_model"] = p_logreg

        cand_logreg = safe_candidate_metrics(y_valid, p_logreg)
        prob_logreg = evaluate_problem_level(valid_df, "score_model")
        fold_metrics_rows.append(
            {
                "model_name": "logistic_regression",
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                **cand_logreg,
                **prob_logreg,
            }
        )

        for feat, imp in get_logreg_importance(logreg, feature_cols).items():
            importance_rows.append(
                {
                    "model_name": "logistic_regression",
                    "fold": fold_idx,
                    "feature": feat,
                    "importance": imp,
                }
            )

        # LightGBM binary
        lgbm = make_lgbm_binary(random_state=random_state + fold_idx)
        lgbm.fit(x_train, y_train)
        p_lgbm = lgbm.predict_proba(x_valid)[:, 1]
        valid_df["score_model"] = p_lgbm

        cand_lgbm = safe_candidate_metrics(y_valid, p_lgbm)
        prob_lgbm = evaluate_problem_level(valid_df, "score_model")
        fold_metrics_rows.append(
            {
                "model_name": "lightgbm_binary",
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                **cand_lgbm,
                **prob_lgbm,
            }
        )

        for feat, imp in get_lgbm_importance(lgbm, feature_cols).items():
            importance_rows.append(
                {
                    "model_name": "lightgbm_binary",
                    "fold": fold_idx,
                    "feature": feat,
                    "importance": imp,
                }
            )

    return pd.DataFrame(fold_metrics_rows), pd.DataFrame(importance_rows), splits


def run_family_ablation(
    data: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for combo_name, combo_features in ABLATION_CONFIGS.items():
        for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
            train_df = data.iloc[train_idx].copy()
            valid_df = data.iloc[valid_idx].copy()

            x_train = train_df[combo_features]
            y_train = train_df["is_correct"].astype(int).to_numpy()
            x_valid = valid_df[combo_features]
            y_valid = valid_df["is_correct"].astype(int).to_numpy()

            model = make_lgbm_binary(random_state=random_state + 100 + fold_idx)
            model.fit(x_train, y_train)
            p = model.predict_proba(x_valid)[:, 1]
            valid_df["score_model"] = p

            cand = safe_candidate_metrics(y_valid, p)
            prob = evaluate_problem_level(valid_df, "score_model")

            rows.append(
                {
                    "row_type": "fold",
                    "model_name": "lightgbm_binary",
                    "feature_combo": combo_name,
                    "fold": fold_idx,
                    "feature_count": len(combo_features),
                    "features": "|".join(combo_features),
                    **cand,
                    **prob,
                }
            )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["model_name", "feature_combo"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            feature_count=("feature_count", "first"),
            candidate_auc_mean=("candidate_auc", "mean"),
            candidate_auc_std=("candidate_auc", "std"),
            problem_top1_accuracy_mean=("problem_top1_accuracy", "mean"),
            problem_top1_accuracy_std=("problem_top1_accuracy", "std"),
            mixed_v1_problem_top1_accuracy_mean=("mixed_v1_problem_top1_accuracy", "mean"),
            delta_vs_mixed_v1_mean=("delta_vs_mixed_v1", "mean"),
            delta_vs_mixed_v1_std=("delta_vs_mixed_v1", "std"),
        )
        .sort_values(["problem_top1_accuracy_mean", "candidate_auc_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["row_type"] = "summary"

    return pd.concat([df, summary], ignore_index=True, sort=False)


def model_summary(fold_df: pd.DataFrame) -> List[Dict[str, object]]:
    out = []
    for model_name, sub in fold_df.groupby("model_name"):
        out.append(
            {
                "model_name": model_name,
                "folds": int(sub["fold"].nunique()),
                "candidate_auc_mean": float(sub["candidate_auc"].mean()),
                "candidate_auc_std": float(sub["candidate_auc"].std()),
                "problem_top1_accuracy_mean": float(sub["problem_top1_accuracy"].mean()),
                "problem_top1_accuracy_std": float(sub["problem_top1_accuracy"].std()),
                "mixed_v1_problem_top1_accuracy_mean": float(sub["mixed_v1_problem_top1_accuracy"].mean()),
                "delta_vs_mixed_v1_mean": float(sub["delta_vs_mixed_v1"].mean()),
                "delta_vs_mixed_v1_std": float(sub["delta_vs_mixed_v1"].std()),
            }
        )
    return out


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
    data["sid"] = data["sid"].astype(str)
    data["group_id"] = data["cache_key"].astype(str) + "::" + data["problem_id"].astype(str)

    for col in FEATURE_COLUMNS:
        if col in BOOL_COLUMNS:
            data[col] = parse_bool_series(data[col])
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    fold_metrics_df, importance_by_fold_df, splits = run_cv_baselines(
        data=data,
        feature_cols=FEATURE_COLUMNS,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
    )

    importance_summary_df = aggregate_importance(importance_by_fold_df)
    family_ablation_df = run_family_ablation(
        data=data,
        splits=splits,
        random_state=int(args.random_state),
    )

    for p in [
        args.fold_metrics_output,
        args.importance_summary_output,
        args.family_ablation_output,
        args.summary_output,
        args.importance_by_fold_output,
    ]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    fold_metrics_df.to_csv(args.fold_metrics_output, index=False)
    importance_summary_df.to_csv(args.importance_summary_output, index=False)
    family_ablation_df.to_csv(args.family_ablation_output, index=False)
    importance_by_fold_df.to_csv(args.importance_by_fold_output, index=False)

    top_logreg = importance_summary_df[importance_summary_df["model_name"] == "logistic_regression"].head(8)
    top_lgbm = importance_summary_df[importance_summary_df["model_name"] == "lightgbm_binary"].head(8)

    ablation_summary = family_ablation_df[family_ablation_df["row_type"] == "summary"].copy()
    ablation_summary = ablation_summary.sort_values(
        ["problem_top1_accuracy_mean", "candidate_auc_mean"], ascending=[False, False]
    )

    summary = {
        "task": "ml_ds_science_feature_audit",
        "input": args.input,
        "target_cache_keys": TARGET_CACHE_KEYS,
        "n_splits": int(args.n_splits),
        "group_id": "cache_key::problem_id",
        "data_shape": {
            "rows": int(len(data)),
            "problems": int(data["group_id"].nunique()),
            "cache_breakdown": data.groupby("cache_key").size().astype(int).to_dict(),
        },
        "baseline_cv_summary": model_summary(fold_metrics_df),
        "top_features": {
            "logistic_regression": top_logreg.to_dict(orient="records"),
            "lightgbm_binary": top_lgbm.to_dict(orient="records"),
        },
        "family_ablation_summary": ablation_summary.to_dict(orient="records"),
        "outputs": {
            "fold_metrics_csv": args.fold_metrics_output,
            "importance_summary_csv": args.importance_summary_output,
            "family_ablation_csv": args.family_ablation_output,
            "importance_by_fold_csv": args.importance_by_fold_output,
        },
    }

    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={len(data)} problems={data['group_id'].nunique()}")
    print(f"wrote {args.fold_metrics_output}")
    print(f"wrote {args.importance_summary_output}")
    print(f"wrote {args.family_ablation_output}")
    print(f"wrote {args.importance_by_fold_output}")
    print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
