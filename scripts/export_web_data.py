#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import math
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
WEB_PUBLIC_DATA_DIR = ROOT / "web" / "public" / "data"
WEB_FILES_DIR = WEB_PUBLIC_DATA_DIR / "files"
sys.path.insert(0, str(ROOT))
from nad.io import build_problem_catalog, load_nad_next_index, NadNextLoader


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_artifact(source: Path, group: str) -> dict[str, str]:
    source = source.resolve()
    target_dir = WEB_FILES_DIR / group
    ensure_dir(target_dir)
    target_path = target_dir / source.name
    shutil.copy2(source, target_path)
    return {
        "label": source.name,
        "originalPath": rel_repo_path(source),
        "publicPath": f"data/files/{group}/{source.name}",
    }


def copy_visual(source: Path, group: str, title: str, caption: str | None = None) -> dict[str, str]:
    source = source.resolve()
    target_dir = WEB_FILES_DIR / group
    ensure_dir(target_dir)
    target_path = target_dir / source.name
    shutil.copy2(source, target_path)
    visual = {
        "title": title,
        "publicPath": f"data/files/{group}/{source.name}",
    }
    if caption:
        visual["caption"] = caption
    return visual


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in {None, ""}:
                    parsed[key] = None
                    continue
                lowered = value.lower()
                if lowered == "true":
                    parsed[key] = True
                    continue
                if lowered == "false":
                    parsed[key] = False
                    continue
                try:
                    number = float(value)
                    parsed[key] = int(number) if number.is_integer() else number
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def average_numeric_dicts(items: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = OrderedDict()
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                keys[key] = None
    out: dict[str, float | None] = {}
    for key in keys:
        values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float)) and not math.isnan(float(item[key]))]
        out[key] = mean(values) if values else None
    return out


def summarize_candidate_scores(score_map: dict[str, dict[str, float]]) -> dict[str, float]:
    flattened = [float(score) for sid_scores in score_map.values() for score in sid_scores.values()]
    if not flattened:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "score_range": 0.0,
            "problem_count": 0.0,
            "candidate_count": 0.0,
        }
    return {
        "mean_score": mean(flattened),
        "std_score": pstdev(flattened) if len(flattened) > 1 else 0.0,
        "max_score": max(flattened),
        "min_score": min(flattened),
        "score_range": max(flattened) - min(flattened),
        "problem_count": float(len(score_map)),
        "candidate_count": float(sum(len(sid_scores) for sid_scores in score_map.values())),
    }


def select_note_lines(payload: dict[str, Any], keys: list[str]) -> list[str]:
    lines: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            preview = ", ".join(str(item) for item in value[:6])
            lines.append(f"{key}: {preview}")
        elif isinstance(value, dict):
            preview = ", ".join(f"{k}={v}" for k, v in list(value.items())[:4])
            lines.append(f"{key}: {preview}")
    return lines


def build_early_stop_cards() -> dict[str, Any]:
    variants_eval = read_json(RESULT_DIR / "early_stop_mean_confidence_variants_eval.json")
    dynamics_summary = read_json(RESULT_DIR / "dynamics_v2_local_batch_summary.json")
    router_policy = read_json(RESULT_DIR / "dynamics_benchmark_router.json")
    router_notes = read_json(RESULT_DIR / "early_stop_dynamics_router_conservative_submit_notes.json")
    trimmed_report = read_json(RESULT_DIR / "early_stop_mean_confidence_plus_dyn_conservative_trimmed_mean_logprob_report.json")

    confidence_only = variants_eval["evaluations"]["confidence_only"]
    confidence_plus = variants_eval["evaluations"]["confidence_plus_dynamics_conservative"]
    delta_plus = variants_eval["delta"]["confidence_plus_minus_confidence_only"]

    dynamics_per_cache: dict[str, dict[str, float | None]] = {}
    for cache_key, payload in dynamics_summary["per_benchmark"].items():
        recommended_mode = payload["decision"]["recommended_mode"]
        metrics = payload["modes"][recommended_mode]["metrics"]
        dynamics_per_cache[cache_key] = {
            "AUC-AUROC": metrics.get("AUC-AUROC"),
            "AUC-SelAcc": metrics.get("AUC-SelAcc"),
            "Stop@100%": metrics.get("Stop@100"),
            "AUROC@100%": payload["modes"][recommended_mode]["budget_curves"][-1]["AUROC"],
        }
    dynamics_overall = average_numeric_dicts(list(dynamics_per_cache.values()))

    cards = [
        {
            "id": "early_stop_router_final",
            "task": "early_stop",
            "title": "early_stop_dynamics_router_conservative_submit",
            "shortDescription": "最终提交路线：按 benchmark 做 conservative router，仅对确认稳定收益的 DS-R1 cache 启用 dynamics plugin。",
            "status": "final_best",
            "sourceFiles": [
                "result/early_stop_dynamics_router_conservative_submit.json",
                "result/early_stop_dynamics_router_conservative_submit_notes.json",
                "result/dynamics_benchmark_router.json",
                "result/early_stop_mean_confidence_variants_eval.json",
            ],
            "overall": confidence_plus["overall"],
            "perCache": confidence_plus["per_cache"],
            "notes": [
                "offline 指标引用 conservative confidence+dynamics 对照评估，用于页面统一比较。",
                "主提交 route: aime24 -> rho_tail_plus_neg_A_accel",
                "主提交 route: aime25 -> neg_A_accel_only",
                "主提交 route: hmmt25 -> rho_tail_plus_neg_A_accel",
                "DS-R1/brumo25, DS-R1/gpqa, DS-R1/lcb_v5 保守禁用。",
            ],
            "tags": ["final", "router", "conservative"],
            "timelineGroup": "early_stop",
        },
        {
            "id": "early_stop_confidence_only",
            "task": "early_stop",
            "title": "early_stop_mean_confidence",
            "shortDescription": "纯 confidence 主线，作为后续所有 dynamics / router 路线的离线比较基线。",
            "status": "important",
            "sourceFiles": [
                "result/early_stop_mean_confidence.json",
                "result/early_stop_mean_confidence_report.json",
                "result/early_stop_mean_confidence_variants_eval.json",
            ],
            "overall": confidence_only["overall"],
            "perCache": confidence_only["per_cache"],
            "notes": [
                "baseline: prefix_mean_logprob + cache_budget_quantile",
                "提供完整 labeled cache 对照指标。",
            ],
            "tags": ["baseline", "confidence"],
            "timelineGroup": "early_stop",
        },
        {
            "id": "early_stop_dynamics_local",
            "task": "early_stop",
            "title": "early_stop_dynamics_v2_local",
            "shortDescription": "局部 dynamics plugin 基准测试；先按 benchmark 审核覆盖与 delta，再决定是否进入 router。",
            "status": "important",
            "sourceFiles": [
                "result/early_stop_dynamics_v2_local.json",
                "result/early_stop_dynamics_v2_local_report.json",
                "result/dynamics_v2_local_batch_summary.json",
            ],
            "overall": dynamics_overall,
            "perCache": dynamics_per_cache,
            "notes": [
                dynamics_summary["final_recommendation_text"],
                "该卡片展示每个 benchmark 推荐 mode 的离线指标。",
            ],
            "tags": ["dynamics", "local-eval"],
            "timelineGroup": "early_stop",
            "visuals": [
                dict(copy_visual(RESULT_DIR / "dynamics_full/phase_plane.png", "visuals", "Dynamics Phase Plane", "展示动力学状态在相平面上的轨迹与分布。"), interactiveKind="dynamics", interactiveDataPath="data/dynamics_interactive.json", interactiveKey="phase_plane"),
                dict(copy_visual(RESULT_DIR / "dynamics_full/rho_timeseries.png", "visuals", "Rho Time Series", "展示关键 dynamics 信号随 token 位置变化。"), interactiveKind="dynamics", interactiveDataPath="data/dynamics_interactive.json", interactiveKey="rho_timeseries"),
                dict(copy_visual(RESULT_DIR / "dynamics_full/statistics_comparison.png", "visuals", "Statistics Comparison", "对比不同 benchmark / mode 下的统计量变化。"), interactiveKind="dynamics", interactiveDataPath="data/dynamics_interactive.json", interactiveKey="statistics_comparison"),
                dict(copy_visual(RESULT_DIR / "dynamics_full/param_ratios.png", "visuals", "Parameter Ratios", "展示不同参数比率在 dynamics 分析中的差异。"), interactiveKind="dynamics", interactiveDataPath="data/dynamics_interactive.json", interactiveKey="param_ratios"),
                dict(copy_visual(RESULT_DIR / "dynamics_full/coherence_comparison.png", "visuals", "Coherence Comparison", "展示 coherence 相关指标的对比。"), interactiveKind="dynamics", interactiveDataPath="data/dynamics_interactive.json", interactiveKey="coherence_comparison"),
            ],
        },
        {
            "id": "early_stop_confidence_plus_dynamics",
            "task": "early_stop",
            "title": "early_stop_mean_confidence_plus_dyn_conservative",
            "shortDescription": "在 confidence 主线上叠加 conservative dynamics policy，验证 selective enabling 是否比全局启用更稳。",
            "status": "important",
            "sourceFiles": [
                "result/early_stop_mean_confidence_plus_dyn_conservative.json",
                "result/early_stop_mean_confidence_plus_dyn_conservative_report.json",
                "result/early_stop_mean_confidence_variants_eval.json",
            ],
            "overall": confidence_plus["overall"],
            "perCache": confidence_plus["per_cache"],
            "notes": [
                "only_expected_changed = True",
                f"policy_enabled_caches: {', '.join(variants_eval['plugin_checks']['policy_enabled_caches'])}",
                f"overall ΔAUC-AUROC: {delta_plus['overall']['AUC-AUROC']:.4f}",
                f"overall ΔAUC-SelAcc: {delta_plus['overall']['AUC-SelAcc']:.4f}",
            ],
            "tags": ["confidence", "dynamics", "conservative"],
            "timelineGroup": "early_stop",
        },
        {
            "id": "early_stop_trimmed_logprob",
            "task": "early_stop",
            "title": "early_stop_mean_confidence_plus_dyn_conservative_trimmed_mean_logprob",
            "shortDescription": "trimmed mean logprob 版本，说明 aggregation 有增益，但最终展示主线仍保守落在 router 策略而非继续扩大 patch。",
            "status": "promising",
            "sourceFiles": [
                "result/early_stop_mean_confidence_plus_dyn_conservative_trimmed_mean_logprob.json",
                "result/early_stop_mean_confidence_plus_dyn_conservative_trimmed_mean_logprob_report.json",
            ],
            "overall": {
                "enabled_cache_count": 3,
                "disabled_cache_count": 9,
            },
            "perCache": {
                cache_key: {
                    "mean_abs_delta": delta["mean_abs_delta"],
                    "max_abs_delta": delta["max_abs_delta"],
                    "change_rate": delta["changed_samples"] / delta["total_samples"] if delta["total_samples"] else 0.0,
                }
                for cache_key, delta in trimmed_report["summary"]["cache_deltas"].items()
            },
            "notes": [
                "适合作为 aggregation 方向证据，不单独替换保守主提交叙事。",
                "page 图表主看上面三张可直接对齐 AUC 指标的卡片。",
            ],
            "tags": ["trimmed", "aggregation"],
            "timelineGroup": "early_stop",
        },
        {
            "id": "early_stop_v6_stable",
            "task": "early_stop",
            "title": "early_stop_v6_1_stable",
            "shortDescription": "中间阶段稳定版，具备工程演进价值，但不是最终采用路线。",
            "status": "deprecated",
            "sourceFiles": [
                "result/early_stop_v6_1_stable.json",
                "result/early_stop_v6_1_stable_report.json",
            ],
            "notes": [
                "保留为历史节点，用于说明从 smoke/stable 到 benchmark-selective router 的演进。",
            ],
            "tags": ["legacy"],
            "timelineGroup": "early_stop",
        },
    ]

    return {
        "generatedAt": now_iso(),
        "task": "early_stop",
        "finalBestId": "early_stop_router_final",
        "highlightedIds": [
            "early_stop_router_final",
            "early_stop_confidence_only",
            "early_stop_dynamics_local",
            "early_stop_confidence_plus_dynamics",
        ],
        "cards": cards,
        "metricOptions": [
            "AUC-AUROC",
            "AUC-SelAcc",
            "AUROC@10%",
            "AUROC@50%",
            "AUROC@100%",
            "Stop@100%",
            "Earliest>0.6",
        ],
        "metricHints": {
            "AUC-AUROC": "area under AUROC-vs-budget curve",
            "AUC-SelAcc": "area under selected accuracy-vs-budget curve",
            "Stop@100%": "selected accuracy at full budget",
        },
        "conclusions": [
            "benchmark-selective enabling 比全局启用更稳。",
            "DS-R1/aime24、DS-R1/aime25、DS-R1/hmmt25 是稳定收益区。",
            "router 的价值在于保守裁剪，而不是扩大 plugin 覆盖面。",
        ],
        "routerPolicy": router_policy["policies"]["conservative"],
        "routerNotes": select_note_lines(router_notes, ["cache_count", "problem_count"]),
    }


def build_best_of_n_card(
    *,
    card_id: str,
    title: str,
    status: str,
    result_file: str,
    description: str,
    notes_file: str | None = None,
    extra_sources: list[str] | None = None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    result_path = RESULT_DIR / result_file
    payload = read_json(result_path)
    per_cache = {
        cache_key: summarize_candidate_scores(cache_scores)
        for cache_key, cache_scores in payload["scores"].items()
    }
    overall = average_numeric_dicts(list(per_cache.values()))

    source_files = [f"result/{result_file}"]
    notes: list[str] = []
    if notes_file:
        notes_payload = read_json(RESULT_DIR / notes_file)
        source_files.append(f"result/{notes_file}")
        notes.extend(
            select_note_lines(
                notes_payload,
                [
                    "metric",
                    "reduction",
                    "group_size",
                    "topk",
                    "max_gap",
                    "threshold",
                    "candidate_count",
                    "applied_count",
                ],
            )
        )
    if extra_sources:
        source_files.extend(extra_sources)
    if extra_notes:
        notes.extend(extra_notes)

    return {
        "id": card_id,
        "task": "best_of_n",
        "title": title,
        "shortDescription": description,
        "status": status,
        "sourceFiles": source_files,
        "overall": overall,
        "perCache": per_cache,
        "notes": notes,
        "tags": ["best-of-n"],
        "timelineGroup": "best_of_n",
    }


def build_best_of_n_cards() -> dict[str, Any]:
    mixed_v6_metrics = read_json(RESULT_DIR / "mixed_v6_local_head_metrics.json")
    mixed_v6_summary = read_json(RESULT_DIR / "mixed_v6_final_check_summary.json")

    cards = [
        build_best_of_n_card(
            card_id="bon_mixed_v2_logprob",
            title="nad_mixed_v2_aime_top2_gap1e3_logprob",
            status="final_best",
            result_file="best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json",
            notes_file="best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit_notes.json",
            description="最终主线：仅在 AIME 四个 cache 做 very small top2 gap + tok_logprob patch，其他 cache 保守不动。",
            extra_notes=[
                "tie-break 维持 tok_logprob，避免把 candidate-level 试验扩展成全局提交风险。",
            ],
        ),
        build_best_of_n_card(
            card_id="bon_mixed_v1_complete",
            title="nad_mixed_v1_complete",
            status="important",
            result_file="best_of_n_nad_mixed_v1_complete.json",
            notes_file="best_of_n_nad_mixed_v1_complete_notes.json",
            description="完整 baseline / 拼装版，为后续所有 targeted patch 提供统一起点。",
        ),
        build_best_of_n_card(
            card_id="bon_mixed_v2_top3_selfcert",
            title="nad_mixed_v2_aime_top3_selfcert",
            status="important",
            result_file="best_of_n_nad_mixed_v2_aime_top3_selfcert_submit.json",
            notes_file="best_of_n_nad_mixed_v2_aime_top3_selfcert_submit_notes.json",
            description="AIME 定向 self-cert top3 patch；说明 targeted patch 有效，但最终不如 top2 logprob 稳。",
        ),
        build_best_of_n_card(
            card_id="bon_cluster_router",
            title="nad_mixed_v5_cluster_router",
            status="deprecated",
            result_file="best_of_n_nad_mixed_v5_cluster_router_submit.json",
            notes_file="best_of_n_nad_mixed_v5_cluster_router_submit_notes.json",
            description="cluster router 尝试过更复杂的结构化 patch，但最终没有成为主提交路线。",
        ),
        build_best_of_n_card(
            card_id="bon_local_head",
            title="nad_mixed_v6_aime_local_binary_corrector",
            status="promising",
            result_file="best_of_n_nad_mixed_v6_local_head_submit.json",
            notes_file="best_of_n_nad_mixed_v6_local_head_submit_notes.json",
            description="candidate-level local head 路线。离线分析有解释价值，但最终只落地成极小范围修补证据。",
            extra_sources=[
                "result/mixed_v6_local_head_metrics.json",
                "result/mixed_v6_final_check_summary.json",
            ],
            extra_notes=[
                f"chosen_threshold: {mixed_v6_metrics['chosen_threshold']['threshold']}",
                f"valid_auc: {mixed_v6_metrics['valid_auc']}",
                f"head_global net_gain@lambda=1: {next(row['net_gain'] for row in mixed_v6_summary['summary_rows'] if row['lambda_fp'] == 1.0 and row['strategy_name'] == 'head_global_threshold')}",
            ],
        ),
        build_best_of_n_card(
            card_id="bon_activation_a1",
            title="A1 = medoid + activation tie-break",
            status="deprecated",
            result_file="best_of_n_nad_mixed_v1_complete.json",
            description="第一轮 activation 方向验证。图像上能看到现象，但作为主 tie-break 不够稳。",
            extra_sources=[
                "result/a1_medoid_activation_aime24.json",
                "result/a1_medoid_activation_aime24_accuracy.json",
            ],
            extra_notes=[
                "单 cache 最小真实实验：A1 = medoid + activation tie-break",
                "图像观察主要来自题 61 / 70。",
            ],
        ),
        build_best_of_n_card(
            card_id="bon_mixed_v3_tailveto",
            title="nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto",
            status="deprecated",
            result_file="best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json",
            notes_file="best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit_notes.json",
            description="把 activation tail warning 挂到最强 logprob baseline 上的 very small veto 尝试；本地有小收益，但 leaderboard 不稳。",
            extra_notes=[
                "可触发的 DS-R1/aime24 题号：62、80、85。",
                "本地曾有 +1，但最终 leaderboard 反馈不好。",
            ],
        ),
        build_best_of_n_card(
            card_id="bon_em_regularized_m4",
            title="best_of_n_em_regularized_m4_seed42_keepall",
            status="deprecated",
            result_file="best_of_n_em_regularized_m4_seed42_keepall.json",
            notes_file="best_of_n_em_regularized_m4_seed42_keepall_notes.json",
            description="EM regularized 路线保留为探索证据，但没有超过更小、更可控的 AIME patch 主线。",
        ),
    ]

    for card in cards:
        if card["id"] == "bon_activation_a1":
            card["visuals"] = [
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_61.png"),
                        "visuals",
                        "activation_61",
                        "题 61：A1 阶段重点观察到的 activation 轨迹差异。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="61",
                ),
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_70.png"),
                        "visuals",
                        "activation_70",
                        "题 70：与题 61 类似，但趋势更弱。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="70",
                ),
            ]
        elif card["id"] == "bon_mixed_v3_tailveto":
            card["visuals"] = [
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_78.png"),
                        "visuals",
                        "activation_78",
                        "尾段 warning 观察样例之一：尾段新增 neuron 很少。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="78",
                ),
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_80.png"),
                        "visuals",
                        "activation_80",
                        "mixed_v3 本地曾改对的一题，对应 activation tail warning 观察。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="80",
                ),
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_82.png"),
                        "visuals",
                        "activation_82",
                        "尾段平台化现象的另一个样例。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="82",
                ),
                dict(
                    copy_visual(
                        Path("/home/jovyan/work/activation_85.png"),
                        "visuals",
                        "activation_85",
                        "可触发但最终没有转成稳定外部收益的样例。",
                    ),
                    interactiveKind="activation",
                    interactiveDataPath="data/activation_cases.json",
                    interactiveKey="85",
                ),
            ]

    return {
        "generatedAt": now_iso(),
        "task": "best_of_n",
        "finalBestId": "bon_mixed_v2_logprob",
        "highlightedIds": [
            "bon_mixed_v2_logprob",
            "bon_mixed_v1_complete",
            "bon_mixed_v3_tailveto",
            "bon_activation_a1",
            "bon_local_head",
        ],
        "cards": cards,
        "metricOptions": [
            "mean_score",
            "std_score",
            "max_score",
            "min_score",
            "score_range",
            "problem_count",
            "candidate_count",
        ],
        "metricHints": {
            "mean_score": "mean candidate score across all cached candidates",
            "score_range": "max - min candidate score within each cache",
        },
        "conclusions": [
            "very small AIME-only patch 比统一大改更稳。",
            "复杂 router / head / EM 路线有局部信号，但没稳定转成更强主提交。",
            "保守保留 baseline 其余 cache，是当前最稳的工程决策。",
        ],
    }


def build_activation_cases_data() -> dict[str, Any]:
    cache_dir = ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610"
    loader = NadNextLoader(cache_dir)
    problems = build_problem_catalog(load_nad_next_index(cache_dir))
    accuracy = read_json(RESULT_DIR / "a1_medoid_activation_aime24_accuracy.json")
    notes = {
        "61": "A1 阶段重点案例，观察 activation 轨迹形态。",
        "70": "与 61 类似，但图上分离趋势更弱。",
        "78": "tail warning 新补充观察案例。",
        "80": "mixed_v3 本地曾改对的一题。",
        "82": "tail signal 存在但没转成稳定收益。",
        "85": "触发但没转成外部收益。",
    }
    selector_map = {
        "medoid": "medoid",
        "file:./plugins/medoid_activation_tiebreak.py:MedoidActivationTieBreakSelector": "activation_tiebreak",
    }
    cases: dict[str, Any] = {}
    for problem_id in ["61", "70", "78", "80", "82", "85"]:
        grouped_runs = []
        per_problem = accuracy.get("per_problem", {}).get(problem_id, {}).get("selectors", {})
        selected_by_sample: dict[int, list[str]] = {}
        highlighted = []
        for selector_key, selector_label in selector_map.items():
            info = per_problem.get(selector_key)
            if not info:
                continue
            sample_id = int(info["sample_id"])
            selected_by_sample.setdefault(sample_id, []).append(selector_label)
            highlighted.append({
                "label": selector_label,
                "sampleId": sample_id,
                "isCorrect": info.get("is_correct"),
            })
        problem_runs = problems[problem_id]["correct_runs"] + problems[problem_id]["incorrect_runs"]
        for run in problem_runs:
            sample_id = int(run["sample_id"])
            tokens = loader.get_slice_ids_for_sample(sample_id).tolist()
            neuron_counts = loader.get_neuron_cumcnt_for_sample(sample_id).tolist()
            entropy_sums = loader.get_slice_entropy_sum_for_sample(sample_id).tolist()
            grouped_runs.append({
                "sampleId": sample_id,
                "pointCount": len(tokens),
                "maxToken": max(tokens) if tokens else 0,
                "maxNeurons": max(neuron_counts) if neuron_counts else 0,
                "highlightTags": selected_by_sample.get(sample_id, []),
                "points": [
                    {
                        "token": int(token),
                        "neurons": int(neurons),
                        "entropySum": float(entropy),
                        "sampleId": sample_id,
                        "highlightTags": selected_by_sample.get(sample_id, []),
                    }
                    for token, neurons, entropy in zip(tokens, neuron_counts, entropy_sums)
                ],
            })
        cases[problem_id] = {
            "problemId": problem_id,
            "note": notes.get(problem_id),
            "runs": grouped_runs,
            "highlighted": highlighted,
        }
    return {
        "generatedAt": now_iso(),
        "cases": cases,
    }


def build_dynamics_interactive_data() -> dict[str, Any]:
    merged_rows = load_csv_rows(RESULT_DIR / "dynamics_full/dynamics_statistics_DS-R1_merged.csv")
    parameter_rows = load_csv_rows(RESULT_DIR / "dynamics_full/dynamics_parameters.csv")
    return {
        "generatedAt": now_iso(),
        "charts": {
            "phase_plane": {
                "title": "Dynamics Phase Plane",
                "description": "以 rho_tail 为横轴、A_accel 为纵轴，查看不同 benchmark 的动力学状态分布。",
                "xKey": "rho_tail",
                "yKey": "A_accel",
                "rows": merged_rows,
            },
            "rho_timeseries": {
                "title": "Rho / Progress Scatter",
                "description": "用 T_p_norm 近似横向进度，观察 rho_tail 在不同 run 上的分布。",
                "xKey": "T_p_norm",
                "yKey": "rho_tail",
                "rows": merged_rows,
            },
            "statistics_comparison": {
                "title": "Statistics Comparison",
                "description": "对比总 token 长度 T 与最大激活规模 N_max。",
                "xKey": "T",
                "yKey": "N_max",
                "rows": merged_rows,
            },
            "param_ratios": {
                "title": "Parameter Ratios",
                "description": "对比 sigma_alpha_ratio 与 beta_gamma_ratio 两个参数比率。",
                "xKey": "sigma_alpha_ratio",
                "yKey": "beta_gamma_ratio",
                "rows": parameter_rows,
            },
            "coherence_comparison": {
                "title": "Coherence Comparison",
                "description": "以 psi_mid 和 rho_c_corr 展示 coherence 相关的动力学关系。",
                "xKey": "psi_mid",
                "yKey": "rho_c_corr",
                "rows": merged_rows,
            },
        },
    }


def build_timeline() -> dict[str, Any]:
    nodes = [
        {
            "id": "timeline-es-confidence",
            "task": "early_stop",
            "title": "confidence-only baseline 固化",
            "status": "important",
            "summary": "先用纯 confidence 路线建立完整 labeled benchmark 对照，为后续 plugin/router 判断提供统一参照。",
        },
        {
            "id": "timeline-es-dyn-local",
            "task": "early_stop",
            "title": "dynamics_v2_local 做 benchmark 级审计",
            "status": "important",
            "summary": "不做全局启用，逐 benchmark 看 AUC 与 Stop@100 变化，再决定是否值得进入 router。",
        },
        {
            "id": "timeline-es-router",
            "task": "early_stop",
            "title": "conservative router 成为最终提交",
            "status": "final_best",
            "summary": "只在确认收益的 DS-R1/aime24、aime25、hmmt25 启用 dynamics；其他 cache 保守禁用。",
        },
        {
            "id": "timeline-bon-v1",
            "task": "best_of_n",
            "title": "mixed_v1_complete 作为统一 baseline",
            "status": "important",
            "summary": "先把跨 cache baseline 拼完整，再在此基础上做 targeted patch，而不是直接推翻整体结构。",
        },
        {
            "id": "timeline-bon-v2-selfcert",
            "task": "best_of_n",
            "title": "AIME 定向 top3/self-cert patch",
            "status": "important",
            "summary": "验证 very small patch 对 AIME 有价值，但仍需继续收缩风险面。",
        },
        {
            "id": "timeline-bon-v2-final",
            "task": "best_of_n",
            "title": "top2 gap + tok_logprob 成为最终主线",
            "status": "final_best",
            "summary": "最终选更小、更稳的 AIME-only patch，其余 cache 保持 baseline，不做统一激进修改。",
        },
        {
            "id": "timeline-bon-v5",
            "task": "best_of_n",
            "title": "cluster router 没有转成主提交",
            "status": "deprecated",
            "summary": "更复杂的结构化 router 增加了系统复杂度，但没有带来足够稳定的整体收益。",
        },
        {
            "id": "timeline-bon-v6",
            "task": "best_of_n",
            "title": "local head 提供候选级信号",
            "status": "promising",
            "summary": "candidate-level 模型有解释价值，但最终只支持极小 patch，不足以替代保守主线。",
        },
        {
            "id": "timeline-cross-1",
            "task": "cross",
            "title": "小 patch 胜过大一统改写",
            "status": "important",
            "summary": "两个任务都收敛到相同结论：局部、可验证的 patch 比全局重写更稳，更适合最终提交。",
        },
    ]

    return {"generatedAt": now_iso(), "nodes": nodes}


def build_data_index(include_exported_files: bool = False) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []

    early_files = [
        RESULT_DIR / "early_stop_dynamics_router_conservative_submit.json",
        RESULT_DIR / "early_stop_dynamics_router_conservative_submit_notes.json",
        RESULT_DIR / "early_stop_mean_confidence_variants_eval.json",
        RESULT_DIR / "early_stop_dynamics_v2_local_report.json",
        RESULT_DIR / "dynamics_v2_local_batch_summary.json",
        RESULT_DIR / "dynamics_benchmark_router.json",
        RESULT_DIR / "dynamics_benchmark_router.md",
    ]
    groups.append(
        {
            "name": "Early Stop",
            "description": "最终提交、router 依据、以及 confidence/dynamics 对照评估。",
            "files": [copy_artifact(path, "early_stop") for path in early_files],
        }
    )

    bon_files = [
        RESULT_DIR / "best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json",
        RESULT_DIR / "best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit_notes.json",
        RESULT_DIR / "best_of_n_nad_mixed_v1_complete.json",
        RESULT_DIR / "best_of_n_nad_mixed_v1_complete_notes.json",
        RESULT_DIR / "best_of_n_nad_mixed_v6_local_head_submit.json",
        RESULT_DIR / "best_of_n_nad_mixed_v6_local_head_submit_notes.json",
        RESULT_DIR / "mixed_v6_local_head_metrics.json",
        RESULT_DIR / "mixed_v6_final_check_summary.json",
        RESULT_DIR / "specialists_inventory.json",
        RESULT_DIR / "compare_mixedv2_vs_specialists_v1_report.json",
        RESULT_DIR / "a1_medoid_activation_aime24.json",
        RESULT_DIR / "a1_medoid_activation_aime24_accuracy.json",
        RESULT_DIR / "best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json",
        RESULT_DIR / "best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit_notes.json",
    ]
    groups.append(
        {
            "name": "Best-of-N",
            "description": "最终主线、baseline、candidate-level head 试验，以及和 specialists 的对照产物。",
            "files": [copy_artifact(path, "best_of_n") for path in bon_files],
        }
    )

    if include_exported_files:
        exported_files = [
            WEB_PUBLIC_DATA_DIR / "early_stop_cards.json",
            WEB_PUBLIC_DATA_DIR / "best_of_n_cards.json",
            WEB_PUBLIC_DATA_DIR / "research_timeline.json",
            WEB_PUBLIC_DATA_DIR / "activation_cases.json",
            WEB_PUBLIC_DATA_DIR / "dynamics_interactive.json",
            WEB_PUBLIC_DATA_DIR / "data_index.json",
        ]
        groups.append(
            {
                "name": "Web Exports",
                "description": "前端当前直接消费的导出数据。",
                "files": [copy_artifact(path, "web_exports") for path in exported_files],
            }
        )

    return {
        "generatedAt": now_iso(),
        "groups": groups,
    }


def main() -> None:
    ensure_dir(WEB_PUBLIC_DATA_DIR)
    ensure_dir(WEB_FILES_DIR)

    early_stop_cards = build_early_stop_cards()
    best_of_n_cards = build_best_of_n_cards()
    timeline = build_timeline()
    activation_cases = build_activation_cases_data()
    dynamics_interactive = build_dynamics_interactive_data()

    write_json(WEB_PUBLIC_DATA_DIR / "early_stop_cards.json", early_stop_cards)
    write_json(WEB_PUBLIC_DATA_DIR / "best_of_n_cards.json", best_of_n_cards)
    write_json(WEB_PUBLIC_DATA_DIR / "research_timeline.json", timeline)
    write_json(WEB_PUBLIC_DATA_DIR / "activation_cases.json", activation_cases)
    write_json(WEB_PUBLIC_DATA_DIR / "dynamics_interactive.json", dynamics_interactive)

    data_index = build_data_index(include_exported_files=False)
    write_json(WEB_PUBLIC_DATA_DIR / "data_index.json", data_index)

    data_index = build_data_index(include_exported_files=True)
    write_json(WEB_PUBLIC_DATA_DIR / "data_index.json", data_index)

    print(f"Wrote data exports to {WEB_PUBLIC_DATA_DIR}")


if __name__ == "__main__":
    main()
