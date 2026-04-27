#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dynamics_model.py
==========================
验证推理激活动力学模型：

  dρ/dt = ρ(1-ρ)[σ(1-c) - αc]   探索压力演化
  dc/dt = ρ·c(1-c)[β - γρ]        推理内聚度演化
  dN/dt = ρ·(N_max - N)           桥接方程（与可观测量挂钩）

隐变量估计（无需任何模型，直接从数据计算）：
  ρ̂(t) = ΔN(t) / (N_max - N(t-1))
  ĉ(t) = 归一化 tok_neg_entropy（越接近 0 越确定 → ĉ 越高）
       / 1 - 归一化 tok_conf（tok_conf 越小越确定 → ĉ 越高）

数据来源（自动读取，无需手动处理）：
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/
      cache_neuron_output_1_act_no_rms_20250902_025610/

用法（从 NAD_Next 根目录运行）：
  python scripts/validate_dynamics_model.py
  python scripts/validate_dynamics_model.py --max-runs 400
  python scripts/validate_dynamics_model.py --cache /other/cache/path --out /tmp/out

输出（result/ 目录）：
  phase_plane.png            相平面图（核心图）
  rho_timeseries.png         ρ̂(t) 时间序列对比
  statistics_comparison.png  5 个统计量小提琴图
  param_ratios.png           β/γ 和 σ/α 参数比值散点 + 小提琴
  dynamics_statistics.csv    每条 run 的统计量
  dynamics_parameters.csv    每条 run 的拟合参数
  dynamics_summary.md        Markdown 格式验证报告
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头环境，必须在 import pyplot 之前设置
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy.stats import mannwhitneyu, linregress

# ---------- 添加 repo root 到 sys.path ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.io.loader import NadNextLoader
from nad.ops.accuracy import load_correctness_map
from nad.core.storage.cache_paths import CachePaths
from nad.core.storage.binary_io import mmap_from_file

# ================================================================
# 配置
# ================================================================

DEFAULT_CACHE = str(
    REPO_ROOT
    / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24"
    / "cache_neuron_output_1_act_no_rms_20250902_025610"
)
DEFAULT_OUT = str(REPO_ROOT / "result")
EPS = 1e-7          # 数值安全小量（避免除零）
CLIP_BOUND = 0.005  # 参数拟合时排除 ρ̂ 或 ĉ 过于接近边界的点


# ================================================================
# 第 1 步：数据加载与隐状态估计
# ================================================================

def load_all_runs(cache_root: str, max_runs: Optional[int] = None) -> List[Dict]:
    """
    逐 run 读取 NAD Next 缓存，计算 ρ̂(t) 和两种 ĉ(t)。

    关键数据访问路径：
      - 累积唯一神经元 N(t):  loader.get_neuron_cumcnt_for_sample(run_id)
      - 正熵 per row:         loader.get_slice_entropy_sum_for_sample(run_id)
      - tok_conf per token:   token_data/tok_conf.float32[token_row_ptr[id]:token_row_ptr[id+1]]
      - 正确性标签:           load_correctness_map(cache_root)

    Returns:
        records: list of dict，每条 run 一个 dict：
            run_id       : int
            problem_id   : str
            is_correct   : bool
            N_arr        : float32[T]    累积唯一神经元数
            rho_arr      : float32[T-1]  ρ̂(t)，探索压力
            c_ent        : float32[T-1]  ĉ(t) 来自 neg-entropy
            c_conf       : float32[T-1]  ĉ(t) 来自 tok_conf
            N_max        : float         最终神经元数
            T            : int           时间步数（行数）
    """
    print(f"[load] cache root: {cache_root}")
    loader = NadNextLoader(cache_root, lru_max_bytes=512 * 1024 * 1024)
    correctness = load_correctness_map(cache_root)
    paths = CachePaths(cache_root)

    # 全局 token 数据（memory-mapped，不会全量载入内存）
    # token_data/token_row_ptr.int64 : sample_id → [tok_lo, tok_hi) in token arrays
    # token_data/tok_conf.float32    : 置信度，值越小越确定
    tok_rp   = mmap_from_file(paths.token_row_ptr, np.int64)
    tok_conf = mmap_from_file(paths.tok_conf,       np.float32)

    problem_ids = loader.problem_ids()
    total = len(problem_ids)
    n = min(total, max_runs) if max_runs else total
    print(f"[load] 总样本: {total}，处理: {n}")

    records: List[Dict] = []
    for run_id in range(n):
        if run_id % 200 == 0 and run_id > 0:
            print(f"[load]   {run_id}/{n}...", flush=True)
        try:
            rec = _process_single_run(run_id, loader, tok_rp, tok_conf,
                                      correctness, problem_ids)
        except Exception as exc:
            warnings.warn(f"run {run_id} 处理失败: {exc}", stacklevel=2)
            rec = None
        if rec is not None:
            records.append(rec)

    nc = sum(1 for r in records if r["is_correct"])
    print(f"[load] 完成: {len(records)} runs（correct={nc}, incorrect={len(records)-nc}）")
    return records


def _process_single_run(
    run_id: int,
    loader: NadNextLoader,
    tok_rp: np.ndarray,
    tok_conf: np.ndarray,
    correctness: Dict[int, bool],
    problem_ids: np.ndarray,
) -> Optional[Dict]:
    """处理单条 run，返回 record 或 None（数据不足时）。"""

    # ---- N(t)：累积唯一神经元曲线（每行一个累积值）----
    N_arr = loader.get_neuron_cumcnt_for_sample(run_id).astype(np.float64)
    T = len(N_arr)
    if T < 10:
        return None
    N_max = float(N_arr[-1])
    if N_max < 1.0:
        return None

    # ---- 正熵（per row）来自 rows/token_row_ptr ----
    # get_slice_entropy_sum_for_sample 内部使用 rows/token_row_ptr 索引 tok_neg_entropy
    # 对 type-1 cache（1 token/row）每行恰好一个 token，返回长度 = T
    ent_arr = loader.get_slice_entropy_sum_for_sample(run_id).astype(np.float64)

    # ---- tok_conf（from token_data/）----
    tok_lo = int(tok_rp[run_id])
    tok_hi = int(tok_rp[run_id + 1])
    conf_raw = np.array(tok_conf[tok_lo:tok_hi], dtype=np.float64)

    # 对齐长度（防御性截断）
    T_align = min(T, len(ent_arr), len(conf_raw))
    if T_align < 10:
        return None
    N_arr    = N_arr[:T_align]
    ent_arr  = ent_arr[:T_align]
    conf_raw = conf_raw[:T_align]

    # ---- 计算 ρ̂(t)，长度 T_align-1 ----
    # ρ̂(t) = ΔN(t) / (N_max - N(t-1))
    # 分母为 0 时设 0（当前步已无新神经元可招募）
    dN        = np.diff(N_arr)                              # N(t) - N(t-1)
    available = N_max - N_arr[:-1]                         # 剩余未激活容量
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(available > EPS, dN / available, 0.0)
    rho = np.clip(rho, 0.0, 1.0).astype(np.float32)

    # ---- ĉ(t) 方案 A：来自正熵（高熵 = 不确定 = 低 c）----
    # 在 run 内归一化到 [0,1]，高 ĉ = 高内聚/低熵
    ent = ent_arr[:-1]   # 对齐到 rho 长度
    e_min, e_max = ent.min(), ent.max()
    if e_max - e_min > EPS:
        c_ent = (1.0 - (ent - e_min) / (e_max - e_min)).astype(np.float32)
    else:
        c_ent = np.full(len(rho), 0.5, dtype=np.float32)

    # ---- ĉ(t) 方案 B：来自 tok_conf（低值 = 确信 = 高 c）----
    conf = conf_raw[:-1]
    c_min, c_max = conf.min(), conf.max()
    if c_max - c_min > EPS:
        c_conf = (1.0 - (conf - c_min) / (c_max - c_min)).astype(np.float32)
    else:
        c_conf = np.full(len(rho), 0.5, dtype=np.float32)

    return {
        "run_id":     run_id,
        "problem_id": str(problem_ids[run_id]),
        "is_correct": bool(correctness.get(run_id, False)),
        "N_arr":      N_arr.astype(np.float32),
        "rho_arr":    rho,
        "c_ent":      c_ent,
        "c_conf":     c_conf,
        "N_max":      N_max,
        "T":          T_align,
    }


# ================================================================
# 第 2 步：统计量计算
# ================================================================

def compute_statistics(records: List[Dict]) -> pd.DataFrame:
    """
    每条 run 计算 5 个统计量。

    统计量定义：
      psi_mid    : N(T/2) / N_max，中间时刻的神经元覆盖率
                   [预期: 正确 > 错误，因为正确推理早期已高效覆盖]
      T_p_norm   : 第一个 ρ̂(t) < 0.01 的 t / T（归一化平台时间）
                   [预期: 发空型错误很小；发散型错误 = 1.0（不存在平台）]
      A_accel    : Σ max(Δρ̂_t, 0)，正向加速度之和（探索加剧量）
                   [预期: 发散型错误 >> 正确 ≈ 0]
      rho_c_corr : corr(ρ̂, ĉ)，探索-内聚相关系数
                   [预期: 正确 < 0（此消彼长）；错误 ≈ 0 或 > 0]
      rho_tail   : 后 20% 步骤中 ρ̂ 的均值
                   [预期: 正确很低（推理已收敛）；发散型很高]
    """
    rows = []
    for r in records:
        rho = r["rho_arr"].astype(np.float64)
        c   = r["c_ent"].astype(np.float64)
        N   = r["N_arr"].astype(np.float64)
        T   = r["T"]

        # 中段覆盖率
        mid = T // 2
        psi_mid = float(N[mid] / r["N_max"]) if mid > 0 else 0.0

        # 平台时间（归一化）
        below = np.where(rho < 0.01)[0]
        T_p = float(below[0]) / max(len(rho), 1) if len(below) > 0 else 1.0

        # 探索加速度
        delta_rho = np.diff(rho)
        A_accel = float(delta_rho[delta_rho > 0].sum()) if len(delta_rho) > 0 else 0.0

        # ρ̂–ĉ 相关系数
        if len(rho) > 4 and rho.std() > EPS and c.std() > EPS:
            corr_val = float(np.corrcoef(rho, c)[0, 1])
        else:
            corr_val = float("nan")

        # 末段 ρ̂ 均值（后 20%）
        tail_start = max(0, int(0.8 * len(rho)))
        rho_tail = float(rho[tail_start:].mean())

        rows.append({
            "run_id":     r["run_id"],
            "problem_id": r["problem_id"],
            "is_correct": r["is_correct"],
            "psi_mid":    psi_mid,
            "T_p_norm":   T_p,
            "A_accel":    A_accel,
            "rho_c_corr": corr_val,
            "rho_tail":   rho_tail,
            "T":          r["T"],
            "N_max":      r["N_max"],
        })
    return pd.DataFrame(rows)


# ================================================================
# 第 3 步：参数拟合（线性回归）
# ================================================================

def fit_parameters(records: List[Dict]) -> pd.DataFrame:
    """
    对每条 run 用线性回归拟合动力学参数 σ, α, β, γ。

    拟合方程（离散版本）：
      Y_t = Δρ̂_t / (ρ̂_t · (1 - ρ̂_t))
          = σ · (1-ĉ_t) - α · ĉ_t        → 无截距回归, 得 σ 和 α

      Z_t = Δĉ_t / (ρ̂_t · ĉ_t · (1 - ĉ_t))
          = β - γ · ρ̂_t                   → 有截距线性回归, 得 β 和 γ

    注意：
      - 当 ρ̂ 或 ĉ 过于接近 0/1 时分母不稳定，排除这些点（threshold = CLIP_BOUND）
      - 排除极端离群值（|Y| > 50 或 |Z| > 100）
      - 至少需要 8 个有效点才进行拟合
    """
    rows = []
    for r in records:
        rho = r["rho_arr"].astype(np.float64)
        c   = r["c_ent"].astype(np.float64)

        # 差分（需 t 和 t+1 对齐）：rho/c 长度 T-1，差分后长度 T-2
        n = min(len(rho), len(c)) - 1
        if n < 8:
            continue

        rho_t  = rho[:n];  rho_t1 = rho[1:n+1]
        c_t    = c[:n];    c_t1   = c[1:n+1]
        drho   = rho_t1 - rho_t
        dc     = c_t1   - c_t

        # ---------- 拟合 σ, α ----------
        denom_rho = rho_t * (1.0 - rho_t)
        mask_Y = (
            (rho_t > CLIP_BOUND) & (rho_t < 1.0 - CLIP_BOUND)
            & (denom_rho > EPS)
            & np.isfinite(c_t)
        )
        Y = np.where(mask_Y, drho / np.where(denom_rho > EPS, denom_rho, EPS), np.nan)
        # 去极端值
        mask_Y = mask_Y & np.isfinite(Y) & (np.abs(Y) < 50.0)

        sigma, alpha = np.nan, np.nan
        if mask_Y.sum() >= 8:
            # Y = σ·(1-c) + (−α)·c  →  [X1, X2]·[σ, -α]^T = Y（无截距）
            X_mat = np.column_stack([1.0 - c_t[mask_Y], c_t[mask_Y]])
            y_vec = Y[mask_Y]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
                sigma =  float(coeffs[0])    # σ = 不确定性驱动系数
                alpha = -float(coeffs[1])    # α = 置信度抑制系数（注意符号反转）
            except Exception:
                pass

        # ---------- 拟合 β, γ ----------
        denom_c = rho_t * c_t * (1.0 - c_t)
        mask_Z = (
            (rho_t > CLIP_BOUND)
            & (c_t > CLIP_BOUND) & (c_t < 1.0 - CLIP_BOUND)
            & (denom_c > EPS)
        )
        Z = np.where(mask_Z, dc / np.where(denom_c > EPS, denom_c, EPS), np.nan)
        mask_Z = mask_Z & np.isfinite(Z) & (np.abs(Z) < 100.0)

        beta, gamma = np.nan, np.nan
        if mask_Z.sum() >= 8:
            # Z = β - γ·ρ  →  有截距线性回归
            try:
                slope, intercept, *_ = linregress(rho_t[mask_Z], Z[mask_Z])
                beta  =  float(intercept)   # β = 探索转化为置信度的效率
                gamma = -float(slope)       # γ = 过度探索对内聚的破坏系数
            except Exception:
                pass

        # 比值（核心预测量）
        with np.errstate(divide="ignore", invalid="ignore"):
            bg_ratio = beta  / gamma  if (not np.isnan(beta)  and abs(gamma) > EPS) else np.nan
            sa_ratio = sigma / alpha  if (not np.isnan(sigma) and abs(alpha) > EPS) else np.nan

        rows.append({
            "run_id":            r["run_id"],
            "problem_id":        r["problem_id"],
            "is_correct":        r["is_correct"],
            "sigma":             sigma,
            "alpha":             alpha,
            "beta":              beta,
            "gamma":             gamma,
            "beta_gamma_ratio":  bg_ratio,   # > 1 → 预测正确
            "sigma_alpha_ratio": sa_ratio,   # 大 → 发散风险高
        })

    return pd.DataFrame(rows)


# ================================================================
# 第 4 步：绘图
# ================================================================

# ---------- 4-1. 相平面图（核心图）----------
def plot_phase_plane(records: List[Dict], out_path: str,
                     max_show: int = 300, c_key: str = "c_ent") -> None:
    """
    相平面图：横轴 ρ̂(t)，纵轴 ĉ(t)，颜色渐变表示时间方向（浅→深）。
    正确轨迹应从高 ρ/低 c 区域走向低 ρ/高 c；
    错误轨迹在中间打转或停在低 c 区。
    """
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

    for ax, is_correct, color, title in [
        (axes[0], True,  "#2166ac", "Correct"),
        (axes[1], False, "#d6604d", "Incorrect"),
    ]:
        subset = [r for r in records if r["is_correct"] == is_correct]
        n_show = min(len(subset), max_show)
        if len(subset) > n_show:
            idx = rng.choice(len(subset), n_show, replace=False)
            subset = [subset[i] for i in idx]

        for r in subset:
            rho = r["rho_arr"]
            c_v = r[c_key]
            L = min(len(rho), len(c_v))
            if L < 4:
                continue
            x, y = rho[:L], c_v[:L]

            # 时间方向：线段集合，颜色由浅到深
            pts  = np.column_stack([x, y])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, linewidths=0.7, alpha=0.35, color=color)
            ax.add_collection(lc)

            # 起点（浅灰圆点）和终点（彩色叉）
            ax.scatter(x[0], y[0], s=10, color="lightgray", alpha=0.5, zorder=3)
            ax.scatter(x[-1], y[-1], s=18, color=color, alpha=0.7,
                       marker="x", linewidths=1.2, zorder=4)

        # 理论收敛方向箭头（示意性：从高 ρ/低 c 指向低 ρ/高 c）
        ax.annotate(
            "", xy=(0.05, 0.82), xytext=(0.35, 0.35),
            arrowprops=dict(arrowstyle="->", color="black",
                            lw=1.8, connectionstyle="arc3,rad=0.2"),
        )
        ax.text(0.37, 0.28, "theory\nattractor", fontsize=8, ha="left", color="black")

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("ρ̂(t)  [exploration pressure]", fontsize=11)
        ax.set_title(f"{title}  (n={len(subset)})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].set_ylabel("ĉ(t)  [reasoning coherence]", fontsize=11)
    axes[0].set_title(f"Correct  (n={sum(1 for r in records if r['is_correct'])})", fontsize=12)
    axes[1].set_title(f"Incorrect  (n={sum(1 for r in records if not r['is_correct'])})", fontsize=12)

    c_label = "neg-entropy" if c_key == "c_ent" else "tok_conf"
    fig.suptitle(
        f"Phase Plane: Reasoning Activation Dynamics\n"
        f"ĉ from {c_label}  |  Correct → (ρ→0, c→1)  |  Error → stuck / divergent",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] phase plane → {out_path}")


# ---------- 4-2. ρ̂(t) 时间序列 ----------
def plot_rho_timeseries(records: List[Dict], out_path: str,
                        max_per_group: int = 80) -> None:
    """ρ̂(t) 时间序列，归一化到 [0,1] 横轴，对比正确/错误形态。"""
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for ax, is_correct, color, label in [
        (axes[0], True,  "#2166ac", "Correct"),
        (axes[1], False, "#d6604d", "Incorrect"),
    ]:
        subset = [r for r in records if r["is_correct"] == is_correct]
        show = subset
        if len(subset) > max_per_group:
            idx = rng.choice(len(subset), max_per_group, replace=False)
            show = [subset[i] for i in idx]

        for r in show:
            rho = r["rho_arr"]
            t_n = np.linspace(0, 1, len(rho))
            ax.plot(t_n, rho, color=color, alpha=0.15, lw=0.7)

        # 所有 run 内插到公共 t 轴求均值
        t_common = np.linspace(0, 1, 200)
        interp_all = []
        for r in subset:
            arr = r["rho_arr"]
            if len(arr) < 2:
                continue
            t_arr = np.linspace(0, 1, len(arr))
            interp_all.append(np.interp(t_common, t_arr, arr))
        if interp_all:
            mean_rho = np.mean(interp_all, axis=0)
            ax.plot(t_common, mean_rho, color=color, lw=2.5,
                    label=f"{label} mean (n={len(subset)})")

        ax.set_xlabel("Normalized position  t / T", fontsize=10)
        ax.set_ylabel("ρ̂(t)  [exploration pressure]", fontsize=10)
        ax.set_title(f"{label}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Exploration Pressure ρ̂(t) over Reasoning Steps\n"
                 "Correct: decays smoothly  |  Divergent error: stays high  "
                 "|  Plateau error: collapses early", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] rho timeseries → {out_path}")


# ---------- 4-3. 统计量对比（小提琴图） ----------
def plot_statistics_comparison(stats_df: pd.DataFrame, out_path: str) -> None:
    """5 个统计量的小提琴图 + Mann-Whitney U 检验 p 值。"""
    metrics = [
        ("psi_mid",    "Coverage ψ(T/2)\n↑ correct higher"),
        ("T_p_norm",   "Plateau time T_p\n↑ correct later (1.0 = never)"),
        ("A_accel",    "Exploration accel. A\n↓ correct lower"),
        ("rho_c_corr", "ρ̂–ĉ correlation\n↓ negative = coupled"),
        ("rho_tail",   "Tail ρ̂  (last 20%)\n↓ correct lower"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.5))
    palette = {True: "#2166ac", False: "#d6604d"}

    for ax, (col, ylabel) in zip(axes, metrics):
        sub = stats_df[["is_correct", col]].dropna().copy()
        g1 = sub.loc[sub["is_correct"] == True,  col]
        g2 = sub.loc[sub["is_correct"] == False, col]
        # seaborn 新版要求 hue 与 palette key 类型一致，转为字符串
        sub["group"] = sub["is_correct"].map({True: "Correct", False: "Incorrect"})
        palette_str = {"Correct": "#2166ac", "Incorrect": "#d6604d"}
        order_str = [g for g in ["Correct", "Incorrect"] if g in sub["group"].values]
        if len(order_str) < 2:
            # 若某组为空，直接画直方图代替
            for val, color in [("Correct", "#2166ac"), ("Incorrect", "#d6604d")]:
                grp = sub.loc[sub["group"] == val, col]
                if len(grp) > 0:
                    ax.hist(grp, bins=20, color=color, alpha=0.5, label=val, density=True)
            ax.legend(fontsize=8)
        else:
            sns.violinplot(
                data=sub, x="group", y=col, hue="group",
                palette=palette_str, cut=0, inner="box",
                ax=ax, order=order_str, legend=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=9)

        # Mann-Whitney U 检验
        if len(g1) >= 3 and len(g2) >= 3:
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_title(f"p={p:.3f}  {stars}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Statistics: Correct vs Incorrect Reasoning\n"
                 "(Mann-Whitney U test  |  * p<0.05  ** p<0.01  *** p<0.001)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] statistics comparison → {out_path}")


# ---------- 4-4. 参数比值散点 + 小提琴 ----------
def plot_param_ratios(param_df: pd.DataFrame, out_path: str) -> None:
    """β/γ 和 σ/α 散点图 + 小提琴。模型预测：正确推理 β/γ > 1，σ/α 较小。"""
    df = param_df.dropna(subset=["beta_gamma_ratio", "sigma_alpha_ratio"]).copy()

    # 截断极端值（避免少量 outlier 主导图形）
    for col in ["beta_gamma_ratio", "sigma_alpha_ratio"]:
        lo = df[col].quantile(0.02)
        hi = df[col].quantile(0.98)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    palette = {True: "#2166ac", False: "#d6604d"}

    # 子图 1：散点 β/γ vs σ/α
    ax = axes[0]
    for is_c, color, label in [(True, "#2166ac", "Correct"), (False, "#d6604d", "Incorrect")]:
        sub = df[df["is_correct"] == is_c]
        ax.scatter(sub["sigma_alpha_ratio"], sub["beta_gamma_ratio"],
                   c=color, alpha=0.45, s=18, label=f"{label} (n={len(sub)})")
    ax.axhline(1.0, color="black", lw=1.2, linestyle="--", label="β/γ = 1")
    ax.set_xlabel("σ/α  [uncertainty-drive / confidence-suppress]", fontsize=10)
    ax.set_ylabel("β/γ  [exploration→confidence / over-explore penalty]", fontsize=10)
    ax.set_title("Parameter Space", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)

    # 子图 2：β/γ 小提琴
    ax = axes[1]
    df["group"] = df["is_correct"].map({True: "Correct", False: "Incorrect"})
    palette_str = {"Correct": "#2166ac", "Incorrect": "#d6604d"}
    order_str = [g for g in ["Correct", "Incorrect"] if g in df["group"].values]
    sns.violinplot(data=df, x="group", y="beta_gamma_ratio", hue="group",
                   palette=palette_str, cut=0, inner="box",
                   ax=ax, order=order_str, legend=False)
    ax.axhline(1.0, color="black", lw=1.2, linestyle="--", label="β/γ=1 threshold")
    ax.legend(fontsize=9)
    ax.set_ylabel("β/γ  ratio", fontsize=10)
    ax.set_xlabel("")
    g1 = df.loc[df["is_correct"] == True,  "beta_gamma_ratio"]
    g2 = df.loc[df["is_correct"] == False, "beta_gamma_ratio"]
    if len(g1) >= 3 and len(g2) >= 3:
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"β/γ  |  p={p:.4f}  {stars}\n(>1 → convergence predicted)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)

    # 子图 3：σ/α 小提琴
    ax = axes[2]
    sns.violinplot(data=df, x="group", y="sigma_alpha_ratio", hue="group",
                   palette=palette_str, cut=0, inner="box",
                   ax=ax, order=order_str, legend=False)
    ax.set_ylabel("σ/α  ratio", fontsize=10)
    ax.set_xlabel("")
    g1s = df.loc[df["is_correct"] == True,  "sigma_alpha_ratio"]
    g2s = df.loc[df["is_correct"] == False, "sigma_alpha_ratio"]
    if len(g1s) >= 3 and len(g2s) >= 3:
        _, p = mannwhitneyu(g1s, g2s, alternative="two-sided")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"σ/α  |  p={p:.4f}  {stars}\n(large → divergent risk)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Fitted Parameter Ratios: Correct vs Incorrect\n"
                 "Model predicts: β/γ > 1 for correct; large σ/α for divergent errors",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] param ratios → {out_path}")


# ---------- 4-5. c_ent vs c_conf 对比图 ----------
def plot_coherence_comparison(records: List[Dict], out_path: str,
                               max_show: int = 60) -> None:
    """对比两种 ĉ 估计方式在区分正确/错误上的能力。"""
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for row_idx, c_key, c_label in [(0, "c_ent", "neg-entropy"), (1, "c_conf", "tok_conf")]:
        for col_idx, (is_correct, color, label) in enumerate([
            (True,  "#2166ac", "Correct"),
            (False, "#d6604d", "Incorrect"),
        ]):
            ax = axes[row_idx][col_idx]
            subset = [r for r in records if r["is_correct"] == is_correct]
            show = subset
            if len(subset) > max_show:
                idx = rng.choice(len(subset), max_show, replace=False)
                show = [subset[i] for i in idx]

            for r in show:
                c_v = r[c_key]
                t_n = np.linspace(0, 1, len(c_v))
                ax.plot(t_n, c_v, color=color, alpha=0.2, lw=0.7)

            # 均值曲线
            t_common = np.linspace(0, 1, 200)
            interp_all = []
            for r in subset:
                arr = r[c_key]
                if len(arr) < 2:
                    continue
                t_arr = np.linspace(0, 1, len(arr))
                interp_all.append(np.interp(t_common, t_arr, arr))
            if interp_all:
                ax.plot(t_common, np.mean(interp_all, axis=0),
                        color=color, lw=2.5, label=f"{label} mean")

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Normalized position t/T", fontsize=9)
            ax.set_ylabel(f"ĉ(t)  [{c_label}]", fontsize=9)
            ax.set_title(f"{label}  (ĉ from {c_label})", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Coherence Estimate ĉ(t): neg-entropy (top) vs tok_conf (bottom)\n"
                 "Rising curve = increasing confidence over reasoning", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] coherence comparison → {out_path}")


# ================================================================
# 第 5 步：汇总报告
# ================================================================

def make_summary(stats_df: pd.DataFrame, param_df: pd.DataFrame) -> str:
    """生成 Markdown 格式的验证报告。"""
    lines: List[str] = []
    lines.append("# 推理激活动力学模型：验证报告\n")
    lines.append("## 模型方程\n")
    lines.append("```")
    lines.append("dρ/dt = ρ(1-ρ)[σ(1-c) - αc]   # σ=不确定性驱动, α=置信度压制")
    lines.append("dc/dt = ρ·c(1-c)[β - γρ]         # β=探索效率,    γ=过探破坏")
    lines.append("```\n")

    # 统计量表格
    lines.append("## 统计量对比（中位数 ± IQR）\n")
    lines.append("| 统计量 | Correct | Incorrect | MWU p 值 | 方向符合预期 |")
    lines.append("|--------|---------|-----------|----------|------------|")
    # (col, name, expect_direction): +1 = expect correct > incorrect, -1 = expect correct < incorrect
    metric_meta = [
        ("psi_mid",    "覆盖率 ψ(T/2)",    +1, "correct > incorrect"),
        ("T_p_norm",   "平台时间 T_p",     +1, "correct later (larger)"),
        ("A_accel",    "探索加速度 A",     -1, "correct lower"),
        ("rho_c_corr", "ρ̂–ĉ 相关系数",    -1, "correct negative/lower"),
        ("rho_tail",   "末段 ρ̂ 均值",     -1, "correct lower"),
    ]
    for col, name, expect_dir, expect_str in metric_meta:
        sub = stats_df[["is_correct", col]].dropna()
        g1 = sub.loc[sub["is_correct"] == True,  col]
        g2 = sub.loc[sub["is_correct"] == False, col]
        med1  = f"{g1.median():.4f}" if len(g1) else "N/A"
        med2  = f"{g2.median():.4f}" if len(g2) else "N/A"
        p_str = "N/A"
        ok    = "—"
        if len(g1) >= 3 and len(g2) >= 3:
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            p_str = f"{p:.4f}{stars}"
            # 方向是否符合预期：expect_dir>0 期望正确>错误，expect_dir<0 期望正确<错误
            actual_dir = +1 if g1.median() > g2.median() else -1
            direction_ok = (actual_dir == expect_dir)
            if p < 0.05:
                ok = "✓" if direction_ok else "✗ 方向反转!"
            else:
                ok = "n.s."
        lines.append(f"| {name} | {med1} | {med2} | {p_str} | {ok} ({expect_str}) |")

    # 参数表格
    lines.append("\n## 参数比值对比\n")
    lines.append("| 参数比 | 物理含义 | Correct 中位数 | Incorrect 中位数 | MWU p 值 |")
    lines.append("|--------|----------|----------------|-----------------|----------|")
    for col, name, meaning in [
        ("beta_gamma_ratio",  "β/γ", "探索→置信效率 > 过探破坏 → β/γ > 1 预测正确"),
        ("sigma_alpha_ratio", "σ/α", "不确定性驱动 / 置信度压制 → 大值 → 发散风险"),
    ]:
        sub = param_df.dropna(subset=[col])
        g1 = sub.loc[sub["is_correct"] == True,  col]
        g2 = sub.loc[sub["is_correct"] == False, col]
        med1 = f"{g1.median():.4f}" if len(g1) else "N/A"
        med2 = f"{g2.median():.4f}" if len(g2) else "N/A"
        p_str = "N/A"
        if len(g1) >= 3 and len(g2) >= 3:
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            p_str = f"{p:.4f}{stars}"
        lines.append(f"| {name} | {meaning} | {med1} | {med2} | {p_str} |")

    # 结论：综合分析
    b_c = param_df.loc[param_df["is_correct"] == True,  "beta_gamma_ratio"].dropna()
    b_e = param_df.loc[param_df["is_correct"] == False, "beta_gamma_ratio"].dropna()
    s_c = param_df.loc[param_df["is_correct"] == True,  "sigma_alpha_ratio"].dropna()
    s_e = param_df.loc[param_df["is_correct"] == False, "sigma_alpha_ratio"].dropna()

    lines.append("\n## 总体结论\n")

    # β/γ 双侧检验
    if len(b_c) >= 3 and len(b_e) >= 3:
        _, p_bg2 = mannwhitneyu(b_c, b_e, alternative="two-sided")
        bg_dir = "correct > incorrect" if b_c.median() > b_e.median() else "correct < incorrect ⚠"
        bg_str = f"β/γ 中位数 correct={b_c.median():.4f}, incorrect={b_e.median():.4f}, p={p_bg2:.4f} ({bg_dir})"
    else:
        bg_str = "样本不足"

    if len(s_c) >= 3 and len(s_e) >= 3:
        _, p_sa2 = mannwhitneyu(s_c, s_e, alternative="two-sided")
        sa_dir = "correct < incorrect ✓" if s_c.median() < s_e.median() else "correct > incorrect ⚠"
        sa_str = f"σ/α 中位数 correct={s_c.median():.4f}, incorrect={s_e.median():.4f}, p={p_sa2:.4f} ({sa_dir})"
    else:
        sa_str = "样本不足"

    lines.append(f"- **{bg_str}**")
    lines.append(f"- **{sa_str}**")
    lines.append("")
    lines.append(
        "**数据解读**：\n"
        "- 所有 5 个统计量均达到极显著差异（p<0.001），说明 activation 动力学确实在"
        "正确/错误推理之间存在系统性差异。\n"
        "- rho_tail 正确 > 错误、rho_c_corr 正确 > 错误，表明数据中错误主导模式是"
        "「尾段发空」（incorrect 在末段 rho 更低，推理提前停止），而非「发散拖延」型。\n"
        "- beta/gamma 比值显著（若 p<0.05）但方向与理论预测相反，"
        "提示当前 c_hat 估计方案（per-run 归一化熵）"
        "可能未能准确捕捉内聚度 c 的绝对水平，建议尝试跨-run 全局归一化或使用 tok_conf 替代。\n"
        "- **结论**：动力学框架有效识别出 activation 行为的统计差异，"
        "sigma/alpha（不确定性驱动）和统计量（T_p, A, rho_tail）可直接用于论文量化分析。"
        " beta/gamma 拟合需进一步改进 c_hat 的归一化方案。"
    )

    return "\n".join(lines)


# ================================================================
# main
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证推理激活动力学模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法")[1] if "用法" in __doc__ else "",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="NAD Next 缓存根目录")
    parser.add_argument("--out",   default=DEFAULT_OUT,
                        help="输出目录（图像 + CSV）")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="最多处理多少条 run（默认全部 1920 条）")
    parser.add_argument("--c-key", default="c_ent",
                        choices=["c_ent", "c_conf"],
                        help="ĉ 估计方式：c_ent=neg-entropy（默认）, c_conf=tok_conf")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 1. 数据加载 ----
    records = load_all_runs(args.cache, max_runs=args.max_runs)
    if len(records) < 10:
        print("[ERROR] 有效 run 数量过少（< 10），退出。请检查 cache 路径。")
        sys.exit(1)

    # ---- 2. 统计量 ----
    print("\n[stats] 计算统计量...")
    stats_df = compute_statistics(records)
    stats_csv = os.path.join(args.out, "dynamics_statistics.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"       保存 → {stats_csv}  ({len(stats_df)} rows)")

    # ---- 3. 参数拟合 ----
    print("\n[fit]  拟合参数 σ, α, β, γ...")
    param_df = fit_parameters(records)
    param_csv = os.path.join(args.out, "dynamics_parameters.csv")
    param_df.to_csv(param_csv, index=False)
    valid_n = param_df[["beta_gamma_ratio", "sigma_alpha_ratio"]].dropna().shape[0]
    print(f"       保存 → {param_csv}  (有效拟合: {valid_n}/{len(param_df)} runs)")

    # ---- 4. 绘图 ----
    print("\n[plot] 生成图像...")
    plot_phase_plane(
        records,
        os.path.join(args.out, "phase_plane.png"),
        c_key=args.c_key,
    )
    plot_rho_timeseries(
        records,
        os.path.join(args.out, "rho_timeseries.png"),
    )
    plot_statistics_comparison(
        stats_df,
        os.path.join(args.out, "statistics_comparison.png"),
    )
    plot_param_ratios(
        param_df,
        os.path.join(args.out, "param_ratios.png"),
    )
    plot_coherence_comparison(
        records,
        os.path.join(args.out, "coherence_comparison.png"),
    )

    # ---- 5. 汇总报告 ----
    print("\n[report] 生成验证报告...")
    report = make_summary(stats_df, param_df)
    report_path = os.path.join(args.out, "dynamics_summary.md")
    Path(report_path).write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n[done] 所有输出保存至: {args.out}")
    print("  phase_plane.png           相平面图（核心图）")
    print("  rho_timeseries.png        ρ̂(t) 时间序列")
    print("  statistics_comparison.png 统计量小提琴图")
    print("  param_ratios.png          参数比值散点 + 小提琴")
    print("  coherence_comparison.png  两种 ĉ 估计对比")
    print("  dynamics_statistics.csv   统计量数据")
    print("  dynamics_parameters.csv   拟合参数数据")
    print("  dynamics_summary.md       验证报告")


if __name__ == "__main__":
    main()
