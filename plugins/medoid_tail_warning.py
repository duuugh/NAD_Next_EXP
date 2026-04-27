from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.ops.uniques import extract_tokenwise_counts


class MedoidTailWarningSelector(Selector):
    """
    Version A:
      1. Use medoid as the base selector.
      2. Only inspect the top-2 medoid candidates when they are extremely close.
      3. Do not let activation features directly pick a winner.
      4. Only veto the medoid winner when its tail looks clearly over-flat
         and the runner-up does not show the same warning.

    Tail warning intuition
    ----------------------
    Some incorrect runs appear to keep generating tokens while adding very few
    new unique neurons near the end. We model that as:
      - early plateau: the curve reaches ~final level too early
      - weak tail growth: the final tail contributes too little new mass

    This plugin is intentionally conservative:
      - only top-2 medoid candidates are compared
      - only a small medoid gap triggers inspection
      - warning acts as veto, not as a direct replacement score
    """

    def __init__(
        self,
        gap_abs: float = 0.002,
        tail_start: float = 0.85,
        plateau_fraction: float = 0.98,
        tail_new_ratio_warn: float = 0.010,
        plateau_progress_warn: float = 0.80,
        min_tail_ratio_advantage: float = 0.008,
        min_plateau_progress_advantage: float = 0.05,
        min_points: int = 8,
    ) -> None:
        self.gap_abs = float(gap_abs)
        self.tail_start = float(tail_start)
        self.plateau_fraction = float(plateau_fraction)
        self.tail_new_ratio_warn = float(tail_new_ratio_warn)
        self.plateau_progress_warn = float(plateau_progress_warn)
        self.min_tail_ratio_advantage = float(min_tail_ratio_advantage)
        self.min_plateau_progress_advantage = float(min_plateau_progress_advantage)
        self.min_points = max(3, int(min_points))
        self._context: SelectorContext | None = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    @staticmethod
    def _mean_distances(D: np.ndarray) -> np.ndarray:
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"Expected square distance matrix, got shape={D.shape}")
        return np.asarray(D.mean(axis=1), dtype=np.float64)

    @staticmethod
    def _safe_progress(tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.float64)
        if tokens.size == 0:
            return np.array([], dtype=np.float64)
        if tokens.size == 1:
            return np.array([0.0], dtype=np.float64)
        span = float(tokens[-1] - tokens[0])
        if span <= 1e-12:
            return np.linspace(0.0, 1.0, num=tokens.size, dtype=np.float64)
        return (tokens - tokens[0]) / span

    def _tail_metrics(self, run_id: int) -> Optional[Dict[str, float]]:
        if self._context is None:
            return None

        cache = self._context.cache
        rows_srp = getattr(cache, "rows_sample_row_ptr", None)
        rows_rp = getattr(cache, "rows_row_ptr", None)
        rows_keys = getattr(cache, "rows_keys", None)
        rows_slice_ids = getattr(cache, "rows_slice_ids", None)
        rows_trp = getattr(cache, "rows_token_row_ptr", None)

        if any(x is None for x in (rows_srp, rows_rp, rows_keys)):
            return None

        tokens, counts = extract_tokenwise_counts(
            run_id=run_id,
            rows_srp=rows_srp,
            rows_rp=rows_rp,
            rows_keys=rows_keys,
            rows_slice_ids=rows_slice_ids,
            rows_trp=rows_trp,
            token_axis="row",
        )
        if tokens.size < self.min_points or counts.size < self.min_points:
            return None

        progress = self._safe_progress(tokens)
        if progress.size < self.min_points:
            return None

        counts = counts.astype(np.float64, copy=False)
        final_count = float(counts[-1])
        denom = max(final_count, 1.0)

        tail_idx = int(np.searchsorted(progress, self.tail_start, side="left"))
        tail_idx = min(max(tail_idx, 0), counts.size - 1)
        tail_start_count = float(counts[tail_idx])
        tail_new_ratio = max(0.0, final_count - tail_start_count) / denom

        plateau_target = self.plateau_fraction * final_count
        plateau_hits = np.where(counts >= plateau_target)[0]
        plateau_idx = int(plateau_hits[0]) if plateau_hits.size else counts.size - 1
        plateau_progress = float(progress[plateau_idx])

        return {
            "tail_new_ratio": float(tail_new_ratio),
            "plateau_progress": plateau_progress,
            "final_count": final_count,
            "num_points": float(counts.size),
        }

    def _is_warned(self, metrics: Dict[str, float]) -> bool:
        return (
            metrics["tail_new_ratio"] <= self.tail_new_ratio_warn
            and metrics["plateau_progress"] <= self.plateau_progress_warn
        )

    def _runner_is_healthier(
        self,
        base_metrics: Dict[str, float],
        runner_metrics: Dict[str, float],
    ) -> bool:
        better_tail = (
            runner_metrics["tail_new_ratio"]
            >= base_metrics["tail_new_ratio"] + self.min_tail_ratio_advantage
        )
        later_plateau = (
            runner_metrics["plateau_progress"]
            >= base_metrics["plateau_progress"] + self.min_plateau_progress_advantage
        )
        return bool(better_tail or later_plateau)

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        mean_d = self._mean_distances(D)
        order = np.argsort(mean_d, kind="stable")
        base_idx = int(order[0])

        if D.shape[0] <= 1 or order.size < 2 or self._context is None:
            return base_idx

        runner_idx = int(order[1])
        gap = float(mean_d[runner_idx] - mean_d[base_idx])
        if gap > self.gap_abs:
            return base_idx

        run_ids = self._context.run_ids
        base_metrics = self._tail_metrics(int(run_ids[base_idx]))
        runner_metrics = self._tail_metrics(int(run_ids[runner_idx]))
        if base_metrics is None or runner_metrics is None:
            return base_idx

        base_warned = self._is_warned(base_metrics)
        runner_warned = self._is_warned(runner_metrics)

        if base_warned and (not runner_warned) and self._runner_is_healthier(base_metrics, runner_metrics):
            return runner_idx
        return base_idx
