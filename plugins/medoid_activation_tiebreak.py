from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.ops.uniques import extract_tokenwise_counts


class MedoidActivationTieBreakSelector(Selector):
    """
    A1 prototype:
      1. Use medoid (minimum mean distance) as the base selector.
      2. Only when the top medoid candidates are very close, use token-wise
         cumulative-unique endpoint features as a local tie-break.
      3. Prefer candidates that finish earlier and end with fewer cumulative
         unique neurons.

    Notes
    -----
    - This selector does NOT assume any fixed geometric quadrant rule.
    - It uses endpoint features that are directly implied by the activation plots:
      shorter token trajectory and lower cumulative unique-neuron endpoint.
    - It is intentionally conservative: if row-bank data is missing or the
      medoid winner is not close to the runner-up, it falls back to plain medoid.
    """

    def __init__(self, gap_abs: float = 0.01, top_m: int = 2):
        self.gap_abs = float(gap_abs)
        self.top_m = max(2, int(top_m))
        self._context: SelectorContext | None = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    @staticmethod
    def _mean_distances(D: np.ndarray) -> np.ndarray:
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"Expected square distance matrix, got shape={D.shape}")
        return np.asarray(D.mean(axis=1), dtype=np.float64)

    def _close_candidates(self, mean_d: np.ndarray) -> List[int]:
        order = np.argsort(mean_d, kind="stable")
        best = float(mean_d[order[0]])
        candidates: List[int] = []
        for idx in order:
            if len(candidates) >= self.top_m:
                break
            if float(mean_d[idx]) <= best + self.gap_abs:
                candidates.append(int(idx))
        return candidates

    def _endpoint_features(self, run_id: int) -> Tuple[float, float]:
        if self._context is None:
            return float("inf"), float("inf")

        cache = self._context.cache
        rows_srp = getattr(cache, "rows_sample_row_ptr", None)
        rows_rp = getattr(cache, "rows_row_ptr", None)
        rows_keys = getattr(cache, "rows_keys", None)
        rows_slice_ids = getattr(cache, "rows_slice_ids", None)
        rows_trp = getattr(cache, "rows_token_row_ptr", None)

        if any(x is None for x in (rows_srp, rows_rp, rows_keys)):
            return float("inf"), float("inf")

        tokens, counts = extract_tokenwise_counts(
            run_id=run_id,
            rows_srp=rows_srp,
            rows_rp=rows_rp,
            rows_keys=rows_keys,
            rows_slice_ids=rows_slice_ids,
            rows_trp=rows_trp,
            token_axis="row",
        )
        if tokens.size == 0 or counts.size == 0:
            return float("inf"), float("inf")

        end_token_pos = float(tokens[-1])
        end_cum_unique = float(counts[-1])
        return end_cum_unique, end_token_pos

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        mean_d = self._mean_distances(D)
        order = np.argsort(mean_d, kind="stable")
        base_idx = int(order[0])

        if D.shape[0] <= 1:
            return base_idx

        candidates = self._close_candidates(mean_d)
        if len(candidates) < 2:
            return base_idx

        if self._context is None:
            return base_idx

        run_ids = self._context.run_ids
        scored: List[Tuple[float, float, float, int]] = []
        for idx in candidates:
            end_cum_unique, end_token_pos = self._endpoint_features(int(run_ids[idx]))
            scored.append((end_cum_unique, end_token_pos, float(mean_d[idx]), int(idx)))

        valid = [item for item in scored if np.isfinite(item[0]) and np.isfinite(item[1])]
        if not valid:
            return base_idx

        valid.sort()
        return int(valid[0][3])
