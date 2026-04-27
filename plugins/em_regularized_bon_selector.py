"""
E_M-regularized Best-of-N selector.

Paper:
  Revisiting the (Sub)Optimality of Best-of-N for Inference-Time Alignment

Core idea:
  - Given N candidate answers with reward scores
  - Let k = ceil(N / M)
  - Sort by reward descending, breaking ties randomly
  - Uniformly sample one answer from the top-k set

This plugin supports two usage styles:
1. In pure Python, call `select_from_answer_score_list(...)`
2. In NAD, use as an external selector with a score file:
     --user-selector 'file:./plugins/em_regularized_bon_selector.py:EMRegularizedBoNSelector'

Expected score-file format for NAD mode:
{
  "problem_id_1": {"run_id_1": 0.91, "run_id_2": 0.73, ...},
  "problem_id_2": {...}
}

Notes:
- This selector does NOT estimate rewards itself.
- It only consumes the current candidate set and their given reward scores.
- Tie-breaking is randomized as required by the paper-style prompt.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext


AnswerScore = Tuple[Any, float]


def _validate_and_normalize_answer_score_list(answer_score_list: Sequence[Any]) -> List[AnswerScore]:
    normalized: List[AnswerScore] = []
    for item in answer_score_list:
        if isinstance(item, dict):
            if 'answer' not in item or 'score' not in item:
                raise ValueError("Each dict item must contain 'answer' and 'score'.")
            answer = item['answer']
            score = float(item['score'])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            answer = item[0]
            score = float(item[1])
        else:
            raise ValueError(
                "Each item must be either {'answer': ..., 'score': ...} or a 2-tuple/list (answer, score)."
            )
        normalized.append((answer, score))
    if not normalized:
        raise ValueError("answer_score_list must not be empty.")
    return normalized


def select_from_answer_score_list(
    answer_score_list: Sequence[Any],
    M: float = 4,
    rng: random.Random | None = None,
) -> Any:
    """
    Apply E_M-regularized Best-of-N to a list of (answer, reward_score).

    Args:
        answer_score_list: list of (answer, score) pairs or dicts with keys answer/score
        M: hyperparameter >= 1; k = ceil(N / M)
        rng: optional random.Random instance for reproducibility

    Returns:
        The selected answer
    """
    if M < 1:
        raise ValueError("M must be >= 1.")

    rng = rng or random.Random()
    items = _validate_and_normalize_answer_score_list(answer_score_list)
    N = len(items)
    k = int(math.ceil(N / float(M)))

    decorated = []
    for idx, (answer, score) in enumerate(items):
        decorated.append((float(score), rng.random(), idx, answer))

    decorated.sort(key=lambda x: (-x[0], x[1]))
    top_k = decorated[:k]
    chosen = rng.choice(top_k)
    return chosen[3]


class EMRegularizedBoNSelector(Selector):
    """
    NAD external selector version of E_M-regularized Best-of-N.

    It expects an external reward score file mapping:
      problem_id -> run_id -> reward_score

    Parameters
    ----------
    M : float
        Hyperparameter in k = ceil(N / M). Must be >= 1.
    score_file : str
        Path to JSON reward-score file.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(self, M: float = 4, score_file: str | None = None, seed: int | None = None):
        if M < 1:
            raise ValueError("M must be >= 1.")
        if not score_file:
            raise ValueError("score_file is required for EMRegularizedBoNSelector.")
        self.M = float(M)
        self.score_file = str(score_file)
        self.seed = seed
        self._context: SelectorContext | None = None
        self._score_map: Dict[str, Dict[str, float]] | None = None
        self._base_rng = random.Random(seed)

    def bind(self, context: SelectorContext) -> None:
        self._context = context
        if self._score_map is None:
            self._score_map = json.loads(Path(self.score_file).read_text())

    def _problem_scores(self) -> Dict[str, float]:
        assert self._context is not None
        assert self._score_map is not None
        problem_id = str(self._context.problem_id)
        if problem_id not in self._score_map:
            raise KeyError(f"Problem {problem_id} missing from score file {self.score_file}")
        raw = self._score_map[problem_id]
        return {str(k): float(v) for k, v in raw.items()}

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("EMRegularizedBoNSelector requires bind(context) before select().")

        problem_scores = self._problem_scores()
        run_ids = list(self._context.run_ids)
        indexed_scores: List[Tuple[int, float]] = []
        for local_idx, run_id in enumerate(run_ids):
            key = str(run_id)
            if key not in problem_scores:
                raise KeyError(
                    f"Run ID {run_id} for problem {self._context.problem_id} missing from score file {self.score_file}"
                )
            indexed_scores.append((local_idx, problem_scores[key]))

        N = len(indexed_scores)
        k = int(math.ceil(N / self.M))

        # Per-problem RNG so selection is randomized but reproducible under a fixed seed.
        rng = random.Random(self._base_rng.random())

        decorated = []
        for local_idx, score in indexed_scores:
            decorated.append((float(score), rng.random(), local_idx))

        decorated.sort(key=lambda x: (-x[0], x[1]))
        top_k = decorated[:k]
        chosen = rng.choice(top_k)
        return int(chosen[2])
