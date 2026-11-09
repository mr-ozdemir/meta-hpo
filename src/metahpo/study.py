"""Results of an optimisation: every evaluated trial and the best one."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from metahpo.optimizers.base import Direction


@dataclass(frozen=True)
class Trial:
    """One evaluated (or cached) candidate.

    `meta` is the candidate's meta (phase, generation, individual), `start_time` the wall-clock time in
    seconds since the epoch when the evaluation started (NaN when unknown) and `duration` the seconds it took
    (0 for a cached repeat). The two timing fields are measurements, not part of a trial's identity: they are
    left out of == and repr, so the same seed still gives equal trials.
    """

    number: int
    params: dict[str, Any]
    score: float
    cached: bool
    x: NDArray[np.float64] = field(repr=False)
    meta: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default=math.nan, repr=False, compare=False)
    duration: float = field(default=0.0, repr=False, compare=False)

    @property
    def failed(self) -> bool:
        """True when the user's function returned NaN or an infinite value."""
        return not math.isfinite(self.score)


@dataclass
class Study:
    """Every trial of one search and the reason it stopped.

    `stop_reason` is 'budget' (max_evals real calls made), 'exhausted' (every parameter set of a space
    without Float evaluated), 'repeats' (too many cached repeats in a row), 'interrupted' (KeyboardInterrupt)
    or 'error' (the objective raised); it is empty while the search runs.
    """

    direction: Direction
    trials: list[Trial] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def scores(self) -> NDArray[np.float64]:
        return np.array([t.score for t in self.trials], dtype=np.float64)

    @property
    def best_trial(self) -> Trial:
        finite = [t for t in self.trials if not t.failed]
        if not finite:
            raise ValueError("no trial returned a finite score")
        pick = min if self.direction == "min" else max
        return pick(finite, key=lambda t: t.score)

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best_trial.params

    @property
    def best_score(self) -> float:
        return self.best_trial.score

    @property
    def n_calls(self) -> int:
        """Calls of the user's function; cached repeats are not counted."""
        return sum(not t.cached for t in self.trials)
