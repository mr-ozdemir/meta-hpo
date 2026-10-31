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
    number: int
    params: dict[str, Any]
    score: float
    cached: bool
    x: NDArray[np.float64] = field(repr=False)

    @property
    def failed(self) -> bool:
        """True when the user's function returned NaN or an infinite value."""
        return not math.isfinite(self.score)


@dataclass
class Study:
    direction: Direction
    trials: list[Trial] = field(default_factory=list)

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
