"""Results of an optimisation: every evaluated trial and the best one."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from metahpo.optimizers.base import Direction


def _same(a: Any, b: Any) -> bool:
    """Equality that always gives a bool: the same object, numpy arrays element by element (two arrays of the
    same shape; an array never equals another type), dicts, lists and tuples item by item, two NaN floats as
    equal, and a comparison that fails as unequal."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        both = isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        return bool(both and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


@dataclass(frozen=True)
class Trial:
    """One evaluated (or cached) candidate.

    `meta` is the candidate's meta (phase, generation, individual), `start_time` the wall-clock time in
    seconds since the epoch when the evaluation started (NaN when unknown) and `duration` the seconds it took
    (0 for a cached repeat). The two timing fields are measurements, not part of a trial's identity: they are
    left out of == and repr, so the same seed still gives equal trials.

    `pruned` is True for a trial that the objective ended by raising metahpo.TrialPruned; its score is then
    its last reported intermediate value (NaN when it reported none), as in Optuna, and it is never the best
    trial. `intermediate_values` maps each step the objective reported (see metahpo.prune) to its value; it
    is empty for an objective that reports nothing. A cached repeat carries the outcome of the trial it
    repeats.
    """

    number: int
    params: dict[str, Any]
    score: float
    cached: bool
    x: NDArray[np.float64] = field(repr=False)
    meta: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default=math.nan, repr=False, compare=False)
    duration: float = field(default=0.0, repr=False, compare=False)
    pruned: bool = False
    intermediate_values: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def failed(self) -> bool:
        """True when the user's function returned NaN or an infinite value; a pruned trial is not failed."""
        return not self.pruned and not math.isfinite(self.score)

    @property
    def state(self) -> str:
        """'complete' for a finite score, 'pruned' for a pruned trial and 'failed' for a NaN or infinite
        score."""
        if self.pruned:
            return "pruned"
        return "failed" if self.failed else "complete"

    def __eq__(self, other: object) -> bool:
        """True when number, params, score, cached flag, x, meta, the pruned flag and the intermediate
        values are equal: x element by element, two NaN scores (failed trials) count as equal, and params and
        meta may hold numpy arrays. The timing fields are left out. A value of another class gives
        NotImplemented."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, Trial)
        return (
            self.number == other.number
            and self.cached == other.cached
            and _same(self.score, other.score)
            and bool(np.array_equal(self.x, other.x))
            and _same(self.params, other.params)
            and _same(self.meta, other.meta)
            and self.pruned == other.pruned
            and _same(self.intermediate_values, other.intermediate_values)
        )


@dataclass
class Study:
    """Every trial of one search and the reason it stopped.

    `stop_reason` is 'budget' (max_evals real calls made), 'exhausted' (every parameter set of a space
    without Float evaluated), 'repeats' (too many cached repeats in a row), 'interrupted' (KeyboardInterrupt),
    'error' (the objective raised) or the reason of the stopping rule that ended the run: 'patience',
    'time' or 'target' for the built-in rules, the `reason` of a Stopper of your own (see metahpo.stop); it is
    empty while the search runs.
    """

    direction: Direction
    trials: list[Trial] = field(default_factory=list)
    stop_reason: str = ""

    def __eq__(self, other: object) -> bool:
        """True when direction, stop_reason and every trial are equal (see Trial.__eq__); a value of another
        class gives NotImplemented."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, Study)
        return (
            self.direction == other.direction
            and self.stop_reason == other.stop_reason
            and len(self.trials) == len(other.trials)
            and all(a == b for a, b in zip(self.trials, other.trials, strict=True))
        )

    @property
    def scores(self) -> NDArray[np.float64]:
        return np.array([t.score for t in self.trials], dtype=np.float64)

    @property
    def best_trial(self) -> Trial:
        finite = [t for t in self.trials if t.state == "complete"]
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
