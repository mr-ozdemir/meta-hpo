"""Common ask/tell interface of all metahpo optimizers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from metahpo.space import Space

Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Candidate:
    """A point to evaluate: `x` in the unit cube, `params` as the user's function receives them, and `key`,
    the hashable decoded identity used for caching."""

    id: int
    x: NDArray[np.float64]
    params: dict[str, Any]
    key: tuple[Any, ...]


class Optimizer:
    """ask() returns the candidates that can be evaluated now; tell() reports their scores.

    An optimizer never hands out a parameter set it has handed out before, as long as it finds a new one in
    `max_tries` attempts; only then may a repeat come back, which a cache answers without a new evaluation.

    Scores are given in the user's direction; internally everything is minimised, and a NaN or infinite
    score becomes the worst possible value for the direction (-inf when maximising, +inf when minimising).

    Every optimizer keeps this contract: tell() accepts only candidates this optimizer handed out and has not
    been told yet, each at most once per call, and every score must convert with float(). The whole batch is
    checked first, so a failing batch raises and leaves the optimizer unchanged. The optimizer keeps its own
    copy of what it handed out, so changing `cand.params` does not change the search. cancel() withdraws
    pending candidates that will not be evaluated.
    """

    def __init__(self, space: Space, direction: Direction = "min", seed: int | None = None) -> None:
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
        self.space = space
        self.direction: Direction = direction
        self.rng = np.random.default_rng(seed)
        self._uniforms: list[float] = []
        self._next_id = 0
        self.best_x: NDArray[np.float64] | None = None
        self.best_f = math.inf
        self.max_tries = 100
        self._seen: set[tuple[Any, ...]] = set()

    def ask(self) -> list[Candidate]:
        raise NotImplementedError

    def _uniform(self) -> float:
        """Next U(0, 1) draw; drawn from the generator in blocks, so single draws cost no numpy call."""
        if not self._uniforms:
            self._uniforms = self.rng.random(4096).tolist()[::-1]
        return self._uniforms.pop()

    def tell(self, candidates: Sequence[Candidate], scores: Sequence[float]) -> None:
        raise NotImplementedError

    def cancel(self, candidates: Sequence[Candidate]) -> None:
        """Withdraws pending candidates that will not be evaluated, so the optimizer can ask again.

        optimize() calls it for the candidates of a batch that the budget cuts off. An optimizer that does
        not override it keeps such candidates pending.
        """
        raise NotImplementedError

    @property
    def best_params(self) -> dict[str, Any] | None:
        return None if self.best_x is None else self.space.to_dicts(self.best_x)[0]

    @property
    def best_score(self) -> float:
        return self.best_f if self.direction == "min" else -self.best_f

    def _internal(self, score: float) -> float:
        """User score to the minimised value."""
        if not math.isfinite(score):
            return math.inf
        return score if self.direction == "min" else -score

    def _fresh(self, n: int) -> list[Candidate]:
        """n random candidates whose parameter sets are new, resampling a repeat up to `max_tries` times."""
        cands = self._candidates(self.space.sample(self.rng, n))
        taken = set(self._seen)
        for t, cand in enumerate(cands):
            tries = 1
            while cand.key in taken and tries < self.max_tries and len(taken) < self.space.size:
                keys, params = self.space.rows(x := self.space.sample(self.rng, 1))
                cand = Candidate(cand.id, x[0], params[0], keys[0])
                tries += 1
            cands[t] = cand
            taken.add(cand.key)
        return cands

    def _candidates(self, xs: NDArray[np.float64]) -> list[Candidate]:
        xs = np.atleast_2d(xs)
        keys, params = self.space.rows(xs)
        start = self._next_id
        self._next_id += len(keys)
        rows = zip(xs, params, keys, strict=True)
        return [Candidate(start + n, x, p, k) for n, (x, p, k) in enumerate(rows)]
