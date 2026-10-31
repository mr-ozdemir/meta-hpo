"""Artificial Bee Colony (Karaboga & Basturk, 2007) behind the ask/tell interface.

The phases run in the same order and, apart from the greedy acceptance described below, with the same
rules as the original hyp_optimizer implementation: employed bees one food source after another, onlookers
chosen by roulette sweep (p = 0.9 * fit / max + 0.1) with each improvement visible to the next onlooker,
and at most one scout per cycle. Only the initial population is handed out as a batch, because its
evaluations are independent.

A neighbour differs from its food source in one coordinate, so only that coordinate is decoded again;
the rest of the decoded parameters are copied from the source. A neighbour whose parameter set was already
handed out is dropped and another partner food k is tried; after every other food has been tried, the
coordinate j is drawn again too. The same holds for scouts and the initial population.

Greedy acceptance compares the raw minimised scores: a neighbour replaces its food source when its score is
strictly lower. This deliberately deviates from the original, which compares the fitness 1 / (1 + f): that
transform loses precision near 0 (the fitness of 1e-18 equals the fitness of 0), so a strictly better
neighbour would be rejected. Both orders agree wherever the fitness keeps the scores apart; the fitness is
still used for the onlooker probabilities.

The optimizer keeps its own copy of what it issued (x, key and params) with each pending candidate. tell()
accepts only those candidates and works from the stored copies, so a candidate of another optimizer with the
same id, or a user who changes `cand.params`, cannot change the search.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from metahpo import operators
from metahpo.optimizers.base import Candidate, Direction, Optimizer
from metahpo.space import Space


@dataclass
class _Issued:
    """What the optimizer handed out for one pending candidate: its phase, food source and own copies.

    `new_key` is True when issuing it added the key to the seen set, so a cancel removes the key again.
    """

    kind: str
    food: int
    x: NDArray[np.float64]
    key: tuple[Any, ...]
    params: dict[str, Any]
    new_key: bool


class ABCOptimizer(Optimizer):
    def __init__(
        self,
        space: Space,
        direction: Direction = "min",
        colony_size: int = 10,
        limit: int = 15,
        seed: int | None = None,
        max_tries: int = 100,
    ) -> None:
        super().__init__(space, direction, seed)
        if max_tries < 1:
            raise ValueError("max_tries must be at least 1")
        self.max_tries = max_tries
        if colony_size < 4:
            raise ValueError("colony_size must be at least 4 (two food sources)")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.food_number = sn = colony_size // 2
        self.limit = limit
        self.foods = np.zeros((sn, space.dim))
        self.f = [float("inf")] * sn
        self.fit = [0.0] * sn
        self.trial = [0] * sn
        self.prob = [1.0] * sn
        self._keys: list[tuple[Any, ...]] = [()] * sn
        self._params: list[dict[str, Any]] = [{}] * sn
        self.phase = "init"
        self.cycle = 0
        self._employed_i = 0
        self._onlooker_i = 0
        self._onlooker_t = 0
        self._initialised = [False] * sn
        self._pending: dict[int, _Issued] = {}

    def ask(self) -> list[Candidate]:
        if self.phase == "init":
            waiting = {e.food for e in self._pending.values() if e.kind == "init"}
            todo = [i for i, done in enumerate(self._initialised) if not done and i not in waiting]
            if not todo:
                raise RuntimeError("tell the pending initial candidates before asking again")
            return self._issue(self._fresh(len(todo)), "init", todo)
        if self._pending:
            raise RuntimeError("tell the pending candidate before asking again")
        if self.phase == "scout":
            worst = max(range(self.food_number), key=self.trial.__getitem__)
            if self.trial[worst] >= self.limit:
                return self._issue(self._fresh(1), "scout", [worst])
            self._end_cycle()
        if self.phase == "employed":
            return self._issue([self._neighbour(self._employed_i)], "employed", [self._employed_i])
        while self._uniform() >= self.prob[self._onlooker_i]:
            self._onlooker_i = (self._onlooker_i + 1) % self.food_number
        return self._issue([self._neighbour(self._onlooker_i)], "onlooker", [self._onlooker_i])

    def tell(self, candidates: Sequence[Candidate], scores: Sequence[float]) -> None:
        """Applies the scores of pending candidates of this optimizer.

        The whole batch is checked first: every candidate must be pending, issued by this optimizer (same
        id, key and x) and told once, and every score must convert with float(). A failing batch raises
        and leaves the optimizer unchanged.
        """
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")
        entries = self._claim(candidates)
        values = [self._internal(float(score)) for score in scores]
        for cand, entry, f in zip(candidates, entries, values, strict=True):
            del self._pending[cand.id]
            if f < self.best_f or self.best_x is None:
                self.best_f, self.best_x = f, entry.x.copy()
            i = entry.food
            if entry.kind == "init":
                self._replace(i, entry, f)
                self._initialised[i] = True
                if all(self._initialised):
                    self.phase = "employed"
            elif entry.kind == "scout":
                self._replace(i, entry, f)
                self._end_cycle()
            else:
                if f < self.f[i]:
                    self._replace(i, entry, f)
                else:
                    self.trial[i] += 1
                self._advance(entry.kind)

    def cancel(self, candidates: Sequence[Candidate]) -> None:
        """Withdraws pending candidates without a score; they count as never handed out.

        Their parameter sets may be handed out again, and the next ask() hands out new candidates for the
        same food sources. The batch is checked like in tell(); a failing batch leaves the optimizer
        unchanged.
        """
        entries = self._claim(candidates)
        for cand, entry in zip(candidates, entries, strict=True):
            del self._pending[cand.id]
            if entry.new_key:
                heir = next((e for e in self._pending.values() if e.key == entry.key), None)
                if heir is None:
                    self._seen.discard(entry.key)
                else:
                    heir.new_key = True

    def _claim(self, candidates: Sequence[Candidate]) -> list[_Issued]:
        """The pending entries of `candidates`; ValueError, before any change, unless each is pending,
        issued by this optimizer and named once."""
        entries = []
        ids = set()
        for cand in candidates:
            entry = self._pending.get(cand.id)
            if entry is None or cand.id in ids:
                raise ValueError(f"candidate {cand.id} was not asked for or was already told")
            if cand.key != entry.key or not np.array_equal(cand.x, entry.x):
                raise ValueError(f"candidate {cand.id} was not handed out by this optimizer")
            ids.add(cand.id)
            entries.append(entry)
        return entries

    def _neighbour(self, i: int) -> Candidate:
        """Moves one random coordinate j of food i towards or away from a random other food k.

        A move to a parameter set handed out before is retried with another k, and with another j once
        every other food has been tried, up to `max_tries` attempts.
        """
        sn, d = self.food_number, self.space.dim
        for attempt in range(self.max_tries):
            if attempt % (sn - 1) == 0:
                j = min(int(self._uniform() * d), d - 1)
            k = min(int(self._uniform() * (sn - 1)), sn - 2)
            k += k >= i
            phi = 2.0 * self._uniform() - 1.0
            v = operators.neighbour_value(float(self.foods[i, j]), float(self.foods[k, j]), phi)
            key, params = self.space.move(self._keys[i], self._params[i], j, v)
            if key not in self._seen or len(self._seen) >= self.space.size:
                break
        x = self.foods[i].copy()
        x[j] = v
        cand = Candidate(self._next_id, x, params, key)
        self._next_id += 1
        return cand

    def _issue(self, cands: list[Candidate], kind: str, foods: list[int]) -> list[Candidate]:
        for cand, i in zip(cands, foods, strict=True):
            new_key = cand.key not in self._seen
            self._pending[cand.id] = _Issued(kind, i, cand.x.copy(), cand.key, dict(cand.params), new_key)
            self._seen.add(cand.key)
        return cands

    def _replace(self, i: int, entry: _Issued, f: float) -> None:
        self.foods[i] = entry.x
        self._keys[i] = entry.key
        self._params[i] = entry.params
        self.f[i] = f
        self.fit[i] = operators.fitness_scalar(f)
        self.trial[i] = 0

    def _advance(self, kind: str) -> None:
        if kind == "employed":
            self._employed_i += 1
            if self._employed_i == self.food_number:
                self.prob = operators.onlooker_probabilities(np.array(self.fit)).tolist()
                self.phase = "onlooker"
                self._onlooker_i = 0
                self._onlooker_t = 0
        else:
            self._onlooker_t += 1
            self._onlooker_i = (self._onlooker_i + 1) % self.food_number
            if self._onlooker_t == self.food_number:
                self.phase = "scout"

    def _end_cycle(self) -> None:
        self.cycle += 1
        self.phase = "employed"
        self._employed_i = 0
