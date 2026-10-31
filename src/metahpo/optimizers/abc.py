"""Artificial Bee Colony (Karaboga & Basturk, 2007) behind the ask/tell interface.

The phases run in the same order and with the same rules as the original hyp_optimizer implementation:
employed bees one food source after another, onlookers chosen by roulette sweep (p = 0.9 * fit / max + 0.1)
with each improvement visible to the next onlooker, and at most one scout per cycle. Only the initial
population is handed out as a batch, because its evaluations are independent.

A neighbour differs from its food source in one coordinate, so only that coordinate is decoded again;
the rest of the decoded parameters are copied from the source. A neighbour whose parameter set was already
handed out is dropped and another partner food k is tried; after every other food has been tried, the
coordinate j is drawn again too. The same holds for scouts and the initial population.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from metahpo import operators
from metahpo.optimizers.base import Candidate, Direction, Optimizer
from metahpo.space import Space


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
        self._pending: dict[int, tuple[str, int]] = {}

    def ask(self) -> list[Candidate]:
        if self.phase == "init":
            waiting = {i for kind, i in self._pending.values() if kind == "init"}
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
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")
        for cand, score in zip(candidates, scores, strict=True):
            if cand.id not in self._pending:
                raise ValueError(f"candidate {cand.id} was not asked for or was already told")
            kind, i = self._pending.pop(cand.id)
            f = self._internal(float(score))
            if f < self.best_f or self.best_x is None:
                self.best_f, self.best_x = f, cand.x.copy()
            if kind == "init":
                self._replace(i, cand, f)
                self._initialised[i] = True
                if all(self._initialised):
                    self.phase = "employed"
            elif kind == "scout":
                self._replace(i, cand, f)
                self._end_cycle()
            else:
                if operators.fitness_scalar(f) > self.fit[i]:
                    self._replace(i, cand, f)
                else:
                    self.trial[i] += 1
                self._advance(kind)

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
            self._pending[cand.id] = (kind, i)
            self._seen.add(cand.key)
        return cands

    def _replace(self, i: int, cand: Candidate, f: float) -> None:
        self.foods[i] = cand.x
        self._keys[i] = cand.key
        self._params[i] = dict(cand.params)
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
