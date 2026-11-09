"""Particle Swarm Optimization behind the ask/tell interface.

The update is the inertia-weight form of Shi & Eberhart (1998) on top of Kennedy & Eberhart (1995):

    v <- w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x),   then v is clipped to [-v_max, v_max]
    x <- x + v

with r1 and r2 drawn uniformly per particle and coordinate: in ask(), particle by particle in index order,
r1 = rng.random(d) and then r2 = rng.random(d) for every attempt. The defaults w = 0.7298 and
c1 = c2 = 1.49618 are the constriction constants of Clerc & Kennedy (2002) for phi = 4.1, which this form
reproduces exactly; `operators.constriction` computes chi for other phi. The topology is the global best
(star) of the 1995 paper, velocities start at zero, and `max_velocity` is measured in unit-cube widths
(Eberhart & Shi 2000: v_max equal to the range).

Only the initial population and each iteration's flight are handed out as batches, because their
evaluations are independent. The iterations are synchronous: every particle flies once per iteration.
The guide g is frozen when the first candidate of an iteration is asked and stays until the last candidate
of the iteration is told, so the swarm's next flight does not depend on the order in which a batch is
told; only best_params may differ between orders when two candidates tie on the best score.

A coordinate that leaves the unit cube is placed on the wall and its velocity is set to 0 (absorbing walls,
Robinson & Rahmat-Samii 2004). A personal best is replaced only by a strictly lower internal score.

A flight whose parameter set was already handed out is drawn again with new r1 and r2, up to `max_tries`
attempts; only then may a repeat come back, which the cache of optimize() answers without a new evaluation.

The optimizer keeps its own copy of what it issued (x, key, params and the velocity that produced x) with
each pending candidate. tell() accepts only those candidates, checks the whole batch before applying it and
draws no random number, so a failing batch leaves the optimizer unchanged.
"""

from __future__ import annotations

import math
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
    """What the optimizer handed out for one pending candidate: its kind, particle and own copies.

    `new_key` is True when issuing it added the key to the seen set, so a cancel removes the key again.
    """

    kind: str
    particle: int
    x: NDArray[np.float64]
    velocity: NDArray[np.float64]
    key: tuple[Any, ...]
    params: dict[str, Any]
    new_key: bool


def _coefficient(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return number


class PSOOptimizer(Optimizer):
    def __init__(
        self,
        space: Space,
        direction: Direction = "min",
        swarm_size: int = 20,
        inertia: float = 0.7298,
        cognitive: float = 1.49618,
        social: float = 1.49618,
        max_velocity: float = 1.0,
        seed: int | None = None,
        max_tries: int = 100,
    ) -> None:
        super().__init__(space, direction, seed)
        if max_tries < 1:
            raise ValueError("max_tries must be at least 1")
        self.max_tries = max_tries
        try:
            whole = int(swarm_size)
        except (OverflowError, ValueError):
            whole = 0
        if whole != swarm_size or whole < 2:
            raise ValueError(f"swarm_size must be an integer of at least 2, got {swarm_size!r}")
        self.inertia = _coefficient("inertia", inertia)
        self.cognitive = _coefficient("cognitive", cognitive)
        self.social = _coefficient("social", social)
        self.max_velocity = float(max_velocity)
        if not math.isfinite(self.max_velocity) or self.max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be a finite number > 0, got {max_velocity!r}")
        self.swarm_size = n = whole
        d = space.dim
        self.positions = np.zeros((n, d))
        self.velocities = np.zeros((n, d))
        self.pbest = np.zeros((n, d))
        self.pbest_f = [math.inf] * n
        self.f = [math.inf] * n
        self.phase = "init"
        self.iteration = 0
        self._initialised = [False] * n
        self._flown = [False] * n
        self._guide: NDArray[np.float64] | None = None
        self._pending: dict[int, _Issued] = {}

    @property
    def guide(self) -> NDArray[np.float64] | None:
        """The global best g that the current iteration flies towards; None between iterations."""
        return None if self._guide is None else self._guide.copy()

    def ask(self) -> list[Candidate]:
        waiting = {e.particle for e in self._pending.values()}
        if self.phase == "init":
            todo = [i for i, done in enumerate(self._initialised) if not done and i not in waiting]
            if not todo:
                raise RuntimeError("tell the pending initial candidates before asking again")
            cands = self._fresh(len(todo))
            zero = np.zeros(self.space.dim)
            return self._issue(cands, "init", todo, [zero] * len(todo))
        todo = [i for i, done in enumerate(self._flown) if not done and i not in waiting]
        if not todo:
            raise RuntimeError("tell the pending candidates before asking again")
        if self._guide is None:
            best = min(range(self.swarm_size), key=self.pbest_f.__getitem__)
            self._guide = self.pbest[best].copy()
        taken = set(self._seen)
        xs, vs = [], []
        for i in todo:
            x, v = self._flight(i, self._guide, taken)
            taken.add(self.space.rows(x)[0][0])
            xs.append(x)
            vs.append(v)
        return self._issue(self._candidates(np.array(xs)), "fly", todo, vs)

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
            for twin in self._pending.values():
                if twin.key == entry.key:
                    twin.new_key = False
            i = entry.particle
            self.positions[i] = entry.x
            self.velocities[i] = entry.velocity
            self.f[i] = f
            if entry.kind == "init":
                self._set_pbest(i, entry, f)
                self._initialised[i] = True
                if all(self._initialised):
                    self.phase = "fly"
            else:
                if f < self.pbest_f[i]:
                    self._set_pbest(i, entry, f)
                self._flown[i] = True
                if all(self._flown):
                    self.iteration += 1
                    self._flown = [False] * self.swarm_size
                    self._guide = None

    def cancel(self, candidates: Sequence[Candidate]) -> None:
        """Withdraws pending candidates without a score; they count as never handed out.

        Their parameter sets may be handed out again unless another candidate with the same key is still
        pending or has been told, and the next ask() hands out new candidates for the same particles, with new
        random draws and the same frozen guide. The batch is checked like in
        tell(); a failing batch leaves the optimizer unchanged.
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

    def _flight(
        self, i: int, guide: NDArray[np.float64], taken: set[tuple[Any, ...]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One velocity and position update of particle i towards its pbest and the frozen guide.

        A flight to a parameter set in `taken` is drawn again with new r1 and r2, up to `max_tries`
        attempts; the last attempt is returned even when it is a repeat.
        """
        d = self.space.dim
        x_i, v_i, p_i = self.positions[i], self.velocities[i], self.pbest[i]
        for _ in range(self.max_tries):
            r1, r2 = self.rng.random(d), self.rng.random(d)
            v = operators.pso_velocity(
                v_i, x_i, p_i, guide, self.inertia, self.cognitive, self.social, r1, r2, self.max_velocity
            )
            x, v = operators.pso_absorb(x_i + v, v)
            key = self.space.rows(x)[0][0]
            if key not in taken or len(self._seen) >= self.space.size:
                break
        return x, v

    def _issue(
        self,
        cands: list[Candidate],
        kind: str,
        particles: list[int],
        velocities: list[NDArray[np.float64]],
    ) -> list[Candidate]:
        cands = [self._with_meta(c, kind, self.iteration, i) for c, i in zip(cands, particles, strict=True)]
        for cand, i, v in zip(cands, particles, velocities, strict=True):
            new_key = cand.key not in self._seen
            self._pending[cand.id] = _Issued(
                kind, i, cand.x.copy(), np.array(v, dtype=np.float64), cand.key, dict(cand.params), new_key
            )
            self._seen.add(cand.key)
        return cands

    @property
    def generation(self) -> int:
        """Completed iterations: every particle has flown once per iteration."""
        return self.iteration

    def population(self) -> dict[str, Any] | None:
        """Particles: positions, current scores, personal-best scores and speeds (velocity norms)."""
        return {
            "x": self.positions.copy(),
            "score": [self._external(f) for f in self.f],
            "pbest_score": [self._external(f) for f in self.pbest_f],
            "speed": np.linalg.norm(self.velocities, axis=1).tolist(),
        }

    def _set_pbest(self, i: int, entry: _Issued, f: float) -> None:
        self.pbest[i] = entry.x
        self.pbest_f[i] = f
