"""Enhanced bare-bones particle swarm optimization (Zhang, Lim & Liu 2023) behind the ask/tell interface.

Bare-bones PSO (Kennedy 2003) has no velocity: a particle is placed by sampling around its guides. This
variant follows the full model of Zhang, Lim & Liu (2023, Algorithm 1): a second remote leader, a neighbouring
elite solution per particle, five search operations that a particle cycles through when it stagnates, and,
after every iteration, a root-finding refinement of the leader and the four next-best particles whose
method is chosen by Q-learning.

Leaders, frozen when the first move of an iteration is asked: G is the best point evaluated so far; the
second leader S is, among the five best other particles, the one whose position correlates least with G
(Pearson correlation, MATLAB corr2). The neighbour of particle i is a random better particle whose
correlation with G is below -0.2, else the better particle with the lowest correlation, else G.

The five operations, one random coefficient per coordinate, with c = (neighbour + G) / 2: (1) Levy step
towards (S + G) / 2; (2) Cauchy step towards c; (3) Gaussian around c with standard deviation
|neighbour - G|; (4) per coordinate (2) or (3); (5) c plus a Levy step along the difference of two random
particles, crossed with the current position at rate `crossover`. Every particle starts with operation 1 and
moves on to the next operation (5 wraps to 1) whenever its move does not give a strictly better score. A
move always replaces the particle's position (there is no acceptance test in bare-bones PSO).

Refinement: each refined particle keeps a 4 x 4 Q-table (state = last method, action = next method, initial
state IQI) and picks the action with the highest Q, ties at random. The actions are inverse quadratic
interpolation and the secant method with random partner particles, Steffensen's method (a probe and a
trial), and one ITP step on the segment towards a random partner. The reward is +1 when the best point the
method produced is strictly better than the refined base, else -1, and Q is updated with
Q[s, a] += q_alpha * (reward + q_gamma * max Q[a, :] - Q[s, a]).

The root methods work on h = s (f - target), so the point they aim at is the best score the function can
return. Here f is the internally minimised score and target is the best score in the user's direction:
by default 0.0 for 'min' (losses) and 1.0 for 'max' (accuracy-like metrics). Comparisons never use h.
A single ITP step is taken without a sign condition; equal h values give the midpoint.

Deliberate deviations from the paper: per-particle operation index (the paper's Algorithm 1 writes one
global Flag, its text describes stagnation per particle); one ITP step instead of a bracketing loop;
personal bests are not kept (the paper updates them but no equation uses them); Steffensen's increment is
h / h_max with h_max the largest |h| of the swarm and the base, so the probe stays within one cube width and
the search does not depend on the scale of the objective; ties and comparisons are relative only.

Only the initial population is handed out as a batch; every move and every root trial depends on the state
the previous tell produced. tell() checks the whole batch first and draws no random number, so a failing
batch leaves the optimizer unchanged. Coordinates are repaired after every operation: NaN or infinite
values take the parent's coordinate, the rest are clipped to [0, 1]. A candidate whose parameter set was
already handed out is drawn again, up to `max_tries` attempts.
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

FloatArray = NDArray[np.float64]

IQI, SECANT, STEFFENSEN, ITP = 0, 1, 2, 3
REMOTE = -0.2
ELITES = 5


@dataclass
class _Issued:
    """What the optimizer handed out for one pending candidate: its kind, particle and own copies.

    `new_key` is True when issuing it added the key to the seen set, so a cancel removes the key again.
    """

    kind: str
    particle: int
    x: FloatArray
    key: tuple[Any, ...]
    params: dict[str, Any]
    new_key: bool


def _in_range(name: str, value: float, low: float, high: float, low_open: bool, high_open: bool) -> float:
    number = float(value)
    below = number <= low if low_open else number < low
    above = number >= high if high_open else number > high
    if not math.isfinite(number) or below or above:
        lo, hi = "(" if low_open else "[", ")" if high_open else "]"
        raise ValueError(f"{name} must lie in {lo}{low}, {high}{hi}, got {value!r}")
    return number


class BBPSOOptimizer(Optimizer):
    def __init__(
        self,
        space: Space,
        direction: Direction = "min",
        swarm_size: int = 20,
        target: float | None = None,
        crossover: float = 0.5,
        q_alpha: float = 0.1,
        q_gamma: float = 0.9,
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
        if whole != swarm_size or whole < 6:
            raise ValueError(f"swarm_size must be an integer of at least 6, got {swarm_size!r}")
        self.crossover = _in_range("crossover", crossover, 0.0, 1.0, False, False)
        self.q_alpha = _in_range("q_alpha", q_alpha, 0.0, 1.0, True, False)
        self.q_gamma = _in_range("q_gamma", q_gamma, 0.0, 1.0, False, True)
        default = 0.0 if direction == "min" else 1.0
        self.target = default if target is None else float(target)
        if not math.isfinite(self.target):
            raise ValueError(f"target must be a finite number, got {target!r}")
        self._t_int = self.target if direction == "min" else -self.target
        self.swarm_size = n = whole
        self.positions = np.zeros((n, space.dim))
        self.f = [math.inf] * n
        self.flags = [1] * n
        self.q = np.zeros((n, 4, 4))
        self.q_state = [IQI] * n
        self.phase = "init"
        self.iteration = 0
        self._initialised = [False] * n
        self._owner = 0
        self._move_i = 0
        self._guide: FloatArray | None = None
        self._second: FloatArray | None = None
        self._slots: list[tuple[int, FloatArray, float]] = []
        self._slot_i = 0
        self._slot_stage = "choose"
        self._slot_action = IQI
        self._slot_best: tuple[FloatArray, float] = (np.zeros(space.dim), math.inf)
        self._slot_x: FloatArray | None = None
        self._slot_probe: tuple[FloatArray, float, float] = (np.zeros(space.dim), 0.0, 1.0)
        self._pending: dict[int, _Issued] = {}

    @property
    def guide(self) -> FloatArray | None:
        """The leader G that the current iteration's moves follow; None before the first move is asked."""
        return None if self._guide is None else self._guide.copy()

    @property
    def second_leader(self) -> FloatArray | None:
        """The second leader S of the current iteration; None before the first move is asked."""
        return None if self._second is None else self._second.copy()

    def ask(self) -> list[Candidate]:
        if self.phase == "init":
            waiting = {e.particle for e in self._pending.values()}
            todo = [i for i, done in enumerate(self._initialised) if not done and i not in waiting]
            if not todo:
                raise RuntimeError("tell the pending initial candidates before asking again")
            return self._issue(self._fresh(len(todo)), "init", todo)
        if self._pending:
            raise RuntimeError("tell the pending candidate before asking again")
        while True:
            if self.phase == "move":
                if self._guide is None:
                    self._guide = self._best_x().copy()
                    self._second = self._second_leader(self._guide)
                i = self._move_i
                x = self._propose_new(i, self.flags[i], self._neighbour(i))
                return self._issue(self._candidates(x), "move", [i])
            if self._slot_i >= len(self._slots):
                self.iteration += 1
                self.phase = "move"
                self._move_i = 0
                self._guide = None
                continue
            particle, base_x, base_f = self._slots[self._slot_i]
            if self._slot_stage in ("probe", "trial") and self._slot_x is not None:
                kind = "probe" if self._slot_stage == "probe" else "root"
                return self._issue(self._candidates(self._slot_x), kind, [particle])
            self._slot_action = self._choose_action(particle)
            self._slot_best = (base_x, base_f)
            h_a = self._h(base_f)
            if h_a == 0.0:
                self._finish_slot()
                continue
            if self._slot_action == STEFFENSEN:
                h_max = max([abs(h_a)] + [abs(self._h(v)) for v in self.f if math.isfinite(v)])
                h_norm = h_a / h_max
                self._slot_probe = (base_x, h_norm, h_max)
                self._slot_x = operators.repair(base_x + h_norm, base_x)
                self._slot_stage = "probe"
                return self._issue(self._candidates(self._slot_x), "probe", [particle])
            trial = self._root_candidate(self._slot_action, particle, base_x, h_a)
            if trial is None:
                self._finish_slot()
                continue
            self._slot_x = trial
            self._slot_stage = "trial"
            return self._issue(self._candidates(trial), "root", [particle])

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
                self._owner = entry.particle
            for twin in self._pending.values():
                if twin.key == entry.key:
                    twin.new_key = False
            i = entry.particle
            if entry.kind == "init":
                self.positions[i], self.f[i] = entry.x, f
                self._initialised[i] = True
                if all(self._initialised):
                    self.phase = "move"
            elif entry.kind == "move":
                old = self.f[i]
                self.positions[i], self.f[i] = entry.x, f
                if not f < old:
                    self.flags[i] = self.flags[i] % 5 + 1
                self._move_i += 1
                if self._move_i == self.swarm_size:
                    self._build_slots()
            elif entry.kind == "probe":
                self._take_better(entry.x, f)
                a, h_norm, h_max = self._slot_probe
                h_probe = self._h(f) / h_max if math.isfinite(f) else math.nan
                if not math.isfinite(h_probe) or operators.small_difference(h_probe, h_norm):
                    self._finish_slot()
                else:
                    self._slot_x = operators.repair(operators.steffensen_step(a, h_norm, h_probe), a)
                    self._slot_stage = "trial"
            else:
                self._take_better(entry.x, f)
                self._finish_slot()

    def cancel(self, candidates: Sequence[Candidate]) -> None:
        """Withdraws pending candidates without a score; they count as never handed out.

        Their parameter sets may be handed out again unless another candidate with the same key is still
        pending or has been told. The next ask() repeats the step: a move is drawn again, a probe or root
        trial is handed out again. The batch is checked like in tell(); a failing batch leaves the optimizer
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

    def _best_x(self) -> FloatArray:
        return self._position(0) if self.best_x is None else self.best_x

    def _h(self, f: float) -> float:
        """Distance of an internal score to the target: the function whose root the root methods seek."""
        return f - self._t_int

    def _ranking(self, exclude: int | None = None) -> list[int]:
        """Particles with a finite score, best first, ties by index."""
        finite = [j for j in range(self.swarm_size) if math.isfinite(self.f[j]) and j != exclude]
        return sorted(finite, key=lambda j: (self.f[j], j))

    def _pick(self, ids: list[int]) -> int:
        return ids[int(self.rng.integers(len(ids)))]

    def _least_correlated(self, ids: list[int], guide: FloatArray) -> int:
        values = [operators.correlation(self.positions[j], guide) for j in ids]
        lowest = min(values)
        return self._pick([j for j, v in zip(ids, values, strict=True) if abs(v - lowest) <= 1e-12])

    def _position(self, j: int) -> FloatArray:
        return np.array(self.positions[j], dtype=np.float64)

    def _second_leader(self, guide: FloatArray) -> FloatArray:
        ids = self._ranking(exclude=self._owner)[:ELITES]
        if not ids:
            return np.array(guide, dtype=np.float64)
        return self._position(self._least_correlated(ids, guide))

    def _neighbour(self, i: int) -> FloatArray:
        guide = self._guide if self._guide is not None else self._best_x()
        fitter = [j for j in range(self.swarm_size) if j != i and self.f[j] < self.f[i]]
        remote = [j for j in fitter if operators.correlation(self.positions[j], guide) < REMOTE]
        if remote:
            return self._position(self._pick(remote))
        if fitter:
            return self._position(self._least_correlated(fitter, guide))
        return np.array(guide, dtype=np.float64)

    def _levy(self, d: int) -> FloatArray:
        """Levy(0, 1) draws; a normal draw of exactly 0 gives inf, which repair turns into the parent's
        coordinate."""
        with np.errstate(divide="ignore"):
            return operators.levy_scale(self.rng.standard_normal(d))

    def _propose(self, i: int, mode: int, nb: FloatArray) -> FloatArray:
        """The raw position of one of the five operations for particle i (before repair)."""
        assert self._guide is not None and self._second is not None
        d = self.space.dim
        x, g = self.positions[i], self._guide
        center, spread = (nb + g) / 2.0, np.abs(nb - g)
        if mode == 1:
            return operators.move_two_leaders(x, g, self._second, self._levy(d))
        if mode == 2:
            return operators.move_cauchy(x, center, operators.cauchy_scale(self.rng.random(d)))
        if mode == 3:
            return operators.move_gauss(center, spread, self.rng.standard_normal(d))
        if mode == 4:
            cauchy = operators.cauchy_scale(self.rng.random(d))
            normal = self.rng.standard_normal(d)
            return operators.move_mixed(x, center, spread, cauchy, normal, self.rng.random(d) > 0.5)
        others = [j for j in range(self.swarm_size) if j != i]
        q, r = (others[k] for k in self.rng.choice(len(others), size=2, replace=False))
        levy = self._levy(d)
        cross = self.rng.random(d) <= self.crossover
        return operators.move_levy_difference(x, center, self.positions[q], self.positions[r], levy, cross)

    def _propose_new(self, i: int, mode: int, nb: FloatArray) -> FloatArray:
        """A repaired move of particle i, drawn again while its parameter set was handed out before."""
        x = self.positions[i]
        for _ in range(self.max_tries):
            y = operators.repair(self._propose(i, mode, nb), x)
            if self._new_key(y):
                break
        return y

    def _new_key(self, x: FloatArray) -> bool:
        return self.space.rows(x)[0][0] not in self._seen or len(self._seen) >= self.space.size

    def _choose_action(self, particle: int) -> int:
        row = self.q[particle, self.q_state[particle]]
        best = [a for a in range(4) if row[a] == row.max()]
        return best[0] if len(best) == 1 else self._pick(best)

    def _root_candidate(self, action: int, owner: int, a: FloatArray, h_a: float) -> FloatArray | None:
        """A repaired IQI, secant or ITP point from base a; partners are random finite particles other than
        the owner, drawn again while the point is degenerate or a repeat. None when no point exists."""
        pool = self._ranking(exclude=owner)
        needed = 2 if action == IQI else 1
        if len(pool) < needed:
            return None
        last = None
        for _ in range(self.max_tries):
            picks = [pool[k] for k in self.rng.choice(len(pool), size=needed, replace=False)]
            b, h_b = self.positions[picks[0]], self._h(self.f[picks[0]])
            if action == SECANT:
                if operators.small_difference(h_a, h_b):
                    continue
                raw = operators.secant_point(a, b, h_a, h_b)
            elif action == IQI:
                c, h_c = self.positions[picks[1]], self._h(self.f[picks[1]])
                pairs = ((h_a, h_b), (h_a, h_c), (h_b, h_c))
                if any(operators.small_difference(p, q) for p, q in pairs):
                    continue
                raw = operators.iqi_point(a, b, c, h_a, h_b, h_c)
            else:
                if float(np.max(np.abs(b - a))) <= 1e-12:
                    continue
                raw = a + operators.itp_point(0.0, 1.0, h_a, h_b, 0) * (b - a)
            last = operators.repair(raw, a)
            if self._new_key(last):
                break
        return last

    def _build_slots(self) -> None:
        """The leader with base (G, best score) and the four best other particles with their own base."""
        slots = [(self._owner, self._best_x().copy(), self.best_f)]
        for j in self._ranking(exclude=self._owner)[: ELITES - 1]:
            slots.append((j, self.positions[j].copy(), self.f[j]))
        self._slots = [s for s in slots if math.isfinite(s[2])]
        self._slot_i = 0
        self._slot_stage = "choose"
        self._slot_x = None
        self.phase = "enhance"

    def _take_better(self, x: FloatArray, f: float) -> None:
        if f < self._slot_best[1]:
            self._slot_best = (x.copy(), f)

    def _finish_slot(self) -> None:
        """Reward, Q-learning update, state change and replacement for the current slot."""
        particle, _, base_f = self._slots[self._slot_i]
        action, state = self._slot_action, self.q_state[particle]
        best_x, best_f = self._slot_best
        reward = 1.0 if best_f < base_f else -1.0
        q = self.q[particle]
        q[state, action] += self.q_alpha * (reward + self.q_gamma * float(q[action].max()) - q[state, action])
        self.q_state[particle] = action
        if best_f < base_f:
            self.positions[particle], self.f[particle] = best_x, best_f
        self._slot_i += 1
        self._slot_stage = "choose"
        self._slot_x = None

    def _issue(self, cands: list[Candidate], kind: str, particles: list[int]) -> list[Candidate]:
        for cand, i in zip(cands, particles, strict=True):
            new_key = cand.key not in self._seen
            self._pending[cand.id] = _Issued(kind, i, cand.x.copy(), cand.key, dict(cand.params), new_key)
            self._seen.add(cand.key)
        return cands
