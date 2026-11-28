"""Stopping rules that end a search before its budget: `optimize(..., stop=[...])`.

A rule is a Stopper. Three ship with the library: Patience (no improvement for a number of calls, stop
reason 'patience'), MaxTime (a limit on the search time, 'time') and TargetScore (a score that is good
enough, 'target'). A rule of your own subclasses Stopper, sets `reason` and overrides should_stop().
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from metahpo.optimizers.base import _init_settings, _integer
from metahpo.report import RunState

RESERVED_REASONS = ("budget", "exhausted", "repeats", "interrupted", "error")
"""The stop reasons the search gives itself; no rule may use them as its reason."""


class Stopper:
    """A rule that ends a search early: subclass it, set `reason` and override should_stop().

    `reason` is the stop_reason a run records when this rule ends it: a non-empty string other than the
    reasons in RESERVED_REASONS; optimize() rejects any other value with ValueError.

    optimize() calls should_stop() of every rule exactly once after every trial, cached repeats included, in
    the order the rules were given: after the trial is in state.study.trials, state.best and state.calls are
    updated and the callbacks received on_trial. Every rule is called for every trial, also when an earlier
    rule already said stop, so a rule that counts stays in step; the first rule in the order that says stop
    gives the reason. The run then tells the optimizer the evaluated part of the current batch, withdraws the
    rest and ends as usual (last generation, on_run_end, results folder, console line) without a further
    call of the objective. A rule's reason wins over 'budget' when the same trial used up the budget.

    Every run works on its own copy (copy.deepcopy) of the rules, so a rule may keep state in its
    attributes: it starts fresh in every run, ends its own run only, and the object passed to optimize() is
    never changed.
    """

    reason: str = ""

    def should_stop(self, state: RunState) -> bool:
        """True when the run should end after the trial just recorded, state.study.trials[-1]."""
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """The constructor parameters, each read from the attribute of the same name (a parameter without
        such an attribute is left out), read the same way as Optimizer.settings(). config.json records them
        with the class name, and resume() builds Patience, MaxTime and TargetScore again from them."""
        return _init_settings(self)


class Patience(Stopper):
    """Stops when `evals` calls of the objective in a row brought no improvement; stop reason 'patience'.

    Calls count as for max_evals: a cached repeat is no call. An improvement is a finite score better than
    the reference, in the run's direction, by more than `min_delta`; the first finite score of a run is
    always one, and only an improvement moves the reference. A NaN or infinite score and a pruned trial are
    calls without improvement. `evals` is an integer of at least 1 and `min_delta` a finite number >= 0.
    """

    reason = "patience"

    def __init__(self, evals: int, min_delta: float = 0.0) -> None:
        self.evals = _integer("evals", evals, 1)
        self.min_delta = _non_negative("min_delta", min_delta)
        self._reference: float | None = None
        self._improved_at = 0

    def should_stop(self, state: RunState) -> bool:
        trial = state.study.trials[-1]
        score = trial.score
        if not trial.pruned and self._improves(score, state.direction):
            self._reference = score
            self._improved_at = state.calls
        return state.calls - self._improved_at >= self.evals

    def _improves(self, score: float, direction: str) -> bool:
        """True when `score` is finite and beats the reference by more than min_delta, or is the first
        finite score."""
        if not math.isfinite(score):
            return False
        if self._reference is None:
            return True
        gain = self._reference - score if direction == "min" else score - self._reference
        return gain > self.min_delta


class MaxTime(Stopper):
    """Stops when the run has searched for at least seconds + 60 * minutes + 3600 * hours seconds; stop
    reason 'time'.

    The time is the run's search time at the end of the trial just recorded, state.elapsed() at its
    start_time + duration, which leaves out the time a run was stopped. It is checked after every trial and
    never interrupts a running evaluation: the run ends after the first evaluation that finishes at or past
    the limit. The answer depends only on what the results folder records (the start of the run, its pauses
    and the trial's start and duration), so resume() replays it exactly and a run that MaxTime ended comes
    back without a call. Every part is a finite number >= 0 and their sum is positive.
    """

    reason = "time"

    def __init__(self, seconds: float = 0.0, minutes: float = 0.0, hours: float = 0.0) -> None:
        self.seconds = _non_negative("seconds", seconds)
        self.minutes = _non_negative("minutes", minutes)
        self.hours = _non_negative("hours", hours)
        self._limit = self.seconds + 60.0 * self.minutes + 3600.0 * self.hours
        if not math.isfinite(self._limit) or self._limit <= 0.0:
            raise ValueError(
                f"MaxTime needs a positive, finite time, got seconds={seconds!r}, minutes={minutes!r}, "
                f"hours={hours!r}"
            )

    def should_stop(self, state: RunState) -> bool:
        trial = state.study.trials[-1]
        return state.elapsed(trial.start_time + trial.duration, trial.number) >= self._limit


class TargetScore(Stopper):
    """Stops when the best score so far is at least as good as `score` in the run's direction (at most
    `score` when minimising, at least `score` when maximising); stop reason 'target'. `score` is finite."""

    reason = "target"

    def __init__(self, score: float) -> None:
        self.score = float(score)
        if not math.isfinite(self.score):
            raise ValueError(f"score must be a finite number, got {score!r}")

    def should_stop(self, state: RunState) -> bool:
        best = state.best
        if best is None:
            return False
        return best.score <= self.score if state.direction == "min" else best.score >= self.score


def _non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return number


def _rules(stop: Stopper | Sequence[Stopper] | None) -> list[Stopper]:
    """The rules of optimize(stop=...): None, one Stopper or a sequence of them. TypeError for anything else,
    ValueError for a rule whose reason is empty, not a string or one of RESERVED_REASONS."""
    if stop is None:
        return []
    rules = [stop] if isinstance(stop, Stopper) else stop
    if not isinstance(rules, Sequence) or not all(isinstance(rule, Stopper) for rule in rules):
        raise TypeError("stop must be a Stopper, a sequence of Stoppers or None")
    for rule in rules:
        reason = rule.reason
        if not isinstance(reason, str) or not reason.strip() or reason in RESERVED_REASONS:
            raise ValueError(
                f"the reason of {type(rule).__name__} must be a non-empty string other than "
                f"{', '.join(RESERVED_REASONS)}; got {reason!r}"
            )
    return list(rules)
