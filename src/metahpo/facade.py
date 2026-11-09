"""One-call optimisation: `optimize(function, space, direction=...)`."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from metahpo.optimizers.abc import ABCOptimizer
from metahpo.optimizers.base import Candidate, Direction, Optimizer
from metahpo.report import (
    DEFAULT_OUTPUT,
    Callback,
    ConsoleReporter,
    ResultWriter,
    RunState,
    resolve_output,
    resolve_verbose,
)
from metahpo.space import Space
from metahpo.study import Study, Trial

Objective = Callable[[dict[str, Any]], float]

_MAX_REPEATS = 1000


def optimize(
    function: Objective,
    space: Space,
    direction: Direction = "min",
    max_evals: int = 500,
    optimizer: Optimizer | None = None,
    seed: int | None = None,
    cache: bool = True,
    output: str | os.PathLike[str] | None = DEFAULT_OUTPUT,
    verbose: int = 1,
    log_every: int = 1,
    callbacks: Sequence[Callback] | None = None,
) -> Study:
    """Search `space` for the parameters that minimise or maximise `function(params) -> score`.

    `function` receives a dict of parameter values and returns a score (accuracy, F1, loss, ...); what it
    does inside, including its own training budget, is entirely up to the user. `max_evals` counts the
    calls of `function`: with `cache=True` a repeated parameter set reuses the earlier score and does not
    use up the budget. The search stops early only when the space is exhausted: every parameter set of a
    space without Float has been evaluated, or `_MAX_REPEATS` candidates in a row were repeats. The default
    optimizer is ABCOptimizer.

    `function` receives a copy of the candidate's params, so changing the dict does not reach the optimizer.
    When the budget cuts a batch short, the candidates that are not evaluated are withdrawn with
    `optimizer.cancel()`, so a passed optimizer can still be used with ask/tell afterwards; an optimizer
    that does not implement cancel() keeps them pending.

    Reporting (see metahpo.report): `verbose=1` prints one line per generation (every `log_every`
    generations), `verbose=2` one line per trial as well, `verbose=0` nothing. `output` is the folder below
    which a time-stamped results folder is written (default 'hpo_results' in the working directory; None
    switches it off). `callbacks` receive the same events. A KeyboardInterrupt stops the search and returns
    the study so far with stop_reason 'interrupted'; an exception raised by `function` is re-raised after the
    results so far are written. Every trial records its phase, generation and individual, its start time and
    its duration.
    """
    if max_evals < 1:
        raise ValueError("max_evals must be at least 1")
    opt = optimizer or ABCOptimizer(space, direction=direction, seed=seed)
    if opt.direction != direction or opt.space is not space:
        raise ValueError("the optimizer must use the same space and direction as optimize()")
    level = resolve_verbose(verbose)
    listeners: list[Callback] = []
    if level > 0:
        listeners.append(ConsoleReporter(level, log_every))
    folder = resolve_output(output)
    if folder is not None:
        listeners.append(ResultWriter(folder))
    listeners.extend(callbacks or [])
    study = Study(direction)
    state = RunState(space, opt, direction, max_evals, seed, study)
    run = _Run(function, space, opt, cache, max_evals, state, listeners)
    for listener in listeners:
        listener.on_run_start(state)
    try:
        run.loop()
    except KeyboardInterrupt:
        run.abandon("interrupted")
    except BaseException:
        run.abandon("error")
        run.finish()
        raise
    run.finish()
    return study


class _Run:
    """The search loop of optimize() with its bookkeeping, split out so interrupts can tidy up."""

    def __init__(
        self,
        function: Objective,
        space: Space,
        opt: Optimizer,
        cache: bool,
        max_evals: int,
        state: RunState,
        listeners: list[Callback],
    ) -> None:
        self.function = function
        self.space = space
        self.opt = opt
        self.cache = cache
        self.max_evals = max_evals
        self.state = state
        self.listeners = listeners
        self.seen: dict[tuple[Any, ...], float] = {}
        self.repeats = 0
        self.batch: list[Candidate] = []
        self.scores: list[float] = []
        self.generation = opt.generation
        self.generation_trials: list[Trial] = []

    def loop(self) -> None:
        study, state = self.state.study, self.state
        while (
            state.calls < self.max_evals and self.repeats < _MAX_REPEATS and len(self.seen) < self.space.size
        ):
            batch = self.opt.ask()
            self.batch, cut = batch[: self.max_evals - state.calls], batch[self.max_evals - state.calls :]
            if cut:
                self._cancel(cut)
            self.scores = []
            for cand in self.batch:
                self._evaluate(cand)
            self.opt.tell(self.batch, self.scores)
            self.batch, self.scores = [], []
            self._check_generation()
        if state.calls >= self.max_evals:
            study.stop_reason = "budget"
        elif len(self.seen) >= self.space.size:
            study.stop_reason = "exhausted"
        else:
            study.stop_reason = "repeats"

    def _evaluate(self, cand: Candidate) -> None:
        state = self.state
        cached = self.cache and cand.key in self.seen
        start = time.time()
        clock = time.perf_counter()
        score = self.seen[cand.key] if cached else float(self.function(dict(cand.params)))
        duration = 0.0 if cached else time.perf_counter() - clock
        if self.cache:
            self.seen.setdefault(cand.key, score)
        state.calls += not cached
        self.repeats = self.repeats + 1 if cached else 0
        self.scores.append(score)
        trial = Trial(
            len(state.study.trials), cand.params, score, cached, cand.x, dict(cand.meta), start, duration
        )
        state.study.trials.append(trial)
        best = state.best
        new_best = state.better(score, best.score if best is not None else float("nan"))
        if new_best:
            state.best = trial
        self.generation_trials.append(trial)
        for listener in self.listeners:
            listener.on_trial(state, trial, new_best)

    def _check_generation(self) -> None:
        now = self.opt.generation
        if now == self.generation:
            return
        finished = [
            t for t in self.generation_trials if t.meta.get("generation", self.generation) == self.generation
        ]
        self.generation_trials = [t for t in self.generation_trials if t not in finished]
        self._emit_generation(self.generation, finished)
        self.generation = now

    def _emit_generation(self, generation: int, trials: list[Trial]) -> None:
        population = self.opt.population()
        for listener in self.listeners:
            listener.on_generation(self.state, generation, trials, population)

    def _cancel(self, cands: list[Candidate]) -> None:
        try:
            self.opt.cancel(cands)
        except NotImplementedError:
            pass

    def abandon(self, reason: str) -> None:
        """Tells the evaluated part of an unfinished batch, withdraws the rest and records the reason."""
        self.state.study.stop_reason = reason
        done, rest = self.batch[: len(self.scores)], self.batch[len(self.scores) :]
        try:
            if done:
                self.opt.tell(done, self.scores)
            if rest:
                self._cancel(rest)
        except Exception:
            pass
        self.batch, self.scores = [], []

    def finish(self) -> None:
        """Emits the last, possibly unfinished generation and the end of the run."""
        if self.generation_trials:
            self._emit_generation(self.generation, self.generation_trials)
            self.generation_trials = []
        for listener in self.listeners:
            listener.on_run_end(self.state)
