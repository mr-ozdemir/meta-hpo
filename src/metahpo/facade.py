"""One-call optimisation: `optimize(function, space, direction=...)`, and `resume(function, folder)` for a
search that stopped."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np

from metahpo.optimizers.abc import ABCOptimizer
from metahpo.optimizers.base import Candidate, Direction, Optimizer
from metahpo.optimizers.bbpso import BBPSOOptimizer
from metahpo.optimizers.pso import PSOOptimizer
from metahpo.report import (
    DEFAULT_OUTPUT,
    Callback,
    ConsoleReporter,
    ResultWriter,
    RunState,
    _iso,
    _Logged,
    _paused_at,
    _read_config,
    _read_run,
    _rebuild_space,
    _RunLog,
    _space_matches,
    _version,
    resolve_output,
    resolve_verbose,
)
from metahpo.space import Space
from metahpo.study import Study, Trial

Objective = Callable[[dict[str, Any]], float]
OptimizerFactory = Callable[[Any], Optimizer]
Seed = int | Sequence[int | None] | None

_MAX_REPEATS = 1000


@overload
def optimize(
    function: Objective,
    space: Space,
    direction: Direction = "min",
    max_evals: int = 500,
    optimizer: Optimizer | OptimizerFactory | None = None,
    seed: Seed = None,
    cache: bool = True,
    output: str | os.PathLike[str] | None = DEFAULT_OUTPUT,
    verbose: int = 1,
    log_every: int = 1,
    callbacks: Sequence[Callback] | None = None,
    runs: Literal[1] = 1,
) -> Study: ...


@overload
def optimize(
    function: Objective,
    space: Space,
    direction: Direction = "min",
    max_evals: int = 500,
    optimizer: Optimizer | OptimizerFactory | None = None,
    seed: Seed = None,
    cache: bool = True,
    output: str | os.PathLike[str] | None = DEFAULT_OUTPUT,
    verbose: int = 1,
    log_every: int = 1,
    callbacks: Sequence[Callback] | None = None,
    runs: int = 1,
) -> Study | list[Study]: ...


def optimize(
    function: Objective,
    space: Space,
    direction: Direction = "min",
    max_evals: int = 500,
    optimizer: Optimizer | OptimizerFactory | None = None,
    seed: Seed = None,
    cache: bool = True,
    output: str | os.PathLike[str] | None = DEFAULT_OUTPUT,
    verbose: int = 1,
    log_every: int = 1,
    callbacks: Sequence[Callback] | None = None,
    runs: int = 1,
) -> Study | list[Study]:
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

    Repeated runs: `runs` repeats the whole search that many times, each run with a fresh optimizer, its own
    cache and the full `max_evals`; with `runs=1` optimize() returns a Study, with more runs a list of
    Study, one per run. `seed` is None (every run random), one integer (every run uses it) or a sequence
    with one seed per run. An optimizer object holds the state of one search, so it is accepted only with
    `runs=1`; for more runs pass a function that builds one from a seed, for example
    `optimizer=lambda seed: PSOOptimizer(space, direction='max', seed=seed)`. A KeyboardInterrupt stops the
    remaining runs as well. All runs are written to the same results folder (run_001, run_002, ...).

    A run that stopped, for any reason, continues from its results folder with resume().
    """
    if max_evals < 1:
        raise ValueError("max_evals must be at least 1")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be an integer of at least 1")
    seeds = _run_seeds(seed, runs)
    make = _optimizer_factory(optimizer, space, direction, runs)
    folder = resolve_output(output)
    writer = None if folder is None else ResultWriter(folder)
    listeners = _listeners(verbose, log_every, writer, callbacks)

    def starts() -> Iterator[_Start]:
        for number, run_seed in enumerate(seeds, start=1):
            opt = _checked(make(run_seed), direction)
            if opt.space is not space:
                raise ValueError("the optimizer must use the same space and direction as optimize()")
            yield _Start(number, run_seed, opt)

    studies = _search(function, starts(), direction, max_evals, cache, runs, seeds, listeners)
    return studies[0] if runs == 1 else studies


def resume(
    function: Objective,
    folder: str | os.PathLike[str],
    *,
    optimizer: Optimizer | OptimizerFactory | None = None,
    space: Space | None = None,
    max_evals: int | None = None,
    verbose: int | None = None,
    log_every: int = 1,
    callbacks: Sequence[Callback] | None = None,
) -> Study | list[Study]:
    """Continues the search of a results folder that stopped, and returns what optimize() would have returned.

    `folder` is the time-stamped results folder of an optimize() call (the one that holds config.json). The
    space, direction, optimizer and its settings, budget, cache flag, runs, seeds and the state of every
    run's random generator come from the folder; `function` is the objective, which is never stored. A run
    that stopped for any reason, a KeyboardInterrupt, an error in the objective, a killed process or a power
    cut, continues exactly where it stopped: resume() builds the optimizer again, restores the generator
    state of the start of the run (run_NNN/state.json), and answers the trials logged in trials.jsonl from
    the log instead of calling `function`. Because the optimizers draw their random numbers only from that
    generator and depend only on the scores they are told, the search then goes on as if it had never
    stopped, also for a run without a seed. The trial that was running when the run stopped was not logged
    and is evaluated again.

    The result is a Study, or a list of Study when the folder holds several runs: finished runs are
    restored without a call, the unfinished run is continued and runs that never started are run with the
    seeds config.json records. A folder whose runs all finished evaluates nothing. `max_evals` sets a new
    budget for every run; the search then gives the result of a run started with that budget. It must be
    at least the number of calls a run already logged, else ValueError.

    `optimizer` is needed only when the folder was written by an optimizer other than ABCOptimizer,
    PSOOptimizer and BBPSOOptimizer: an Optimizer object (one run) or a function seed -> optimizer, as in
    optimize(); a folder of another optimizer without it raises ValueError. `space` is needed only when a
    Categorical has choices that JSON cannot hold (classes, functions, ...); it must be the space that
    config.json describes. An optimizer or space that does not match the folder raises ValueError.

    Before anything in the folder changes, resume() replays every log on a copy (copy.deepcopy) of its
    optimizer: when a candidate differs from the logged trial (another meta-hpo version, optimizer setting,
    space or an edited log) it raises RuntimeError naming the run and the trial, and the folder stays as it
    is. Then it drops a last line of trials.jsonl that a kill cut short, appends the new trials and writes
    every other file of the folder again from the start, into the same folder; `output` and METAHPO_OUTPUT
    do not apply. 0.5.x folders have no state.json: their runs are repeated from the recorded seed, a run
    without a seed raises ValueError unless `optimizer` is given, their Categorical choices need `space`, a
    non-finite score they logged as null comes back as NaN, and runs that never started run without a seed.
    A state.json that is empty or not valid JSON, as a power cut at the start of a run can leave it, counts as
    missing: the run is repeated from its seed, and without a seed it needs `optimizer`.

    `verbose`, `log_every` and `callbacks` work as in optimize() (verbose None is the default level 1,
    which METAHPO_VERBOSE replaces). The console prints one 'resumed' line for the restored trials of a run
    instead of their usual lines; callbacks receive the events of the restored trials too, with
    RunState.replaying True. Elapsed times, durations and the console clock leave out the time the run was
    stopped (see metahpo.report).
    """
    root = Path(folder)
    config = _read_config(root)
    runs = int(config["runs"])
    direction = cast(Direction, config["direction"])
    cache = bool(config.get("cache", True))
    seeds = [*config["seeds"], *[None] * (runs - len(config["seeds"]))]
    logs = [_read_run(root / f"run_{number:03d}") for number in range(1, runs + 1)]
    budget = _resume_budget(max_evals, config["max_evals"], logs)
    make = _rebuild(optimizer, space, config, direction, runs)
    starts = [
        _resume_start(number, seed, log, make, config, cache, budget, optimizer is not None, root)
        for number, (seed, log) in enumerate(zip(seeds, logs, strict=True), start=1)
    ]
    for log in logs:
        if log is not None:
            log.drop_torn_tail()
    entry = {
        "time": _iso(time.time()),
        "version": _version(),
        "max_evals": budget,
        "trials": sum(len(log.trials) for log in logs if log is not None),
    }
    writer = ResultWriter._continuing(root, config, entry)
    listeners = _listeners(1 if verbose is None else verbose, log_every, writer, callbacks)
    studies = _search(function, starts, direction, budget, cache, runs, seeds, listeners)
    return studies[0] if runs == 1 else studies


def _listeners(
    verbose: int, log_every: int, writer: ResultWriter | None, callbacks: Sequence[Callback] | None
) -> list[Callback]:
    """The console (unless the level is 0), the results writer (when there is one) and the callbacks."""
    level = resolve_verbose(verbose)
    listeners: list[Callback] = []
    if level > 0:
        listeners.append(ConsoleReporter(level, log_every))
    if writer is not None:
        listeners.append(writer)
    listeners.extend(callbacks or [])
    return listeners


def _checked(opt: Any, direction: Direction) -> Optimizer:
    """The optimizer a function or a folder gave, checked for its type and direction."""
    if not isinstance(opt, Optimizer):
        raise TypeError("the optimizer function must return an Optimizer")
    if opt.direction != direction:
        raise ValueError("the optimizer must use the same space and direction as optimize()")
    return opt


class _Log:
    """The logged trials of a run that resume() answers from, in order, each checked against the candidate
    the optimizer hands out."""

    def __init__(self, trials: Sequence[_Logged] = (), where: str = "") -> None:
        self.trials = list(trials)
        self.where = where
        self.next = 0

    def take(self, cand: Candidate, cached: bool, number: int) -> _Logged | None:
        """The logged trial `number` when the log holds it, None after the last one; RuntimeError when the
        candidate is not the one that was logged (another position, cached flag or trial number)."""
        if self.next == len(self.trials):
            return None
        logged = self.trials[self.next]
        if (logged.trial, logged.cached, logged.x) != (number, cached, [float(v) for v in cand.x]):
            raise RuntimeError(
                f"{self.where} cannot be continued: trial {number} differs from the logged trial, so the "
                "folder was written by another meta-hpo version, optimizer, space or cache setting, or its "
                "log was edited"
            )
        self.next += 1
        return logged


@dataclass
class _Start:
    """One run to search: its number, seed and optimizer; for a run that resume() continues also its log,
    start time, earlier pauses and the search seconds at the end of its last logged trial."""

    number: int
    seed: Any
    opt: Optimizer
    log: _Log = field(default_factory=_Log)
    started: float | None = None
    pauses: list[tuple[int, float]] = field(default_factory=list)
    elapsed: float = 0.0


def _search(
    function: Objective,
    starts: Iterable[_Start],
    direction: Direction,
    max_evals: int,
    cache: bool,
    runs: int,
    seeds: list[Any],
    listeners: list[Callback],
) -> list[Study]:
    """The runs of optimize() and resume(), one after the other; a KeyboardInterrupt ends the rest."""
    studies: list[Study] = []
    for start in starts:
        study = Study(direction)
        studies.append(study)
        opt = start.opt
        state = RunState(
            opt.space,
            opt,
            direction,
            max_evals,
            start.seed,
            study,
            run=start.number,
            runs=runs,
            seeds=seeds,
            cache=cache,
            restored=len(start.log.trials),
        )
        if start.started is not None:
            state.started = state._replayed_until = start.started
            state.replaying = state.restored > 0
            state.pauses = [*start.pauses, (state.restored, time.time() - start.started - start.elapsed)]
        run = _Run(function, opt.space, opt, cache, max_evals, state, listeners, start.log)
        for listener in listeners:
            listener.on_run_start(state)
        try:
            run.loop()
        except KeyboardInterrupt:
            run.abandon("interrupted")
            run.finish()
            break
        except BaseException:
            run.abandon("error")
            run.finish()
            raise
        run.finish()
    return studies


_BUILT_IN: dict[str, type[Optimizer]] = {
    cls.__name__: cls for cls in (ABCOptimizer, PSOOptimizer, BBPSOOptimizer)
}


def _resume_budget(max_evals: int | None, recorded: int, logs: list[_RunLog | None]) -> int:
    """The budget of a resume: `max_evals` or the recorded one, at least the calls every run logged."""
    budget = recorded if max_evals is None else max_evals
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("max_evals must be an integer of at least 1")
    for number, log in enumerate(logs, start=1):
        if log is not None and budget < log.needed:
            raise ValueError(
                f"max_evals={budget} is smaller than what run {number} already logged; give at least "
                f"max_evals={log.needed}"
            )
    return int(budget)


def _rebuild(
    optimizer: Optimizer | OptimizerFactory | None,
    space: Space | None,
    config: dict[str, Any],
    direction: Direction,
    runs: int,
) -> OptimizerFactory:
    """A function seed -> optimizer for resume(): the given optimizer, or the one config.json names, built
    with its settings on the given space or on the space config.json describes."""
    if optimizer is not None:
        return _optimizer_factory(optimizer, cast(Space, space), direction, runs)
    name = config["optimizer"]["name"]
    cls = _BUILT_IN.get(name)
    if cls is None:
        raise ValueError(
            f"the folder was written by {name}, which resume() cannot build again; pass optimizer=, an "
            "optimizer object or a function seed -> optimizer as in optimize()"
        )
    if space is None:
        space = _rebuild_space(config["space"])
    if space is None:
        raise ValueError(
            "the search space has a Categorical whose choices config.json stores only as text; pass space="
        )
    settings = config["optimizer"]["settings"]
    built = space
    return lambda seed: cls(built, direction=direction, seed=seed, **settings)


def _resume_start(
    number: int,
    seed: Any,
    log: _RunLog | None,
    make: OptimizerFactory,
    config: dict[str, Any],
    cache: bool,
    budget: int,
    given: bool,
    root: Path,
) -> _Start:
    """The start of one run of a resume: its optimizer with the generator state of the run's start, checked
    by replaying the log on a copy; a run that never logged starts fresh."""
    opt = _checked(make(seed), cast(Direction, config["direction"]))
    if not _space_matches(opt.space, config["space"]):
        raise ValueError(f"the space of the optimizer is not the space that {root / 'config.json'} describes")
    if log is None:
        return _Start(number, seed, opt)
    where = f"run {number} of {root}"
    if log.state is not None:
        opt.rng.bit_generator.state = log.state["generator"]
    elif log.trials and seed is None and not given:
        raise ValueError(
            f"the generator state of {where} is missing or unreadable (state.json) and the run has no seed, "
            "so its search cannot be repeated; pass optimizer= built with the seed it used"
        )
    _check_replay(copy.deepcopy(opt), cache, budget, log.trials, where)
    state = log.state or {}
    started = float(state.get("started", log.trials[0].start if log.trials else time.time()))
    pauses = [(int(n), float(s)) for n, s in state.get("pauses", [])]
    last = log.trials[-1] if log.trials else None
    elapsed = 0.0 if last is None else last.end - started - _paused_at(pauses, last.trial)
    return _Start(number, seed, opt, _Log(log.trials, where), started, pauses, elapsed)


class _ReplayEnd(Exception):
    """Ends the check of a log at the first trial that the log does not hold."""


def _stop(params: dict[str, Any]) -> float:
    raise _ReplayEnd


def _check_replay(opt: Optimizer, cache: bool, max_evals: int, trials: list[_Logged], where: str) -> None:
    """Replays the logged trials on `opt` without listeners; RuntimeError at the first candidate that differs
    from its logged trial, or when the search ends before the log does."""
    study = Study(opt.direction)
    state = RunState(opt.space, opt, opt.direction, max_evals, None, study, cache=cache)
    log = _Log(trials, where)
    try:
        _Run(_stop, opt.space, opt, cache, max_evals, state, [], log).loop()
    except _ReplayEnd:
        return
    if log.next < len(trials):
        raise RuntimeError(
            f"{where} cannot be continued: the search ends after trial {log.next - 1} but the log holds "
            f"{len(trials)} trials, so the folder was written by another meta-hpo version, optimizer, space "
            "or cache setting, or its log was edited"
        )


def _run_seeds(seed: Seed, runs: int) -> list[Any]:
    """One seed per run: None for every run, the same integer for every run, or the given sequence."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return [seed] * runs
    seeds = list(seed)
    if len(seeds) != runs:
        raise ValueError(f"seed holds {len(seeds)} values but runs={runs}; give one seed per run")
    return seeds


def _optimizer_factory(
    optimizer: Optimizer | OptimizerFactory | None, space: Space, direction: Direction, runs: int
) -> OptimizerFactory:
    """A function seed -> optimizer for every run."""
    if optimizer is None:
        return lambda seed: ABCOptimizer(space, direction=direction, seed=seed)
    if isinstance(optimizer, Optimizer):
        if runs > 1:
            raise ValueError(
                "an optimizer object holds the state of one search and cannot be reused for several runs; "
                "pass a function that builds one from a seed, for example "
                "optimizer=lambda seed: PSOOptimizer(space, direction='max', seed=seed)"
            )
        instance = optimizer
        return lambda seed: instance
    if not callable(optimizer):
        raise TypeError("optimizer must be an Optimizer, a function seed -> Optimizer, or None")
    return optimizer


class _Run:
    """The search loop of optimize() and resume() with its bookkeeping, split out so interrupts can tidy up.

    `log` holds the trials a resume answers from the results folder instead of calling the function.
    """

    def __init__(
        self,
        function: Objective,
        space: Space,
        opt: Optimizer,
        cache: bool,
        max_evals: int,
        state: RunState,
        listeners: list[Callback],
        log: _Log,
    ) -> None:
        self.function = function
        self.space = space
        self.opt = opt
        self.cache = cache
        self.max_evals = max_evals
        self.state = state
        self.listeners = listeners
        self.log = log
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
        logged = self.log.take(cand, cached, len(state.study.trials))
        if logged is not None:
            score = self.seen[cand.key] if cached else logged.score
            start, duration = logged.start, logged.duration
            state._replayed_until = logged.end
        else:
            state.replaying = False
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
