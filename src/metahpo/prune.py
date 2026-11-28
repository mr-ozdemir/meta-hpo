"""Pruning: stopping a trial early when it is clearly worse than the others at the same step.

An objective with a parameter named `trial` receives a RunningTrial. It reports its intermediate values,
such as the validation accuracy after every epoch, with `trial.report(value, step)`, asks
`trial.should_prune()` and stops its trial by raising TrialPruned:

    def train(params, trial):
        model = build(params)
        for epoch in range(20):
            accuracy = fit_one_epoch(model)
            trial.report(accuracy, step=epoch)
            if trial.should_prune():
                raise mh.TrialPruned()
        return accuracy

This is the interface of Optuna. A Pruner decides from the trials of the run so far; three ship with the
library: MedianPruner (the default of optimize()), PercentilePruner and NopPruner, which never prunes.
A pruner of your own subclasses Pruner and overrides prune().
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

from metahpo.optimizers.base import _init_settings, _integer
from metahpo.study import Study


class TrialPruned(Exception):
    """Raised by the objective to end its trial as pruned.

    The trial is recorded with state 'pruned' and its last reported value as its score (NaN when it
    reported none); it counts as a call toward max_evals. An objective without a `trial` parameter may
    raise it as well.
    """


class Pruner:
    """Decides whether a running trial is pruned: subclass it and override prune().

    optimize() asks prune() only when the objective calls `trial.should_prune()`, and never for a trial of
    the optimizer's initial population. Every run works on its own copy (copy.deepcopy) of the pruner. A
    pruner should decide from `study` and `trial` alone: resume() does not ask it again for the trials it
    restores from the results folder, so a pruner that keeps state of its own would decide differently in a
    resumed run.
    """

    def prune(self, study: Study, trial: RunningTrial) -> bool:
        """True when `trial` should be pruned now. `study` holds the trials of the run so far, the running
        trial not included; `trial.intermediate_values` holds what the running trial reported."""
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """The constructor parameters, each read from the attribute of the same name, read the same way as
        Optimizer.settings(). config.json records them with the class name, and resume() builds the pruners
        of the library again from them."""
        return _init_settings(self)


class NopPruner(Pruner):
    """Never prunes: `optimize(..., pruner=NopPruner())` switches pruning off."""

    def prune(self, study: Study, trial: RunningTrial) -> bool:
        return False


class PercentilePruner(Pruner):
    """Prunes a trial whose best intermediate value so far is in the worst `100 - percentile` percent of
    the values the completed trials reported at the same step; the rule and its settings are those of
    Optuna's PercentilePruner.

    `percentile=25.0` keeps a trial only while it is among the best quarter. The reference is the completed
    trials of the run (state 'complete', cached repeats not counted) that reported a value at exactly the
    step the trial reported last; numpy's nanpercentile gives the threshold.

    The trial is not pruned while fewer than `n_startup_trials` trials have completed, while its last step
    is below `n_warmup_steps`, or when fewer than `n_min_trials` completed trials reported that step. After
    the warm-up a trial is judged once per `interval_steps` steps, at the first step it reports in each
    interval counted from `n_warmup_steps`. A trial whose every reported value is NaN is pruned. Steps are
    assumed to start at 0.
    """

    def __init__(
        self,
        percentile: float,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float, np.integer, np.floating)):
            raise ValueError(f"percentile must be a number between 0 and 100, got {percentile!r}")
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError(f"percentile must be a number between 0 and 100, got {percentile!r}")
        self.percentile = float(percentile)
        self.n_startup_trials = _integer("n_startup_trials", n_startup_trials, 0)
        self.n_warmup_steps = _integer("n_warmup_steps", n_warmup_steps, 0)
        self.interval_steps = _integer("interval_steps", interval_steps, 1)
        self.n_min_trials = _integer("n_min_trials", n_min_trials, 1)

    def prune(self, study: Study, trial: RunningTrial) -> bool:
        completed = [t for t in study.trials if t.state == "complete" and not t.cached]
        if not completed or len(completed) < self.n_startup_trials:
            return False
        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False
        values = trial.intermediate_values
        if not _first_in_interval(step, values, self.n_warmup_steps, self.interval_steps):
            return False
        best = _best_value(list(values.values()), study.direction)
        if math.isnan(best):
            return True
        reference = [t.intermediate_values[step] for t in completed if step in t.intermediate_values]
        threshold = _threshold(reference, study.direction, self.percentile, self.n_min_trials)
        if math.isnan(threshold):
            return False
        return best > threshold if study.direction == "min" else best < threshold


class MedianPruner(PercentilePruner):
    """Prunes a trial whose best intermediate value so far is worse than the median of the values the
    completed trials reported at the same step: PercentilePruner with percentile 50, as Optuna's
    MedianPruner. It is the default pruner of optimize(); it prunes only trials that report and ask."""

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        super().__init__(50.0, n_startup_trials, n_warmup_steps, interval_steps, n_min_trials=n_min_trials)


class RunningTrial:
    """The trial being evaluated, as an objective with a parameter named `trial` receives it.

    `number` is the trial's number in the run and `params` a copy of its parameters. report() records an
    intermediate value, should_prune() asks the run's pruner, and raising TrialPruned ends the trial as
    pruned. The handle works only while the objective runs; afterwards report() and should_prune() raise
    RuntimeError.
    """

    def __init__(
        self, number: int, params: dict[str, Any], study: Study, pruner: Pruner, protected: bool = False
    ) -> None:
        self.number = number
        self.params = dict(params)
        self._study = study
        self._pruner = pruner
        self._protected = protected
        self._values: dict[int, float] = {}
        self._repeated: list[int] = []
        self._open = True

    @property
    def intermediate_values(self) -> dict[int, float]:
        """A copy of the reported values, step -> value, in the order they were reported."""
        return dict(self._values)

    @property
    def last_step(self) -> int | None:
        """The largest step reported so far; None before the first report."""
        return max(self._values) if self._values else None

    def report(self, value: float, step: int) -> None:
        """Records `value` (anything float() converts) as the intermediate value at `step`, a Python or
        numpy integer >= 0, such as the epoch. A step reported before keeps its first value; the later ones
        are ignored, and optimize() warns about them once when the call has finished. ValueError for a
        value float() does not convert or a step that is not such an integer."""
        self._check_open()
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"the reported value must be a number, got {value!r}") from None
        index = _integer("step", step, 0)
        if index in self._values:
            self._repeated.append(index)
            return
        self._values[index] = number

    def should_prune(self) -> bool:
        """True when the run's pruner says to prune the trial now; the objective then raises TrialPruned.
        Always False for a trial of the optimizer's initial population, which is never pruned, and before
        the first report."""
        self._check_open()
        if self._protected or not self._values:
            return False
        return bool(self._pruner.prune(self._study, self))

    def _check_open(self) -> None:
        if not self._open:
            raise RuntimeError(f"trial {self.number} has finished; its handle can no longer be used")

    def _close(self) -> None:
        self._open = False


def _first_in_interval(step: int, values: dict[int, float], warmup: int, interval: int) -> bool:
    """True when `step` is the first step reported in its interval of `interval` steps counted from
    `warmup`, as in Optuna: no other reported step lies between the start of that interval and `step`."""
    start = (step - warmup) // interval * interval + warmup
    earlier = functools.reduce(lambda last, s: s if last < s != step else last, values, -1)
    return earlier < start


def _best_value(values: list[float], direction: str) -> float:
    """The best of the reported values in the run's direction, NaN values left out; NaN when all are NaN."""
    known = [v for v in values if not math.isnan(v)]
    if not known:
        return math.nan
    return min(known) if direction == "min" else max(known)


def _threshold(values: list[float], direction: str, percentile: float, n_min_trials: int) -> float:
    """The value at `percentile` of the reference values counted from the best end (from the top when
    maximising), NaN values left out; NaN when fewer than `n_min_trials` values are given or all are NaN."""
    if len(values) < n_min_trials or all(math.isnan(v) for v in values):
        return math.nan
    q = percentile if direction == "min" else 100.0 - percentile
    return float(np.nanpercentile(np.array(values, dtype=np.float64), q))


def _pruner(pruner: Pruner | None) -> Pruner:
    """The pruner of optimize(pruner=...): MedianPruner() for None, else a Pruner; TypeError otherwise."""
    if pruner is None:
        return MedianPruner()
    if not isinstance(pruner, Pruner):
        raise TypeError("pruner must be a Pruner or None")
    return pruner
