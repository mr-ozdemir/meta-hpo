"""One-call optimisation: `optimize(function, space, direction=...)`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from metahpo.optimizers.abc import ABCOptimizer
from metahpo.optimizers.base import Direction, Optimizer
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
    """
    if max_evals < 1:
        raise ValueError("max_evals must be at least 1")
    opt = optimizer or ABCOptimizer(space, direction=direction, seed=seed)
    if opt.direction != direction or opt.space is not space:
        raise ValueError("the optimizer must use the same space and direction as optimize()")
    study = Study(direction)
    seen: dict[tuple[Any, ...], float] = {}
    calls = repeats = 0
    while calls < max_evals and repeats < _MAX_REPEATS and len(seen) < space.size:
        batch = opt.ask()
        cands, cut = batch[: max_evals - calls], batch[max_evals - calls :]
        if cut:
            try:
                opt.cancel(cut)
            except NotImplementedError:
                pass
        scores = []
        for cand in cands:
            cached = cache and cand.key in seen
            score = seen[cand.key] if cached else float(function(dict(cand.params)))
            if cache:
                seen.setdefault(cand.key, score)
            calls += not cached
            repeats = repeats + 1 if cached else 0
            scores.append(score)
            study.trials.append(Trial(len(study.trials), cand.params, score, cached, cand.x))
        opt.tell(cands, scores)
    return study
