"""Evaluation budget of optimize() as exact counts: trials per run and calls of the user's function.

Oracles: the optimize docstring ('max_evals counts the optimizer's evaluations', 'with cache=True a repeated
parameter set reuses the earlier score instead of calling function again') and the task specification
('exact max_evals trials, even inside the initial batch'). KPIs: trials == max_evals exactly, calls of the
function <= max_evals, trial numbers 0 .. max_evals - 1 in order.

Owned elsewhere, not asserted here: max_evals < 1 (error-path tests), max_evals = 3 (tests/test_abc.py) and
the default budget (API smoke test), and which trials are cache hits (facade-integration).

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: Small budgets compared with the number of trials and of calls of the user's function.
"""

import pytest

from metahpo import Float, Space, optimize

ANY_SEED = 0


def _spy_objective():
    """A spy objective (sum of squares) and the list in which it records the params of every call."""
    calls = []

    def spy_objective(params):
        calls.append(params)
        return sum(v * v for v in params.values())

    return spy_objective, calls


def _trial_numbers(study):
    return [t.number for t in study.trials]


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="one-cuts-the-initial-batch-of-5-to-1"),
        pytest.param(50, id="many-over-several-cycles"),
    ],
)
def test_optimize_should_return_exactly_max_evals_numbered_trials_and_call_at_most_that_often(max_evals):
    """Given max_evals, when optimize() runs, then max_evals trials come back in order and calls <= that.

    Oracle: task specification (exact max_evals trials, even inside the initial batch) and the optimize
    docstring (max_evals counts evaluations; the cache can only save calls). Inputs: 1 is the smallest valid
    budget and ends the run inside the default initial batch of colony_size // 2 = 5; 50 runs the facade loop
    many times. The first trial can never be a cache hit, so the function is called at least once; for
    max_evals = 1 the call count is therefore exactly 1.
    """
    space = Space({"x0": Float(-5.0, 5.0), "x1": Float(-5.0, 5.0), "x2": Float(-5.0, 5.0)})
    spy_objective, calls = _spy_objective()

    study = optimize(spy_objective, space, max_evals=max_evals, seed=ANY_SEED)

    assert len(study.trials) == max_evals
    assert _trial_numbers(study) == list(range(max_evals))
    assert 1 <= len(calls) <= max_evals
