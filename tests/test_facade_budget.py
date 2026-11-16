"""Evaluation budget of optimize() as exact counts: trials per run and calls of the user's function.

Oracles: the optimize docstring ('max_evals counts the calls of function', 'with cache=True a repeated
parameter set reuses the earlier score and does not use up the budget', 'The search stops early only when
the space is exhausted'), the Optimizer docstring (a repeat comes back only when `max_tries` attempts find no
new parameter set) and the task specification ('exact max_evals trials, even inside the initial batch').
KPIs: calls of the function == max_evals unless the space is exhausted, trials >= calls, trial numbers
0, 1, 2, ... in order.

Owned elsewhere, not asserted here: max_evals < 1 (error-path tests), max_evals = 3 (tests/test_abc.py) and
the default budget (API smoke test), and which trials are cache hits (facade-integration).

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: Small budgets compared with the number of trials and of calls of the user's function, a mixed space
whose optimizer hands out repeats, and an exhausted discrete space with and without the cache; budgets 30
and 50 of the three optimizers, with and without the cache and Patience, on a space where a budget that cuts
a PSO flight keeps a cached repeat before the cut.
"""

import pytest

from metahpo import (
    ABCOptimizer,
    BBPSOOptimizer,
    Categorical,
    Float,
    Int,
    Patience,
    PSOOptimizer,
    Space,
    optimize,
)

ANY_SEED = 0
MIXED_BUDGET = 60
EXHAUSTED_BUDGET = 300
DISCRETE_SIZE = 3 * 2


def _spy_objective():
    """A spy objective (sum of squares) and the list in which it records the params of every call."""
    calls = []

    def spy_objective(params):
        calls.append(params)
        return sum(v * v for v in params.values())

    return spy_objective, calls


def _trial_numbers(study):
    return [t.number for t in study.trials]


def _discrete_space():
    """Int(1, 3) x Categorical(['p', 'q']): 6 parameter sets, far fewer than EXHAUSTED_BUDGET."""
    return Space({"a": Int(1, 3), "b": Categorical(["p", "q"])})


def _mixed_space():
    """The discrete space plus a Float, so the space is never exhausted."""
    return Space({"a": Int(1, 3), "b": Categorical(["p", "q"]), "c": Float(0.0, 1.0)})


def _spy_mixed_objective():
    """A spy objective for the discrete and mixed spaces (a, plus 1 for 'q', plus c when present) and the
    list in which it records the params of every call."""
    calls = []

    def spy_mixed_objective(params):
        calls.append(params)
        return params["a"] + (params["b"] == "q") + params.get("c", 0.0)

    return spy_mixed_objective, calls


def _distinct(calls):
    return len({tuple(sorted(params.items())) for params in calls})


CUT_SEEDS = (1, 3, 4)
"""Seeds whose PSO run with max_evals 30 on the cut space keeps a cached repeat before the cut of its second
flight, so 0.7.0 asked the swarm again and its 31st trial was not the 31st trial of a longer run."""

CUT_OPTIMIZERS = [
    pytest.param(lambda space, seed: ABCOptimizer(space, "max", seed=seed), id="abc"),
    pytest.param(lambda space, seed: PSOOptimizer(space, "max", seed=seed), id="pso"),
    pytest.param(lambda space, seed: BBPSOOptimizer(space, "max", seed=seed), id="bbpso"),
]


def _cut_space():
    """Int(1, 4) x Categorical(['gini', 'entropy']) x Float(0, 1): particles that sit on a wall of the Float
    hand out repeats, which the cache answers without a call."""
    return Space({"depth": Int(1, 4), "crit": Categorical(["gini", "entropy"]), "f": Float(0.0, 1.0)})


def _rounded(params):
    """A score rounded to two decimals, so many parameter sets tie."""
    return round(params["depth"] * 0.1 + (params["crit"] == "gini") * 0.05 + params["f"] * 0.01, 2)


def _cut_run(make, seed, max_evals, cache, stop):
    """optimize() maximising _rounded on the cut space with the optimizer `make` builds."""
    space = _cut_space()
    return optimize(
        _rounded, space, "max", max_evals, optimizer=make(space, seed), cache=cache, output=None, stop=stop
    )


def _identity(study):
    """Every trial's number, unit-cube position, cached flag and exact score."""
    return [(t.number, t.x.tolist(), t.cached, repr(t.score)) for t in study.trials]


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


def test_optimize_should_call_the_function_exactly_max_evals_times_when_the_optimizer_hands_out_repeats():
    """Given a mixed space and an ABCOptimizer with max_tries=1, when optimize() runs with max_evals = 60,
    then the function is called exactly 60 times, Study.n_calls is 60 and the cached repeats come on top as
    extra trials, numbered in order.

    Oracle: optimize docstring ('max_evals counts the calls of function'; a cached repeat 'does not use up
    the budget') and Study.n_calls docstring. Input: with max_tries=1 a neighbour that keeps the Int or
    Categorical value is handed out as it is (Optimizer docstring), so repeats occur (asserted); the Float
    keeps the space from being exhausted, so the run can only end on the budget.
    """
    space = _mixed_space()
    optimizer = ABCOptimizer(space, seed=ANY_SEED, max_tries=1)
    spy_objective, calls = _spy_mixed_objective()

    study = optimize(spy_objective, space, max_evals=MIXED_BUDGET, optimizer=optimizer)
    assert len(study.trials) > MIXED_BUDGET

    assert (study.n_calls, len(calls)) == (MIXED_BUDGET, MIXED_BUDGET)
    assert _distinct(calls) == MIXED_BUDGET
    assert _trial_numbers(study) == list(range(len(study.trials)))


def test_optimize_should_stop_after_one_call_per_parameter_set_when_a_discrete_space_is_exhausted():
    """Given Int(1, 3) x Categorical(['p', 'q']) with 6 parameter sets and the default cache, when optimize()
    runs with max_evals = 300, then the function is called exactly 6 times, once with each parameter set,
    and the run ends long before 300 trials.

    Oracle: optimize docstring ('The search stops early only when the space is exhausted: every parameter
    set of a space without Float has been evaluated'), Space.size (the product 3 * 2) and the Study.n_calls
    docstring. A loop that waited for 300 calls would never end on this space; one that stopped on trials
    would call the function 300 times without the cache.
    """
    space = _discrete_space()
    spy_objective, calls = _spy_mixed_objective()

    study = optimize(spy_objective, space, max_evals=EXHAUSTED_BUDGET, seed=ANY_SEED)

    assert (study.n_calls, len(calls), _distinct(calls)) == (DISCRETE_SIZE, DISCRETE_SIZE, DISCRETE_SIZE)
    assert space.size == DISCRETE_SIZE
    assert len(study.trials) < EXHAUSTED_BUDGET


def test_optimize_should_call_the_function_max_evals_times_when_cache_is_false_on_an_exhausted_space():
    """Given the same 6-set discrete space with cache=False, when optimize() runs with max_evals = 300, then
    the function is called 300 times, repeats included, and no trial is cached.

    Oracle: optimize docstring (only `cache=True` reuses the earlier score, so without it every repeat is a
    call of the function and uses up the budget) and the Study.n_calls docstring. Pair of the exhausted-space
    test above: the early stop is a property of the cache, so 300 calls over 6 sets must repeat sets.
    """
    space = _discrete_space()
    spy_objective, calls = _spy_mixed_objective()

    study = optimize(spy_objective, space, max_evals=EXHAUSTED_BUDGET, seed=ANY_SEED, cache=False)

    assert (len(study.trials), study.n_calls, len(calls)) == (EXHAUSTED_BUDGET,) * 3
    assert _distinct(calls) == DISCRETE_SIZE
    assert not any(t.cached for t in study.trials)


@pytest.mark.parametrize("seed", CUT_SEEDS)
@pytest.mark.parametrize(
    "stop", [pytest.param(None, id="no-rule"), pytest.param(Patience(12), id="patience")]
)
@pytest.mark.parametrize("cache", [pytest.param(True, id="cache"), pytest.param(False, id="no-cache")])
@pytest.mark.parametrize("make", CUT_OPTIMIZERS)
def test_optimize_should_try_the_first_trials_of_a_larger_budget_when_its_budget_cuts_a_batch(
    make, cache, stop, seed
):
    """Given each optimizer with and without the cache and with no rule or Patience(12), on the cut space
    with the seeds 1, 3 and 4, when optimize() runs with max_evals 30 and with 50, then the trials of the
    smaller budget are the first trials of the larger one: number, position, cached flag and score.

    Oracle: optimize docstring (max_evals counts the calls of function; the candidates of a batch the budget
    cuts short that are not evaluated are withdrawn) and resume docstring (a new max_evals gives the result
    of a run started with that budget, which needs the trials of the smaller budget to come first).
    Decisive: PSO's swarm of 20 makes the budget of 30 cut the second flight, and for these seeds the part
    before the cut holds a cached repeat; a loop that cuts the batch to the remaining calls first and asks
    the swarm again for the call left over gives a 31st trial that the run with 50 never tries.
    """
    small = _cut_run(make, seed, 30, cache, stop)

    large = _cut_run(make, seed, 50, cache, stop)

    assert _identity(small) == _identity(large)[: len(small.trials)]
