"""Equivalence classes of optimize(): evaluation budget, configuration consistency and a single-point corner.

Oracles: the optimize docstring ('max_evals counts the calls of function'; with cache=True 'a repeated
parameter set reuses the earlier score and does not use up the budget'; default optimizer ABCOptimizer),
README Behaviour (cache), the Study.n_calls docstring and the task spec for the facade (exactly max_evals
trials even inside the initial batch, ValueError for max_evals < 1 or a mismatched optimizer space or
direction, user exceptions propagate). The default colony_size 10 gives SN = 10 // 2 = 5 initial points
(Config.py FOOD_NUMBER = NUMBER_OF_POPULATION / 2).

Class table (feature, viewpoint, class id, definition, representative, expected behaviour, covering test)

optimize, budget (SN = 5)
  M0  max_evals < 1                 0 (edge), -3 (interior)   ValueError, function never called
      test_optimize_should_raise_value_error_without_calling_the_function_when_max_evals_is_below_one
  M1  1 <= max_evals < SN           1 (edge), 2 (interior)    exactly max_evals trials and calls
  M2  max_evals == SN               5                          exactly max_evals trials and calls
  M3  max_evals == SN + 1           6 (first employed bee)     exactly max_evals trials and calls
  M4  several cycles                40 (interior)              exactly max_evals trials and calls
      test_optimize_should_run_exactly_max_evals_uncached_trials_when_max_evals_is_positive
optimize, configuration consistency
  M5  optimizer over another space            ValueError; same dim and names, other bound of x
      test_optimize_should_raise_value_error_when_the_optimizer_searches_another_space
  M6  optimizer with another direction        ValueError
      test_optimize_should_raise_value_error_when_the_optimizer_has_another_direction
  M7  function raises                         the same exception propagates, no further call
      test_optimize_should_propagate_the_objective_exception_when_the_objective_raises
  M8  consistent explicit optimizer, default optimizer: the valid side of M5/M6, used by every budget case

Edge cases: max_evals 0/1, SN, SN + 1. Corner case where several edges meet: Int(2, 2) with a one-choice
Categorical (one point in the whole space) under cache=True and a budget of SN + 1 that crosses from the
initial batch into the employed phase:
      test_optimize_should_call_the_function_once_when_the_space_has_a_single_point

Assertions are seed-independent (counts and the single point); ANY_SEED only makes the runs reproducible.

Not covered (unspecified): an optimizer over an equal but distinct Space object (the spec says 'same
space' without defining equality, so M5 uses a space with the same dimension and names but another
bound); which evaluations the cache skips for a continuous space; max_evals of a non-integer type.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: One representative per budget and configuration class, plus the single-point corner.
"""

from unittest.mock import Mock

import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

ANY_SEED = 0
SN_OF_DEFAULT_COLONY = 10 // 2
CONSTANT_SCORE = 1.0
CALLS_BEFORE_FAILURE = 6


class ObjectiveFailure(Exception):
    """Raised by the objective double to check that optimize lets it through unchanged."""


def _unit_square():
    return Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})


@pytest.mark.parametrize(
    "max_evals",
    [pytest.param(0, id="M0-zero-edge"), pytest.param(-3, id="M0-negative-interior")],
)
def test_optimize_should_raise_value_error_without_calling_the_function_when_max_evals_is_below_one(
    max_evals,
):
    """Given max_evals below 1, when optimize is called, then ValueError names max_evals, no call is made.

    Oracle: task spec (ValueError for max_evals < 1). match= separates it from the optimizer-mismatch
    error of the same call. Postcondition: the objective was never called.
    """
    calls = []

    def spy_objective(params):
        calls.append(params)
        return CONSTANT_SCORE

    with pytest.raises(ValueError, match="max_evals"):
        optimize(spy_objective, _unit_square(), max_evals=max_evals, seed=ANY_SEED)

    assert calls == []


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="M1-one-edge"),
        pytest.param(2, id="M1-inside-initial-batch-interior"),
        pytest.param(SN_OF_DEFAULT_COLONY, id="M2-equal-to-sn"),
        pytest.param(SN_OF_DEFAULT_COLONY + 1, id="M3-sn-plus-one-first-employed-bee"),
        pytest.param(40, id="M4-several-cycles-interior"),
    ],
)
def test_optimize_should_run_exactly_max_evals_uncached_trials_when_max_evals_is_positive(max_evals):
    """Given a budget class and cache=False, when optimize runs, then it makes exactly max_evals trials.

    Oracle: optimize docstring (max_evals counts evaluations), task spec (exact max_evals trials even inside
    the initial batch) and Study.n_calls docstring (every uncached trial is a call). Trial numbers 0..m-1
    show that no trial is lost or repeated. Postcondition: the study and the objective's call record.
    """
    calls = []

    def spy_objective(params):
        calls.append(params)
        return params["x"]

    study = optimize(spy_objective, _unit_square(), max_evals=max_evals, seed=ANY_SEED, cache=False)

    assert (len(study.trials), study.n_calls, len(calls)) == (max_evals, max_evals, max_evals)
    assert [t.number for t in study.trials] == list(range(max_evals))


def test_optimize_should_raise_value_error_when_the_optimizer_searches_another_space():
    """Given an ABCOptimizer over another 2-d space, when optimize gets the unit square, then ValueError.

    Oracle: task spec (ValueError for a mismatched optimizer space) and the facade message 'the same
    space and direction'. The optimizer space has the same dimension 2 and the same names x, y as the
    unit square and differs only in the bound of x (Float(0, 2)), so it is decisive: a check of dim or of
    the names alone would accept it (site: space comparison in optimize). Postcondition: none
    (configuration error).
    """
    optimizer = ABCOptimizer(Space({"x": Float(0.0, 2.0), "y": Float(0.0, 1.0)}), seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space"):
        optimize(lambda params: CONSTANT_SCORE, _unit_square(), max_evals=1, optimizer=optimizer)


def test_optimize_should_raise_value_error_when_the_optimizer_has_another_direction():
    """Given an ABCOptimizer with direction 'max', when optimize runs with 'min', then ValueError is raised.

    Oracle: task spec (ValueError for a mismatched optimizer direction). Postcondition: none.
    """
    space = _unit_square()
    optimizer = ABCOptimizer(space, direction="max", seed=ANY_SEED)

    with pytest.raises(ValueError, match="direction"):
        optimize(lambda params: CONSTANT_SCORE, space, direction="min", max_evals=1, optimizer=optimizer)


def test_optimize_should_propagate_the_objective_exception_when_the_objective_raises():
    """Given an objective failing on its 7th call, when optimize runs 40 evaluations, then the error escapes.

    Oracle: task spec (user exceptions propagate). The 7th call is the second employed bee (after SN = 5
    initial points and one employed bee), so the failure happens inside the single-candidate phases.
    Postcondition: the objective was called exactly 7 times, no call after the failure.
    """
    mock_objective = Mock(
        side_effect=[CONSTANT_SCORE] * CALLS_BEFORE_FAILURE + [ObjectiveFailure("seventh call")]
    )

    with pytest.raises(ObjectiveFailure):
        optimize(mock_objective, _unit_square(), max_evals=40, seed=ANY_SEED, cache=False)

    assert mock_objective.call_count == CALLS_BEFORE_FAILURE + 1


def test_optimize_should_call_the_function_once_when_the_space_has_a_single_point():
    """Given a one-point space and cache=True, when 6 evaluations are allowed, then one function call serves
    the whole initial batch and the run stops.

    Oracle: README Behaviour and the optimize docstring (a repeated parameter set reuses the earlier score;
    the search stops once every parameter set of a space without Float has been evaluated), Study.n_calls
    docstring and the Int / Categorical classes I2 and C2 (the only point is a = 2, b = 'only'). Budget
    SN + 1 = 6 would allow a first employed bee after the initial batch of SN = 5, but the one point is
    evaluated by then, so the 4 repeats of the batch are the only cached trials. Postcondition: the trials,
    their cached flags and the call record.
    """
    space = Space({"a": Int(2, 2), "b": Categorical(["only"])})
    calls = []

    def spy_objective(params):
        calls.append(params)
        return CONSTANT_SCORE

    study = optimize(spy_objective, space, max_evals=SN_OF_DEFAULT_COLONY + 1, seed=ANY_SEED, cache=True)

    assert [t.params for t in study.trials] == [{"a": 2, "b": "only"}] * SN_OF_DEFAULT_COLONY
    assert {type(t.params["a"]) for t in study.trials} == {int}
    assert [t.cached for t in study.trials] == [False] + [True] * (SN_OF_DEFAULT_COLONY - 1)
    assert (study.n_calls, len(calls)) == (1, 1)
