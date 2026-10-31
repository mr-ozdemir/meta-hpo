"""Boundary tests of optimize(): the evaluation budget, extreme scores and the all-NaN run.

Oracles: the task specification of optimize (exactly max_evals trials, even inside the initial batch;
ValueError for max_evals < 1; the default optimizer is ABCOptimizer), the optimize docstring, the README
Behaviour section (NaN or infinity is the worst value for the direction; the same seed reproduces the same
search), the Trial.failed docstring (non-finite score), the Study.best_trial rule (ValueError when no trial is
finite) and the onlooker_probabilities docstring (all-zero fitness gives equal probabilities).

The default colony has FOOD_NUMBER = 10 // 2 = 5 initial sources, so 4, 5 and 6 evaluations are the values
just inside, exactly at and just past the end of the initial batch. Objectives are plain functions that
record their calls (spy) or return a constant (stub); ANY_SEED is fixed for determinism only.

Rare paths are built directly (P-T7-29): a NaN objective makes every fitness 0, which is the only way to
reach the onlooker all-zero guard through the public API.

Unspecified, therefore not asserted: the type of Trial.score when the objective returns a numpy scalar or an
int (only the value and the failed flag are checked). Already covered elsewhere and not repeated here: the
invalid side max_evals = 0 and the mismatched space or direction errors. The infeasible-requirement
inventory (P-T4-07) is in the module docstring of test_abc_boundaries.py.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: Budgets at the edges of the initial batch, extreme scores and an all-NaN run.
"""

import math
import sys

import numpy as np
import pytest

from metahpo import ABCOptimizer, Float, Space, optimize

ANY_SEED = 0
FOOD_NUMBER = 10 // 2
IRRELEVANT_SCORE = 0.0


def _unit_space_3d():
    """Three Float(0, 1) parameters; distinct random points give distinct parameter sets."""
    return Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0), "c": Float(0.0, 1.0)})


def _spy_objective(calls):
    """An objective that records every params dict it receives in `calls` and returns their sum."""

    def spy_objective(params):
        calls.append(params)
        return sum(params.values())

    return spy_objective


def _stub_objective(value):
    """An objective that ignores its params and returns `value`."""

    def stub_objective(params):
        return value

    return stub_objective


def _failed_flags(study):
    """Trial.failed of every trial, in trial order."""
    return [t.failed for t in study.trials]


def _trial_scores(study):
    """Trial.score of every trial, in trial order."""
    return [t.score for t in study.trials]


def _stacked_x(points):
    """The x arrays of trials or candidates stacked into one (n, d) array."""
    return np.array([p.x for p in points])


def test_optimize_should_run_exactly_one_evaluation_when_max_evals_is_one():
    """Given max_evals = 1, when optimize runs, then there is exactly one trial and one objective call.

    Oracle: task specification 'exact max_evals trials (even inside the initial batch)' and 'ValueError for
    max_evals < 1'. 1 is the valid side of that threshold (P-T1-40); the invalid side 0 is in the error tests.
    """
    calls = []

    study = optimize(_spy_objective(calls), _unit_space_3d(), max_evals=1, seed=ANY_SEED)

    assert (len(study.trials), len(calls)) == (1, 1)


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(FOOD_NUMBER - 1, id="one-short-of-the-initial-batch"),
        pytest.param(FOOD_NUMBER, id="exactly-the-initial-batch"),
        pytest.param(FOOD_NUMBER + 1, id="first-evaluation-after-the-initial-batch"),
    ],
)
def test_optimize_should_run_exactly_max_evals_calls_when_the_budget_ends_around_the_initial_batch(max_evals):
    """Given the default colony (5 sources), when optimize runs with 4, 5 or 6 evaluations, then trials and
    calls equal max_evals.

    Oracle: task specification 'exact max_evals trials (even inside the initial batch)'; Study.n_calls
    docstring. The three budgets are both sides of the end of the initial batch (P-T1-40). The parameter sets
    are distinct, so the cache does not reduce the calls.
    """
    calls = []

    study = optimize(_spy_objective(calls), _unit_space_3d(), max_evals=max_evals, seed=ANY_SEED)

    assert (len(study.trials), study.n_calls, len(calls)) == (max_evals, max_evals, max_evals)


def test_optimize_should_evaluate_a_neighbour_of_source_0_when_the_budget_is_one_past_the_initial_batch():
    """Given max_evals = FOOD_NUMBER + 1, when optimize runs, then the last trial differs from trial 0 in at
    most one coordinate.

    Oracle: task specification (after the initial batch the employed phase starts with source 0) and
    Karaboga & Basturk (2007) (a neighbour changes one coordinate j). Trial 0 is initial source 0; a random
    initial point would differ in all three coordinates, so the count is decisive (P-T1-40).
    """
    study = optimize(
        _stub_objective(IRRELEVANT_SCORE), _unit_space_3d(), max_evals=FOOD_NUMBER + 1, seed=ANY_SEED
    )

    assert np.count_nonzero(study.trials[-1].x != study.trials[0].x) <= 1


def test_optimize_should_use_the_given_optimizer_when_its_space_object_and_direction_match():
    """Given an ABCOptimizer on the same Space object with direction 'max', when optimize(direction='max')
    runs, then its trials are that optimizer's initial points.

    Oracle: task specification (optimizer=... is used; only a mismatched space or direction raises) and README
    'the same seed reproduces the same search': a twin optimizer with the same seed asks the same initial
    batch. This is the valid side of the space/direction check (P-T1-44). max_evals = 3 stays inside the
    batch.
    """
    space = _unit_space_3d()
    given = ABCOptimizer(space, direction="max", seed=ANY_SEED)
    twin_batch = ABCOptimizer(space, direction="max", seed=ANY_SEED).ask()

    study = optimize(_stub_objective(0.0), space, direction="max", max_evals=3, optimizer=given)

    np.testing.assert_array_equal(_stacked_x(study.trials), _stacked_x(twin_batch[:3]))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(sys.float_info.max, id="largest-finite-float"),
        pytest.param(-sys.float_info.max, id="most-negative-finite-float"),
        pytest.param(1e308, id="near-largest-finite"),
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(np.float64(2.5), id="numpy-float64-return"),
        pytest.param(7, id="python-int-return"),
    ],
)
def test_optimize_should_record_extreme_finite_scores_as_successful_trials(value):
    """Given an objective returning one extreme finite value, when optimize runs, then every trial keeps that
    score and is not failed.

    Oracle: Trial.failed docstring 'True when the user's function returned NaN or an infinite value', so a
    finite score is a success. The values are the implicit limits of the float type, signed zero and the
    numpy and int return types (P-T1-42).
    """
    objective = _stub_objective(value)

    study = optimize(objective, _unit_space_3d(), max_evals=3, seed=ANY_SEED)

    assert _trial_scores(study) == [value] * 3
    assert _failed_flags(study) == [False] * 3


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.inf, id="plus-infinity"),
        pytest.param(-math.inf, id="minus-infinity"),
        pytest.param(math.nan, id="nan"),
    ],
)
def test_optimize_should_mark_trials_failed_when_the_objective_returns_a_non_finite_score(value):
    """Given an objective returning inf, -inf or NaN, when optimize runs, then every trial is failed.

    Oracle: Trial.failed docstring (NaN or infinite means failed). These are the other side of the finite
    limits in the previous test (P-T1-42).
    """
    objective = _stub_objective(value)

    study = optimize(objective, _unit_space_3d(), max_evals=3, seed=ANY_SEED)

    assert _failed_flags(study) == [True] * 3


@pytest.mark.filterwarnings("error")
def test_optimize_should_finish_the_budget_without_warnings_when_every_score_is_nan():
    """Given a NaN objective, when optimize runs 30 evaluations with warnings turned into errors, then all 30
    trials exist and are failed.

    Oracle: task specification (exactly max_evals trials), Trial.failed docstring, onlooker_probabilities
    docstring 'all-zero fitness gives equal probabilities' (so no division by a zero maximum happens). 30 is
    past 5 initial and 5 employed evaluations, so the onlooker phase with all fitness 0 runs: the all-zero
    guard, a code-level boundary (P-T1-44), reached by construction (P-T7-29).
    """
    objective = _stub_objective(math.nan)

    study = optimize(objective, _unit_space_3d(), max_evals=30, seed=ANY_SEED)

    assert _failed_flags(study) == [True] * 30


@pytest.mark.parametrize("attribute", ["best_trial", "best_params", "best_score"])
def test_study_should_raise_value_error_for_the_best_when_every_score_is_nan(attribute):
    """Given a study whose objective always returned NaN, when a best attribute is read, then ValueError is
    raised.

    Oracle: task specification 'best_trial (ValueError if no finite trial)'; best_params and best_score are
    read from best_trial. An all-failed study is the empty side of 'at least one finite trial' (P-T2-33).
    """
    study = optimize(_stub_objective(math.nan), _unit_space_3d(), max_evals=6, seed=ANY_SEED)

    with pytest.raises(ValueError):
        getattr(study, attribute)
