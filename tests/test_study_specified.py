"""Resolved ambiguities and unspecified behaviour of the results (study.py) and of optimize() (facade.py).

The test pins one question that a plain reading of the API leaves open; its docstring names the spec
sentence that answers it. Studies are built directly from Trial objects, so the scores are exactly the
programmed ones.

Resolved here:
- Does a non-finite score that a naive max() or min() would pick (+inf when maximising, -inf when
  minimising) ever become Study.best_trial? README Behaviour 'A score of NaN or infinity counts as the worst
  possible value for the chosen direction' and the Trial.failed docstring 'True when the user's function
  returned NaN or an infinite value'.

Resolved by existing tests, not repeated:
- Study.best_score under direction='max' is on the user's scale: tests/test_abc.py::
  test_direction_max_and_nan_handling (best_score > 0.95 for an objective in [0, 1]).
- A cached repeat is not a call and does not consume budget (optimize docstring 'max_evals counts the
  calls of function: with cache=True a repeated parameter set reuses the earlier score and does not use up
  the budget'; Study.n_calls docstring 'cached repeats are not counted'): tests/test_abc.py::
  test_cache_skips_repeated_parameter_sets (6 calls, one per parameter set, plus cached trials) and, exactly,
  the equivalence-class test test_optimize_should_call_the_function_once_when_the_space_has_a_single_point.
- max_evals < 1 raises ValueError and max_evals = 1 gives exactly one trial: the error-path test
  test_optimize_should_raise_value_error_when_max_evals_is_below_one and the boundary test
  test_optimize_should_run_exactly_one_evaluation_when_max_evals_is_one. The lower bound is stated only in
  the optimize() error message, not in the README or a docstring.

Unspecified behaviour. No spec source (README, CHANGELOG, docstrings, the original ABC.py, Karaboga &
Basturk 2007) states the following, so no test asserts it. 'Observed in 0.1.0.dev0' notes are not
guaranteed.
- UNSPECIFIED: the base of Trial.number (0 or 1) and of Candidate.id. Observed in 0.1.0.dev0: both start at 0
  and count up by one. Several other items assert that the trials of optimize() are numbered
  0..max_evals - 1; they are listed in this item's conflict report and are settled at integration either by
  a spec sentence that fixes the base or by reducing them to the specified part (len(study.trials) ==
  max_evals). Trials that a test builds itself with chosen numbers do not depend on this.
- UNSPECIFIED: which trial Study.best_trial returns when several finite trials share the best score.
  Observed in 0.1.0.dev0: the first one, for both directions.
- UNSPECIFIED: a user function that modifies the params dict it receives. Observed in 0.1.0.dev0: the change
  is visible in Trial.params and leaks into the params of later candidates, so params can disagree with
  Trial.x.
- UNSPECIFIED: a non-integer max_evals. Observed in 0.1.0.dev0: 2.5 raises TypeError.

Input space: Every list of trials with any scores and cached flags, for both directions.
Subset: One case per resolved ambiguity; unspecified behaviour is listed, not asserted.
"""

import math

import numpy as np
import pytest

from metahpo import Study, Trial

UNUSED_X = np.zeros(1)


def _trials(scores):
    """Trials numbered in list order with the given scores; params and x do not matter here."""
    return [Trial(n, {"n": n}, score, False, UNUSED_X) for n, score in enumerate(scores)]


@pytest.mark.parametrize(
    "direction, scores, expected",
    [
        pytest.param(
            "max", [math.nan, math.inf, -3.0], -3.0, id="max-nan-and-plus-inf-before-a-negative-score"
        ),
        pytest.param(
            "min", [math.nan, -math.inf, 5.0], 5.0, id="min-nan-and-minus-inf-before-a-positive-score"
        ),
    ],
)
def test_study_best_trial_should_be_the_finite_trial_when_the_others_are_non_finite(
    direction, scores, expected
):
    """Given NaN, the naively best infinity and one poor finite score, when best_trial and best_score are
    read, then best_trial is the finite trial (index 2) and best_score is its score.

    Oracle: README Behaviour 'A score of NaN or infinity counts as the worst possible value for the chosen
    direction'; Trial.failed docstring. Inputs: +inf under 'max' and -inf under 'min' are what a plain
    max() / min() over the scores would return, and the finite score (-3.0 under 'max', 5.0 under 'min') is
    poor, so a filter that treated only NaN as failed returns the infinity (site: Study.best_trial finite
    filter). Postcondition: none, the properties are read-only.
    """
    trials = _trials(scores)
    study = Study(direction, trials)

    best_trial, best_score = study.best_trial, study.best_score

    assert best_trial is trials[2]
    assert best_score == expected
