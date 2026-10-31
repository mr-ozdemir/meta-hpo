"""Equivalence classes of Trial and Study: score classes, best trial per direction and call counting.

Oracles: Trial.failed docstring ('True when the user's function returned NaN or an infinite value'), README
Behaviour (NaN or infinity counts as the worst value for the direction), the task spec for Study
(best_trial raises ValueError when no trial is finite; best_params and best_score come from it) and the
Study.n_calls docstring ('cached repeats are not counted').

Class table (feature, viewpoint, class id, definition, representative, expected behaviour, covering test)

Trial, score classes
  TF1 finite score    0.5            failed is False
  TF2 NaN             nan            failed is True
  TF3 +inf            inf            failed is True
  TF4 -inf            -inf           failed is True
      test_trial_should_report_failed_only_when_the_score_is_not_finite
Study, best trial per direction
  B1  min with finite trials   [2.0, inf, 1.0, -inf, nan, 3.0]   trial 2, score 1.0
  B2  max with finite trials   same scores                       trial 5, score 3.0
      test_study_should_pick_the_best_finite_trial_for_the_direction_when_infinities_are_present
  B3  no finite trial          [nan, inf]                        ValueError
      test_study_best_trial_should_raise_value_error_when_no_trial_is_finite
Study, call counting (cached flags)
  N1  no trial                 []                     0
  N2  no cached trial          [False, False]         2
  N3  cached and uncached      [False, True, True]    1
      test_study_should_count_only_uncached_trials_as_calls

Edge cases: -inf under 'min' and +inf under 'max' (would be the extreme if not filtered), the empty study.
Corner cases: none.

Not covered (unspecified): ties between equal best scores (which trial wins); best_params identity versus
copy; Study with a direction other than 'min' and 'max'.

Trials are built directly with the public Trial constructor; x is irrelevant to every behaviour here.

Input space: Every list of trials with any scores and cached flags, for both directions.
Subset: One representative per score class, direction and cached flag.
"""

import math

import numpy as np
import pytest

from metahpo import Study, Trial

IRRELEVANT_X = np.zeros(1)
IRRELEVANT_SCORE = 0.0
MIXED_SCORES = [2.0, math.inf, 1.0, -math.inf, math.nan, 3.0]


def _trial(number, score, cached=False):
    return Trial(number, {"n": number}, score, cached, IRRELEVANT_X)


def _study(direction, scores):
    return Study(direction, [_trial(n, s) for n, s in enumerate(scores)])


def _study_with_cached_flags(flags):
    return Study("min", [_trial(n, IRRELEVANT_SCORE, cached) for n, cached in enumerate(flags)])


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        pytest.param(0.5, False, id="TF1-finite"),
        pytest.param(math.nan, True, id="TF2-nan"),
        pytest.param(math.inf, True, id="TF3-plus-inf"),
        pytest.param(-math.inf, True, id="TF4-minus-inf"),
    ],
)
def test_trial_should_report_failed_only_when_the_score_is_not_finite(score, expected):
    """Given a trial with a score of one class, when failed is read, then it is True for non-finite scores.

    Oracle: Trial.failed docstring (NaN or an infinite value). Postcondition: none (read-only property).
    """
    trial = _trial(0, score)

    failed = trial.failed

    assert failed is expected


@pytest.mark.parametrize(
    ("direction", "expected_number", "expected_score"),
    [
        pytest.param("min", 2, 1.0, id="B1-min-ignores-minus-inf"),
        pytest.param("max", 5, 3.0, id="B2-max-ignores-plus-inf"),
    ],
)
def test_study_should_pick_the_best_finite_trial_for_the_direction_when_infinities_are_present(
    direction, expected_number, expected_score
):
    """Given scores [2, inf, 1, -inf, nan, 3], when best_trial is read, then it has the extreme finite score.

    Oracle: README Behaviour (NaN or infinity is the worst value, so never the best) and the task spec
    (failed = non-finite; best_trial among finite trials). -inf under 'min' and +inf under 'max' are the
    discriminating values a plain min/max would return. Postcondition: none (read-only properties).
    """
    study = _study(direction, MIXED_SCORES)

    best = study.best_trial

    assert (best.number, study.best_score, study.best_params) == (
        expected_number,
        expected_score,
        {"n": expected_number},
    )


def test_study_best_trial_should_raise_value_error_when_no_trial_is_finite():
    """Given a study whose trials scored NaN and +inf, when best_trial is read, then ValueError is raised.

    Oracle: task spec (best_trial raises ValueError if no finite trial). Postcondition: none.
    """
    study = _study("min", [math.nan, math.inf])

    with pytest.raises(ValueError):
        _ = study.best_trial


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param([], 0, id="N1-empty-study"),
        pytest.param([False, False], 2, id="N2-no-cached-trial"),
        pytest.param([False, True, True], 1, id="N3-cached-repeats"),
    ],
)
def test_study_should_count_only_uncached_trials_as_calls(flags, expected):
    """Given trials with the listed cached flags, when n_calls is read, then it counts the uncached ones.

    Oracle: Study.n_calls docstring (calls of the user's function; cached repeats are not counted).
    Postcondition: none (read-only property).
    """
    study = _study_with_cached_flags(flags)

    n_calls = study.n_calls

    assert n_calls == expected
