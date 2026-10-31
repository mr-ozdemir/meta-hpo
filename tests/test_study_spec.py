"""Specification gaps of Trial and Study beyond tests/test_abc.py, which reads them only through optimize().

Trials are built directly with the public Trial and Study API, so no optimizer is involved. Oracles: the
metahpo.study module docstring 'every evaluated trial and the best one', the Trial.failed and Study.n_calls
docstrings, README Behaviour bullet 1 and the computed spec 'best_trial (ValueError if no finite trial)'.

Unspecified and therefore not asserted: which of several equal best scores best_trial returns, and the
dtype and order of Study.scores.

Input space: Every list of trials with any scores and cached flags, for both directions.
Subset: One test per specification gap of Trial and Study.
"""

import math

import numpy as np
import pytest

from metahpo import Study, Trial

IRRELEVANT_X = np.zeros(1)


def _trial(number, score, cached=False):
    """A Trial whose params record its number, so the chosen trial can be recognised."""
    return Trial(number, {"i": number}, score, cached, IRRELEVANT_X)


def _study(direction, scores, cached=None):
    """A Study holding one trial per score, numbered from 0."""
    flags = cached or [False] * len(scores)
    return Study(direction, [_trial(n, s, c) for n, (s, c) in enumerate(zip(scores, flags, strict=True))])


@pytest.mark.parametrize(
    ("score", "expected_failed"),
    [
        pytest.param(math.nan, True, id="nan"),
        pytest.param(math.inf, True, id="plus_inf"),
        pytest.param(-math.inf, True, id="minus_inf"),
        pytest.param(0.0, False, id="zero"),
        pytest.param(-1e308, False, id="largest_negative_finite"),
        pytest.param(1e308, False, id="largest_positive_finite"),
    ],
)
def test_trial_failed_should_be_true_exactly_for_non_finite_scores(score, expected_failed):
    """Given a Trial with one score, when failed is read, then it is True only for NaN and both infinities.

    Oracle: Trial.failed docstring 'True when the user's function returned NaN or an infinite value.' Classes:
    the three non-finite values and, on the other side, zero (a falsy finite value) and the largest finite
    magnitudes of both signs, the boundary just inside the finite range. test_abc.py only sees NaN.
    Postcondition: none, failed is a pure property.
    """
    trial = _trial(0, score)

    failed = trial.failed

    assert failed is expected_failed


@pytest.mark.parametrize("attribute", ["best_trial", "best_params", "best_score"])
@pytest.mark.parametrize(
    ("direction", "scores"),
    [
        pytest.param("min", [], id="no_trials"),
        pytest.param("max", [math.nan, math.inf], id="only_failed_trials"),
    ],
)
def test_study_best_should_raise_value_error_when_no_trial_is_finite(direction, scores, attribute):
    """Given a Study without trials or with only failed trials, when a best attribute is read, then
    ValueError is raised.

    Oracle: computed spec 'best_trial (ValueError if no finite trial)'; best_params and best_score are the
    best trial's fields (study docstring 'every evaluated trial and the best one'). Classes: zero trials and
    a non-empty list without a finite trial; no existing test reaches this error.
    """
    study = _study(direction, scores)

    with pytest.raises(ValueError):
        getattr(study, attribute)


@pytest.mark.parametrize(
    ("direction", "expected_number", "expected_score"),
    [pytest.param("min", 3, 1.0, id="min"), pytest.param("max", 1, 3.0, id="max")],
)
def test_study_best_trial_should_skip_failed_trials_in_the_users_direction(
    direction, expected_number, expected_score
):
    """Given trials scored [nan, 3, -inf, 1, inf], when the best is read, then it is trial 3 (score 1) for
    'min' and trial 1 (score 3) for 'max'.

    Oracle: README 'A score of NaN or infinity counts as the worst possible value for the chosen direction.'
    and the study docstring. Decisive inputs: -inf would win 'min' and +inf would win 'max' if failed trials
    were compared, and NaN comes first, where it would break a plain min or max. test_abc.py checks only
    that the best trial under 'max' is not failed.
    """
    study = _study(direction, [math.nan, 3.0, -math.inf, 1.0, math.inf])

    best = study.best_trial

    assert best.number == expected_number
    assert study.best_score == expected_score
    assert study.best_params == {"i": expected_number}


def test_study_n_calls_should_not_count_cached_trials():
    """Given four trials of which the second and fourth are cached, when n_calls is read, then it is 2.

    Oracle: Study.n_calls docstring 'Calls of the user's function; cached repeats are not counted.' Class:
    an exact count with cached and uncached trials interleaved; test_abc.py only bounds n_calls by 6 through
    optimize(). The NaN score shows that failed trials still count as calls.
    """
    study = _study("min", [1.0, 1.0, math.nan, 2.0], cached=[False, True, False, True])

    n_calls = study.n_calls

    assert n_calls == 2
