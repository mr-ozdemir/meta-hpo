"""Optimizer base tests added after the stdlib coverage measurement: direction check and the reported best.

They close base line 38 (edge 37L, unknown direction) and the min/max sides of best_score at base line 65,
plus the dict side of best_params at base line 61, through the public ABCOptimizer.

Oracles: the Optimizer docstring ('Scores are given in the user's direction; internally everything is
minimised'), the task specification (direction 'min' | 'max'; best_params and best_score) and the library's
convention that invalid argument values raise ValueError.

Unspecified, therefore not asserted here (listed as excluded in the coverage report): that the abstract
Optimizer.ask and Optimizer.tell raise NotImplementedError (base lines 48 and 57, no spec sentence), and the
value of best_params before any tell (base line 61, None side; only the return annotation hints at it).

Input space: Every direction string, seed and sequence of told scores, which is unbounded.
Subset: The base branches that the coverage measurement found unreached.
"""

import pytest

from metahpo import ABCOptimizer, Float, Space

ANY_SEED = 0


def _space():
    """Two linear floats; the content is irrelevant to the direction and best bookkeeping."""
    return Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param("maximize", id="near_miss_spelling"),
        pytest.param("MIN", id="wrong_case_of_a_valid_value"),
    ],
)
def test_optimizer_should_raise_value_error_when_direction_is_neither_min_nor_max(direction):
    """Given a direction outside {'min', 'max'}, when ABCOptimizer is built, then ValueError is raised.

    Oracle: task specification 'direction min|max' with ValueError for invalid arguments. Inputs: a near-miss
    spelling and a case variant of a valid value are the likeliest user mistakes. Site: base line 38 (edge
    37L).
    """
    space = _space()

    with pytest.raises(ValueError):
        ABCOptimizer(space, direction=direction, seed=ANY_SEED)


@pytest.mark.parametrize(
    ("direction", "best_index", "expected_score"),
    [
        pytest.param("min", 1, -2.0, id="min_reports_smallest"),
        pytest.param("max", 2, 7.5, id="max_reports_largest_not_negated"),
    ],
)
def test_optimizer_best_should_be_the_extreme_told_score_and_its_params_in_the_user_direction(
    direction, best_index, expected_score
):
    """Given scores 3.0, -2.0, 7.5, 0.0, 1.0 told for the initial batch, when best_* is read, then the
    extreme.

    Oracle: Optimizer docstring 'Scores are given in the user's direction; internally everything is
    minimised', so best_score is the told value itself (7.5 for max, not -7.5) and best_params are those of
    the candidate that received it. Inputs: mixed signs make a missing or doubled negation visible. Sites:
    base line 65 both sides, line 61 dict side.
    """
    opt = ABCOptimizer(_space(), direction=direction, seed=ANY_SEED)
    batch = opt.ask()
    opt.tell(batch, [3.0, -2.0, 7.5, 0.0, 1.0])

    best_score = opt.best_score
    best_params = opt.best_params

    assert best_score == expected_score
    assert best_params == batch[best_index].params
