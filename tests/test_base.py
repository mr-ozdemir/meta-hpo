"""Unit tests of the common Optimizer contract (optimizers/base.py), including its error path.

Unit tests of the common Optimizer contract (optimizers/base.py), on a one-parameter Space.

Grouped by method: Optimizer.__init__ (direction check), Optimizer.ask and Optimizer.tell (abstract in the
base class), Optimizer.best_params, and the score mapping that Optimizer._internal applies, observed only
through the public best_score and best_params after a single public ask/tell of an ABCOptimizer.

Unspecified, not asserted: best_score before any tell, and the private uniform block size.

Error path of the common optimizer constructor: the direction check in Optimizer.__init__.

Covered raise site: base.py, Optimizer.__init__, ValueError "direction". The full inventory is in the module
docstring of test_space_errors.py.

Unspecified, not asserted: non-string directions such as None. Only 'min' and 'max' are specified.

Input space: Every direction string, seed and sequence of told scores, which is unbounded.
Subset: One-parameter spaces with literal scores for both directions, including NaN and the infinities, and
one invalid direction.
"""

import math

import pytest

from metahpo import ABCOptimizer, Float, Optimizer, Space

ANY_SEED = 0


COLONY_SIZE = 10


def _unit_space():
    """A fresh one-parameter Space; its bounds do not matter for these tests."""
    return Space({"x": Float(0.0, 1.0)})


def test_optimizer_init_should_raise_value_error_when_direction_is_unknown():
    """Given a direction outside {'min', 'max'}, when Optimizer is built, then ValueError is raised.

    Oracle: task specification 'direction min|max' and the Optimizer.__init__ message 'direction must be ...'.
    Input: 'up' is an arbitrary value from the invalid class. Site: Optimizer.__init__ direction check ->
    raise.
    """
    space = _unit_space()

    with pytest.raises(ValueError, match="direction"):
        Optimizer(space, direction="up")


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda opt: opt.ask(), id="ask"),
        pytest.param(lambda opt: opt.tell([], []), id="tell-with-empty-lists"),
    ],
)
def test_optimizer_ask_and_tell_should_raise_not_implemented_error_when_called_on_base_class(call):
    """Given the base Optimizer, when ask() or tell([], []) is called, then NotImplementedError is raised.

    Oracle: the Optimizer docstring defines only the shared ask/tell interface; the search itself belongs to
    the concrete optimizers, so the base methods are abstract. Inputs: empty lists for tell show that the
    error does not come from input validation. Sites: Optimizer.ask raise, Optimizer.tell raise.
    """
    opt = Optimizer(_unit_space(), seed=ANY_SEED)

    with pytest.raises(NotImplementedError):
        call(opt)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda space: Optimizer(space, seed=ANY_SEED), id="base-optimizer"),
        pytest.param(lambda space: ABCOptimizer(space, seed=ANY_SEED), id="abc-optimizer"),
    ],
)
def test_optimizer_best_params_should_be_none_when_nothing_was_told(build):
    """Given a fresh optimizer, when best_params is read before any tell, then it is None.

    Oracle: the public return type 'dict[str, Any] | None' of Optimizer.best_params; before any tell no point
    has a score, so None is the only value the type leaves for 'no best yet'. 'is None' rules out an empty
    dict. Site: Optimizer.best_params, branch 'best_x is None'.
    """
    opt = build(_unit_space())

    params = opt.best_params

    assert params is None


def test_optimizer_best_should_skip_infinite_and_nan_scores_when_maximising():
    """Given the initial batch told [inf, 1.0, 4.0, nan, 2.0] under 'max', when best is read, then it is 4.0.

    Oracle: Optimizer docstring 'a NaN or infinite score becomes the worst possible value for the direction'
    and README Behaviour bullet 1; the maximum of the finite scores is 4.0, at batch position 2. Inputs: +inf
    comes first, so a mapping that only treats NaN as failed reports inf; a missing sign flip for 'max'
    reports -4.0 or 1.0 (sites: Optimizer._internal non-finite check and sign, Optimizer.best_score sign).
    The batch size is colony_size // 2 = 5. One ask and one tell only.
    """
    opt = ABCOptimizer(_unit_space(), direction="max", colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands, [math.inf, 1.0, 4.0, math.nan, 2.0])

    best_score, best_params = opt.best_score, opt.best_params

    assert best_score == 4.0
    assert best_params == cands[2].params


def test_optimizer_best_should_skip_minus_infinite_and_nan_scores_when_minimising():
    """Given the initial batch told [-inf, 3.0, 1.0, nan, 2.0] under 'min', when best is read, then it is 1.0.

    Oracle: Optimizer docstring, a non-finite score becomes +inf when minimising; README Behaviour bullet 1;
    the minimum of the finite scores is 1.0, at batch position 2. Input: -inf is the value a naive minimiser
    takes as best, so a NaN-only failed check reports -inf (site: Optimizer._internal non-finite check).
    One ask and one tell only.
    """
    opt = ABCOptimizer(_unit_space(), direction="min", colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands, [-math.inf, 3.0, 1.0, math.nan, 2.0])

    best_score, best_params = opt.best_score, opt.best_params

    assert best_score == 1.0
    assert best_params == cands[2].params


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param("up", id="arbitrary-word"),
        pytest.param("MIN", id="case-variant-of-min"),
    ],
)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda space, direction: Optimizer(space, direction=direction), id="base-optimizer"),
        pytest.param(
            lambda space, direction: ABCOptimizer(space, direction=direction, seed=ANY_SEED),
            id="abc-optimizer",
        ),
    ],
)
def test_optimizer_should_raise_value_error_when_direction_is_unknown(build, direction):
    """Given a direction other than 'min' or 'max', when an optimizer is built, then ValueError is raised.

    Oracle: Optimizer docstring and the Direction type (only 'min' | 'max'), plus the message 'direction must
    be ...' in Optimizer.__init__. Inputs: 'up' is an arbitrary invalid word; 'MIN' is a discriminating
    case-variant that shows the comparison is exact. ABCOptimizer reaches the same site through its base
    constructor; the keyword 'direction' rules out its own colony_size and limit sites. ANY_SEED is fixed
    because the check does not depend on random draws.
    """
    space = Space({"x": Float(0.0, 1.0)})

    with pytest.raises(ValueError, match="direction") as excinfo:
        build(space, direction)

    assert excinfo.type is ValueError
