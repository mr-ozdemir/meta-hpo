"""Error paths and output coverage of the search space.

Error paths of the search-space constructors: Float, Int, Categorical and Space.

Exception coverage inventory of src/metahpo. It lists every raise statement other than NotImplementedError,
with the keyword of its message, the exact exception class and the test or parametrize id that covers it.

space.py, Float.__post_init__, ValueError "low < high":
    test_space_errors.py::test_parameter_should_raise_value_error_when_arguments_are_invalid
    [float-low-equals-high, float-low-above-high]
space.py, Float.__post_init__, ValueError "log-scaled Float":
    same test [float-log-low-zero, float-log-low-negative]
space.py, Int.__post_init__, ValueError "low <= high":
    same test [int-low-above-high]
space.py, Int.__post_init__, ValueError "log-scaled Int":
    same test [int-log-low-zero, int-log-low-negative]
space.py, Categorical.__post_init__, ValueError "at least one choice":
    same test [categorical-empty]
space.py, Space.__init__, ValueError "at least one parameter":
    same test [space-empty]
space.py, Space.__init__, TypeError "not a metahpo parameter":
    test_space_errors.py::test_space_should_raise_type_error_when_a_value_is_not_a_parameter
base.py, Optimizer.__init__, ValueError "direction":
    test_base.py::test_optimizer_should_raise_value_error_when_direction_is_unknown
abc.py, ABCOptimizer.__init__, ValueError "colony_size":
    test_abc_protocol.py::test_abc_optimizer_should_raise_value_error_when_colony_size_or_limit_is_too_small
    [colony-3-boundary, colony-0]
abc.py, ABCOptimizer.__init__, ValueError "limit":
    same test [limit-0-boundary, limit-negative]
abc.py, ABCOptimizer.ask, RuntimeError "initial":
    test_abc_protocol.py::test_abc_optimizer_ask_should_raise_runtime_error_when_all_initial_candidates_are_pending
abc.py, ABCOptimizer.ask, RuntimeError "pending candidate":
    test_abc_protocol.py::test_abc_optimizer_ask_should_raise_runtime_error_when_an_employed_candidate_is_pending
abc.py, ABCOptimizer.tell, ValueError "same length":
    test_abc_protocol.py::test_abc_optimizer_tell_should_raise_value_error_when_lengths_differ
abc.py, ABCOptimizer.tell, ValueError "not asked":
    test_abc_protocol.py::test_abc_optimizer_tell_should_raise_value_error_when_candidate_is_unknown_or_already_told
study.py, Study.best_trial, ValueError "finite":
    test_study.py::test_study_best_trial_should_raise_value_error_when_no_trial_has_a_finite_score
facade.py, optimize, ValueError "max_evals":
    test_facade.py::test_optimize_should_raise_value_error_when_max_evals_is_below_one
facade.py, optimize, ValueError "same space":
    test_facade.py::test_optimize_should_raise_value_error_when_optimizer_space_or_direction_differs
Pass-through of a user exception raised by the objective (no raise site of its own, optimize has no
try/except around the call):
    test_facade.py::test_optimize_should_propagate_objective_exception_unchanged_when_objective_raises

That is 17 raise sites plus the pass-through. Excluded as unspecified: the NotImplementedError hooks
Parameter.kind, Parameter.decode, Parameter.decode_one, Parameter.to_python (space.py) and Optimizer.ask,
Optimizer.tell (base.py). No spec source gives them a contract.

match= is used only where one call has several ValueError paths: Float and Int each have two, so their
cases carry a keyword. Categorical and the empty Space have a single ValueError path per call, so their
cases check the type only.

Unspecified, not asserted: non-finite bounds such as Float(nan, 1.0), and an inverted range combined with
log=True. Which check fires first there is an implementation detail.

Output coverage of Space for Int and Categorical: every possible output value is reached.

The centre of share k, u_k = (k + 0.5) / n, lies away from the floating-point share edges, so with equal
shares it must decode to exactly the k-th value. Only the linear Int is covered: the log-scaled Int has no
equal-share specification, so its outputs are unspecified here. Postconditions concern the returned values
only; Space keeps no state between calls.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: One invalid argument per constructor error, and the share centres of every Int and Categorical output.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(lambda: Float(1.0, 1.0), "low < high", id="float-low-equals-high"),
        pytest.param(lambda: Float(2.0, 1.0), "low < high", id="float-low-above-high"),
        pytest.param(lambda: Float(0.0, 1.0, log=True), "log-scaled Float", id="float-log-low-zero"),
        pytest.param(lambda: Float(-1.0, 1.0, log=True), "log-scaled Float", id="float-log-low-negative"),
        pytest.param(lambda: Int(5, 4), "low <= high", id="int-low-above-high"),
        pytest.param(lambda: Int(0, 10, log=True), "log-scaled Int", id="int-log-low-zero"),
        pytest.param(lambda: Int(-1, 10, log=True), "log-scaled Int", id="int-log-low-negative"),
        pytest.param(lambda: Categorical([]), None, id="categorical-empty"),
        pytest.param(lambda: Space({}), None, id="space-empty"),
    ],
)
def test_parameter_should_raise_value_error_with_its_site_keyword_when_arguments_are_invalid(factory, match):
    """Given one invalid constructor argument, when the parameter or Space is built, then ValueError is
    raised.

    Oracle: README parameter table (Float in [low, high], Int low..high inclusive, Categorical one of
    choices), the Int docstring and the constructor messages in space.py. Inputs: low == high is the
    boundary of Float; Int(5, 4) is one step past the valid Int(3, 3); the log cases keep low < high so
    only the log check (site: low <= 0 guard) can fire, and 0 is its boundary while -1 is the negative
    class. The keyword tells the two Float sites and the two Int sites apart.
    """
    with pytest.raises(ValueError, match=match) as excinfo:
        factory()

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        pytest.param(lambda: Int(3, 3), (3, 3, False), id="int-low-equals-high"),
        pytest.param(
            lambda: Float(1e-12, 1.0, log=True), (1e-12, 1.0, True), id="float-log-low-tiny-positive"
        ),
        pytest.param(lambda: Int(1, 10, log=True), (1, 10, True), id="int-log-low-one"),
    ],
)
def test_parameter_should_keep_its_bounds_when_arguments_are_on_the_valid_side_of_a_boundary(
    factory, expected
):
    """Given the valid neighbour of a raising boundary, when the parameter is built, then it keeps low, high,
    log.

    Oracle: Int docstring 'integers low..high inclusive' (so low == high is valid) and the log messages
    'needs low > 0' in space.py (so any positive low is valid). These cases pair with the raising cases
    above and pin each boundary from the other side. Postcondition: the stored fields equal the arguments.
    """
    parameter = factory()

    assert (parameter.low, parameter.high, parameter.log) == expected


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param({"a": 3}, id="int-literal"),
        pytest.param({"a": Float}, id="class-not-instance"),
        pytest.param({"a": Float(0.0, 1.0), "b": 3}, id="second-entry-invalid"),
    ],
)
def test_space_should_raise_type_error_when_a_value_is_not_a_parameter(mapping):
    """Given a mapping with a value that is not a parameter instance, when Space is built, then TypeError is
    raised.

    Oracle: task specification 'invalid parameters raise ValueError/TypeError' and the message
    'is not a metahpo parameter' in Space.__init__. Inputs: a plain int literal; the Float class instead of
    an instance (site: isinstance check, not a name check); a valid first entry followed by an invalid one,
    which shows a valid entry does not mask a later invalid one. Checking the exact type also shows the
    error is not a ValueError.
    """
    with pytest.raises(TypeError) as excinfo:
        Space(mapping)

    assert excinfo.type is TypeError


def _share_centres(n):
    """Column of the n share centres (k + 0.5) / n, shape (n, 1)."""
    return ((np.arange(n) + 0.5) / n)[:, None]


@pytest.mark.parametrize(
    ("low", "high"),
    [
        pytest.param(-2, 3, id="negative-zero-positive"),
        pytest.param(5, 5, id="single-value"),
    ],
)
def test_space_decode_should_map_each_int_share_centre_to_its_integer(low, high):
    """Given Int(low, high), when the centre of every share is decoded, then the raw values are low..high in
    order.

    Oracle: Int docstring 'integers low..high inclusive; every integer gets an equal share' and Space.decode
    docstring 'integers as whole numbers'. Inputs: -2..3 covers negative, zero and positive values; Int(5, 5)
    is the n = 1 boundary.
    """
    space = Space({"n": Int(low, high)})
    u = _share_centres(high - low + 1)

    raw = space.decode(u)

    np.testing.assert_array_equal(raw[:, 0], np.arange(low, high + 1))


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        pytest.param(-2, 3, [-2, -1, 0, 1, 2, 3], id="negative-zero-positive"),
        pytest.param(5, 5, [5], id="single-value"),
    ],
)
def test_space_rows_should_give_every_integer_as_python_int_for_the_int_share_centres(low, high, expected):
    """Given Int(low, high), when rows() decodes every share centre, then params hold low..high as Python
    ints.

    Oracle: Int docstring (equal shares, low..high inclusive) and Space.rows docstring 'params ... with
    Python types'. The type check is decisive because -2.0 == -2 would hide a float or numpy leak.
    """
    space = Space({"n": Int(low, high)})
    u = _share_centres(high - low + 1)

    keys, params = space.rows(u)

    assert [p["n"] for p in params] == expected
    assert {type(p["n"]) for p in params} == {int}


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param(["a", "b", "c"], id="three"),
        pytest.param(["only"], id="single"),
    ],
)
def test_space_decode_should_map_each_categorical_share_centre_to_its_index(choices):
    """Given Categorical(choices), when the centre of every share is decoded, then the raw values are 0..n-1
    in order.

    Oracle: Categorical docstring 'every choice gets an equal share' and Space.decode docstring 'categories
    as indices'. The single choice is the n = 1 boundary.
    """
    space = Space({"c": Categorical(choices)})
    u = _share_centres(len(choices))

    raw = space.decode(u)

    np.testing.assert_array_equal(raw[:, 0], np.arange(len(choices)))


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param(["a", "b", "c"], id="three"),
        pytest.param(["only"], id="single"),
    ],
)
def test_space_to_dicts_should_give_every_choice_in_order_for_the_categorical_share_centres(choices):
    """Given Categorical(choices), when to_dicts() decodes every share centre, then each choice appears in
    order.

    Oracle: Categorical docstring 'one of choices; every choice gets an equal share' and the README parameter
    table. The single choice is the n = 1 boundary.
    """
    space = Space({"c": Categorical(choices)})
    u = _share_centres(len(choices))

    params = space.to_dicts(u)

    assert params == [{"c": choice} for choice in choices]
