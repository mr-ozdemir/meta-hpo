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
space.py, Categorical.__post_init__, ValueError "distinct":
    test_space_errors.py::test_categorical_should_raise_value_error_when_two_choices_have_the_same_type_and_compare_equal
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
abc.py, ABCOptimizer._claim (tell and cancel), ValueError "not asked":
    test_abc_protocol.py::test_abc_optimizer_tell_should_raise_value_error_when_candidate_is_unknown_or_already_told
    test_abc_protocol.py::test_abc_optimizer_cancel_should_raise_value_error_and_cancel_nothing_when_one_candidate_is_not_pending
abc.py, ABCOptimizer._claim (tell and cancel), ValueError "not handed out":
    test_abc_protocol.py::test_abc_optimizer_tell_should_raise_value_error_when_a_candidate_of_another_optimizer_has_a_pending_id
study.py, Study.best_trial, ValueError "finite":
    test_study.py::test_study_best_trial_should_raise_value_error_when_no_trial_has_a_finite_score
facade.py, optimize, ValueError "max_evals":
    test_facade.py::test_optimize_should_raise_value_error_when_max_evals_is_below_one
facade.py, optimize, ValueError "same space":
    test_facade.py::test_optimize_should_raise_value_error_when_optimizer_space_or_direction_differs
Pass-through of a user exception raised by the objective (no raise site of its own, optimize has no
try/except around the call):
    test_facade.py::test_optimize_should_propagate_objective_exception_unchanged_when_objective_raises

That is 19 raise sites plus the pass-through. Excluded as unspecified: the NotImplementedError hooks
Parameter.kind, Parameter.decode, Parameter.decode_one, Parameter.to_python (space.py) and Optimizer.ask,
Optimizer.tell (base.py). No spec source gives them a contract. Optimizer.cancel (base.py) has one, because
optimize() relies on it:
    test_base.py::test_optimizer_cancel_should_raise_not_implemented_error_when_called_on_base_class

match= is used only where one call has several ValueError paths: Float and Int each have two, so their
cases carry a keyword. The empty Space has a single ValueError path, so its case checks the type only; the
empty Categorical case does too, and the duplicate-choice cases carry the keyword 'distinct'.

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


class _ListChoice(list):
    """A list subclass: unhashable, and equal to a plain list with the same items, but of another type."""


class _IncomparableChoice:
    """A choice object whose == raises, like a numpy array of several elements in a boolean context."""

    __hash__ = None

    def __eq__(self, other):
        raise RuntimeError("cannot compare")


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param(["a", "a"], id="equal-strings"),
        pytest.param([1, 2, 1], id="equal-ints-apart"),
        pytest.param([0.5, 0.5], id="equal-floats"),
        pytest.param([(1, 2), (1, 2)], id="equal-tuples"),
        pytest.param([[1], [1]], id="equal-unhashable-lists"),
        pytest.param([{"k": 1}, {"k": 1}], id="equal-unhashable-dicts"),
    ],
)
def test_categorical_should_raise_value_error_when_two_choices_have_the_same_type_and_compare_equal(choices):
    """Given a choice list with two choices of the same type that compare equal, when Categorical is built,
    then ValueError ('distinct') is raised.

    Oracle: Categorical docstring ('Choices must be distinct: two choices of the same type that compare
    equal raise ValueError, because the cache tells parameter sets apart by the index of the choice').
    Inputs: hashable and unhashable choices, so both the set lookup and the pairwise comparison are
    exercised; [1, 2, 1] shows the duplicate need not be adjacent.
    """
    with pytest.raises(ValueError, match="distinct") as excinfo:
        Categorical(choices)

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param([1, True, 1.0], id="equal-values-of-different-types"),
        pytest.param([[1], [2], (1,)], id="distinct-unhashable-lists"),
        pytest.param([[1], _ListChoice([1])], id="equal-unhashable-values-of-different-types"),
        pytest.param([_IncomparableChoice(), _IncomparableChoice()], id="comparison-raises"),
        pytest.param([None, "None", 0], id="none-and-lookalikes"),
    ],
)
def test_categorical_should_keep_every_choice_when_choices_differ_in_type_or_cannot_be_compared(choices):
    """Given choices that compare equal only across types, unhashable distinct choices, or choices whose ==
    raises, when Categorical is built, then it keeps every choice object in order.

    Oracle: Categorical docstring (only 'two choices of the same type that compare equal' are rejected) and
    the README ('one of choices'). Inputs: 1 == True == 1.0 holds in Python, but the types differ, so the
    function sees different values; the same holds for the unhashable [1] and a list subclass holding 1;
    a comparison that raises counts as different.
    """
    parameter = Categorical(choices)

    assert len(parameter.choices) == len(choices)
    assert all(kept is given for kept, given in zip(parameter.choices, choices, strict=True))
