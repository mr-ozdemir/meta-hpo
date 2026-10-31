"""Equivalence classes of the ABC fitness and onlooker formulas in metahpo.operators.

Oracles: the original hyp_optimizer ABC.py (calculate_fitness: 1 / (fun + 1) for fun >= 0, 1 + abs(fun)
otherwise; calculate_probabilities: 0.9 * fit / max_fit + 0.1), Karaboga & Basturk (2007) and the
docstrings of fitness ('+inf gives 0') and onlooker_probabilities ('all-zero fitness gives equal
probabilities').

Class table (feature, viewpoint 'formula classes', class id, definition, representative, expected,
covering test)

fitness and fitness_scalar
  Fi1  f > 0        f = 1.0 (interior)   1 / (1 + f) = 0.5
       tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas (existing)
  Fi2  f == 0       f = 0.0, -0.0        1.0, where both branches meet
       existing test for 0.0; for -0.0
       test_fitness_scalar_should_give_one_on_both_branches_when_f_is_signed_zero
       and test_fitness_should_give_one_on_both_branches_when_f_is_signed_zero
  Fi3  f < 0        f = -2.0 (interior)  1 + |f| = 3.0
       tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas (existing)
  Fi4  f == +inf    f = inf              0.0
       tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas (existing)
onlooker_probabilities
  O1   max > 0      [0.5, 1.0] (ratio 0.5 is interior)   [0.55, 1.0]
       tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas (existing)
  O2   all zero     zeros(3)             ones
       tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas (existing)

Strict partition (P-T1-38): the classes f >= 0 and f < 0 meet at zero. -0.0 compares equal to 0.0 but
carries a sign, so it is the one value that could fall into both classes; the overlap is harmless only
if both formulas agree there (1 / (1 + 0) = 1 + |0| = 1), which the test below checks.

Edge cases: f = 0 and -0.0 (branch meeting point), f = +inf (implicit bound). Corner cases: none.

Not covered (unspecified, so no class): fitness(NaN), fitness(-inf) and negative fitness values in
onlooker_probabilities; the optimizer maps every non-finite score to +inf before calling them.

Input space: Three float64 values for neighbour_value and float64 arrays of any length for fitness and
onlooker_probabilities.
Subset: One representative per equivalence class of the fitness and onlooker formulas.
"""

import numpy as np
import pytest

from metahpo.operators import fitness, fitness_scalar


@pytest.mark.parametrize(
    "f",
    [
        pytest.param(0.0, id="Fi2-positive-zero"),
        pytest.param(-0.0, id="Fi2-negative-zero-discriminating"),
    ],
)
def test_fitness_scalar_should_give_one_on_both_branches_when_f_is_signed_zero(f):
    """Given a signed zero, when fitness_scalar runs, then it returns exactly 1.0.

    Oracle: ABC.py calculate_fitness (fun >= 0 gives 1 / (fun + 1)); -0.0 >= 0 holds, and the other branch
    1 + |-0.0| would also give 1.0, so the overlap of the two classes is consistent. Site: the f >= 0
    comparison of fitness_scalar. Postcondition: none (pure function).
    """
    value = fitness_scalar(f)

    assert value == 1.0


def test_fitness_should_give_one_on_both_branches_when_f_is_signed_zero():
    """Given the array [0.0, -0.0], when fitness runs, then it returns [1.0, 1.0].

    Oracle: ABC.py calculate_fitness and the fitness docstring (1 / (1 + f) for f >= 0, 1 + |f| otherwise);
    both formulas give 1 at the meeting point. Site: the np.where branch selection of fitness, a separate
    implementation from fitness_scalar. Postcondition: none (pure function).
    """
    values = np.array([0.0, -0.0])

    result = fitness(values)

    np.testing.assert_array_equal(result, np.array([1.0, 1.0]))
