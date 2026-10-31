"""Resolved ambiguities and unspecified behaviour of the optimizer contract (optimizers/base.py), observed
through ABCOptimizer, plus the unspecified behaviour of operators.py and optimizers/abc.py.

Each test pins one question that a plain reading of the API leaves open; its docstring names the spec
sentence that answers it. Only public ask/tell and best_score / best_params are used.

Resolved here:
- Is best_score the internally minimised (negated) value or the user's value when maximising? Optimizer
  docstring 'Scores are given in the user's direction; internally everything is minimised', and the README
  'Ask and tell' example, which prints opt.best_score under direction='max'.
- Does a non-finite score that a naive max() or min() would pick (+inf when maximising, -inf when
  minimising, NaN for both) ever become the best? README Behaviour 'A score of NaN or infinity counts as the
  worst possible value for the chosen direction', CHANGELOG 'NaN and infinite scores are treated as the
  worst value for the chosen direction', and the Optimizer docstring '(-inf when maximising, +inf when
  minimising)'. tests/test_abc.py::test_direction_max_and_nan_handling covers NaN through optimize() only.

Resolved by existing tests, not repeated:
- fitness of +inf is exactly 0 (operators.fitness docstring '+inf gives 0'):
  tests/test_abc.py::test_fitness_and_probabilities_match_the_original_formulas.
- all-zero fitness gives onlooker probabilities [1.0, 1.0, 1.0], not 1 / n (onlooker_probabilities
  docstring 'all-zero fitness gives equal probabilities'; p is a per-source acceptance threshold of the
  roulette sweep, whose largest value is 0.9 * 1 + 0.1 = 1.0): same test. The original ABC.py
  calculate_probabilities computes 0 / 0 here (NaN), so equal probabilities are the library's own
  resolution of that case.
- colony_size < 4 and limit < 1 raise ValueError, colony_size = 4 with limit = 1 is accepted: the error-path
  tests test_abc_optimizer_should_raise_value_error_when_colony_size_or_limit_is_too_small and
  test_abc_optimizer_should_hand_out_two_initial_candidates_when_colony_size_and_limit_are_minimal, and
  tests/test_abc.py::test_protocol_errors (colony_size = 3). These bounds are stated only in the
  ABCOptimizer.__init__ error messages, not in the README or a docstring.

Derived elsewhere, not stated by a README or docstring sentence (kept with the oracle each test cites, so
they are not listed as unspecified below):
- best_params is None before any tell: only the public return annotation dict | None says so, a weak
  oracle (base-study-units, test_optimizer_best_params_should_be_none_when_nothing_was_told; and
  abc-protocol-invariants, test_abc_protocol.py::
  test_abc_optimizer_best_params_should_be_none_when_nothing_was_told).
- a refused call whose whole input is rejected (mismatched lengths, a tell naming only candidates that are
  not pending, an ask while candidates are pending) leaves the protocol state unchanged: the refused-input
  self-loop convention of rule P-T7-49, which abc-mealy-model states as its model precondition and checks
  in test_abc_model.py::test_abc_optimizer_should_leave_protocol_state_unchanged_when_input_is_rejected.

Unspecified behaviour. No spec source (README, CHANGELOG, docstrings, the original ABC.py, Karaboga &
Basturk 2007) states the following, so no test asserts it. 'Observed in 0.1.0.dev0' notes are not
guaranteed.
- UNSPECIFIED: operators.fitness and fitness_scalar of -inf. Observed in 0.1.0.dev0: inf.
- UNSPECIFIED: operators.fitness and fitness_scalar of NaN. Observed in 0.1.0.dev0: nan.
- UNSPECIFIED: whether fitness of +inf emits a numpy warning. Observed in 0.1.0.dev0: none.
- UNSPECIFIED: operators.onlooker_probabilities with negative or mixed-sign fitness. Observed in 0.1.0.dev0:
  all values <= 0 give ones, and a positive maximum with a negative entry gives a negative value
  ([-1.0, 2.0] gives [-0.35, 1.0]). The optimizer never passes such input, since its fitness is >= 0.
- UNSPECIFIED: best_score before any tell. Observed in 0.1.0.dev0: +inf for 'min', -inf for 'max'.
- UNSPECIFIED: best_params when every told score is non-finite. The original ABC.py gives only an analogue
  (lines 64, 87-88: the global best starts at food 0 and is replaced only on a strict '<'). It does not
  cover told order or non-finite scores (ABC.py has no NaN handling, and the library tracks the best in
  told order across all candidates), so this stays unspecified. Observed in 0.1.0.dev0: the params of the
  first told candidate. (best_score is then the worst value for the direction, which follows from the
  Optimizer docstring.)
- UNSPECIFIED: which candidate best_params reports when two told finite scores are equal, outside the one
  case the ABC.py analogue covers. ABC.py lines 64 and 87-88 keep the earliest food when the whole initial
  batch is evaluated in food order, so a tie inside the initial batch told at once in ask order keeps the
  first candidate. Ties told out of ask order (partial initial tells) or in later phases are not covered,
  since ABC.py has no told order. Observed in 0.1.0.dev0: the first told candidate wins in both cases.
- UNSPECIFIED: the state after a tell that is rejected part-way, that is a pending candidate followed by an
  unknown or already-told one in the same call. Observed in 0.1.0.dev0: ABCOptimizer.tell applies the
  candidates before the offending id (best_score changes) and then raises ValueError; the optimizer stays
  usable.
- UNSPECIFIED: whether a refused ask or tell consumes random draws.
- UNSPECIFIED: whether tell modifies the scores sequence or the candidates it receives. Observed in
  0.1.0.dev0: it does not.

Input space: Every direction string, seed and sequence of told scores, which is unbounded.
Subset: One case per resolved ambiguity; unspecified behaviour is listed, not asserted.
"""

import math

import pytest

from metahpo import ABCOptimizer, Float, Space

ANY_SEED = 0
TWO_SOURCES = 4


def _two_source_optimizer(direction):
    """ABCOptimizer on one Float with colony_size=4, so its first ask hands out exactly 2 initial
    candidates."""
    space = Space({"x": Float(0.0, 1.0)})
    return ABCOptimizer(space, direction=direction, colony_size=TWO_SOURCES, seed=ANY_SEED)


@pytest.mark.parametrize(
    "scores, best_index",
    [
        pytest.param([0.3, 0.7], 1, id="larger-told-last"),
        pytest.param([0.7, 0.3], 0, id="larger-told-first"),
    ],
)
def test_optimizer_best_score_should_be_the_users_largest_score_when_direction_is_max(scores, best_index):
    """Given direction='max' and the initial batch told 0.3 and 0.7 in either order, when best_score and
    best_params are read, then best_score is 0.7 (not -0.7) and best_params are those of the 0.7 candidate.

    Oracle: Optimizer docstring 'Scores are given in the user's direction; internally everything is
    minimised'; README 'Ask and tell' prints opt.best_score under direction='max'. Inputs: 0.7 against -0.7
    tells the user scale from the negated internal value (site: Optimizer.best_score sign), and both orders
    rule out a 'last told wins' or 'first told wins' best. Postcondition: best_score and best_params.
    """
    opt = _two_source_optimizer("max")
    cands = opt.ask()

    opt.tell(cands, scores)

    assert opt.best_score == 0.7
    assert opt.best_params == cands[best_index].params


@pytest.mark.parametrize(
    "direction, scores, expected",
    [
        pytest.param("max", [math.inf, -1e6], -1e6, id="max-plus-inf-first"),
        pytest.param("max", [-1e6, math.inf], -1e6, id="max-plus-inf-last"),
        pytest.param("max", [math.nan, -1e6], -1e6, id="max-nan-first"),
        pytest.param("max", [-1e6, math.nan], -1e6, id="max-nan-last"),
        pytest.param("min", [-math.inf, 1e6], 1e6, id="min-minus-inf-first"),
        pytest.param("min", [1e6, -math.inf], 1e6, id="min-minus-inf-last"),
        pytest.param("min", [math.nan, 1e6], 1e6, id="min-nan-first"),
        pytest.param("min", [1e6, math.nan], 1e6, id="min-nan-last"),
    ],
)
def test_optimizer_best_score_should_be_the_finite_score_when_the_other_score_is_non_finite(
    direction, scores, expected
):
    """Given the initial batch told one non-finite and one very poor finite score, when best_score is read,
    then it is the finite score.

    Oracle: README Behaviour 'A score of NaN or infinity counts as the worst possible value for the chosen
    direction'; Optimizer docstring 'a NaN or infinite score becomes the worst possible value for the
    direction (-inf when maximising, +inf when minimising)'. Inputs: +inf under 'max' and -inf under 'min'
    are the values a naive max() / min() would pick, NaN is the other non-finite class, and -1e6 / 1e6 are
    finite but very poor, so any finite score must still win. Non-finite first makes it the provisional best
    that must be replaced; non-finite last must not replace a finite best (site: Optimizer._internal
    non-finite mapping). Postcondition: best_score.
    """
    opt = _two_source_optimizer(direction)
    cands = opt.ask()

    opt.tell(cands, scores)

    assert opt.best_score == expected
