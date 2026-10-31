"""Generated valid inputs for the ABC operators, checked against invariants instead of hand-written values.

Generators (P-T7-25): every case is drawn from a numpy Generator seeded from the test parameters plus a named
SALT_* constant, one distinct salt per test. Unit-cube values mix uniform draws with the ends 0.0, TOP and
1.0; phi mixes uniform draws in [-1, 1] with -1.0, 0.0 and 1.0; minimisation values f mix ordinary
magnitudes of both signs with 0.0, -0.0, +inf, +-1e308 and +-sys.float_info.max; fitness vectors are
non-negative and not all zero.

Budget and size (P-T7-20): every test is parametrized on the number of cases (20 or 300) and on the input
size per case (1 or 500 values) and asserts how many values it generated.

Harness (P-T7-14, P-T7-15, P-T7-18): one generic _run_case(target, *args) calls any operator and never raises;
_fuzz records (id, input, outcome) per case and only after the loop turns each failing case into a line
'Test #<id> IN=<repr> OUT=<ok|Type: msg>'.

Oracles (relations, never the operator formula recomputed): neighbour_value docstring (phi in [-1, 1],
result clipped to [0, 1]); fitness docstring (1 / (1 + f) for f >= 0, 1 + |f| otherwise, +inf gives 0: both
branches decrease in f, the first lies in [0, 1], the second above 1); fitness_scalar docstring ('`fitness`
for one value'); onlooker_probabilities docstring (p_i = 0.9 * fit_i / max(fit) + 0.1, so p lies in
[0.1, 1.0], is 1.0 where fit is largest and keeps the order of fit).

Rounding: onlooker_probabilities is compared with ROUNDING_TOL above 1.0 only; the docstring formula
evaluated left to right can exceed 1.0 by one ulp.

Unspecified, therefore not asserted: NaN inputs, phi outside [-1, 1], unit values outside [0, 1] and negative
fitness values (abnormal inputs belong to another work item).

Input space: Three float64 values for neighbour_value and float64 arrays of any length for fitness and
onlooker_probabilities.
Subset: Seeded generated valid inputs, checked against invariants.
"""

import math
import sys

import numpy as np
import pytest

from metahpo.operators import fitness, fitness_scalar, neighbour_value, onlooker_probabilities

TOP = math.nextafter(1.0, 0.0)
UNIT_ENDS = (0.0, TOP, 1.0)
PHI_ENDS = (-1.0, 0.0, 1.0)
HUGE = sys.float_info.max
F_EDGES = (0.0, -0.0, math.inf, 1e308, -1e308, HUGE, -HUGE)
ROUNDING_TOL = 2 * math.ulp(1.0)
SALT_NEIGHBOUR = 11
SALT_FITNESS = 12
SALT_PROBABILITIES = 13
SIZES = [pytest.param(n_cases, size, id=f"{n_cases}-{size}") for n_cases in (20, 300) for size in (1, 500)]


def _run_case(target, *args, **kwargs):
    """Calls target(*args, **kwargs); returns ('ok', value) or ('error', type name, message), never raises."""
    try:
        return ("ok", target(*args, **kwargs))
    except Exception as exc:
        return ("error", type(exc).__name__, str(exc))


def _failure_line(record, check):
    """The report line of one record, or '' when its outcome is ok and check finds no violated invariant."""
    case_id, case, outcome = record
    problems = check(case, outcome[1]) if outcome[0] == "ok" else []
    if outcome[0] == "ok" and not problems:
        return ""
    shown = "ok VIOLATED=" + repr(problems)[:400] if outcome[0] == "ok" else f"{outcome[1]}: {outcome[2]}"
    return f"Test #{case_id} IN={repr(case)[:400]} OUT={shown}"


def _fuzz(cases, target, check):
    """Runs target on every case, records (id, input, outcome), and after the loop returns the failure
    lines."""
    records = [(case_id, case, _run_case(target, *case)) for case_id, case in enumerate(cases)]
    lines = [_failure_line(record, check) for record in records]
    return [line for line in lines if line]


def _value_count(cases):
    """The total number of generated values: the length of the first input sequence of every case."""
    return sum(len(case[0]) for case in cases)


def _mixed(rng, size, draw, edges):
    """size values: seeded draws with the edge values, in random order, overwriting random positions."""
    values = draw(size)
    positions = rng.permutation(size)[: len(edges)]
    values[positions] = rng.permutation(edges)[: len(positions)]
    return values


def _neighbour_cases(rng, n_cases, size):
    """Cases (x_ij, x_kj, phi) as Python float lists: unit values with the ends and phi with -1, 0 and 1."""
    cases = []
    for _ in range(n_cases):
        x_ij = _mixed(rng, size, rng.random, np.array(UNIT_ENDS))
        x_kj = _mixed(rng, size, rng.random, np.array(UNIT_ENDS[::-1]))
        phi = _mixed(rng, size, lambda n: rng.uniform(-1.0, 1.0, n), np.array(PHI_ENDS))
        cases.append((x_ij.tolist(), x_kj.tolist(), phi.tolist()))
    return cases


def _neighbour_values(x_ij, x_kj, phi):
    """Target: neighbour_value for every triple of the case."""
    return [neighbour_value(a, b, p) for a, b, p in zip(x_ij, x_kj, phi, strict=True)]


def _neighbour_problems(case, values):
    """Violations of 'v in [0, 1], v moves from x_ij by at most |x_ij - x_kj|, in the direction of
    phi * (x_ij - x_kj), and v == x_ij when phi == 0 or x_ij == x_kj'."""
    problems = []
    for a, b, p, v in zip(*case, values, strict=True):
        step = p * (a - b)
        bad_range = type(v) is not float or not 0.0 <= v <= 1.0
        bad_size = abs(v - a) > abs(a - b)
        bad_sign = (v - a) * step < 0.0
        bad_fixed = (p == 0.0 or a == b) and v != a
        if bad_range or bad_size or bad_sign or bad_fixed:
            problems.append((a, b, p, v))
    return problems


def _f_draw(rng):
    """A draw function for f: magnitudes 10**U(-6, 6) with a random sign."""
    return lambda n: rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-6.0, 6.0, n)


def _fitness_cases(rng, n_cases, size):
    """Cases (f,) of float64 arrays mixing ordinary values of both signs with F_EDGES."""
    return [(_mixed(rng, size, _f_draw(rng), np.array(F_EDGES)),) for _ in range(n_cases)]


def _fitness_problems(case, fit):
    """Violations of the fitness branches, of agreement with fitness_scalar and of 'decreasing in f'."""
    (f,) = case
    problems = [] if fit.shape == f.shape else [f"shape {fit.shape}"]
    scalar = [fitness_scalar(float(v)) for v in f]
    if fit.tolist() != scalar:
        problems.append("fitness differs from fitness_scalar")
    upper = (f >= 0) & np.isfinite(f)
    if not np.all((fit[upper] > 0.0) & (fit[upper] <= 1.0)) or not np.all(fit[f < 0] > 1.0):
        problems.append("branch range")
    if not np.all(fit[f == math.inf] == 0.0):
        problems.append("+inf does not give 0")
    order = np.argsort(f, kind="stable")
    if np.any(np.diff(fit[order]) > 0.0):
        problems.append("fitness increases with f")
    return problems


def _probability_cases(rng, n_cases, size):
    """Cases (fit,) of non-negative fitness vectors that are not all zero, with zeros mixed in."""
    cases = []
    for _ in range(n_cases):
        fit = rng.random(size) * 10.0 ** rng.uniform(-3.0, 3.0)
        fit[rng.random(size) < 0.2] = 0.0
        fit[int(rng.integers(size))] = float(rng.uniform(0.5, 2.0))
        cases.append((fit,))
    return cases


def _probability_problems(case, prob):
    """Violations of 'p in [0.1, 1.0], p is 1.0 where fit is largest, p keeps the order of fit', each up to
    ROUNDING_TOL above 1.0."""
    (fit,) = case
    problems = [] if prob.shape == fit.shape else [f"shape {prob.shape}"]
    if not np.all((prob >= 0.1) & (prob <= 1.0 + ROUNDING_TOL)):
        problems.append("p outside [0.1, 1.0]")
    if not np.all(np.abs(prob[fit == fit.max()] - 1.0) <= ROUNDING_TOL):
        problems.append("p of the largest fitness is not 1.0")
    order = np.argsort(fit, kind="stable")
    if np.any(np.diff(prob[order]) < 0.0):
        problems.append("p does not keep the order of fit")
    return problems


@pytest.mark.parametrize(("n_cases", "size"), SIZES)
def test_neighbour_value_should_stay_in_the_unit_interval_and_move_towards_phi_times_difference(
    n_cases, size
):
    """Given generated unit values x_ij, x_kj (ends 0.0, TOP, 1.0 included) and phi in [-1, 1] (-1, 0, 1
    included), when neighbour_value runs, then v is a float in [0, 1], moves from x_ij by at most
    |x_ij - x_kj| in the direction of phi * (x_ij - x_kj), and stays at x_ij when phi or the difference is 0.

    Oracle: neighbour_value docstring (v = x_ij + phi * (x_ij - x_kj), phi in [-1, 1], clipped to [0, 1]);
    clipping only shortens the step, so the relations hold with and without clipping. Postcondition: none.
    """
    cases = _neighbour_cases(np.random.default_rng([n_cases, size, SALT_NEIGHBOUR]), n_cases, size)

    failures = _fuzz(cases, _neighbour_values, _neighbour_problems)

    assert failures == []
    assert _value_count(cases) == n_cases * size


@pytest.mark.parametrize(("n_cases", "size"), SIZES)
def test_fitness_should_agree_with_fitness_scalar_and_decrease_in_f_on_generated_values(n_cases, size):
    """Given generated minimisation values of both signs with 0.0, -0.0, +inf, +-1e308 and +-float max, when
    fitness runs on the array, then it equals fitness_scalar per element, lies in (0, 1] for finite f >= 0,
    above 1 for f < 0, is 0 for +inf, and never increases when f increases.

    Oracle: fitness and fitness_scalar docstrings (both branches of 1 / (1 + f) and 1 + |f| decrease in f;
    fitness_scalar is fitness for one value). Postcondition: none (pure function).
    """
    cases = _fitness_cases(np.random.default_rng([n_cases, size, SALT_FITNESS]), n_cases, size)

    failures = _fuzz(cases, fitness, _fitness_problems)

    assert failures == []
    assert _value_count(cases) == n_cases * size


@pytest.mark.parametrize(("n_cases", "size"), SIZES)
def test_onlooker_probabilities_should_lie_in_point_one_to_one_and_keep_the_order_of_fitness(n_cases, size):
    """Given generated non-negative fitness vectors with zeros and at least one positive value, when
    onlooker_probabilities runs, then every p lies in [0.1, 1.0], p is 1.0 where fit is largest, and p never
    decreases when fit increases.

    Oracle: onlooker_probabilities docstring (p_i = 0.9 * fit_i / max(fit) + 0.1, which is 1.0 for the
    largest fit_i in real arithmetic). The upper end allows ROUNDING_TOL because evaluating 0.9 * fit_i first
    can round to 1.0000000000000002; the exact 1.0 of the original ABC.py (0.9 * (fit_i / max) + 0.1) is
    reported separately as a deviation. Postcondition: none (pure function).
    """
    cases = _probability_cases(np.random.default_rng([n_cases, size, SALT_PROBABILITIES]), n_cases, size)

    failures = _fuzz(cases, onlooker_probabilities, _probability_problems)

    assert failures == []
    assert _value_count(cases) == n_cases * size
