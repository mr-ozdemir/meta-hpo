"""Unit tests of the PSO building blocks in operators.py: velocity, absorbing walls and constriction.

Oracles: the docstrings of operators.pso_velocity (Shi & Eberhart 1998 velocity, clipped to
[-max_velocity, max_velocity]), operators.pso_absorb (positions clipped to [0, 1], velocity of every clipped
coordinate set to 0) and operators.constriction (Clerc & Kennedy 2002, chi = 2 / |2 - phi - sqrt(phi^2 -
4 phi)| for phi > 4). Literal inputs are powers of two where possible, so the expected values are exact.

Input space: Three to ten float64 arrays or numbers per call, which is unbounded.
Subset: Hand-computed literal vectors for each formula, both clip bounds and their interior neighbours, the
exact wall values 0 and 1, and the constriction boundary phi = 4 with values on both sides.
"""

import math

import numpy as np
import pytest

from metahpo import operators

CHI_PHI_4_1 = 0.7298437881283576


def _velocity(max_velocity):
    """pso_velocity on one fixed literal vector with the given clip bound."""
    return operators.pso_velocity(
        np.array([0.125, -0.125]),
        np.array([0.25, 0.75]),
        np.array([0.5, 0.25]),
        np.array([0.75, 0.0]),
        0.5,
        1.0,
        2.0,
        np.array([0.5, 0.5]),
        np.array([0.25, 0.25]),
        max_velocity,
    )


def test_pso_velocity_should_combine_inertia_cognitive_and_social_terms_when_no_bound_is_hit():
    """Given v, x, pbest, gbest, w = 0.5, c1 = 1, c2 = 2, r1 = 0.5 and r2 = 0.25, when the velocity is
    computed with a bound of 1, then each coordinate is w v + c1 r1 (p - x) + c2 r2 (g - x).

    Oracle: pso_velocity docstring. Coordinate 0: 0.0625 + 0.125 + 0.25 = 0.4375; coordinate 1:
    -0.0625 - 0.25 - 0.375 = -0.6875. Distinct coefficients make a swapped term visible.
    """
    expected = [0.4375, -0.6875]

    velocity = _velocity(1.0)

    assert velocity.tolist() == expected


def test_pso_velocity_should_clip_each_coordinate_to_the_bound_on_both_sides():
    """Given the same literal vector as the unclipped test, when the bound is 0.25, then the positive
    coordinate is cut to 0.25 and the negative one to -0.25.

    Oracle: pso_velocity docstring ('clipped to [-max_velocity, max_velocity] coordinate by coordinate').
    """
    expected = [0.25, -0.25]

    velocity = _velocity(0.25)

    assert velocity.tolist() == expected


def test_pso_velocity_should_keep_a_value_exactly_at_the_bound():
    """Given the same literal vector, when the bound equals the largest magnitude 0.6875, then nothing
    changes.

    Oracle: pso_velocity docstring; a bound that equals the value is inside the closed interval.
    """
    expected = [0.4375, -0.6875]

    velocity = _velocity(0.6875)

    assert velocity.tolist() == expected


def test_pso_velocity_should_be_zero_when_every_coefficient_is_zero():
    """Given w = c1 = c2 = 0, when the velocity is computed, then it is 0 in every coordinate.

    Oracle: pso_velocity formula; each term carries one of the three coefficients.
    """
    expected = [0.0, 0.0, 0.0]

    velocity = operators.pso_velocity(
        np.full(3, 0.5), np.zeros(3), np.ones(3), np.ones(3), 0.0, 0.0, 0.0, np.ones(3), np.ones(3), 1.0
    )

    assert velocity.tolist() == expected


def test_pso_absorb_should_place_outside_coordinates_on_the_wall_and_zero_their_velocity():
    """Given positions -0.5, 0.25 and 1.5 with velocities -0.625, 0.125 and 0.75, when the walls absorb them,
    then the positions become 0, 0.25 and 1 and the velocities 0, 0.125 and 0.

    Oracle: pso_absorb docstring. The interior coordinate keeps both values.
    """
    expected = ([0.0, 0.25, 1.0], [0.0, 0.125, 0.0])

    x, v = operators.pso_absorb(np.array([-0.5, 0.25, 1.5]), np.array([-0.625, 0.125, 0.75]))

    assert (x.tolist(), v.tolist()) == expected


def test_pso_absorb_should_keep_the_velocity_of_a_coordinate_exactly_on_a_wall():
    """Given positions exactly 0 and 1, when the walls absorb them, then positions and velocities stay.

    Oracle: pso_absorb docstring: only clipped coordinates lose their velocity, and 0 and 1 are inside the
    closed cube.
    """
    expected = ([0.0, 1.0], [-0.25, 0.25])

    x, v = operators.pso_absorb(np.array([0.0, 1.0]), np.array([-0.25, 0.25]))

    assert (x.tolist(), v.tolist()) == expected


def test_pso_absorb_should_zero_the_velocity_just_outside_each_wall():
    """Given positions one ulp outside each wall, when the walls absorb them, then they land on the walls with
    zero velocity.

    Oracle: pso_absorb docstring; the smallest overshoot is decisive for a strict versus non-strict
    comparison.
    """
    expected = ([0.0, 1.0], [0.0, 0.0])

    x, v = operators.pso_absorb(np.array([-5e-324, np.nextafter(1.0, 2.0)]), np.array([-0.5, 0.5]))

    assert (x.tolist(), v.tolist()) == expected


def test_constriction_should_give_the_clerc_kennedy_chi_for_phi_4_1():
    """Given phi1 = phi2 = 2.05, when chi is computed, then it is 0.72984378813 and chi * 2.05 is the
    standard acceleration 1.49618.

    Oracle: constriction docstring and Clerc & Kennedy (2002).
    """
    expected_chi = CHI_PHI_4_1

    chi = operators.constriction(2.05, 2.05)

    assert chi == pytest.approx(expected_chi, abs=1e-12)
    assert chi * 2.05 == pytest.approx(1.49618, abs=1e-5)


@pytest.mark.parametrize(
    ("phi1", "phi2"),
    [
        pytest.param(2.0, 2.0, id="phi-4"),
        pytest.param(1.0, 1.0, id="phi-2"),
        pytest.param(0.0, 0.0, id="phi-0"),
        pytest.param(math.inf, 1.0, id="phi-inf"),
        pytest.param(math.nan, 3.0, id="phi-nan"),
    ],
)
def test_constriction_should_raise_value_error_when_phi_is_not_a_finite_number_above_4(phi1, phi2):
    """Given phi1 + phi2 <= 4, infinite or NaN, when chi is computed, then ValueError is raised.

    Oracle: constriction docstring ('defined for finite phi > 4'); phi = 4 is the boundary value, and an
    infinite phi would otherwise give NaN.
    """
    with pytest.raises(ValueError, match="phi"):
        operators.constriction(phi1, phi2)


def test_constriction_should_be_defined_just_above_phi_4():
    """Given phi = 4 + 1e-9, when chi is computed, then it is finite and close to 1.

    Oracle: constriction docstring; at phi -> 4 the formula tends to 2 / |2 - 4| = 1.
    """
    expected = 1.0

    chi = operators.constriction(2.0, 2.0 + 1e-9)

    assert chi == pytest.approx(expected, abs=1e-3)
