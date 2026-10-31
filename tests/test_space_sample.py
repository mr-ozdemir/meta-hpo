"""Space.sample (space.py) with a real numpy Generator: integration with the external random source.

Oracle: the Space.sample docstring ("n uniform points of the unit cube, shape (n, d)"), README ("Optimizers
search the unit cube [0, 1]^d", "The same seed reproduces the same search") and numpy Generator.random,
which returns float64 values in [0, 1). No values are captured from a seeded run; runs are only compared
with each other or checked for seed-independent properties.

Unspecified, not asserted: sample with n = 0 or negative n.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: A few fixed seeds of a real numpy Generator, compared with each other or checked for seed-independent
properties.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

SAMPLE_SEED = 5
OTHER_SEED = 6


def _mixed_space():
    """Every parameter kind, so the column count d = 4 comes from the space, not from one kind."""
    return Space(
        {
            "lr": Float(1e-4, 1e-2, log=True),
            "n": Int(16, 256),
            "act": Categorical(["relu", "tanh"]),
            "k": Categorical([3, 5, 7]),
        }
    )


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(1, id="one-row-stays-2d"),
        pytest.param(7, id="typical"),
        pytest.param(1000, id="many-draws-check-bounds"),
    ],
)
def test_space_sample_should_return_a_float64_unit_cube_array_of_shape_n_by_d_when_given_a_real_generator(n):
    """Given a mixed space and np.random.default_rng, when sample(rng, n) is called, then the result is a
    float64 ndarray of shape (n, 4) with every value in [0, 1).

    n=1 is the smallest row count (must stay 2-D, not (4,)); n=1000 gives 4000 draws for the half-open bounds.
    Postcondition: none besides advancing the given generator.
    """
    space = _mixed_space()

    u = space.sample(np.random.default_rng(SAMPLE_SEED), n)

    assert type(u) is np.ndarray
    assert u.dtype == np.float64
    assert u.shape == (n, space.dim)
    assert space.dim == 4
    assert bool((u >= 0.0).all()) is True
    assert bool((u < 1.0).all()) is True


def test_space_sample_should_return_identical_arrays_when_generators_share_a_seed():
    """Given two generators built from the same seed, when each samples 7 points, then the arrays are equal.

    Oracle: README "The same seed reproduces the same search"; the space adds no randomness of its own.
    """
    space = _mixed_space()

    u = space.sample(np.random.default_rng(SAMPLE_SEED), 7)
    u_again = space.sample(np.random.default_rng(SAMPLE_SEED), 7)

    np.testing.assert_array_equal(u, u_again)


def test_space_sample_should_return_different_arrays_when_generators_have_different_seeds():
    """Given generators with different seeds, when each samples 7 points, then the arrays differ.

    Discriminating input against a sampler that ignores the given generator (site: Space.sample rng use).
    """
    space = _mixed_space()

    u = space.sample(np.random.default_rng(SAMPLE_SEED), 7)
    u_other = space.sample(np.random.default_rng(OTHER_SEED), 7)

    assert np.array_equal(u, u_other) is False
