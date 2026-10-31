"""Reproducibility of the optimizer's random stream, with and without a refill of its uniform block.

Oracles: README Behaviour ('The same seed reproduces the same search'), the Optimizer docstring and the task
specification of optimizers/base.py (numpy Generator, uniforms drawn in blocks of 4096). Seeded runs are
compared only with each other; no candidate value of a seeded run is asserted. OTHER_SEED is a
non-triviality control: if every seed gave the same candidates, equality of two same-seed runs would prove
nothing.

Gray-box risk: information source - base.py _uniform() hands out uniforms from a block of 4096 and refills it
when empty, while Space.sample (initial batch and scouts) draws from the same generator directly; risk - the
draw order or the stream changes across a refill, so a long run is not reproducible; targeted test - a run
short enough to stay inside the first block and a run long enough for several refills.

Unspecified, therefore not asserted: the exact candidates of any seed, and the block size itself (the test
must also pass if uniforms were drawn one by one).

Input space: Every direction string, seed and sequence of told scores, which is unbounded.
Subset: A run short enough to stay inside the first block of uniforms and a run long enough for several
refills.
"""

import numpy as np
import pytest

from metahpo import ABCOptimizer, Float, Space

ANY_SEED = 0
OTHER_SEED = 1


def sum_of_squares(params):
    """Deterministic objective, so equal candidates always get equal scores."""
    return sum(v * v for v in params.values())


def _run(space, seed, n_asks):
    """Default ABCOptimizer on `space` with `seed`: the initial batch plus `n_asks` asks, each told at once.

    Returns the x of every candidate (one row each), their params, best_score and best_params.
    """
    opt = ABCOptimizer(space, seed=seed)
    xs, params = [], []
    initial = opt.ask()
    opt.tell(initial, [sum_of_squares(c.params) for c in initial])
    xs.extend(c.x for c in initial)
    params.extend(c.params for c in initial)
    for _ in range(n_asks):
        batch = opt.ask()
        opt.tell(batch, [sum_of_squares(c.params) for c in batch])
        xs.extend(c.x for c in batch)
        params.extend(c.params for c in batch)
    return np.vstack(xs), params, opt.best_score, opt.best_params


@pytest.mark.parametrize(
    "n_asks",
    [
        pytest.param(20, id="short-no-block-refill"),
        pytest.param(3000, id="long-several-block-refills"),
    ],
)
def test_abc_optimizer_should_repeat_every_candidate_when_run_twice_with_the_same_seed(n_asks):
    """Given two default optimizers with the same seed, when both run n_asks asks, then all candidates match.

    Oracle: README 'The same seed reproduces the same search'. Inputs: 20 asks need about 3 uniforms each
    (coordinate j, partner k, phi), roughly 60 plus a few roulette draws, well inside the first block of 4096,
    so the refill runs 0 times. 3000 asks need at least 3 * 3000 = 9000 > 2 * 4096 uniforms, so the block is
    refilled at least twice; limit 15 lets scouts fire, whose Space.sample draws interleave with the block.
    Then: x of every candidate, every params dict, best_score and best_params are equal, and the OTHER_SEED
    control run differs.
    """
    space = Space({"x0": Float(-5.0, 5.0), "x1": Float(-5.0, 5.0), "x2": Float(-5.0, 5.0)})

    xs_a, params_a, best_score_a, best_params_a = _run(space, ANY_SEED, n_asks)
    xs_b, params_b, best_score_b, best_params_b = _run(space, ANY_SEED, n_asks)
    xs_control, _, _, _ = _run(space, OTHER_SEED, n_asks)

    np.testing.assert_array_equal(xs_b, xs_a)
    assert params_b == params_a
    assert best_score_b == best_score_a
    assert best_params_b == best_params_a
    assert np.array_equal(xs_control, xs_a) is False
