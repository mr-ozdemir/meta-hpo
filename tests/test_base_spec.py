"""Specification gaps of the common optimizer interface (metahpo.optimizers.base) beyond tests/test_abc.py.

ABCOptimizer is the only concrete Optimizer, so it stands in for the interface. Every test names the class
it adds beyond test_abc.py. Oracles: README.md, the Optimizer and Candidate docstrings and the computed spec
of the ask/tell interface. Seeded runs are only compared with each other or checked for seed-independent
properties; no seeded trajectory is asserted.

Unspecified and therefore not asserted: best_params and best_score before the first tell, best_score when
every told score is non-finite, whether best_score also counts candidates that ABC rejected, and the
numeric order of candidate ids (only their uniqueness follows from the protocol).

Input space: Every direction string, seed and sequence of told scores, which is unbounded.
Subset: One test per specification gap of the common optimizer interface.
"""

import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space

ANY_SEED = 0
SHARED_SEED = 11


def _unit_space(d):
    """d linear floats on [0, 1]; the decoded value equals the unit coordinate."""
    return Space({f"x{i}": Float(0.0, 1.0) for i in range(d)})


def _drive(opt, n_asks, score_of):
    """Asks and tells n_asks single candidates after the initial batch; returns the asked candidates."""
    asked = []
    for _ in range(n_asks):
        (cand,) = opt.ask()
        opt.tell([cand], [score_of(cand)])
        asked.append(cand)
    return asked


def _finish_initial_batch(opt, score_of):
    """Asks the initial batch and tells score_of for each candidate; returns the batch."""
    cands = opt.ask()
    opt.tell(cands, [score_of(c) for c in cands])
    return cands


def _coordinate_sum(cand):
    return float(cand.x.sum())


def _scores(cands, score_of):
    """score_of of every candidate, in asking order."""
    return [score_of(c) for c in cands]


def _points(cands):
    """The x of every candidate stacked into an (n, d) array, in asking order."""
    return np.stack([c.x for c in cands])


def _keys(cands):
    return [c.key for c in cands]


def _params(cands):
    return [c.params for c in cands]


def _ids(cands):
    return [c.id for c in cands]


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param("maximize", id="long_word"),
        pytest.param("MIN", id="wrong_case"),
        pytest.param("", id="empty"),
    ],
)
def test_optimizer_should_raise_value_error_when_direction_is_not_min_or_max(direction):
    """Given a direction other than 'min' or 'max', when the optimizer is built, then ValueError is raised.

    Oracle: computed spec "direction 'min'|'max'" and README, which names only those two words. Class: the
    invalid side of direction; test_abc.py only uses valid directions. 'MIN' shows that case matters.
    """
    space = _unit_space(2)

    with pytest.raises(ValueError):
        ABCOptimizer(space, direction=direction, seed=ANY_SEED)


@pytest.mark.parametrize(
    ("direction", "bad_score", "best_index", "expected_best"),
    [
        pytest.param("min", math.nan, 2, 1.0, id="min_nan"),
        pytest.param("min", math.inf, 2, 1.0, id="min_plus_inf"),
        pytest.param("min", -math.inf, 2, 1.0, id="min_minus_inf"),
        pytest.param("max", math.inf, 4, 5.0, id="max_plus_inf"),
        pytest.param("max", -math.inf, 4, 5.0, id="max_minus_inf"),
    ],
)
def test_optimizer_should_treat_non_finite_score_as_worst(direction, bad_score, best_index, expected_best):
    """Given an initial batch whose first score is NaN or an infinity, when the batch is told, then the best
    score and params are those of the best finite score in the user's direction.

    Oracle: README 'A score of NaN or infinity counts as the worst possible value for the chosen direction.'
    and the Optimizer docstring '(-inf when maximising, +inf when minimising)'. The bad score comes first, so
    it would win if it were not mapped to the worst value. Decisive classes: -inf under 'min' and +inf under
    'max', which win a naive comparison; NaN with 'max' is left out because test_abc.py covers it.
    """
    space = _unit_space(3)
    opt = ABCOptimizer(space, direction=direction, colony_size=10, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [bad_score, 3.0, 1.0, 2.0, 5.0])

    assert opt.best_score == expected_best
    assert opt.best_params == space.to_dicts(cands[best_index].x)[0]


def test_optimizer_should_report_best_in_user_direction_when_maximising():
    """Given direction 'max' and the score a - b + n, when 80 single candidates have been told, then
    best_score is the largest told score and best_params are the params of that candidate.

    Oracle: Optimizer docstring 'Scores are given in the user's direction' and the README ask-and-tell
    example, which reads opt.best_params and opt.best_score. Class: ask/tell best tracking under 'max',
    where scores are negated internally; test_abc.py only reads best values through optimize().
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0), "n": Int(1, 5)})
    opt = ABCOptimizer(space, direction="max", colony_size=6, seed=ANY_SEED)

    def score_of(cand):
        return cand.params["a"] - cand.params["b"] + cand.params["n"]

    batch = _finish_initial_batch(opt, score_of)
    asked = batch + _drive(opt, 80, score_of)
    scores = _scores(asked, score_of)
    best = asked[int(np.argmax(scores))]

    assert opt.best_score == max(scores)
    assert opt.best_params == space.to_dicts(best.x)[0]


def test_candidate_should_carry_key_and_params_decoded_from_its_x():
    """Given a space of all five kinds and limit 2, when 500 candidates are asked, then every key and params
    equal rows() of the candidate's x, and every x is a point of the unit cube of dimension d.

    Oracle: Candidate docstring '`x` in the unit cube, `params` as the user's function receives them, and
    `key`, the hashable decoded identity used for caching.' The run covers all three origins: the initial
    batch, one-coordinate neighbours and scouts, which limit = 2 makes frequent. Class: key identity, which
    test_abc.py never asserts.
    """
    space = Space(
        {
            "a": Float(-1.0, 1.0),
            "b": Float(1e-4, 1e-2, log=True),
            "c": Int(3, 9),
            "d": Int(1, 1000, log=True),
            "e": Categorical(["x", "y", "z"]),
        }
    )
    opt = ABCOptimizer(space, colony_size=6, limit=2, seed=ANY_SEED)
    asked = _finish_initial_batch(opt, _coordinate_sum) + _drive(opt, 500, _coordinate_sum)
    xs = _points(asked)

    keys, params = space.rows(xs)

    assert _keys(asked) == keys
    assert _params(asked) == params
    assert xs.shape == (len(asked), space.dim)
    assert xs.min() >= 0.0
    assert xs.max() <= 1.0


def test_candidate_ids_should_be_unique_over_a_run():
    """Given one optimizer, when the initial batch and 300 single candidates are asked, then no id repeats.

    Oracle: computed spec, telling 'an unknown/already-told candidate -> ValueError' identifies candidates by
    their id, so one optimizer must never hand out the same id twice. Class: id uniqueness, not covered by
    test_abc.py.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=2, seed=ANY_SEED)

    asked = _finish_initial_batch(opt, _coordinate_sum) + _drive(opt, 300, _coordinate_sum)
    ids = _ids(asked)

    assert len(set(ids)) == len(ids)


def test_optimizer_should_repeat_candidates_past_a_uniform_block_when_seed_is_equal():
    """Given two optimizers with the same SHARED_SEED and limit 5, when each asks 2000 single candidates,
    then the two candidate sequences are identical.

    Oracle: README 'The same `seed` reproduces the same search.' and the computed spec 'uniforms drawn in
    blocks of 4096'. Every neighbour uses at least three uniforms (j, k, phi), so 2000 asks need more than
    6000 draws and cross a block refill; limit = 5 interleaves scout samples from the generator with the
    blocks. Any seed works; both runs only need the same one. test_abc.py compares seeded runs only through
    optimize() with the default limit of 15.
    """
    first = ABCOptimizer(_unit_space(3), colony_size=10, limit=5, seed=SHARED_SEED)
    second = ABCOptimizer(_unit_space(3), colony_size=10, limit=5, seed=SHARED_SEED)

    first_asked = _finish_initial_batch(first, _coordinate_sum) + _drive(first, 2000, _coordinate_sum)
    second_asked = _finish_initial_batch(second, _coordinate_sum) + _drive(second, 2000, _coordinate_sum)

    np.testing.assert_array_equal(_points(first_asked), _points(second_asked))
