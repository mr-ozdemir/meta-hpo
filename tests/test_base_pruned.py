"""Unit tests of pruned candidates in tell(): the `pruned` keyword of ABCOptimizer, PSOOptimizer and
BBPSOOptimizer and the ranking the optimizer base gives them.

Oracles: the Optimizer docstring (one `pruned` entry per candidate, None for a candidate whose objective
returned, else the last step reported or -1; a pruned candidate ranks below every finite score and above a NaN
or infinite one, of two pruned candidates the later step and then the better last value ranks higher, and it
otherwise counts as a NaN score), Optimizer._ranked, and the module docstrings of abc (a pruned neighbour
never replaces a food source with a finite score), pso (a pruned flight moves the particle but replaces its
personal best only when that best is NaN, infinite or a lower-ranked pruned flight) and bbpso (the particle
keeps its position and score and moves on to the next operation). The optimizers are driven by hand through
ask and tell with chosen scores, so every expected state is read off the scores.

Unspecified, therefore not asserted: which candidates the optimizers hand out after a pruned one (that depends
on their random draws), and the error text beyond naming `pruned`.

Input space: Every batch of candidates with any scores and any `pruned` entries, for each optimizer and both
directions, which is unbounded. Subset: One pruned candidate against a food source or personal best that is
finite, NaN, or pruned at an earlier, the same or a later step; a better and a worse last value; both
directions; entries None, -1, 0 and 3; an entry list of the wrong length and invalid entries (-2, a float, a
bool, a string).
"""

import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, BBPSOOptimizer, Float, PSOOptimizer, Space

ANY_SEED = 0

NAN = math.nan


def _space():
    """A fresh two-parameter Float space."""
    return Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})


def _abc(direction, init_scores, init_pruned=None):
    """An ABCOptimizer with two food sources, initialised with `init_scores` (and `init_pruned`), that has
    handed out its first employed neighbour, for food source 0."""
    opt = ABCOptimizer(_space(), direction=direction, colony_size=4, seed=ANY_SEED)
    opt.tell(opt.ask(), init_scores, pruned=init_pruned)
    return opt, opt.ask()


@pytest.mark.parametrize(
    ("direction", "food", "last"),
    [
        pytest.param("min", 1.0, 0.5, id="min"),
        pytest.param("max", 1.0, 2.0, id="max"),
    ],
)
def test_abc_should_never_replace_a_finite_food_source_with_a_pruned_neighbour(direction, food, last):
    """Given an ABC food source 0 with the finite score `food` and its employed neighbour pruned at step 3
    with a last value that would be better, when the neighbour is told, then the source keeps its position and
    score and its trial counter rises to 1.

    Oracle: abc module docstring (a pruned neighbour never replaces a food source with a finite score, so it
    adds to the source's trial counter) and Optimizer (a pruned candidate ranks below every finite score).
    """
    opt, (neighbour,) = _abc(direction, [food, food])
    before = opt.foods[0].copy()

    opt.tell([neighbour], [last], pruned=[3])

    assert np.array_equal(opt.foods[0], before)
    assert opt.f[0] == opt._internal(food)
    assert opt.trial[0] == 1


@pytest.mark.parametrize(
    ("init_pruned", "init_score", "step", "last", "replaced"),
    [
        pytest.param(None, NAN, 0, 5.0, True, id="nan-source-loses-to-pruned"),
        pytest.param([1, None], 0.5, 2, 0.9, True, id="later-step-wins"),
        pytest.param([2, None], 0.5, 2, 0.4, True, id="same-step-better-value-wins"),
        pytest.param([2, None], 0.5, 2, 0.6, False, id="same-step-worse-value-loses"),
        pytest.param([3, None], 0.5, 2, 0.1, False, id="earlier-step-loses"),
        pytest.param([0, None], 0.5, -1, 0.1, False, id="no-report-loses-to-step-0"),
    ],
)
def test_abc_should_rank_a_pruned_neighbour_against_a_nan_or_pruned_food_source(
    init_pruned, init_score, step, last, replaced
):
    """Given an ABC food source 0 that a NaN score or a pruned trial gave (minimising), when its employed
    neighbour pruned at `step` with last value `last` is told, then it replaces the source exactly when it
    ranks higher: always against NaN, by the later step, and at the same step by the better last value.

    Oracle: Optimizer docstring (a pruned candidate ranks above a NaN score; of two pruned candidates the one
    that reported a later step ranks higher, and at the same step the better last value; -1 for no report) and
    the abc module docstring (it replaces a source whose score is NaN or that a lower-ranked pruned trial
    gave).
    """
    opt, (neighbour,) = _abc("min", [init_score, 1.0], init_pruned)
    position = neighbour.x.copy()

    opt.tell([neighbour], [last], pruned=[step])

    assert np.array_equal(opt.foods[0], position) is replaced
    assert opt.trial[0] == int(not replaced)


def test_abc_should_count_a_pruned_food_source_as_a_nan_score_for_the_onlooker_probabilities():
    """Given two ABC food sources, one pruned and one with the score 1.0, when the employed phase ends with
    two neighbours that do not replace them, then the pruned source has fitness 0 and the onlooker probability
    0.1, as a NaN score would.

    Oracle: abc module docstring (a pruned food source counts as a NaN score for the fitness and the onlooker
    probabilities) and the onlooker rule p = 0.9 * fit / max + 0.1.
    """
    opt, (first,) = _abc("min", [0.5, 1.0], [2, None])
    opt.tell([first], [0.9], pruned=[1])
    (second,) = opt.ask()

    opt.tell([second], [2.0])

    assert opt.fit[0] == 0.0
    assert opt.prob == pytest.approx([0.1, 1.0])


def _pso(direction, scores):
    """A PSOOptimizer of two particles initialised with `scores` that has handed out its first flight."""
    opt = PSOOptimizer(_space(), direction=direction, swarm_size=2, seed=ANY_SEED)
    opt.tell(opt.ask(), scores)
    return opt, opt.ask()


@pytest.mark.parametrize(
    ("direction", "scores", "flown"),
    [
        pytest.param("min", [1.0, 2.0], [0.1, 0.2], id="min"),
        pytest.param("max", [1.0, 2.0], [5.0, 6.0], id="max"),
    ],
)
def test_pso_should_move_a_pruned_particle_but_keep_its_finite_personal_best(direction, scores, flown):
    """Given two PSO particles with finite personal bests, when both flights are told as pruned at step 4 with
    last values that would be better, then both particles move to their flights while their personal bests,
    with their scores, and the next guide stay those of the initial population.

    Oracle: pso module docstring (a pruned flight moves the particle like any other but replaces its personal
    best only when that best is NaN, infinite or a lower-ranked pruned flight; the guide is the personal best
    of the highest rank).
    """
    opt, flights = _pso(direction, scores)
    pbest, pbest_f = opt.pbest.copy(), list(opt.pbest_f)
    best_init = int(np.argmin([opt._internal(s) for s in scores]))

    opt.tell(flights, flown, pruned=[4, 4])
    opt.ask()

    assert np.array_equal(opt.positions, np.array([c.x for c in flights]))
    assert np.array_equal(opt.pbest, pbest)
    assert opt.pbest_f == pbest_f
    assert np.array_equal(opt.guide, pbest[best_init])


def test_pso_should_replace_a_nan_personal_best_with_a_pruned_flight():
    """Given a PSO particle whose initial score was NaN, when its flight is told as pruned at step 0, then its
    personal best moves to the flight, while the other particle, pruned against a finite best, keeps its own.

    Oracle: pso module docstring (a pruned flight replaces a personal best that has a NaN score) and Optimizer
    (a pruned candidate ranks above a NaN score).
    """
    opt, flights = _pso("min", [NAN, 1.0])
    kept = opt.pbest[1].copy()

    opt.tell(flights, [0.5, 0.5], pruned=[0, 0])

    assert np.array_equal(opt.pbest[0], flights[0].x)
    assert np.array_equal(opt.pbest[1], kept)


def test_bbpso_should_keep_a_particle_in_place_when_its_move_is_pruned_and_try_the_next_operation():
    """Given a BBPSO swarm after its initial population, when the first move is told as pruned with a last
    value that would be the best of the run, then the particle keeps its position and score, its operation
    index moves from 1 to 2, and the best point of the run stays an initial one.

    Oracle: bbpso module docstring (a pruned move: the particle keeps its position and score and moves on to
    the next operation) and Optimizer (a pruned candidate otherwise counts as a NaN score, so it is never the
    best).
    """
    opt = BBPSOOptimizer(_space(), swarm_size=6, seed=ANY_SEED)
    opt.tell(opt.ask(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    (move,) = opt.ask()
    particle = move.meta["individual"]
    position, score, best = opt.positions[particle].copy(), opt.f[particle], opt.best_f

    opt.tell([move], [-100.0], pruned=[2])

    assert np.array_equal(opt.positions[particle], position)
    assert opt.f[particle] == score
    assert opt.flags[particle] == 2
    assert opt.best_f == best


@pytest.mark.parametrize(
    "pruned",
    [
        pytest.param([None], id="too-short"),
        pytest.param([None, None, None], id="too-long"),
        pytest.param([-2, None], id="below-minus-one"),
        pytest.param([1.0, None], id="float"),
        pytest.param([True, None], id="bool"),
        pytest.param(["1", None], id="string"),
    ],
)
def test_tell_should_reject_invalid_pruned_entries_and_leave_the_optimizer_unchanged(pruned):
    """Given an ABC optimizer with its two initial candidates pending, when they are told with a `pruned` list
    of the wrong length or with an entry that is not None or an integer of at least -1, then tell() raises
    ValueError naming pruned or the step, and both candidates are still pending.

    Oracle: Optimizer docstring (the whole batch is checked first, so a failing batch raises and leaves the
    optimizer unchanged) and Optimizer._ranked (one entry per candidate, each None or an integer >= -1).
    """
    opt = ABCOptimizer(_space(), colony_size=4, seed=ANY_SEED)
    batch = opt.ask()

    with pytest.raises(ValueError, match="pruned"):
        opt.tell(batch, [1.0, 2.0], pruned=pruned)
    assert sorted(opt._pending) == [c.id for c in batch]


@pytest.mark.parametrize(
    ("scores", "pruned", "ranks", "values"),
    [
        pytest.param([1.0, 2.0], None, [(0.0, 1.0), (0.0, 2.0)], [1.0, 2.0], id="finite"),
        pytest.param([NAN, math.inf], None, [(2.0, 0.0), (2.0, 0.0)], [math.inf, math.inf], id="failed"),
        pytest.param(
            [0.5, NAN], [3, -1], [(1.0, -3.0, 0.5), (1.0, 1.0, math.inf)], [math.inf, math.inf], id="pruned"
        ),
    ],
)
def test_ranked_should_give_finite_pruned_and_failed_scores_their_documented_ranks(
    scores, pruned, ranks, values
):
    """Given scores with and without `pruned` entries when minimising, when the optimizer base ranks them,
    then a finite score ranks (0, value), a NaN or infinite one (2, 0) and a pruned one (1, -last step, last
    value), and the value of every pruned or failed score is +inf.

    Oracle: Optimizer._ranked docstring.
    """
    opt = ABCOptimizer(_space(), seed=ANY_SEED)

    ranked = opt._ranked(scores, pruned)

    assert [rank for _, rank in ranked] == ranks
    assert [value for value, _ in ranked] == values
