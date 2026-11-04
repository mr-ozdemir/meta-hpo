"""Unit tests of PSOOptimizer: swarm state, flight rules, bookkeeping, determinism and convergence.

Oracles: the module docstring of optimizers/pso.py (Shi & Eberhart 1998 update with the Clerc & Kennedy 2002
constants, star topology, synchronous iterations, zero initial velocities, velocity clamping in unit-cube
widths, absorbing walls, strict personal-best update, frozen guide, duplicate regeneration), the docstrings
of PSOOptimizer.tell, PSOOptimizer.cancel and PSOOptimizer.guide, the Optimizer docstring (the contract of
every optimizer) and the issue #18 theory write-up.

State is observed through the public attributes positions, velocities, pbest, pbest_f, phase, iteration and
guide, which the module documents; the pending table is never read.

Unspecified, therefore not asserted: exact candidate values (they depend on the random stream); the order of
the particles inside a batch beyond one candidate per particle; how many attempts a regeneration needed.

Input space: Every space, swarm configuration, seed and ask/tell sequence, which is unbounded.
Subset: One- to four-dimensional spaces with linear, log, Int and Categorical parameters; the defaults, the
boundary values of every constructor check, coefficient settings that isolate one term of the update
(social only, inertia only, everything zero), scripted constant, tied, improving and NaN scores, and seeded
runs on the sphere and Rastrigin functions.
"""

import math

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, PSOOptimizer, Space, optimize

ANY_SEED = 0


OTHER_SEED = 1


SWARM = 20


ITERATIONS = 10


def _box(d=2):
    """A fresh d-dimensional box [-5, 5]^d of linear Floats named x0, x1, ..."""
    return Space({f"x{j}": Float(-5.0, 5.0) for j in range(d)})


def _mixed():
    """A fresh mixed space with a linear Float, a log Float, an Int and a Categorical."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _sphere(params):
    """Sum of the squared numeric parameter values; 0 at the centre of the box."""
    return sum(v * v for v in params.values())


def _rastrigin(params):
    """Rastrigin function of the numeric parameter values; global minimum 0 at the origin."""
    values = list(params.values())
    return 10.0 * len(values) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in values)


def _mixed_score(params):
    """A smooth score on the mixed space with its minimum near lr = 1e-3, units = 128, gelu, dropout 0.2."""
    act = {"relu": 0.1, "tanh": 0.2, "gelu": 0.0}[params["act"]]
    lr = (math.log10(params["lr"]) + 3.0) ** 2
    return lr + ((params["units"] - 128) / 256) ** 2 + (params["dropout"] - 0.2) ** 2 + act


def _tell_sphere(opt):
    """Asks once and tells the batch its sphere scores; returns the batch."""
    cands = opt.ask()
    opt.tell(cands, [_sphere(c.params) for c in cands])
    return cands


def _run_iterations(opt, score, iterations):
    """Runs the initial batch and `iterations` flights with the given score; returns every batch's keys."""
    keys = []
    for _ in range(iterations + 1):
        cands = opt.ask()
        keys.append([c.key for c in cands])
        opt.tell(cands, [score(c.params) for c in cands])
    return keys


def _max_speed(opt, score, iterations):
    """The largest velocity magnitude seen after each tell over `iterations` flights."""
    speeds = []
    for _ in range(iterations + 1):
        cands = opt.ask()
        opt.tell(cands, [score(c.params) for c in cands])
        speeds.append(float(np.max(np.abs(opt.velocities))))
    return max(speeds)


def _grid():
    """A fresh 2-D Int space with 11 x 11 values."""
    return Space({"a": Int(0, 10), "b": Int(0, 10)})


def _distinct_keys(opt):
    """Distinct keys handed out by the initial batch and four flights on a quadratic bowl."""
    batches = _run_iterations(opt, lambda p: (p["a"] - 3) ** 2 + (p["b"] - 7) ** 2, 4)
    return len({key for batch in batches for key in batch})


def _found_mixed_optimum(seed):
    """True when 300 calls of PSOOptimizer with `seed` end below 0.05 with activation 'gelu'."""
    space = _mixed()
    study = optimize(_mixed_score, space, max_evals=300, optimizer=PSOOptimizer(space, seed=seed), seed=seed)
    return study.best_score < 0.05 and study.best_params["act"] == "gelu"


def test_pso_optimizer_should_store_the_documented_defaults():
    """Given a PSOOptimizer built with a space and a seed only, when its settings are read, then they are the
    documented defaults: 20 particles, w = 0.7298, c1 = c2 = 1.49618, max_velocity 1, max_tries 100,
    direction 'min'.

    Oracle: module docstring of optimizers/pso.py (Clerc & Kennedy constants, v_max equal to the range) and
    the issue #18 parameter table.
    """
    expected = (20, 0.7298, 1.49618, 1.49618, 1.0, 100, "min")

    opt = PSOOptimizer(_box(), seed=ANY_SEED)

    settings = (opt.swarm_size, opt.inertia, opt.cognitive, opt.social, opt.max_velocity, opt.max_tries)
    assert (*settings, opt.direction) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"swarm_size": 1}, id="swarm-1"),
        pytest.param({"swarm_size": 0}, id="swarm-0"),
        pytest.param({"swarm_size": 2.5}, id="swarm-fraction"),
        pytest.param({"swarm_size": math.inf}, id="swarm-inf"),
        pytest.param({"swarm_size": math.nan}, id="swarm-nan"),
        pytest.param({"inertia": -0.1}, id="inertia-negative"),
        pytest.param({"inertia": math.nan}, id="inertia-nan"),
        pytest.param({"cognitive": -1.0}, id="cognitive-negative"),
        pytest.param({"cognitive": math.inf}, id="cognitive-inf"),
        pytest.param({"social": math.nan}, id="social-nan"),
        pytest.param({"max_velocity": 0.0}, id="max-velocity-0"),
        pytest.param({"max_velocity": -0.5}, id="max-velocity-negative"),
        pytest.param({"max_velocity": math.inf}, id="max-velocity-inf"),
        pytest.param({"max_tries": 0}, id="max-tries-0"),
        pytest.param({"direction": "up"}, id="direction-unknown"),
    ],
)
def test_pso_optimizer_should_raise_value_error_when_a_setting_is_invalid(kwargs):
    """Given one invalid constructor argument, when PSOOptimizer is built, then ValueError is raised.

    Oracle: issue #18 parameter table (swarm_size an integer >= 2, so also not inf or NaN; coefficients
    finite and >= 0; max_velocity finite and > 0; max_tries >= 1) and the Optimizer direction check.
    """
    space = _box()

    with pytest.raises(ValueError):
        PSOOptimizer(space, seed=ANY_SEED, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"swarm_size": 2}, id="swarm-2"),
        pytest.param({"inertia": 0.0, "cognitive": 0.0, "social": 0.0}, id="coefficients-0"),
        pytest.param({"max_velocity": 1e-12}, id="max-velocity-tiny"),
        pytest.param({"max_tries": 1}, id="max-tries-1"),
    ],
)
def test_pso_optimizer_should_accept_the_smallest_valid_settings(kwargs):
    """Given the smallest valid value of a setting, when PSOOptimizer is built and asked once, then the first
    batch holds one candidate per particle.

    Oracle: issue #18 parameter table; each value is the boundary of its check.
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED, **kwargs)

    cands = opt.ask()

    assert len(cands) == opt.swarm_size


def test_pso_ask_should_hand_out_the_whole_swarm_as_distinct_initial_candidates():
    """Given a fresh PSOOptimizer, when ask() is called, then it returns swarm_size candidates with distinct
    keys, every x in the unit cube, and the phase is 'init'.

    Oracle: module docstring ('Only the initial population and each iteration's flight are handed out as
    batches') and the Optimizer docstring (a parameter set is not handed out twice).
    """
    opt = PSOOptimizer(_box(3), seed=ANY_SEED)

    cands = opt.ask()

    xs = np.array([c.x for c in cands])
    assert (len(cands), len({c.key for c in cands}), opt.phase) == (SWARM, SWARM, "init")
    assert bool(np.all((xs >= 0.0) & (xs <= 1.0))) is True


def test_pso_ask_should_raise_runtime_error_while_the_rest_of_a_partially_told_initial_batch_is_pending():
    """Given the initial batch told for its first 5 candidates only, when ask() is called, then RuntimeError
    is raised and the phase stays 'init'.

    Oracle: issue #18 FSM: ask hands out only particles that are neither told nor pending, and raises when
    every remaining particle of the phase is pending (the ABC convention for partial initial tells).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands[:5], [1.0] * 5)

    with pytest.raises(RuntimeError, match="pending"):
        opt.ask()
    assert opt.phase == "init"


def test_pso_ask_should_raise_runtime_error_when_the_whole_initial_batch_is_pending():
    """Given the initial batch handed out and not told, when ask() is called again, then RuntimeError is
    raised.

    Oracle: issue #18 FSM: ask raises only when every particle of the phase is pending.
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    opt.ask()

    with pytest.raises(RuntimeError, match="pending"):
        opt.ask()


def test_pso_ask_should_raise_runtime_error_when_the_whole_flight_is_pending():
    """Given the initial batch told and the first flight handed out, when ask() is called again, then
    RuntimeError is raised.

    Oracle: issue #18 FSM (fly -> fly only after a tell).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    opt.ask()

    with pytest.raises(RuntimeError, match="pending"):
        opt.ask()


def test_pso_tell_should_move_to_the_flight_phase_after_the_last_initial_candidate():
    """Given the initial batch, when it is told completely, then the phase is 'fly', the iteration is 0, the
    positions equal the initial x and the personal bests equal the positions.

    Oracle: module docstring (velocities start at zero; pbest starts at the initial position) and the
    issue #18 FSM ('init -> fly: the last initial particle is told').
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)

    cands = _tell_sphere(opt)

    xs = np.array([c.x for c in cands])
    assert (opt.phase, opt.iteration) == ("fly", 0)
    assert np.array_equal(opt.positions, xs) and np.array_equal(opt.pbest, xs)
    assert bool(np.all(opt.velocities == 0.0)) is True


def test_pso_flight_should_hand_out_one_candidate_per_particle_inside_the_cube():
    """Given the initial batch told, when the first flight is asked, then it holds swarm_size candidates with
    distinct keys and every x inside the unit cube.

    Oracle: module docstring ('every particle flies once per iteration'; absorbing walls keep the cube).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)

    cands = opt.ask()

    xs = np.array([c.x for c in cands])
    assert (len(cands), len({c.key for c in cands})) == (SWARM, SWARM)
    assert bool(np.all((xs >= 0.0) & (xs <= 1.0))) is True


def test_pso_flight_should_close_the_iteration_and_release_the_guide_after_the_last_tell():
    """Given one full flight told, when the state is read, then the iteration is 1 and the guide is None.

    Oracle: module docstring ('The guide g is frozen when the first candidate of an iteration is asked and
    stays until the last candidate of the iteration is told') and the PSOOptimizer.guide docstring.
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)

    _tell_sphere(opt)

    assert (opt.iteration, opt.guide) == (1, None)


def test_pso_guide_should_be_the_best_personal_best_when_the_flight_is_asked():
    """Given the initial batch told its sphere scores, when the first flight is asked, then the guide equals
    the personal best with the lowest score.

    Oracle: module docstring (star topology: the global best guides every particle).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    expected = opt.pbest[int(np.argmin(opt.pbest_f))]

    opt.ask()

    assert np.array_equal(opt.guide, expected)


def test_pso_guide_should_stay_frozen_while_part_of_the_flight_is_told_better_scores():
    """Given a flight asked, when its first half is told scores far below every earlier score, then the guide
    is unchanged while best_score already follows the new scores.

    Oracle: module docstring (frozen guide) and the Optimizer docstring (best_score follows every tell).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    cands = opt.ask()
    frozen = opt.guide

    opt.tell(cands[:10], [-100.0] * 10)

    assert np.array_equal(opt.guide, frozen)
    assert opt.best_score == -100.0


def test_pso_next_flight_should_not_depend_on_the_order_in_which_a_flight_is_told():
    """Given two PSOOptimizers with the same seed and the same scores, when one tells its flight in ask order
    and the other in reverse order, then the next flights carry the same keys.

    Oracle: module docstring ('the result does not depend on the order in which a batch is told').
    """
    first, second = PSOOptimizer(_box(), seed=ANY_SEED), PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(first)
    _tell_sphere(second)
    a, b = first.ask(), second.ask()
    first.tell(a, [_sphere(c.params) for c in a])
    second.tell(b[::-1], [_sphere(c.params) for c in b[::-1]])

    keys_first, keys_second = [c.key for c in first.ask()], [c.key for c in second.ask()]

    assert keys_first == keys_second


def test_pso_flight_should_move_every_coordinate_towards_the_guide_when_only_the_social_term_acts():
    """Given w = 0, c1 = 0 and c2 = 1, when the first flight is told, then every coordinate moved towards the
    guide and not past it.

    Oracle: velocity formula of the module docstring: with only c2 r2 (g - x) and r2 in [0, 1) the step has
    the sign of g - x and a smaller magnitude.
    """
    opt = PSOOptimizer(_box(3), inertia=0.0, cognitive=0.0, social=1.0, seed=ANY_SEED)
    _tell_sphere(opt)
    old = opt.positions.copy()
    cands = opt.ask()
    guide = opt.guide

    opt.tell(cands, [_sphere(c.params) for c in cands])

    step, towards = opt.positions - old, guide - old
    assert bool(np.all(step * towards >= 0.0)) is True
    assert bool(np.all(np.abs(step) <= np.abs(towards))) is True


def test_pso_flight_should_repeat_the_initial_positions_when_every_coefficient_is_zero():
    """Given w = c1 = c2 = 0 and max_tries = 1, when the first flight is asked, then every candidate repeats
    the key of its particle's initial candidate.

    Oracle: velocity formula (zero coefficients give zero velocity) and the Optimizer docstring (after
    max_tries attempts a repeat may come back).
    """
    opt = PSOOptimizer(_box(), inertia=0.0, cognitive=0.0, social=0.0, max_tries=1, seed=ANY_SEED)
    initial = _tell_sphere(opt)

    flight = opt.ask()

    assert [c.key for c in flight] == [c.key for c in initial]


def test_pso_velocity_should_stay_within_max_velocity_over_several_iterations():
    """Given max_velocity = 0.05, when ten flights are told, then no velocity component ever exceeds 0.05.

    Oracle: module docstring (velocity clamping in unit-cube widths).
    """
    opt = PSOOptimizer(_box(3), max_velocity=0.05, seed=ANY_SEED)

    speed = _max_speed(opt, _sphere, ITERATIONS)

    assert 0.0 < speed <= 0.05


def test_pso_walls_should_absorb_a_particle_that_overshoots_the_upper_bound():
    """Given a 1-D space whose score -x puts the guide at the largest initial x, and w = 0, c1 = 0, c2 = 3,
    when the first flight is told, then at least one particle sits exactly on the wall 1, every particle on
    the wall has velocity 0 and every position is inside [0, 1].

    Oracle: module docstring (absorbing walls: a coordinate that leaves the cube is placed on the wall and its
    velocity is set to 0). With c2 = 3 a particle overshoots whenever 3 r2 (g - x) > 1 - x.
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = PSOOptimizer(space, inertia=0.0, cognitive=0.0, social=3.0, seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [-c.params["x"] for c in init])

    flight = opt.ask()
    opt.tell(flight, [-c.params["x"] for c in flight])

    on_wall = opt.positions[:, 0] == 1.0
    assert int(np.sum(on_wall)) >= 1
    assert bool(np.all(opt.velocities[on_wall] == 0.0)) is True
    assert bool(np.all((opt.positions >= 0.0) & (opt.positions <= 1.0))) is True


def test_pso_pbest_should_ignore_a_tie_and_take_a_strictly_lower_score():
    """Given the initial batch told 1.0 everywhere, when the first flight is told 1.0 everywhere, then the
    personal bests stay at the initial positions; when the second flight is told 0.5 everywhere, then they
    move to the positions of that flight.

    Oracle: module docstring ('A personal best is replaced only by a strictly lower internal score').
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial, [1.0] * SWARM)
    tied = opt.ask()
    opt.tell(tied, [1.0] * SWARM)
    after_tie = opt.pbest.copy()

    better = opt.ask()
    opt.tell(better, [0.5] * SWARM)

    assert np.array_equal(after_tie, np.array([c.x for c in initial]))
    assert np.array_equal(opt.pbest, np.array([c.x for c in better]))
    assert opt.pbest_f == [0.5] * SWARM


@pytest.mark.parametrize(
    ("direction", "expected"),
    [pytest.param("min", -3.0, id="min"), pytest.param("max", 7.0, id="max")],
)
def test_pso_best_score_should_be_the_extreme_told_score_for_the_direction(direction, expected):
    """Given scores from -3 to 7 told to the initial batch, when best_score is read, then it is -3 for 'min'
    and 7 for 'max', with best_params of the candidate that scored it.

    Oracle: Optimizer docstring (scores are given in the user's direction) and best_params.
    """
    opt = PSOOptimizer(_box(), direction=direction, seed=ANY_SEED)
    cands = opt.ask()
    scores = [float(s) for s in np.linspace(-3.0, 7.0, SWARM)]

    opt.tell(cands, scores)

    best = cands[scores.index(expected)]
    assert (opt.best_score, opt.best_params) == (expected, best.params)


@pytest.mark.parametrize(
    ("direction", "worst"),
    [pytest.param("min", math.inf, id="min"), pytest.param("max", -math.inf, id="max")],
)
def test_pso_should_report_the_worst_score_and_the_first_params_when_every_initial_score_is_nan(
    direction, worst
):
    """Given every initial score NaN, when best_score and best_params are read, then best_score is the worst
    value for the direction and best_params are the params of the first candidate; a finite flight score then
    becomes the best.

    Oracle: Optimizer docstring ('a NaN or infinite score becomes the worst possible value for the
    direction').
    """
    opt = PSOOptimizer(_box(), direction=direction, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands, [math.nan] * SWARM)
    nan_state = (opt.best_score, opt.best_params)
    flight = opt.ask()

    opt.tell(flight, [2.0] * SWARM)

    assert nan_state == (worst, cands[0].params)
    assert opt.best_score == 2.0


def test_pso_should_hand_out_the_same_candidates_for_the_same_seed():
    """Given two PSOOptimizers with the same seed on separately built mixed spaces, when ten iterations run
    with the same scores, then every batch carries the same keys; a third with another seed differs.

    Oracle: README Behaviour ('The same seed reproduces the same search') and the Optimizer seed.
    """
    first = _run_iterations(PSOOptimizer(_mixed(), seed=ANY_SEED), _mixed_score, ITERATIONS)
    second = _run_iterations(PSOOptimizer(_mixed(), seed=ANY_SEED), _mixed_score, ITERATIONS)

    other = _run_iterations(PSOOptimizer(_mixed(), seed=OTHER_SEED), _mixed_score, ITERATIONS)

    assert first == second
    assert first != other


def test_pso_should_evaluate_each_value_of_a_three_value_space_once_and_stop():
    """Given Space({'k': Int(0, 2)}) and optimize(max_evals=50), when the search runs, then the function is
    called exactly 3 times, once per value.

    Oracle: Optimizer docstring (no parameter set twice while a new one exists; repeats are answered by the
    cache) and the optimize docstring (a space without Float stops once every set has been evaluated).
    """
    space = Space({"k": Int(0, 2)})
    calls = []

    study = optimize(
        lambda p: calls.append(p["k"]) or p["k"],
        space,
        max_evals=50,
        optimizer=PSOOptimizer(space, seed=ANY_SEED),
        seed=ANY_SEED,
    )

    assert (study.n_calls, sorted(calls)) == (3, [0, 1, 2])


def test_pso_flight_should_hand_out_more_distinct_keys_on_a_discrete_space_with_regeneration():
    """Given a 2-D Int space with 121 values and 20 particles, when the initial batch and four flights are
    told once with max_tries = 100 and once with max_tries = 1, then the run with regeneration hands out
    more distinct keys.

    Oracle: Optimizer docstring ('never hands out a parameter set it has handed out before, as long as it
    finds a new one in max_tries attempts') and the module docstring (a repeated flight is drawn again). A
    converged particle (pbest = g = x, velocity 0) cannot leave its cell, so repeats remain possible; only
    the effect of regeneration is asserted.
    """
    regenerated = _distinct_keys(PSOOptimizer(_grid(), seed=ANY_SEED))

    single = _distinct_keys(PSOOptimizer(_grid(), max_tries=1, seed=ANY_SEED))

    assert regenerated > single


def test_pso_cancel_should_let_the_next_ask_fly_exactly_the_cancelled_particles():
    """Given a flight asked, when its last 8 candidates are cancelled and ask() is called, then 8 candidates
    come back; telling all of them closes the iteration.

    Oracle: PSOOptimizer.cancel docstring ('the next ask() hands out new candidates for the same particles,
    with new random draws and the same frozen guide').
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    cands = opt.ask()
    guide = opt.guide
    opt.cancel(cands[-8:])
    opt.tell(cands[:-8], [_sphere(c.params) for c in cands[:-8]])

    again = opt.ask()
    same_guide = np.array_equal(opt.guide, guide)
    opt.tell(again, [_sphere(c.params) for c in again])

    assert (len(again), same_guide, opt.iteration) == (8, True, 1)


def test_pso_cancel_should_let_the_next_ask_hand_out_the_cancelled_initial_particles():
    """Given the initial batch, when 5 candidates are cancelled and the rest told, then ask() returns 5
    candidates and telling them ends the initial phase.

    Oracle: PSOOptimizer.cancel docstring and the optimize docstring (cut candidates are withdrawn).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()
    opt.cancel(cands[:5])
    opt.tell(cands[5:], [_sphere(c.params) for c in cands[5:]])

    again = opt.ask()
    opt.tell(again, [_sphere(c.params) for c in again])

    assert (len(again), opt.phase) == (5, "fly")


def test_pso_cancel_should_raise_value_error_for_a_candidate_that_is_not_pending():
    """Given a told initial batch, when one of its candidates is cancelled, then ValueError is raised.

    Oracle: PSOOptimizer.cancel docstring ('The batch is checked like in tell()').
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = _tell_sphere(opt)

    with pytest.raises(ValueError, match="not asked for"):
        opt.cancel(cands[:1])


def test_pso_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length():
    """Given the initial batch, when tell() gets one score fewer than candidates, then ValueError is raised.

    Oracle: PSOOptimizer.tell docstring (the whole batch is checked first).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell(cands, [1.0] * (SWARM - 1))


def test_pso_should_reach_the_sphere_minimum_within_forty_iterations():
    """Given the 2-D sphere on [-5, 5]^2, when optimize() runs 820 calls (the initial batch and 40 flights of
    20 particles), then the best score is below 1e-3.

    Oracle: CONTRIBUTING (a new optimizer needs a convergence test on a benchmark function); the sphere is
    unimodal, where PSO with the Clerc & Kennedy constants converges (Clerc & Kennedy 2002).
    """
    space = _box()

    study = optimize(
        _sphere, space, max_evals=820, optimizer=PSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    assert study.best_score < 1e-3


def test_pso_should_improve_on_its_initial_batch_on_rastrigin():
    """Given the 2-D Rastrigin function on [-5, 5]^2, when optimize() runs 820 calls, then the best score is
    below the best score of the 20 initial candidates.

    Oracle: CONTRIBUTING (convergence test on a benchmark function); Rastrigin is multimodal, so only progress
    over random sampling is asserted.
    """
    space = _box()

    study = optimize(
        _rastrigin, space, max_evals=820, optimizer=PSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    assert study.best_score < min(t.score for t in study.trials[:SWARM])


def test_pso_should_find_the_optimum_region_of_a_mixed_space_for_most_seeds():
    """Given the mixed space and a score with its minimum 0 near lr = 1e-3, units = 128, gelu and dropout 0.2,
    when optimize() runs 300 calls for each of 10 seeds, then at least 9 runs end below 0.05 with 'gelu'.

    Oracle: README Behaviour and the module docstring (log, Int and Categorical parameters move through the
    unit cube and are decoded by the Space). One run in ten may converge prematurely on another category,
    which is a known weakness of global-best PSO (issue #18 theory, section 9).
    """
    found = [_found_mixed_optimum(seed) for seed in range(10)]

    assert sum(found) >= 9


def test_pso_should_evaluate_the_same_points_when_maximising_minus_f_or_minimising_f():
    """Given the sphere and its negative, when optimize() runs 200 calls with 'min' on f and 'max' on -f and
    the same seed, then both studies evaluate the same keys.

    Oracle: Optimizer docstring ('internally everything is minimised').
    """
    space = _box()

    low = optimize(_sphere, space, max_evals=200, optimizer=PSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED)
    high = optimize(
        lambda p: -_sphere(p),
        space,
        direction="max",
        max_evals=200,
        optimizer=PSOOptimizer(space, direction="max", seed=ANY_SEED),
        seed=ANY_SEED,
    )

    assert [tuple(t.x) for t in low.trials] == [tuple(t.x) for t in high.trials]


def test_pso_should_evaluate_the_same_points_when_the_objective_is_scaled_by_1e_minus_20():
    """Given the sphere and the sphere times 1e-20, when optimize() runs 200 calls on each with the same seed,
    then both studies evaluate the same points.

    Oracle: Optimizer docstring (scores are only compared); PSO uses no fitness transform, so a positive
    scale changes no comparison.
    """
    space = _box(3)

    plain = optimize(
        _sphere, space, max_evals=200, optimizer=PSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )
    scaled = optimize(
        lambda p: 1e-20 * _sphere(p),
        space,
        max_evals=200,
        optimizer=PSOOptimizer(space, seed=ANY_SEED),
        seed=ANY_SEED,
    )

    assert [tuple(t.x) for t in plain.trials] == [tuple(t.x) for t in scaled.trials]


def test_pso_cancel_should_handle_candidates_that_share_a_key_in_one_batch():
    """Given a two-value space and four particles, so that the initial batch must repeat a key, when the
    first candidate of a repeated key is cancelled and then its two twins, then ask() hands out three
    candidates again and the batch can be told completely.

    Oracle: PSOOptimizer.cancel docstring (cancelled candidates count as never handed out; the batch is
    checked like in tell()) and the Optimizer docstring (a repeat may come back once no new set exists).
    Seed 0 gives the keys (1,), (0,), (0,), (0,), so candidates 1 to 3 share the key (0,).
    """
    space = Space({"k": Int(0, 1)})
    opt = PSOOptimizer(space, swarm_size=4, seed=ANY_SEED)
    cands = opt.ask()
    opt.cancel(cands[1:2])
    opt.cancel(cands[2:])
    opt.tell(cands[:1], [1.0])

    again = opt.ask()
    opt.tell(again, [0.0] * len(again))

    assert [c.key for c in cands] == [(1.0,), (0.0,), (0.0,), (0.0,)]
    assert (len(again), opt.phase) == (3, "fly")


def _replayed_flight(rng, opt):
    """The x and velocity of every particle's next flight, recomputed from the public state of opt and a twin
    generator: r1 = rng.random(d), then r2 = rng.random(d), particle by particle in index order."""
    d = opt.positions.shape[1]
    xs, vs = [], []
    for x, v, p in zip(opt.positions, opt.velocities, opt.pbest, strict=True):
        r1, r2 = rng.random(d), rng.random(d)
        raw = opt.inertia * v + opt.cognitive * r1 * (p - x) + opt.social * r2 * (opt.guide - x)
        step = np.clip(raw, -opt.max_velocity, opt.max_velocity)
        moved = x + step
        xs.append(np.clip(moved, 0.0, 1.0))
        vs.append(np.where((moved < 0.0) | (moved > 1.0), 0.0, step))
    return np.array(xs), np.array(vs)


def test_pso_flights_should_follow_the_velocity_update_with_r1_and_r2_drawn_per_particle():
    """Given a 3-D box, max_velocity = 0.2 and a twin numpy generator with the same seed, when the initial
    batch and two flights are told sphere scores, then every flight's x and the stored velocities equal
    w v + c1 r1 (pbest - x) + c2 r2 (g - x), clipped and absorbed, with r1 and r2 drawn in the documented
    order.

    Oracle: module docstring of optimizers/pso.py (velocity update, clipping, absorbing walls, draw order of
    r1 and r2, zero initial velocity). The second flight has non-zero velocities and personal bests that
    differ from the positions (the odd particles are told 100 more in the first flight, so they keep their
    initial personal best), so every term of the update is decisive. max_tries = 1 keeps one draw of r1
    and r2 per particle: the guide particle cannot move in the first flight, and regeneration would draw
    again for it.
    """
    opt = PSOOptimizer(_box(3), max_velocity=0.2, max_tries=1, seed=ANY_SEED)
    rng = np.random.default_rng(ANY_SEED)
    rng.random((SWARM, 3))
    _tell_sphere(opt)
    first = opt.ask()
    expected_x1, expected_v1 = _replayed_flight(rng, opt)
    opt.tell(first, [_sphere(c.params) + 100.0 * (n % 2) for n, c in enumerate(first)])
    second = opt.ask()
    expected_x2, expected_v2 = _replayed_flight(rng, opt)

    opt.tell(second, [_sphere(c.params) for c in second])

    assert np.allclose([c.x for c in first], expected_x1, rtol=0.0, atol=1e-12)
    assert np.allclose([c.x for c in second], expected_x2, rtol=0.0, atol=1e-12)
    assert np.allclose(opt.velocities, expected_v2, rtol=0.0, atol=1e-12)
    assert bool(np.any(expected_v1 != 0.0)) is True


def test_pso_guide_should_not_follow_a_personal_best_that_changes_during_the_iteration():
    """Given particle 3 made the global best in the first flight, when the second flight is asked and only
    the candidate of particle 3 is told a score below every other, then the guide keeps the value it had when
    the flight was asked although the personal best of particle 3 moved.

    Oracle: module docstring ('The guide g is frozen when the first candidate of an iteration is asked').
    Particle 3 has a non-zero velocity in the second flight, so its personal best really moves.
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    first = opt.ask()
    scores = [_sphere(c.params) for c in first]
    scores[3] = -1.0
    opt.tell(first, scores)
    second = opt.ask()
    frozen = opt.guide

    opt.tell(second[3:4], [-2.0])

    assert np.array_equal(opt.guide, frozen)
    assert not np.array_equal(opt.pbest[3], frozen)


def test_pso_flight_should_regenerate_repeats_inside_the_flights_of_a_discrete_space():
    """Given a 2-D Int space with 121 values and 20 particles, when the initial batch and six flights are
    told, then the flights alone repeat fewer keys (of earlier batches or of the same batch) with
    max_tries = 100 than with max_tries = 1.

    Oracle: module docstring ('A flight whose parameter set was already handed out is drawn again with new r1
    and r2, up to max_tries attempts'). Only flight batches are counted, so the regeneration of the initial
    batch does not carry the result.
    """
    regenerated = _flight_repeats(PSOOptimizer(_grid(), seed=ANY_SEED))

    single = _flight_repeats(PSOOptimizer(_grid(), max_tries=1, seed=ANY_SEED))

    assert regenerated < single


def _flight_repeats(opt):
    """Keys of the six flights after the initial batch that were handed out before, in earlier batches or
    earlier in the same flight."""
    batches = _run_iterations(opt, lambda p: (p["a"] - 3) ** 2 + (p["b"] - 7) ** 2, 6)
    seen = set(batches[0])
    repeats = 0
    for batch in batches[1:]:
        for key in batch:
            repeats += key in seen
            seen.add(key)
    return repeats


def test_pso_cancel_should_keep_a_key_reserved_when_its_twin_in_the_batch_was_already_told():
    """Given a flight on Int(0, 9) whose candidates 1 and 2 share the key (9,), when candidate 2 is told and
    candidate 1 is cancelled, then the re-asked candidate does not repeat any told key.

    Oracle: Optimizer docstring (a parameter set handed out before is not handed out again while a new one is
    found) and the PSOOptimizer.cancel docstring (a cancelled key is freed only when no candidate with the
    same key is pending or told). Regression: cancel() used to free (9,) after its twin was told.
    """
    space = Space({"k": Int(0, 9)})
    opt = PSOOptimizer(space, swarm_size=3, inertia=0.0, cognitive=0.0, social=2.0, seed=28)
    init = opt.ask()
    opt.tell(init, [-c.params["k"] for c in init])
    flight = opt.ask()
    opt.tell([flight[2], flight[0]], [-flight[2].params["k"], -flight[0].params["k"]])
    opt.cancel(flight[1:2])

    again = opt.ask()

    told = {c.key for c in init} | {flight[0].key, flight[2].key}
    assert [c.key for c in flight] == [(8.0,), (9.0,), (9.0,)]
    assert again[0].key not in told


def test_pso_cancel_should_let_a_cancelled_key_be_handed_out_again():
    """Given Int(0, 3) and two particles, when the whole initial batch with keys (2,) and (3,) is cancelled,
    then the next initial batch may hand out (3,) again.

    Oracle: PSOOptimizer.cancel docstring ('they count as never handed out'). Without freeing, (2,) and (3,)
    would stay reserved and the regeneration would push both new candidates to (0,) and (1,). Seed 1 is
    chosen because its second batch reuses (3,).
    """
    opt = PSOOptimizer(Space({"k": Int(0, 3)}), swarm_size=2, seed=1)
    first = opt.ask()
    opt.cancel(first)

    again = opt.ask()

    assert ([c.key for c in first], [c.key for c in again]) == ([(2.0,), (3.0,)], [(0.0,), (3.0,)])


def test_pso_init_should_set_the_personal_bests_to_the_initial_positions_even_for_nan_scores():
    """Given every initial score NaN, when the batch is told, then every personal best equals its initial x
    and every personal-best score is +inf.

    Oracle: issue #18 theory (p_i = x_i at initialisation) and the Optimizer docstring (NaN is the worst
    score, +inf internally).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [math.nan] * SWARM)

    assert np.array_equal(opt.pbest, np.array([c.x for c in cands]))
    assert opt.pbest_f == [math.inf] * SWARM


def test_pso_guide_should_be_the_lowest_index_particle_when_every_personal_best_ties():
    """Given every initial score equal, when the first flight is asked, then the guide is the personal best of
    particle 0.

    Oracle: issue #18 theory equation 6 (argmin of the personal-best scores, lowest index on ties).
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands, [3.0] * SWARM)

    opt.ask()

    assert np.array_equal(opt.guide, cands[0].x)


def test_pso_social_only_flight_should_move_some_coordinate():
    """Given w = 0, c1 = 0 and c2 = 1, when the first flight is told, then at least one coordinate of some
    particle changed.

    Oracle: velocity formula of the module docstring; together with the direction test this rules out a
    social term that is dropped, which would leave every particle in place.
    """
    opt = PSOOptimizer(_box(3), inertia=0.0, cognitive=0.0, social=1.0, seed=ANY_SEED)
    _tell_sphere(opt)
    old = opt.positions.copy()

    _tell_sphere(opt)

    assert bool(np.any(opt.positions != old)) is True


def test_pso_tell_should_draw_no_random_number():
    """Given a flight asked, when it is told, then the state of the optimizer generator is unchanged.

    Oracle: module docstring ('tell() ... draws no random number'); this keeps a failing tell and a twin in
    step and makes the next flight independent of the tell order.
    """
    opt = PSOOptimizer(_box(), seed=ANY_SEED)
    _tell_sphere(opt)
    cands = opt.ask()
    before = opt.rng.bit_generator.state

    opt.tell(cands, [_sphere(c.params) for c in cands])

    assert opt.rng.bit_generator.state == before


def test_pso_flight_should_not_repeat_a_key_inside_one_flight_while_a_new_one_is_reachable():
    """Given Int(0, 9), four particles, w = 0, c1 = 0, c2 = 1 and seed 1, when the first flight is asked after
    the initial keys (5,), (9,), (1,), (3,), then its four keys are distinct: (5,), (6,), (4,), (3,).

    Oracle: Optimizer docstring (a parameter set is not handed out twice while a new one is found) and the
    module docstring (a flight is drawn again). Without the check against keys issued earlier in the same
    flight, seed 1 hands out (4,) twice.
    """
    space = Space({"k": Int(0, 9)})
    opt = PSOOptimizer(space, swarm_size=4, inertia=0.0, cognitive=0.0, social=1.0, seed=1)
    init = opt.ask()
    opt.tell(init, [abs(c.params["k"] - 5) for c in init])

    flight = opt.ask()

    assert [c.key for c in init] == [(5.0,), (9.0,), (1.0,), (3.0,)]
    assert [c.key for c in flight] == [(5.0,), (6.0,), (4.0,), (3.0,)]


def test_pso_cancel_should_keep_a_key_reserved_while_its_twin_is_still_pending():
    """Given the seed-28 flight on Int(0, 9) whose candidates 1 and 2 share the key (9,), when candidate 1 is
    cancelled while candidate 2 is still pending and ask() is called, then the new candidate does not take
    (9,) again.

    Oracle: PSOOptimizer.cancel docstring (a cancelled key is freed only when no candidate with the same key
    is pending or told). Without handing the key over to the pending twin, the new candidate is (9,).
    """
    space = Space({"k": Int(0, 9)})
    opt = PSOOptimizer(space, swarm_size=3, inertia=0.0, cognitive=0.0, social=2.0, seed=28)
    init = opt.ask()
    opt.tell(init, [-c.params["k"] for c in init])
    flight = opt.ask()
    opt.cancel(flight[1:2])

    again = opt.ask()

    assert again[0].key != (9.0,)
