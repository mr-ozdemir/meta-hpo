"""Unit tests of BBPSOOptimizer: swarm state, operations, leaders, refinement, bookkeeping and convergence.

Oracles: the module docstring of optimizers/bbpso.py (Zhang, Lim & Liu 2023 full model with the stated
deviations: per-particle operation index, unconditional replacement, refinement of the leader and the four
next-best particles with Q-learning over IQI, secant, Steffensen and one ITP step, root methods on
h = s (f - target), Steffensen increment h / h_max, no acceptance test, repair after every operation), the
docstrings of BBPSOOptimizer.tell and BBPSOOptimizer.cancel, the Optimizer docstring and the BBPSO theory
write-up.

State is observed through the public attributes positions, f (internal minimised scores), flags, q,
q_state, phase, iteration, target, best_x and best_f. A test may set entries of q before the refinement to
choose the action, because q is the documented Q-table and the action is its row maximum.

Unspecified, therefore not asserted: exact move positions (random draws), which particles are refined
beyond the leader, and how many attempts a regeneration needed.

Input space: Every space, configuration, seed and ask/tell sequence, which is unbounded.
Subset: One- to four-dimensional spaces of every parameter kind; defaults and the boundary values of every
constructor check; scripted constant, improving, worse and NaN scores that force each branch of the move and
refinement rules; seeded runs on the sphere and Rastrigin functions.
"""

import math

import numpy as np
import pytest

from metahpo import BBPSOOptimizer, Categorical, Float, Int, Space, operators, optimize

ANY_SEED = 0


OTHER_SEED = 1


SWARM = 20


IQI, SECANT, STEFFENSEN, ITP = 0, 1, 2, 3


def _box(d=2):
    """A fresh d-dimensional box [-5, 5]^d of linear Floats."""
    return Space({f"x{j}": Float(-5.0, 5.0) for j in range(d)})


def _unit():
    """A fresh one-parameter space on [0, 1]."""
    return Space({"x": Float(0.0, 1.0)})


def _mixed():
    """A fresh mixed space with a log Float, an Int, a Categorical and a linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _sphere(params):
    """Sum of the squared parameter values; 0 at the centre."""
    return sum(v * v for v in params.values())


def _rastrigin(params):
    """Rastrigin function of the parameter values; global minimum 0 at the origin."""
    values = list(params.values())
    return 10.0 * len(values) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in values)


def _mixed_score(params):
    """A smooth score with its minimum near lr = 1e-3, units = 128, gelu and dropout 0.2."""
    act = {"relu": 0.1, "tanh": 0.2, "gelu": 0.0}[params["act"]]
    return (math.log10(params["lr"]) + 3.0) ** 2 + ((params["units"] - 128) / 256) ** 2 + act


def _told(opt, score):
    """Asks once and tells every candidate its score; returns the candidates."""
    cands = opt.ask()
    opt.tell(cands, [score(c.params) for c in cands])
    return cands


def _to_refinement(opt, score):
    """Tells the initial batch and the swarm_size moves of the first iteration; returns the moves."""
    _told(opt, score)
    return [_told(opt, score)[0] for _ in range(opt.swarm_size)]


def _refinement_asks(opt, score):
    """Counts the asks of the refinement block until the first move of the next iteration is handed out."""
    count = 0
    cands = opt.ask()
    while opt.phase == "enhance":
        count += 1
        opt.tell(cands, [score(c.params) for c in cands])
        cands = opt.ask()
    opt.tell(cands, [score(c.params) for c in cands])
    return count


def _tell_moves(opt, scores):
    """Asks one move per score and tells it that score."""
    for score in scores:
        opt.tell(opt.ask(), [score])


def _told_many(opt, score, asks):
    """Asks and tells `asks` times with the given score."""
    for _ in range(asks):
        _told(opt, score)


def _keys(opt, score, asks):
    """Keys handed out by `asks` asks with the given score."""
    return [c.key for _ in range(asks) for c in _told(opt, score)]


def _found_mixed_optimum(seed):
    """True when 300 calls with `seed` end below 0.05 with activation 'gelu'."""
    space = _mixed()
    study = optimize(
        _mixed_score, space, max_evals=300, optimizer=BBPSOOptimizer(space, seed=seed), seed=seed
    )
    return study.best_score < 0.05 and study.best_params["act"] == "gelu"


def _two_finite_particles(direction):
    """A 1-D BBPSOOptimizer whose initial batch scores 0.90 (particle 0) and 0.88 (particle 1), every other
    particle NaN, and whose first moves score 0.5 (particle 0), 0.88 (particle 1) and NaN elsewhere, so the
    leader base is particle 0's initial point and particle 1 is the only possible partner."""
    opt = BBPSOOptimizer(_unit(), direction=direction, seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [0.90, 0.88] + [math.nan] * (SWARM - 2))
    moves = []
    for i in range(SWARM):
        cand = opt.ask()[0]
        opt.tell([cand], [0.5 if i == 0 else 0.88 if i == 1 else math.nan])
        moves.append(cand)
    return opt, init, moves


def test_bbpso_optimizer_should_store_the_documented_defaults():
    """Given a BBPSOOptimizer built with a space and a seed, when its settings are read, then they are 20
    particles, target 0.0 for 'min', crossover 0.5, q_alpha 0.1, q_gamma 0.9 and max_tries 100.

    Oracle: bbpso module docstring and the BBPSO theory parameter table.
    """
    expected = (20, 0.0, 0.5, 0.1, 0.9, 100)

    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)

    assert (opt.swarm_size, opt.target, opt.crossover, opt.q_alpha, opt.q_gamma, opt.max_tries) == expected


def test_bbpso_optimizer_should_default_the_target_to_one_when_maximising():
    """Given direction 'max' and no target, when the target is read, then it is 1.0; an explicit target is
    kept.

    Oracle: bbpso module docstring ('by default 0.0 for min (losses) and 1.0 for max (accuracy-like
    metrics)').
    """
    maximising = BBPSOOptimizer(_box(), direction="max", seed=ANY_SEED)

    explicit = BBPSOOptimizer(_box(), direction="max", target=0.0, seed=ANY_SEED)

    assert (maximising.target, explicit.target) == (1.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"swarm_size": 5}, id="swarm-5"),
        pytest.param({"swarm_size": 6.5}, id="swarm-fraction"),
        pytest.param({"swarm_size": math.inf}, id="swarm-inf"),
        pytest.param({"crossover": -0.1}, id="crossover-negative"),
        pytest.param({"crossover": 1.5}, id="crossover-above-1"),
        pytest.param({"crossover": math.nan}, id="crossover-nan"),
        pytest.param({"q_alpha": 0.0}, id="q-alpha-0"),
        pytest.param({"q_alpha": 1.5}, id="q-alpha-above-1"),
        pytest.param({"q_gamma": 1.0}, id="q-gamma-1"),
        pytest.param({"q_gamma": -0.1}, id="q-gamma-negative"),
        pytest.param({"target": math.nan}, id="target-nan"),
        pytest.param({"target": math.inf}, id="target-inf"),
        pytest.param({"max_tries": 0}, id="max-tries-0"),
        pytest.param({"direction": "up"}, id="direction-unknown"),
    ],
)
def test_bbpso_optimizer_should_raise_value_error_when_a_setting_is_invalid(kwargs):
    """Given one invalid constructor argument, when BBPSOOptimizer is built, then ValueError is raised.

    Oracle: BBPSO theory parameter table (swarm_size an integer >= 6; crossover in [0, 1]; q_alpha in
    (0, 1]; q_gamma in [0, 1); target finite; max_tries >= 1) and the Optimizer direction check.
    """
    space = _box()

    with pytest.raises(ValueError):
        BBPSOOptimizer(space, seed=ANY_SEED, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"swarm_size": 6}, id="swarm-6"),
        pytest.param({"crossover": 0.0}, id="crossover-0"),
        pytest.param({"crossover": 1.0}, id="crossover-1"),
        pytest.param({"q_alpha": 1.0}, id="q-alpha-1"),
        pytest.param({"q_gamma": 0.0}, id="q-gamma-0"),
        pytest.param({"max_tries": 1}, id="max-tries-1"),
    ],
)
def test_bbpso_optimizer_should_run_a_whole_iteration_with_the_smallest_valid_settings(kwargs):
    """Given the boundary value of one setting, when the initial batch, the moves and the refinement run on
    the sphere, then the second iteration starts.

    Oracle: BBPSO theory parameter table; each value is the boundary of its check.
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED, **kwargs)
    _to_refinement(opt, _sphere)

    _refinement_asks(opt, _sphere)

    assert (opt.iteration, opt.phase) == (1, "move")


def test_bbpso_ask_should_hand_out_the_whole_swarm_as_distinct_initial_candidates():
    """Given a fresh BBPSOOptimizer, when ask() is called, then 20 candidates with distinct keys come back
    and the phase is 'init'.

    Oracle: module docstring ('Only the initial population is handed out as a batch').
    """
    opt = BBPSOOptimizer(_box(3), seed=ANY_SEED)

    cands = opt.ask()

    assert (len(cands), len({c.key for c in cands}), opt.phase) == (SWARM, SWARM, "init")


def test_bbpso_ask_should_raise_runtime_error_while_initial_candidates_are_pending():
    """Given the initial batch told for 5 candidates only, when ask() is called, then RuntimeError is raised
    and the phase stays 'init'.

    Oracle: the ask/tell convention of the library (the remaining particles are pending).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands[:5], [1.0] * 5)

    with pytest.raises(RuntimeError, match="pending"):
        opt.ask()
    assert opt.phase == "init"


def test_bbpso_ask_should_raise_runtime_error_while_a_move_is_pending():
    """Given a move handed out and not told, when ask() is called, then RuntimeError is raised.

    Oracle: module docstring ('every move ... depends on the state the previous tell produced').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, _sphere)
    opt.ask()

    with pytest.raises(RuntimeError, match="pending"):
        opt.ask()


def test_bbpso_moves_should_come_one_at_a_time_and_start_the_refinement_after_the_last_particle():
    """Given the initial batch told, when 20 moves are asked and told, then each ask hands out one candidate
    in phase 'move' and the 20th tell switches the phase to 'enhance'.

    Oracle: module docstring (one move per particle per iteration, refinement after every iteration).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, _sphere)

    moves = [len(_told(opt, _sphere)) for _ in range(SWARM)]

    assert (moves, opt.phase) == ([1] * SWARM, "enhance")


def test_bbpso_move_should_replace_the_position_even_when_the_score_is_worse():
    """Given the initial batch told 1.0 everywhere, when particle 0's move is told 9.0, then particle 0's
    position and score are those of the move.

    Oracle: module docstring ('A move always replaces the particle's position (there is no acceptance test in
    bare-bones PSO)').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [1.0] * SWARM)
    move = opt.ask()

    opt.tell(move, [9.0])

    assert np.array_equal(opt.positions[0], move[0].x)
    assert opt.f[0] == 9.0


def test_bbpso_flags_should_advance_only_for_particles_whose_move_was_not_strictly_better():
    """Given the initial batch told 1.0, when particle 0's move is told 0.5 and every other move 1.0 (a tie)
    or 2.0, then particle 0 keeps operation 1 and every other particle moves on to operation 2.

    Oracle: module docstring ('moves on to the next operation ... whenever its move does not give a
    strictly better score'); the index is per particle.
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [1.0] * SWARM)

    _tell_moves(opt, [0.5, 1.0] + [2.0] * (SWARM - 2))

    assert opt.flags == [1] + [2] * (SWARM - 1)


def test_bbpso_flags_should_cycle_through_the_five_operations_and_wrap_to_the_first():
    """Given a particle whose every move is worse, when six iterations run, then its operation index goes
    2, 3, 4, 5, 1, 2.

    Oracle: module docstring (operation 5 wraps to 1).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, lambda p: 1.0)

    seen = [_flag_after_iteration(opt) for _ in range(6)]

    assert seen == [2, 3, 4, 5, 1, 2]


def _flag_after_iteration(opt):
    """Runs one iteration in which every move and refinement trial scores worse than before (increasing
    scores), and returns particle 5's operation index."""
    for _ in range(opt.swarm_size):
        opt.tell(opt.ask(), [1.0 + opt.iteration + 1.0])
    _refinement_asks(opt, lambda p: 1e9)
    return opt.flags[5]


def test_bbpso_refinement_should_take_between_one_and_ten_evaluations_per_iteration():
    """Given the first iteration on the sphere, when the refinement runs, then it asks between 1 and 10
    candidates (at most five slots, at most two each) and the next iteration starts.

    Oracle: module docstring (leader and the four next-best particles; Steffensen uses a probe and a trial).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)

    count = _refinement_asks(opt, _sphere)

    assert 1 <= count <= 10
    assert opt.iteration == 1


def test_bbpso_refinement_should_reward_minus_one_when_every_trial_is_worse():
    """Given the first iteration on the sphere, when every refinement trial is told 1e9, then every refined
    particle has exactly one non-zero Q entry, -0.1 in the IQI row, and its state is that action.

    Oracle: module docstring (reward -1; Q[s, a] += q_alpha (reward + q_gamma max Q[a, :] - Q[s, a]) with all
    zeros gives -0.1; the state becomes the chosen action; the initial state is IQI).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)

    _refinement_asks(opt, lambda p: 1e9)

    refined = [p for p in range(SWARM) if np.any(opt.q[p] != 0.0)]
    entries = [(np.count_nonzero(opt.q[p]), float(opt.q[p, IQI, opt.q_state[p]])) for p in refined]
    assert 1 <= len(refined) <= 5
    assert entries == [(1, pytest.approx(-0.1))] * len(refined)


def test_bbpso_refinement_should_reward_plus_one_and_replace_the_particle_when_a_trial_is_better():
    """Given the first iteration on the sphere, when every refinement trial is told -1e9, then every refined
    particle's Q entry is +0.1 and the leader's slot ends with the trial's score -1e9.

    Oracle: module docstring (reward +1; the refined particle takes the better point).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)

    _refinement_asks(opt, lambda p: -1e9)

    refined = [p for p in range(SWARM) if np.any(opt.q[p] != 0.0)]
    assert [float(opt.q[p, IQI, opt.q_state[p]]) for p in refined] == [pytest.approx(0.1)] * len(refined)
    assert -1e9 in opt.f
    assert opt.best_score == -1e9


def test_bbpso_steffensen_should_ask_a_normalised_probe_and_then_the_steffensen_trial():
    """Given Steffensen made the best action for every particle, when the refinement starts, then the first
    candidate is the probe repair(G + h / h_max) and, after the probe is told, the second is
    repair(G - hn^2 / (hp - hn)).

    Oracle: module docstring (Steffensen's increment is h / h_max with h_max the largest |h| of the swarm and
    the base; root methods work on h = f - target, target 0 for 'min').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)
    opt.q[:, IQI, STEFFENSEN] = 1.0
    base, h_a = opt.best_x.copy(), opt.best_f
    h_max = max([abs(h_a)] + [abs(v) for v in opt.f if math.isfinite(v)])
    h_norm = h_a / h_max

    probe = opt.ask()
    opt.tell(probe, [_sphere(probe[0].params)])
    trial = opt.ask()

    h_probe = _sphere(probe[0].params) / h_max
    expected_trial = np.clip(base - h_norm**2 / (h_probe - h_norm), 0.0, 1.0)
    assert np.allclose(probe[0].x, np.clip(base + h_norm, 0.0, 1.0), rtol=0.0, atol=1e-12)
    assert np.allclose(trial[0].x, expected_trial, rtol=0.0, atol=1e-12)


def test_bbpso_secant_should_step_beyond_the_leader_away_from_a_worse_partner_when_maximising():
    """Given direction 'max', target 1, the leader at accuracy 0.90 and the only partner at 0.88, when the
    secant method refines the leader, then the trial is G + 5 (G - b), clipped to [0, 1].

    Oracle: module docstring (h = s (f - target): h = 0.10 at the leader and 0.12 at the partner; the secant
    root is 5 segments beyond the leader on the side away from the partner). With the raw accuracy the step
    would be 45 segments towards and beyond the partner.
    """
    opt, init, moves = _two_finite_particles("max")
    opt.q[:, IQI, SECANT] = 1.0
    leader, partner = init[0].x, moves[1].x

    trial = opt.ask()

    assert np.allclose(trial[0].x, np.clip(leader + 5.0 * (leader - partner), 0.0, 1.0), rtol=0.0, atol=1e-9)


def test_bbpso_itp_should_take_one_step_just_beyond_the_leader_without_a_sign_change():
    """Given the same leader and partner as the secant test, when one ITP step refines the leader, then the
    trial is G + t (b - G) with t = -0.024.

    Oracle: module docstring ('A single ITP step is taken without a sign condition') and itp_point for
    h = 0.10 and 0.12.
    """
    opt, init, moves = _two_finite_particles("max")
    opt.q[:, IQI, ITP] = 1.0
    leader, partner = init[0].x, moves[1].x

    trial = opt.ask()

    assert np.allclose(
        trial[0].x, np.clip(leader - 0.024 * (partner - leader), 0.0, 1.0), rtol=0.0, atol=1e-9
    )


def test_bbpso_iqi_should_skip_the_slot_when_fewer_than_two_partners_exist():
    """Given only one finite partner, when IQI is the best action, then the leader's slot hands out nothing
    and gets reward -1 at once, and the next ask belongs to another slot or to the next iteration.

    Oracle: module docstring (IQI uses two random partner particles) and the BBPSO theory (a slot without a
    candidate is finished inside ask with reward -1). The Q entry starts at 1.0 and max Q[IQI, :] is that
    same entry, so the update gives 1 + 0.1 (-1 + 0.9 * 1 - 1) = 0.89.
    """
    opt, _, _ = _two_finite_particles("max")
    opt.q[:, IQI, IQI] = 1.0

    opt.ask()

    assert float(opt.q[0, IQI, IQI]) == pytest.approx(1.0 + 0.1 * (-1.0 + 0.9 * 1.0 - 1.0))


def test_bbpso_should_skip_the_refinement_when_every_score_is_nan():
    """Given every initial score and every move NaN, when the next candidate is asked, then no refinement
    candidate is handed out: the second iteration's first move comes back.

    Oracle: module docstring (refinement bases need a finite score) and the Optimizer docstring (NaN is the
    worst score).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, lambda p: math.nan)
    _told_many(opt, lambda p: math.nan, SWARM)

    opt.ask()

    assert (opt.phase, opt.iteration) == ("move", 1)


def test_bbpso_best_score_should_follow_the_direction():
    """Given scores from -3 to 7 told to the initial batch, when best_score is read, then it is -3 for 'min'
    and 7 for 'max'.

    Oracle: Optimizer docstring (scores are given in the user's direction).
    """
    low, high = BBPSOOptimizer(_box(), seed=ANY_SEED), BBPSOOptimizer(_box(), direction="max", seed=ANY_SEED)
    scores = [float(s) for s in np.linspace(-3.0, 7.0, SWARM)]

    low.tell(low.ask(), scores)
    high.tell(high.ask(), scores)

    assert (low.best_score, high.best_score) == (-3.0, 7.0)


def test_bbpso_should_hand_out_the_same_candidates_for_the_same_seed():
    """Given two BBPSOOptimizers with the same seed and a third with another, when 80 asks run with the same
    scores, then the first two hand out the same keys and the third differs.

    Oracle: README Behaviour ('The same seed reproduces the same search').
    """
    first = _keys(BBPSOOptimizer(_mixed(), seed=ANY_SEED), _mixed_score, 80)
    second = _keys(BBPSOOptimizer(_mixed(), seed=ANY_SEED), _mixed_score, 80)

    other = _keys(BBPSOOptimizer(_mixed(), seed=OTHER_SEED), _mixed_score, 80)

    assert first == second
    assert first != other


def test_bbpso_should_evaluate_each_value_of_a_four_value_space_once_and_stop():
    """Given Space({'k': Int(0, 3)}) and optimize(max_evals=40), when the search runs, then the function is
    called exactly 4 times.

    Oracle: Optimizer docstring (no parameter set twice while a new one exists) and the optimize docstring (a
    space without Float stops once every set has been evaluated).
    """
    space = Space({"k": Int(0, 3)})

    study = optimize(
        lambda p: p["k"], space, max_evals=40, optimizer=BBPSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    assert study.n_calls == 4


def test_bbpso_cancel_should_let_the_next_ask_repeat_the_move_of_the_same_particle():
    """Given a move handed out, when it is cancelled and 20 moves are then asked and told, then the 20th tell
    starts the refinement, so the cancelled move was asked again for the same particle.

    Oracle: BBPSOOptimizer.cancel docstring ('The next ask() repeats the step: a move is drawn again').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, _sphere)
    opt.cancel(opt.ask())

    _told_many(opt, _sphere, SWARM)

    assert opt.phase == "enhance"


def test_bbpso_cancel_should_hand_out_the_same_probe_again():
    """Given a Steffensen probe handed out, when it is cancelled and ask() is called, then the same x comes
    back.

    Oracle: BBPSOOptimizer.cancel docstring ('a probe or root trial is handed out again').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)
    opt.q[:, IQI, STEFFENSEN] = 1.0
    probe = opt.ask()
    opt.cancel(probe)

    again = opt.ask()

    assert np.array_equal(again[0].x, probe[0].x)


def test_bbpso_cancel_should_raise_value_error_for_a_candidate_that_is_not_pending():
    """Given a told initial batch, when one of its candidates is cancelled, then ValueError is raised.

    Oracle: BBPSOOptimizer.cancel docstring ('The batch is checked like in tell()').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    cands = _told(opt, _sphere)

    with pytest.raises(ValueError, match="not asked for"):
        opt.cancel(cands[:1])


def test_bbpso_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length():
    """Given the initial batch, when tell() gets one score fewer than candidates, then ValueError is raised.

    Oracle: BBPSOOptimizer.tell docstring (the whole batch is checked first).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    cands = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell(cands, [1.0] * (SWARM - 1))


def test_bbpso_tell_should_draw_no_random_number():
    """Given a move asked, when it is told, then the state of the optimizer generator is unchanged.

    Oracle: module docstring ('tell() checks the whole batch first and draws no random number').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _told(opt, _sphere)
    cands = opt.ask()
    before = opt.rng.bit_generator.state

    opt.tell(cands, [_sphere(c.params) for c in cands])

    assert opt.rng.bit_generator.state == before


def test_bbpso_flags_should_stay_between_one_and_five_over_a_long_run():
    """Given 600 calls on Rastrigin, when the operation indices are read, then every one lies in 1..5 and
    every position lies in the unit cube.

    Oracle: module docstring (five operations; repair clips to [0, 1]).
    """
    space = _box()
    opt = BBPSOOptimizer(space, seed=ANY_SEED)

    optimize(_rastrigin, space, max_evals=600, optimizer=opt, seed=ANY_SEED)

    assert set(opt.flags) <= {1, 2, 3, 4, 5}
    assert bool(np.all((opt.positions >= 0.0) & (opt.positions <= 1.0))) is True


def test_bbpso_should_reach_the_sphere_minimum_within_300_calls():
    """Given the 2-D sphere on [-5, 5]^2, when optimize() runs 300 calls, then the best score is below 1e-3.

    Oracle: CONTRIBUTING (a convergence test on a benchmark function); the sphere is unimodal with its
    minimum 0 equal to the default target, where the root steps aim straight at the optimum.
    """
    space = _box()

    study = optimize(
        _sphere, space, max_evals=300, optimizer=BBPSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    assert study.best_score < 1e-3


def test_bbpso_should_improve_on_its_initial_batch_on_rastrigin():
    """Given the 2-D Rastrigin function, when optimize() runs 300 calls, then the best score is below the best
    of the 20 initial candidates.

    Oracle: CONTRIBUTING (convergence test); Rastrigin is multimodal, so only progress is asserted.
    """
    space = _box()

    study = optimize(
        _rastrigin, space, max_evals=300, optimizer=BBPSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    assert study.best_score < min(t.score for t in study.trials[:SWARM])


def test_bbpso_should_find_the_optimum_region_of_a_mixed_space_for_most_seeds():
    """Given the mixed space and its score, when optimize() runs 300 calls for each of 10 seeds, then at
    least 9 runs end below 0.05 with 'gelu'.

    Oracle: README Behaviour and the module docstring (every parameter kind is decoded by the Space).
    """
    found = sum(map(_found_mixed_optimum, range(10)))

    assert found >= 9


def test_bbpso_should_evaluate_the_same_points_when_maximising_minus_f_with_target_zero():
    """Given the sphere minimised and its negative maximised, both with target 0, when optimize() runs 200
    calls with the same seed, then both studies evaluate the same points.

    Oracle: module docstring (h = s (f - target) with f the internally minimised score, so both runs see
    the same h) and the Optimizer docstring (internally everything is minimised).
    """
    space = _box()

    low = optimize(
        _sphere, space, max_evals=200, optimizer=BBPSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )
    high = optimize(
        lambda p: -_sphere(p),
        space,
        direction="max",
        max_evals=200,
        optimizer=BBPSOOptimizer(space, direction="max", target=0.0, seed=ANY_SEED),
        seed=ANY_SEED,
    )

    assert [tuple(t.x) for t in low.trials] == [tuple(t.x) for t in high.trials]


def test_bbpso_refinement_should_skip_a_base_that_already_sits_on_the_target():
    """Given the initial batch with particle 0 at score 0.0 (the default target for 'min') and every other
    particle at 1.0, and moves scoring 1.0, when the refinement starts, then the leader's slot hands out
    nothing and its Q-table gets exactly one entry of -0.1.

    Oracle: module docstring (the root methods aim at the target; a base with h = 0 has nothing to find)
    and the BBPSO theory (a slot without a candidate is finished inside ask with reward -1).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [0.0] + [1.0] * (SWARM - 1))
    _tell_moves(opt, [1.0] * SWARM)

    opt.ask()

    assert int(np.count_nonzero(opt.q[0])) == 1
    assert float(opt.q[0].sum()) == pytest.approx(-0.1)


def test_bbpso_steffensen_should_finish_the_slot_when_the_probe_scores_nan():
    """Given Steffensen made the best action, when its probe is told NaN, then no trial follows for that slot
    and its Q entry gets reward -1: 1 + 0.1 (-1 + 0.9 * max Q[STEFFENSEN, :] - 1) = 0.8, because the
    Steffensen row is still all zeros.

    Oracle: module docstring (reward -1 unless a point strictly better than the base was produced) and the
    Optimizer docstring (NaN is the worst score).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)
    opt.q[:, IQI, STEFFENSEN] = 1.0
    probe = opt.ask()

    opt.tell(probe, [math.nan])

    leaders = [p for p in range(SWARM) if opt.q[p, IQI, STEFFENSEN] != 1.0]
    assert len(leaders) == 1
    assert float(opt.q[leaders[0], IQI, STEFFENSEN]) == pytest.approx(0.8)


def test_bbpso_cancel_should_handle_initial_candidates_that_share_a_key():
    """Given Int(0, 4) and six particles, so that seed 0 hands out (4,) twice in the initial batch, when the
    first (4,) is cancelled while its twin is pending and the rest is told, then ask() hands out one candidate
    and telling it ends the initial phase.

    Oracle: BBPSOOptimizer.cancel docstring (a cancelled key stays reserved while another candidate with the
    same key is pending) and the Optimizer docstring (a repeat may come back once no new set exists).
    """
    opt = BBPSOOptimizer(Space({"k": Int(0, 4)}), swarm_size=6, seed=ANY_SEED)
    cands = opt.ask()
    opt.cancel(cands[4:5])
    rest = [c for n, c in enumerate(cands) if n != 4]
    opt.tell(rest, [float(c.params["k"]) for c in rest])

    again = opt.ask()
    opt.tell(again, [0.0])

    assert [c.key for c in cands] == [(3.0,), (1.0,), (0.0,), (2.0,), (4.0,), (4.0,)]
    assert (len(again), opt.phase) == (1, "move")


def test_bbpso_cancel_should_keep_the_key_of_the_first_twin_when_the_repeated_twin_is_cancelled():
    """Given the seed-0 initial batch on Int(0, 4) with (4,) at candidates 4 and 5, when candidate 5 (the
    repeat) is cancelled and the rest is told, then ask() hands out one candidate for that particle.

    Oracle: BBPSOOptimizer.cancel docstring (only a candidate that added its key to the seen set frees it).
    """
    opt = BBPSOOptimizer(Space({"k": Int(0, 4)}), swarm_size=6, seed=ANY_SEED)
    cands = opt.ask()
    opt.cancel(cands[5:6])
    opt.tell(cands[:5], [float(c.params["k"]) for c in cands[:5]])

    again = opt.ask()

    assert len(again) == 1
    assert cands[5].key == cands[4].key


def _refine_only(opt, score):
    """Runs the refinement block with the given score and stops at the first move of the next iteration,
    which is returned untold."""
    cands = opt.ask()
    while opt.phase == "enhance":
        opt.tell(cands, [score(c.params) for c in cands])
        cands = opt.ask()
    return cands


def _bowl(params):
    """A quadratic bowl on the Int grid with its minimum at (3, 7)."""
    return (params["a"] - 3) ** 2 + (params["b"] - 7) ** 2


def _move_repeats(opt, tries):
    """After the initial batch, sets max_tries and counts the first iteration's moves that repeat a key
    handed out before."""
    seen = {c.key for c in _told(opt, _bowl)}
    opt.max_tries = tries
    repeats = 0
    for _ in range(opt.swarm_size):
        cand = _told(opt, _bowl)[0]
        repeats += cand.key in seen
        seen.add(cand.key)
    return repeats


def test_bbpso_refinement_should_visit_the_leader_and_the_four_next_best_particles():
    """Given 20 particles with finite scores on the sphere, when every refinement trial is told 1e9, then
    exactly five particles have a Q entry, the leader among them.

    Oracle: module docstring ('the leader and the four next-best particles').
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)

    _refine_only(opt, lambda p: 1e9)

    refined = [p for p in range(SWARM) if np.any(opt.q[p] != 0.0)]
    assert len(refined) == 5


def test_bbpso_refinement_should_put_a_better_trial_into_the_refined_particle():
    """Given the first iteration on the sphere, when every refinement trial is told -1e9, then before any
    further move is told, some particle holds the score -1e9 and the position of a refinement trial.

    Oracle: module docstring ('the refined particle takes the better point', reward +1).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)

    _refine_only(opt, lambda p: -1e9)

    assert -1e9 in opt.f
    holders = [i for i in range(SWARM) if opt.f[i] == -1e9]
    assert any(np.array_equal(opt.positions[i], opt.best_x) for i in holders)


def test_bbpso_second_leader_should_be_the_least_correlated_of_the_five_best_other_particles():
    """Given the first iteration asked on a 4-D box, when the second leader is read, then it is the position,
    among the five best particles other than the leader's owner, whose correlation with G is lowest.

    Oracle: module docstring ('the second leader S is, among the five best other particles, the one whose
    position correlates least with G'). In four dimensions the correlations are distinct.
    """
    opt = BBPSOOptimizer(_box(4), seed=ANY_SEED)
    init = _told(opt, _sphere)
    owner = int(np.argmin(opt.f))

    opt.ask()

    others = sorted((j for j in range(SWARM) if j != owner), key=lambda j: (opt.f[j], j))[:5]
    expected = min(others, key=lambda j: operators.correlation(init[j].x, opt.guide))
    assert np.array_equal(opt.second_leader, init[expected].x)
    assert np.array_equal(opt.guide, init[owner].x)


def test_bbpso_gaussian_operation_should_move_away_from_g_when_a_better_neighbour_exists():
    """Given every particle set to operation 3 after the first iteration, when the second iteration's moves
    are asked, then some move differs from G: the neighbour differs from G, so the Gaussian spread
    |neighbour - G| is not zero.

    Oracle: module docstring (operation 3 samples around c = (neighbour + G) / 2 with standard deviation
    |neighbour - G|; the neighbour is a better particle when one exists).
    """
    opt = BBPSOOptimizer(_box(), seed=ANY_SEED)
    _to_refinement(opt, _sphere)
    first = _refine_only(opt, _sphere)
    opt.cancel(first)
    opt.flags[:] = [3] * SWARM

    moves = [c.x for c in _told_list(opt, _sphere, SWARM)]

    assert any(not np.array_equal(x, opt.guide) for x in moves)


def _told_list(opt, score, asks):
    """Asks and tells `asks` times; returns every candidate."""
    out = []
    for _ in range(asks):
        out.extend(_told(opt, score))
    return out


def test_bbpso_crossover_zero_should_keep_every_coordinate_in_operation_five():
    """Given crossover 0 and every particle set to operation 5, when a move is asked, then it hands out the
    particle's current position.

    Oracle: module docstring (operation 5 is crossed with the current position at rate crossover); with
    rate 0 no coordinate takes the donor, so the move repeats the position and regeneration cannot help.
    """
    opt = BBPSOOptimizer(_box(), crossover=0.0, max_tries=3, seed=ANY_SEED)
    _told(opt, _sphere)
    opt.flags[:] = [5] * SWARM
    before = opt.positions.copy()

    move = opt.ask()

    assert np.array_equal(move[0].x, before[0])


def test_bbpso_moves_should_avoid_earlier_keys_on_a_discrete_space_with_regeneration():
    """Given an 11 x 11 Int grid and the same initial batch, when the first iteration's 20 moves are drawn
    with max_tries = 100, then none repeats an earlier key, while with max_tries = 1 some do.

    Oracle: module docstring ('A candidate whose parameter set was already handed out is drawn again, up to
    max_tries attempts'); the grid has 121 values, so new ones are always available in the first iteration.
    """
    regenerated = _move_repeats(BBPSOOptimizer(Space({"a": Int(0, 10), "b": Int(0, 10)}), seed=ANY_SEED), 100)

    single = _move_repeats(BBPSOOptimizer(Space({"a": Int(0, 10), "b": Int(0, 10)}), seed=ANY_SEED), 1)

    assert regenerated == 0 < single


def test_bbpso_cancel_should_keep_a_key_reserved_when_its_twin_in_the_batch_was_already_told():
    """Given an initial batch on Int(0, 19) built with max_tries = 1, whose candidates 2, 7 and 10 share the
    key (14,), when every candidate except candidate 2 is told, candidate 2 is cancelled and ask() is called
    with max_tries = 100, then the re-asked candidate repeats none of the told keys.

    Oracle: BBPSOOptimizer.cancel docstring (a cancelled key is freed only when no candidate with the same
    key is pending or told). Without that rule seed 14 hands out (14,) again.
    """
    opt = BBPSOOptimizer(Space({"k": Int(0, 19)}), swarm_size=12, seed=14, max_tries=1)
    batch = opt.ask()
    told = [c for n, c in enumerate(batch) if n != 2]
    opt.tell(told, [float(c.params["k"]) for c in told])
    opt.cancel(batch[2:3])
    opt.max_tries = 100

    again = opt.ask()

    assert [batch[n].key for n in (2, 7, 10)] == [(14.0,)] * 3
    assert again[0].key not in {c.key for c in told}
