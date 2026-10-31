"""Boundary tests of ABCOptimizer: thresholds of the colony, acceptance, scout and best-tracking rules.

Oracles: the task specification of ABCOptimizer (colony_size // 2 sources asked as one batch; greedy
acceptance on fitness, strictly greater replaces, otherwise trial += 1; a scout when max trial >= limit, at
most one per cycle; the cycle counter increments when the scout step finishes; a pending candidate makes
ask raise RuntimeError), the abc.py module docstring, the Optimizer docstring (NaN or infinite scores are
the worst value for the direction), the original hyp_optimizer ABC.py (trial.argmax picks the first index
among ties; calculate_probabilities) and Karaboga & Basturk (2007) (partner k != i, phi in [-1, 1]).

Observation is through public ask/tell, Candidate fields and the read-only counters phase, cycle, trial and
food_number; no test reads underscore members or the food matrix.

Rare branches are reached by construction, not by random search (P-T7-29): a constant objective makes every
neighbour fail and every fitness equal, so all onlooker probabilities are 1.0 and the order of single asks
in a cycle is fixed (employed 0, 1, ..., then onlookers 0, 1, ...). With colony_size = 4 (two sources) one
cycle is four single asks, so trial == [2, 2] after cycle 0 whatever the seed. NaN objectives drive the
first-tell-is-NaN branch. ANY_SEED is fixed for determinism only; no assertion depends on the random draws.
LARGE_LIMIT keeps scouts out of tests that do not target them.

Infeasible or out-of-scope test requirements found while reading the code (P-T4-07):
- abc.py _neighbour clamps min(int(u * d), d - 1) and min(int(u * (sn - 1)), sn - 2): redundant decisions.
  For u in [0, 1) and an integer n < 2**53, u * n rounds to a value below n, so the clamp never changes the
  index; they are also not reachable through the public API, because the uniforms are internal.
- abc.py __init__ prob = [1.0] * sn: a dead store. prob is only read in the onlooker phase, and _advance
  always overwrites it when the employed phase ends.
- space.py Parameter base methods raising NotImplementedError: executable, but Parameter is not in
  metahpo.__all__ and the spec does not describe it, so it is outside the public contract.
- space.py linear-Int clip, lower side: low + floor(u * n) >= low for every u >= 0, so it is never active
  (a redundant decision). Its upper side (u = 1.0) is feasible and tested.
Feasible and tested: the Categorical and Int n - 1 clamp at u = 1.0 and the log-Int clip
(test_space_boundaries.py), the onlooker all-zero guard (test_facade_boundaries.py, all-NaN run), the
init-batch RuntimeError, the scout threshold and the first-tell branch (this module).

Unspecified, therefore not asserted:
- infinite Float bounds;
- reusing an optimizer after a truncated initial batch whose rest is never told;
- telling a candidate of a foreign optimizer that has a colliding id;
- tell([], []) and the same candidate twice inside one tell call;
- best_params before any tell, and which candidate best_params reports when every told score is NaN;
- which candidate is best when two told scores are equal (tie order of best).

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: Both sides of every threshold of the colony, acceptance, scout and best-tracking rules.
"""

import math
import sys

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space

ANY_SEED = 0
LARGE_LIMIT = 10**6
CONSTANT_SCORE = 1.0
SOURCE_SCORE = 0.5
WORSE_SCORE = 0.75
REPEAT_RUN_STEPS = 60


def _unit_space_3d():
    """Three Float(0, 1) parameters: a neighbour changes one of three coordinates."""
    return Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0), "c": Float(0.0, 1.0)})


def _ask_and_tell_initial(opt, scores):
    """Asks the initial batch, tells it the given scores in order and returns the candidates."""
    initial = opt.ask()
    opt.tell(initial, scores)
    return initial


def _single_steps(opt, score, steps):
    """Asks and tells `steps` single candidates, each told `score`, and returns the candidates in ask
    order."""
    asked = []
    for _ in range(steps):
        (cand,) = opt.ask()
        opt.tell([cand], [score])
        asked.append(cand)
    return asked


def _two_sources_after_one_failing_cycle(limit):
    """colony_size = 4 with constant scores driven through one whole cycle, so trial == [2, 2]."""
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, limit=limit, seed=ANY_SEED)
    _ask_and_tell_initial(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _single_steps(opt, CONSTANT_SCORE, 2 * opt.food_number)
    return opt


@pytest.mark.parametrize(
    ("colony_size", "expected_sources"),
    [
        pytest.param(4, 2, id="smallest-valid-colony"),
        pytest.param(5, 2, id="odd-colony-rounds-down"),
        pytest.param(6, 3, id="next-even-colony"),
    ],
)
def test_abc_ask_should_hand_out_half_the_colony_rounded_down_when_colony_size_is_at_or_above_four(
    colony_size, expected_sources
):
    """Given colony_size 4, 5 or 6, when the first ask runs, then it returns colony_size // 2 initial
    candidates.

    Oracle: task specification (colony_size >= 4; colony_size // 2 sources asked as one batch) and the
    original ABC.py FOOD_NUMBER = colony / 2. 4 is the valid side of the threshold whose invalid side (3) is
    in tests/test_abc.py; 5 shows floor division and 6 the next step (P-T1-40). Postcondition: food_number
    holds the same count.
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=colony_size, seed=ANY_SEED)

    initial = opt.ask()

    assert len(initial) == expected_sources
    assert opt.food_number == expected_sources


@pytest.mark.parametrize(
    ("deciding_score", "expected_trial"),
    [
        pytest.param(SOURCE_SCORE, [2, 1], id="equal-score-equal-fitness-keeps-source"),
        pytest.param(
            math.nextafter(SOURCE_SCORE, 0.0), [2, 1], id="one-ulp-better-score-rounds-to-equal-fitness"
        ),
        pytest.param(SOURCE_SCORE - 2.0**-20, [0, 1], id="strictly-greater-fitness-replaces-source"),
    ],
)
def test_abc_tell_should_replace_a_source_only_when_the_candidate_fitness_is_strictly_greater(
    deciding_score, expected_trial
):
    """Given source 0 with one failure and its onlooker candidate, when the candidate is told a score near the
    source's, then only strictly greater fitness resets trial[0].

    Oracle: task specification 'greedy acceptance on fitness (strictly greater replaces, else trial += 1)',
    original ABC.py 'if fitness_sol > self.fitness[i]', fitness docstring 1 / (1 + f). Hand calculation:
    equal score gives equal fitness; 1 + (0.5 - 2**-54) rounds to 1.5 in binary64, so the one-ulp-better
    score also has equal fitness (implicit boundary, P-T1-42); 0.5 - 2**-20 gives a larger fitness. trial[0]
    is 1 before the tell, so 'replaced' (0) and 'kept' (2) differ from 'nothing happened' (1). Arrange: both
    sources scored 0.5, both employed told 0.75 (worse), so the onlooker probabilities are equal and the
    next ask is the onlooker of source 0. Postcondition: trial[1] is untouched.
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    _ask_and_tell_initial(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _single_steps(opt, WORSE_SCORE, opt.food_number)
    (onlooker,) = opt.ask()
    assert (opt.phase, opt.trial) == ("onlooker", [1, 1])

    opt.tell([onlooker], [deciding_score])

    assert opt.trial == expected_trial


@pytest.mark.parametrize(
    ("employed_score", "expected_trial_0"),
    [
        pytest.param(math.inf, 1, id="infinite-score-zero-fitness-is-not-greater-than-zero"),
        pytest.param(sys.float_info.max, 0, id="largest-finite-score-has-positive-fitness"),
    ],
)
def test_abc_tell_should_replace_an_infinite_source_only_when_the_candidate_score_is_finite(
    employed_score, expected_trial_0
):
    """Given source 0 scored inf, when its employed candidate is told inf or float max, then only the finite
    score replaces it.

    Oracle: fitness docstring (+inf gives 0; finite f >= 0 gives 1 / (1 + f) > 0), strict greedy acceptance
    from the task specification. Hand calculation: 1 / (1 + 1.797e308) is about 5.6e-309, a positive
    subnormal, so it is strictly greater than 0. This is the implicit boundary between the largest finite
    float and infinity (P-T1-42). Postcondition: none beyond trial[0].
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    _ask_and_tell_initial(opt, [math.inf, 0.0])
    (employed,) = opt.ask()

    opt.tell([employed], [employed_score])

    assert opt.trial[0] == expected_trial_0


SOURCE_MOVE_SEEDS = [
    pytest.param(0, id="seed-0"),
    pytest.param(1, id="seed-1"),
    pytest.param(2, id="seed-2"),
    pytest.param(3, id="seed-3"),
]


def _employed_of_source_0_after_its_onlooker_is_told(seed, deciding_score):
    """Two sources scored SOURCE_SCORE, both employed told WORSE_SCORE, onlooker 0 told `deciding_score`,
    onlooker 1 told WORSE_SCORE; then the employed candidate of source 0 in cycle 1 is asked.

    Returns a dict with 'initial' (initial source 0), 'onlooker' (onlooker candidate of source 0) and
    'employed' (employed candidate of source 0 in cycle 1).
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, limit=LARGE_LIMIT, seed=seed)
    initial = _ask_and_tell_initial(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _single_steps(opt, WORSE_SCORE, opt.food_number)
    (onlooker,) = opt.ask()
    opt.tell([onlooker], [deciding_score])
    _single_steps(opt, WORSE_SCORE, 1)
    (employed,) = opt.ask()
    return {"initial": initial[0], "onlooker": onlooker, "employed": employed}


@pytest.mark.parametrize("seed", SOURCE_MOVE_SEEDS)
@pytest.mark.parametrize(
    ("deciding_score", "expected_source"),
    [
        pytest.param(SOURCE_SCORE, "initial", id="equal-fitness-keeps-the-source-point"),
        pytest.param(
            math.nextafter(SOURCE_SCORE, 0.0), "initial", id="one-ulp-better-score-keeps-the-source-point"
        ),
        pytest.param(
            SOURCE_SCORE - 2.0**-20, "onlooker", id="strictly-greater-fitness-moves-the-source-point"
        ),
    ],
)
def test_abc_tell_should_move_the_source_to_the_candidate_only_when_the_candidate_fitness_is_strictly_greater(
    seed, deciding_score, expected_source
):
    """Given onlooker 0 told a score at the greedy boundary, when the employed bee of source 0 is asked next
    cycle, then it neighbours `expected_source`: the onlooker candidate for strictly greater fitness, initial
    source 0 otherwise.

    Oracle: task specification 'greedy acceptance on fitness (strictly greater replaces)', original ABC.py
    'self.foods[i][:] = np.copy(solution)' inside 'if fitness_sol > self.fitness[i]', and Karaboga & Basturk
    (2007): a neighbour differs from its source in one coordinate. So the next candidate of source 0 differs
    in at most one coordinate from the source point, named by the expected_source column: 'onlooker' when it
    replaced the source, 'initial' when it did not. Hand calculation of the deciding scores as in the trial
    test above (P-T1-40, P-T1-42). Order: all fitness values are equal when the probabilities are computed,
    so every probability is 1.0 and the next asks are onlooker 1 and then (no scout under LARGE_LIMIT)
    employed 0.
    Site: the source update of the greedy branch, which the trial counters alone do not observe. Seeds: the
    assertion holds for every seed; it is decisive only when the employed bee moves another coordinate than
    the onlooker did, which is not the case for seed 0, so four fixed seeds are used.
    """
    observed = _employed_of_source_0_after_its_onlooker_is_told(seed, deciding_score)

    assert int(np.count_nonzero(observed["employed"].x != observed[expected_source].x)) <= 1


def _first_onlooker_after_infinite_source_0_is_told(seed, employed_score):
    """Two sources scored inf, the employed bee of source 0 told `employed_score` and that of source 1 told
    inf; then the first onlooker is asked.

    Returns a dict with 'initial' (initial source 0), 'employed' (employed candidate of source 0) and
    'onlooker' (first onlooker candidate).
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, limit=LARGE_LIMIT, seed=seed)
    initial = _ask_and_tell_initial(opt, [math.inf, math.inf])
    (employed,) = opt.ask()
    opt.tell([employed], [employed_score])
    _single_steps(opt, math.inf, 1)
    (onlooker,) = opt.ask()
    return {"initial": initial[0], "employed": employed, "onlooker": onlooker}


@pytest.mark.parametrize("seed", SOURCE_MOVE_SEEDS)
@pytest.mark.parametrize(
    ("employed_score", "expected_source"),
    [
        pytest.param(math.inf, "initial", id="infinite-score-keeps-the-source-point"),
        pytest.param(sys.float_info.max, "employed", id="largest-finite-score-moves-the-source-point"),
    ],
)
def test_abc_tell_should_move_an_infinite_source_to_the_candidate_only_when_the_candidate_score_is_finite(
    seed, employed_score, expected_source
):
    """Given source 0 scored inf and its employed bee told inf or float max, when the first onlooker is asked,
    then it neighbours `expected_source`: the employed candidate for a finite score, initial source 0 for inf.

    Oracle: fitness docstring (+inf gives 0, float max gives about 5.6e-309 > 0), strict greedy acceptance
    from the task specification and original ABC.py, and Karaboga & Basturk (2007) (one coordinate per
    neighbour). The first onlooker is the one of source 0: original ABC.py starts the roulette sweep at i = 0,
    and onlooker_probabilities gives p[0] = 0.9 * fit / max + 0.1 = 1.0 when source 0 holds the only positive
    fitness and ones when all fitness values are zero, so a uniform draw below 1.0 always selects it (the
    subnormal rounding reported as defect D3 gives slightly more than 1.0 here, which selects it as well).
    The expected_source column names the source point: 'employed' when the finite score replaced the source,
    'initial' when inf kept it. This is the source side of the finite/infinite boundary (P-T1-42). Seeds as
    in the previous test: the assertion holds for every seed and is decisive when the two moves touch
    different coordinates.
    """
    observed = _first_onlooker_after_infinite_source_0_is_told(seed, employed_score)

    assert int(np.count_nonzero(observed["onlooker"].x != observed[expected_source].x)) <= 1


@pytest.mark.parametrize(
    ("first_score", "expected_best", "expected_index"),
    [
        pytest.param(math.inf, 0.25, 1, id="infinite-score-is-worst-when-maximising"),
        pytest.param(
            sys.float_info.max, sys.float_info.max, 0, id="largest-finite-score-is-best-when-maximising"
        ),
    ],
)
def test_abc_best_score_should_ignore_infinity_but_take_the_largest_float_when_direction_is_max(
    first_score, expected_best, expected_index
):
    """Given direction 'max', when the initial batch is told [first_score, 0.25], then inf is never best but
    float max is.

    Oracle: README 'a score of NaN or infinity counts as the worst possible value for the chosen direction'
    and the Optimizer docstring. inf and float max are the two sides of the implicit finite/infinite boundary
    (P-T1-42). Postcondition: best_params are the params of the best candidate.
    """
    opt = ABCOptimizer(_unit_space_3d(), direction="max", colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [first_score, 0.25])

    assert opt.best_score == expected_best
    assert opt.best_params == initial[expected_index].params


@pytest.mark.parametrize(
    ("limit", "expected_phase", "expected_cycle"),
    [
        pytest.param(2, "scout", 0, id="max-trial-equal-to-limit-issues-a-scout"),
        pytest.param(3, "employed", 1, id="max-trial-one-below-limit-skips-the-scout"),
    ],
)
def test_abc_ask_should_issue_a_scout_only_when_the_largest_trial_reaches_limit(
    limit, expected_phase, expected_cycle
):
    """Given two sources with trial == [2, 2] after one failing cycle, when the next ask runs, then a scout is
    pending only if 2 >= limit.

    Oracle: task specification 'scout if max trial >= limit ... cycle counter increments when the scout step
    finishes' and original ABC.py send_scout_bees 'np.amax(self.trial) >= self.conf.LIMIT'. limit = 2 and
    limit = 3 are both sides of the threshold (P-T1-40). With a scout pending the cycle is not finished yet
    (phase 'scout', cycle 0); without one the scout step finishes inside ask (cycle 1, phase 'employed').
    """
    opt = _two_sources_after_one_failing_cycle(limit)
    assert (opt.phase, opt.cycle, opt.trial) == ("scout", 0, [2, 2])

    opt.ask()

    assert (opt.phase, opt.cycle) == (expected_phase, expected_cycle)


def test_abc_tell_should_finish_the_cycle_when_the_scout_candidate_is_told():
    """Given a pending scout (limit = 2, trial == [2, 2]), when it is told, then the cycle counter becomes 1
    and the next phase is employed.

    Oracle: task specification 'cycle counter increments when the scout step finishes' (P-T1-40, the side
    where the scout fires). Postcondition: the replaced source starts again with trial 0.
    """
    opt = _two_sources_after_one_failing_cycle(limit=2)
    (scout,) = opt.ask()

    opt.tell([scout], [CONSTANT_SCORE])

    assert (opt.phase, opt.cycle) == ("employed", 1)
    assert opt.trial[0] == 0


def test_abc_scout_should_replace_the_first_source_when_trials_tie():
    """Given limit = 1 and trial == [2, 2], when the scout is asked and told, then source 0 is replaced and
    source 1 keeps trial 2.

    Oracle: original ABC.py 'self.init(self.trial.argmax(axis=0))': numpy argmax returns the first index among
    ties. A tie is the code-level boundary of the argmax choice (P-T1-44), reached by construction (P-T7-29).
    """
    opt = _two_sources_after_one_failing_cycle(limit=1)
    (scout,) = opt.ask()

    opt.tell([scout], [CONSTANT_SCORE])

    assert opt.trial == [0, 2]


def test_abc_ask_should_not_issue_a_second_scout_in_the_same_cycle_when_another_source_is_over_limit():
    """Given limit = 1 and a told scout for source 0 while trial[1] == 2 >= limit, when the next ask runs,
    then it is the employed bee of source 0.

    Oracle: task specification 'at most one per cycle', abc.py docstring 'at most one scout per cycle',
    Karaboga & Basturk (2007). limit = 1 is the smallest valid limit, so every source is over it after one
    failure (P-T1-40, P-T1-44). Postcondition: phase is employed and trial is unchanged by the ask.
    """
    opt = _two_sources_after_one_failing_cycle(limit=1)
    (scout,) = opt.ask()
    opt.tell([scout], [CONSTANT_SCORE])
    assert opt.trial == [0, 2]

    (employed,) = opt.ask()

    assert (opt.phase, opt.cycle, opt.trial) == ("employed", 1, [0, 2])
    assert np.count_nonzero(employed.x != scout.x) <= 1


def test_abc_scout_should_replace_the_source_with_the_largest_trial_when_it_is_unique():
    """Given limit = 1 after the tie scout of cycle 0, when cycle 1 fails everywhere and its scout is told,
    then source 1 (trial 4 > 2) is replaced.

    Oracle: original ABC.py trial.argmax and the task specification 'replace that source'. Hand count: after
    the first scout trial == [0, 2]; four failing single asks give [2, 4], so the argmax is 1. Together with
    the tie test this separates 'first argmax' from 'always source 0' (P-T1-44).
    """
    opt = _two_sources_after_one_failing_cycle(limit=1)
    _single_steps(opt, CONSTANT_SCORE, 1)
    _single_steps(opt, CONSTANT_SCORE, 2 * opt.food_number)
    assert (opt.phase, opt.cycle, opt.trial) == ("scout", 1, [2, 4])

    _single_steps(opt, CONSTANT_SCORE, 1)

    assert (opt.cycle, opt.trial) == (2, [2, 0])


def _neighbour_xs_with_fixed_two_sources(steps):
    """1-d space, two sources scored 0.0 and every later candidate told 1.0, so the sources never move.

    Returns (initial source values, values of the `steps` single candidates in ask order).
    """
    opt = ABCOptimizer(Space({"a": Float(0.0, 1.0)}), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    initial = _ask_and_tell_initial(opt, [0.0, 0.0])
    asked = _single_steps(opt, 1.0, steps)
    return np.array([c.x[0] for c in initial]), np.array([c.x[0] for c in asked])


def test_abc_neighbour_should_always_use_the_other_source_as_partner_when_there_are_two_sources():
    """Given two fixed sources in a 1-d space, when 40 single candidates are asked, then each is x_i + phi *
    (x_i - x_k) with k the other source.

    Oracle: Karaboga & Basturk (2007) v_ij = x_ij + phi_ij (x_ij - x_kj) with k != i and phi in [-1, 1], and
    neighbour_value clipping to [0, 1]. With two sources the partner range has one element, the code-level
    boundary of the partner choice (P-T1-44). A self-partner would give v == x_i exactly. Candidate order is
    0, 1, 0, 1, ... because equal fitness makes every onlooker probability 1.0. Each candidate must be
    clipped (0 or 1) or move by at most |x_i - x_k| and differ from x_i; phi == 0 exactly has probability 0.
    """
    steps = 40
    sources, values = _neighbour_xs_with_fixed_two_sources(steps)
    own = sources[np.tile([0, 1], steps // 2)]
    other = sources[np.tile([1, 0], steps // 2)]

    clipped = (values == 0.0) | (values == 1.0)
    moved_within_partner_gap = (values != own) & (np.abs(values - own) <= np.abs(own - other) * (1 + 1e-12))

    np.testing.assert_array_equal(clipped | moved_within_partner_gap, np.full(steps, True))


def test_abc_should_reach_the_same_state_when_the_initial_batch_is_told_in_reverse_order():
    """Given two optimizers with the same seed, when one tells the initial batch in order and the other in
    reverse (scores alongside), then best and next candidate agree.

    Oracle: abc.py docstring 'only the initial population is handed out as a batch, because its evaluations
    are independent', task specification 'may be told partially', README 'the same seed reproduces the same
    search'. The order of independent results is a permutation that must not matter (P-T2-33). The scores
    are distinct, because the tie order of best is unspecified. Postcondition: best_score, best_params and
    the next candidate.
    """
    in_order = ABCOptimizer(_unit_space_3d(), colony_size=6, seed=ANY_SEED)
    reversed_order = ABCOptimizer(_unit_space_3d(), colony_size=6, seed=ANY_SEED)
    distinct_scores = [0.75, 0.25, 0.5]
    first, second = in_order.ask(), reversed_order.ask()
    in_order.tell(first, distinct_scores)

    reversed_order.tell(second[::-1], distinct_scores[::-1])

    assert reversed_order.best_score == in_order.best_score
    assert reversed_order.best_params == in_order.best_params
    np.testing.assert_array_equal(reversed_order.ask()[0].x, in_order.ask()[0].x)


@pytest.mark.parametrize(
    ("direction", "worst"),
    [
        pytest.param("min", math.inf, id="min-worst-is-plus-inf"),
        pytest.param("max", -math.inf, id="max-worst-is-minus-inf"),
    ],
)
def test_abc_best_score_should_be_the_worst_value_when_every_initial_score_is_nan(direction, worst):
    """Given a fresh optimizer, when its whole initial batch is told NaN, then best_score is the worst value
    of the direction.

    Oracle: Optimizer docstring 'a NaN or infinite score becomes the worst possible value for the direction
    (-inf when maximising, +inf when minimising)'. The NaN-first tell reaches the code-level branch that
    records a best point even for a non-finite score (best_x is None, P-T1-44, P-T7-29), but that branch only
    affects best_params, which is unspecified after an all-NaN batch and not asserted; best_score does not
    depend on the branch. The assertion covers the specified part only: best_score is the worst value.
    """
    opt = ABCOptimizer(_unit_space_3d(), direction=direction, colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [math.nan, math.nan])

    assert opt.best_score == worst


def test_abc_best_should_move_to_the_first_finite_score_when_all_earlier_scores_were_nan():
    """Given an initial batch told all NaN, when the first employed candidate is told 0.25, then it becomes
    the best.

    Oracle: Optimizer docstring (NaN is the worst value), so any finite score is better; best_params are that
    candidate's params. This is the side right after the NaN-first branch (P-T1-44).
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, seed=ANY_SEED)
    _ask_and_tell_initial(opt, [math.nan, math.nan])
    (employed,) = opt.ask()

    opt.tell([employed], [0.25])

    assert (opt.best_score, opt.best_params) == (0.25, employed.params)


def test_abc_best_should_stay_when_a_nan_score_follows_a_finite_one():
    """Given a best score 0.25 from the initial batch, when the next candidate is told NaN, then the best is
    unchanged.

    Oracle: Optimizer docstring (NaN is the worst value, so it never improves on a finite best).
    Postcondition: best_params are still those of the candidate scored 0.25.
    """
    opt = ABCOptimizer(_unit_space_3d(), colony_size=4, seed=ANY_SEED)
    initial = _ask_and_tell_initial(opt, [0.25, 0.5])
    (employed,) = opt.ask()

    opt.tell([employed], [math.nan])

    assert (opt.best_score, opt.best_params) == (0.25, initial[0].params)


def test_abc_ask_should_raise_runtime_error_when_one_initial_candidate_is_still_pending():
    """Given the default colony (5 sources) with 4 initial candidates told, when ask runs again, then
    RuntimeError is raised.

    Oracle: task specification 'init ... may be told partially' and 'asking again with a pending candidate ->
    RuntimeError'. One pending candidate is the boundary between 'all pending' (tested elsewhere) and 'none
    pending' (next test) (P-T1-44). Precondition: the phase is still init.
    """
    opt = ABCOptimizer(_unit_space_3d(), seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial[:-1], [CONSTANT_SCORE] * (len(initial) - 1))
    assert (len(initial), opt.phase) == (5, "init")

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_ask_should_hand_out_one_employed_candidate_when_the_last_initial_candidate_is_told_late():
    """Given 4 of 5 initial candidates told, when the last one is told and ask runs, then one candidate comes
    back in the employed phase.

    Oracle: task specification (init may be told partially; employed phase hands out one candidate per ask).
    This is the 'none pending' side of the partial-tell boundary (P-T1-44). Postcondition: the candidate
    differs from initial source 0 in at most one coordinate (Karaboga & Basturk 2007 neighbour).
    """
    opt = ABCOptimizer(_unit_space_3d(), seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial[:-1], [CONSTANT_SCORE] * (len(initial) - 1))
    opt.tell(initial[-1:], [CONSTANT_SCORE])

    asked = opt.ask()

    assert (len(asked), opt.phase) == (1, "employed")
    assert np.count_nonzero(asked[0].x != initial[0].x) <= 1


@pytest.mark.parametrize(
    ("max_tries", "expected_repeat"),
    [
        pytest.param(1, True, id="one-try-hands-out-the-repeat"),
        pytest.param(100, False, id="default-tries-find-a-new-set"),
    ],
)
def test_abc_ask_should_hand_out_a_repeated_parameter_set_only_when_max_tries_is_too_small(
    max_tries, expected_repeat
):
    """Given Int(1, 3) x Categorical(['p', 'q']) x Float(0, 1) and a constant objective, when the initial
    batch and 60 single rounds are asked with max_tries 1 or the default 100, then a key is handed out twice
    with max_tries=1 and never with 100.

    Oracle: Optimizer docstring (a parameter set handed out before comes back only when no new one is found
    in `max_tries` attempts). Input: 1 is the smallest valid value (0 raises, see test_abc_protocol.py), so
    a neighbour that keeps the Int or Categorical value of its source is handed out at once; with 100
    attempts the Float coordinate is drawn sooner or later and gives a new set, because the Float makes the
    space infinite. Same seed for both cases, so only max_tries differs.
    """
    opt = ABCOptimizer(
        Space({"a": Int(1, 3), "b": Categorical(["p", "q"]), "c": Float(0.0, 1.0)}),
        max_tries=max_tries,
        seed=ANY_SEED,
    )
    initial = _ask_and_tell_initial(opt, [CONSTANT_SCORE] * opt.food_number)

    singles = _single_steps(opt, CONSTANT_SCORE, REPEAT_RUN_STEPS)

    keys = [cand.key for cand in initial + singles]
    assert (len(set(keys)) < len(keys)) is expected_repeat
