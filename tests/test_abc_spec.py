"""Specification gaps of ABCOptimizer beyond tests/test_abc.py.

Oracles: the metahpo.optimizers.abc module docstring, the computed spec of the phases and protocol errors,
and the original hyp_optimizer ABC.py (send_employed_bees lines 113-121, send_onlooker_bees lines 129-149,
send_scout_bees lines 151-153, init lines 70-81) with Karaboga & Basturk (2007). Observation uses only
candidates returned by ask(), best values, exceptions and the documented counters phase, cycle and trial.

Food sources are followed with a reference model built from public candidates: the sources start as the
initial batch, and a neighbour belongs to the source it differs from in at most one coordinate (abc.py: 'A
neighbour differs from its food source in one coordinate'). With continuous floats a neighbour differs
from every other source in at least two coordinates.

Unspecified and therefore not asserted: what a candidate whose id collides with a pending id of another
optimizer does, whether tell() is atomic when a later candidate in the list is invalid, and the exact
onlooker sequence of a seeded run.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: One test per specification gap that test_abc.py leaves open.
"""

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Float, Space

ANY_SEED = 0
PROBABILITY_SEED = 0
WITNESS_SEED = 3
TIE_SCORE = 1.0
WORSE_SCORE = 50.0


def _unit_space(d):
    """d linear floats on [0, 1]."""
    return Space({f"x{i}": Float(0.0, 1.0) for i in range(d)})


def _finish_initial_batch(opt, scores):
    """Asks the initial batch, tells the given scores and returns the batch."""
    cands = opt.ask()
    opt.tell(cands, scores)
    return cands


def _tell_single(opt, n_asks, score):
    """Asks and tells n_asks single candidates with one constant score; returns them."""
    asked = []
    for _ in range(n_asks):
        (cand,) = opt.ask()
        opt.tell([cand], [score])
        asked.append(cand)
    return asked


def _copied_points(cands):
    """Independent copies of the x of each candidate, the starting food sources of the reference model."""
    return [c.x.copy() for c in cands]


def _source_of(sources, x):
    """(index, number of differing coordinates) of the model source that differs least from x."""
    distances = [int(np.count_nonzero(x != s)) for s in sources]
    i = int(np.argmin(distances))
    return i, distances[i]


def _onlooker_sources_after_employed(opt, employed_scores, onlooker_score, sources):
    """Tells the employed phase with employed_scores, then the onlooker phase with one score.

    An employed candidate that improves its source (a negative score here) replaces it in the model.
    Returns the model source index of every onlooker candidate.
    """
    for score in employed_scores:
        (cand,) = opt.ask()
        i, _ = _source_of(sources, cand.x)
        opt.tell([cand], [score])
        if score < 0:
            sources[i] = cand.x.copy()
    picked = []
    while opt.phase == "onlooker":
        (cand,) = opt.ask()
        picked.append(_source_of(sources, cand.x)[0])
        opt.tell([cand], [onlooker_score])
    return picked


def _improving_run(opt, sources, n_asks, first_score):
    """Tells n_asks single candidates with strictly decreasing scores, so every candidate is accepted.

    Returns (largest coordinate distance of an onlooker candidate to its model source, number of witness
    onlookers). A witness is an onlooker on a source already improved in the same onlooker phase whose
    candidate differs from the pre-improvement source in more than one coordinate, so it can only be a
    neighbour of the improved source.
    """
    score = first_score
    largest_distance = 0
    witnesses = 0
    before_improvement = {}
    for _ in range(n_asks):
        phase = opt.phase
        (cand,) = opt.ask()
        i, distance = _source_of(sources, cand.x)
        if phase == "onlooker":
            largest_distance = max(largest_distance, distance)
            old = before_improvement.get(i)
            if old is not None and np.count_nonzero(cand.x != old) > 1:
                witnesses += 1
            before_improvement.setdefault(i, sources[i].copy())
        else:
            before_improvement = {}
        opt.tell([cand], [score])
        score -= 1.0
        sources[i] = cand.x.copy()
    return largest_distance, witnesses


def _scout_cycles(opt, n_asks):
    """Tells n_asks single candidates: ties for neighbours, WORSE_SCORE for scouts. Returns the cycle of each
    scout. A candidate is a scout when the phase is still 'scout' after ask() returned it."""
    cycles = []
    for _ in range(n_asks):
        cycle = opt.cycle
        (cand,) = opt.ask()
        if opt.phase == "scout":
            cycles.append(cycle)
            opt.tell([cand], [WORSE_SCORE])
        else:
            opt.tell([cand], [TIE_SCORE])
    return cycles


def _exhaust_all_sources_twice(opt):
    """Ties the initial batch of three sources and one employed plus one onlooker visit per source.

    With equal fitness every onlooker probability is 0.9 * 1 + 0.1 = 1, so the roulette sweep visits sources
    0, 1, 2 in order and every trial counter ends at 2 whatever the seed. Returns the initial batch.
    """
    batch = _finish_initial_batch(opt, [TIE_SCORE] * 3)
    _tell_single(opt, 6, TIE_SCORE)
    return batch


@pytest.mark.parametrize("limit", [pytest.param(0, id="zero"), pytest.param(-1, id="negative")])
def test_abc_should_raise_value_error_when_limit_is_below_one(limit):
    """Given limit below 1, when ABCOptimizer is built, then ValueError naming the limit is raised.

    Oracle: computed spec 'limit=15 (>=1)' and 'invalid colony_size/limit -> ValueError'. Class: the invalid
    side of limit; test_abc.py only covers colony_size = 3. The valid boundary limit = 1 is used below.
    """
    space = _unit_space(2)

    with pytest.raises(ValueError, match="limit"):
        ABCOptimizer(space, limit=limit, seed=ANY_SEED)


@pytest.mark.parametrize(
    ("source_score", "new_score", "expected_trials"),
    [
        pytest.param(1.0, 0.5, [0, 0], id="better_replaces"),
        pytest.param(1.0, 1.0, [1, 0], id="tie_keeps_source"),
        pytest.param(1.0, 2.0, [1, 0], id="worse_keeps_source"),
        pytest.param(-1.0, -2.0, [0, 0], id="better_in_negative_branch"),
        pytest.param(-1.0, -1.0, [1, 0], id="tie_in_negative_branch"),
        pytest.param(1e-17, 0.0, [1, 0], id="better_score_equal_fitness_keeps_source"),
    ],
)
def test_abc_employed_tell_should_reset_trial_only_on_strictly_greater_fitness(
    source_score, new_score, expected_trials
):
    """Given two sources told source_score, when the first employed bee (source 0) is told new_score, then
    trial[0] is reset to 0 only when the fitness is strictly greater and otherwise becomes 1.

    Oracle: ABC.py line 113 'if fitness_sol > self.fitness[i]:' (trial = 0, else trial + 1) and the fitness
    formula 1 / (1 + f) for f >= 0, 1 + |f| for f < 0. Decisive inputs: the tie, the negative branch, and
    1e-17 versus 0.0, where the raw score is better but 1 / (1 + 1e-17) == 1 / (1 + 0) in float64, which
    tells a comparison on fitness apart from one on raw scores. Source 1 is untouched and stays at 0.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=4, seed=ANY_SEED)
    _finish_initial_batch(opt, [source_score] * 2)
    (cand,) = opt.ask()

    opt.tell([cand], [new_score])

    assert opt.trial == expected_trials


def test_abc_onlookers_should_follow_probabilities_of_the_employed_result():
    """Given five sources where source 0 is best after initialisation and source 1 becomes far better in the
    employed phase, when the onlooker phase runs, then source 1 is visited more often than source 0.

    Oracle: abc.py docstring 'onlookers chosen by roulette sweep (p = 0.9 * fit / max + 0.1)' and the
    original loop order send_employed_bees, calculate_probabilities, send_onlooker_bees. Hand calculation:
    after the employed phase fit = [1, 1e6 + 1, ~1e-6, ~1e-6, ~1e-6], so p_1 = 1 and every other p is about
    0.1; probabilities taken before the employed phase would give p_0 = 1 instead, and never recomputed
    ones would visit every source once. Onlookers are told the worst score, so no source changes. With
    p_1 = 1 against p_0 of about 0.1 the inequality is a statistical expectation, not a certainty: it holds
    for PROBABILITY_SEED and for 497 of the seeds 0..499, so this test keeps its own seed constant.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=10, seed=PROBABILITY_SEED)
    batch = _finish_initial_batch(opt, [0.0, 1e6, 1e6, 1e6, 1e6])
    sources = _copied_points(batch)

    picked = _onlooker_sources_after_employed(opt, [1e6, -1e6, 1e6, 1e6, 1e6], 1e6, sources)

    assert len(picked) == 5
    assert picked.count(1) > picked.count(0)


def test_abc_onlooker_should_build_on_a_source_improved_earlier_in_the_same_phase():
    """Given every tell improving its source, when 600 single candidates run, then every onlooker candidate
    is a one-coordinate neighbour of the current model source, including onlookers that revisit a source
    improved earlier in the same onlooker phase.

    Oracle: abc.py docstring 'with each improvement visible to the next onlooker' and ABC.py
    send_onlooker_bees, which writes self.foods[i] inside the onlooker loop (lines 140-146). WITNESS_SEED 3
    with colony_size 6 and four coordinates was chosen because it produces witness revisits; without a
    witness the distance check could not tell an updated source from a stale copy. limit = 10**6 rules out
    scouts.
    """
    opt = ABCOptimizer(_unit_space(4), colony_size=6, limit=10**6, seed=WITNESS_SEED)
    batch = _finish_initial_batch(opt, [100.0] * 3)
    sources = _copied_points(batch)

    largest_distance, witnesses = _improving_run(opt, sources, 600, 99.0)

    assert largest_distance <= 1
    assert witnesses >= 1


def test_abc_should_issue_exactly_one_scout_per_cycle_when_limit_is_one():
    """Given limit = 1 and tied neighbour scores, when 60 single candidates run, then every cycle that sent
    a scout sent exactly one, and at least five cycles did.

    Oracle: abc.py docstring 'at most one scout per cycle' and ABC.py send_scout_bees, which calls init for
    one source when max(trial) >= LIMIT. limit = 1 is the smallest valid limit, so a source is exhausted in
    every cycle and the at-most-one rule is the only thing that keeps the count at one.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=1, seed=ANY_SEED)
    _finish_initial_batch(opt, [TIE_SCORE] * 3)

    scout_cycles = _scout_cycles(opt, 60)

    assert sorted(set(scout_cycles)) == sorted(scout_cycles)
    assert len(scout_cycles) >= 5


@pytest.mark.parametrize(
    ("limit", "expect_scout"),
    [
        pytest.param(2, True, id="max_trial_equals_limit"),
        pytest.param(3, False, id="max_trial_one_below_limit"),
    ],
)
def test_abc_should_send_a_scout_only_when_max_trial_reaches_limit(limit, expect_scout):
    """Given every trial counter at 2, when the scout step asks, then a scout is issued for limit 2 and none
    for limit 3.

    Oracle: ABC.py line 152 'if np.amax(self.trial) >= self.conf.LIMIT'. Boundary pair: trial == limit
    scouts, trial == limit - 1 does not. Without a scout ask() ends the cycle and returns an employed bee,
    so the phase after ask() tells the two apart.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=limit, seed=ANY_SEED)
    _exhaust_all_sources_twice(opt)

    opt.ask()

    assert (opt.phase == "scout") is expect_scout


def test_abc_scout_should_reset_the_trial_of_the_first_most_exhausted_source():
    """Given trial counters [2, 2, 2] and limit 2, when the scout is asked and told, then trial becomes
    [0, 2, 2].

    Oracle: ABC.py line 153 'self.init(self.trial.argmax(axis=0))', whose argmax picks the first of equal
    maxima, and init line 81 'self.trial[index] = 0'. Class: the tie-break between equally exhausted
    sources, which test_abc.py does not look at.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=2, seed=ANY_SEED)
    _exhaust_all_sources_twice(opt)
    (scout,) = opt.ask()

    opt.tell([scout], [WORSE_SCORE])

    assert opt.trial == [0, 2, 2]


def test_abc_scout_should_replace_its_source_even_when_the_scout_score_is_worse():
    """Given a scout for source 0 told a worse score than the source had, when the next employed bee for
    source 0 is asked, then it is a one-coordinate neighbour of the scout point, not of the old source.

    Oracle: ABC.py init (lines 70-81) assigns the random point to foods[index] without comparing scores.
    Class: unconditional replacement; test_abc.py only checks that a scout occurs.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=2, seed=ANY_SEED)
    batch = _exhaust_all_sources_twice(opt)
    (scout,) = opt.ask()
    opt.tell([scout], [WORSE_SCORE])

    (employed,) = opt.ask()

    assert np.count_nonzero(employed.x != scout.x) <= 1
    assert np.count_nonzero(employed.x != batch[0].x) >= 2


def test_abc_cycle_should_increment_when_the_scout_tell_finishes():
    """Given a pending scout in cycle 0, when it is told, then cycle is 1 and the phase is 'employed'.

    Oracle: computed spec 'cycle counter increments when the scout step finishes' and ABCAlgorithm order
    send_scout_bees then increase_cycle. Class: the branch with a scout; test_abc.py covers only the cycle
    increment without a scout (limit = 1000).
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=2, seed=ANY_SEED)
    _exhaust_all_sources_twice(opt)
    (scout,) = opt.ask()
    assert (opt.cycle, opt.phase) == (0, "scout")

    opt.tell([scout], [WORSE_SCORE])

    assert (opt.cycle, opt.phase) == (1, "employed")


@pytest.mark.parametrize(
    ("colony_size", "expected_sources"),
    [pytest.param(4, 2, id="smallest_colony"), pytest.param(7, 3, id="odd_colony")],
)
def test_abc_initial_batch_should_hold_colony_size_floor_half_candidates(colony_size, expected_sources):
    """Given colony_size 4 or 7, when the initial batch is asked, then it holds colony_size // 2 candidates.

    Oracle: computed spec 'colony_size//2 sources asked as one batch'. Boundaries: 4 is the smallest valid
    colony and 7 is odd, where the floor division matters; test_abc.py only uses 10.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=colony_size, seed=ANY_SEED)

    batch = opt.ask()

    assert len(batch) == expected_sources


@pytest.mark.parametrize(
    "n_told", [pytest.param(0, id="whole_batch_pending"), pytest.param(1, id="part_pending")]
)
def test_abc_ask_should_raise_runtime_error_while_initial_candidates_are_pending(n_told):
    """Given an initial batch of three with none or one told, when ask() is called, then RuntimeError is
    raised.

    Oracle: computed spec 'asking again with a pending candidate -> RuntimeError' and 'may be told
    partially'. Class: pending candidates of the initial batch; test_abc.py covers only a pending employed
    candidate.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=7, seed=ANY_SEED)
    batch = opt.ask()
    opt.tell(batch[:n_told], [TIE_SCORE] * n_told)

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_should_enter_employed_phase_when_initial_batch_is_told_in_two_parts():
    """Given an initial batch of three told as one and then two candidates, when the next candidate is
    asked, then the phase is 'employed' and ask() returns a single candidate.

    Oracle: computed spec 'init (colony_size//2 sources asked as one batch, may be told partially)' and
    'employed (one source after another, one candidate per ask)'. Class: a split tell of the initial batch.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=7, seed=ANY_SEED)
    batch = opt.ask()
    opt.tell(batch[:1], [1.0])
    opt.tell(batch[1:], [2.0, 3.0])

    asked = opt.ask()

    assert opt.phase == "employed"
    assert len(asked) == 1


def test_abc_tell_should_raise_value_error_when_lengths_differ():
    """Given an initial batch of five, when tell() gets four scores, then ValueError about the length is
    raised.

    Oracle: computed spec 'mismatched lengths -> ValueError'. Class: length mismatch, absent from test_abc.py.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=10, seed=ANY_SEED)
    batch = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell(batch, [TIE_SCORE] * (len(batch) - 1))


def test_abc_tell_should_raise_value_error_when_candidate_was_never_asked_for():
    """Given a candidate built by hand with an id the optimizer never issued, when it is told, then
    ValueError is raised.

    Oracle: computed spec 'telling an unknown/already-told candidate -> ValueError'. Class: an unknown id;
    test_abc.py covers only an already-told candidate. The candidate is built from the real space.rows().
    """
    space = _unit_space(3)
    opt = ABCOptimizer(space, colony_size=10, seed=ANY_SEED)
    _finish_initial_batch(opt, [TIE_SCORE] * 5)
    x = np.array([0.5, 0.5, 0.5])
    (key,), (params,) = space.rows(x[None])
    stranger = Candidate(10**6, x, params, key)

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([stranger], [TIE_SCORE])
