"""Equivalence classes of ABCOptimizer: configuration, protocol, score classes, greedy acceptance and scout.

Oracles: the original hyp_optimizer ABC.py (send_employed_bees / send_onlooker_bees: 'fitness_sol >
self.fitness[i]' replaces and resets trial, otherwise trial + 1; calculate_probabilities; send_scout_bees:
'np.amax(self.trial) >= LIMIT' re-initialises 'self.trial.argmax()'), Config.py (FOOD_NUMBER =
NUMBER_OF_POPULATION / 2), Karaboga & Basturk (2007), the abc.py module docstring (employed bees one source
after another, roulette sweep, at most one scout per cycle, only the initial population as a batch), the
Optimizer docstring (scores in the user's direction, NaN or infinite score is the worst value: +inf when
minimising, -inf when maximising), README Behaviour, and the task spec for the protocol errors and for the
cycle counter (it increments when the scout step finishes).

Class table (feature, viewpoint, class id, definition, representative, expected behaviour, covering test)

ABC, configuration
  K1  colony_size < 4        0, 1, 2, 3 (exhaustive)     ValueError
      test_abc_optimizer_should_raise_value_error_when_colony_size_is_0_to_3
  K2  colony_size even >= 4  4 (edge), 6 .. 12           SN = colony_size // 2 sources, first ask SN
  K3  colony_size odd >= 5   5, 7 .. 11 (11 interior)    SN = colony_size // 2 (floor)
      test_abc_optimizer_should_use_half_the_colony_as_food_sources_when_colony_size_is_4_to_12
  L1  limit < 1              0 (edge), -1 (interior)     ValueError
      test_abc_optimizer_should_raise_value_error_when_limit_is_below_one
  L2  limit >= 1             1 (edge), 2, 3 (interior)   valid; scouts when max(trial) reaches limit
      test_abc_optimizer_should_scout_the_first_source_with_the_most_trials_when_max_trial_reaches_limit
      (limit 1 with trial [1, 1], limit 2 with trial [2, 2] and [0, 2])
      test_abc_optimizer_should_start_the_next_cycle_without_a_scout_when_no_trial_reaches_limit (limit 3)
ABC, protocol (ask / tell)
  T1  whole initial batch told at once       phase becomes employed
      tests/test_abc.py::test_phase_order_and_evaluation_counts (existing)
  T2  initial batch told in two parts        init until the last part, then employed
      test_abc_optimizer_should_stay_in_init_until_every_initial_candidate_is_told_when_told_in_parts
  T3  candidate never asked                  ValueError ('not asked')
      test_abc_optimizer_tell_should_raise_value_error_when_candidate_was_never_asked
  T4  candidate already told                 ValueError
      tests/test_abc.py::test_protocol_errors (existing)
  T5  scores and candidates differ in length ValueError ('same length')
      test_abc_optimizer_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length
  T6  ask while candidates are pending       RuntimeError; init phase here, employed phase in
      tests/test_abc.py::test_protocol_errors (existing)
      test_abc_optimizer_ask_should_raise_runtime_error_when_initial_candidates_are_pending
ABC, score handling per direction (direction x score class, full product)
  D1 min/finite 0.5 -> 0.5   D2 min/NaN -> +inf   D3 min/+inf -> +inf   D4 min/-inf -> +inf
  D5 max/finite 0.5 -> 0.5   D6 max/NaN -> -inf   D7 max/+inf -> -inf   D8 max/-inf -> -inf
      test_abc_optimizer_should_report_the_worst_value_for_the_direction_unless_the_score_is_finite
ABC, greedy acceptance (fitness of the new candidate against the source)
  A1  strictly greater fitness   replaces the source (trial 0, its fitness is the new reference)
      test_abc_optimizer_should_compare_later_candidates_with_the_new_fitness_when_a_source_was_replaced
      test_abc_optimizer_should_reset_the_trial_counter_when_an_onlooker_is_strictly_better
  A2  equal fitness (edge)       source kept, trial + 1
  A3  lower fitness              source kept, trial + 1
      test_abc_optimizer_should_keep_the_source_and_count_a_trial_when_fitness_is_not_strictly_greater
ABC, scout
  Sc1 max(trial) < limit     trial [2, 2], limit 3 (limit - 1, edge): no scout; the cycle increments at
                             that ask and an employed bee follows
      test_abc_optimizer_should_start_the_next_cycle_without_a_scout_when_no_trial_reaches_limit
  Sc2 max(trial) >= limit    trial [2, 2], limit 2 (== limit, edge) and trial [1, 1], limit 1: exactly one
                             scout on the first index of max(trial), its trial is 0 even for a worse
                             score, the cycle increments after its tell
      test_abc_optimizer_should_scout_the_first_source_with_the_most_trials_when_max_trial_reaches_limit
                             the scout point becomes the source although its score is worse (no greedy
                             comparison): later neighbours start from it and are compared with its fitness
      test_abc_optimizer_should_take_the_scout_point_as_the_new_source_when_its_score_is_worse

Edge cases: colony_size 3/4, limit 0/1, equal fitness, ties in max(trial), and the meeting point of Sc1
and Sc2: trial [2, 2] scouts with limit 2 (max(trial) == limit) and does not scout with limit 3
(max(trial) == limit - 1). Corner case: max(trial) == limit together with a trial tie, where the edge of
limit and the tie rule of argmax meet (Sc2 tie case).

The acceptance and scout scenarios use colony_size 4 (two sources) and constant scores, so the employed
order (source 0 then 1) and the onlooker sweep are fixed by the spec: with prob 1.0 for a source
(fitness equal to the maximum, 0.9 * 1 + 0.1) the roulette draw r < 1.0 always selects it. Every
assertion is seed-independent; ANY_SEED only makes the runs reproducible. Observations use the documented
counters phase, cycle, trial and food_number only.

Covered (exhaustive claims are limited to the enumerated values): colony_size 0..12 only (0..3 invalid,
4..12 valid). A pass shows agreement with the stated spec assumption SN = colony_size // 2 (Config.py
FOOD_NUMBER = int(NUMBER_OF_POPULATION / 2) and the task spec), not with every colony size.

Not covered (unspecified): direction strings other than 'min' and 'max'; tell with a mix of valid and
invalid candidates (partial application); candidates of another optimizer whose id happens to be pending;
best_params after only non-finite scores; colony sizes above 12 and non-integer colony_size or limit.

Requirement issue: ABC.py increments the cycle once per cycle after the scout step; the library increments
it at the ask when no scout runs and at the scout's tell otherwise. The tests follow the task spec ('the
cycle counter increments when the scout step finishes'), which both timings satisfy.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: One representative per equivalence class of configuration, protocol use, score and acceptance.
"""

import math

import pytest

from metahpo import ABCOptimizer, Candidate, Float, Space

ANY_SEED = 0
TWO_SOURCE_COLONY = 4
LIMIT_NEVER_REACHED = 1000
SCOUT_LIMIT = 2
LIMIT_ONE_ABOVE_MAX_TRIAL = 3
LIMIT_EDGE = 1
INITIAL_SCORE = 1.0
BETTER_SCORE = 0.5
WORSE_SCORE = 5.0
NEVER_ASKED_ID = 999


def _unit_square():
    return Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})


def _two_source_abc_after_initial_batch(limit):
    opt = ABCOptimizer(_unit_square(), colony_size=TWO_SOURCE_COLONY, limit=limit, seed=ANY_SEED)
    batch = opt.ask()
    opt.tell(batch, [INITIAL_SCORE] * len(batch))
    return opt


def _ask_and_tell(opt, score):
    (cand,) = opt.ask()
    opt.tell([cand], [score])
    return cand


def _tell_single_steps(opt, scores):
    return [_ask_and_tell(opt, score) for score in scores]


@pytest.mark.parametrize("colony_size", [pytest.param(c, id=f"K1-colony_size_{c}") for c in range(0, 4)])
def test_abc_optimizer_should_raise_value_error_when_colony_size_is_0_to_3(colony_size):
    """Given colony_size 0..3 (exhaustive), when ABCOptimizer is built, then ValueError names colony_size.

    Oracle: task spec (colony_size >= 4, invalid colony_size -> ValueError); ABC needs SN = colony_size // 2
    >= 2 sources so that a neighbour source k != i exists. match= separates it from the limit error of the
    same constructor. Postcondition: none (construction fails).
    """
    space = _unit_square()

    with pytest.raises(ValueError, match="colony_size"):
        ABCOptimizer(space, colony_size=colony_size, seed=ANY_SEED)


@pytest.mark.parametrize("colony_size", [pytest.param(c, id=f"K2-K3-colony_size_{c}") for c in range(4, 13)])
def test_abc_optimizer_should_use_half_the_colony_as_food_sources_when_colony_size_is_4_to_12(colony_size):
    """Given colony_size 4..12 (exhaustive), when it is asked once, then SN = colony_size // 2.

    Oracle: Config.py FOOD_NUMBER = int(NUMBER_OF_POPULATION / 2) and the task spec (init asks
    colony_size // 2 sources as one batch). Odd sizes check the floor. The claim is limited to 4..12.
    Postcondition: food_number and the size of the first batch.
    """
    sn = colony_size // 2
    opt = ABCOptimizer(_unit_square(), colony_size=colony_size, seed=ANY_SEED)

    batch = opt.ask()

    assert (opt.food_number, len(batch)) == (sn, sn)


@pytest.mark.parametrize(
    "limit",
    [pytest.param(0, id="L1-limit-zero-edge"), pytest.param(-1, id="L1-limit-negative-interior")],
)
def test_abc_optimizer_should_raise_value_error_when_limit_is_below_one(limit):
    """Given limit 0 or -1, when ABCOptimizer is built, then ValueError names limit.

    Oracle: task spec (limit >= 1, invalid limit -> ValueError). Postcondition: none (construction fails).
    """
    space = _unit_square()

    with pytest.raises(ValueError, match="limit"):
        ABCOptimizer(space, limit=limit, seed=ANY_SEED)


def test_abc_optimizer_should_stay_in_init_until_every_initial_candidate_is_told_when_told_in_parts():
    """Given the 5 initial candidates, when 2 and then 3 are told, then phase is init and then employed.

    Oracle: task spec (the initial batch may be told partially) and the abc.py docstring (employed bees
    start after the initial population). Postcondition: phase after each part.
    """
    opt = ABCOptimizer(_unit_square(), seed=ANY_SEED)
    batch = opt.ask()

    opt.tell(batch[:2], [INITIAL_SCORE] * 2)
    phase_after_first_part = opt.phase
    opt.tell(batch[2:], [INITIAL_SCORE] * 3)

    assert (phase_after_first_part, opt.phase) == ("init", "employed")


def test_abc_optimizer_tell_should_raise_value_error_when_candidate_was_never_asked():
    """Given a pending initial batch, when a candidate id never handed out is told, then ValueError.

    Oracle: task spec (telling an unknown candidate -> ValueError). match= separates it from the length
    error of tell. Postcondition: none asserted (partial application is unspecified).
    """
    opt = ABCOptimizer(_unit_square(), seed=ANY_SEED)
    batch = opt.ask()
    stranger = Candidate(NEVER_ASKED_ID, batch[0].x, batch[0].params, batch[0].key)

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([stranger], [INITIAL_SCORE])


def test_abc_optimizer_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length():
    """Given one pending employed candidate, when it is told two scores, then ValueError names the length.

    Oracle: task spec (mismatched lengths -> ValueError). Postcondition: none asserted (unspecified).
    """
    opt = _two_source_abc_after_initial_batch(LIMIT_NEVER_REACHED)
    (cand,) = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell([cand], [INITIAL_SCORE, INITIAL_SCORE])


def test_abc_optimizer_ask_should_raise_runtime_error_when_initial_candidates_are_pending():
    """Given the whole initial batch pending, when ask is called again, then RuntimeError is raised.

    Oracle: task spec (asking again with a pending candidate -> RuntimeError; the initial population is one
    batch). The employed-phase case is in tests/test_abc.py. Postcondition: none (protocol error).
    """
    opt = ABCOptimizer(_unit_square(), seed=ANY_SEED)
    opt.ask()

    with pytest.raises(RuntimeError):
        opt.ask()


@pytest.mark.parametrize(
    ("direction", "score", "expected"),
    [
        pytest.param("min", 0.5, 0.5, id="D1-min-finite"),
        pytest.param("min", math.nan, math.inf, id="D2-min-nan"),
        pytest.param("min", math.inf, math.inf, id="D3-min-plus-inf"),
        pytest.param("min", -math.inf, math.inf, id="D4-min-minus-inf-discriminating"),
        pytest.param("max", 0.5, 0.5, id="D5-max-finite"),
        pytest.param("max", math.nan, -math.inf, id="D6-max-nan"),
        pytest.param("max", math.inf, -math.inf, id="D7-max-plus-inf-discriminating"),
        pytest.param("max", -math.inf, -math.inf, id="D8-max-minus-inf"),
    ],
)
def test_abc_optimizer_should_report_the_worst_value_for_the_direction_unless_the_score_is_finite(
    direction, score, expected
):
    """Given a direction and one candidate told a score class, when best_score is read, then it maps it.

    Oracle: README Behaviour (NaN or infinity counts as the worst possible value for the direction) and the
    Optimizer docstring (+inf when minimising, -inf when maximising; scores in the user's direction).
    -inf under 'min' and +inf under 'max' are the discriminating inputs: taken literally they would be the
    best scores. Telling one candidate of five also exercises a partial initial tell.
    Postcondition: best_score of the optimizer.
    """
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), direction=direction, seed=ANY_SEED)
    batch = opt.ask()

    opt.tell(batch[:1], [score])

    assert opt.best_score == expected


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(INITIAL_SCORE, id="A2-equal-fitness-edge"),
        pytest.param(2.0, id="A3-lower-fitness"),
    ],
)
def test_abc_optimizer_should_keep_the_source_and_count_a_trial_when_fitness_is_not_strictly_greater(score):
    """Given two sources scored 1.0, when employed bee 0 gets an equal or worse score, then trial is [1, 0].

    Oracle: ABC.py send_employed_bees ('fitness_sol > self.fitness[i]' replaces, else trial + 1) with
    fitness 1 / (1 + f): equal score 1.0 gives 0.5 = 0.5, score 2.0 gives 1/3 < 0.5. The equal case is the
    edge of 'strictly greater' and guards against '>=' (site: greedy comparison in tell).
    Postcondition: trial counters.
    """
    opt = _two_source_abc_after_initial_batch(LIMIT_NEVER_REACHED)

    _ask_and_tell(opt, score)

    assert opt.trial == [1, 0]


def test_abc_optimizer_should_reset_the_trial_counter_when_an_onlooker_is_strictly_better():
    """Given trial [1, 1] after two failed employed bees, when onlooker 0 scores better, then trial is [0, 1].

    Oracle: ABC.py send_onlooker_bees (better fitness: trial = 0) and calculate_probabilities: equal
    fitness 0.5 gives prob [1.0, 1.0], so the sweep starts at source 0 and r < 1.0 always selects it. The
    trial of source 0 is 1 before, so a reset is visible, unlike for the first employed bee (trial already 0).
    Postcondition: trial counters.
    """
    opt = _two_source_abc_after_initial_batch(LIMIT_NEVER_REACHED)
    _tell_single_steps(opt, [WORSE_SCORE, WORSE_SCORE])
    assert (opt.phase, opt.trial) == ("onlooker", [1, 1])

    _ask_and_tell(opt, BETTER_SCORE)

    assert opt.trial == [0, 1]


def test_abc_optimizer_should_compare_later_candidates_with_the_new_fitness_when_a_source_was_replaced():
    """Given source 0 replaced by score 0.5, when an onlooker on it gets 0.5 again, then it is a trial.

    Oracle: ABC.py send_employed_bees (replacement stores fitness_sol as self.fitness[i]) and
    send_onlooker_bees (strictly greater needed). After the replacement source 0 has fitness 2/3, the
    maximum, so prob[0] = 1.0 and the first onlooker works on source 0; score 0.5 again is equal, so trial
    goes to 1. Had the source kept the old fitness 0.5, the onlooker would have been accepted (trial 0).
    Postcondition: trial counters after the replacement and after the onlooker.
    """
    opt = _two_source_abc_after_initial_batch(LIMIT_NEVER_REACHED)

    _ask_and_tell(opt, BETTER_SCORE)
    trial_after_replacement = list(opt.trial)
    _ask_and_tell(opt, WORSE_SCORE)
    phase_before_probe = opt.phase
    _ask_and_tell(opt, BETTER_SCORE)

    assert (trial_after_replacement, phase_before_probe, opt.trial) == ([0, 0], "onlooker", [1, 1])


def test_abc_optimizer_should_start_the_next_cycle_without_a_scout_when_no_trial_reaches_limit():
    """Given trial [2, 2] and limit 3 after the onlookers, when ask runs, then cycle 1 starts at once.

    Oracle: ABC.py send_scout_bees (no scout while amax(trial) < LIMIT), increase_cycle, and the task spec
    (the cycle counter increments when the scout step finishes, here at once). max(trial) == limit - 1 is
    the last value of Sc1; the same trial [2, 2] scouts with limit 2 in the Sc2 test, so the pair pins the
    comparison to '>=' (site: scout check in ask, guards against '>' and against 'limit - 1'). The following
    tell of a worse score raises trial[0], which shows that the asked candidate was the employed bee of
    source 0. Postcondition: cycle, phase and trial.
    """
    opt = _two_source_abc_after_initial_batch(LIMIT_ONE_ABOVE_MAX_TRIAL)
    _tell_single_steps(opt, [WORSE_SCORE] * 4)
    assert (opt.phase, opt.cycle, opt.trial) == ("scout", 0, [2, 2])

    (cand,) = opt.ask()
    cycle_and_phase_after_ask = (opt.cycle, opt.phase)
    opt.tell([cand], [WORSE_SCORE])

    assert (cycle_and_phase_after_ask, opt.trial) == ((1, "employed"), [3, 2])


@pytest.mark.parametrize(
    ("limit", "cycle_scores", "expected_trial"),
    [
        pytest.param(
            SCOUT_LIMIT,
            [WORSE_SCORE] * 4,
            [0, 2],
            id="Sc2-tie-2-2-equal-to-limit-first-argmax-is-source-0-corner",
        ),
        pytest.param(
            SCOUT_LIMIT,
            [WORSE_SCORE, WORSE_SCORE, BETTER_SCORE, WORSE_SCORE],
            [0, 0],
            id="Sc2-trial-0-2-argmax-is-source-1",
        ),
        pytest.param(
            LIMIT_EDGE,
            [BETTER_SCORE, BETTER_SCORE, WORSE_SCORE, WORSE_SCORE],
            [0, 1],
            id="L2-limit-1-edge-trial-1-1",
        ),
    ],
)
def test_abc_optimizer_should_scout_the_first_source_with_the_most_trials_when_max_trial_reaches_limit(
    limit, cycle_scores, expected_trial
):
    """Given max(trial) == limit after the onlookers, when the scout gets a worse score, then its trial is 0.

    Oracle: ABC.py send_scout_bees (amax(trial) >= LIMIT re-initialises trial.argmax(), the first maximum)
    and init(index) (trial = 0); abc.py docstring (at most one scout per cycle); task spec (cycle increments
    when the scout step finishes). Trial [2, 2] with limit 2 checks the first-index tie rule at
    max(trial) == limit; trial [0, 2] (onlooker on source 0 accepted) distinguishes 'first argmax' from
    'always source 0'; limit 1 is the edge of L2: two better employed scores give equal fitness 2/3 and
    prob [1.0, 1.0], then two worse onlookers give trial [1, 1]. Trial 0 after the worse score 5.0 shows
    the scout is not a greedy trial (that would give trial + 1). After the scout the phase is employed
    although the other source may still have trial >= limit, so no second scout runs in the cycle. That the
    scout point becomes the source is checked in the next test. Postcondition: cycle after the ask and
    after the tell, phase and trial.
    """
    opt = _two_source_abc_after_initial_batch(limit)
    _tell_single_steps(opt, cycle_scores)
    assert (opt.phase, max(opt.trial)) == ("scout", limit)

    (cand,) = opt.ask()
    cycle_after_ask = opt.cycle
    opt.tell([cand], [WORSE_SCORE])

    assert (cycle_after_ask, opt.cycle, opt.phase, opt.trial) == (0, 1, "employed", expected_trial)


def test_abc_optimizer_should_take_the_scout_point_as_the_new_source_when_its_score_is_worse():
    """Given a scout on source 0 told the worse score 5.0, when the next employed bee gets 1.0, then it wins.

    Oracle: ABC.py send_scout_bees / init(index) (the scout point replaces the source unconditionally and
    its fitness becomes the reference) and send_employed_bees (neighbour of the source in one coordinate,
    'fitness_sol > self.fitness[i]' replaces). The scout point has fitness 1/(1 + 5) = 1/6; the probe score
    1.0 has fitness 0.5 > 1/6, so it is accepted and trial[0] stays 0. Had the old source (score 1.0,
    fitness 0.5) been kept, the equal fitness would give trial 1 or more: this is the decisive input (site:
    scout branch of tell, guards against resetting trial without replacing the source). The probe also
    keeps every coordinate of the scout point except the moved one, so with two dimensions at least one
    coordinate equals the scout's (guards against keeping the old position).
    Postcondition: trial counters and the probe's position relative to the scout point.
    """
    opt = _two_source_abc_after_initial_batch(SCOUT_LIMIT)
    _tell_single_steps(opt, [WORSE_SCORE] * 4)
    scout = _ask_and_tell(opt, WORSE_SCORE)
    assert (opt.phase, opt.trial) == ("employed", [0, 2])

    (probe,) = opt.ask()
    opt.tell([probe], [INITIAL_SCORE])
    coordinates_kept_from_scout = int(sum(probe.x == scout.x))

    assert opt.trial == [0, 2]
    assert coordinates_kept_from_scout >= 1
