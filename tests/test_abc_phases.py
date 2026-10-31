"""Phase schedule of ABCOptimizer checked against the original hyp_optimizer implementation.

Oracles: the original ABC.py (initial, send_employed_bees, calculate_probabilities, send_onlooker_bees,
send_scout_bees, increase_cycle), the loop order of ABCAlgorithm.py (employed -> calculate_probabilities ->
onlooker -> memorize_best_source -> scout -> increase_cycle), Config.py
(FOOD_NUMBER = NumberOfPopulation / 2), Karaboga & Basturk (2007), the abc.py module docstring and the task
specification of the ABC phases.

How sources are observed: only public data is read. Every candidate is matched to the food source it was
built from by comparing its x with the x of the tracked sources: a neighbour changes one coordinate of its
source, so it differs from it in at most one coordinate and, in three continuous dimensions, from every other
source in more. The tracked sources start as the x of the initial batch and are replaced only where the
scenario makes the replacement certain by the spec: a strictly better score under the greedy rule of ABC.py
('if fitness_sol > self.fitness[i]'), or a scout, which ABC.py init() installs without comparison. If the
library replaced a different source, or none, the following candidates match no tracked source and the
source sequences asserted below contain None. opt.phase and opt.cycle are read right after each ask and
opt.cycle again after each tell; foods, fit, prob and pending candidates are never read.

Seeds: ANY_SEED is used where the asserted property holds for every seed. DISCRIMINATING_SEED is used where
the property also holds for every seed but the seed decides whether a wrong implementation shows itself; the
docstring of that test says why. Probabilities quoted in docstrings follow from the onlooker probabilities
p = 0.9 * fit / max(fit) + 0.1 of ABC.py calculate_probabilities.

Unspecified, therefore not asserted:
- phase and cycle read between the last onlooker tell and the next ask: ABC.py increments the cycle at the
  end of its loop, while the library decides on the scout inside the next ask() and shows phase 'scout' with
  the old cycle until then. Every phase and cycle assertion here reads them right after an ask (or after a
  scout tell), where both readings agree;
- what happens to the cycle if a scout candidate is never told;
- exact candidate values: ABC.py rounds the neighbour value and draws k differently, so only the schedule and
  the selection rules are taken from it.

Owned elsewhere: colony sizes 0 and 3 and the initial batches of colony_size 4 and 10 (error-path and API
contract tests), trial-counter values (abc-acceptance-mutation-guided), move() against rows()
(abc-protocol-invariants) and cache hits (facade-integration).

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: Phase schedules of small colonies, compared step by step with the rules of the original ABC.py.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from metahpo import ABCOptimizer, Float, Space

ANY_SEED = 0
DISCRIMINATING_SEED = 0
LIMIT_NEVER_REACHED = 1000
LOWEST_LIMIT = 1
CONSTANT_SCORE = 1.0
IMPROVING_SCORE = 0.0
WORSE_THAN_EVERY_SOURCE = 1e9
WORST_SCORE = math.inf
THREE_CYCLES = 3
SN_CASES = [
    pytest.param(4, id="colony-4-sn-2-minimum"),
    pytest.param(6, id="colony-6-sn-3"),
    pytest.param(10, id="colony-10-sn-5-default"),
]


@dataclass(frozen=True)
class _Step:
    """One ask in the employed, onlooker or scout phase and its tell, as seen through the public API."""

    batch_size: int
    phase: str
    cycle: int
    cycle_after_tell: int
    source: int | None
    x: np.ndarray
    source_x: np.ndarray | None


def _source_of(x, sources):
    """Index of the only tracked source that x differs from in at most one coordinate, else None."""
    matches = [i for i, s in enumerate(sources) if np.count_nonzero(x != s) <= 1]
    return matches[0] if len(matches) == 1 else None


def _drive(opt, initial_scores, rule, n_asks):
    """Tells the initial batch `initial_scores`, then asks and tells `n_asks` times with scores from `rule`.

    rule(phase, source, cycle) returns (score, replaced): replaced is the source index that the spec says this
    tell replaces, or None. Returns one _Step per ask.
    """
    initial = opt.ask()
    opt.tell(initial, list(initial_scores))
    sources = [c.x.copy() for c in initial]
    steps = []
    for _ in range(n_asks):
        batch = opt.ask()
        cand = batch[0]
        phase, cycle = opt.phase, opt.cycle
        source = _source_of(cand.x, sources)
        source_x = None if source is None else sources[source].copy()
        score, replaced = rule(phase, source, cycle)
        opt.tell(batch, [score] * len(batch))
        steps.append(_Step(len(batch), phase, cycle, opt.cycle, source, cand.x.copy(), source_x))
        if replaced is not None:
            sources[replaced] = cand.x.copy()
    return steps


def _phases(steps):
    return [s.phase for s in steps]


def _sources(steps):
    return [s.source for s in steps]


def _batch_sizes(steps):
    return [s.batch_size for s in steps]


def _asks_per_cycle(steps):
    """How many asks read each cycle value right after the ask."""
    return dict(Counter(s.cycle for s in steps))


def _sources_per_cycle(steps, phase):
    """Source indices of the steps of one phase, one list per cycle (cycle read after the ask)."""
    cycles = sorted({s.cycle for s in steps})
    return [[s.source for s in steps if s.phase == phase and s.cycle == c] for c in cycles]


def _changed_coordinates(steps, phase):
    """For each step of `phase`, the number of coordinates in which it differs from its matched source."""
    return [int(np.count_nonzero(s.x != s.source_x)) for s in steps if s.phase == phase]


def _first_of_each(sequences):
    return [seq[0] for seq in sequences]


def _all_sources(sequences):
    return [src for seq in sequences for src in seq]


def _has_consecutive_zero_visits(sequences):
    """True if some sequence visits source 0 twice in a row."""
    return any(a == b == 0 for seq in sequences for a, b in zip(seq[:-1], seq[1:], strict=True))


def _same_phase_onlooker_pairs_on_source_zero(steps):
    """(first, second) onlooker steps of each two-onlooker cycle whose onlookers both came from source 0."""
    by_cycle = Counter(s.cycle for s in steps if s.phase == "onlooker")
    pairs = [[s for s in steps if s.phase == "onlooker" and s.cycle == c] for c in sorted(by_cycle)]
    return [(a, b) for a, b in pairs if a.source == 0 and b.source == 0]


def _second_within_one_coordinate_of_first(pairs):
    return [int(np.count_nonzero(b.x != a.x)) <= 1 for a, b in pairs]


def _largest_change_of_second_from_phase_start(pairs):
    """Largest number of coordinates in which a second onlooker differs from source 0 before the phase."""
    return max(int(np.count_nonzero(b.x != a.source_x)) for a, b in pairs)


def _scout_cycles(steps):
    """(cycle after the scout ask, cycle after its tell) for every scout step."""
    return [(s.cycle, s.cycle_after_tell) for s in steps if s.phase == "scout"]


def _scout_sources(steps):
    return [s.source for s in steps if s.phase == "scout"]


def _constant_rule(score, scout_score=CONSTANT_SCORE, scout_targets=()):
    """Every bee gets `score`, so no employed or onlooker tell replaces a source; the scout of cycle c gets
    `scout_score` and replaces scout_targets[c]."""

    def rule(phase, source, cycle):
        if phase == "scout":
            return scout_score, scout_targets[cycle]
        return score, None

    return rule


def _dominant_source_rule():
    """Onlookers on source 0 get -1, -2, -3, ... (each strictly better: fitness 1 + |f| grows) and replace it;
    every other candidate gets an infinite score, which never replaces (fitness 0 is not > any fitness)."""
    told = []

    def rule(phase, source, cycle):
        if phase == "onlooker" and source == 0:
            told.append(source)
            return -float(len(told)), 0
        return WORST_SCORE, None

    return rule


def _first_onlooker_improves_rule():
    """The first onlooker gets IMPROVING_SCORE (fitness 1.0 > 0.5) and replaces its source; all other
    candidates get CONSTANT_SCORE, equal to the initial scores, which never replaces."""
    improved = []

    def rule(phase, source, cycle):
        if phase == "onlooker" and not improved:
            improved.append(source)
            return IMPROVING_SCORE, source
        return CONSTANT_SCORE, None

    return rule


def _source_zero_improves_rule(scout_score, scout_target):
    """Candidates of source 0 get -1, -2, ... and replace it (its trial stays 0); other bees get
    CONSTANT_SCORE (equal, so their trial grows); the scout gets `scout_score` and replaces `scout_target`."""
    told = []

    def rule(phase, source, cycle):
        if phase == "scout":
            return scout_score, scout_target
        if source == 0:
            told.append(source)
            return -float(len(told)), 0
        return CONSTANT_SCORE, None

    return rule


@pytest.mark.parametrize(
    ("colony_size", "expected_batch"),
    [
        pytest.param(5, 2, id="odd-5-next-to-minimum-drops-one-bee"),
        pytest.param(6, 3, id="even-6"),
        pytest.param(7, 3, id="odd-7-drops-one-bee"),
    ],
)
def test_abc_optimizer_should_ask_half_the_colony_as_initial_batch_when_colony_size_is_odd_or_even(
    colony_size, expected_batch
):
    """Given an ABCOptimizer of colony_size, when ask() is called first, then colony_size // 2 come back.

    Oracle: Config.py FOOD_NUMBER = int(NumberOfPopulation / 2) and ABC.py initial(), which evaluates every
    food source once; the abc.py docstring says only the initial population is handed out as a batch. The odd
    sizes discriminate floor division from rounding up (5 -> 2, not 3; 7 -> 3, not 4). Postcondition: the
    optimizer stays in phase 'init' of cycle 0 while the batch is pending.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, seed=ANY_SEED)

    batch = opt.ask()

    assert len(batch) == expected_batch
    assert opt.phase == "init"
    assert opt.cycle == 0


def test_abc_optimizer_should_raise_value_error_when_colony_size_gives_a_single_food_source():
    """Given colony_size 2 (one food source), when ABCOptimizer is built, then ValueError is raised.

    Oracle: task specification (invalid colony_size raises ValueError, colony_size >= 4) and Karaboga &
    Basturk (2007), whose neighbour needs a second source k != i. This is the infeasible one-source run of
    the sources-per-phase loop; the zero-source run (colony_size 0) and the boundary 3 are in the error-path
    tests.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})

    with pytest.raises(ValueError, match="colony_size"):
        ABCOptimizer(space, colony_size=2, seed=ANY_SEED)


@pytest.mark.parametrize("colony_size", SN_CASES)
def test_abc_optimizer_should_run_sn_employed_then_sn_onlooker_asks_each_cycle_when_no_scout_fires(
    colony_size,
):
    """Given limit 1000 (never reached), when 3 cycles run, then each is SN employed then SN onlooker asks.

    Oracle: ABCAlgorithm.py loop order employed -> calculate_probabilities -> onlooker -> scout ->
    increase_cycle, with 'i < FOOD_NUMBER' in send_employed_bees and 't < FOOD_NUMBER' in send_onlooker_bees.
    SN = 2, 3 and 5 run the sources-per-phase loop at its minimum and with many sources. The phase is read
    right after each ask; the reading before an ask at a cycle boundary is unspecified (module docstring).
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    sn = colony_size // 2
    expected = (["employed"] * sn + ["onlooker"] * sn) * THREE_CYCLES

    steps = _drive(opt, [CONSTANT_SCORE] * sn, _constant_rule(CONSTANT_SCORE), THREE_CYCLES * 2 * sn)

    assert _phases(steps) == expected


@pytest.mark.parametrize("colony_size", SN_CASES)
def test_abc_optimizer_should_count_one_cycle_per_2sn_single_asks_when_the_scout_is_skipped(colony_size):
    """Given no scout, when 3 * 2SN + 1 asks run, then cycles 0-2 get 2SN asks each and the last opens 3.

    Oracle: ABC.py send_scout_bees does nothing while max(trial) < LIMIT, but ABCAlgorithm.py still calls
    increase_cycle in every loop. KPIs as exact integers: evaluations per cycle = 2SN, candidates per ask = 1.
    The cycle counted for an ask is the value read right after it. Gray-box risk: information source - abc.py
    ask() ends a cycle inside its 'scout' branch when no source is exhausted; risk - the counter is off by one
    at a boundary, or a skipped scout does not count the cycle; targeted test - read the counter after every
    ask across three boundaries. This runs the cycle loop 0 times (cycle 0), once and several times.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    sn = colony_size // 2
    n_asks = THREE_CYCLES * 2 * sn + 1

    steps = _drive(opt, [CONSTANT_SCORE] * sn, _constant_rule(CONSTANT_SCORE), n_asks)

    assert _asks_per_cycle(steps) == {0: 2 * sn, 1: 2 * sn, 2: 2 * sn, 3: 1}
    assert _batch_sizes(steps) == [1] * n_asks


@pytest.mark.parametrize("colony_size", SN_CASES)
def test_abc_optimizer_should_send_employed_bees_to_sources_in_index_order_when_every_score_is_equal(
    colony_size,
):
    """Given equal scores, when 3 cycles run, then the n-th employed bee of every cycle comes from source n.

    Oracle: ABC.py send_employed_bees, i = 0 .. FOOD_NUMBER - 1. Equal scores are the decisive input: the
    greedy rule replaces only on strictly greater fitness, so no source moves and the initial x identifies
    every candidate; if an equal score replaced a source, later candidates of that source would match no
    tracked source. Each employed candidate differs from its source in exactly one coordinate (Karaboga &
    Basturk 2007: one param2change; a zero change would need phi == 0 or x_kj == x_ij).
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    sn = colony_size // 2

    steps = _drive(opt, [CONSTANT_SCORE] * sn, _constant_rule(CONSTANT_SCORE), THREE_CYCLES * 2 * sn)

    assert _sources_per_cycle(steps, "employed") == [list(range(sn))] * THREE_CYCLES
    assert _changed_coordinates(steps, "employed") == [1] * (THREE_CYCLES * sn)


@pytest.mark.parametrize("colony_size", SN_CASES)
def test_abc_optimizer_should_start_onlookers_at_source_zero_and_sweep_in_order_when_fitness_is_equal(
    colony_size,
):
    """Given equal scores, when 3 cycles run, then each cycle's onlookers visit sources 0, 1, ..., SN - 1.

    Oracle: ABC.py send_onlooker_bees starts with i = 0 in every phase, takes source i when r < prob[i] and
    moves on with i = (i + 1) % FOOD_NUMBER; calculate_probabilities gives 0.9 * 1 + 0.1 = 1.0 for equal
    fitness, and r in [0, 1) is always below 1.0, so the skip loop runs 0 times whatever the seed.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    sn = colony_size // 2

    steps = _drive(opt, [CONSTANT_SCORE] * sn, _constant_rule(CONSTANT_SCORE), THREE_CYCLES * 2 * sn)

    assert _sources_per_cycle(steps, "onlooker") == [list(range(sn))] * THREE_CYCLES


def test_abc_optimizer_should_keep_phase_start_onlooker_probabilities_when_an_onlooker_improves_a_source():
    """Given equal fitness, when the first onlooker improves source 0, then the next ones visit 1-4 in order.

    Oracle: ABCAlgorithm.py calls calculate_probabilities once, between send_employed_bees and
    send_onlooker_bees, and send_onlooker_bees only reads prob, so p stays 1.0 for every source during the
    phase. colony_size 10 gives five sources. DISCRIMINATING_SEED was chosen so that an implementation which
    recomputed p after the improvement (p = 0.55 for the other sources) skips a source in this phase; with the
    fixed p the expected order holds for every seed.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=10, limit=LIMIT_NEVER_REACHED, seed=DISCRIMINATING_SEED)
    sn = 10 // 2

    steps = _drive(opt, [CONSTANT_SCORE] * sn, _first_onlooker_improves_rule(), 2 * sn)

    assert _sources_per_cycle(steps, "onlooker") == [[0, 1, 2, 3, 4]]


def test_abc_optimizer_should_build_the_next_onlooker_from_the_improved_x_when_an_onlooker_improves_it():
    """Given source 0 improved by every onlooker, when 20 cycles run, then later bees start from the new x.

    Gray-box risk: information source - abc.py tell() replaces the source in place during the onlooker
    phase while p stays fixed; risk - the next onlooker of the same source is built from the stale x; targeted
    test - two onlookers on the same source in one phase. Oracle: ABC.py send_onlooker_bees writes
    self.foods[i] inside the loop, so the next calculate_neighbour_solution(i) reads the new x (Karaboga &
    Basturk 2007 greedy selection). Setup: colony_size 4, initial scores [0.0, inf], every employed candidate
    inf, so p = [1.0, 0.1] in every cycle; onlookers on source 0 get -1, -2, ... and always replace it, and a
    cycle visits [0, 0] with probability 0.9. Then: in every [0, 0] cycle the second onlooker differs from the
    first one's x in at most one coordinate ('<= 1' because clipping at 0 or 1 can give a zero change), and
    the next cycle's employed bees match the tracked, improved sources. The last assertion is a precondition
    guard: some [0, 0] cycle changed two different coordinates, the situation in which a stale x becomes
    visible; over 20 cycles it fails only with negligible probability for any seed.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    cycles = 20

    steps = _drive(opt, [IMPROVING_SCORE, WORST_SCORE], _dominant_source_rule(), cycles * 4)
    pairs = _same_phase_onlooker_pairs_on_source_zero(steps)

    assert _second_within_one_coordinate_of_first(pairs) == [True] * len(pairs)
    assert _sources_per_cycle(steps, "employed") == [[0, 1]] * cycles
    assert _largest_change_of_second_from_phase_start(pairs) == 2


@pytest.mark.parametrize(
    "colony_size",
    [
        pytest.param(4, id="sn-2-revisit-skips-one-source"),
        pytest.param(10, id="sn-5-revisit-skips-four-sources"),
    ],
)
def test_abc_optimizer_should_skip_the_other_sources_when_one_source_dominates_the_onlooker_probabilities(
    colony_size,
):
    """Given p = 1.0 for source 0, 0.1 elsewhere, when 10 cycles run, then onlookers start at 0, revisit it.

    Oracle: ABC.py send_onlooker_bees (i = 0 at phase start, take i when r < prob[i], i = (i + 1) %
    FOOD_NUMBER) and calculate_probabilities (fitness 0 of an infinite score gives 0.1, the best gives 1.0).
    Two consecutive visits of source 0 mean that the skip loop passed over all SN - 1 other sources: once for
    SN = 2, four times for SN = 5; the zero-skip run is the equal-fitness sweep test. The first visit is
    source 0 for every seed (r < 1.0) and every onlooker matches a tracked source. A consecutive revisit has
    probability 0.9 (SN = 2) or 0.9 ** 4 (SN = 5) per cycle, so over 10 cycles the last assertion is a
    precondition guard that holds for any seed with overwhelming probability.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LIMIT_NEVER_REACHED, seed=ANY_SEED)
    sn = colony_size // 2
    cycles = 10
    initial_scores = [IMPROVING_SCORE] + [WORST_SCORE] * (sn - 1)

    steps = _drive(opt, initial_scores, _dominant_source_rule(), cycles * 2 * sn)
    onlooker_sources = _sources_per_cycle(steps, "onlooker")

    assert _first_of_each(onlooker_sources) == [0] * cycles
    assert None not in _all_sources(onlooker_sources)
    assert _has_consecutive_zero_visits(onlooker_sources) is True


@pytest.mark.parametrize(
    ("colony_size", "scout_targets"),
    [
        pytest.param(4, (0, 1, 0), id="sn-2"),
        pytest.param(10, (0, 1, 2), id="sn-5"),
    ],
)
def test_abc_optimizer_should_send_at_most_one_scout_per_cycle_when_every_source_is_exhausted(
    colony_size, scout_targets
):
    """Given limit 1 and equal scores (all exhausted), when 3 cycles run, then each ends with one scout.

    Oracle: ABC.py send_scout_bees is a single 'if np.amax(trial) >= LIMIT: init(argmax)', followed by
    increase_cycle. limit 1 is the lowest valid value: after one employed and one onlooker visit every trial
    is 2, so every cycle, cycle 0 included, has a scout and costs 2SN + 1 evaluations (KPI as an exact
    integer). The scout is a random point, a neighbour of no tracked source; the cycle is unchanged after the
    scout ask and advances by one when its tell finishes the scout step. scout_targets are the argmax sources
    derived by hand from the trial counts (SN = 2: [2, 2], [2, 4], [4, 2]; SN = 5: [2] * 5, [2, 4, 4, 4, 4],
    ...), needed only to keep tracking the replaced sources.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, limit=LOWEST_LIMIT, seed=ANY_SEED)
    sn = colony_size // 2
    rule = _constant_rule(CONSTANT_SCORE, scout_targets=scout_targets)
    expected_phases = (["employed"] * sn + ["onlooker"] * sn + ["scout"]) * THREE_CYCLES

    steps = _drive(opt, [CONSTANT_SCORE] * sn, rule, THREE_CYCLES * (2 * sn + 1))

    assert _phases(steps) == expected_phases
    assert _scout_cycles(steps) == [(0, 1), (1, 2), (2, 3)]
    assert _scout_sources(steps) == [None] * THREE_CYCLES


@pytest.mark.parametrize(
    ("make_rule", "replaced"),
    [
        pytest.param(
            lambda: _constant_rule(CONSTANT_SCORE, scout_score=WORSE_THAN_EVERY_SOURCE, scout_targets=(0,)),
            0,
            id="tied-trials-replace-lowest-index-0",
        ),
        pytest.param(
            lambda: _source_zero_improves_rule(WORSE_THAN_EVERY_SOURCE, 1),
            1,
            id="source-1-has-more-trials",
        ),
    ],
)
def test_abc_optimizer_should_replace_the_most_exhausted_source_with_the_scout_even_when_it_scores_worse(
    make_rule, replaced
):
    """Given limit 1 and a scout told 1e9 (worse than all), when cycle 1 starts, then its bees use the scout.

    Oracle: ABC.py send_scout_bees -> init(self.trial.argmax(axis=0)); numpy argmax returns the first maximum,
    and init() overwrites the source and resets its trial with no comparison. Cases: equal scores tie every
    trial at 2, so source 0 (lowest index) is replaced; giving every candidate of source 0 a strictly better
    score keeps its trial at 0 while source 1 reaches at least 1, so source 1 is replaced. 1e9 is decisive: a
    greedy scout would keep the old source, and the next employed bee of that index would match no tracked
    source. Then: the employed bees of cycle 1 match the tracked sources in index order, and the one of the
    replaced index is a one-coordinate neighbour of the scout x. make_rule builds a fresh scoring rule per
    test, so no scoring state is shared between runs.
    """
    space = Space({"x0": Float(0.0, 1.0), "x1": Float(0.0, 1.0), "x2": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=LOWEST_LIMIT, seed=ANY_SEED)
    sn = 4 // 2
    rule = make_rule()

    steps = _drive(opt, [CONSTANT_SCORE] * sn, rule, 2 * sn + 1 + sn)
    scout = steps[2 * sn]
    next_employed = steps[2 * sn + 1 :]

    assert scout.phase == "scout"
    assert _sources(next_employed) == [0, 1]
    assert int(np.count_nonzero(next_employed[replaced].x != scout.x)) == 1
