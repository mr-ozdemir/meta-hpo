"""Coverage-driven tests of ABCOptimizer: statements, branches and the stdlib coverage measurement.

ABCOptimizer tests added after the stdlib coverage measurement: arguments, init batch, tell errors, phases.

They close the red lines and one-sided branches of abc.py found by the baseline trace and branch probe:
line 37 and edge 36L (limit below one), the init branch of ask with a partially told batch (line 57 loop,
line 58 conditions, edge 59->60L, line 60), line 77 and edge 76L (length mismatch), edge 83->84L (first
told score non-finite), and the definition-use pairs of the onlooker probabilities when every fitness is zero
and of the scout step (replace and skip).

Oracles: the task specification of ABCOptimizer (colony_size >= 4, limit >= 1, colony_size // 2 sources
asked as one batch that may be told partially, one candidate per ask afterwards, greedy acceptance on
strictly greater fitness, scout when max trial >= limit, at most one per cycle, cycle counted when the scout
step finishes, RuntimeError on ask with a pending candidate, ValueError on unknown candidates and length
mismatch), the original hyp_optimizer ABC.py (send_onlooker_bees roulette sweep, send_scout_bees with
argmax, i.e. the first maximal trial), Karaboga & Basturk 2007 (SN = colony size / 2, greedy selection),
operators docstrings (fitness 1 / (1 + f), +inf gives 0; all-zero fitness gives equal probabilities) and the
Optimizer docstring (NaN or inf is the worst value for the direction).

Observation uses the public API only: Candidate fields recorded from earlier asks, the documented counters
phase, cycle, trial and food_number, best_score, and exceptions. A neighbour differs from its source in at
most one coordinate, while two independent uniform points differ in every coordinate with probability one;
that is how a candidate is matched to its source without reading internal state.

Unspecified, therefore not asserted: telling a candidate of another optimizer whose id happens to be pending
here (ids are per-optimizer counters; the test below uses a foreign id that is not pending), and the value
of best_params when every told score is non-finite.

Unit tests of ABCOptimizer protocol statements that the repository tests do not reach (optimizers/abc.py).

Grouped by method: ABCOptimizer.__init__ (limit check), ABCOptimizer.ask (initial batch still pending),
ABCOptimizer.tell (initial phase kept until every initial source is told, length check). Each test uses at
most the initial ask and tell, no cycles.

The case 'ask returns only the untold sources that are not pending' cannot arise through the public
API, because the first ask always issues every initial source at once.

Unspecified, not asserted: whether a rejected tell leaves its candidates pending.

Branch-coverage tests of Optimizer._internal / best_score and ABCOptimizer.ask / tell / _advance.

Control flow graphs (course notation: nodes are the real line numbers of the named file, straight-line code is
merged, every decision has one T and one F edge, s and e are entry and exit, a raise goes to e; a conditional
expression on one line is split into t and f nodes; loops use the while and for templates):

Optimizer._internal, optimizers/base.py 67-71
    s   -> 69
    69  -T-> 70 -> e          score not finite: return +inf
    69  -F-> 71
    71  -T-> 71t -> e         direction == "min": return score
    71  -F-> 71f -> e         return -score

Optimizer.best_score, optimizers/base.py 63-65
    s   -> 65
    65  -T-> 65t -> e         direction == "min": best_f
    65  -F-> 65f -> e         -best_f

ABCOptimizer.ask, optimizers/abc.py 55-73
    s   -> 56
    56  -T-> 57-58 -> 59      phase == "init"; the waiting and todo comprehensions are merged
    56  -F-> 62
    59  -T-> 60 -> e          nothing left to hand out: raise RuntimeError
    59  -F-> 61 -> e          issue the initial batch
    62  -T-> 63 -> e          a candidate is pending: raise RuntimeError
    62  -F-> 64
    64  -T-> 65 -> 66         phase == "scout"; 65 picks the first most-tried source
    64  -F-> 69
    66  -T-> 67 -> e          trial[worst] >= limit: issue a scout
    66  -F-> 68 -> 69         _end_cycle (cycle += 1, phase = "employed")
    69  -T-> 70 -> e          phase == "employed": issue an employed neighbour
    69  -F-> 71
    71  -T-> 72 -> 71         while uniform >= prob[i]: i = (i + 1) % SN
    71  -F-> 73 -> e          issue an onlooker neighbour

ABCOptimizer.tell, optimizers/abc.py 75-98
    s   -> 76
    76  -T-> 77 -> e          lengths differ: raise ValueError
    76  -F-> 78
    78  -T-> 79               for: next (candidate, score) pair
    78  -F-> e                for: pairs exhausted
    79  -T-> 80 -> e          id not pending: raise ValueError
    79  -F-> 81-82 -> 83      pop the pending entry, f = _internal(score)
    83  -T-> 84 -> 85         f < best_f or best_x is None: new best
    83  -F-> 85
    85  -T-> 86-87 -> 88      kind == "init": replace, mark initialised
    85  -F-> 90
    88  -T-> 89 -> 78         all initialised: phase = "employed"
    88  -F-> 78
    90  -T-> 91-92 -> 78      kind == "scout": replace unconditionally, end the cycle
    90  -F-> 94
    94  -T-> 95 -> 98         fitness strictly greater: replace
    94  -F-> 97 -> 98         otherwise trial += 1
    98  -> 78                 _advance(kind)

ABCOptimizer._advance, optimizers/abc.py 128-140
    s   -> 129
    129 -T-> 130 -> 131       employed: employed_i += 1
    129 -F-> 137-138 -> 139   onlooker: onlooker_t += 1, onlooker_i = (onlooker_i + 1) % SN
    131 -T-> 132-135 -> e     last employed source: probabilities, phase = "onlooker"
    131 -F-> e
    139 -T-> 140 -> e         last onlooker: phase = "scout"
    139 -F-> e

Test requirements (branch coverage, one per T/F edge):
    BASE-1  69 T (BASE-1min under "min", BASE-1max under "max")    BASE-2  69 F
    BASE-3  71 T        BASE-4  71 F        BASE-5  65 T        BASE-6  65 F
    ABC-ASK-1   56 T    ABC-ASK-2   56 F    ABC-ASK-3   59 T    ABC-ASK-4   59 F
    ABC-ASK-5   62 T    ABC-ASK-6   62 F    ABC-ASK-7   64 T    ABC-ASK-8   64 F    ABC-ASK-9   66 T
    ABC-ASK-10  66 F
    ABC-ASK-11  69 T    ABC-ASK-12  69 F    ABC-ASK-13  71 T    ABC-ASK-14  71 F
    ABC-TELL-1  76 T    ABC-TELL-2  76 F    ABC-TELL-3  78 T    ABC-TELL-4  78 F    ABC-TELL-5  79 T
    ABC-TELL-6  79 F
    ABC-TELL-7  83 T, as ABC-TELL-7a (f < best_f decides) and ABC-TELL-7b (only best_x is None is true)
    ABC-TELL-8  83 F    ABC-TELL-9  85 T    ABC-TELL-10 85 F    ABC-TELL-11 88 T    ABC-TELL-12 88 F
    ABC-TELL-13 90 T    ABC-TELL-14 90 F    ABC-TELL-15 94 T    ABC-TELL-16 94 F
    ABC-ADV-1  129 T    ABC-ADV-2  129 F    ABC-ADV-3  131 T    ABC-ADV-4  131 F    ABC-ADV-5  139 T
    ABC-ADV-6  139 F

Coverage level: 42 of 42 feasible branch requirements (100 percent; 46 of 46 with the sub-items 1min/1max and
7a/7b), all by this module. Every edge is feasible.

Infeasible paths (not edges, so no requirement is lost): ask 66 F -> 68 -> 69 F, because _end_cycle always
sets phase = "employed". The ADV edges are executed by every test that drives a full employed and onlooker
phase; the existing tests/test_abc.py::test_phase_order_and_evaluation_counts asserts the phase sequence.

Unspecified, therefore not asserted: tell([], []); whether a rejected tell (wrong length, unknown id) leaves
the state unchanged; best_params before any tell; the onlooker probability vector itself (it is internal
layout; OP-13 is tested directly in test_branches_operators.py).

Observation is through ask/tell, Candidate fields recorded from earlier asks, best_params/best_score and the
read-only counters phase, cycle and trial. A constant rejected score keeps every food source equal to the
candidate it came from, so earlier Candidate.x values identify the sources. ANY_SEED is fixed for
determinism; every assertion holds for any seed except where a docstring states a probability.

Box types (P-T4-08): white-box tests are designed from the edges above with crafted inputs (the docstring
names the edge each input forces); black-box tests are designed from the README, the docstrings, the task
spec and the original hyp_optimizer ABC.py. ask, tell, _advance, _internal and best_score each have both.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: One scenario per statement or branch that the coverage measurement and the branch inventory found
unreached.
"""

import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Float, Space

ANY_SEED = 0


IRRELEVANT_SCORE = 0.0


NEVER_REACHED_LIMIT = 1000


def _unit_space(dim):
    """`dim` linear floats on [0, 1]; continuous, so independent points differ in every coordinate."""
    return Space({f"x{j}": Float(0.0, 1.0) for j in range(dim)})


def _finish_initial_batch(opt, scores):
    """Asks the initial batch, tells `scores` in order and returns the batch."""
    batch = opt.ask()
    opt.tell(batch, scores)
    return batch


def _single_rounds(opt, count, score):
    """Runs `count` single ask/tell rounds telling `score` each time; returns the asked candidates."""
    asked = []
    for _ in range(count):
        (candidate,) = opt.ask()
        opt.tell([candidate], [score])
        asked.append(candidate)
    return asked


def _changed_coordinates(x, reference):
    """Number of coordinates in which x differs from reference."""
    return int(np.count_nonzero(np.asarray(x) != np.asarray(reference)))


def _sources_of(candidates, sources):
    """For each candidate, the indices of the sources it differs from in at most one coordinate."""
    return [[i for i, s in enumerate(sources) if _changed_coordinates(c.x, s.x) <= 1] for c in candidates]


def _ids(candidates):
    """The ids of `candidates`."""
    return [c.id for c in candidates]


def test_abc_optimizer_should_raise_value_error_when_limit_is_zero():
    """Given limit=0, when ABCOptimizer is built, then ValueError is raised.

    Oracle: task specification 'limit >= 1', invalid limit -> ValueError. Input: 0 is the invalid value next
    to the smallest valid limit 1. Site: abc line 37 (edge 36L). The keyword tells it apart from the
    colony_size error of the same constructor.
    """
    space = _unit_space(2)

    with pytest.raises(ValueError, match="limit"):
        ABCOptimizer(space, limit=0, seed=ANY_SEED)


@pytest.mark.parametrize(
    "colony_size",
    [pytest.param(4, id="smallest_valid_colony"), pytest.param(5, id="odd_colony_rounds_down")],
)
def test_abc_optimizer_should_ask_two_initial_candidates_when_colony_size_is_four_or_five_and_limit_is_one(
    colony_size,
):
    """Given colony_size 4 or 5 and limit 1, when the first ask runs, then two sources are handed out.

    Oracle: task specification 'colony_size // 2 sources asked as one batch', colony_size >= 4 and
    limit >= 1; Karaboga & Basturk 2007 SN = colony size / 2. Inputs: 4 and limit 1 are the smallest valid
    values (valid side of both boundaries), 5 shows floor division for an odd colony.
    """
    opt = ABCOptimizer(_unit_space(2), colony_size=colony_size, limit=1, seed=ANY_SEED)

    batch = opt.ask()

    assert opt.food_number == 2
    assert len(batch) == 2


def test_abc_ask_should_raise_runtime_error_when_part_of_the_initial_batch_is_still_untold():
    """Given 2 of 5 initial candidates told in reverse order, when ask() runs again, then RuntimeError.

    Oracle: task specification 'init sources asked as one batch, may be told partially; asking again with a
    pending candidate -> RuntimeError'. Input: telling a reversed subset splits the batch into told and still
    pending sources. Sites: abc line 57 loop, both line-58 conditions false, line 60 (edge 59->60L).
    """
    opt = ABCOptimizer(_unit_space(3), seed=ANY_SEED)
    batch = opt.ask()
    opt.tell([batch[1], batch[0]], [1.0, 2.0])
    assert opt.phase == "init"

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_ask_should_hand_out_one_employed_candidate_when_the_rest_of_a_partially_told_batch_is_told():
    """Given 2 of 5 initial candidates told and then the other 3, when ask() runs, then one employed
    candidate.

    Oracle: task specification (init batch told partially; employed phase hands out one candidate per ask).
    Postcondition: phase 'employed' with that single candidate pending.
    """
    opt = ABCOptimizer(_unit_space(3), seed=ANY_SEED)
    batch = opt.ask()
    opt.tell([batch[1], batch[0]], [1.0, 2.0])
    opt.tell(batch[2:], [3.0, 4.0, 5.0])

    after_init = opt.ask()

    assert len(after_init) == 1
    assert opt.phase == "employed"


def test_abc_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length():
    """Given one pending candidate, when tell() gets it with no score, then ValueError is raised.

    Oracle: task specification 'mismatched lengths -> ValueError'. Input: one candidate and an empty score
    list is the smallest mismatch. Site: abc line 77 (edge 76L). The keyword tells it apart from the
    'not asked' error of the same method.
    """
    opt = ABCOptimizer(_unit_space(2), seed=ANY_SEED)
    batch = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell([batch[0]], [])


def test_abc_tell_should_raise_value_error_when_the_candidate_comes_from_another_optimizer():
    """Given a candidate asked from another optimizer, when this optimizer is told it, then ValueError.

    Oracle: task specification 'telling an unknown candidate -> ValueError'. Input: the foreign optimizer is
    advanced past its initial batch, so its candidate id is not among this optimizer's pending ids (a
    colliding id is unspecified, see the module docstring).
    """
    space = _unit_space(2)
    opt = ABCOptimizer(space, seed=ANY_SEED)
    pending = opt.ask()
    other = ABCOptimizer(space, seed=1)
    _finish_initial_batch(other, [IRRELEVANT_SCORE] * other.food_number)
    (foreign,) = other.ask()
    assert foreign.id not in _ids(pending)

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([foreign], [IRRELEVANT_SCORE])


@pytest.mark.parametrize(
    ("direction", "worst"),
    [
        pytest.param("min", math.inf, id="min_worst_is_plus_inf"),
        pytest.param("max", -math.inf, id="max_worst_is_minus_inf"),
    ],
)
def test_abc_best_score_should_be_the_worst_value_for_the_direction_when_every_initial_score_is_non_finite(
    direction, worst
):
    """Given NaN, +inf and -inf told for the whole initial batch, when best_score is read, then the worst
    value.

    Oracle: Optimizer docstring and README Behaviour: a NaN or infinite score becomes the worst value for the
    direction (+inf when minimising, -inf when maximising). Input: -inf would be 'best' for a naive min and
    +inf for a naive max. Site: abc edge 83->84L (best not yet set and the first value not better).
    """
    opt = ABCOptimizer(_unit_space(2), direction=direction, seed=ANY_SEED)
    _finish_initial_batch(opt, [math.nan, math.inf, -math.inf, math.nan, math.nan])

    best_score = opt.best_score

    assert best_score == worst


def test_abc_onlookers_should_visit_every_source_once_in_order_when_every_score_is_nan():
    """Given NaN for every initial and employed candidate, when SN onlookers are asked, then sources 0, 1, 2.

    Oracle: NaN becomes +inf internally (Optimizer docstring) whose fitness is 0 (operators docstring), so
    every employed candidate is rejected (greedy rule: strictly greater fitness) and the sources stay at the
    initial points; operators.onlooker_probabilities gives equal probabilities 1 for all-zero fitness, and the
    ABC.py roulette sweep 'if r < prob[i]' then takes every source in order. Seed-independent: a uniform draw
    is always below 1. Site: the all-zero branch of the probabilities used by the onlooker phase.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=6, limit=NEVER_REACHED_LIMIT, seed=ANY_SEED)
    sources = _finish_initial_batch(opt, [math.nan] * opt.food_number)
    _single_rounds(opt, opt.food_number, math.nan)
    assert opt.phase == "onlooker"

    onlookers = _single_rounds(opt, opt.food_number, math.nan)

    assert _sources_of(onlookers, sources) == [[0], [1], [2]]


def test_abc_scout_should_take_the_first_most_tried_source_and_end_the_cycle_at_limit():
    """Given two sources both at trial 2 = limit, when the scout candidate is asked and told, then source 0
    resets.

    Oracle: ABC.py send_scout_bees 'if np.amax(trial) >= LIMIT: self.init(trial.argmax())' (argmax is the
    first maximum) and the task specification (at most one scout per cycle; the cycle counter increments when
    the scout step finishes). Setup: initial scores 1.0, every later score 5.0 has lower fitness, so all 2
    employed and 2 onlooker candidates are rejected; equal fitness gives probabilities 1, so the onlookers
    take sources 0 then 1 for any seed. Input: the tie [2, 2] decides first-maximum tie-breaking.
    Postcondition: trial [0, 2], cycle 1, phase 'employed' although source 1 is still at the limit.
    """
    opt = ABCOptimizer(_unit_space(2), colony_size=4, limit=2, seed=ANY_SEED)
    _finish_initial_batch(opt, [1.0, 1.0])
    _single_rounds(opt, 4, 5.0)
    assert (opt.phase, opt.trial, opt.cycle) == ("scout", [2, 2], 0)

    (scout,) = opt.ask()
    opt.tell([scout], [5.0])

    assert opt.trial == [0, 2]
    assert opt.cycle == 1
    assert opt.phase == "employed"


def test_abc_scout_should_replace_the_source_even_when_the_scout_scores_worse():
    """Given a scout told 5.0 for a source that scored 1.0, when the next employed bee is asked, it moves from
    the scout.

    Oracle: ABC.py send_scout_bees calls init(i), which replaces the source without a greedy comparison, and
    the employed sweep restarts at source 0 (task specification). Input: a scout score worse than the
    source's own decides between unconditional replacement and greedy selection. Observation: the employed
    candidate differs from the scout point in at most one coordinate; a neighbour of the old source would
    differ from it in both coordinates.
    """
    opt = ABCOptimizer(_unit_space(2), colony_size=4, limit=2, seed=ANY_SEED)
    _finish_initial_batch(opt, [1.0, 1.0])
    _single_rounds(opt, 4, 5.0)
    (scout,) = opt.ask()
    opt.tell([scout], [5.0])

    (employed,) = opt.ask()

    assert _changed_coordinates(employed.x, scout.x) <= 1


def test_abc_ask_should_start_the_next_cycle_without_a_scout_when_no_trial_reaches_limit():
    """Given limit 1000 and a finished onlooker phase, when ask() runs, then an employed candidate of cycle 1.

    Oracle: ABC.py send_scout_bees does nothing when max trial < limit and increase_cycle follows; task
    specification 'cycle counter increments when the scout step finishes'. Input: limit 1000 cannot be reached
    in one cycle, so the scout step is skipped for any seed. Postcondition: one candidate, phase 'employed'
    already at ask time (a scout candidate would leave the phase at 'scout'), cycle 1.
    """
    opt = ABCOptimizer(_unit_space(3), colony_size=10, limit=NEVER_REACHED_LIMIT, seed=ANY_SEED)
    _finish_initial_batch(opt, [IRRELEVANT_SCORE] * opt.food_number)
    _single_rounds(opt, 2 * opt.food_number, IRRELEVANT_SCORE)
    assert (opt.phase, opt.cycle) == ("scout", 0)

    batch = opt.ask()

    assert len(batch) == 1
    assert opt.phase == "employed"
    assert opt.cycle == 1


COLONY_SIZE = 10


IRRELEVANT_TOLD_SCORE = 1.0


def _one_parameter_space():
    """A fresh one-parameter Space; its bounds do not matter for these tests."""
    return Space({"x": Float(0.0, 1.0)})


def test_abc_init_should_raise_value_error_when_limit_is_zero():
    """Given limit=0, when ABCOptimizer is built, then ValueError is raised.

    Oracle: task specification 'limit=15 (>=1)' and 'invalid colony_size/limit -> ValueError'. Input: 0 is
    just below the smallest valid limit, decisive against a '< 0' bound (site: ABCOptimizer.__init__ limit
    check -> raise). The keyword separates it from the colony_size error of the same constructor.
    """
    space = _one_parameter_space()

    with pytest.raises(ValueError, match="limit"):
        ABCOptimizer(space, limit=0, seed=ANY_SEED)


def test_abc_init_should_accept_limit_one_and_issue_the_initial_batch():
    """Given limit=1, when ABCOptimizer is built and asked, then it returns colony_size // 2 candidates.

    Oracle: task specification 'limit (>=1)' and 'init (colony_size//2 sources asked as one batch)'. Input:
    1 is the smallest valid limit, the valid side of the boundary above.
    """
    opt = ABCOptimizer(_one_parameter_space(), colony_size=COLONY_SIZE, limit=1, seed=ANY_SEED)

    cands = opt.ask()

    assert len(cands) == COLONY_SIZE // 2


def test_abc_ask_should_raise_runtime_error_when_initial_batch_is_partially_told():
    """Given an initial batch of 5 of which only 2 are told, when ask() is called again, then RuntimeError.

    Oracle: task specification 'init ... may be told partially' and 'asking again with a pending candidate ->
    RuntimeError'. Input: a partial tell leaves 3 initial candidates pending and none untold, so the initial
    phase has nothing new to issue (site: ABCOptimizer.ask, initial phase with no source left -> raise).
    """
    opt = ABCOptimizer(_one_parameter_space(), colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands[:2], [IRRELEVANT_TOLD_SCORE] * 2)

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_phase_should_stay_init_when_initial_batch_is_partially_told():
    """Given an initial batch of 5 of which only 2 are told, when phase is read, then it is still 'init'.

    Oracle: task specification phase order 'init (colony_size//2 sources asked as one batch, may be told
    partially)' then 'employed'; the initial phase ends only when every initial source has a score. Input: a
    partial tell of 2 of 5 is decisive against an end-of-init check that fires on the first told source
    (site: ABCOptimizer.tell 'if all(initialised)' -> phase switch). Postcondition: phase is 'init'.
    """
    opt = ABCOptimizer(_one_parameter_space(), colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands[:2], [IRRELEVANT_TOLD_SCORE] * 2)

    phase = opt.phase

    assert phase == "init"


def test_abc_ask_should_return_one_candidate_when_rest_of_partially_told_batch_is_told():
    """Given an initial batch told in two parts (2, then 3), when ask() is called, then it returns 1
    candidate.

    Oracle: task specification 'init ... may be told partially' and 'employed (one candidate per ask)'.
    Input: the split tell is the positive counterpart of the test above; the initial phase ends only when
    every source is told. Postcondition: the optimizer is in the employed phase.
    """
    opt = ABCOptimizer(_one_parameter_space(), colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell(cands[:2], [IRRELEVANT_TOLD_SCORE] * 2)
    opt.tell(cands[2:], [IRRELEVANT_TOLD_SCORE] * 3)

    next_cands = opt.ask()

    assert len(next_cands) == 1
    assert opt.phase == "employed"


def test_abc_tell_should_raise_value_error_when_one_score_is_missing():
    """Given 5 candidates and 4 scores, when tell() is called, then ValueError is raised.

    Oracle: task specification 'mismatched lengths -> ValueError'. Input: the score list is one short, the
    nearest invalid length (site: ABCOptimizer.tell length check -> raise). The keyword separates it from the
    'not asked' error of the same call.
    """
    opt = ABCOptimizer(_one_parameter_space(), colony_size=COLONY_SIZE, seed=ANY_SEED)
    cands = opt.ask()
    scores = [IRRELEVANT_TOLD_SCORE] * (len(cands) - 1)

    with pytest.raises(ValueError, match="same length"):
        opt.tell(cands, scores)


LARGE_LIMIT = 10**6


SOURCE_SCORE = 1.0


REJECTED_SCORE = 5.0


BETTER_SCORE = 0.5


MUCH_WORSE_SCORE = 99.0


TRIAL_AFTER_ONE_REJECTED_CYCLE = 2


LIMIT_EQUAL_TO_TRIAL = TRIAL_AFTER_ONE_REJECTED_CYCLE


LIMIT_ONE_ABOVE_TRIAL = TRIAL_AFTER_ONE_REJECTED_CYCLE + 1


def _unit_space_2d():
    """Two Float(0, 1) parameters: a neighbour differs from its source in at most one of two coordinates."""
    return Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})


def _told_initial_batch(opt, scores):
    """Asks the initial batch, tells it `scores` in index order and returns the candidates."""
    cands = opt.ask()
    opt.tell(cands, scores)
    return cands


def _tell_single_steps_until(opt, phase, score):
    """Asks and tells one candidate at a time with `score` until opt.phase == `phase`."""
    while opt.phase != phase:
        (cand,) = opt.ask()
        opt.tell([cand], [score])


def _coordinates_changed(a, b):
    """Number of coordinates in which the unit-cube points a and b differ."""
    return int(np.count_nonzero(a != b))


def _sources_matching(x, sources):
    """Indices of the sources from which x differs in at most one coordinate (x is a neighbour of them)."""
    return [t for t, source in enumerate(sources) if _coordinates_changed(x, source) <= 1]


def _first_onlooker_targets(opt, sources, cycles, score):
    """Runs single ask/tell steps with `score` until `cycles` cycles end.

    Returns the target source of the first onlooker of each cycle and, for every onlooker, the number of
    sources it is a neighbour of.
    """
    first_targets, match_counts = [], []
    previous_phase = opt.phase
    while opt.cycle < cycles:
        (cand,) = opt.ask()
        if opt.phase == "onlooker":
            matches = _sources_matching(cand.x, sources)
            match_counts.append(len(matches))
            if previous_phase == "employed" and matches:
                first_targets.append(matches[0])
        previous_phase = opt.phase
        opt.tell([cand], [score])
    return first_targets, match_counts


@pytest.fixture
def scout_ready():
    """colony_size=4, limit=2, initial scores [1, 1], then one cycle of rejected scores: trial == [2, 2].

    The limit equals the largest trial (the boundary of 'trial >= limit'), so the scout is sent on equality.
    Returns (opt, initial candidates, scout candidate) right after the scout ask.
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, limit=LIMIT_EQUAL_TO_TRIAL, seed=ANY_SEED)
    cands = _told_initial_batch(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _tell_single_steps_until(opt, "scout", REJECTED_SCORE)
    (scout,) = opt.ask()
    return opt, cands, scout


def test_optimizer_should_report_inf_best_and_keep_the_first_candidate_when_minimising_non_finite_scores():
    """Under 'min', NaN and +inf both become +inf, the best score is +inf, the first candidate stays best.

    Given ABCOptimizer(direction='min', colony_size=4) and its two initial candidates.
    When they are told [nan, inf] in index order: nan forces 69 T (BASE-1min) and enters the best through
    'best_x is None' only (83 T, 7b); inf < inf is false with best_x set (83 F).
    Then best_score is +inf (65 T) and best_params are those of the first candidate.
    Covers: BASE-1, BASE-5, ABC-TELL-3, ABC-TELL-4, ABC-TELL-6, ABC-TELL-7, ABC-TELL-8, ABC-TELL-9,
    ABC-TELL-11, ABC-TELL-12
    Box: white
    Source: Optimizer docstring ('+inf when minimising'); ABC.py 87-88 (global best starts at food 0) and 64
    (strict '<').
    """
    opt = ABCOptimizer(_unit_space_2d(), direction="min", colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [math.nan, math.inf])

    assert opt.best_score == math.inf
    assert opt.best_params == cands[0].params


def test_optimizer_should_report_minus_inf_best_score_when_maximising_non_finite_scores():
    """Under 'max', NaN and +inf are the worst value, so the best score reported to the user is -inf.

    Given ABCOptimizer(direction='max', colony_size=4) and its two initial candidates.
    When they are told [nan, inf]: both force 69 T under 'max' (BASE-1max), the first enters through 7b.
    Then best_score is -inf (65 F), not +inf.
    Covers: BASE-1, BASE-6, ABC-TELL-7, ABC-TELL-8
    Box: black
    Source: README Behaviour ('NaN or infinity counts as the worst possible value for the chosen direction');
    Optimizer docstring ('-inf when maximising').
    """
    opt = ABCOptimizer(_unit_space_2d(), direction="max", colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [math.nan, math.inf])

    assert opt.best_score == -math.inf


def test_optimizer_should_treat_positive_infinity_as_worst_when_maximising():
    """Under 'max' a finite score beats +inf, although +inf is numerically larger.

    Given ABCOptimizer(direction='max', colony_size=4); +inf is the discriminating input, because a plain
    negation would turn it into the best internal value.
    When the initial candidates are told [inf, 0.3]: inf takes 69 T and enters through 7b; 0.3 takes 69 F and
    71 F (negated to -0.3) and replaces it through 7a.
    Then best_score is 0.3 and best_params are those of the second candidate.
    Covers: BASE-1, BASE-2, BASE-4, BASE-6, ABC-TELL-7
    Box: black
    Source: README Behaviour (NaN or infinity is the worst value for the direction); Optimizer docstring.
    """
    opt = ABCOptimizer(_unit_space_2d(), direction="max", colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [math.inf, 0.3])

    assert opt.best_score == 0.3
    assert opt.best_params == cands[1].params


def test_optimizer_should_track_the_smallest_score_when_minimising_finite_scores():
    """Under 'min' a strictly smaller later score becomes the best.

    Given ABCOptimizer(direction='min', colony_size=4).
    When the initial candidates are told [2.0, -1.0]: both take 69 F and 71 T, and -1.0 < 2.0 takes 83 T (7a).
    Then best_score is -1.0 (65 T) and best_params are those of the second candidate.
    Covers: BASE-2, BASE-3, BASE-5, ABC-TELL-7
    Box: white
    Source: Optimizer docstring (scores in the user's direction, minimised internally); ABC.py 64.
    """
    opt = ABCOptimizer(_unit_space_2d(), direction="min", colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [2.0, -1.0])

    assert opt.best_score == -1.0
    assert opt.best_params == cands[1].params


def test_optimizer_should_keep_the_first_best_when_a_later_score_only_equals_it():
    """An equal later score does not replace the best (strict comparison).

    Given ABCOptimizer(direction='min', colony_size=4).
    When the initial candidates are told [1.0, 1.0]: the first enters through 7a (1.0 < inf), the second gives
    1.0 < 1.0 false with best_x set (83 F); a '<=' comparison would pick the second.
    Then best_params are those of the first candidate and best_score is 1.0.
    Covers: ABC-TELL-7, ABC-TELL-8
    Box: black
    Source: ABC.py 64 (strict 'self.f[i] < self.globalOpt') and 87-88.
    """
    opt = ABCOptimizer(_unit_space_2d(), direction="min", colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    opt.tell(cands, [SOURCE_SCORE, SOURCE_SCORE])

    assert opt.best_params == cands[0].params
    assert opt.best_score == SOURCE_SCORE


def test_abc_should_increment_trial_when_an_employed_score_only_equals_its_source():
    """Greedy acceptance: an equal fitness is not an improvement, so the trial counter grows.

    Given colony_size=4, limit=LARGE_LIMIT, initial scores [1, 1] and one cycle of rejected scores, so trial
    == [2, 2] and the next ask ends the cycle (66 F) and hands out the employed neighbour of source 0.
    When that neighbour is told 1.0: fitness 0.5 equals the source fitness, forcing 94 F ('>=' would accept).
    Then trial == [3, 2].
    Covers: ABC-ASK-2, ABC-ASK-6, ABC-ASK-7, ABC-ASK-8, ABC-ASK-10, ABC-ASK-11, ABC-ASK-12, ABC-ASK-14,
    ABC-TELL-14, ABC-TELL-16, ABC-ADV-1, ABC-ADV-2, ABC-ADV-3, ABC-ADV-4, ABC-ADV-5, ABC-ADV-6
    Box: black
    Source: ABC.py 113-121 (strict '>' replaces, otherwise trial + 1); task spec (greedy acceptance on
    fitness).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    _told_initial_batch(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _tell_single_steps_until(opt, "scout", SOURCE_SCORE)
    (employed,) = opt.ask()

    opt.tell([employed], [SOURCE_SCORE])

    assert opt.trial == [3, 2]


def test_abc_should_reset_trial_when_an_employed_score_is_strictly_better_than_its_source():
    """Greedy acceptance: a strictly better fitness replaces the source and resets its trial counter.

    Given the same state as the equal-score test: trial == [2, 2] and the employed neighbour of source 0.
    When it is told 0.5: fitness 1 / 1.5 > 0.5 forces 94 T, the other side of the equality boundary.
    Then trial == [0, 2]: source 0 is reset, source 1 is untouched.
    Covers: ABC-TELL-14, ABC-TELL-15, ABC-ADV-1, ABC-ADV-4
    Box: black
    Source: ABC.py 113-119 (trial = 0 on replacement); task spec (strictly greater replaces).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    _told_initial_batch(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _tell_single_steps_until(opt, "scout", SOURCE_SCORE)
    (employed,) = opt.ask()

    opt.tell([employed], [BETTER_SCORE])

    assert opt.trial == [0, 2]


def test_abc_should_hand_out_a_random_scout_when_the_largest_trial_reaches_limit(scout_ready):
    """With max trial >= limit the scout step issues a fresh random point, not a neighbour.

    Given colony_size=4, limit=2 and trial == [2, 2] after a cycle of rejected scores: trial == limit is the
    boundary of '>=' and forces 66 T on the tie (a '>' comparison would end the cycle instead).
    When the scout ask has been made (fixture scout_ready).
    Then the phase is still 'scout', the cycle is still 0, and the candidate differs from both sources in
    every coordinate (a neighbour would differ in at most one).
    Covers: ABC-ASK-2, ABC-ASK-6, ABC-ASK-7, ABC-ASK-9, ABC-ADV-1, ABC-ADV-2, ABC-ADV-3, ABC-ADV-4, ABC-ADV-5,
    ABC-ADV-6
    Box: white
    Source: ABC.py 151-153 (scout if amax(trial) >= LIMIT; init draws every coordinate at random, 70-74); task
    spec (cycle counter increments when the scout step finishes).
    """
    opt, cands, scout = scout_ready

    changed = [_coordinates_changed(scout.x, cands[0].x), _coordinates_changed(scout.x, cands[1].x)]

    assert opt.phase == "scout"
    assert opt.cycle == 0
    assert changed == [2, 2]


def test_abc_should_replace_the_first_most_tried_source_even_with_a_worse_score_when_the_scout_is_told(
    scout_ready,
):
    """The scout replaces the first most-tried source unconditionally and ends the cycle.

    Given the scout candidate of a trial tie [2, 2] with limit 2 (fixture scout_ready).
    When it is told 99.0, much worse than the source score 1.0 (85 F, 90 T: no fitness comparison).
    Then the cycle is 1, trial == [0, 2] (source 0, the first index of the tie, was replaced), and the next
    employed neighbour of source 0 differs from the scout point in at most one coordinate.
    Covers: ABC-TELL-10, ABC-TELL-13
    Box: black
    Source: ABC.py 153 (trial.argmax: first index on a tie) and 70-81 (init replaces unconditionally); ABC.py
    155-157 (cycle count); task spec.
    """
    opt, _, scout = scout_ready

    opt.tell([scout], [MUCH_WORSE_SCORE])
    cycle, trial = opt.cycle, list(opt.trial)
    (next_employed,) = opt.ask()

    assert cycle == 1
    assert trial == [0, 2]
    assert _coordinates_changed(next_employed.x, scout.x) <= 1


def test_abc_should_send_no_second_scout_in_a_cycle_when_another_trial_is_still_at_limit(scout_ready):
    """At most one scout per cycle: source 1 keeps trial 2 >= limit 2, yet the next ask is an employed bee.

    Given the scout of the trial tie [2, 2] has been told (trial == [0, 2], limit 2).
    When the next candidate is asked.
    Then the phase is 'employed' and the cycle is 1.
    Covers: ABC-ASK-2, ABC-ASK-6, ABC-ASK-8, ABC-ASK-11, ABC-TELL-13
    Box: black
    Source: abc.py module docstring and task spec ('at most one scout per cycle'); ABC.py 151-153.
    """
    opt, _, scout = scout_ready
    opt.tell([scout], [MUCH_WORSE_SCORE])

    opt.ask()

    assert opt.trial[1] == 2
    assert opt.phase == "employed"
    assert opt.cycle == 1


def test_abc_should_end_the_cycle_on_ask_and_send_an_employed_bee_when_no_trial_reaches_limit():
    """Below the limit the scout step does nothing; the cycle ends and employed bee 0 starts the next one.

    Given colony_size=4, limit=3, initial scores [1, 1] and one cycle of rejected scores, so trial == [2, 2]
    (the largest trial is limit - 1, the negative side of the scout boundary), the sources are still the
    initial candidates and phase == 'scout' with cycle 0.
    When the next candidate is asked: max trial 2 < limit 3 forces 66 F, then 68 and 69 T.
    Then the cycle is 1, the phase is 'employed', and the candidate differs from initial candidate 0 in at
    most one coordinate (it is a neighbour of source 0, not a random scout).
    Covers: ABC-ASK-7, ABC-ASK-10, ABC-ASK-11
    Box: black
    Source: task spec ('cycle counter increments when the scout step finishes'); ABC.py 152 (no scout when
    amax(trial) < LIMIT) and 155-157.
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, limit=LIMIT_ONE_ABOVE_TRIAL, seed=ANY_SEED)
    cands = _told_initial_batch(opt, [SOURCE_SCORE, SOURCE_SCORE])
    _tell_single_steps_until(opt, "scout", REJECTED_SCORE)

    (cand,) = opt.ask()

    assert opt.cycle == 1
    assert opt.phase == "employed"
    assert _coordinates_changed(cand.x, cands[0].x) <= 1


def test_abc_should_refuse_to_ask_when_the_initial_batch_is_told_only_partially():
    """While some initial candidates are still pending, ask raises RuntimeError and the phase stays 'init'.

    Given colony_size=8, so the initial batch has sn = 8 // 2 = 4 candidates.
    When only candidate 2 is told (88 F) and ask is called again: nothing is left to hand out (59 T).
    Then ask raises RuntimeError and the phase is 'init'.
    Covers: ABC-ASK-1, ABC-ASK-3, ABC-ASK-4, ABC-TELL-9, ABC-TELL-12
    Box: white
    Source: task spec (initial candidates asked as one batch, may be told partially; asking again with a
    pending candidate raises RuntimeError).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=8, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell([cands[2]], [SOURCE_SCORE])
    assert opt.phase == "init"

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_should_start_the_employed_phase_when_the_last_initial_candidate_is_told_out_of_order():
    """The initial batch may be told in parts and out of order; the last part starts the employed phase.

    Given colony_size=8 (4 initial candidates) told as [2], then [0, 3] (88 F each time).
    When the remaining candidate 1 is told (88 T) and the next candidate is asked.
    Then the phase is 'employed' and ask returns exactly one candidate.
    Covers: ABC-ASK-1, ABC-ASK-2, ABC-ASK-4, ABC-ASK-6, ABC-ASK-11, ABC-TELL-3, ABC-TELL-4, ABC-TELL-6,
    ABC-TELL-9, ABC-TELL-11, ABC-TELL-12
    Box: black
    Source: task spec (initial candidates asked as one batch, may be told partially); abc.py module docstring.
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=8, seed=ANY_SEED)
    cands = opt.ask()
    opt.tell([cands[2]], [1.0])
    opt.tell([cands[0], cands[3]], [2.0, 3.0])

    opt.tell([cands[1]], [4.0])
    next_cands = opt.ask()

    assert opt.phase == "employed"
    assert len(next_cands) == 1


def test_abc_should_refuse_to_ask_when_an_employed_candidate_is_still_pending():
    """After the initial batch, a second ask while the employed candidate is untold raises RuntimeError.

    Given colony_size=4 with its initial batch told, so the phase is 'employed', and one employed candidate
    asked but not told (that first ask takes 56 F and 62 F).
    When ask is called again: the pending candidate forces 62 T.
    Then ask raises RuntimeError.
    Covers: ABC-ASK-2, ABC-ASK-5, ABC-ASK-6
    Box: white
    Source: task spec (asking again with a pending candidate raises RuntimeError).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, seed=ANY_SEED)
    _told_initial_batch(opt, [SOURCE_SCORE, SOURCE_SCORE])
    opt.ask()
    assert opt.phase == "employed"

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_tell_should_raise_value_error_when_candidates_and_scores_differ_in_length():
    """Two candidates with one score are rejected.

    Given the two initial candidates of a colony_size=4 optimizer.
    When they are told with a single score (length 2 against 1 forces 76 T).
    Then tell raises ValueError (message keyword 'same length').
    Covers: ABC-TELL-1
    Box: black
    Source: task spec (mismatched lengths raise ValueError).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, seed=ANY_SEED)
    cands = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell(cands, [SOURCE_SCORE])


def test_abc_tell_should_raise_value_error_when_the_candidate_was_never_asked_for():
    """A candidate id the optimizer never issued is rejected.

    Given a colony_size=4 optimizer whose initial batch has ids 0 and 1, and a Candidate with id 999 carrying
    the fields of candidate 0 (76 F, then 79 T on the unknown id).
    When it is told.
    Then tell raises ValueError (message keyword 'not asked').
    Covers: ABC-TELL-2, ABC-TELL-3, ABC-TELL-5
    Box: black
    Source: task spec (telling an unknown candidate raises ValueError).
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, seed=ANY_SEED)
    cands = opt.ask()
    stranger = Candidate(999, cands[0].x, cands[0].params, cands[0].key)

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([stranger], [SOURCE_SCORE])


def test_abc_should_skip_a_low_probability_source_in_the_onlooker_sweep_when_scores_are_crafted():
    """Crafted scores make source 0 improbable, so the first onlooker of a cycle often lands on source 1.

    Given colony_size=4, limit=LARGE_LIMIT and initial scores [1e12, 0.0], so the onlooker probabilities are
    about [0.1, 1.0]; every later candidate is told 1e13, worse than both sources, so the sources and the
    probabilities never change and the initial candidates identify the sources.
    When 30 cycles are run.
    Then every onlooker is a neighbour of exactly one source, and at least one first onlooker of a cycle
    targets source 1. The sweep restarts at source 0 each cycle, so that onlooker proves the skip edge 71 T
    was taken. Per cycle this happens with probability about 0.9, so the check fails for a seed only with
    probability about 0.1 ** 30.
    Covers: ABC-ASK-8, ABC-ASK-12, ABC-ASK-13, ABC-ASK-14, ABC-ADV-2, ABC-ADV-5, ABC-ADV-6
    Box: white
    Source: ABC.py 129-149 (sweep from i = 0, select when r < prob[i], i = (i + 1) % SN); ABC.py 124-127.
    """
    opt = ABCOptimizer(_unit_space_2d(), colony_size=4, limit=LARGE_LIMIT, seed=ANY_SEED)
    cands = _told_initial_batch(opt, [1e12, 0.0])
    sources = [cands[0].x, cands[1].x]

    first_targets, match_counts = _first_onlooker_targets(opt, sources, cycles=30, score=1e13)

    assert set(match_counts) == {1}
    assert 1 in first_targets
