"""ABCOptimizer ask/tell protocol: candidate invariants, init protocol, best tracking and misuse.

ABCOptimizer ask/tell contract: candidate invariants, neighbour moves, init protocol and best tracking.

Everything is observed through the public API: the Candidate objects that ask() returns, best_params and
best_score, and the documented read-only counters phase, cycle and trial. Food sources are never read; a
source is identified among the earlier candidates, because every food source is the x of a candidate that
was told earlier (abc.py replaces a source only with a told candidate).

Oracles: Candidate docstring (x in the unit cube, params as the user's function receives them, key the
hashable decoded identity); README ("Optimizers search the unit cube [0, 1]^d", ask/tell example with
best_params and best_score); Optimizer docstring (scores in the user's direction, a NaN or infinite score
is the worst value for the direction); abc.py module docstring (only the initial population is a batch, a
neighbour differs from its food source in one coordinate, at most one scout per cycle); the original
hyp_optimizer ABC.py (calculate_neighbour_solution copies the source and changes one random param2change,
send_employed_bees loops from source 0 each cycle, send_scout_bees replaces a single argmax source);
Karaboga & Basturk (2007), v_ij = x_ij + phi_ij (x_ij - x_kj) for one randomly chosen j; task
specification (init may be told partially, asking with a pending candidate raises RuntimeError).

Seeds: ANY_SEED is arbitrary and fixed for reproducibility. The asserted properties (invariants, at most
one changed coordinate, running best) do not depend on the seed; the long runs only need some
replacements and some scouts, which colony_size=6 with limit=3 gives (the tests assert that those paths
ran). FAULT_SEED is another arbitrary seed, so the fault-injection run is not the same trajectory.

Unspecified, not asserted:
- best_params while only non-finite scores have been told (the implementation returns the first told
  candidate, not None);
- which of several equally good candidates is the best; scores are chosen so ties come only from equal
  params;
- the state after a rejected tell, and whether a multi-candidate tell is atomic;
- re-asking only the untold indices of the initial batch: after the first init ask every untold index is
  pending until it is told, so that branch cannot be reached through the public API.

Error paths of ABCOptimizer: constructor arguments and ask/tell protocol misuse.

Covered raise sites (abc.py): ABCOptimizer.__init__ "colony_size" and "limit", ABCOptimizer.ask
RuntimeError "initial" and "pending candidate", ABCOptimizer.tell ValueError "same length" and "not asked".
The full inventory is in the module docstring of test_space_errors.py.

Each call under test has several error paths of the same class, so every case carries a short keyword
that ties it to its own site.

Unspecified, not asserted:
- what ask() returns after the initial batch was told partially (the spec only says it may be told partially);
- whether optimizer state is unchanged after a failed tell, so every offending tell uses a one-element list
  and nothing is asserted about state afterwards;
- which check fires first when colony_size and limit are both invalid, so each case has one invalid argument.

ANY_SEED is fixed for determinism; none of these error paths depends on random draws. IRRELEVANT_SCORE is a
dummy score that only moves the optimizer to the next phase; phase progress does not depend on its value.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: Seeded ask/tell runs on small mixed spaces for the invariants, and one misuse per protocol error.
"""

import math
from collections import Counter

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Categorical, Float, Int, Space

ANY_SEED = 0


FAULT_SEED = 4


IRRELEVANT_SCORE = 1.0


LONG_RUN_STEPS = 300


STEPS_AFTER_FAULTS = 120


SCOUT_RUN_ASKS = 60


CYCLES_AFTER_IMPROVEMENT = 10


NOT_BETTER_SCORE = 10.0


CLEARLY_BETTER_SCORE = 0.0


CHOICE_PENALTY = {"x": 0.0, "y": 0.05, "z": 0.2}


WORST_FOR = {"min": math.inf, "max": -math.inf}


SIGN_FOR = {"min": 1.0, "max": -1.0}


def _mixed_space():
    """A fresh 3-dimensional Space: one Float, one Int, one Categorical (no log kinds: exact decodes)."""
    return Space({"a": Float(-1.0, 1.0), "b": Int(1, 9), "c": Categorical(["x", "y", "z"])})


def _score_depending_on_every_param(params):
    """Finite, non-negative score that changes with every parameter, so sources are replaced now and then."""
    return abs(params["a"] - 0.3) + 0.1 * params["b"] + CHOICE_PENALTY[params["c"]]


def _scores_in_direction(direction):
    """Score schedule: told number 0 gets NaN, every later candidate the direction-signed objective."""

    def score_of(told_number, params):
        if told_number == 0:
            return math.nan
        return SIGN_FOR[direction] * _score_depending_on_every_param(params)

    return score_of


def _plain_scores(told_number, params):
    """Score schedule that ignores the told number."""
    return _score_depending_on_every_param(params)


def _candidate_violations(space, cand):
    """Names of the Candidate invariants that cand breaks; an empty list when it satisfies all of them."""
    expected_params = space.to_dicts(cand.x)[0]
    checks = {
        "x is not an array of shape (d,)": isinstance(cand.x, np.ndarray) and cand.x.shape == (space.dim,),
        "x leaves the closed unit cube": bool(np.all(cand.x >= 0.0) and np.all(cand.x <= 1.0)),
        "params differ from to_dicts(x)": cand.params == expected_params,
        "param types differ from to_dicts(x)": {n: type(v) for n, v in cand.params.items()}
        == {n: type(v) for n, v in expected_params.items()},
        "key differs from decode(x)": cand.key == tuple(space.decode(cand.x)[0]),
    }
    return [name for name, ok in checks.items() if not ok]


def _new_record():
    """Empty observation lists, filled in ask/tell order by _observe_ask and _tell_one."""
    return {
        "candidates": [],
        "phases": [],
        "violations": [],
        "told_scores": [],
        "best_scores": [],
        "best_params": [],
    }


def _observe_ask(opt, cands, record):
    """Records each asked candidate, the phase after the ask and its invariant violations, before any tell."""
    for cand in cands:
        record["candidates"].append(cand)
        record["phases"].append(opt.phase)
        record["violations"].extend((cand.id, v) for v in _candidate_violations(opt.space, cand))


def _tell_one(opt, cand, score_of, record):
    """Tells one candidate and records the told score and best_score/best_params right after the tell."""
    score = score_of(len(record["told_scores"]), cand.params)
    opt.tell([cand], [score])
    record["told_scores"].append(score)
    record["best_scores"].append(opt.best_score)
    record["best_params"].append(opt.best_params)


def _run_single_steps(opt, score_of, steps, record):
    """Runs `steps` rounds of ask, record, tell; returns the record."""
    for _ in range(steps):
        cands = opt.ask()
        _observe_ask(opt, cands, record)
        for cand in cands:
            _tell_one(opt, cand, score_of, record)
    return record


def _drive(opt, score_of, steps):
    """Asks the initial batch, tells it one candidate per call (init may be told partially), then runs
    `steps` single rounds; returns the record."""
    record = _new_record()
    initial = opt.ask()
    _observe_ask(opt, initial, record)
    for cand in initial:
        _tell_one(opt, cand, score_of, record)
    return _run_single_steps(opt, score_of, steps, record)


def _expected_running_best(record, direction, space):
    """Hand oracle over the told scores: after every tell, the extreme finite score so far (the worst value
    while none is finite) and to_dicts of the first candidate that reached it; plus the first finite index."""
    better = (lambda s, b: s < b) if direction == "min" else (lambda s, b: s > b)
    best, best_x, first_finite = WORST_FOR[direction], None, None
    scores, params = [], []
    for n, (cand, score) in enumerate(zip(record["candidates"], record["told_scores"], strict=True)):
        if math.isfinite(score) and (best_x is None or better(score, best)):
            best, best_x = score, cand.x
            first_finite = n if first_finite is None else first_finite
        scores.append(best)
        params.append(None if best_x is None else space.to_dicts(best_x)[0])
    return scores, params, first_finite


def _single_coordinate_change(space, source, cand):
    """The set of changed coordinates when cand is a one-coordinate neighbour of source (at most one x
    coordinate differs, params and key are equal outside it); None otherwise."""
    diff = [int(j) for j in np.flatnonzero(cand.x != source.x)]
    if len(diff) > 1:
        return None
    same_params = all(cand.params[n] == source.params[n] for j, n in enumerate(space.names) if j not in diff)
    same_key = all(cand.key[j] == source.key[j] for j in range(space.dim) if j not in diff)
    return set(diff) if same_params and same_key else None


def _neighbour_source_report(space, record):
    """Looks for an earlier candidate that each employed or onlooker candidate is a one-coordinate
    neighbour of; returns (ids without one, coordinates changed by neighbours differing in exactly one)."""
    unmatched, changed = [], set()
    for n, (cand, phase) in enumerate(zip(record["candidates"], record["phases"], strict=True)):
        if phase not in ("employed", "onlooker"):
            continue
        earlier = record["candidates"][:n]
        matches = [m for m in (_single_coordinate_change(space, s, cand) for s in earlier) if m is not None]
        if not matches:
            unmatched.append(cand.id)
        changed.update(*[m for m in matches if len(m) == 1][:1])
    return unmatched, changed


def _scout_observations(opt, score, limit, asks):
    """Runs `asks` single rounds with a constant score; returns (scout asks per cycle, number of exhausted
    sources at each scout step)."""
    scout_cycles, exhausted = [], []
    for _ in range(asks):
        if opt.phase == "scout":
            exhausted.append(sum(t >= limit for t in opt.trial))
        (cand,) = opt.ask()
        if opt.phase == "scout":
            scout_cycles.append(opt.cycle)
        opt.tell([cand], [score])
    return Counter(scout_cycles), exhausted


def _ask_and_tell_one_by_one(opt, scores):
    """Asks one candidate per score and tells it that score; returns the candidates in order."""
    told = []
    for score in scores:
        (cand,) = opt.ask()
        opt.tell([cand], [score])
        told.append(cand)
    return told


def _first_candidate_of_each_cycle(opt, score, cycles):
    """Runs single rounds with a constant score; returns the first candidate of each of the next `cycles`
    cycles, i.e. the first ask after opt.cycle changed, which is employed bee 0."""
    firsts, last_cycle = [], opt.cycle
    while len(firsts) < cycles:
        (cand,) = opt.ask()
        if opt.cycle != last_cycle:
            firsts.append(cand)
            last_cycle = opt.cycle
        opt.tell([cand], [score])
    return firsts


def _hamming(source, cands):
    """Number of differing x coordinates between source and each candidate."""
    return [int(np.count_nonzero(c.x != source.x)) for c in cands]


def _neighbour_flags(space, source, cands):
    """True for each candidate that is a one-coordinate neighbour of source."""
    return [_single_coordinate_change(space, source, c) is not None for c in cands]


def _ids(record):
    return [c.id for c in record["candidates"]]


def _all_finite(values):
    return all(math.isfinite(v) for v in values)


def test_abc_optimizer_ask_should_return_candidates_that_satisfy_the_invariants_at_every_step():
    """Given a seeded run over a mixed space, when each candidate is checked right after ask(), then
    none breaks a Candidate invariant.

    Oracle: Candidate docstring, README unit cube [0, 1]^d (closed, since clipping gives exactly 0.0 or
    1.0), Space.to_dicts/decode docstrings. Inputs: the initial batch (sampled points decoded by rows) and
    the single asks (neighbours decoded by Space.move, scouts by rows) are different code paths; the run
    must contain all of them, which the phase set asserts. Checking before the tell names the candidate id
    and the broken invariant at the step that produced it. Ids must be strictly increasing, hence unique.
    """
    space = _mixed_space()
    opt = ABCOptimizer(space, colony_size=6, limit=3, seed=ANY_SEED)

    record = _drive(opt, _plain_scores, LONG_RUN_STEPS)

    assert record["violations"] == []
    assert _ids(record) == sorted(set(_ids(record)))
    assert set(record["phases"]) == {"init", "employed", "onlooker", "scout"}


def test_abc_optimizer_ask_should_move_one_coordinate_of_a_source_when_phase_is_employed_or_onlooker():
    """Given a seeded run, when an employed or onlooker candidate is asked, then it is a one-coordinate
    neighbour of an earlier candidate.

    Oracle: abc.py module docstring ("A neighbour differs from its food source in one coordinate"),
    Karaboga & Basturk (2007) v_ij with a single j, original ABC.py calculate_neighbour_solution.
    "Neighbour" means: at most one x coordinate differs, and params and key are equal outside it (copied
    from the source). Zero differing coordinates is allowed, because phi * (x_ij - x_kj) can be 0 and
    clipping can return the old value. Scouts are random points and are skipped. A move of two coordinates,
    or params copied from another source, finds no matching earlier candidate.
    """
    space = _mixed_space()
    opt = ABCOptimizer(space, colony_size=6, limit=3, seed=ANY_SEED)
    record = _drive(opt, _plain_scores, LONG_RUN_STEPS)

    unmatched, changed = _neighbour_source_report(space, record)

    assert unmatched == []


def test_abc_optimizer_ask_should_spread_neighbour_moves_over_every_coordinate():
    """Given a seeded 3-dimensional run of LONG_RUN_STEPS steps, when the single changed coordinate of each
    neighbour is collected, then all 3 coordinates occur.

    Oracle: Karaboga & Basturk (2007) and original ABC.py (param2change = random.randint(0, D - 1)): the
    coordinate is drawn at random, so about 250 moves over 3 coordinates leave none out. A neighbour that
    always moves the same coordinate fails (site: ABCOptimizer._neighbour choice of j).
    """
    space = _mixed_space()
    opt = ABCOptimizer(space, colony_size=6, limit=3, seed=ANY_SEED)
    record = _drive(opt, _plain_scores, LONG_RUN_STEPS)

    unmatched, changed = _neighbour_source_report(space, record)

    assert changed == set(range(space.dim))


def test_abc_optimizer_ask_should_issue_at_most_one_scout_per_cycle_when_several_sources_are_exhausted():
    """Given limit=1 and a constant score, when the run reaches scout steps with several exhausted
    sources, then each cycle that has a scout ask has exactly one.

    Oracle: abc.py module docstring ("at most one scout per cycle"), original ABC.py send_scout_bees (one
    argmax source is replaced). Input: with a constant score no neighbour is ever strictly better, so every
    source reaches trial >= limit=1 and the decisive situation, two or more exhausted sources at one scout
    step, occurs (asserted). A scout step that replaced every exhausted source would ask several scouts in
    one cycle. opt.trial and opt.cycle are read-only public counters.
    """
    limit = 1
    opt = ABCOptimizer(_mixed_space(), colony_size=6, limit=limit, seed=ANY_SEED)
    _drive(opt, _plain_scores, 0)

    scouts_per_cycle, exhausted = _scout_observations(opt, IRRELEVANT_SCORE, limit, SCOUT_RUN_ASKS)

    assert max(scouts_per_cycle.values()) == 1
    assert max(exhausted) >= 2


@pytest.mark.parametrize("direction", ["min", "max"])
def test_abc_optimizer_tell_should_keep_the_best_at_the_best_finite_told_candidate(direction):
    """Given a run whose first told score is NaN, when best_score and best_params are read after every
    tell, then they follow the best finite score told so far.

    Contract (postcondition of tell): best_score equals the extreme finite score told so far in the user's
    direction (so it never gets worse), or the worst value for the direction (+inf for min, -inf for max)
    while only non-finite scores were told; best_params equals to_dicts of the x of the first candidate
    that reached it. Oracle: Optimizer docstring (user's direction, NaN is the worst value), README
    ask/tell example. Expected values come from a hand model over the told scores, not from the optimizer.
    Inputs: the leading NaN covers the all-non-finite state and the switch to finite; for max the scores
    are negated and negative, so a missing sign flip would pick the minimum. best_params is not compared
    before the first finite tell (unspecified).
    """
    space = _mixed_space()
    opt = ABCOptimizer(space, direction=direction, colony_size=6, limit=3, seed=ANY_SEED)

    record = _drive(opt, _scores_in_direction(direction), LONG_RUN_STEPS)

    expected_scores, expected_params, first_finite = _expected_running_best(record, direction, space)
    assert record["best_scores"] == expected_scores
    assert record["best_params"][first_finite:] == expected_params[first_finite:]


def test_abc_optimizer_ask_should_raise_runtime_error_when_part_of_the_initial_batch_is_pending():
    """Given 1 of 3 initial candidates told, when ask() is called, then RuntimeError is raised.

    Precondition of ask: no candidate may be pending. Oracle: task specification (init may be told
    partially; asking again with a pending candidate raises RuntimeError) and the message 'tell the
    pending initial candidates' in ABCOptimizer.ask; match='initial' separates it from the employed-phase
    RuntimeError of the same call. The case with no initial candidate told belongs to the error-path tests;
    here a tell has already changed state (site: the init branch of ask, which must exclude pending
    indices and not only told ones).
    """
    opt = ABCOptimizer(_mixed_space(), colony_size=6, seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial[:1], [IRRELEVANT_SCORE])

    with pytest.raises(RuntimeError, match="initial") as excinfo:
        opt.ask()

    assert excinfo.type is RuntimeError


def test_abc_optimizer_tell_should_leave_the_init_phase_after_the_last_initial_candidate_out_of_order():
    """Given 3 initial candidates told one per call in the order 0, 2, 1, when phase is read after each
    tell and ask() is called, then phase is init, init, employed and ask returns one candidate.

    Definition-use pair across methods: tell defines the initialised flags and the phase, the next ask
    uses them. Oracle: task specification (init may be told partially), abc.py docstring (only the initial
    population is a batch, employed bees go one source after another). Input: telling index 2 before
    index 1 shows that the switch waits for every index, not for the last position.
    """
    colony_size = 6
    opt = ABCOptimizer(_mixed_space(), colony_size=colony_size, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial[0:1], [3.0])
    phase_after_first = opt.phase
    opt.tell(initial[2:3], [2.0])
    phase_after_second = opt.phase
    opt.tell(initial[1:2], [1.0])
    phase_after_last = opt.phase
    next_candidates = opt.ask()

    assert len(initial) == colony_size // 2
    assert [phase_after_first, phase_after_second, phase_after_last] == ["init", "init", "employed"]
    assert len(next_candidates) == 1
    assert opt.phase == "employed"


def test_abc_optimizer_best_params_should_be_none_when_nothing_was_told():
    """Given the initial batch asked but not told, when best_params is read, then it is None.

    Oracle (derived): the public annotation of Optimizer.best_params is dict | None, and no score has been
    reported yet. Asking first shows that handing out candidates does not set a best.
    """
    opt = ABCOptimizer(_mixed_space(), colony_size=6, seed=ANY_SEED)
    opt.ask()

    best_params = opt.best_params

    assert best_params is None


@pytest.mark.parametrize(
    ("direction", "expected_score", "best_single"),
    [
        pytest.param("min", 1.0, 1, id="min-best-is-second-single-ask"),
        pytest.param("max", 9.0, 2, id="max-best-is-third-single-ask"),
    ],
)
def test_abc_optimizer_best_params_should_decode_the_best_told_candidate_when_it_came_from_a_single_ask(
    direction, expected_score, best_single
):
    """Given distinct scores 5, 2, 8 for the initial batch and 6, 1, 9, 3 for four single asks, when the
    best is read, then it is the best of those candidates.

    Definition-use pair across methods: tell defines the best x and score, the best_params and best_score
    properties use them. Oracle: Optimizer docstring (scores in the user's direction), README ask/tell
    example. Inputs: all scores distinct, so there is no tie; the best for both directions comes from a
    single ask, not from the initial batch, and the max case picks 9.0 where a missing sign flip would
    give 1.0.
    """
    space = _mixed_space()
    opt = ABCOptimizer(space, direction=direction, colony_size=6, seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial, [5.0, 2.0, 8.0])

    singles = _ask_and_tell_one_by_one(opt, [6.0, 1.0, 9.0, 3.0])

    best = singles[best_single]
    assert opt.best_score == expected_score
    assert opt.best_params == best.params
    assert opt.best_params == space.to_dicts(best.x)[0]


def test_abc_optimizer_ask_should_build_later_neighbours_from_the_improved_source():
    """Given source 0 replaced by a strictly better employed candidate, when the first candidate of each of
    the next 10 cycles is asked, then each one is a one-coordinate neighbour of the improved candidate.

    Definition-use pair across methods: tell replaces source 0 (x, key and params), the next cycle's ask
    builds employed bee 0 from it. Oracle: Karaboga & Basturk (2007) greedy selection, original ABC.py
    send_employed_bees (foods[i] replaced when fitness is greater, loop from i = 0 each cycle). Inputs:
    every other tell scores NOT_BETTER_SCORE, never strictly better than itself or than
    CLEARLY_BETTER_SCORE, so source 0 stays the improved candidate; limit=1000 rules out scouts. The
    improved candidate differs from the old source in one coordinate (arranged precondition); some first
    candidate moves another coordinate, so it is 2 away from the old source (asserted) and would not be a
    neighbour of the improved one if the replacement was lost.
    """
    space = Space(
        {"a": Float(-1.0, 1.0), "b": Int(1, 9), "c": Categorical(["x", "y", "z"]), "d": Float(0.0, 1.0)}
    )
    opt = ABCOptimizer(space, colony_size=6, limit=1000, seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial, [NOT_BETTER_SCORE] * len(initial))
    (improved,) = opt.ask()
    opt.tell([improved], [CLEARLY_BETTER_SCORE])

    firsts = _first_candidate_of_each_cycle(opt, NOT_BETTER_SCORE, CYCLES_AFTER_IMPROVEMENT)

    assert np.count_nonzero(improved.x != initial[0].x) == 1
    assert _neighbour_flags(space, improved, firsts) == [True] * CYCLES_AFTER_IMPROVEMENT
    assert max(_hamming(initial[0], firsts)) == 2


def test_abc_optimizer_should_keep_working_when_the_initial_batch_is_told_in_pieces_with_a_lost_worker():
    """Given a 5-candidate initial batch told as [3] with NaN, then [0, 4], then [2], then [1], when
    STEPS_AFTER_FAULTS normal rounds follow, then best_score and every later candidate stay valid.

    Fault injection: the NaN for index 3 simulates a failed worker, the split and shuffled tells slow
    workers. Oracle: task specification (init may be told partially), Optimizer docstring (NaN is the
    worst value, so best_score is +inf for min after it), Candidate docstring for the later candidates.
    Postconditions: best_score is +inf after the NaN alone, finite from the first finite tell on and after
    every later tell, and every later candidate satisfies the candidate invariants.
    """
    opt = ABCOptimizer(_mixed_space(), colony_size=10, limit=3, seed=FAULT_SEED)
    initial = opt.ask()
    score = _score_depending_on_every_param

    opt.tell([initial[3]], [math.nan])
    best_after_lost_worker = opt.best_score
    opt.tell([initial[0], initial[4]], [score(initial[0].params), score(initial[4].params)])
    best_after_first_finite = opt.best_score
    opt.tell([initial[2]], [score(initial[2].params)])
    opt.tell([initial[1]], [score(initial[1].params)])
    record = _run_single_steps(opt, _plain_scores, STEPS_AFTER_FAULTS, _new_record())

    assert best_after_lost_worker == math.inf
    assert math.isfinite(best_after_first_finite)
    assert _all_finite(record["best_scores"])
    assert record["violations"] == []


DUMMY_SCORE = 0.0


STRANGER_ID = 999


def _finish_initial_batch(opt):
    """Asks the initial batch and tells it with dummy scores, which moves the optimizer to the employed
    phase."""
    initial = opt.ask()
    opt.tell(initial, [DUMMY_SCORE] * len(initial))


def _optimizer_with_pending_employed_and_stranger():
    """An optimizer with one employed candidate pending, plus a public Candidate whose id was never issued."""
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=4, seed=ANY_SEED)
    _finish_initial_batch(opt)
    opt.ask()
    stranger = Candidate(STRANGER_ID, np.zeros(1), {}, ())
    return opt, stranger


def _optimizer_with_told_employed():
    """An optimizer whose first employed candidate was asked and told once, successfully."""
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=4, seed=ANY_SEED)
    _finish_initial_batch(opt)
    (employed,) = opt.ask()
    opt.tell([employed], [DUMMY_SCORE])
    return opt, employed


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"colony_size": 3}, "colony_size", id="colony-3-boundary"),
        pytest.param({"colony_size": 0}, "colony_size", id="colony-0"),
        pytest.param({"limit": 0}, "limit", id="limit-0-boundary"),
        pytest.param({"limit": -1}, "limit", id="limit-negative"),
    ],
)
def test_abc_optimizer_should_raise_value_error_when_colony_size_or_limit_is_too_small(kwargs, match):
    """Given one too-small colony_size or limit, when ABCOptimizer is built, then ValueError comes from that
    site.

    Oracle: task specification (colony_size >= 4, limit >= 1, invalid values raise ValueError) and the
    messages in ABCOptimizer.__init__. Inputs: 3 and 0 are one below the boundary and one past it, 0 and -1
    likewise for limit; the other argument keeps its valid default (colony_size=10, limit=15). The keyword
    of each message does not occur in the other, so a case fails if the wrong site fires.
    """
    space = Space({"x": Float(0.0, 1.0)})

    with pytest.raises(ValueError, match=match) as excinfo:
        ABCOptimizer(space, seed=ANY_SEED, **kwargs)

    assert excinfo.type is ValueError


def test_abc_optimizer_should_hand_out_two_initial_candidates_when_colony_size_and_limit_are_minimal():
    """Given colony_size=4 and limit=1, when the optimizer is built and asked once, then it hands out 2
    sources.

    Oracle: task specification (colony_size >= 4, limit >= 1, init asks colony_size // 2 sources as one
    batch). These are the valid sides of both boundaries tested above.
    """
    colony_size = 4
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=colony_size, limit=1, seed=ANY_SEED)

    initial = opt.ask()

    assert len(initial) == colony_size // 2


def test_abc_optimizer_ask_should_raise_runtime_error_when_all_initial_candidates_are_pending():
    """Given the whole initial batch asked and none told, when ask() is called again, then RuntimeError is
    raised.

    Oracle: task specification 'asking again with a pending candidate -> RuntimeError' and the message
    'tell the pending initial candidates' in ABCOptimizer.ask. The keyword 'initial' does not occur in the
    employed-phase message, so this pins the init site.
    """
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=4, seed=ANY_SEED)
    opt.ask()

    with pytest.raises(RuntimeError, match="initial") as excinfo:
        opt.ask()

    assert excinfo.type is RuntimeError


def test_abc_optimizer_ask_should_raise_runtime_error_when_an_employed_candidate_is_pending():
    """Given one employed candidate asked and not told, when ask() is called again, then RuntimeError is
    raised.

    Oracle: task specification 'asking again with a pending candidate -> RuntimeError' and the message
    'tell the pending candidate' in ABCOptimizer.ask. 'pending candidate' does not match the init message
    ('pending initial candidates'), so the test fails if the init site fired instead. The dummy scores only
    move the optimizer into the employed phase.
    """
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=4, seed=ANY_SEED)
    _finish_initial_batch(opt)
    (employed,) = opt.ask()

    with pytest.raises(RuntimeError, match="pending candidate") as excinfo:
        opt.ask()

    assert excinfo.type is RuntimeError


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([], id="scores-empty"),
        pytest.param([DUMMY_SCORE], id="scores-one-short"),
        pytest.param([DUMMY_SCORE] * 3, id="scores-one-long"),
    ],
)
def test_abc_optimizer_tell_should_raise_value_error_when_lengths_differ(scores):
    """Given the 2 initial candidates, when tell() gets a different number of scores, then ValueError is
    raised.

    Oracle: task specification 'mismatched lengths -> ValueError' and the message 'must have the same
    length' in ABCOptimizer.tell. Inputs: no scores at all, one too few and one too many; the last two show
    the check fails in both directions. The keyword rules out the 'not asked' site of the same call.
    """
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    with pytest.raises(ValueError, match="same length") as excinfo:
        opt.tell(initial, scores)

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_optimizer_with_pending_employed_and_stranger, id="unknown-id-while-another-is-pending"),
        pytest.param(_optimizer_with_told_employed, id="employed-told-twice"),
    ],
)
def test_abc_optimizer_tell_should_raise_value_error_when_candidate_is_unknown_or_already_told(arrange):
    """Given a candidate that is not pending, when tell() gets it, then ValueError is raised.

    Oracle: task specification 'telling an unknown/already-told candidate -> ValueError' and the message
    'was not asked for or was already told' in ABCOptimizer.tell. Inputs: STRANGER_ID can never be pending
    because only 3 ids were issued, and it is told while another candidate IS pending, so a tell that
    ignores ids and consumes any pending entry would not raise; the second case tells the employed candidate
    a second time. A candidate from another optimizer is not used because its ids could clash.
    """
    opt, offending = arrange()

    with pytest.raises(ValueError, match="not asked") as excinfo:
        opt.tell([offending], [DUMMY_SCORE])

    assert excinfo.type is ValueError
