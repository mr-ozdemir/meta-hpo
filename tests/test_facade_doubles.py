"""optimize() against doubles: a scripted optimizer and stubs, mocks and spies of the user objective.

Facade bookkeeping tested against a scripted optimizer double, plus the optimizer contract the
double shares with the real ABCOptimizer.

ScriptedOptimizer replaces the optimizer behind optimize() so the facade's own bookkeeping (exact budget,
trial records, what is told back to the optimizer, the space/direction check) is tested without the ABC
search. It enters only through the public optimizer= parameter and subclasses the public Optimizer. Because
it is extra code the facade tests rely on, it has its own tests, and the contract tests apply one shared
driver to both the double and the real ABCOptimizer so the two cannot drift apart unnoticed. Two tests run
optimize() with the real ABCOptimizer to close the gap the double leaves on the partial-batch path.

Oracles: optimize docstring (max_evals counts evaluations, cache reuse), the task specification (exact
max_evals trials even inside the initial batch; ValueError for a mismatched optimizer space or direction;
ABC init may be told partially; tell errors), the Optimizer docstring (ask() hands out candidates, tell()
reports their scores in the user's direction), the Candidate docstring (key is the hashable decoded
identity), Space.rows/decode docstrings and the README (optimizers search the unit cube [0, 1]^d).

Unspecified, not asserted: whether an equal but distinct Space object is accepted by optimize() (the code
checks identity); Trial.number values and the order of Study.trials; which candidates of a truncated batch
are evaluated (only that exactly the evaluated ones are told); whether the facade tells once per batch or
per candidate; how many times ask() is called; whether candidate ids increase (only uniqueness is needed
by the tell protocol); what a real ABCOptimizer does on a later ask() after a truncated run.

optimize() (facade.py) driven by doubles of the user objective: stubs, sequenced stubs, mocks and a spy.

Only the user objective is replaced; Space and the default ABCOptimizer are the real ones with a fixed seed.
Expected values are read off the values programmed into each double, never captured from a seeded run.
ANY_SEED is used where the asserted property does not depend on the seed; SPY_SEED only fixes the run.

Unspecified, recorded as current behaviour or not asserted:
- Propagation of an exception raised by the objective is not documented in README, CHANGELOG or the
  docstrings; the tests record the current behaviour (the exception escapes unchanged and no further call is
  made). Whether a partial Study is reachable afterwards is unspecified and not asserted.
- Tie-breaking of best_trial among equal finite scores is unspecified, so with a constant stub only
  membership of best_params among the trial params is asserted.
- Acceptance of numpy scalar scores that are not float subclasses (np.float32, np.int64) rests on the float()
  conversion in facade.py, not on README text, which only says the function returns a score.
- Study.scores keeps the raw non-finite value of a failed trial; README's "counts as the worst value" is
  about ranking, so NaN marks only the NaN failures and ~isfinite marks every failure.
- The order of calls (call k is the k-th non-cached trial) follows from the sequential loop in facade.py.
- The exact number of calls of an all-NaN objective with cache=True depends on repeated keys and is not
  asserted.

Branch map of optimize() exercised here (P-T3-06 full-coverage criterion): cached trial (one-point space,
tiny space), first occurrence with cache=True (same tests, trial 0), cache=False (sequenced stubs,
every-trial test), truncated initial batch (three-value sequence), exception path (raising stubs). The
max_evals and optimizer-mismatch ValueError branches belong to the facade validation items.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: A scripted optimizer and objective stubs, mocks and spies chosen per facade path.
"""

import itertools
import math
from collections import Counter
from itertools import combinations
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Categorical, Float, Int, Optimizer, Space, Study, optimize

ANY_SEED = 0


CONTRACT_COLONY_SIZE = 6


N_CONTRACT_ASKS = 8


FACADE_SCRIPT = (
    ((0.1, 0.1), (0.5, 0.1), (0.9, 0.1)),
    ((0.1, 0.9),),
    ((0.5, 0.9),),
)


FACADE_PARAM_VALUES = (("a", 1), ("b", 1), ("c", 1), ("a", 2), ("b", 2))


FACADE_KEYS = ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (0.0, 2.0), (1.0, 2.0))


FACADE_RETURNS = (0.5, 0.25, 0.75, 1.5, 2.5)


CONTRACT_SCRIPT = (
    ((0.1, 0.1), (0.0, 0.0), (1.0, 1.0)),
    ((0.5, 0.9),),
    ((0.2, 0.3),),
    ((0.9, 0.1),),
    ((0.5, 0.1),),
    ((0.1, 0.9),),
    ((0.7, 0.6),),
    ((0.35, 0.5),),
    ((0.95, 0.95),),
)


class ScriptExhausted(Exception):
    """Raised by ScriptedOptimizer when it is asked for a batch after its script has run out."""


class ScriptedOptimizer(Optimizer):
    """Scripted optimizer double: hands out the batches of `script` in order and records every ask and tell.

    Each batch is a tuple of unit-cube points. x is built first and params/key come from the real
    space.rows(x). Ids come from its own counter starting at 0. ask() raises RuntimeError while candidates
    are pending and ScriptExhausted when the script has no batch left, so a facade that over-asks fails
    loudly. tell() raises ValueError ('same length') for mismatched lengths and ValueError ('not asked')
    for a candidate that is not pending, as ABCOptimizer does. It checks calls, not search state.
    """

    def __init__(self, space, direction, script):
        super().__init__(space, direction, ANY_SEED)
        self.script = list(script)
        self.asked = []
        self.tells = []
        self.pending = {}
        self.issued = 0

    def ask(self):
        if self.pending:
            raise RuntimeError("tell the pending candidates before asking again")
        if not self.script:
            raise ScriptExhausted("the script has no batch left")
        xs = np.array(self.script.pop(0), dtype=np.float64)
        keys, params = self.space.rows(xs)
        batch = [
            Candidate(self.issued + n, x, p, k)
            for n, (x, p, k) in enumerate(zip(xs, params, keys, strict=True))
        ]
        self.issued += len(batch)
        self.pending.update({c.id: c for c in batch})
        self.asked.append(batch)
        return batch

    def tell(self, candidates, scores):
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")
        if any(c.id not in self.pending for c in candidates):
            raise ValueError("a candidate was not asked for or was already told")
        self.tells.append((list(candidates), list(scores)))
        for c in candidates:
            del self.pending[c.id]


def _two_param_space():
    """A fresh 6-combination space with distinct, non-equal choices: k in a/b/c, n in 1..2."""
    return Space({"k": Categorical(["a", "b", "c"]), "n": Int(1, 2)})


def stub_objective(params):
    """Stub objective: a=0, b=1, c=2 plus n, so every value is a hand-checkable whole number."""
    return {"a": 0.0, "b": 1.0, "c": 2.0}[params["k"]] + params["n"]


def _scripted_facade_optimizer(space, direction="min"):
    """ScriptedOptimizer over FACADE_SCRIPT: batches of 3, 1, 1 with five distinct parameter sets."""
    return ScriptedOptimizer(space, direction, FACADE_SCRIPT)


def _mock_objective():
    """Mock objective returning FACADE_RETURNS in call order; a sixth call raises StopIteration."""
    return mock.Mock(side_effect=list(FACADE_RETURNS))


def _drive(opt, n_asks):
    """Asks n_asks times, tells each batch in full with stub_objective; returns all candidates in order."""
    seen = []
    for _ in range(n_asks):
        batch = opt.ask()
        opt.tell(batch, [stub_objective(c.params) for c in batch])
        seen.extend(batch)
    return seen


def _params_tuple(params):
    """A hashable form of a params dict."""
    return tuple(sorted(params.items()))


def _pair_counts(pairs):
    """Order-free multiset of (params, score) pairs."""
    return Counter((_params_tuple(p), s) for p, s in pairs)


def _call_pairs(mock_objective):
    """(params, returned score) of every call of the mock objective, using the scripted FACADE_RETURNS."""
    called = [c.args[0] for c in mock_objective.call_args_list]
    return list(zip(called, FACADE_RETURNS[: len(called)], strict=True))


def _told_pairs_of_fake(fake):
    """(params, score) of every candidate the facade told to the scripted double, over all tells."""
    return [(c.params, s) for cands, scores in fake.tells for c, s in zip(cands, scores, strict=True)]


def _told_pairs_of_spy(spy):
    """(params, score) of every candidate passed to a spied tell(), over all calls."""
    return [(c.params, s) for call in spy.call_args_list for c, s in zip(*call.args, strict=True)]


def _told_ids_of_spy(spy):
    """Ids of every candidate passed to a spied tell(), over all calls."""
    return [c.id for call in spy.call_args_list for c in call.args[0]]


def _trial_pairs(study):
    """(params, score) of every trial of `study`."""
    return [(t.params, t.score) for t in study.trials]


def _trial_param_set(study):
    """Set of hashable params of every trial of `study`."""
    return {_params_tuple(t.params) for t in study.trials}


def _decoded_trial_params(space, study):
    """The decoding of each trial's x, in trial order."""
    return [space.to_dicts(t.x)[0] for t in study.trials]


def _trial_params(study):
    """Params of each trial, in trial order."""
    return [t.params for t in study.trials]


def _scripted_params():
    """Hand-decoded params of FACADE_SCRIPT (Categorical and Int give each value an equal share of [0, 1])."""
    return [{"k": k, "n": n} for k, n in FACADE_PARAM_VALUES]


def _batch_sizes(fake):
    """Number of candidates in each batch the scripted double handed out."""
    return [len(batch) for batch in fake.asked]


def _told_ids_of_fake(fake):
    """Candidate ids of each recorded tell of the scripted double."""
    return [[c.id for c in cands] for cands, _ in fake.tells]


def _told_scores_of_fake(fake):
    """Score list of each recorded tell of the scripted double."""
    return [scores for _, scores in fake.tells]


def _key_param_agreement(cands):
    """Set of (keys equal, params equal) over all candidate pairs; {(True, True), (False, False)} means keys
    are equal exactly when params are, and that both kinds of pair occurred."""
    return {(a.key == b.key, a.params == b.params) for a, b in combinations(cands, 2)}


def _distinct_key_count(cands):
    """Number of distinct keys; building the set needs every key to be hashable."""
    return len({c.key for c in cands})


def _distinct_params_count(cands):
    """Number of distinct decoded parameter sets."""
    return len({_params_tuple(c.params) for c in cands})


def _stacked_x(cands):
    """All candidate points as one (n, d) array."""
    return np.array([c.x for c in cands])


def _params_of(cands):
    """Params of each candidate."""
    return [c.params for c in cands]


def _decoded_params_of(space, cands):
    """The decoding of each candidate's x."""
    return [space.to_dicts(c.x)[0] for c in cands]


def _keys_of(cands):
    """Key of each candidate."""
    return [c.key for c in cands]


def _ids(cands):
    """Id of each candidate."""
    return [c.id for c in cands]


CONTRACT_OPTIMIZERS = [
    pytest.param(lambda space: ScriptedOptimizer(space, "min", CONTRACT_SCRIPT), id="scripted"),
    pytest.param(
        lambda space: ABCOptimizer(space, colony_size=CONTRACT_COLONY_SIZE, seed=ANY_SEED),
        id="abc",
    ),
]


def test_scripted_optimizer_should_hand_out_batches_of_the_scripted_sizes_when_each_batch_is_told():
    """Given the double over FACADE_SCRIPT, when it is asked three times with a full tell after each ask, then
    the batches have 3, 1 and 1 candidates.

    Oracle: the double's own docstring (P-T3-07: the fake the facade tests rely on gets its own tests).
    Input: a multi-candidate batch like ABC init followed by single batches like the later ABC phases.
    """
    fake = _scripted_facade_optimizer(_two_param_space())

    _drive(fake, len(FACADE_SCRIPT))

    assert _batch_sizes(fake) == [3, 1, 1]


def test_scripted_optimizer_should_derive_params_and_keys_from_the_real_space_when_handing_out_candidates():
    """Given the double over FACADE_SCRIPT, when all three batches are handed out, then each candidate has the
    hand-decoded params and the raw-row key of its scripted x.

    Oracle: hand decoding by the Categorical/Int docstrings (equal share of [0, 1] per value, e.g. u=0.5 of
    three choices is index 1 'b', u=0.9 of Int(1, 2) is 2) and Space.rows/decode docstrings (key is the raw
    row, categories as indices, integers as whole numbers).
    """
    fake = _scripted_facade_optimizer(_two_param_space())

    cands = _drive(fake, len(FACADE_SCRIPT))

    assert _params_of(cands) == _scripted_params()
    assert _keys_of(cands) == list(FACADE_KEYS)


def test_scripted_optimizer_should_record_each_tell_with_its_candidates_and_scores():
    """Given the double over FACADE_SCRIPT, when its three batches are told with stub_objective scores, then
    the recorded tells hold the ids and scores in the order they were passed.

    Oracle: the double's docstring and hand calculation of stub_objective: a1=1, b1=2, c1=3, a2=2, b2=3.
    """
    fake = _scripted_facade_optimizer(_two_param_space())

    _drive(fake, len(FACADE_SCRIPT))

    assert _told_ids_of_fake(fake) == [[0, 1, 2], [3], [4]]
    assert _told_scores_of_fake(fake) == [[1.0, 2.0, 3.0], [2.0], [3.0]]


def test_scripted_optimizer_should_raise_script_exhausted_when_asked_past_the_end_of_its_script():
    """Given the double whose three scripted batches were all handed out and told, when ask() is called again,
    then ScriptExhausted is raised instead of an empty batch or a loop.

    Oracle: the double's docstring (fail loudly). Input: the boundary just past the end of the script.
    """
    fake = _scripted_facade_optimizer(_two_param_space())
    _drive(fake, len(FACADE_SCRIPT))

    with pytest.raises(ScriptExhausted):
        fake.ask()


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_should_issue_unique_candidate_ids_across_asks(make_optimizer):
    """Given the scripted double or a real ABCOptimizer, when it is driven through N_CONTRACT_ASKS asks with
    full tells, then no candidate id repeats.

    Oracle: tell() identifies candidates by id (abc.py tell, task specification: telling an unknown or
    already-told candidate is a ValueError), so ids must be unique. Increasing order is base-class
    behaviour only and is not asserted.
    """
    opt = make_optimizer(_two_param_space())

    ids = _ids(_drive(opt, N_CONTRACT_ASKS))

    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_should_hand_out_points_of_the_closed_unit_cube(make_optimizer):
    """Given the scripted double or a real ABCOptimizer, when it is driven through N_CONTRACT_ASKS asks, then
    every candidate x is a d-vector inside [0, 1].

    Oracle: README 'Optimizers search the unit cube [0, 1]^d' and the Candidate docstring. Input: the script
    contains the corners (0, 0) and (1, 1), the closed bounds.
    """
    space = _two_param_space()
    opt = make_optimizer(space)

    xs = _stacked_x(_drive(opt, N_CONTRACT_ASKS))

    assert xs.shape[1] == space.dim
    assert xs.min() >= 0.0
    assert xs.max() <= 1.0


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_should_give_each_candidate_the_params_decoded_from_its_x(make_optimizer):
    """Given the scripted double or a real ABCOptimizer, when it is driven through N_CONTRACT_ASKS asks, then
    each candidate's params equal space.to_dicts of its x.

    Oracle: Candidate docstring (params as the user's function receives them for the point x) and the
    Space.to_dicts docstring.
    """
    space = _two_param_space()
    opt = make_optimizer(space)

    cands = _drive(opt, N_CONTRACT_ASKS)

    assert _params_of(cands) == _decoded_params_of(space, cands)


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_should_give_equal_hashable_keys_exactly_when_params_are_equal(make_optimizer):
    """Given the scripted double or a real ABCOptimizer on a space with distinct, non-equal choices, when it
    is driven through N_CONTRACT_ASKS asks, then two candidates share a key exactly when their params are
    equal, and the keys can be collected in a set.

    Oracle: Candidate docstring ('key, the hashable decoded identity used for caching') and Space.rows
    ('equal keys mean the user's function sees equal parameters'). Input: 10 candidates over 6 parameter
    sets make an equal pair certain by pigeonhole for ABC; the script has two different x decoding to
    {'k': 'a', 'n': 1}, so the check is not vacuous for the double either. The agreement set also requires
    an unequal pair, which seed 0 provides.
    """
    opt = make_optimizer(_two_param_space())

    cands = _drive(opt, N_CONTRACT_ASKS)

    assert _key_param_agreement(cands) == {(True, True), (False, False)}
    assert _distinct_key_count(cands) == _distinct_params_count(cands)


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_tell_should_raise_value_error_when_a_candidate_is_told_twice(make_optimizer):
    """Given the scripted double or a real ABCOptimizer whose last candidate was already told, when that
    candidate is told again, then ValueError is raised.

    Oracle: task specification 'telling an unknown/already-told candidate -> ValueError'. The keyword
    separates it from the length error of the same call.
    """
    opt = make_optimizer(_two_param_space())
    last = _drive(opt, N_CONTRACT_ASKS)[-1]

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([last], [stub_objective(last.params)])


@pytest.mark.parametrize(
    "unknown_id_after",
    [
        pytest.param(lambda last_id: last_id + 1, id="next-unissued-id"),
        pytest.param(lambda last_id: -1, id="negative"),
    ],
)
@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_tell_should_raise_value_error_when_the_candidate_id_was_never_issued(
    make_optimizer, unknown_id_after
):
    """Given the scripted double or a real ABCOptimizer after N_CONTRACT_ASKS asks, when a candidate with an
    id it never handed out is told, then ValueError is raised.

    Oracle: task specification 'telling an unknown candidate -> ValueError'. Inputs: the id right after the
    largest issued one (decisive against an 'id <= next id' check) and -1 below every issued id.
    """
    opt = make_optimizer(_two_param_space())
    last = _drive(opt, N_CONTRACT_ASKS)[-1]
    stranger = Candidate(unknown_id_after(last.id), last.x, last.params, last.key)

    with pytest.raises(ValueError, match="not asked"):
        opt.tell([stranger], [stub_objective(last.params)])


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_tell_should_raise_value_error_when_scores_and_candidates_differ_in_length(make_optimizer):
    """Given the scripted double or a real ABCOptimizer with one freshly asked, pending candidate, when it is
    told with an empty score list, then ValueError is raised.

    Oracle: task specification 'mismatched lengths -> ValueError'. Input: the candidate is pending, so the
    error cannot come from the already-told check; the keyword separates the two paths.
    """
    opt = make_optimizer(_two_param_space())
    _drive(opt, N_CONTRACT_ASKS)
    pending = opt.ask()

    with pytest.raises(ValueError, match="same length"):
        opt.tell(pending, [])


@pytest.mark.parametrize("make_optimizer", CONTRACT_OPTIMIZERS)
def test_optimizer_ask_should_raise_runtime_error_when_a_candidate_is_pending(make_optimizer):
    """Given the scripted double or a real ABCOptimizer past its initial batch with one untold candidate, when
    ask() is called again, then RuntimeError is raised.

    Oracle: task specification 'asking again with a pending candidate -> RuntimeError'.
    """
    opt = make_optimizer(_two_param_space())
    _drive(opt, N_CONTRACT_ASKS)
    opt.ask()

    with pytest.raises(RuntimeError):
        opt.ask()


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="smallest-budget"),
        pytest.param(2, id="inside-first-batch"),
        pytest.param(3, id="first-batch-boundary"),
        pytest.param(4, id="one-into-second-batch"),
    ],
)
def test_optimize_should_evaluate_exactly_max_evals_times_when_the_scripted_batch_is_larger(max_evals):
    """Given the scripted double with batches of 3, 1, 1 and distinct parameter sets, when optimize() runs
    with max_evals, then the Study has max_evals trials and the objective was called max_evals times.

    Oracle: optimize docstring ('max_evals counts the optimizer's evaluations') and the task specification
    'exact max_evals trials (even inside the initial batch)'. Inputs: both sides of the first batch boundary
    of 3; distinct parameter sets keep the cache out of the count.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space)
    mock_objective = _mock_objective()

    study = optimize(mock_objective, space, max_evals=max_evals, optimizer=fake)

    assert len(study.trials) == max_evals
    assert mock_objective.call_count == max_evals


def test_optimize_should_record_each_evaluated_params_with_the_score_the_objective_returned():
    """Given the scripted double and a mock objective returning 0.5, 0.25, 0.75, 1.5, 2.5, when optimize()
    runs all 5 evaluations, then the trials hold exactly the (params, score) pairs of the objective calls.

    Oracle: Trial fields (params, score) in study.py and the optimize docstring. Input: distinct, unsorted,
    non-zero scores, so a swapped, sorted or negated score is visible.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space)
    mock_objective = _mock_objective()

    study = optimize(mock_objective, space, max_evals=len(FACADE_RETURNS), optimizer=fake)

    assert _pair_counts(_trial_pairs(study)) == _pair_counts(_call_pairs(mock_objective))


def test_optimize_should_record_trial_x_that_decodes_to_the_trial_params():
    """Given the scripted double, when optimize() runs all 5 evaluations, then every trial's x decodes to that
    trial's params.

    Oracle: Trial carries the unit-cube point x of its params; Space.to_dicts docstring. Input: five distinct
    points, so an x copied from another candidate decodes to other params.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space)

    study = optimize(_mock_objective(), space, max_evals=len(FACADE_RETURNS), optimizer=fake)

    assert _decoded_trial_params(space, study) == _trial_params(study)


@pytest.mark.parametrize(
    "direction, expected_best",
    [pytest.param("min", 0.25, id="min-picks-0.25"), pytest.param("max", 2.5, id="max-picks-2.5")],
)
def test_optimize_should_return_a_study_in_the_requested_direction_when_the_optimizer_agrees(
    direction, expected_best
):
    """Given the scripted double built with `direction` and a mock objective returning 0.5, 0.25, 0.75, 1.5,
    2.5, when optimize() runs all 5 evaluations with the same direction, then the Study has that direction
    and its best score is the extreme for it.

    Oracle: optimize docstring ('minimise or maximise') and Study.best_trial (min or max by direction).
    Input: the best value is neither first nor last for 'min'.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space, direction)

    study = optimize(
        _mock_objective(), space, direction=direction, max_evals=len(FACADE_RETURNS), optimizer=fake
    )

    assert type(study) is Study
    assert study.direction == direction
    assert study.best_score == expected_best


def test_optimize_should_evaluate_the_candidates_of_the_injected_optimizer():
    """Given the scripted double passed as optimizer=, when optimize() runs 5 evaluations, then the evaluated
    parameter sets are exactly the five scripted ones.

    Oracle: optimize signature and docstring (the default ABCOptimizer is used only without an optimizer);
    the double is an Optimizer subclass on the same Space object, entering through the public parameter.
    Input: five distinct hand-decoded parameter sets that a fresh default ABC would not reproduce.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space)

    study = optimize(_mock_objective(), space, max_evals=len(FACADE_RETURNS), optimizer=fake)

    assert _trial_param_set(study) == {_params_tuple(p) for p in _scripted_params()}


def test_optimize_should_raise_value_error_without_evaluating_when_the_optimizer_uses_another_space():
    """Given the scripted double built on a Space with other parameters, when optimize() is called with it,
    then ValueError is raised, the objective is never called and the double is never asked.

    Oracle: task specification 'ValueError for mismatched optimizer space'; the check is an argument check
    of optimize(). Input: the other space differs in content as well as identity, so the case holds under
    either comparison; the directions agree, so only the space check can fire.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(Space({"y": Int(0, 1)}))
    mock_objective = _mock_objective()

    with pytest.raises(ValueError, match="same space"):
        optimize(mock_objective, space, max_evals=len(FACADE_RETURNS), optimizer=fake)

    assert mock_objective.call_count == 0
    assert fake.asked == []


@pytest.mark.parametrize(
    "fake_direction, optimize_direction",
    [
        pytest.param("max", "min", id="max-optimizer-min-call"),
        pytest.param("min", "max", id="min-optimizer-max-call"),
    ],
)
def test_optimize_should_raise_value_error_without_evaluating_when_the_optimizer_uses_another_direction(
    fake_direction, optimize_direction
):
    """Given the scripted double on the same Space object but the other direction, when optimize() is called
    with it, then ValueError is raised, the objective is never called and the double is never asked.

    Oracle: task specification 'ValueError for mismatched optimizer direction'. Input: both orientations;
    the space is the same object, so only the direction check can fire.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space, fake_direction)
    mock_objective = _mock_objective()

    with pytest.raises(ValueError, match="same space"):
        optimize(
            mock_objective, space, direction=optimize_direction, max_evals=len(FACADE_RETURNS), optimizer=fake
        )

    assert mock_objective.call_count == 0
    assert fake.asked == []


@pytest.mark.parametrize(
    "max_evals",
    [pytest.param(2, id="truncated-first-batch"), pytest.param(5, id="whole-script")],
)
def test_optimize_should_tell_exactly_the_evaluated_candidates_with_their_scores(max_evals):
    """Given the scripted double with batches of 3, 1, 1, when optimize() runs max_evals evaluations, then the
    candidates told back are exactly the evaluated ones, each with the score the objective returned for it.

    Oracle: Optimizer docstring ('tell() reports their scores') and the task specification (ABC init may be
    told partially). Input: max_evals=2 cuts the first batch, the case where what goes to the optimizer is
    not visible in the Study, so the tell calls themselves are checked; an unevaluated candidate told with
    any score would show up. 5 uses the whole script.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space)
    mock_objective = _mock_objective()

    optimize(mock_objective, space, max_evals=max_evals, optimizer=fake)

    assert _pair_counts(_told_pairs_of_fake(fake)) == _pair_counts(_call_pairs(mock_objective))


def test_optimize_should_tell_scores_unchanged_in_the_user_direction_when_maximising():
    """Given the scripted double with direction 'max' and a mock objective returning 0.5, 0.25, 0.75, 1.5,
    2.5, when optimize() runs all 5 evaluations, then every told score equals the returned score, not negated.

    Oracle: Optimizer docstring 'Scores are given in the user's direction; internally everything is
    minimised'. Input: 'max' is the direction where a facade could wrongly negate; no score is 0.0, so
    negation would be visible.
    """
    space = _two_param_space()
    fake = _scripted_facade_optimizer(space, "max")
    mock_objective = _mock_objective()

    optimize(mock_objective, space, direction="max", max_evals=len(FACADE_RETURNS), optimizer=fake)

    assert _pair_counts(_told_pairs_of_fake(fake)) == _pair_counts(_call_pairs(mock_objective))


@pytest.mark.parametrize("direction", ["min", "max"])
def test_optimize_should_tell_the_real_abc_only_the_evaluated_part_of_its_initial_batch(direction):
    """Given a real ABCOptimizer with colony_size=6 (initial batch of 3) and a spy on its tell, when
    optimize() runs 2 evaluations, then the run ends with 2 trials and the ABC was told exactly those trials'
    params and scores.

    Oracle: task specification (exact max_evals; ABC init may be told partially) and the Optimizer docstring.
    Input: the same truncation boundary (2 < 3) the double was used for, now with the real component, so a
    difference between the double and ABC on the partial-batch path is caught. Seed 0 is arbitrary.
    """
    space = _two_param_space()
    opt = ABCOptimizer(space, direction, colony_size=CONTRACT_COLONY_SIZE, seed=ANY_SEED)

    with mock.patch.object(opt, "tell", wraps=opt.tell) as spy:
        study = optimize(stub_objective, space, direction=direction, max_evals=2, optimizer=opt)

    assert len(study.trials) == 2
    assert _pair_counts(_told_pairs_of_spy(spy)) == _pair_counts(_trial_pairs(study))


def test_optimize_should_tell_the_real_abc_every_trial_once_when_running_past_the_initial_batch():
    """Given a real ABCOptimizer with colony_size=6 and a spy on its tell, when optimize() runs 12 evaluations
    on a 6-combination space, then the told candidates have 12 distinct ids and their (params, score) pairs
    are exactly the trials' pairs, cached repeats included.

    Oracle: Optimizer docstring (every handed-out candidate's score is told), README/optimize docstring
    (a cached repeat reuses its earlier score). Input: 12 goes past the initial batch of 3 into single
    employed and onlooker asks; 12 trials over 6 parameter sets force cache hits. Seed 0 is arbitrary.
    """
    space = _two_param_space()
    opt = ABCOptimizer(space, colony_size=CONTRACT_COLONY_SIZE, seed=ANY_SEED)

    with mock.patch.object(opt, "tell", wraps=opt.tell) as spy:
        study = optimize(stub_objective, space, max_evals=12, optimizer=opt)

    assert len(set(_told_ids_of_spy(spy))) == 12
    assert _pair_counts(_told_pairs_of_spy(spy)) == _pair_counts(_trial_pairs(study))


SPY_SEED = 1


STUB_SCORE = 2.5


ONE_POINT_SCORE = 0.7


IRRELEVANT_SCORE = 1.0


MIXED_INVALID_SCORES = [math.nan, math.inf, 3.0, -math.inf, 1.0, 2.0]


MIXED_INVALID_FAILED = [True, True, False, True, False, False]


def _unit_float_space():
    """One continuous parameter: every candidate is a fresh point, so trial order is the only structure."""
    return Space({"x": Float(0.0, 1.0)})


def _tiny_space():
    """Four points (2 integers x 2 choices), far fewer than the budget, so parameter sets must repeat."""
    return Space({"i": Int(0, 1), "c": Categorical(["a", "b"])})


def _mixed_space():
    """Every parameter kind; choices 3, 5, 7 are disjoint from the category indices 0..2."""
    return Space(
        {
            "lr": Float(1e-4, 1e-2, log=True),
            "n": Int(16, 256),
            "act": Categorical(["relu", "tanh"]),
            "k": Categorical([3, 5, 7]),
        }
    )


def _real_objective(params):
    """A cheap real objective that depends on all four parameters of the mixed space."""
    sign = 1 if params["act"] == "relu" else -1
    return params["lr"] * params["n"] + sign * params["k"]


def _counting_stub_objective():
    """Stub returning 0.0, 1.0, 2.0, ... so a reused score is told apart from a repeated call."""
    counter = itertools.count()
    return Mock(side_effect=lambda params: float(next(counter)))


def _params_key(params):
    """Hashable form of a params dict."""
    return tuple(sorted(params.items()))


def _first_occurrence(trials):
    """True for each trial whose params did not appear in an earlier trial."""
    return [t.params not in [u.params for u in trials[:i]] for i, t in enumerate(trials)]


def _first_score_by_params(trials):
    """Score of the first trial of every distinct params dict."""
    return {_params_key(t.params): t.score for t in reversed(trials)}


def test_optimize_should_report_the_stub_score_in_every_trial_and_the_best_when_the_objective_is_constant():
    """Given a stub objective that always returns 2.5, when optimize runs 30 evaluations, then every trial
    score and best_score are 2.5 and best_params is one of the trial params.

    Oracle: the stub's programmed value (README Quick start: best_score comes from the user's scores).
    2.5 is non-integer and differs from its fitness 1/(1+2.5), so truncation or a fitness leak would show;
    30 evaluations go past the 5-candidate initial batch into single-candidate asks.
    """
    space = _unit_float_space()

    def stub_objective(params):
        return STUB_SCORE

    study = optimize(stub_objective, space, max_evals=30, seed=ANY_SEED)

    assert len(study.trials) == 30
    assert study.scores.tolist() == [STUB_SCORE] * 30
    assert [t.failed for t in study.trials] == [False] * 30
    assert study.best_score == STUB_SCORE
    assert type(study.best_score) is float
    assert study.best_params in [t.params for t in study.trials]


def test_optimize_should_fail_every_trial_when_the_objective_always_returns_nan():
    """Given a stub objective that always returns NaN, when optimize runs 150 evaluations, then the run
    finishes and every trial is failed with a NaN score.

    Oracle: README Behaviour (NaN counts as the worst value) and the Trial.failed docstring. 150 evaluations
    exceed colony_size//2 * limit, so the employed, onlooker (all-zero fitness) and scout steps all run.
    """
    space = _unit_float_space()

    def stub_objective(params):
        return math.nan

    study = optimize(stub_objective, space, max_evals=150, seed=ANY_SEED)

    assert len(study.trials) == 150
    assert [t.failed for t in study.trials] == [True] * 150
    assert np.isnan(study.scores).tolist() == [True] * 150


@pytest.mark.parametrize("attribute", ["best_trial", "best_score", "best_params"])
def test_study_should_raise_value_error_for_best_when_every_trial_returned_nan(attribute):
    """Given a study whose objective always returned NaN, when a best accessor is read, then ValueError.

    Oracle: study.py best_trial raises ValueError when no trial returned a finite score; best_score and
    best_params are separate public entry points, so each is a case.
    """

    def stub_objective(params):
        return math.nan

    study = optimize(stub_objective, _unit_float_space(), max_evals=12, seed=ANY_SEED)

    with pytest.raises(ValueError):
        getattr(study, attribute)


def test_optimize_should_mark_non_finite_scores_failed_and_ignore_minus_inf_when_direction_is_min():
    """Given a stub returning nan, inf, 3.0, -inf, 1.0, 2.0 in turn and cache=False, when optimize runs six
    evaluations minimising, then exactly the non-finite trials fail and the best is 1.0 at trial 4.

    Oracle: hand-derived from the sequence and README Behaviour ("NaN or infinity counts as the worst
    possible value"). -inf is decisive: a plain minimum over raw scores would pick trial 3. Six evaluations
    are the 5-candidate batch plus one employed candidate, so both tell shapes receive the sequence.
    """
    stub_objective = Mock(side_effect=MIXED_INVALID_SCORES)

    study = optimize(
        stub_objective, _unit_float_space(), direction="min", max_evals=6, seed=ANY_SEED, cache=False
    )

    assert [t.failed for t in study.trials] == MIXED_INVALID_FAILED
    assert study.best_score == 1.0
    assert study.best_trial.number == 4
    assert stub_objective.call_count == 6
    assert study.n_calls == 6


def test_optimize_should_pick_the_largest_finite_score_and_ignore_plus_inf_when_direction_is_max():
    """Given the same nan, inf, 3.0, -inf, 1.0, 2.0 stub and cache=False, when optimize maximises over six
    evaluations, then the failed trials are the same and the best is 3.0 at trial 2.

    Oracle: hand-derived from the sequence; README Behaviour ("worst possible value for the chosen
    direction"). +inf is decisive here: a plain maximum over raw scores would pick trial 1.
    """
    stub_objective = Mock(side_effect=MIXED_INVALID_SCORES)

    study = optimize(
        stub_objective, _unit_float_space(), direction="max", max_evals=6, seed=ANY_SEED, cache=False
    )

    assert [t.failed for t in study.trials] == MIXED_INVALID_FAILED
    assert study.best_score == 3.0
    assert study.best_trial.number == 2
    assert stub_objective.call_count == 6


@pytest.mark.parametrize(
    ("direction", "expected_score", "expected_number"),
    [pytest.param("min", 1.0, 1, id="min-picks-middle"), pytest.param("max", 3.0, 0, id="max-picks-first")],
)
def test_optimize_should_pick_the_hand_derived_best_when_the_budget_truncates_the_initial_batch(
    direction, expected_score, expected_number
):
    """Given a stub returning exactly 3.0, 1.0, 2.0 and max_evals=3 < colony_size//2, when optimize runs,
    then three trials carry those scores and the best follows the direction.

    Oracle: the sequence itself and the exact max_evals budget (facade.py docstring). A fourth call would
    raise StopIteration from the exhausted stub, so over-calling fails the test; min in the middle and max
    first defeat 'first' and 'last' shortcuts.
    """
    stub_objective = Mock(side_effect=[3.0, 1.0, 2.0])

    study = optimize(
        stub_objective, _unit_float_space(), direction=direction, max_evals=3, seed=ANY_SEED, cache=False
    )

    assert study.scores.tolist() == [3.0, 1.0, 2.0]
    assert stub_objective.call_count == 3
    assert study.best_score == expected_score
    assert study.best_trial.number == expected_number


@pytest.mark.parametrize(
    ("side_effect", "error", "expected_calls"),
    [
        pytest.param(ValueError("first"), ValueError, 1, id="first-call-user-value-error"),
        pytest.param([1.0, 2.0, RuntimeError("boom")], RuntimeError, 3, id="inside-initial-batch"),
        pytest.param([1.0] * 5 + [RuntimeError("boom")], RuntimeError, 6, id="first-employed-ask"),
    ],
)
def test_optimize_should_propagate_the_objective_exception_and_stop_calling_when_the_objective_raises(
    side_effect, error, expected_calls
):
    """Given a stub that returns some scores and then raises, when optimize runs with a larger budget, then
    the user's own exception reaches the caller and no call follows it.

    Not documented in README or docstrings: records current behaviour. max_evals=10 exceeds every
    expected_calls, so stopping comes from the exception, not the budget; the raise is placed on the first
    call, inside the batched initial population and on the first single-candidate ask.
    """
    stub_objective = Mock(side_effect=side_effect)

    with pytest.raises(error):
        optimize(stub_objective, _unit_float_space(), max_evals=10, seed=ANY_SEED, cache=False)

    assert stub_objective.call_count == expected_calls


def test_optimize_should_finish_with_a_valid_best_when_invalid_scores_come_before_valid_ones():
    """Given a stub returning nan, nan, 5.0, 4.0, 6.0, 7.0, 3.0 and cache=False, when optimize runs seven
    evaluations, then the first two trials fail and the best is the valid 3.0 of the last call.

    Oracle: hand-derived from the sequence (finite minimum 3.0 at index 6); README Behaviour. This is the
    invalid-then-valid path: the last value lands on a single-candidate ask after the initial batch.
    """
    stub_objective = Mock(side_effect=[math.nan, math.nan, 5.0, 4.0, 6.0, 7.0, 3.0])

    study = optimize(stub_objective, _unit_float_space(), max_evals=7, seed=ANY_SEED, cache=False)

    assert [t.failed for t in study.trials] == [True, True, False, False, False, False, False]
    assert study.best_trial.failed is False
    assert study.best_score == 3.0
    assert study.best_trial.number == 6
    assert stub_objective.call_count == 7


def test_optimize_should_call_the_objective_once_and_cache_the_rest_when_the_space_has_one_point():
    """Given a one-point space and a mock objective returning 0.7, when optimize runs 20 evaluations with
    cache=True, then the objective is called once and every later trial is cached with 0.7.

    Postcondition besides the returned Study: each cached trial causes no call of the objective. Oracle:
    README Behaviour (cache reuses the earlier score), facade.py docstring, Study.n_calls docstring.
    Trials 1..4 repeat inside the initial batch and 5..19 on single asks.
    """
    mock_objective = Mock(return_value=ONE_POINT_SCORE)

    study = optimize(mock_objective, Space({"c": Categorical(["a"])}), max_evals=20, seed=ANY_SEED)

    assert mock_objective.call_count == 1
    assert study.n_calls == 1
    assert mock_objective.call_args.args[0] == {"c": "a"}
    assert [t.cached for t in study.trials] == [False] + [True] * 19
    assert study.scores.tolist() == [ONE_POINT_SCORE] * 20


def test_optimize_should_call_the_objective_once_per_distinct_parameter_set_when_cache_is_true():
    """Given a four-point space and a counting stub returning a new value per call, when optimize runs 50
    evaluations with cache=True, then calls equal distinct parameter sets, follow trial order, and every
    repeat is cached with the score of its first occurrence.

    Oracle: README Behaviour (cache), space.rows docstring (equal keys mean equal parameters), Study.n_calls
    docstring. The per-call value is decisive: a constant stub cannot tell a reused score from a new call.
    """
    mock_objective = _counting_stub_objective()

    study = optimize(mock_objective, _tiny_space(), max_evals=50, seed=ANY_SEED, cache=True)

    first = _first_occurrence(study.trials)
    first_score = _first_score_by_params(study.trials)
    assert len(study.trials) == 50
    assert mock_objective.call_count == study.n_calls
    assert mock_objective.call_count == len(first_score)
    assert mock_objective.call_count <= 4
    assert [c.args[0] for c in mock_objective.call_args_list] == [
        t.params for t in study.trials if not t.cached
    ]
    assert [t.cached for t in study.trials] == [not f for f in first]
    assert [t.score for t in study.trials] == [first_score[_params_key(t.params)] for t in study.trials]


def test_optimize_should_call_the_objective_on_every_trial_when_cache_is_false():
    """Given the same four-point space and cache=False, when optimize runs 50 evaluations, then the
    objective is called 50 times and no trial is cached.

    Pair of the cache=True test: only the flag differs. Oracle: facade.py docstring (only cache=True reuses
    a repeated set) and Study.n_calls docstring.
    """
    mock_objective = Mock(return_value=IRRELEVANT_SCORE)

    study = optimize(mock_objective, _tiny_space(), max_evals=50, seed=ANY_SEED, cache=False)

    assert mock_objective.call_count == 50
    assert study.n_calls == 50
    assert [t.cached for t in study.trials] == [False] * 50


def test_optimize_should_send_typed_in_range_parameters_to_the_objective_when_a_spy_records_the_calls():
    """Given a spy wrapping a real objective over a mixed space, when optimize runs 150 evaluations, then
    every recorded argument has all names, Python types and values inside the declared ranges or choices.

    Oracle: README Search space table (Float in [low, high], Int low..high inclusive, Categorical one of
    choices) and space.rows docstring (params have Python types). type(...) is used so whole floats and
    numpy scalars fail; choices 3, 5, 7 catch indices sent instead of choices.
    """
    space = _mixed_space()
    spy_objective = Mock(wraps=_real_objective)

    optimize(spy_objective, space, max_evals=150, seed=SPY_SEED)

    recorded = [c.args[0] for c in spy_objective.call_args_list]
    assert {frozenset(p) for p in recorded} == {frozenset(space.names)}
    assert {type(p["lr"]) for p in recorded} == {float}
    assert {1e-4 <= p["lr"] <= 1e-2 for p in recorded} == {True}
    assert {type(p["n"]) for p in recorded} == {int}
    assert {16 <= p["n"] <= 256 for p in recorded} == {True}
    assert {p["act"] for p in recorded} <= {"relu", "tanh"}
    assert {p["k"] for p in recorded} <= {3, 5, 7}
    assert {type(p["k"]) for p in recorded} == {int}


def test_optimize_should_keep_the_real_scores_and_count_every_call_when_a_spy_wraps_the_objective():
    """Given a spy wrapping a real objective, when optimize runs with cache=True, then each trial score is
    the real objective's value, recorded calls equal n_calls, and no parameter set is sent twice.

    Oracle: a spy only records (P-T3-12), so scores must equal the real function; README Behaviour (cache)
    and Study.n_calls docstring for the call accounting.
    """
    spy_objective = Mock(wraps=_real_objective)

    study = optimize(spy_objective, _mixed_space(), max_evals=150, seed=SPY_SEED)

    recorded = [c.args[0] for c in spy_objective.call_args_list]
    assert [t.score for t in study.trials] == [_real_objective(t.params) for t in study.trials]
    assert len(recorded) == study.n_calls
    assert spy_objective.call_count == study.n_calls
    assert len({_params_key(p) for p in recorded}) == len(recorded)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(np.float64(2.5), 2.5, id="float64"),
        pytest.param(np.float32(2.5), 2.5, id="float32-exact"),
        pytest.param(np.int64(2), 2.0, id="int64"),
        pytest.param(np.float32(0.1), 0.10000000149011612, id="float32-widened-not-rounded"),
    ],
)
def test_optimize_should_store_python_float_scores_when_the_objective_returns_numpy_scalars(value, expected):
    """Given a real objective returning a numpy scalar, when optimize runs 12 evaluations, then every score
    is a Python float equal to the widened value and no trial fails.

    Support for np.float32 and np.int64 rests on the float() conversion in facade.py, not on README text.
    np.float32(0.1) is decisive: its exact value 0.10000000149011612 must be kept, not rounded to 0.1.
    12 evaluations cross from the initial batch into single asks.
    """

    def numpy_objective(params):
        return value

    study = optimize(numpy_objective, _unit_float_space(), max_evals=12, seed=ANY_SEED)

    assert study.best_score == expected
    assert type(study.best_score) is float
    assert {type(t.score) for t in study.trials} == {float}
    assert study.scores.dtype == np.float64
    assert [t.failed for t in study.trials] == [False] * 12


def test_study_should_expose_float64_scores_with_raw_non_finite_values_when_some_trials_failed():
    """Given a stub returning nan, 2.0, inf, 1.0, -inf, nan, 4.0 and cache=False, when optimize runs seven
    evaluations, then Study.scores is a float64 array whose NaN entries are the NaN trials and whose
    non-finite entries are exactly the failed trials.

    Oracle: study.py (scores built as float64, failed means not finite) and the programmed sequence. inf
    trials are failed but not NaN, so the NaN mask and the failed mask must differ at trials 2 and 4.
    """
    stub_objective = Mock(side_effect=[math.nan, 2.0, math.inf, 1.0, -math.inf, math.nan, 4.0])

    study = optimize(stub_objective, _unit_float_space(), max_evals=7, seed=ANY_SEED, cache=False)

    scores = study.scores
    assert type(scores) is np.ndarray
    assert scores.dtype == np.float64
    assert scores.shape == (7,)
    assert np.isnan(scores).tolist() == [True, False, False, False, False, True, False]
    assert (~np.isfinite(scores)).tolist() == [t.failed for t in study.trials]
    assert [t.failed for t in study.trials] == [True, False, True, False, True, True, False]
    assert scores[[1, 3, 6]].tolist() == [2.0, 1.0, 4.0]
