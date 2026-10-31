"""Model-based tests of the ABCOptimizer ask/tell protocol: a Mealy machine model and the W method.

Model (rule P-T7-42). The ask/tell protocol of ABCOptimizer(Space({"x": Float(0, 1)}), colony_size=4,
limit=1) is modelled as a Mealy machine (S, s0, X, Y, delta, lambda) held in TRANSITIONS, which maps each
(state, input) to (next state, output), or to INAPPLICABLE where the input cannot be formed:

- S = STATES: INIT (nothing asked), INIT_WAIT2 (both initial candidates pending), INIT_WAIT1 (one
  initial candidate pending), IDLE (initialised, nothing pending), WAIT (initialised, one candidate
  pending); s0 = INIT.
- X = INPUTS, as the driver _Driver forms them from public ask/tell calls only (rule P-T7-48):
  ask = opt.ask(); tell_first = tell the oldest pending candidate; tell_all = tell every pending candidate;
  tell_stale = tell one candidate that is not pending (the last told one, or a Candidate with the never
  issued id NEVER_ISSUED_ID while nothing was told yet); tell_mismatch = tell the pending candidates with one
  score too many (tell([], [score]) when nothing is pending).
- Y = OUTPUTS: 'batch:<n>' for an ask that returns n candidates, the exception class name for a refused
  call, 'ok' for an accepted tell.

Oracles for the table: the task specification and README "Ask and tell" (init asks colony_size // 2 = 2
sources as one batch, which may be told partially; afterwards one candidate per ask in the employed,
onlooker and scout phases; asking again with a pending candidate raises RuntimeError; telling an unknown or
already-told candidate raises ValueError; mismatched lengths raise ValueError), the ABC module docstring
("Only the initial population is handed out as a batch") and the Optimizer docstring ("ask() returns the
candidates that can be evaluated now; tell() reports their scores").

Refused and inapplicable inputs (rule P-T7-49). A refused input (RuntimeError from ask, ValueError from
tell_stale or tell_mismatch) is observable and keeps the state: the self-loop convention of the rule
source, which the library documents nowhere; the rejected-input test (name ending in
_leave_protocol_state_unchanged_when_input_is_rejected) checks that the implementation meets it, because
the W-method suite is complete only under it. tell_first and tell_all are INAPPLICABLE in INIT and IDLE
(nothing is pending, so there is nothing to tell): the driver then makes no call, outputs 'n/a' and the
model state stays unchanged.

Minimal model (rule P-T7-50). INIT_WAIT1 and WAIT are equivalent, so W is built on the 4-state quotient
MIN_TRANSITIONS (INIT, INIT_WAIT2, PENDING1, IDLE); CLASS_OF maps each state to its class.

Reset (rule P-T7-53). Every input sequence runs on a freshly built optimizer, so it starts in s0.

Unspecified, not asserted:
- that a refused call keeps the protocol state is not written in README, CHANGELOG or a docstring; it is
  checked only as the model's refused-input precondition;
- whether a refused call consumes random draws: no x values are compared after a refusal;
- tell() with a list that mixes pending and non-pending candidates: the driver never sends one;
- the range of candidate ids: NEVER_ISSUED_ID = -1 is used because no id handed out in these short runs is
  negative, not because a spec source forbids negative ids;
- scores do not influence protocol outputs in any spec source, so every tell uses IRRELEVANT_SCORE;
- the protocol for colony_size > 4: the model is for colony_size = 4 only.

Seeds: ANY_SEED is fixed for determinism; no protocol output depends on random draws. SAME_SEED is the seed
shared by the instances of the reset tests; any fixed value works because only runs are compared.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: The W-method sequences of the Mealy machine model of the ask/tell protocol.
"""

import itertools
from types import MappingProxyType

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Float, Space

ANY_SEED = 0
SAME_SEED = 7
IRRELEVANT_SCORE = 0.0
NEVER_ISSUED_ID = -1
COLONY_SIZE = 4
LIMIT = 1

INIT = "INIT"
INIT_WAIT2 = "INIT_WAIT2"
INIT_WAIT1 = "INIT_WAIT1"
IDLE = "IDLE"
WAIT = "WAIT"
PENDING1 = "PENDING1"

ASK = "ask"
TELL_FIRST = "tell_first"
TELL_ALL = "tell_all"
TELL_STALE = "tell_stale"
TELL_MISMATCH = "tell_mismatch"

BATCH_OF_ONE = "batch:1"
BATCH_OF_TWO = "batch:2"
REFUSED_ASK = "RuntimeError"
REFUSED_TELL = "ValueError"
ACCEPTED = "ok"
NOT_APPLICABLE = "n/a"
INAPPLICABLE = "INAPPLICABLE"

STATES = (INIT, INIT_WAIT2, INIT_WAIT1, IDLE, WAIT)
S0 = INIT
INPUTS = (ASK, TELL_FIRST, TELL_ALL, TELL_STALE, TELL_MISMATCH)
OUTPUTS = (BATCH_OF_ONE, BATCH_OF_TWO, REFUSED_ASK, REFUSED_TELL, ACCEPTED)

TRANSITIONS = MappingProxyType(
    {
        (INIT, ASK): (INIT_WAIT2, BATCH_OF_TWO),
        (INIT, TELL_FIRST): INAPPLICABLE,
        (INIT, TELL_ALL): INAPPLICABLE,
        (INIT, TELL_STALE): (INIT, REFUSED_TELL),
        (INIT, TELL_MISMATCH): (INIT, REFUSED_TELL),
        (INIT_WAIT2, ASK): (INIT_WAIT2, REFUSED_ASK),
        (INIT_WAIT2, TELL_FIRST): (INIT_WAIT1, ACCEPTED),
        (INIT_WAIT2, TELL_ALL): (IDLE, ACCEPTED),
        (INIT_WAIT2, TELL_STALE): (INIT_WAIT2, REFUSED_TELL),
        (INIT_WAIT2, TELL_MISMATCH): (INIT_WAIT2, REFUSED_TELL),
        (INIT_WAIT1, ASK): (INIT_WAIT1, REFUSED_ASK),
        (INIT_WAIT1, TELL_FIRST): (IDLE, ACCEPTED),
        (INIT_WAIT1, TELL_ALL): (IDLE, ACCEPTED),
        (INIT_WAIT1, TELL_STALE): (INIT_WAIT1, REFUSED_TELL),
        (INIT_WAIT1, TELL_MISMATCH): (INIT_WAIT1, REFUSED_TELL),
        (IDLE, ASK): (WAIT, BATCH_OF_ONE),
        (IDLE, TELL_FIRST): INAPPLICABLE,
        (IDLE, TELL_ALL): INAPPLICABLE,
        (IDLE, TELL_STALE): (IDLE, REFUSED_TELL),
        (IDLE, TELL_MISMATCH): (IDLE, REFUSED_TELL),
        (WAIT, ASK): (WAIT, REFUSED_ASK),
        (WAIT, TELL_FIRST): (IDLE, ACCEPTED),
        (WAIT, TELL_ALL): (IDLE, ACCEPTED),
        (WAIT, TELL_STALE): (WAIT, REFUSED_TELL),
        (WAIT, TELL_MISMATCH): (WAIT, REFUSED_TELL),
    }
)

MIN_STATES = (INIT, INIT_WAIT2, PENDING1, IDLE)
MIN_TRANSITIONS = MappingProxyType(
    {
        (INIT, ASK): (INIT_WAIT2, BATCH_OF_TWO),
        (INIT, TELL_FIRST): INAPPLICABLE,
        (INIT, TELL_ALL): INAPPLICABLE,
        (INIT, TELL_STALE): (INIT, REFUSED_TELL),
        (INIT, TELL_MISMATCH): (INIT, REFUSED_TELL),
        (INIT_WAIT2, ASK): (INIT_WAIT2, REFUSED_ASK),
        (INIT_WAIT2, TELL_FIRST): (PENDING1, ACCEPTED),
        (INIT_WAIT2, TELL_ALL): (IDLE, ACCEPTED),
        (INIT_WAIT2, TELL_STALE): (INIT_WAIT2, REFUSED_TELL),
        (INIT_WAIT2, TELL_MISMATCH): (INIT_WAIT2, REFUSED_TELL),
        (PENDING1, ASK): (PENDING1, REFUSED_ASK),
        (PENDING1, TELL_FIRST): (IDLE, ACCEPTED),
        (PENDING1, TELL_ALL): (IDLE, ACCEPTED),
        (PENDING1, TELL_STALE): (PENDING1, REFUSED_TELL),
        (PENDING1, TELL_MISMATCH): (PENDING1, REFUSED_TELL),
        (IDLE, ASK): (PENDING1, BATCH_OF_ONE),
        (IDLE, TELL_FIRST): INAPPLICABLE,
        (IDLE, TELL_ALL): INAPPLICABLE,
        (IDLE, TELL_STALE): (IDLE, REFUSED_TELL),
        (IDLE, TELL_MISMATCH): (IDLE, REFUSED_TELL),
    }
)
CLASS_OF = MappingProxyType(
    {INIT: INIT, INIT_WAIT2: INIT_WAIT2, INIT_WAIT1: PENDING1, WAIT: PENDING1, IDLE: IDLE}
)

V5 = MappingProxyType(
    {
        INIT: (),
        INIT_WAIT2: (ASK,),
        INIT_WAIT1: (ASK, TELL_FIRST),
        IDLE: (ASK, TELL_ALL),
        WAIT: (ASK, TELL_ALL, ASK),
    }
)
V4 = MappingProxyType(
    {
        INIT: (),
        INIT_WAIT2: (ASK,),
        PENDING1: (ASK, TELL_FIRST),
        IDLE: (ASK, TELL_ALL),
    }
)
W = ((ASK,), (TELL_FIRST, ASK))
EXTRA_STATES_M = 1

TOUR = (
    (ASK, TELL_ALL),
    (ASK, TELL_FIRST, TELL_ALL),
    (
        TELL_STALE,
        TELL_MISMATCH,
        ASK,
        ASK,
        TELL_STALE,
        TELL_MISMATCH,
        TELL_FIRST,
        ASK,
        TELL_STALE,
        TELL_MISMATCH,
        TELL_FIRST,
        ASK,
        ASK,
        TELL_STALE,
        TELL_MISMATCH,
        TELL_FIRST,
        ASK,
        TELL_ALL,
        TELL_STALE,
        TELL_MISMATCH,
    ),
)
TOUR_LAST_EDGE = (IDLE, TELL_MISMATCH)

W_METHOD_ASSUMPTIONS = (
    (
        1,
        "the implementation behaves like an unknown FSM with at most n + m = 4 + 1 protocol states",
        "ASSUMED, not testable black-box; justified by the spec, where protocol outputs depend only on what "
        "is pending and on whether the initial batch is told; the suite adds V.X^2.W (m = 1) as margin",
        ("test_abc_optimizer_should_conform_to_model_when_driven_across_several_cycles",),
    ),
    (
        2,
        "the system has a reliable reset: building a fresh optimizer",
        "checked",
        (
            "test_abc_optimizer_should_hand_out_identical_first_batches_when_built_with_same_seed",
            "test_abc_optimizer_should_not_share_state_when_same_seed_instances_are_interleaved",
        ),
    ),
    (
        3,
        "W is a characterising set of the minimal model",
        "checked",
        ("test_w_set_should_separate_every_pair_of_minimal_states_without_an_na_step",),
    ),
    (
        4,
        "V is a state cover",
        "checked",
        ("test_state_cover_should_reach_every_state_and_contain_epsilon",),
    ),
)


def _step(table, state, symbol):
    """One model step; an inapplicable input outputs 'n/a' and keeps the state."""
    entry = table[(state, symbol)]
    if entry == INAPPLICABLE:
        return state, NOT_APPLICABLE
    return entry


def _run_on_model(inputs, table=TRANSITIONS, s0=S0):
    """Model outputs of an input sequence started in s0."""
    state, outputs = s0, []
    for symbol in inputs:
        state, output = _step(table, state, symbol)
        outputs.append(output)
    return tuple(outputs)


def _walk(table, inputs, s0=S0):
    """Final state, fired applicable edges in order and whether an inapplicable input was met."""
    state, fired, met_inapplicable = s0, [], False
    for symbol in inputs:
        met_inapplicable = met_inapplicable or table[(state, symbol)] == INAPPLICABLE
        if table[(state, symbol)] != INAPPLICABLE:
            fired.append((state, symbol))
        state, _ = _step(table, state, symbol)
    return state, tuple(fired), met_inapplicable


def _applicable_edges(table):
    """Every (state, input) of the table that is not INAPPLICABLE."""
    return frozenset(key for key, entry in table.items() if entry != INAPPLICABLE)


def _inapplicable_cells(table):
    """Every (state, input) of the table that is INAPPLICABLE."""
    return frozenset(key for key, entry in table.items() if entry == INAPPLICABLE)


def _malformed_cells(table, states, inputs, outputs):
    """Cells of states x inputs that are missing or hold neither INAPPLICABLE nor (state, output)."""
    malformed = []
    for key in itertools.product(states, inputs):
        entry = table.get(key, "missing")
        well_formed = entry == INAPPLICABLE or (
            isinstance(entry, tuple) and entry[0] in states and entry[1] in outputs
        )
        if not well_formed:
            malformed.append(key)
    return malformed


def _cover_report(table, cover, s0=S0):
    """For each state of the cover: (state reached by its sequence, whether an inapplicable input was met)."""
    report = {}
    for state, sequence in cover.items():
        reached, _, met_inapplicable = _walk(table, sequence, s0)
        report[state] = (reached, met_inapplicable)
    return report


def _reached_without_inapplicable(states):
    """The cover report expected when every state is reached by its own sequence without 'n/a'."""
    return {s: (s, False) for s in states}


def _set_shape(sequences):
    """(number of sequences, length of the longest one)."""
    return len(sequences), max(len(seq) for seq in sequences)


def _test_names_in_module():
    """Names of the test functions defined in this module."""
    return {name for name in globals() if name.startswith("test_")}


def _assumption_summary(assumptions):
    """(numbers, whether each is only ASSUMED, union of the test names given as checks)."""
    numbers = tuple(number for number, _, _, _ in assumptions)
    assumed = tuple(status.startswith("ASSUMED") for _, _, status, _ in assumptions)
    named = {name for _, _, _, names in assumptions for name in names}
    return numbers, assumed, named


def _edges_fired_by(table, sequences, s0=S0):
    """Union of the applicable edges fired by sequences, each started in s0."""
    fired = set()
    for sequence in sequences:
        fired.update(_walk(table, sequence, s0)[1])
    return frozenset(fired)


def _fire_count(table, sequences, edge, s0=S0):
    """How often edge fires over all sequences, each started in s0."""
    return sum(_walk(table, sequence, s0)[1].count(edge) for sequence in sequences)


def _edge_label(state, symbol, table=TRANSITIONS):
    """The 'x / y' label of an applicable edge."""
    return f"{symbol} / {table[(state, symbol)][1]}"


def _edge_id(state, symbol, table=TRANSITIONS):
    """'qi --x / y--> qj' for an applicable edge."""
    return f"{state} --{_edge_label(state, symbol, table)}--> {table[(state, symbol)][0]}"


def _refine(states, table):
    """Moore partition refinement to the coarsest partition of equivalent states, as a set of frozensets."""
    block_of = {s: tuple(_step(table, s, x)[1] for x in INPUTS) for s in states}
    while True:
        signature = {s: (block_of[s], tuple(block_of[_step(table, s, x)[0]] for x in INPUTS)) for s in states}
        if len(set(signature.values())) == len(set(block_of.values())):
            break
        block_of = signature
    blocks = {}
    for s in states:
        blocks.setdefault(block_of[s], set()).add(s)
    return {frozenset(block) for block in blocks.values()}


def _quotient(table, states, class_of):
    """The table with every state replaced by its class, read from the first state of each class."""
    quotient = {}
    for s in states:
        for x in INPUTS:
            entry = table[(s, x)]
            merged = entry if entry == INAPPLICABLE else (class_of[entry[0]], entry[1])
            quotient.setdefault((class_of[s], x), merged)
    return quotient


def _row_with_successor_class(table, state, class_of):
    """For every input: (output, class of the successor) from state."""
    return tuple((_step(table, state, x)[1], class_of[_step(table, state, x)[0]]) for x in INPUTS)


def _first_separation(table, s, t, suffixes):
    """First suffix whose outputs from s and t first differ at a position where neither is 'n/a'."""
    for w in suffixes:
        pairs = zip(_run_on_model(w, table, s), _run_on_model(w, table, t), strict=True)
        differing = [(pos, a, b) for pos, (a, b) in enumerate(pairs) if a != b]
        if differing and NOT_APPLICABLE not in differing[0][1:]:
            return (w, *differing[0])
    return None


def _separating_sequences(table, s, t, max_length):
    """Every input sequence of length 1..max_length with different model outputs from s and from t."""
    found = []
    for length in range(1, max_length + 1):
        for sequence in itertools.product(INPUTS, repeat=length):
            if _run_on_model(sequence, table, s) != _run_on_model(sequence, table, t):
                found.append(sequence)
    return found


def _outputs_from_each_state(table, states, inputs):
    """Model outputs of inputs from every state."""
    return {s: _run_on_model(inputs, table, s) for s in states}


def _build_w_suite(cover, inputs, w_set, m):
    """V.W, V.X.W, ..., V.X^(m+1).W as concatenations, deduplicated and sorted."""
    suite = set()
    for v in cover.values():
        for k in range(m + 2):
            for xs in itertools.product(inputs, repeat=k):
                for w in w_set:
                    suite.add(v + xs + w)
    return tuple(sorted(suite))


def _layer(cover, inputs, w_set, k):
    """Raw list (with repeats) of v + xs + w for one k."""
    return [v + xs + w for v in cover.values() for xs in itertools.product(inputs, repeat=k) for w in w_set]


def _with_edge(table, edge, entry):
    """A mutant copy of table with one cell replaced."""
    mutant = dict(table)
    mutant[edge] = entry
    return MappingProxyType(mutant)


def _detecting(sequences, mutant, table=TRANSITIONS):
    """The sequences whose whole model output tuple differs between table and mutant."""
    return [seq for seq in sequences if _run_on_model(seq, mutant) != _run_on_model(seq, table)]


def _make_optimizer(seed=ANY_SEED):
    """The optimizer under test: one Float dimension, colony_size=4 (2 sources), limit=1."""
    return ABCOptimizer(Space({"x": Float(0.0, 1.0)}), colony_size=COLONY_SIZE, limit=LIMIT, seed=seed)


class _BatchAfterInitABC(ABCOptimizer):
    """Deliberately faulty optimizer: after initialisation each ask hands its candidate out twice."""

    def ask(self):
        batch = super().ask()
        return batch + batch if len(batch) == 1 else batch


def _make_faulty_optimizer():
    """The deliberately faulty optimizer with the same arguments as _make_optimizer."""
    return _BatchAfterInitABC(
        Space({"x": Float(0.0, 1.0)}), colony_size=COLONY_SIZE, limit=LIMIT, seed=ANY_SEED
    )


class _Driver:
    """Turns model inputs into public ask/tell calls on one optimizer and calls into model outputs."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.pending = []
        self.told = []
        self.asked = []

    def step(self, symbol):
        return getattr(self, "_" + symbol)()

    def _ask(self):
        try:
            batch = self.optimizer.ask()
        except Exception as exc:
            return type(exc).__name__
        self.pending.extend(batch)
        self.asked.extend(batch)
        return f"batch:{len(batch)}"

    def _tell(self, candidates, scores):
        try:
            self.optimizer.tell(candidates, scores)
        except Exception as exc:
            return type(exc).__name__
        return ACCEPTED

    def _tell_first(self):
        if not self.pending:
            return NOT_APPLICABLE
        output = self._tell(self.pending[:1], [IRRELEVANT_SCORE])
        if output == ACCEPTED:
            self.told.append(self.pending.pop(0))
        return output

    def _tell_all(self):
        if not self.pending:
            return NOT_APPLICABLE
        output = self._tell(list(self.pending), [IRRELEVANT_SCORE] * len(self.pending))
        if output == ACCEPTED:
            self.told.extend(self.pending)
            self.pending.clear()
        return output

    def _tell_stale(self):
        stale = self.told[-1] if self.told else Candidate(NEVER_ISSUED_ID, np.zeros(1), {"x": 0.0}, (0.0,))
        return self._tell([stale], [IRRELEVANT_SCORE])

    def _tell_mismatch(self):
        return self._tell(list(self.pending), [IRRELEVANT_SCORE] * (len(self.pending) + 1))


def _drive(driver, inputs):
    """Outputs of driving one driver through inputs."""
    return tuple(driver.step(symbol) for symbol in inputs)


def _run_on_optimizer(inputs, factory=_make_optimizer):
    """Outputs of inputs on a freshly built optimizer, so the sequence starts in s0 (reset)."""
    return _drive(_Driver(factory()), inputs)


def _drive_interleaved(first, second, inputs):
    """Steps two drivers alternately, one input at a time, and returns both output tuples."""
    outputs_first, outputs_second = [], []
    for symbol in inputs:
        outputs_first.append(first.step(symbol))
        outputs_second.append(second.step(symbol))
    return tuple(outputs_first), tuple(outputs_second)


def _asked_points(driver):
    """Every x handed out to the driver, in order, as nested tuples."""
    return tuple(tuple(c.x.tolist()) for c in driver.asked)


def _batch_summary(batch):
    """(ids, x values, params) of an ask result, comparable with ==."""
    return (
        tuple(c.id for c in batch),
        tuple(tuple(c.x.tolist()) for c in batch),
        tuple(c.params for c in batch),
    )


def _mismatch_report(inputs, expected, actual):
    """Comparison report: the input sequence, the first mismatching step and both outputs there."""
    pairs = list(zip(expected, actual, strict=False))
    steps = [n for n, (e, a) in enumerate(pairs) if e != a] or [len(pairs)]
    step = steps[0]
    at_step = (inputs[step : step + 1], expected[step : step + 1], actual[step : step + 1])
    return (
        f"inputs={inputs!r}: first mismatch at step {step}, input {at_step[0]!r}, "
        f"expected {at_step[1]!r}, actual {at_step[2]!r}; expected all={expected!r}; actual all={actual!r}"
    )


def _without_position(outputs, position):
    """outputs with one position removed."""
    return outputs[:position] + outputs[position + 1 :]


APPLICABLE_EDGES = tuple(sorted(_applicable_edges(TRANSITIONS)))
REJECTED_EDGES = tuple(
    edge for edge in APPLICABLE_EDGES if TRANSITIONS[edge][1] in (REFUSED_ASK, REFUSED_TELL)
)
SUITE = _build_w_suite(V4, INPUTS, W, EXTRA_STATES_M)


def _suffix_id(w):
    return ",".join(w)


EDGE_W_CASES = [
    pytest.param(qi, x, w, id=f"{_edge_id(qi, x)} + {_suffix_id(w)}")
    for (qi, x) in APPLICABLE_EDGES
    for w in W
]
REJECTED_W_CASES = [
    pytest.param(s, r, w, id=f"{_edge_id(s, r)} + {_suffix_id(w)}") for (s, r) in REJECTED_EDGES for w in W
]


def test_abc_mealy_model_should_be_total_over_states_and_inputs():
    """The protocol model is a total Mealy machine with exactly four inapplicable cells and 21 edges.

    Given the 5-state TRANSITIONS table, when every cell of STATES x INPUTS is inspected, then each holds
    INAPPLICABLE or (state, output), s0 is a state, INAPPLICABLE sits exactly where nothing is pending to
    tell (tell_first and tell_all in INIT and IDLE) and 25 - 4 = 21 edges remain (3 + 5 + 5 + 3 + 5).
    Oracle: rule P-T7-42 (S, s0, X, Y, delta, lambda) and the protocol rules quoted in the module docstring.
    Postcondition: none, the table is immutable.
    """
    expected_inapplicable = {(INIT, TELL_FIRST), (INIT, TELL_ALL), (IDLE, TELL_FIRST), (IDLE, TELL_ALL)}

    malformed = _malformed_cells(TRANSITIONS, STATES, INPUTS, OUTPUTS)

    assert malformed == []
    assert S0 in STATES
    assert set(_inapplicable_cells(TRANSITIONS)) == expected_inapplicable
    assert len(_applicable_edges(TRANSITIONS)) == 21


def test_abc_mealy_model_should_label_an_edge_as_input_slash_output():
    """An edge label reads 'x / y' and an edge id reads 'qi --x / y--> qj'.

    Given the edge INIT_WAIT2 --tell_first--> INIT_WAIT1 (telling one of two initial candidates), when its
    id is built, then it carries the 'tell_first / ok' label. Oracle: rule P-T7-42 labelling convention;
    'ok' because a partial tell of the initial batch is allowed by the task specification.
    """
    edge = (INIT_WAIT2, TELL_FIRST)

    edge_id = _edge_id(*edge)

    assert edge_id == "INIT_WAIT2 --tell_first / ok--> INIT_WAIT1"


def test_conformance_report_should_name_sequence_step_and_outputs_when_optimizer_deviates():
    """The model/optimizer comparison report names the input sequence, the first bad step and both outputs.

    Given a deliberately faulty ABCOptimizer subclass whose post-init ask returns its candidate twice and
    the shortest sequence reaching the post-init ask, ('ask', 'tell_all', 'ask'), when model and faulty
    optimizer outputs are compared, then the report shows the whole sequence, step 2 (not 0, so a real index
    and not the start), expected 'batch:1' and actual 'batch:2'. Oracle: model edge IDLE --ask / batch:1-->
    WAIT; the report format is the rule P-T7-43 requirement, not library spec.
    """
    inputs = (ASK, TELL_ALL, ASK)
    expected = _run_on_model(inputs)
    actual = _run_on_optimizer(inputs, factory=_make_faulty_optimizer)

    report = _mismatch_report(inputs, expected, actual)

    assert actual != expected
    assert repr(inputs) in report
    assert "step 2" in report
    assert "expected ('batch:1',)" in report
    assert "actual ('batch:2',)" in report


@pytest.mark.parametrize(
    ("table", "cover", "states"),
    [
        pytest.param(TRANSITIONS, V5, STATES, id="five-state-model-V5"),
        pytest.param(MIN_TRANSITIONS, V4, MIN_STATES, id="minimal-model-V4"),
    ],
)
def test_state_cover_should_reach_every_state_and_contain_epsilon(table, cover, states):
    """Each state cover holds the empty sequence and reaches every state of its model from s0.

    Given V5 for the 5-state model and V4 for the minimal model, when each V[s] is walked from s0, then it
    ends in s without an inapplicable input, every state is reached and () is one of the sequences.
    Oracle: rule P-T7-47 adequacy (V reaches every state from s0 and contains epsilon); W-method
    assumption (4). Postcondition: none, model only.
    """
    expected_report = _reached_without_inapplicable(states)

    report = _cover_report(table, cover)

    assert () in cover.values()
    assert report == expected_report


def test_partition_refinement_should_merge_only_init_wait1_and_wait():
    """Minimising the 5-state model merges exactly INIT_WAIT1 and WAIT (both: one candidate pending).

    Given TRANSITIONS with 'n/a' as the output of an inapplicable cell, when partition refinement runs to
    its fixpoint, then the classes are {INIT}, {INIT_WAIT2}, {IDLE}, {INIT_WAIT1, WAIT}. INIT_WAIT2 has the
    same output row as the waiting states and splits only in the second round (tell_first leads to a
    waiting state, not to IDLE), so a single-round refinement would fail here. Oracle: rule P-T7-50.
    """
    expected = {
        frozenset({INIT}),
        frozenset({INIT_WAIT2}),
        frozenset({IDLE}),
        frozenset({INIT_WAIT1, WAIT}),
    }

    partition = _refine(STATES, TRANSITIONS)

    assert partition == expected


def test_merged_states_should_agree_on_every_output_and_successor_class():
    """INIT_WAIT1 and WAIT give the same output and the same successor class for every input.

    Given the two merged states, when their rows (output, class of successor) are read for all five
    inputs, then the rows are equal: ask / RuntimeError to itself, tell_first and tell_all / ok to IDLE,
    tell_stale and tell_mismatch / ValueError to itself. No distinguishing input exists, which is why they
    are merged. Oracle: TRANSITIONS and CLASS_OF (rule P-T7-50).
    """
    expected = (
        (REFUSED_ASK, PENDING1),
        (ACCEPTED, IDLE),
        (ACCEPTED, IDLE),
        (REFUSED_TELL, PENDING1),
        (REFUSED_TELL, PENDING1),
    )

    row_wait1 = _row_with_successor_class(TRANSITIONS, INIT_WAIT1, CLASS_OF)
    row_wait = _row_with_successor_class(TRANSITIONS, WAIT, CLASS_OF)

    assert row_wait1 == expected
    assert row_wait == expected


def test_minimal_model_should_equal_the_quotient_of_the_five_state_model():
    """The declared 4-state MIN_TRANSITIONS is the quotient of TRANSITIONS by CLASS_OF.

    Given CLASS_OF (INIT_WAIT1 and WAIT to PENDING1, every other state to itself), when the quotient table
    is built, then it equals MIN_TRANSITIONS cell for cell, including the four INAPPLICABLE cells.
    Oracle: rule P-T7-50 (minimise before building W).
    """
    class_of = CLASS_OF

    quotient = _quotient(TRANSITIONS, STATES, class_of)

    assert quotient == dict(MIN_TRANSITIONS)


def test_minimal_model_should_have_only_singleton_classes():
    """Partition refinement of MIN_TRANSITIONS leaves every state alone, so the model is minimal.

    Given the 4-state model, when refinement runs to its fixpoint, then there are four singleton classes.
    Oracle: rule P-T7-50 (W existence and size guarantee hold for minimal machines).
    """
    expected = {frozenset({INIT}), frozenset({INIT_WAIT2}), frozenset({PENDING1}), frozenset({IDLE})}

    partition = _refine(MIN_STATES, MIN_TRANSITIONS)

    assert partition == expected


@pytest.mark.parametrize(
    ("s", "t", "expected"),
    [
        pytest.param(INIT, INIT_WAIT2, ((ASK,), 0, BATCH_OF_TWO, REFUSED_ASK), id="INIT-vs-INIT_WAIT2"),
        pytest.param(INIT, PENDING1, ((ASK,), 0, BATCH_OF_TWO, REFUSED_ASK), id="INIT-vs-PENDING1"),
        pytest.param(INIT, IDLE, ((ASK,), 0, BATCH_OF_TWO, BATCH_OF_ONE), id="INIT-vs-IDLE"),
        pytest.param(
            INIT_WAIT2,
            PENDING1,
            ((TELL_FIRST, ASK), 1, REFUSED_ASK, BATCH_OF_ONE),
            id="INIT_WAIT2-vs-PENDING1",
        ),
        pytest.param(INIT_WAIT2, IDLE, ((ASK,), 0, REFUSED_ASK, BATCH_OF_ONE), id="INIT_WAIT2-vs-IDLE"),
        pytest.param(PENDING1, IDLE, ((ASK,), 0, REFUSED_ASK, BATCH_OF_ONE), id="PENDING1-vs-IDLE"),
    ],
)
def test_w_set_should_separate_every_pair_of_minimal_states_without_an_na_step(s, t, expected):
    """W = {(ask), (tell_first, ask)} separates each pair of minimal states at a position without 'n/a'.

    Given one of the 6 unordered pairs of MIN states, when the first w in W whose outputs first differ is
    searched, then it is the expected w with the expected outputs, neither of them 'n/a'. W is built on the
    4-state model because INIT_WAIT1 and WAIT are equivalent, so no set can separate them in the 5-state
    model. INIT_WAIT2 vs PENDING1 is the decisive pair: both refuse ask, only (tell_first, ask) separates
    them, and only because a refused ask is observable. Oracle: rule P-T7-50; W-method assumption (3).
    """
    suffixes = W

    separation = _first_separation(MIN_TRANSITIONS, s, t, suffixes)

    assert separation == expected


def test_w_set_should_respect_the_minimal_machine_size_bound():
    """W holds at most n - 1 sequences, each at most n - 1 long, for the n = 4 minimal model.

    Given W, when its size and longest sequence are measured, then 2 <= 3 and 2 <= 3.
    Oracle: rule P-T7-50 ("at most n - 1 sequences, each of length at most n - 1").
    """
    n = len(MIN_STATES)

    sizes = _set_shape(W)

    assert sizes == (2, 2)
    assert sizes[0] <= n - 1 and sizes[1] <= n - 1


def test_five_state_model_should_have_no_sequence_separating_init_wait1_from_wait():
    """No input sequence up to length 4 separates INIT_WAIT1 from WAIT, so no DS or UIO exists for them.

    Given the 5-state model, when all 5 + 25 + 125 + 625 sequences of length 1..4 are run from both
    states, then none gives different outputs. The 5-state model therefore has no distinguishing sequence
    and these two states have no UIO; this is computed, not assumed. Oracle: rule P-T7-49 ("Not every FSM
    has one").
    """
    max_length = 4

    separators = _separating_sequences(TRANSITIONS, INIT_WAIT1, WAIT, max_length)

    assert separators == []


def test_minimal_model_should_give_four_distinct_outputs_for_tell_first_then_ask():
    """(tell_first, ask) gives a different output tuple from each of the four minimal states.

    Given MIN_TRANSITIONS, when (tell_first, ask) is run from every state, then INIT gives ('n/a',
    'batch:2'), INIT_WAIT2 ('ok', 'RuntimeError'), PENDING1 ('ok', 'batch:1') and IDLE ('n/a', 'batch:1'),
    four distinct tuples. Stated explicitly (rule P-T7-49): this claim, and W's separation of INIT_WAIT2 from
    PENDING1 and of INIT and IDLE from the waiting states, rely on refused inputs being observable
    (RuntimeError from ask, ValueError from tell_stale and tell_mismatch); an inapplicable input makes no
    call, outputs 'n/a' and keeps the state; IDLE and PENDING1 differ here ONLY at the 'n/a' position 0,
    which the driver knows (nothing to tell), not the optimizer, and W never uses an 'n/a' position to
    separate states. Oracle: hand walk of MIN_TRANSITIONS.
    """
    inputs = (TELL_FIRST, ASK)
    expected = {
        INIT: (NOT_APPLICABLE, BATCH_OF_TWO),
        INIT_WAIT2: (ACCEPTED, REFUSED_ASK),
        PENDING1: (ACCEPTED, BATCH_OF_ONE),
        IDLE: (NOT_APPLICABLE, BATCH_OF_ONE),
    }

    outputs = _outputs_from_each_state(MIN_TRANSITIONS, MIN_STATES, inputs)

    assert outputs == expected
    assert len(set(outputs.values())) == len(MIN_STATES)


@pytest.mark.parametrize(("state", "rejected", "w"), REJECTED_W_CASES)
def test_abc_optimizer_should_leave_protocol_state_unchanged_when_input_is_rejected(state, rejected, w):
    """A refused ask or tell is reported and leaves the protocol state as it was.

    Given an optimizer driven by V5[state] into one of the 13 states/inputs whose model output is a refusal,
    when the refused input is followed by w, then the refusal shows at its position and the remaining
    outputs equal those of V5[state] + w on another fresh optimizer. Cases include tell_mismatch with 0, 1
    and 2 pending candidates, tell_stale with a never issued and an already told candidate, and ask with 2
    initial, 1 initial and 1 post-init candidate pending. Only protocol outputs are compared; whether a
    refused call consumes random draws is unspecified and not asserted. Oracle: task specification
    (RuntimeError, ValueError) and the refused-input convention of rule P-T7-49 (self-loop), which the W
    method needs.
    """
    position = len(V5[state])
    expected_refusal = TRANSITIONS[(state, rejected)][1]
    baseline = _run_on_optimizer(V5[state] + w)

    outputs = _run_on_optimizer(V5[state] + (rejected,) + w)

    assert outputs[position] == expected_refusal
    assert _without_position(outputs, position) == baseline


def test_abc_optimizer_should_hand_out_identical_first_batches_when_built_with_same_seed():
    """Two optimizers built with the same seed hand out the same first batch.

    Given two fresh ABCOptimizer instances with SAME_SEED, when each is asked once, then the ids, x values
    and params of both batches are equal. Oracle: README "The same seed reproduces the same search";
    W-method assumption (2), reset = building a fresh optimizer. The x values are compared between runs,
    never with a recorded value.
    """
    first = _make_optimizer(SAME_SEED)
    second = _make_optimizer(SAME_SEED)

    batches = (first.ask(), second.ask())

    assert _batch_summary(batches[0]) == _batch_summary(batches[1])


def test_abc_optimizer_should_not_share_state_when_same_seed_instances_are_interleaved():
    """Two same-seed optimizers stepped alternately behave exactly like one optimizer run alone.

    Given drivers C and D on same-seed optimizers stepped call by call through the W-method sequence
    V4[IDLE] + (ask,) + (tell_first, ask), and driver E on a third same-seed optimizer run alone, when the
    runs end, then C, D and E give the same outputs (those of the model) and hand out the same x values.
    Interleaving catches shared module or global random state that one run after another would miss.
    Oracle: README "The same seed reproduces the same search"; W-method assumption (2).
    """
    inputs = V4[IDLE] + (ASK,) + (TELL_FIRST, ASK)
    driver_c = _Driver(_make_optimizer(SAME_SEED))
    driver_d = _Driver(_make_optimizer(SAME_SEED))
    driver_e = _Driver(_make_optimizer(SAME_SEED))
    outputs_e = _drive(driver_e, inputs)

    outputs_c, outputs_d = _drive_interleaved(driver_c, driver_d, inputs)

    assert outputs_c == outputs_e == _run_on_model(inputs)
    assert outputs_d == outputs_e
    assert _asked_points(driver_c) == _asked_points(driver_e)
    assert _asked_points(driver_d) == _asked_points(driver_e)


@pytest.mark.parametrize("w", [pytest.param(w, id=_suffix_id(w)) for w in W])
def test_abc_optimizer_should_conform_to_model_when_driven_across_several_cycles(w):
    """After 20 told single asks, which span 4 whole cycles with scouts, the protocol is still the model's.

    Given V4[IDLE] followed by 20 rounds of (ask, tell_first) and then w, when it runs on a fresh optimizer,
    then every output equals the model. With 2 sources, limit=1 and one constant score, each neighbour is
    not strictly better, so every source reaches trial 1 and each cycle is 2 employed + 2 onlooker + 1 scout
    asks; 20 rounds therefore pass the onlooker sweep end, the scout step and the cycle end 4 times, which
    the generated suite never reaches. Partial evidence for W-method assumption (1). Oracle: task
    specification (greedy acceptance, scout if max trial >= limit, one candidate per ask after init) and
    the ABC module docstring.
    """
    rounds = 20
    inputs = V4[IDLE] + (ASK, TELL_FIRST) * rounds + w

    outputs = _run_on_optimizer(inputs)

    assert outputs == _run_on_model(inputs), _mismatch_report(inputs, _run_on_model(inputs), outputs)


def test_w_method_assumptions_should_each_be_checked_or_marked_assumed():
    """Each of the four W-method assumptions is listed with its status and names tests of this module.

    Given W_METHOD_ASSUMPTIONS, when its numbers, statuses and named tests are read, then they are 1..4,
    only assumption (1) is ASSUMED and every named test exists here. Oracle: rule P-T7-51 adequacy (all
    four met or the unmet one stated).
    """
    tests_here = _test_names_in_module()

    numbers, assumed, named = _assumption_summary(W_METHOD_ASSUMPTIONS)

    assert numbers == (1, 2, 3, 4)
    assert assumed == (True, False, False, False)
    assert named <= tests_here


def test_transition_tour_should_fire_every_applicable_edge():
    """The three reset-separated tour sequences together fire all 21 applicable edges.

    Given TOUR, when each sequence is walked on the model from s0, then the union of fired edges is the
    whole edge set. Three sequences are needed: INIT, INIT_WAIT2 and INIT_WAIT1 are never re-entered and
    their exits tell_first and tell_all exclude each other, so each sequence starts again from s0 (reset,
    rule P-T7-53). A tour is sufficient only if no state-transfer faults exist and is not adequate alone
    (see the mutant tests). Oracle: rule P-T7-45 ("cover every transition").
    """
    expected = _applicable_edges(TRANSITIONS)

    fired = _edges_fired_by(TRANSITIONS, TOUR)

    assert fired == expected


@pytest.mark.parametrize("sequence", [pytest.param(t, id=f"tour-{n}") for n, t in enumerate(TOUR)])
def test_abc_optimizer_should_conform_to_model_on_transition_tour(sequence):
    """Each transition tour sequence gives the model's outputs on a fresh optimizer.

    Given one tour sequence, when it runs on the model and on a fresh optimizer, then the output tuples
    are equal step by step. Oracle: TRANSITIONS (rule P-T7-43, P-T7-45); conformance on the tour alone
    does not exclude transfer faults.
    """
    expected = _run_on_model(sequence)

    actual = _run_on_optimizer(sequence)

    assert actual == expected, _mismatch_report(sequence, expected, actual)


@pytest.mark.parametrize(
    ("mutant", "expected_detecting"),
    [
        pytest.param(
            _with_edge(TRANSITIONS, (IDLE, ASK), (WAIT, BATCH_OF_TWO)), [TOUR[2]], id="output-fault"
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT, REFUSED_TELL)), [], id="transfer-to-INIT"
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT_WAIT2, REFUSED_TELL)),
            [],
            id="transfer-to-INIT_WAIT2",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT_WAIT1, REFUSED_TELL)),
            [],
            id="transfer-to-INIT_WAIT1",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (WAIT, REFUSED_TELL)), [], id="transfer-to-WAIT"
        ),
    ],
)
def test_transition_tour_should_detect_output_fault_but_miss_transfer_fault_on_its_last_edge(
    mutant, expected_detecting
):
    """The tour catches a wrong output but not a wrong target on the edge it fires last and only once.

    Given an output-fault mutant (IDLE --ask--> WAIT outputs 'batch:2', the post-init phase handing out a
    batch) or a transfer-fault mutant (IDLE --tell_mismatch / ValueError--> moved to a state of another
    minimal class, output kept), when the whole output tuples of every tour sequence are compared with the
    specification, then only the long sequence detects the output fault and no sequence detects the
    transfer fault. The transfer edge is decisive because nothing follows it in the tour. Oracle: rule
    P-T7-45 ("If there are state-transfer faults, a transition tour may not find them") and P-T7-44.
    """
    last_edge_fired_once = _fire_count(TRANSITIONS, TOUR, TOUR_LAST_EDGE) == 1
    last_edge_is_last = _walk(TRANSITIONS, TOUR[2])[1][-1] == TOUR_LAST_EDGE

    detecting = _detecting(TOUR, mutant)

    assert last_edge_fired_once and last_edge_is_last
    assert detecting == expected_detecting


@pytest.mark.parametrize(
    ("mutant", "sequence", "expected_mutant_outputs"),
    [
        pytest.param(
            _with_edge(TRANSITIONS, (IDLE, ASK), (WAIT, BATCH_OF_TWO)),
            V4[IDLE] + W[0],
            (BATCH_OF_TWO, ACCEPTED, BATCH_OF_TWO),
            id="output-fault",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT, REFUSED_TELL)),
            V4[IDLE] + (TELL_MISMATCH,) + W[0],
            (BATCH_OF_TWO, ACCEPTED, REFUSED_TELL, BATCH_OF_TWO),
            id="transfer-to-INIT",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT_WAIT2, REFUSED_TELL)),
            V4[IDLE] + (TELL_MISMATCH,) + W[0],
            (BATCH_OF_TWO, ACCEPTED, REFUSED_TELL, REFUSED_ASK),
            id="transfer-to-INIT_WAIT2",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (INIT_WAIT1, REFUSED_TELL)),
            V4[IDLE] + (TELL_MISMATCH,) + W[0],
            (BATCH_OF_TWO, ACCEPTED, REFUSED_TELL, REFUSED_ASK),
            id="transfer-to-INIT_WAIT1",
        ),
        pytest.param(
            _with_edge(TRANSITIONS, TOUR_LAST_EDGE, (WAIT, REFUSED_TELL)),
            V4[IDLE] + (TELL_MISMATCH,) + W[0],
            (BATCH_OF_TWO, ACCEPTED, REFUSED_TELL, REFUSED_ASK),
            id="transfer-to-WAIT",
        ),
    ],
)
def test_w_method_suite_should_detect_output_fault_and_transfer_fault(
    mutant, sequence, expected_mutant_outputs
):
    """The generated W-method suite holds a sequence whose whole output tuple exposes each mutant.

    Given the output-fault and transfer-fault mutants of the tour test, when a suite sequence that fires
    the faulty edge and then a W suffix runs on the mutant, then its outputs differ from the specification
    exactly where the hand walk says: the immediate output for the output fault, the later W step for the
    transfer fault (whose immediate output 'ValueError' is unchanged, so checking it alone would pass).
    Oracle: hand walk of TRANSITIONS and the mutants; rule P-T7-44 and P-T7-52.
    """
    in_suite = sequence in SUITE
    spec_outputs = _run_on_model(sequence)

    mutant_outputs = _run_on_model(sequence, mutant)

    assert in_suite
    assert mutant_outputs == expected_mutant_outputs
    assert mutant_outputs != spec_outputs


@pytest.mark.parametrize(("qi", "x", "w"), EDGE_W_CASES)
def test_abc_optimizer_should_give_edge_output_and_reach_edge_target(qi, x, w):
    """Each transition gives its output and leaves the optimizer in a state that answers W like its target.

    Given a fresh optimizer brought to qi by V5[qi] (step 1), when input x is given (step 2) and then w,
    then the output at x is the edge output y (step 3) and the outputs of w equal the model outputs of w
    from the minimal class of qj (step 4). Both W elements are needed because INIT_WAIT2 and PENDING1
    targets differ only under (tell_first, ask). Oracle: TRANSITIONS, MIN_TRANSITIONS and CLASS_OF; rule
    P-T7-46 four steps.
    """
    access = V5[qi]
    target, edge_output = TRANSITIONS[(qi, x)]
    expected_w_outputs = _run_on_model(w, MIN_TRANSITIONS, CLASS_OF[target])

    outputs = _run_on_optimizer(access + (x,) + w)

    assert outputs[: len(access)] == _run_on_model(access), f"step 1 failed for {_edge_id(qi, x)}"
    assert outputs[len(access)] == edge_output, f"step 3 failed for {_edge_id(qi, x)}"
    assert outputs[len(access) + 1 :] == expected_w_outputs, f"step 4 failed for {_edge_id(qi, x)} + {w!r}"


def test_w_method_suite_should_equal_the_union_of_v_xk_w_concatenations():
    """The generated suite is exactly V.W u V.X.W u V.X^2.W, with known raw and deduplicated sizes.

    Given V4, INPUTS, W and m = 1, when the suite is generated, then it equals the union of the three raw
    layers built separately: 4 * 2 = 8, 4 * 5 * 2 = 40 and 4 * 25 * 2 = 200 sequences (248). After
    deduplication 7, 38 and 191 remain. Hand count: the prefixes v + xs number 4, 21 and 106 for k <= 0, 1, 2;
    each gives 2 sequences, and p + (ask) equals q + (tell_first, ask) exactly when p = q + (tell_first),
    which happens 1, 4 and 21 times, so 8 - 1, 42 - 4 and 212 - 21. '.' is sequence concatenation and X^k
    the k-fold product. Oracle: rule P-T7-52.
    """
    v_w, v_x_w, v_x2_w = _layer(V4, INPUTS, W, 0), _layer(V4, INPUTS, W, 1), _layer(V4, INPUTS, W, 2)

    suite = _build_w_suite(V4, INPUTS, W, EXTRA_STATES_M)

    assert set(suite) == set(v_w) | set(v_x_w) | set(v_x2_w)
    assert (len(v_w), len(v_x_w), len(v_x2_w)) == (8, 40, 200)
    assert (len(set(v_w)), len(set(v_w) | set(v_x_w)), len(suite)) == (7, 38, 191)


@pytest.mark.parametrize("sequence", [pytest.param(seq, id=_suffix_id(seq)) for seq in SUITE])
def test_abc_optimizer_should_conform_to_model_on_every_w_method_sequence(sequence):
    """Every sequence of the generated W-method suite gives the model's outputs on a fresh optimizer.

    Given one sequence of V4.W u V4.X.W u V4.X^2.W, when it runs on the model and on a fresh optimizer,
    then the output tuples are equal; an inapplicable step outputs 'n/a' on both sides. The suite is
    complete (finds output and state-transfer faults) only under the four W-method assumptions with n = 4
    and m = 1, listed in W_METHOD_ASSUMPTIONS. Oracle: TRANSITIONS (rule P-T7-43, P-T7-52).
    """
    expected = _run_on_model(sequence)

    actual = _run_on_optimizer(sequence)

    assert actual == expected, _mismatch_report(sequence, expected, actual)
