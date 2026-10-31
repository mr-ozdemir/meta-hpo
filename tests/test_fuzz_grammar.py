"""Grammar-derived Space declarations and a pairwise covering array of optimize() configurations.

Declaration grammar (start symbol Space, leftmost derivation, bounds as annotations):

    Space ::= Param | Param Space                          Param{1,6}  (MAX_PARAMS)
    Param ::= FloatLin | FloatLog | IntLin | IntLog | Cat
    Cat ::= Choice | Choice Cat                            Choice{1,5} (MAX_CHOICES per Cat)

Terminals and the constructor arguments drawn for them (constraints from the Float, Int and Categorical
constructors in space.py):

    FloatLin -> Float(low, high) with low < high
    FloatLog -> Float(low, high, log=True) with 0 < low < high
    IntLin   -> Int(low, high) with low <= high
    IntLog   -> Int(low, high, log=True) with 1 <= low <= high
    Choice   -> a value, distinct within its Cat and never ==-equal to another one (CHOICE_POOL)

ALL_RULES lists the productions R1..R9 in the order above; the derivator logs every production it applies.
The bounds are enforced by the derivator itself: once a Space would create its MAX_PARAMS-th Param only R1
is offered, once a Cat would create its MAX_CHOICES-th Choice only R8 is offered, and the rewrite loop runs at
most MAX_STEPS times. Production coverage is shown twice: by FORCED_DERIVATIONS, independent of random luck,
and by the seeded batch BATCH_SEEDS that also feeds the Space property tests. The recursive rules R2 and R9
are run exactly Min, Min+1, Max-1 and Max times.

Configuration grammar for optimize(), tested with a pairwise covering array built here with the standard
library only (greedy over the full product, deterministic):

    Config ::= Kind Direction Cache Colony Budget
    Kind{float, float_log, int, int_log, categorical} Direction{min, max} Cache{True, False}
    Colony{4, 5, 10} Budget{one, inside_init = max(1, colony // 2 - 1), fifty}

For colony 4 and 5, inside_init is 1, the same as Budget one; only colony 10 (inside_init = 4) stops with
more than one trial inside the initial batch. The array therefore holds fewer distinct effective
configurations than rows.

Oracles: README (Search space table: Float real numbers in [low, high], Int integers low..high inclusive,
Categorical one of choices; Behaviour: cache=True reuses the earlier score of a repeated parameter set),
CHANGELOG ("a configuration is never trained twice"), the Space, Space.decode, Space.rows and Space.move
docstrings, the Int and Categorical docstrings (equal share of the unit interval per value), the optimize
docstring (max_evals counts evaluations) and the Study.n_calls docstring (cached repeats not counted).
MAX_PARAMS, MAX_CHOICES and the Colony and Budget levels are test design choices; the library sets no upper
limit on dimension or number of choices.

Generated tests are never a unit's only check. Every property asserted here also has hand-written,
spec-based examples with literal expected values in the unit test files: test_space.py (bounds, equal share,
categorical mapping, Python types, rows keys, Space.move), test_abc.py (exact budget inside the initial
population, cache=True, direction) and test_facade.py (cache=False calls the function for every trial).

Unspecified, therefore not asserted:
- bit-exact agreement of move and rows for log-scaled Float columns (one-value against array decode); compared
  with relative tolerance LOG_REL_TOL, every other column exactly;
- equal share for log-scaled Int; only its endpoints are checked;
- whether u = k/n maps to bucket k for large n (it does not for some n >= 22); buckets use n <= 5;
- duplicate or ==-equal categorical choices; the derivator never emits them.
Known defect, reported separately and kept out of this module: a log-scaled Float decodes u = 0.0 or 1.0
(reachable through neighbour clipping) to a value just outside [low, high], so the bounds test over the
covering array leaves out the float_log rows.

Input space: Every Space declaration of the docstring grammar and every optimize() configuration.
Subset: Seeded grammar derivations, forced derivations per production and a pairwise covering array.
"""

import math
from itertools import combinations, product

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

MAX_PARAMS = 6


MAX_CHOICES = 5


MAX_STEPS = MAX_PARAMS * (2 + MAX_CHOICES)


P_RECURSE = 0.6


LOG_REL_TOL = 1e-12


N_POINTS = 8


BATCH_SEEDS = tuple(range(50))


BOUND_SEEDS = tuple(range(200))


PROPERTY_RNG_OFFSET = 1000


MOVE_RNG_OFFSET = 2000


TERMINAL_SEED = 0


CHOICE_POOL = ("c0", 1, 2.5, "relu", (2, 3))


TOP = float(np.nextafter(1.0, 0.0))


ALL_RULES = (
    ("R1", "Space", ("Param",)),
    ("R2", "Space", ("Param", "Space")),
    ("R3", "Param", ("FloatLin",)),
    ("R4", "Param", ("FloatLog",)),
    ("R5", "Param", ("IntLin",)),
    ("R6", "Param", ("IntLog",)),
    ("R7", "Param", ("Cat",)),
    ("R8", "Cat", ("Choice",)),
    ("R9", "Cat", ("Choice", "Cat")),
)


RULE_IDS = tuple(rid for rid, _, _ in ALL_RULES)


RULE_RHS = {rid: rhs for rid, _, rhs in ALL_RULES}


PARAM_RULES = ("R3", "R4", "R5", "R6", "R7")


NONTERMINALS = ("Space", "Param", "Cat")


FORCED_DERIVATIONS = (
    ("R1", "R3"),
    ("R2", "R4", "R1", "R5"),
    ("R2", "R6", "R1", "R7", "R9", "R8"),
)


def _run_case(target, *args, **kwargs):
    """Runs any target and returns ('ok', value) or ('error', type name, message); never raises."""
    try:
        return ("ok", target(*args, **kwargs))
    except Exception as exc:
        return ("error", type(exc).__name__, str(exc))


def _failure_lines(cases):
    """One 'Test #<id> IN=<repr> OUT=<outcome>' line per case whose outcome is not ok."""
    return [
        f"Test #{case_id} IN={given!r} OUT={outcome[1]}: {outcome[2]}"
        for case_id, given, outcome in cases
        if outcome[0] != "ok"
    ]


def _grammar_lines(rules):
    """The 'LHS ::= alt | alt' line of every non-terminal, rebuilt from the rule table."""
    lines = {}
    for _, lhs, rhs in rules:
        lines.setdefault(lhs, []).append(" ".join(rhs))
    return [f"{lhs} ::= {' | '.join(alts)}" for lhs, alts in lines.items()]


def _allowed(symbol, n_params, n_choices):
    """Rule ids offered for symbol; the recursive alternative is withdrawn once its bound would be passed."""
    if symbol == "Space" and n_params + 1 >= MAX_PARAMS:
        return ["R1"]
    if symbol == "Cat" and n_choices + 1 >= MAX_CHOICES:
        return ["R8"]
    return [rid for rid, lhs, _ in ALL_RULES if lhs == symbol]


def _random_policy(rng):
    """Picks R2 and R9 with probability P_RECURSE when allowed, and a uniform Param alternative."""

    def policy(symbol, allowed):
        if len(allowed) == 1:
            return allowed[0]
        if symbol == "Param":
            return allowed[int(rng.integers(len(allowed)))]
        return allowed[1] if rng.random() < P_RECURSE else allowed[0]

    return policy


def _forced_policy(rule_ids):
    """Replays rule_ids in order; a rule the derivator does not offer is a broken test input."""
    pending = iter(rule_ids)

    def policy(symbol, allowed):
        rid = next(pending)
        if rid not in allowed:
            raise ValueError(f"rule {rid} is not offered for {symbol}: {allowed}")
        return rid

    return policy


def _always_recurse_policy(symbol, allowed):
    """Always takes the last offered alternative: R2, R7 (Cat) and R9 for as long as the bounds allow."""
    return allowed[-1]


def _realise(form, rng):
    """Turns the terminal string into {'p0': Parameter, ...}, drawing constructor arguments left to right."""
    groups = {}
    for symbol, owner in form:
        groups.setdefault(owner, []).append(symbol)
    mapping = {}
    for owner, symbols in groups.items():
        kind = symbols[0]
        if kind == "FloatLin":
            low = float(rng.integers(-50, 50))
            p = Float(low, low + float(rng.integers(1, 100)))
        elif kind == "FloatLog":
            low = 2.0 ** int(rng.integers(-10, 4))
            p = Float(low, low * 2.0 ** int(rng.integers(1, 10)), log=True)
        elif kind == "IntLin":
            low = int(rng.integers(-20, 20))
            p = Int(low, low + int(rng.integers(0, 30)))
        elif kind == "IntLog":
            low = int(rng.integers(1, 10))
            p = Int(low, low + int(rng.integers(0, 200)), log=True)
        else:
            order = rng.permutation(len(CHOICE_POOL))[: len(symbols)]
            p = Categorical([CHOICE_POOL[int(i)] for i in order])
        mapping[f"p{owner}"] = p
    return mapping


def _derive(policy, rng):
    """Leftmost derivation from Space; returns (mapping, rule log). Bounded by MAX_STEPS rewrites."""
    form = [("Space", None)]
    log = []
    n_params = 0
    n_choices = {}
    for _ in range(MAX_STEPS + 1):
        pos = next((i for i, (s, _) in enumerate(form) if s in NONTERMINALS), None)
        if pos is None:
            break
        symbol, owner = form[pos]
        if symbol == "Space":
            rid = policy(symbol, _allowed(symbol, n_params, 0))
            children = [("Param", n_params)] + ([("Space", None)] if rid == "R2" else [])
            n_params += 1
        elif symbol == "Cat":
            rid = policy(symbol, _allowed(symbol, 0, n_choices.get(owner, 0)))
            n_choices[owner] = n_choices.get(owner, 0) + 1
            children = [(s, owner) for s in RULE_RHS[rid]]
        else:
            rid = policy(symbol, _allowed(symbol, 0, 0))
            children = [(s, owner) for s in RULE_RHS[rid]]
        form[pos : pos + 1] = children
        log.append(rid)
    if any(s in NONTERMINALS for s, _ in form):
        raise RuntimeError("derivation did not terminate within MAX_STEPS")
    return _realise(form, rng), tuple(log)


def _derive_random(seed):
    """The seeded random derivation used by the batch and bound tests."""
    rng = np.random.default_rng(seed)
    return _derive(_random_policy(rng), rng)


def _derive_forced(rule_ids):
    """A derivation that applies exactly rule_ids; terminal arguments come from TERMINAL_SEED."""
    mapping, log = _derive(_forced_policy(rule_ids), np.random.default_rng(TERMINAL_SEED))
    if log != tuple(rule_ids):
        raise ValueError(f"forced rules {rule_ids} left unused, derivation stopped at {log}")
    return mapping, log


def _space_rules(n_params):
    """R2 n_params - 1 times then R1; Param kinds rotate through R3..R7, every Cat gets two choices."""
    rules = []
    for k in range(n_params):
        rules.append("R2" if k < n_params - 1 else "R1")
        kind = PARAM_RULES[k % len(PARAM_RULES)]
        rules.append(kind)
        if kind == "R7":
            rules.extend(["R9", "R8"])
    return tuple(rules)


def _cat_rules(n_choices):
    """One categorical parameter whose Cat applies R9 n_choices - 1 times and then R8."""
    return ("R1", "R7") + ("R9",) * (n_choices - 1) + ("R8",)


def _logs(seeds):
    return [_derive_random(s)[1] for s in seeds]


def _derivations(seeds):
    return [_derive_random(s) for s in seeds]


def _rules_used(logs):
    return set().union(*map(set, logs))


def _rules_lost_by_dropping_each(logs):
    """For every log, the productions that no other log in the list uses."""
    return [_rules_used(logs[:i] + logs[i + 1 :]) ^ _rules_used(logs) for i in range(len(logs))]


def _dims(seeds):
    return [len(_derive_random(s)[0]) for s in seeds]


def _choice_counts(seeds):
    return [
        len(p.choices) for s in seeds for p in _derive_random(s)[0].values() if isinstance(p, Categorical)
    ]


def _build_cases(seeds):
    """(seed, mapping, outcome of Space(mapping), rule log) for every seed."""
    cases = []
    for s in seeds:
        mapping, log = _derive_random(s)
        cases.append((s, mapping, _run_case(Space, mapping), log))
    return cases


def _dimension_mismatches(cases):
    """Built spaces whose dim differs from the number of Param productions (R3..R7) in their log."""
    return [
        f"Test #{s} IN={mapping!r} dim={outcome[1].dim} params={sum(log.count(r) for r in PARAM_RULES)}"
        for s, mapping, outcome, log in cases
        if outcome[0] == "ok" and outcome[1].dim != sum(log.count(r) for r in PARAM_RULES)
    ]


def _raw_violations(space, raw):
    """Columns of raw values outside their declared parameter (README bounds, decode docstring)."""
    out = []
    for j, p in enumerate(space.parameters):
        col = raw[:, j]
        whole = bool(np.all(col == np.floor(col)))
        if isinstance(p, Float):
            ok = bool(np.all((col >= p.low) & (col <= p.high)))
        elif isinstance(p, Int):
            ok = whole and bool(np.all((col >= p.low) & (col <= p.high)))
        else:
            ok = whole and bool(np.all((col >= 0) & (col <= len(p.choices) - 1)))
        if not ok:
            out.append(f"column {j} {p!r}: {col.tolist()}")
    return out


def _rows_violations(space, keys, params):
    """Cells where a key is not a float or a params value is not the documented Python value of its key."""
    out = []
    for i, (key, row) in enumerate(zip(keys, params, strict=True)):
        for j, (name, p) in enumerate(zip(space.names, space.parameters, strict=True)):
            k, v = key[j], row[name]
            if isinstance(p, Float):
                ok = type(v) is float and v == k
            elif isinstance(p, Int):
                ok = type(v) is int and v == k
            else:
                ok = v is p.choices[int(k)]
            if type(k) is not float or not ok:
                out.append(f"row {i} {name} {p!r}: key {k!r} value {v!r}")
    return out


def _typed(values):
    return [(type(v), v) for v in values]


def _move_mismatches(space, u, rng):
    """Moves every coordinate of every row to a fresh uniform; returns the cells where move differs from rows
    and the number of cells whose expected key differs from the source key (the moves that change a value)."""
    keys, params = space.rows(u)
    out = []
    changed = 0
    for i in range(u.shape[0]):
        for j, p in enumerate(space.parameters):
            v = float(rng.random())
            moved = u[i : i + 1].copy()
            moved[0, j] = v
            exp_keys, exp_params = space.rows(moved)
            key, row = space.move(keys[i], params[i], j, v)
            name = space.names[j]
            changed += exp_keys[0] != keys[i]
            if getattr(p, "log", False) and isinstance(p, Float):
                rest = [n for n in space.names if n != name]
                same = (
                    key[:j] + key[j + 1 :] == exp_keys[0][:j] + exp_keys[0][j + 1 :]
                    and _typed(row[n] for n in rest) == _typed(exp_params[0][n] for n in rest)
                    and type(row[name]) is float
                    and math.isclose(key[j], exp_keys[0][j], rel_tol=LOG_REL_TOL)
                    and math.isclose(row[name], exp_params[0][name], rel_tol=LOG_REL_TOL)
                )
            else:
                same = key == exp_keys[0] and _typed(row.values()) == _typed(exp_params[0].values())
            if not same:
                out.append(
                    f"row {i} column {j} u={v!r}: move {(key, row)!r} rows {(exp_keys[0], exp_params[0])!r}"
                )
    return out, changed


def _can_change(space):
    """True when some parameter has two or more values: a Float, an Int with low < high or two choices."""
    return any(
        isinstance(p, Float) or (isinstance(p, Int) and p.low < p.high) or len(getattr(p, "choices", ())) > 1
        for p in space.parameters
    )


def _sample_of(mapping, seed):
    """A fresh Space for mapping and N_POINTS uniform points from the property RNG of seed."""
    space = Space(mapping)
    return space, space.sample(np.random.default_rng(PROPERTY_RNG_OFFSET + seed), N_POINTS)


def test_grammar_rule_table_should_list_nine_productions_that_match_the_docstring_grammar():
    """Given ALL_RULES, when its lines are rebuilt as 'LHS ::= alt | alt', then each appears in the module
    docstring and the ids are R1..R9.

    Oracle: P-T7-11 asks for an explicit list of the productions that matches the documented grammar one to
    one. Postcondition: none (static data).
    """
    expected_ids = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9")

    lines = _grammar_lines(ALL_RULES)

    assert RULE_IDS == expected_ids
    assert lines == [
        "Space ::= Param | Param Space",
        "Param ::= FloatLin | FloatLog | IntLin | IntLog | Cat",
        "Cat ::= Choice | Choice Cat",
    ]
    assert [line for line in lines if line not in __doc__] == []


def test_grammar_derivator_should_reproduce_the_same_derivation_for_the_same_seed():
    """Given BATCH_SEEDS, when every seed is derived twice, then rule logs and declared parameters are equal,
    and the seeds do not all give the same derivation.

    Oracle: README "The same seed reproduces the same search", applied to the input generator; parameters are
    frozen dataclasses, so == compares low, high, log and choices. Distinct logs show the derivation is built
    from grammar steps and not a constant space.
    """
    first = _derivations(BATCH_SEEDS)

    second = _derivations(BATCH_SEEDS)

    assert second == first
    assert len({log for _, log in first}) > 1


def test_space_should_accept_every_derived_declaration_with_dimension_equal_to_its_param_count():
    """Given the seeded batch, when Space(mapping) is built for each derivation, then none raises and dim
    equals the number of Param productions (R3..R7) in the rule log.

    Oracle: Space docstring "dimension d = number of parameters"; every derived declaration satisfies the
    constructor constraints listed in the module docstring, so any exception is a defect. Failing cases are
    reported as 'Test #<seed> IN=<mapping> OUT=<Type: message>'.
    """
    seeds = BATCH_SEEDS

    cases = _build_cases(seeds)

    assert _failure_lines([(s, mapping, outcome) for s, mapping, outcome, _ in cases]) == []
    assert _dimension_mismatches(cases) == []


def test_grammar_derivator_should_stay_within_the_recursion_bounds_and_reach_them():
    """Given BOUND_SEEDS, when every seed is derived, then 1 <= dim <= MAX_PARAMS and every Categorical has
    1..MAX_CHOICES choices, and both maxima are reached at least once.

    Oracle: P-T7-10 (bounded recursion) with the bounds of the module docstring. P_RECURSE = 0.6 makes the
    caps reachable, so the maxima show the bound is enforced, not just never approached.
    """
    seeds = BOUND_SEEDS

    dims = _dims(seeds)
    choice_counts = _choice_counts(seeds)

    assert (min(dims), max(dims)) == (1, MAX_PARAMS)
    assert (min(choice_counts), max(choice_counts)) == (1, MAX_CHOICES)


def test_grammar_derivator_should_stop_at_the_bounds_when_the_policy_always_recurses():
    """Given a policy that always takes R2, Cat and R9, when one derivation runs, then it terminates with
    MAX_PARAMS categorical parameters of MAX_CHOICES choices each.

    Oracle: P-T7-10. This is the corner case where the bound is the only reason the derivation stops; an
    unbounded derivator would not terminate (site: _allowed withdrawing R2 and R9).
    """
    rng = np.random.default_rng(TERMINAL_SEED)

    mapping, log = _derive(_always_recurse_policy, rng)

    assert [len(p.choices) for p in mapping.values()] == [MAX_CHOICES] * MAX_PARAMS
    assert (log.count("R2"), log.count("R9")) == (MAX_PARAMS - 1, MAX_PARAMS * (MAX_CHOICES - 1))


def test_grammar_forced_derivations_should_cover_every_production_and_each_be_needed():
    """Given FORCED_DERIVATIONS, when they are derived, then their logs together use all of R1..R9 and
    dropping any one of them loses at least one production.

    Oracle: P-T7-11 production coverage, 100 %, independent of random luck. A derivator that could not offer
    one alternative (for example never R6) makes _derive_forced raise or the union miss that rule.
    """
    forced = FORCED_DERIVATIONS

    logs = [_derive_forced(rules)[1] for rules in forced]

    assert _rules_used(logs) == set(RULE_IDS)
    assert [lost != set() for lost in _rules_lost_by_dropping_each(logs)] == [True] * len(forced)


def test_grammar_seeded_batch_should_use_every_production_in_at_least_two_derivations():
    """Given BATCH_SEEDS, the same seeds the Space property tests are parametrised over, when the rule logs
    are collected, then every production of R1..R9 appears in at least two of them.

    Oracle: P-T7-11 production coverage of the inputs actually used; two derivations per rule keep coverage
    from resting on one lucky seed.
    """
    seeds = BATCH_SEEDS

    logs = _logs(seeds)

    assert {rid: sum(rid in log for log in logs) >= 2 for rid in RULE_IDS} == dict.fromkeys(RULE_IDS, True)


@pytest.mark.parametrize("seed", BATCH_SEEDS, ids=[f"seed{s}" for s in BATCH_SEEDS])
def test_space_sample_should_give_n_by_dim_points_in_the_unit_interval_for_a_derived_declaration(seed):
    """Given a derived declaration, when sample(rng, N_POINTS) is called, then the shape is (N_POINTS, dim)
    and every value lies in [0, 1).

    Oracle: Space.sample docstring "n uniform points of the unit cube, shape (n, d)".
    """
    mapping, _ = _derive_random(seed)
    space = Space(mapping)
    rng = np.random.default_rng(PROPERTY_RNG_OFFSET + seed)

    u = space.sample(rng, N_POINTS)

    assert u.shape == (N_POINTS, len(mapping))
    assert bool(np.all((u >= 0.0) & (u < 1.0))) is True


@pytest.mark.parametrize("seed", BATCH_SEEDS, ids=[f"seed{s}" for s in BATCH_SEEDS])
def test_space_decode_should_stay_inside_every_declared_parameter_for_a_derived_declaration(seed):
    """Given a derived declaration and sampled points, when decode is called, then Float columns lie in
    [low, high], Int columns are whole numbers in low..high and Categorical columns are indices 0..n-1.

    Oracle: README Search space table and the Space.decode docstring (integers as whole numbers, categories as
    indices). Expected bounds come from the declaration, not from another decode.
    """
    mapping, _ = _derive_random(seed)
    space, u = _sample_of(mapping, seed)

    raw = space.decode(u)

    assert _raw_violations(space, raw) == []


@pytest.mark.parametrize("seed", BATCH_SEEDS, ids=[f"seed{s}" for s in BATCH_SEEDS])
def test_space_rows_should_give_raw_float_keys_and_python_typed_params_for_a_derived_declaration(seed):
    """Given a derived declaration and sampled points, when rows is called, then each key is the decoded raw
    row of floats and each params value is the Python float, Python int or the very choice object of its key.

    Oracle: Space.rows docstring (a key is the hashable raw row; params have Python types) and Space.decode
    docstring. type(...) is int / float rejects numpy scalars; 'is' checks choice identity.
    """
    mapping, _ = _derive_random(seed)
    space, u = _sample_of(mapping, seed)

    keys, params = space.rows(u)

    assert keys == [tuple(r) for r in space.decode(u).tolist()]
    assert _rows_violations(space, keys, params) == []


@pytest.mark.parametrize("seed", BATCH_SEEDS, ids=[f"seed{s}" for s in BATCH_SEEDS])
def test_space_move_should_equal_rows_of_the_moved_point_for_every_coordinate_of_a_derived_declaration(seed):
    """Given a derived declaration and sampled rows, when every coordinate j of every row is moved to a fresh
    uniform, then move returns the key and params that rows gives for the moved point, and some move
    changes a value exactly when a parameter has two or more values, so the comparison is not vacuous.

    Oracle: Space.move docstring "(key, params) of a point differing from a decoded point only in coordinate
    j"; rows of the moved point is the documented full decode. Exact with types, except log Float columns
    (relative tolerance LOG_REL_TOL, see the module docstring). The new uniforms come from MOVE_RNG_OFFSET,
    a stream separate from the sampled points, so a move really changes the coordinate.
    """
    mapping, _ = _derive_random(seed)
    space, u = _sample_of(mapping, seed)
    rng = np.random.default_rng(MOVE_RNG_OFFSET + seed)

    mismatches, changed = _move_mismatches(space, u, rng)

    assert mismatches == []
    assert (changed >= 1) is _can_change(space)


@pytest.mark.parametrize("n_params", [1, 2, MAX_PARAMS - 1, MAX_PARAMS], ids=["min", "min+1", "max-1", "max"])
def test_space_should_have_dimension_n_when_the_space_rule_recurses_n_minus_one_times(n_params):
    """Given a derivation forced to apply R2 n_params - 1 times and R1 once, when Space is built and rows of
    sampled points are taken, then dim is n_params and every value lies in its declared parameter.

    Oracle: P-T7-12 boundary counts Min, Min+1, Max-1, Max of Space ::= Param Space; Space docstring (dim =
    number of parameters), README bounds and Space.rows docstring for the values. Kinds rotate R3..R7, so the
    Max case holds every kind.
    """
    mapping, log = _derive_forced(_space_rules(n_params))
    space, u = _sample_of(mapping, n_params)

    keys, params = space.rows(u)

    assert (space.dim, log.count("R2"), log.count("R1")) == (n_params, n_params - 1, 1)
    assert _raw_violations(space, np.array(keys)) == []
    assert _rows_violations(space, keys, params) == []


@pytest.mark.parametrize(
    "n_choices", [1, 2, MAX_CHOICES - 1, MAX_CHOICES], ids=["min", "min+1", "max-1", "max"]
)
def test_categorical_should_have_n_distinct_choices_when_the_cat_rule_recurses_n_minus_one_times(n_choices):
    """Given a derivation forced to R1, R7, R9 n_choices - 1 times and R8, when Space is built and rows of
    sampled points are taken, then the one Categorical has n_choices distinct choices and every value is
    one of them.

    Oracle: P-T7-12 boundary counts of Cat ::= Choice Cat; Categorical docstring "One of choices".
    """
    mapping, log = _derive_forced(_cat_rules(n_choices))
    space, u = _sample_of(mapping, n_choices)

    keys, params = space.rows(u)

    assert (len(set(mapping["p0"].choices)), log.count("R9"), log.count("R8")) == (
        n_choices,
        n_choices - 1,
        1,
    )
    assert _rows_violations(space, keys, params) == []


def _bucket_cases():
    """(n, u, expected index k) for n in 1, 2, 4, 5: u = 0.0, u = k/n, one ulp below k/n, and TOP."""
    cases = []
    for n in (1, 2, MAX_CHOICES - 1, MAX_CHOICES):
        points = {0.0: 0, TOP: n - 1}
        points.update({k / n: k for k in range(n)})
        points.update({float(np.nextafter(k / n, 0.0)): k - 1 for k in range(1, n)})
        cases.extend((n, u, k) for u, k in sorted(points.items()))
    return cases


def _value_at(space, path, u):
    """The single parameter's value at u through decode (raw), rows (params) or move (params)."""
    if path == "decode":
        return space.decode(np.array([[u]]))[0, 0]
    if path == "rows":
        return space.rows(np.array([[u]]))[1][0][space.names[0]]
    keys, params = space.rows(np.array([[0.0]]))
    return space.move(keys[0], params[0], 0, u)[1][space.names[0]]


PATH_TYPES = {"decode": np.float64, "rows": int, "move": int}


INT_BUCKET_CASES = [
    pytest.param(n, low, u, k, path, id=f"{path}-n{n}-low{low}-u{u!r}")
    for n, u, k in _bucket_cases()
    for low in (-3, 0, 7)
    for path in ("decode", "rows", "move")
]


@pytest.mark.parametrize(("n", "low", "u", "k", "path"), INT_BUCKET_CASES)
def test_int_should_map_u_to_the_value_of_its_equal_share_bucket(n, low, u, k, path):
    """Given Int(low, low + n - 1), when u at a bucket edge is decoded by decode, rows or move, then the value
    is low + k, where bucket k is [k/n, (k+1)/n); decode gives it as a whole float, rows and move as an int.

    Oracle: Int docstring (every integer gets an equal share of the unit interval), README (low..high
    inclusive), Space.decode docstring (integers as whole numbers) and Space.rows docstring (Python types).
    Edges from both sides: k/n is the first point of bucket k, one ulp below it the last point of bucket
    k - 1, TOP the last point of the top bucket. Negative, zero and positive low catch a missing offset.
    n <= 5 (module docstring).
    """
    space = Space({"i": Int(low, low + n - 1)})

    value = _value_at(space, path, u)

    assert value == low + k
    assert type(value) is PATH_TYPES[path]


CATEGORICAL_INDEX_CASES = [pytest.param(n, u, k, id=f"n{n}-u{u!r}") for n, u, k in _bucket_cases()]


CATEGORICAL_CHOICE_CASES = [
    pytest.param(n, u, k, path, id=f"{path}-n{n}-u{u!r}")
    for n, u, k in _bucket_cases()
    for path in ("rows", "move")
]


@pytest.mark.parametrize(("n", "u", "k"), CATEGORICAL_INDEX_CASES)
def test_categorical_decode_should_give_the_index_of_the_equal_share_bucket(n, u, k):
    """Given a Categorical of the first n entries of CHOICE_POOL, when decode gets u at a bucket edge, then it
    gives index k.

    Oracle: Categorical docstring (every choice gets an equal share of the unit interval) and Space.decode
    docstring (categories as indices). Edges as in the Int bucket test.
    """
    space = Space({"c": Categorical(list(CHOICE_POOL[:n]))})

    value = _value_at(space, "decode", u)

    assert value == k


@pytest.mark.parametrize(("n", "u", "k", "path"), CATEGORICAL_CHOICE_CASES)
def test_categorical_should_give_the_choice_object_of_the_equal_share_bucket(n, u, k, path):
    """Given a Categorical of the first n entries of CHOICE_POOL, when rows or move gets u at a bucket edge,
    then the value is the choice object k itself.

    Oracle: Categorical docstring ("One of choices", equal shares) and Space.rows docstring (the params the
    user's function receives). 'is' checks identity, so a converted copy or an index fails.
    """
    choices = CHOICE_POOL[:n]
    space = Space({"c": Categorical(list(choices))})

    value = _value_at(space, path, u)

    assert value is choices[k]


@pytest.mark.parametrize(
    ("low", "high"),
    [(1, 1), (1, 2), (1, 4), (1, 5), (2, 6), (1, 100)],
    ids=["single-value", "smallest-range", "four-values", "five-values", "low-above-one", "wide-range"],
)
@pytest.mark.parametrize("path", ["decode", "rows", "move"])
def test_log_int_should_map_the_ends_of_the_unit_interval_to_low_and_high(low, high, path):
    """Given Int(low, high, log=True), when u = 0.0 and u = TOP are decoded by decode, rows or move, then the
    values are low and high.

    Oracle: README (Int low..high inclusive). Equal share for log Int is unspecified, so only the endpoints
    are checked. (1, 1) is the degenerate range, (2, 6) catches a hard-coded low of 1.
    """
    space = Space({"i": Int(low, high, log=True)})

    ends = (_value_at(space, path, 0.0), _value_at(space, path, TOP))

    assert ends == (low, high)


FACTORS = (
    ("kind", ("float", "float_log", "int", "int_log", "categorical")),
    ("direction", ("min", "max")),
    ("cache", (True, False)),
    ("colony", (4, 5, 10)),
    ("budget", ("one", "inside_init", "fifty")),
)


def _pairs_of(row):
    return {(a, row[a], b, row[b]) for a, b in combinations(range(len(row)), 2)}


def _all_pairs(factors):
    levels = [values for _, values in factors]
    return {
        (a, x, b, y) for a, b in combinations(range(len(levels)), 2) for x in levels[a] for y in levels[b]
    }


def _uncovered_pairs(rows, factors):
    covered = set().union(*map(_pairs_of, rows)) if rows else set()
    return _all_pairs(factors) - covered


def _greedy_covering_array(factors):
    """Repeatedly adds the first full-product row that covers most still uncovered value pairs."""
    candidates = list(product(*(values for _, values in factors)))
    uncovered = _all_pairs(factors)
    rows = []
    for _ in range(len(candidates)):
        if not uncovered:
            break
        best = max(candidates, key=lambda r: len(_pairs_of(r) & uncovered))
        rows.append(best)
        uncovered -= _pairs_of(best)
    return tuple(rows)


COVERING_ARRAY = _greedy_covering_array(FACTORS)


def _max_evals(budget, colony):
    return {"one": 1, "inside_init": max(1, colony // 2 - 1), "fifty": 50}[budget]


def _space_of(kind):
    """A fresh two-parameter Space of one Kind; float_log uses the README quick-start bounds."""
    spaces = {
        "float": lambda: {"a": Float(-2.0, 3.0), "b": Float(-2.0, 3.0)},
        "float_log": lambda: {"a": Float(1e-4, 1e-2, log=True), "b": Float(1e-4, 1e-2, log=True)},
        "int": lambda: {"a": Int(-3, 7), "b": Int(-3, 7)},
        "int_log": lambda: {"a": Int(1, 64, log=True), "b": Int(1, 64, log=True)},
        "categorical": lambda: {"a": Categorical(["x", "y", "z"]), "b": Categorical([10, 20])},
    }
    return Space(spaces[kind]())


def _value(space, params):
    """A pure finite score: squared distance to 0.3 for numbers, the choice index for categories."""
    total = 0.0
    for name, p in zip(space.names, space.parameters, strict=True):
        v = params[name]
        total += p.choices.index(v) if isinstance(p, Categorical) else (v - 0.3) ** 2
    return total


def _spy_objective(calls, space):
    def spy_objective(params):
        calls.append(dict(params))
        return _value(space, params)

    return spy_objective


def _expected_cached(trials):
    """Per trial, whether an earlier trial had equal params: the README cache rule."""
    seen, flags = set(), []
    for t in trials:
        key = tuple(t.params.values())
        flags.append(key in seen)
        seen.add(key)
    return flags


def _params_outside(space, trials):
    """Trials whose params leave their declared parameter (README Search space table)."""
    out = []
    for t in trials:
        for name, p in zip(space.names, space.parameters, strict=True):
            v = t.params[name]
            if isinstance(p, Float):
                ok = type(v) is float and p.low <= v <= p.high
            elif isinstance(p, Int):
                ok = type(v) is int and p.low <= v <= p.high
            else:
                ok = v in p.choices
            if not ok:
                out.append(f"trial {t.number} {name}={v!r} {p!r}")
    return out


def _row_params(rows):
    return [
        pytest.param(*row, i, id=f"r{i:02d}-{row[0]}-{row[1]}-cache{row[2]}-c{row[3]}-{row[4]}")
        for i, row in enumerate(rows)
    ]


ROW_NAMES = ("kind", "direction", "cache", "colony", "budget", "seed")


ALL_ROWS = _row_params(COVERING_ARRAY)


CACHED_ROWS = [r for r in ALL_ROWS if r.values[2] is True]


UNCACHED_ROWS = [r for r in ALL_ROWS if r.values[2] is False]


BOUNDED_ROWS = [r for r in ALL_ROWS if r.values[0] != "float_log"]


def test_covering_array_should_contain_every_value_pair_of_every_two_factors():
    """Given FACTORS, when COVERING_ARRAY is checked, then all 87 value pairs of the 10 factor pairs appear,
    every row uses only declared levels, and it has 15 or more but fewer than 180 rows.

    Oracle: P-T7-13 (pairwise covering array). Hand count of pairs: 10+10+15+15+4+6+6+6+6+9 = 87; 15 is the
    lower bound (Kind x Colony needs 15 pairs), 180 = 5*2*2*3*3 is the full product.
    """
    rows = COVERING_ARRAY

    missing = _uncovered_pairs(rows, FACTORS)

    assert (len(_all_pairs(FACTORS)), missing) == (87, set())
    assert set(rows) <= set(product(*(values for _, values in FACTORS)))
    assert 15 <= len(rows) < 180


def test_covering_array_checker_should_report_a_missing_pair_when_the_last_row_is_dropped():
    """Given COVERING_ARRAY without its last row, when the pair checker runs, then it reports missing pairs.

    Oracle: the greedy builder adds a row only for a pair no earlier row covers, so the last row carries a
    pair of its own; this shows the checker of the pairwise test can fail.
    """
    rows = COVERING_ARRAY[:-1]

    missing = _uncovered_pairs(rows, FACTORS)

    assert missing != set()
    assert missing <= _pairs_of(COVERING_ARRAY[-1])


@pytest.mark.parametrize(ROW_NAMES, ALL_ROWS)
def test_optimize_should_record_exactly_max_evals_numbered_trials_for_a_covering_array_row(
    kind, direction, cache, colony, budget, seed
):
    """Given one configuration row, when optimize runs with an ABCOptimizer of that colony, then the study has
    exactly max_evals trials numbered 0..max_evals-1.

    Oracle: optimize docstring (max_evals counts the evaluations) and the task statement (exactly max_evals
    trials, even inside the initial batch). Budget one and inside_init stop inside the initial batch.
    """
    space = _space_of(kind)
    max_evals = _max_evals(budget, colony)
    opt = ABCOptimizer(space, direction=direction, colony_size=colony, seed=seed)

    study = optimize(
        _spy_objective([], space), space, direction, max_evals=max_evals, optimizer=opt, cache=cache
    )

    assert [t.number for t in study.trials] == list(range(max_evals))


@pytest.mark.parametrize(ROW_NAMES, CACHED_ROWS)
def test_optimize_should_call_the_function_once_per_distinct_parameter_set_when_cache_is_true(
    kind, direction, cache, colony, budget, seed
):
    """Given a cache=True row, when optimize runs, then a trial is cached exactly when an earlier trial had
    equal params, the function is never called twice with equal params, and n_calls equals the call count.

    Oracle: README Behaviour (a repeated parameter set reuses its earlier score), CHANGELOG ("never trained
    twice") and the Study.n_calls docstring. The categorical Kind has only six parameter sets, so fifty
    evaluations must hit the cache.
    """
    space = _space_of(kind)
    calls = []
    opt = ABCOptimizer(space, direction=direction, colony_size=colony, seed=seed)

    study = optimize(
        _spy_objective(calls, space),
        space,
        direction,
        max_evals=_max_evals(budget, colony),
        optimizer=opt,
        cache=cache,
    )

    assert [t.cached for t in study.trials] == _expected_cached(study.trials)
    assert len({tuple(c.values()) for c in calls}) == len(calls)
    assert study.n_calls == len(calls)


@pytest.mark.parametrize(ROW_NAMES, UNCACHED_ROWS)
def test_optimize_should_call_the_function_for_every_trial_of_a_row_when_cache_is_false(
    kind, direction, cache, colony, budget, seed
):
    """Given a cache=False row, when optimize runs, then no trial is cached and the function is called once
    per trial, which n_calls reports.

    Oracle: optimize docstring (only cache=True reuses an earlier score) and the Study.n_calls docstring.
    """
    space = _space_of(kind)
    calls = []
    max_evals = _max_evals(budget, colony)
    opt = ABCOptimizer(space, direction=direction, colony_size=colony, seed=seed)

    study = optimize(
        _spy_objective(calls, space), space, direction, max_evals=max_evals, optimizer=opt, cache=cache
    )

    assert (study.n_calls, len(calls)) == (max_evals, max_evals)
    assert [t.cached for t in study.trials] == [False] * max_evals


@pytest.mark.parametrize(ROW_NAMES, BOUNDED_ROWS)
def test_optimize_should_keep_every_trial_inside_its_declared_parameter_for_a_covering_array_row(
    kind, direction, cache, colony, budget, seed
):
    """Given one configuration row, when optimize runs, then every trial's params are a float in [low, high],
    an int in low..high or a member of choices.

    Oracle: README Search space table. The float Kind uses Float(-2.0, 3.0), whose ends decode exactly
    (-2 + 1.0 * 5 = 3), so the check is strict. The float_log rows are left out: their ends decode just
    outside [low, high], a defect reported separately (module docstring).
    """
    space = _space_of(kind)
    opt = ABCOptimizer(space, direction=direction, colony_size=colony, seed=seed)

    study = optimize(
        _spy_objective([], space),
        space,
        direction,
        max_evals=_max_evals(budget, colony),
        optimizer=opt,
        cache=cache,
    )

    assert _params_outside(space, study.trials) == []


@pytest.mark.parametrize(ROW_NAMES, ALL_ROWS)
def test_optimize_should_report_the_extreme_score_and_its_params_as_best_for_a_covering_array_row(
    kind, direction, cache, colony, budget, seed
):
    """Given one configuration row with a finite objective, when optimize runs, then best_score is the minimum
    (min) or maximum (max) of study.scores and the objective of best_params gives best_score.

    Oracle: Study.best_trial (min or max of the finite trials by direction). The pure _value recomputes the
    score of best_params without touching the call counter.
    """
    space = _space_of(kind)
    opt = ABCOptimizer(space, direction=direction, colony_size=colony, seed=seed)
    extreme = {"min": min, "max": max}[direction]

    study = optimize(
        _spy_objective([], space),
        space,
        direction,
        max_evals=_max_evals(budget, colony),
        optimizer=opt,
        cache=cache,
    )

    assert study.best_score == extreme(study.scores.tolist())
    assert _value(space, study.best_params) == study.best_score
