"""Seeded fuzz tests: the library is run on many abnormal inputs, looking for something bad to happen.

Purpose. The normal-use tests (test_space.py, test_abc.py) feed valid, typical inputs and check results.
This module has a different goal: it feeds malformed, extreme and misused inputs and watches for
undocumented exception types and broken invariants. It copies none of their assertions.

Input channels, each with a generator shaped to its type:
(a) declarations: Float/Int/Categorical arguments, Space mapping keys and values, ABCOptimizer
    colony_size/limit/direction/seed and optimize max_evals/direction/optimizer/seed
    (_gen_invalid_parameter_case, _gen_invalid_space_case, _gen_invalid_optimizer_case,
    _gen_invalid_optimize_case, and the single-edit README mutants in OPERATORS);
(b) unit-cube arrays for decode/rows/to_dicts, as float arrays (n, d) with entries in [0, 1] and as a 1-D
    row of length d (_gen_unit_array), and for move a coordinate j with a new unit value u_j (_gen_move);
(c) scores told to ABCOptimizer.tell or returned by the objective: Python floats, np.float64 and np.float32
    mixed with NaN, +-inf, -0.0, +-float max, 1e-308 and 5e-324 (_gen_scores); for NAN_FIRST_SEEDS the first
    score told to ABCOptimizer is NaN;
(d) user callables, i.e. objectives (_objective_from_pool, _objective_raising_on_call, _objective_returning).

Oracle. _run_case(target, *args, **kwargs) is the only runner for every target (Space methods, operators,
ask/tell, optimize, constructors). It records ('ok', value) or ('error', type, message), so the fuzzer itself
never crashes. ORACLE maps each channel to the exception types the spec allows: invalid declarations raise
ValueError or TypeError (README, space.py, abc.py, base.py, facade.py); protocol misuse raises RuntimeError or
ValueError (ABCOptimizer ask/tell); valid unit arrays, scores and operator inputs allow no exception; a
non-numeric objective return may give only ValueError or TypeError. User exceptions must propagate unchanged,
which is checked by identity with pytest.raises, not through the table. _classify turns every outcome into
'ok', 'expected error' or 'finding'; any other library-originated type (IndexError, ZeroDivisionError,
OverflowError, KeyError, AttributeError, ...) is a finding. Cases are collected as (id, input, outcome)
records and asserted after the loop, with one line 'Test <hash><id> IN=<repr> OUT=<ok|Type: msg>' per failing
case, so the seed and the case id reproduce it.

Budget and limits. SEEDS is a fixed tuple of arbitrary seeds and N_CASES a fixed count per seed; every rng is
np.random.default_rng(seed) with seed from SEEDS, and stopping is by budget only, never by a clock. The fuzz
tests are a bounded random search, not a proof of correctness, and they make no coverage claim.

Unspecified, therefore not asserted:
- unit arrays outside the domain (NaN, values below 0 or above 1, wrong width, a 1-D row of length 0);
- tell() with a mix of valid and invalid candidates (it is applied partially) and tell([], []);
- the optimizer state before the first tell;
- seed=-1 (numpy rejects it, the spec only types seed as int) and max_evals that is a float, NaN or inf;
- an objective returning a numeric string such as '1.0' (only 'no undocumented exception type' is checked)
  or an int too large for float (float() raises OverflowError; the return type is only annotated as float);
- np.float32 passed to fitness_scalar (the argument is typed float) and NaN given to the fitness formulas
  beyond agreement of fitness_scalar with fitness;
- which of ValueError and TypeError a particular invalid declaration raises (membership is asserted).

Known library deviations, reported as defects and therefore not asserted here: Float values decoded at the
edge units 0.0, nextafter(0, 1), nextafter(1, 0) and 1.0 can leave [low, high] (so the Float bounds monitor
skips exactly those units); onlooker_probabilities can exceed 1.0 or miss a maximum of exactly 1.0 (so its
upper bound is not monitored); Space accepts infinite, huge or non-integer bounds and rejects non-Mapping
containers with AttributeError. Which exception optimize raises for an optimizer argument that is not an
Optimizer at all (a string or an int) is unspecified, so it is not generated either. None of those inputs is
generated here.

Input space: Every value a caller can pass to the public entry points, which is unbounded.
Subset: Seeded malformed, extreme and misused values for each input channel (declarations, unit-cube arrays,
scores and protocol use), with the forbidden outcomes checked after the loop.
"""

import math
import sys
from types import MappingProxyType

import numpy as np
import pytest

from metahpo import ABCOptimizer, Candidate, Categorical, Float, Int, Space, optimize
from metahpo.operators import fitness, fitness_scalar, onlooker_probabilities

SEEDS = (0, 1, 2, 3, 5, 8, 13, 21)
NAN_FIRST_SEEDS = SEEDS[::2]
N_CASES = 40
N_ROUNDS = 150
N_ACTIONS = 60
D_MAX = 6
N_MAX = 5
MAX_EVALS_MAX = 40
MUTANT_MAX_EVALS = 20
SMALL_BUDGET = 20
README_SEED = 1
ANY_SEED = 0
IRRELEVANT_SCORE = 0.0
HASH = chr(35)
BACKSLASH = chr(92)
FLOAT_MAX = sys.float_info.max

EDGE_U = (0.0, math.nextafter(0.0, 1.0), math.nextafter(1.0, 0.0), 1.0, 0.5)
FLOAT_EDGE_U = frozenset(EDGE_U[:4])
ABNORMAL_SCORES = (math.nan, math.inf, -math.inf, -0.0, FLOAT_MAX, -FLOAT_MAX, 1e-308, 5e-324)
KINDS = ("float", "float_log", "int", "int_log", "categorical")
CHOICE_POOL = (
    "relu",
    "",
    "\x00",
    "a\nb",
    "%s %x %n",
    "{0}",
    ";",
    BACKSLASH,
    '"',
    "\x7f",
    "DROP TABLES",
    None,
    math.nan,
    -0.0,
    0,
    (1, 2),
)

CHECKLIST = MappingProxyType(
    {
        "1 long, short and empty": ("x" * 10_000, ""),
        "2 integer min/max, zero and negative": (0, -1, -5, -(2**31), 2**31 - 1, 2**53, -0.0, FLOAT_MAX),
        "3 special values and characters": ("\x00", "\n", "\r\n", "\x04", "\x1a", None, math.nan),
        "4 format strings": ("%s %x %n", "{0}", "{}"),
        "5 separators, quotes and non-printables": (";", "/", BACKSLASH, '"', "'", "\x07", "\x7f"),
        "6 application keywords": (
            "return",
            "lambda",
            "None",
            "True",
            "nan",
            "inf",
            "min",
            "max",
            "halt",
            "DROP TABLES",
        ),
    }
)

ORACLE = MappingProxyType(
    {
        "declaration": (ValueError, TypeError),
        "protocol": (RuntimeError, ValueError),
        "unit_array": (),
        "score": (),
        "operator": (),
        "user_return": (ValueError, TypeError),
    }
)

NON_PARAMETERS = (1, 2.5, (0, 1), None, "Float", Float, [0, 1])
INVALID_OPTIMIZER_ARGS = MappingProxyType(
    {
        "direction": ("MIN", None, "max ", "", 1),
        "colony_size": (3, 0, -1, None),
        "limit": (0, -1, None),
        "seed": ("a", 1.5),
    }
)
INVALID_OPTIMIZE_ARGS = MappingProxyType(
    {
        "max_evals": (0, -1, None),
        "direction": ("MIN", None, "max ", ""),
        "seed": ("a", 1.5),
    }
)


class FuzzSentinel(Exception):
    """An exception type only the test objective raises, so it can never come from the library."""


def _run_case(target, *args, **kwargs):
    """Runs one case on any target and records ('ok', value) or ('error', type, message); never raises."""
    try:
        return ("ok", target(*args, **kwargs))
    except Exception as exc:
        return ("error", type(exc), str(exc))


def _classify(channel, outcome):
    """Maps an outcome to 'ok', 'expected error' (a type ORACLE allows on the channel) or 'finding'."""
    if outcome[0] == "ok":
        return "ok"
    return "expected error" if issubclass(outcome[1], ORACLE[channel]) else "finding"


def _describe(outcome):
    """The OUT field of a report line: 'ok' or 'Type: message'."""
    return "ok" if outcome[0] == "ok" else f"{outcome[1].__name__}: {outcome[2]}"


def _line(case_id, case_input, outcome, why):
    """One report line naming the case id, its input, its observed outcome and the violated monitor."""
    return f"Test {HASH}{case_id} IN={case_input!r} OUT={_describe(outcome)} WHY={why}"


def _same_float(a, b):
    """Equality of two floats where NaN equals NaN."""
    return a == b or (math.isnan(a) and math.isnan(b))


def _expected_best(told, direction):
    """Best score per the Optimizer docstring: extreme finite score, or the worst value if none is finite."""
    finite = [s for s in told if math.isfinite(s)]
    if not finite:
        return math.inf if direction == "min" else -math.inf
    return min(finite) if direction == "min" else max(finite)


def _gen_parameter(rng, kind):
    """One valid parameter of the given kind, with random moderate bounds or random pool choices."""
    if kind == "float":
        digits = int(rng.integers(1, 4))
        low, high = sorted(round(float(rng.uniform(-10, 10)), digits) for _ in range(2))
        return Float(low, high if high > low else low + 1.0)
    if kind == "float_log":
        low = float(10 ** rng.uniform(-6, 2))
        return Float(low, low * float(10 ** rng.uniform(0.01, 4)), log=True)
    if kind == "int":
        low = int(rng.integers(-50, 50))
        return Int(low, low + int(rng.integers(0, 50)))
    if kind == "int_log":
        low = int(rng.integers(1, 51))
        return Int(low, low + int(rng.integers(0, 501)), log=True)
    picks = rng.choice(len(CHOICE_POOL), size=int(rng.integers(1, 7)), replace=False)
    return Categorical([CHOICE_POOL[int(i)] for i in picks])


def _gen_declared(rng):
    """A random valid declaration {name: parameter} with 1..D_MAX parameters mixing the five kinds."""
    d = int(rng.integers(1, D_MAX + 1))
    return {f"p{j}": _gen_parameter(rng, KINDS[int(rng.integers(len(KINDS)))]) for j in range(d)}


def _gen_unit_array(rng, d):
    """A unit-cube float array, (n, d) with n in 0..N_MAX or a 1-D row of length d, sprinkled with EDGE_U."""
    shape = (d,) if rng.random() < 0.25 else (int(rng.integers(0, N_MAX + 1)), d)
    u = rng.random(shape)
    edge = rng.random(shape) < 0.3
    u[edge] = rng.choice(np.array(EDGE_U), size=int(edge.sum()))
    return u


def _gen_scores(rng, n):
    """n scores mixing ABNORMAL_SCORES, np.float64, np.float32 (also NaN and inf) and plain Python floats."""
    scores = []
    for _ in range(n):
        r = rng.random()
        if r < 0.4:
            scores.append(ABNORMAL_SCORES[int(rng.integers(len(ABNORMAL_SCORES)))])
        elif r < 0.55:
            scores.append(np.float64(rng.normal()))
        elif r < 0.65:
            scores.append(np.float32(rng.normal()))
        elif r < 0.7:
            scores.append(np.float32(math.nan) if rng.random() < 0.5 else np.float32(math.inf))
        else:
            scores.append(float(rng.normal(0, 100)))
    return scores


def _objective_from_pool(pool):
    """A spy objective returning pool[k] on its k-th call; the returned list records every call."""
    calls = []

    def objective(params):
        calls.append(params)
        return pool[len(calls) - 1]

    return objective, calls


def _objective_raising_on_call(call_index, exc):
    """A stub objective returning IRRELEVANT_SCORE until its call number call_index, where it raises exc."""
    calls = []

    def objective(params):
        calls.append(params)
        if len(calls) - 1 == call_index:
            raise exc
        return IRRELEVANT_SCORE

    return objective


def _objective_returning(value):
    """A stub objective that always returns value, whatever the parameters."""

    def objective(params):
        return value

    return objective


def _dummy_objective(params):
    """A dummy objective for cases rejected before any evaluation."""
    return IRRELEVANT_SCORE


def _objective_sum_of_numbers(params):
    """A finite objective that works for any keys: the sum of the int and float parameter values."""
    return float(sum(v for v in params.values() if type(v) in (int, float)))


def _candidate_problems(space, cands):
    """Candidate contract violations: x of shape (d,) in [0, 1], params and key equal to a full decode."""
    problems = []
    for c in cands:
        keys, params = space.rows(c.x)
        if c.x.shape != (space.dim,) or not (0.0 <= c.x.min() and c.x.max() <= 1.0):
            problems.append(f"candidate {c.id} x={c.x!r} not in the unit cube with shape (d,)")
        if c.params != params[0] or c.key != keys[0]:
            problems.append(f"candidate {c.id} params/key {c.params!r}/{c.key!r} differ from the decode")
    return problems


def _value_problems(declared, row_u, params, check_float):
    """Violations of the README value rules for one decoded row; Float edge units are not checked."""
    problems = []
    for j, (name, p) in enumerate(declared.items()):
        v, uj = params[name], float(row_u[j])
        if isinstance(p, Int) and not check_float:
            in_range = type(v) is int and p.low <= v <= p.high
            edges = (uj != 0.0 or v == p.low) and (uj != 1.0 or v == p.high)
            if not (in_range and edges):
                problems.append(f"{name}={v!r} at u={uj!r} breaks Int{(p.low, p.high)}")
        if isinstance(p, Categorical) and not check_float:
            member = any(v is c for c in p.choices)
            edges = (uj != 0.0 or v is p.choices[0]) and (uj != 1.0 or v is p.choices[-1])
            if not (member and edges):
                problems.append(f"{name}={v!r} at u={uj!r} is not the declared choice object")
        if isinstance(p, Float) and check_float and uj not in FLOAT_EDGE_U:
            if not (type(v) is float and math.isfinite(v) and p.low <= v <= p.high):
                problems.append(f"{name}={v!r} at u={uj!r} outside Float{(p.low, p.high)}")
    return problems


def _gen_move(rng, d):
    """A coordinate j in 0..d-1 and a new unit value u_j, an EDGE_U value or a random interior one."""
    j = int(rng.integers(d))
    edge = EDGE_U[int(rng.integers(len(EDGE_U)))]
    interior = float(rng.random())
    return j, (edge if rng.random() < 0.5 else interior)


def _move_and_rows(space, row, j, u_j):
    """move applied to the decoded first row, next to rows() of that row with coordinate j set to u_j."""
    keys, params = space.rows(row)
    changed = row.copy()
    changed[j] = u_j
    ref_keys, ref_params = space.rows(changed)
    return space.move(keys[0], params[0], j, u_j), (ref_keys[0], ref_params[0])


def _collect_unit_array_cases(seed):
    """Records (id, (declared, u, move), outcomes of Space, decode, rows, to_dicts, move) for N_CASES cases.

    The move outcome is recorded only when u has at least one row; it moves the first row.
    """
    rng = np.random.default_rng(seed)
    records = []
    for case in range(N_CASES):
        declared = _gen_declared(rng)
        built = _run_case(Space, declared)
        u = _gen_unit_array(rng, len(declared))
        j, u_j = _gen_move(rng, len(declared))
        outcomes = {"Space": built[:1] if built[0] == "ok" else built}
        if built[0] == "ok":
            outcomes.update(
                {name: _run_case(getattr(built[1], name), u) for name in ("decode", "rows", "to_dicts")}
            )
        if built[0] == "ok" and u.size:
            outcomes["move"] = _run_case(_move_and_rows, built[1], np.atleast_2d(u)[0], j, u_j)
        records.append((f"{seed}-{case}", (declared, u, (j, u_j)), outcomes))
    return records


def _unit_array_failures(seeds, monitor):
    """Report lines of every generated unit-array case whose outcome is not 'ok' or breaks the monitor."""
    failures = []
    for seed in seeds:
        for case_id, case_input, outcomes in _collect_unit_array_cases(seed):
            bad = [(name, o) for name, o in outcomes.items() if _classify("unit_array", o) != "ok"]
            failures += [_line(case_id, case_input, o, f"{name} did not finish") for name, o in bad]
            if not bad:
                failures += [
                    _line(case_id, case_input, ("ok",), why) for why in monitor(case_input, outcomes)
                ]
    return failures


def _structure_monitor(case_input, outcomes):
    """Shapes and consistency: decode (n, d), keys are raw rows, to_dicts equals rows params, key order."""
    declared, u, _ = case_input
    n = 1 if u.ndim == 1 else u.shape[0]
    raw = outcomes["decode"][1]
    keys, params = outcomes["rows"][1]
    dicts = outcomes["to_dicts"][1]
    problems = []
    if raw.shape != (n, len(declared)):
        problems.append(f"decode shape {raw.shape}, expected {(n, len(declared))}")
    if keys != [tuple(r) for r in raw.tolist()]:
        problems.append("rows keys differ from the decoded raw rows")
    if dicts != params or len(dicts) != n:
        problems.append("to_dicts differs from rows params or has the wrong length")
    if not all(
        len(p) == len(declared) and all(a is b for a, b in zip(p, declared, strict=True)) for p in dicts
    ):
        problems.append("params keys are not the declared key objects in declaration order")
    return problems


def _int_and_categorical_monitor(case_input, outcomes):
    """Int values are Python ints in low..high (u=0 gives low, u=1 gives high); choices by identity."""
    declared, u, _ = case_input
    rows = np.atleast_2d(u)
    return [
        why
        for row_u, p in zip(rows, outcomes["to_dicts"][1], strict=False)
        for why in _value_problems(declared, row_u, p, False)
    ]


def _float_monitor(case_input, outcomes):
    """Float values are finite Python floats in [low, high] for every unit value outside FLOAT_EDGE_U."""
    declared, u, _ = case_input
    rows = np.atleast_2d(u)
    return [
        why
        for row_u, p in zip(rows, outcomes["to_dicts"][1], strict=False)
        for why in _value_problems(declared, row_u, p, True)
    ]


def _move_monitor(case_input, outcomes):
    """move(key, params, j, u_j) equals rows() of the row with x_j = u_j: same key, params and value types."""
    if "move" not in outcomes:
        return []
    (moved_key, moved_params), (ref_key, ref_params) = outcomes["move"][1]
    problems = []
    if moved_key != ref_key or moved_params != ref_params:
        problems.append(f"move gave {(moved_key, moved_params)!r}, rows gave {(ref_key, ref_params)!r}")
    if [type(v) for v in moved_params.values()] != [type(v) for v in ref_params.values()]:
        problems.append("move and rows give values of different types")
    return problems


def _abc_score_failures(seeds, direction):
    """Drives ABCOptimizer by ask/tell with generated abnormal scores and reports every monitor violation."""
    failures = []
    for seed in seeds:
        space, records = _collect_abc_score_cases(seed, direction)
        for case_id, scores, outcomes, best, expected_best in records:
            bad = [(name, o) for name, o in outcomes.items() if _classify("score", o) != "ok"]
            failures += [_line(case_id, scores, o, f"{name} raised") for name, o in bad]
            cands = outcomes["ask"][1] if outcomes["ask"][0] == "ok" else []
            failures += [_line(case_id, scores, ("ok",), why) for why in _candidate_problems(space, cands)]
            if not bad and best != expected_best:
                failures.append(_line(case_id, scores, ("ok",), f"best_score {best!r} != {expected_best!r}"))
    return failures


def _collect_abc_score_cases(seed, direction):
    """Records (id, scores, ask/tell outcomes, best_score, expected best) for N_ROUNDS ask/tell rounds.

    For seeds in NAN_FIRST_SEEDS the very first score told is NaN, so a NaN that is not mapped to the worst
    value would become the best score before any finite score arrives.
    """
    rng = np.random.default_rng(seed)
    space = Space(_gen_declared(rng))
    colony_size, limit = int(rng.integers(4, 13)), int(rng.integers(1, 6))
    opt = ABCOptimizer(space, direction, colony_size=colony_size, limit=limit, seed=seed)
    told, records = [], []
    for step in range(N_ROUNDS):
        asked = _run_case(opt.ask)
        cands = asked[1] if asked[0] == "ok" else []
        scores = _gen_scores(rng, len(cands))
        if step == 0 and seed in NAN_FIRST_SEEDS:
            scores[:1] = [math.nan]
        told_outcome = _run_case(opt.tell, cands, scores)
        told += [float(s) for s in scores] if told_outcome[0] == "ok" else []
        outcomes = {"ask": asked, "tell": told_outcome}
        records.append(
            (f"{seed}-{direction}-{step}", scores, outcomes, opt.best_score, _expected_best(told, direction))
        )
        if asked[0] != "ok" or told_outcome[0] != "ok":
            break
    return space, records


def _collect_optimize_score_case(seed, direction):
    """One optimize run whose objective returns abnormal scores; the budget may end inside the init batch."""
    rng = np.random.default_rng(seed)
    declared = _gen_declared(rng)
    space = Space(declared)
    budget = int(rng.integers(1, MAX_EVALS_MAX))
    pool = _gen_scores(rng, budget)
    objective, calls = _objective_from_pool(pool)
    outcome = _run_case(optimize, objective, space, direction=direction, max_evals=budget, seed=seed)
    return space, (f"{seed}-{direction}", (declared, budget, pool), outcome, len(calls))


def _optimize_score_failures(seeds, direction):
    """Report lines of optimize runs breaking the budget, cache, n_calls or best-score rules."""
    failures = []
    for seed in seeds:
        space, (case_id, case_input, outcome, n_called) = _collect_optimize_score_case(seed, direction)
        if _classify("score", outcome) != "ok":
            failures.append(_line(case_id, case_input, outcome, "optimize raised"))
            continue
        failures += [
            _line(case_id, case_input, outcome, why)
            for why in _study_problems(space, case_input, outcome[1], n_called)
        ]
    return failures


def _study_problems(space, case_input, study, n_called):
    """Violations of exact budget, n_calls, cache reuse and best score for one Study."""
    _, budget, _ = case_input
    problems = []
    if len(study.trials) != budget:
        problems.append(f"{len(study.trials)} trials for max_evals={budget}")
    if not (study.n_calls == n_called <= budget):
        problems.append(f"n_calls {study.n_calls}, objective called {n_called} times, budget {budget}")
    keys = space.rows(np.array([t.x for t in study.trials]))[0]
    for i, t in enumerate(study.trials):
        first = keys.index(keys[i])
        repeat_ok = t.cached and _same_float(t.score, study.trials[first].score)
        if (first < i and not repeat_ok) or (first == i and t.cached):
            problems.append(f"trial {i} cached={t.cached} breaks cache reuse of trial {first}")
    finite = [t.score for t in study.trials if math.isfinite(t.score)]
    if finite and study.best_score != _expected_best(finite, study.direction):
        problems.append(f"best_score {study.best_score!r} is not the extreme finite score")
    if not finite and _run_case(lambda: study.best_trial)[:2] != ("error", ValueError):
        problems.append("best_trial did not raise ValueError although no trial is finite")
    return problems


def _gen_operator_values(rng):
    """Minimisation values: the abnormal mix plus normal and log-uniform magnitudes of both signs."""
    base = [0.0, -0.0, 1.0, -1.0, 0.1, -0.2, -3.5, math.inf, -math.inf, math.nan]
    base += [FLOAT_MAX, -FLOAT_MAX, 1e-308, 5e-324, -5e-324]
    normal = [float(v) for v in rng.normal(0, 10, size=N_CASES)]
    sign = rng.choice(np.array([-1.0, 1.0]), size=N_CASES)
    magnitudes = [float(s * 10**e) for s, e in zip(sign, rng.uniform(-300, 300, size=N_CASES), strict=True)]
    return base + normal + magnitudes


def _fitness_agreement_failures(seeds):
    """Report lines where fitness_scalar(v) and fitness([v])[0] disagree (NaN equals NaN) or raise."""
    failures = []
    for seed in seeds:
        values = _gen_operator_values(np.random.default_rng(seed))
        for case, v in enumerate(values + [np.float64(x) for x in values]):
            scalar = _run_case(fitness_scalar, v)
            array = _run_case(fitness, [v])
            case_id = f"{seed}-{case}"
            bad = [o for o in (scalar, array) if _classify("operator", o) != "ok"]
            failures += [_line(case_id, v, o, "operator raised") for o in bad]
            if not bad and not _same_float(float(scalar[1]), float(array[1][0])):
                failures.append(
                    _line(case_id, v, scalar, f"fitness_scalar {scalar[1]!r} != fitness {array[1][0]!r}")
                )
    return failures


def _gen_fitness_array(rng):
    """A fitness array made by fitness() from 1..8 finite or +inf minimisation values, as ABC produces."""
    size = int(rng.integers(1, 9))
    pool = [v for v in _gen_operator_values(rng) if not math.isnan(v) and v != -math.inf]
    if rng.random() < 0.15:
        return fitness(np.full(size, math.inf))
    return fitness(np.array([pool[int(i)] for i in rng.integers(len(pool), size=size)]))


def _probability_failures(seeds):
    """Report lines where onlooker probabilities are not finite, below 0.1, out of order or not all ones."""
    failures = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for case in range(N_CASES):
            fit = _gen_fitness_array(rng)
            outcome = _run_case(onlooker_probabilities, fit)
            case_id = f"{seed}-{case}"
            if _classify("operator", outcome) != "ok":
                failures.append(_line(case_id, fit, outcome, "operator raised"))
                continue
            failures += [_line(case_id, fit, outcome, why) for why in _probability_problems(fit, outcome[1])]
    return failures


def _probability_problems(fit, p):
    """Monitors of p = 0.9 * fit / max(fit) + 0.1 that hold whatever the evaluation order."""
    problems = []
    if p.shape != fit.shape or not np.all(np.isfinite(p)) or p.min() < 0.1:
        problems.append(f"p={p.tolist()!r} not finite, of the wrong shape or below 0.1")
    order = np.argsort(fit, kind="stable")
    if np.any(np.diff(p[order]) < 0):
        problems.append(f"p={p.tolist()!r} does not preserve the order of fit")
    if np.all(fit == 0.0) and p.tolist() != [1.0] * fit.size:
        problems.append(f"all-zero fitness gave p={p.tolist()!r}, expected all ones")
    return problems


def _gen_invalid_parameter_case(rng):
    """A Float, Int or Categorical declaration with exactly one invalid argument, the others random valid."""
    low = round(float(rng.uniform(-10, 10)), 3)
    high = low + round(float(rng.uniform(0.001, 10)), 3)
    pos_low = round(float(rng.uniform(0.001, 10)), 3)
    pos_high = pos_low + round(float(rng.uniform(0.001, 10)), 3)
    ilow = int(rng.integers(-50, 50))
    ihigh = ilow + int(rng.integers(0, 50))
    cases = (
        ("Float low > high", Float, (high, low), {}),
        ("Float low == high", Float, (low, low), {}),
        ("Float NaN low", Float, (math.nan, high), {}),
        ("Float NaN high", Float, (low, math.nan), {}),
        ("Float None low", Float, (None, high), {}),
        ("Float str high", Float, (low, "b"), {}),
        ("log Float low 0", Float, (0.0, pos_high), {"log": True}),
        ("log Float low < 0", Float, (-pos_low, pos_high), {"log": True}),
        ("Int low > high", Int, (ihigh + 1 + int(rng.integers(0, 10)), ihigh), {}),
        ("Int NaN low", Int, (math.nan, ihigh), {}),
        ("Int NaN high", Int, (ilow, math.nan), {}),
        ("Int None high", Int, (ilow, None), {}),
        ("log Int low 0", Int, (0, abs(ihigh) + 1), {"log": True}),
        ("log Int low < 0", Int, (-abs(ilow) - 1, abs(ihigh) + 1), {"log": True}),
        ("Categorical empty list", Categorical, ([],), {}),
        ("Categorical empty string", Categorical, ("",), {}),
        ("Categorical no choices", Categorical, (), {}),
        ("Categorical None", Categorical, (None,), {}),
        ("Categorical int", Categorical, (5,), {}),
    )
    return cases[int(rng.integers(len(cases)))]


def _gen_invalid_space_case(rng):
    """A Space mapping with one value replaced by a non-parameter, or an empty or None mapping."""
    declared = _gen_declared(rng)
    name = list(declared)[int(rng.integers(len(declared)))]
    bad = NON_PARAMETERS[int(rng.integers(len(NON_PARAMETERS)))]
    whole = (("empty mapping", {}), ("None mapping", None), ("empty list", []))
    if rng.random() < 0.25:
        label, mapping = whole[int(rng.integers(len(whole)))]
        return (label, Space, (mapping,), {})
    return (f"Space value {name}={bad!r}", Space, ({**declared, name: bad},), {})


def _gen_invalid_optimizer_case(rng):
    """ABCOptimizer arguments with exactly one of direction, colony_size, limit, seed invalid."""
    space = Space(_gen_declared(rng))
    kwargs = {
        "direction": ("min", "max")[int(rng.integers(2))],
        "colony_size": int(rng.integers(4, 13)),
        "limit": int(rng.integers(1, 6)),
        "seed": int(rng.integers(0, 2**31)),
    }
    slot = sorted(INVALID_OPTIMIZER_ARGS)[int(rng.integers(len(INVALID_OPTIMIZER_ARGS)))]
    values = INVALID_OPTIMIZER_ARGS[slot]
    kwargs[slot] = values[int(rng.integers(len(values)))]
    return (f"ABCOptimizer {slot}={kwargs[slot]!r}", ABCOptimizer, (space,), kwargs)


def _gen_invalid_optimize_case(rng):
    """optimize arguments with exactly one of max_evals, direction, optimizer, seed invalid."""
    declared = _gen_declared(rng)
    space = Space(declared)
    direction = ("min", "max")[int(rng.integers(2))]
    seed = int(rng.integers(0, 2**31))
    kwargs = {"direction": direction, "max_evals": int(rng.integers(1, MAX_EVALS_MAX)), "seed": seed}
    slot = ("max_evals", "direction", "optimizer", "seed")[int(rng.integers(4))]
    if slot == "optimizer":
        other_space = Space({**declared, "extra": Float(0.0, 1.0)})
        opposite = "max" if direction == "min" else "min"
        options = (ABCOptimizer(other_space, direction, seed=seed), ABCOptimizer(space, opposite, seed=seed))
        kwargs["optimizer"] = options[int(rng.integers(2))]
        chosen = kwargs["optimizer"]
        label = f"optimizer on space dim {chosen.space.dim}, direction {chosen.direction}"
    else:
        values = INVALID_OPTIMIZE_ARGS[slot]
        kwargs[slot] = values[int(rng.integers(len(values)))]
        label = f"optimize {slot}={kwargs[slot]!r}"
    return (label, optimize, (_dummy_objective, space), kwargs)


def _collect_declaration_cases(seed, generator):
    """Records (id, (label, args, kwargs), outcome) for N_CASES invalid declarations from generator."""
    rng = np.random.default_rng(seed)
    records = []
    for case in range(N_CASES):
        label, target, args, kwargs = generator(rng)
        records.append((f"{seed}-{case}", (label, args, kwargs), _run_case(target, *args, **kwargs)))
    return records


def _declaration_failures(seeds, generator):
    """Report lines of invalid declarations that were accepted or raised a type outside the oracle."""
    failures = []
    for seed in seeds:
        for case_id, case_input, outcome in _collect_declaration_cases(seed, generator):
            kind = _classify("declaration", outcome)
            if kind != "expected error":
                failures.append(
                    _line(case_id, case_input, outcome, f"{kind}, expected ValueError or TypeError")
                )
    return failures


def _pick_action(rng, pending, told):
    """A random protocol action that is applicable in the shadow state (pending and told candidates)."""
    options = ["ASK"]
    options += ["TELL_ALL", "TELL_LEN"] if pending else []
    options += ["TELL_SUBSET"] if len(pending) >= 2 else []
    options += ["TELL_TOLD"] if told else []
    options += ["TELL_FORGED"] if pending or told else []
    return options[int(rng.integers(len(options)))]


def _collect_protocol_cases(seed):
    """Runs N_ACTIONS random protocol actions against a shadow model, recording outcome and expectation."""
    rng = np.random.default_rng(seed)
    space = Space(_gen_declared(rng))
    direction = ("min", "max")[int(rng.integers(2))]
    colony_size, limit = int(rng.integers(4, 11)), int(rng.integers(1, 6))
    opt = ABCOptimizer(space, direction, colony_size=colony_size, limit=limit, seed=seed)
    pending, told, told_scores, issued, records = [], [], [], [-1], []
    for step in range(N_ACTIONS):
        action = _pick_action(rng, pending, told)
        if action == "ASK":
            expected = ("error", RuntimeError) if pending else ("ok",)
            outcome = _run_case(opt.ask)
            fresh = list(outcome[1]) if outcome[0] == "ok" else []
            pending += fresh
            issued += [c.id for c in fresh]
            payload = [c.id for c in fresh]
        elif action in ("TELL_ALL", "TELL_SUBSET"):
            k = len(pending) if action == "TELL_ALL" else int(rng.integers(1, len(pending)))
            chosen = [pending[int(i)] for i in sorted(rng.choice(len(pending), size=k, replace=False))]
            scores = _gen_scores(rng, k)
            expected = ("ok",)
            outcome = _run_case(opt.tell, chosen, scores)
            done = {c.id for c in chosen} if outcome[0] == "ok" else set()
            told += [c for c in chosen if c.id in done]
            told_scores += [float(s) for c, s in zip(chosen, scores, strict=True) if c.id in done]
            pending = [c for c in pending if c.id not in done]
            payload = ([c.id for c in chosen], scores)
        elif action == "TELL_TOLD":
            cand = told[int(rng.integers(len(told)))]
            expected = ("error", ValueError)
            outcome = _run_case(opt.tell, [cand], _gen_scores(rng, 1))
            payload = cand.id
        elif action == "TELL_FORGED":
            base = (pending + told)[int(rng.integers(len(pending) + len(told)))]
            forged_id = max(issued) + 1000 if rng.random() < 0.5 else -1
            expected = ("error", ValueError)
            outcome = _run_case(
                opt.tell, [Candidate(forged_id, base.x, base.params, base.key)], _gen_scores(rng, 1)
            )
            payload = forged_id
        else:
            delta = 1 if rng.random() < 0.5 else -1
            expected = ("error", ValueError)
            outcome = _run_case(opt.tell, list(pending), _gen_scores(rng, len(pending) + delta))
            payload = (len(pending), delta)
        best_check = (opt.best_score, _expected_best(told_scores, direction)) if told_scores else None
        records.append((f"{seed}-{step}", (action, payload), outcome, expected, best_check))
    return space, records


def _protocol_failures(seeds):
    """Report lines of protocol actions whose outcome differs from the spec or breaks a score monitor."""
    failures = []
    for seed in seeds:
        space, records = _collect_protocol_cases(seed)
        for case_id, case_input, outcome, expected, best_check in records:
            kind = _classify("protocol", outcome)
            if kind == "finding" or outcome[: len(expected)] != expected:
                failures.append(_line(case_id, case_input, outcome, f"{kind}, expected {expected!r}"))
            cands = outcome[1] if case_input[0] == "ASK" and outcome[0] == "ok" else []
            failures += [
                _line(case_id, case_input, outcome, why) for why in _candidate_problems(space, cands)
            ]
            if best_check is not None and best_check[0] != best_check[1]:
                failures.append(
                    _line(case_id, case_input, outcome, f"best_score {best_check[0]!r} != {best_check[1]!r}")
                )
    return failures


def _raw_value(value):
    """A mutation factory that puts value itself into the Space mapping instead of a parameter."""
    return value


README_CONFIG = MappingProxyType(
    {
        "params": (
            ("lr", Float, (1e-4, 1e-2), {"log": True}),
            ("kernel", Categorical, ((3, 5, 7),), {}),
            ("units", Int, (16, 256), {}),
            ("dropout", Float, (0.0, 0.5), {}),
        ),
        "direction": "max",
        "max_evals": MUTANT_MAX_EVALS,
        "colony_size": None,
        "limit": None,
    }
)
SPECIAL_KEYS = tuple(v for item in CHECKLIST.values() for v in item if isinstance(v, str))


def _with_param(config, index, spec):
    """A copy of config with the parameter spec at index replaced, or removed when spec is None."""
    params = list(config["params"])
    params[index : index + 1] = [] if spec is None else [spec]
    return {**config, "params": tuple(params)}


def _pick_index(config, rng, factories):
    """A random index of a parameter whose factory is one of factories."""
    found = [i for i, spec in enumerate(config["params"]) if spec[1] in factories]
    return found[int(rng.integers(len(found)))]


def _op_swap_low_high(config, rng):
    """Swaps low and high of a random Float or Int."""
    i = _pick_index(config, rng, (Float, Int))
    name, factory, (low, high), kw = config["params"][i]
    return _with_param(config, i, (name, factory, (high, low), kw))


def _op_high_equals_low(config, rng):
    """Sets high equal to low on a random Float or Int."""
    i = _pick_index(config, rng, (Float, Int))
    name, factory, (low, _), kw = config["params"][i]
    return _with_param(config, i, (name, factory, (low, low), kw))


def _op_log_low_zero(config, rng):
    """Makes a random Float log-scaled with low 0."""
    i = _pick_index(config, rng, (Float,))
    name, factory, (_, high), kw = config["params"][i]
    return _with_param(config, i, (name, factory, (0.0, high), {**kw, "log": True}))


def _op_log_low_negative(config, rng):
    """Makes a random Float log-scaled with a negative low."""
    i = _pick_index(config, rng, (Float,))
    name, factory, (low, high), kw = config["params"][i]
    return _with_param(config, i, (name, factory, (-abs(low) - 1e-4, high), {**kw, "log": True}))


def _op_nan_bound(config, rng):
    """Replaces one bound of a random Float or Int by NaN."""
    i = _pick_index(config, rng, (Float, Int))
    name, factory, bounds, kw = config["params"][i]
    slot = int(rng.integers(2))
    return _with_param(config, i, (name, factory, bounds[:slot] + (math.nan,) + bounds[slot + 1 :], kw))


def _op_flip_log(config, rng):
    """Flips the log flag of a random Float or Int."""
    i = _pick_index(config, rng, (Float, Int))
    name, factory, bounds, kw = config["params"][i]
    return _with_param(config, i, (name, factory, bounds, {**kw, "log": not kw.get("log", False)}))


def _op_replace_by_tuple(config, rng):
    """Replaces a random parameter by the tuple (0, 1)."""
    i = int(rng.integers(len(config["params"])))
    return _with_param(config, i, (config["params"][i][0], _raw_value, ((0, 1),), {}))


def _op_replace_by_none(config, rng):
    """Replaces a random parameter by None."""
    i = int(rng.integers(len(config["params"])))
    return _with_param(config, i, (config["params"][i][0], _raw_value, (None,), {}))


def _op_replace_by_int(config, rng):
    """Replaces a random parameter by the int 1."""
    i = int(rng.integers(len(config["params"])))
    return _with_param(config, i, (config["params"][i][0], _raw_value, (1,), {}))


def _op_drop_parameter(config, rng):
    """Drops a random parameter."""
    return _with_param(config, int(rng.integers(len(config["params"]))), None)


def _op_empty_choices(config, rng):
    """Empties the choices of the Categorical."""
    i = _pick_index(config, rng, (Categorical,))
    return _with_param(config, i, (config["params"][i][0], Categorical, ((),), {}))


def _op_singleton_choices(config, rng):
    """Keeps one random choice of the Categorical."""
    i = _pick_index(config, rng, (Categorical,))
    name, factory, (choices,), kw = config["params"][i]
    return _with_param(config, i, (name, factory, ((choices[int(rng.integers(len(choices)))],),), kw))


def _op_rename_key_special(config, rng):
    """Renames a random key to a checklist string."""
    i = int(rng.integers(len(config["params"])))
    _, factory, args, kw = config["params"][i]
    return _with_param(config, i, (SPECIAL_KEYS[int(rng.integers(len(SPECIAL_KEYS)))], factory, args, kw))


def _op_bad_direction(config, rng):
    """Sets an invalid direction."""
    return {**config, "direction": ("MIN", None, "")[int(rng.integers(3))]}


def _op_bad_max_evals(config, rng):
    """Sets an invalid max_evals."""
    return {**config, "max_evals": (0, -1, None)[int(rng.integers(3))]}


def _op_bad_colony_size(config, rng):
    """Sets an invalid colony_size on an explicit ABCOptimizer."""
    return {**config, "colony_size": (3, 0, -1, None)[int(rng.integers(4))], "limit": 15}


def _op_bad_limit(config, rng):
    """Sets an invalid limit on an explicit ABCOptimizer."""
    return {**config, "colony_size": 10, "limit": (0, -1, None)[int(rng.integers(3))]}


OPERATORS = MappingProxyType(
    {
        "swap_low_high": _op_swap_low_high,
        "high_equals_low": _op_high_equals_low,
        "log_low_zero": _op_log_low_zero,
        "log_low_negative": _op_log_low_negative,
        "nan_bound": _op_nan_bound,
        "flip_log": _op_flip_log,
        "replace_by_tuple": _op_replace_by_tuple,
        "replace_by_none": _op_replace_by_none,
        "replace_by_int": _op_replace_by_int,
        "drop_parameter": _op_drop_parameter,
        "empty_choices": _op_empty_choices,
        "singleton_choices": _op_singleton_choices,
        "rename_key_special": _op_rename_key_special,
        "bad_direction": _op_bad_direction,
        "bad_max_evals": _op_bad_max_evals,
        "bad_colony_size": _op_bad_colony_size,
        "bad_limit": _op_bad_limit,
    }
)


def _optimize_config(config):
    """Builds the Space (and an ABCOptimizer when colony_size or limit is set) and runs optimize on config."""
    declared = {name: factory(*args, **kw) for name, factory, args, kw in config["params"]}
    space = Space(declared)
    optimizer = None
    if config["colony_size"] is not None or config["limit"] is not None:
        optimizer = ABCOptimizer(
            space,
            config["direction"],
            colony_size=config["colony_size"],
            limit=config["limit"],
            seed=README_SEED,
        )
    study = optimize(
        _objective_sum_of_numbers,
        space,
        direction=config["direction"],
        max_evals=config["max_evals"],
        optimizer=optimizer,
        seed=README_SEED,
    )
    return declared, study


def _collect_mutant_cases(seeds):
    """Records (id, (operator, mutated config), outcome) for every operator, in an rng order per seed."""
    records = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for name in rng.permutation(sorted(OPERATORS)):
            mutant = OPERATORS[str(name)](dict(README_CONFIG), rng)
            records.append((f"{seed}-{name}", (str(name), mutant), _run_case(_optimize_config, mutant)))
    return records


def _mutant_problems(config, declared, study):
    """Monitors of a finished mutant: exact budget, declared keys, README value rules, best is the extreme."""
    problems = []
    if len(study.trials) != config["max_evals"]:
        problems.append(f"{len(study.trials)} trials for max_evals={config['max_evals']}")
    for t in study.trials:
        if list(t.params) != list(declared):
            problems.append(f"trial {t.number} keys {list(t.params)!r}")
        problems += _value_problems(declared, t.x, t.params, False) + _value_problems(
            declared, t.x, t.params, True
        )
    if study.best_score != _expected_best(list(study.scores), config["direction"]):
        problems.append(f"best_score {study.best_score!r} is not the extreme trial score")
    return problems


def _mutant_failures(seeds):
    """Report lines of mutants that raised a type outside the oracle or finished breaking a monitor."""
    failures = []
    for case_id, (name, mutant), outcome in _collect_mutant_cases(seeds):
        kind = _classify("declaration", outcome)
        if kind == "finding":
            failures.append(_line(case_id, (name, mutant), outcome, "finding"))
        if kind == "ok":
            failures += [
                _line(case_id, (name, mutant), outcome, why) for why in _mutant_problems(mutant, *outcome[1])
            ]
    return failures


def _mutant_coverage(seeds):
    """The operator names used and the outcome kinds reached over all mutants."""
    records = _collect_mutant_cases(seeds)
    used = {name for _, (name, _), _ in records}
    kinds = {_classify("declaration", outcome) for _, _, outcome in records}
    return used, kinds


def _fuzz_transcript(seed):
    """repr of every record of the unit-array, protocol and optimize collectors for one seed."""
    unit = _collect_unit_array_cases(seed)
    protocol = _collect_protocol_cases(seed)[1]
    scores = _collect_optimize_score_case(seed, "max")[1]
    return repr((unit, protocol, scores))


def _transcripts(seeds):
    """The fuzz transcript of every seed, in order."""
    return [_fuzz_transcript(seed) for seed in seeds]


def _special_string_space():
    """A Space with every checklist string as a key and every checklist string or object as choices."""
    choices = [v for item in CHECKLIST.values() for v in item if isinstance(v, str)] + [
        None,
        math.nan,
        0,
        -0.0,
    ]
    return Space({key: Categorical(choices) for key in SPECIAL_KEYS}), tuple(choices)


def _special_string_problems(study, choices):
    """Violations of key identity and choice identity over all trials of the special-string run."""
    problems = []
    for t in study.trials:
        if len(t.params) != len(SPECIAL_KEYS) or not all(
            a is b for a, b in zip(t.params, SPECIAL_KEYS, strict=True)
        ):
            problems.append(f"trial {t.number}: keys are not the declared key objects")
        if not all(any(v is c for c in choices) for v in t.params.values()):
            problems.append(f"trial {t.number}: a value is not one of the declared choice objects")
    return problems


def _decode_top_of_int_1_9():
    """Decodes the closed upper unit edge u = 1.0 on Int(1, 9)."""
    return Space({"b": Int(1, 9)}).to_dicts([[1.0]])


def _raise_index_error():
    """A target raising IndexError, a type no oracle channel allows."""
    return [][0]


def _wide_space():
    """A Space of 200 Float parameters (checklist item 1: very long input)."""
    return Space({f"x{j}": Float(0.0, 1.0) for j in range(200)})


def _many_choices_space():
    """A Space with one Categorical of 1000 choices (checklist item 1: very long input)."""
    return Space({"k": Categorical(list(range(1000)))})


TOLD_ORDERS = MappingProxyType({"finite-first": (0, 1), "nan-first": (1, 0)})


def _best_after_one_finite_score(score, direction, order):
    """best_score after telling the 2-source init batch of ABCOptimizer(colony_size=4) score and NaN.

    order gives the positions of (score, NaN) in the told list: (0, 1) tells [score, NaN], (1, 0) tells
    [NaN, score].
    """
    opt = ABCOptimizer(Space({"x": Float(0.0, 1.0)}), direction, colony_size=4, seed=ANY_SEED)
    cands = opt.ask()
    told = [None, None]
    told[order[0]], told[order[1]] = score, math.nan
    opt.tell(cands, told)
    return opt.best_score


def test_space_to_dicts_should_return_known_params_when_smoke_checked_before_fuzzing():
    """Given a mixed Float/Int/Categorical space, when a known row is decoded, then the known params return.

    Smoke check of the target before any fuzzing (P-T7-16). Oracle: README search-space table and the Int and
    Categorical docstrings (equal shares of [0, 1]): u=0.5 on Float(0, 1) is 0.5, u=0.0 on Int(1, 9) is the
    low end 1, u=0.99 on Categorical('xyz') lies in the last third, 'z'. Postcondition: none (pure decode).
    """
    space = Space({"a": Float(0, 1), "b": Int(1, 9), "c": Categorical("xyz")})

    params = space.to_dicts([[0.5, 0.0, 0.99]])

    assert params == [{"a": 0.5, "b": 1, "c": "z"}]


def test_operators_fitness_scalar_should_return_known_values_when_smoke_checked_before_fuzzing():
    """Given f = 3.0 and f = -0.5, when fitness_scalar runs, then the hand-computed 0.25 and 1.5 come back.

    Smoke check of the operator before fuzzing it (P-T7-16). Oracle: operators.fitness docstring, 1 / (1 + 3)
    = 0.25 for f >= 0 and 1 + |-0.5| = 1.5 for f < 0, both exact binary floats. Postcondition: none.
    """
    values = (fitness_scalar(3.0), fitness_scalar(-0.5))

    assert values == (0.25, 1.5)


@pytest.mark.parametrize(
    ("target", "args", "expected"),
    [
        pytest.param(_decode_top_of_int_1_9, (), ("ok", [{"b": 9}]), id="valid-call-records-its-value"),
        pytest.param(Float, (1, 1), ("error", ValueError), id="documented-rejection-records-value-error"),
        pytest.param(
            _raise_index_error, (), ("error", IndexError), id="unlisted-type-is-recorded-not-raised"
        ),
    ],
)
def test_run_case_should_record_the_outcome_without_raising_when_given_a_known_input(target, args, expected):
    """Given a known target, when _run_case runs it, then the recorded (kind, value or type) is the known one.

    Smoke check of the runner before fuzzing (P-T7-16, P-T7-14). Oracle: README (Int low..high inclusive, so
    u=1.0 on Int(1, 9) is 9), Float.__post_init__ (Float(1, 1) violates low < high: ValueError) and the runner
    contract (an exception type outside the oracle is recorded, never raised).
    """
    outcome = _run_case(target, *args)

    assert outcome[:2] == expected


def test_space_should_decode_generated_unit_arrays_to_documented_shapes_and_keys_when_edges_are_sprinkled():
    """Given random mixed spaces and unit arrays with edge values, when decoded, then shapes and keys hold.

    Monitors per case: Space, decode, rows and to_dicts finish; decode gives (n, d), or (1, d) for a 1-D row;
    rows keys are the decoded raw rows; to_dicts equals rows params and has n entries; params keys are the
    declared key objects in order. Oracle: Space.decode, Space.rows and Space.to_dicts docstrings and the
    README unit cube [0, 1]^d (closed, so 0.0 and 1.0 are valid). Edge values 0.0, nextafter(0, 1),
    nextafter(1, 0), 1.0 and 0.5 are boundary inputs; n = 0 and the 1-D row are the batch-shape classes.
    """
    seeds = SEEDS

    failures = _unit_array_failures(seeds, _structure_monitor)

    assert failures == []


def test_space_should_decode_int_and_categorical_to_declared_values_when_unit_arrays_hit_the_edges():
    """Given random spaces and edge-sprinkled unit arrays, when decoded, then Int and Categorical values hold.

    Monitors: an Int value is a Python int in low..high, u=0.0 gives low and u=1.0 gives high (linear and log
    Int, single-value ranges included); a Categorical value is one of the declared choice objects by identity,
    u=0.0 gives the first and u=1.0 the last. The choice pool holds None, NaN, -0.0, 0 and special strings, so
    an equality check could not replace identity. Oracle: README table and the Int/Categorical docstrings
    (every value gets an equal share of [0, 1], so the closed ends map to the first and last value).
    """
    seeds = SEEDS

    failures = _unit_array_failures(seeds, _int_and_categorical_monitor)

    assert failures == []


def test_space_should_keep_float_values_finite_and_inside_bounds_when_unit_values_are_interior():
    """Given random Float bounds, when interior unit values are decoded, then the values stay in range.

    Monitor: every Float value is a finite Python float with low <= v <= high, for every u outside
    FLOAT_EDGE_U. Oracle: README table (Float values are real numbers in [low, high]). The four edge units are
    skipped because the library is known to leave [low, high] there; that is reported as a defect, not hidden
    here.
    """
    seeds = SEEDS

    failures = _unit_array_failures(seeds, _float_monitor)

    assert failures == []


def test_space_move_should_equal_rows_of_the_changed_row_when_coordinate_and_unit_value_are_generated():
    """Given a decoded row, coordinate j and new u_j, when move runs, then it matches rows() of that row.

    Monitor: move(key, params, j, u_j) on the decoded first row gives the same key, the same params and the
    same value types as rows() of that row with x_j replaced by u_j; neither call raises. u_j is an edge unit
    (0.0, nextafter(0, 1), nextafter(1, 0), 1.0, 0.5) or a random interior value, so the closed ends of every
    kind are hit. Oracle: Space.move docstring (the point differs from a decoded point only in coordinate j)
    and Space.rows; the style rule 'move is checked by equality with rows()'.
    """
    seeds = SEEDS

    failures = _unit_array_failures(seeds, _move_monitor)

    assert failures == []


@pytest.mark.parametrize("direction", ["min", "max"])
def test_abc_optimizer_should_keep_candidates_valid_and_best_score_extreme_when_told_abnormal_scores(
    direction,
):
    """Given ABCOptimizer, when abnormal scores are told each round, then candidate and best monitors hold.

    Monitors after every round: ask and tell raise nothing; every Candidate has x of shape (d,) in [0, 1]^d
    and params/key equal to a full decode of x; best_score equals the min ('min') or max ('max') of the finite
    scores told so far, or the worst value (+inf for 'min', -inf for 'max') if none is finite. Oracle: the
    Optimizer docstring (scores in the user's direction, NaN or infinity is the worst value), the Candidate
    docstring and Space.rows. colony_size 4 and limit 1 are reachable minimums, so scouts are frequent. For
    NAN_FIRST_SEEDS the first score told is NaN, the order in which an unmapped NaN would corrupt best_score.
    """
    seeds = SEEDS

    failures = _abc_score_failures(seeds, direction)

    assert failures == []


@pytest.mark.parametrize("direction", ["min", "max"])
def test_optimize_should_keep_budget_cache_and_best_score_rules_when_objective_returns_abnormal_scores(
    direction,
):
    """Given an objective returning abnormal scores, when optimize runs, then budget and cache rules hold.

    Monitors: exactly max_evals trials (max_evals from 1 to 39, so the budget can end inside the initial
    batch); n_calls equals the objective's call count and is at most max_evals; a trial repeating an earlier
    decoded key is cached with the earlier score, a first occurrence is not cached; best_score is the extreme
    finite score, and best_trial raises ValueError when no score is finite. Oracle: optimize docstring and the
    task spec (exact max_evals, cache reuse), README Behaviour bullets and the Study docstrings.
    """
    seeds = SEEDS

    failures = _optimize_score_failures(seeds, direction)

    assert failures == []


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        pytest.param(-0.0, 1.0, id="negative-zero-takes-the-f-ge-0-branch"),
        pytest.param(math.inf, 0.0, id="plus-inf-gives-zero"),
        pytest.param(-math.inf, math.inf, id="minus-inf-gives-1-plus-inf"),
        pytest.param(-3.5, 4.5, id="negative-gives-1-plus-abs"),
        pytest.param(5e-324, 1.0, id="subnormal-rounds-to-one"),
        pytest.param(-5e-324, 1.0, id="negative-subnormal-rounds-to-one"),
        pytest.param(-FLOAT_MAX, FLOAT_MAX, id="minus-float-max-absorbs-the-one"),
        pytest.param(FLOAT_MAX, 5.562684646268003e-309, id="float-max-gives-a-subnormal"),
    ],
)
def test_operators_fitness_scalar_should_return_the_documented_value_when_input_is_abnormal(f, expected):
    """Given an abnormal minimisation value, when fitness_scalar runs, then the hand-computed value returns.

    Oracle: operators.fitness docstring (1 / (1 + f) for f >= 0, 1 + |f| otherwise, +inf gives 0), evaluated
    by hand in IEEE doubles: -0.0 >= 0 holds, 1 + 5e-324 and 1 + FLOAT_MAX round to 1 and FLOAT_MAX, and 1 /
    FLOAT_MAX = 5.562684646268003e-309. Postcondition: none (pure function).
    """
    value = fitness_scalar(f)

    assert value == expected


def test_operators_fitness_scalar_should_agree_with_fitness_when_fed_generated_abnormal_values():
    """Given abnormal, normal, huge or tiny values, when both fitness forms run, then fitness_scalar agrees.

    Monitor: fitness_scalar(v) == float(fitness([v])[0]) for Python floats and np.float64, NaN counting as
    equal, and neither raises. Oracle: fitness_scalar docstring ('fitness for one value'). np.float32 is not
    generated because the argument is typed float.
    """
    seeds = SEEDS

    failures = _fitness_agreement_failures(seeds)

    assert failures == []


def test_operators_onlooker_probabilities_should_be_finite_ordered_and_at_least_0_1_for_generated_fitness():
    """Given fitness arrays from finite and +inf values, when probabilities run, then the monitors hold.

    Monitors: same shape, all finite, every p >= 0.1, p preserves the order of fit, and an all-zero fitness
    (all +inf values) gives exactly all ones. Oracle: onlooker_probabilities docstring (p = 0.9 * fit /
    max(fit) + 0.1, all-zero fitness gives equal probabilities) and Karaboga & Basturk 2007. NaN and -inf are
    not generated because the optimizer never produces them as minimisation values. The upper bound 1.0 is not
    monitored here because the library is known to exceed it; that is reported as a defect.
    """
    seeds = SEEDS

    failures = _probability_failures(seeds)

    assert failures == []


def test_space_declarations_should_raise_only_value_or_type_error_when_one_argument_is_invalid():
    """Given declarations with one invalid part, when they are built, then only ValueError/TypeError occur.

    Inputs: low > high, low == high for Float, NaN bounds, None or str bounds, log with low 0 or negative,
    empty, missing, None or int choices, a non-parameter value in the mapping, and an empty or None mapping.
    Oracle: README and space.py (invalid parameters raise ValueError/TypeError), the Float/Int/Categorical
    __post_init__ rules and Space.__init__. Which of the two types is raised is unspecified.
    """
    seeds = SEEDS

    failures = _declaration_failures(seeds, _gen_invalid_parameter_case) + _declaration_failures(
        seeds, _gen_invalid_space_case
    )

    assert failures == []


def test_abc_optimizer_should_raise_only_value_or_type_error_when_one_constructor_argument_is_invalid():
    """Given ABCOptimizer arguments with one invalid value, when it is built, then only VE or TE occurs.

    Inputs: direction 'MIN', None, 'max ', '' or 1; colony_size 3 (just below 4), 0, -1 or None; limit 0 (just
    below 1), -1 or None; seed 'a' or 1.5. Oracle: ABCOptimizer.__init__ (colony_size >= 4, limit >= 1),
    Optimizer.__init__ (direction is 'min' or 'max'), the seed annotation int | None and the declaration row
    of the oracle table.
    """
    seeds = SEEDS

    failures = _declaration_failures(seeds, _gen_invalid_optimizer_case)

    assert failures == []


def test_optimize_should_raise_only_value_or_type_error_when_one_argument_is_invalid():
    """Given optimize arguments with one invalid value, when optimize is called, then only VE or TE occurs.

    Inputs: max_evals 0 (just below 1), -1 or None; direction 'MIN', None, 'max ' or ''; seed 'a' or 1.5; an
    optimizer on a different space or with the opposite direction. Oracle: optimize docstring and the task
    spec (ValueError for max_evals < 1 and for a mismatched optimizer space or direction), the seed annotation
    int | None and the declaration row of the table.
    """
    seeds = SEEDS

    failures = _declaration_failures(seeds, _gen_invalid_optimize_case)

    assert failures == []


def test_abc_optimizer_should_raise_only_documented_protocol_errors_when_driven_by_random_action_sequences():
    """Given a shadow model, when random ask/tell actions are applied, then each outcome is the specified one.

    Expected: ask with nothing pending is ok; ask while anything is pending (inside the initial batch, also
    after a partial tell, and outside it) raises RuntimeError; telling all or a proper subset of the pending
    candidates is ok; telling an already-told candidate, a forged id (max issued + 1000, or -1) or a score
    list one longer or shorter raises ValueError; no other type occurs; a rejected tell leaves the pending set
    usable; candidates and best_score keep the score monitors after every valid tell. Oracle: the task spec
    protocol errors, the Optimizer and ABCOptimizer docstrings (only the initial population is a batch).
    """
    seeds = SEEDS

    failures = _protocol_failures(seeds)

    assert failures == []


@pytest.mark.parametrize(
    "exc_type", [FuzzSentinel, ZeroDivisionError], ids=["sentinel", "builtin-zero-division"]
)
@pytest.mark.parametrize(
    "call_index",
    [0, 1, 2, 4],
    ids=["first-initial-call", "last-initial-call", "first-employed", "first-onlooker"],
)
def test_optimize_should_propagate_the_same_exception_object_when_objective_raises(exc_type, call_index):
    """Given an objective raising on call call_index, when optimize runs, then that very object propagates.

    colony_size=4 gives two initial calls (0, 1) evaluated as a batch, then employed (2, 3) and onlooker (4)
    single candidates, so both evaluation paths are covered; cache=False makes the call number equal the
    evaluation number. ZeroDivisionError is a type the library itself could raise, so only the identity check
    separates user from library. Oracle: the task spec (user exceptions propagate) and the identity
    requirement of P-T7-19.
    """
    exc = exc_type("raised by the objective")
    objective = _objective_raising_on_call(call_index, exc)
    space = Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, seed=ANY_SEED)

    with pytest.raises(exc_type) as info:
        optimize(objective, space, max_evals=SMALL_BUDGET, optimizer=opt, cache=False)

    assert info.value is exc


@pytest.mark.parametrize(
    "value",
    [None, "abc", 1j, np.array([1.0, 2.0])],
    ids=["none", "non-numeric-string", "complex", "two-element-array"],
)
def test_optimize_should_raise_only_value_or_type_error_when_objective_returns_a_non_number(value):
    """Given an objective returning a non-number, when optimize runs, then only VE or TE occurs.

    Oracle: optimize docstring (function returns a score) and the user_return row of the oracle table; float()
    rejects these values with TypeError or ValueError. A 1-element array is avoided because its conversion
    depends on the numpy version.
    """
    space = Space({"x": Float(0.0, 1.0)})

    outcome = _run_case(optimize, _objective_returning(value), space, max_evals=SMALL_BUDGET, seed=ANY_SEED)

    assert _classify("user_return", outcome) == "expected error"


def test_optimize_should_not_raise_an_undocumented_error_when_objective_returns_a_numeric_string():
    """Given an objective returning '1.0', when optimize runs, then the outcome is ok or ValueError/TypeError.

    Whether a numeric string is accepted is unspecified (the return is annotated as float), so only the
    absence of an undocumented exception type is asserted. Oracle: the user_return row of the oracle table.
    """
    space = Space({"x": Float(0.0, 1.0)})

    outcome = _run_case(optimize, _objective_returning("1.0"), space, max_evals=SMALL_BUDGET, seed=ANY_SEED)

    assert _classify("user_return", outcome) in {"ok", "expected error"}


def test_optimize_should_pass_declared_key_and_choice_objects_when_keys_and_choices_are_checklist_strings():
    """Given checklist strings as keys and choices, when optimize runs, then the declared objects are passed.

    Checklist items (CHECKLIST): 1 a 10_000-character string and ''; 3 NUL, newline, CRLF, EOT, SUB, None,
    NaN; 4 '%s %x %n', '{0}', '{}'; 5 ';', '/', backslash, quotes, BEL, DEL; 6 'return', 'lambda', 'None',
    'True', 'nan', 'inf', 'min', 'max', 'halt', 'DROP TABLES'. Monitors: exactly max_evals trials, params keys
    are the declared key objects in order, and every value is one of the declared choice objects. Oracle:
    README table (one of choices), Space.rows docstring and the optimize budget rule.
    """
    space, choices = _special_string_space()

    study = optimize(_objective_sum_of_numbers, space, max_evals=SMALL_BUDGET, seed=ANY_SEED)

    assert (len(study.trials), _special_string_problems(study, choices)) == (SMALL_BUDGET, [])


def test_optimize_should_call_the_objective_once_when_the_only_parameter_has_one_choice():
    """Given a Space whose only parameter has one choice (item 1), when optimize runs, then n_calls is 1.

    Every candidate decodes to the same parameter set, so with cache=True all trials after the first reuse its
    score. Oracle: README Behaviour (cache reuses a repeated parameter set), Study.n_calls docstring and the
    exact max_evals rule.
    """
    space = Space({"k": Categorical(["only"])})

    study = optimize(_dummy_objective, space, max_evals=SMALL_BUDGET, seed=ANY_SEED)

    assert (len(study.trials), study.n_calls) == (SMALL_BUDGET, 1)


@pytest.mark.parametrize(
    "make_space", [_wide_space, _many_choices_space], ids=["200-parameters", "1000-choices"]
)
def test_optimize_should_spend_the_exact_budget_when_the_space_is_very_large(make_space):
    """Given a very wide space or a 1000-choice list (item 1), when optimize runs, then the budget is exact.

    Oracle: optimize docstring and the task spec (exactly max_evals trials).
    """
    space = make_space()

    study = optimize(_dummy_objective, space, max_evals=SMALL_BUDGET, seed=ANY_SEED)

    assert len(study.trials) == SMALL_BUDGET


def test_optimize_should_record_exactly_one_trial_when_max_evals_is_one():
    """Given max_evals=1 (smallest valid budget), when optimize runs, then exactly one trial exists.

    Oracle: the task spec (exact max_evals trials, even inside the initial batch).
    """
    space = Space({"x": Float(0.0, 1.0)})

    study = optimize(_dummy_objective, space, max_evals=1, seed=ANY_SEED)

    assert len(study.trials) == 1


def test_space_should_return_an_empty_batch_when_the_unit_array_has_zero_rows():
    """Given a unit array of shape (0, 2) (empty input, item 1), when it is decoded, then the batch is empty.

    Oracle: Space.decode docstring ((n, d) to (n, d)) and Space.to_dicts (one dict per row).
    """
    space = Space({"a": Float(0.0, 1.0), "b": Int(1, 3)})
    empty = np.empty((0, 2))

    decoded, dicts = space.decode(empty), space.to_dicts(empty)

    assert (decoded.shape, dicts) == ((0, 2), [])


@pytest.mark.parametrize(
    ("low", "high"),
    [
        pytest.param(0, 0, id="zero-single-value"),
        pytest.param(-5, -1, id="all-negative"),
        pytest.param(-(2**31), 2**31 - 1, id="int32-extremes"),
        pytest.param(0, 2**53, id="largest-exact-float64-integer"),
    ],
)
def test_space_to_dicts_should_return_the_exact_int_end_points_when_bounds_are_zero_negative_or_large(
    low, high
):
    """Given zero, negative or large Int bounds, when u=0 and u=1 are decoded, then low and high return.

    Checklist item 2 (integer zero, negative, min and max). Oracle: README table (integers low..high
    inclusive) and the Int docstring (equal shares of [0, 1], so the closed ends give the end points); the
    values are Python ints. 2**53 is the largest n such that every integer up to n is exactly
    representable in float64; bounds beyond it are a reported defect.
    """
    space = Space({"n": Int(low, high)})

    params = space.to_dicts([[0.0], [1.0]])

    assert (params, type(params[0]["n"]), type(params[1]["n"])) == ([{"n": low}, {"n": high}], int, int)


@pytest.mark.parametrize("told_order", sorted(TOLD_ORDERS))
@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize(
    "score",
    [0.0, -0.0, FLOAT_MAX, -FLOAT_MAX, 1e-308, 5e-324],
    ids=["zero", "negative-zero", "float-max", "minus-float-max", "tiny-normal", "subnormal"],
)
def test_abc_optimizer_should_report_the_only_finite_score_when_it_is_extreme(score, direction, told_order):
    """Given an init batch, when an extreme score and NaN are told in either order, then best_score is it.

    Checklist item 2 (extreme values). Oracle: Optimizer docstring (scores in the user's direction, NaN or
    infinity is the worst value), so the only finite score is the best one in both directions; negating
    +-FLOAT_MAX for 'max' stays finite. The 'nan-first' order is decisive: a NaN told first that is not mapped
    to the worst value becomes best_score and no later comparison with NaN replaces it, while 'finite-first'
    hides that defect. Site: Optimizer._internal, NaN-to-worst mapping.
    """
    expected = score

    best = _best_after_one_finite_score(score, direction, TOLD_ORDERS[told_order])

    assert best == expected


def test_optimize_should_finish_or_raise_value_or_type_error_when_the_readme_config_is_mutated():
    """Given single-edit README config mutants, when each is optimised, then it finishes or raises VE/TE only.

    Mutants: one edit each from OPERATORS (swap or equal bounds, log low 0 or negative, NaN bound, flipped
    log, a parameter replaced by a tuple, None or int, a dropped parameter, empty or singleton choices, a key
    renamed to a checklist string, an invalid direction, max_evals, colony_size or limit); rng chooses order,
    target and value per seed. A finished mutant has exactly max_evals trials, the mutant's keys, README value
    rules (Float edge units excluded, see the module docstring) and best_score equal to the maximum trial
    score. Oracle: README Quick start and table, the validation rules of space.py, abc.py, base.py and
    facade.py.
    """
    seeds = SEEDS

    failures = _mutant_failures(seeds)

    assert failures == []


def test_readme_mutants_should_use_every_operator_and_reach_both_outcomes():
    """Given README mutants over SEEDS, when all are run, then every operator is used and both outcomes occur.

    A permutation of OPERATORS per seed guarantees use of every operator; valid edits (a dropped parameter,
    singleton choices, a key renamed to a checklist string) reach 'ok' on every seed, and an invalid one
    (swapped bounds) reaches 'expected error'.
    Oracle: P-T7-07 (mutate valid inputs) and the oracle table classes; no 'finding' is allowed.
    """
    seeds = SEEDS

    used, kinds = _mutant_coverage(seeds)

    assert (used, kinds) == (set(OPERATORS), {"ok", "expected error"})


def test_fuzz_generators_should_reproduce_identical_outcomes_when_run_twice_with_the_same_seed():
    """Given the unit-array, protocol and optimize collectors, when run twice per seed, then outputs match.

    Budget-only stopping and seeded generators make the fuzz run deterministic; abnormal scores (NaN, inf,
    numpy scalars) are included and compared through repr, where NaN equals NaN. Oracle: README Behaviour (the
    same seed reproduces the same search) and P-T7-21 (identical outcomes across runs).
    """
    seeds = SEEDS[:2]

    first = _transcripts(seeds)
    second = _transcripts(seeds)

    assert first == second
