"""Generated score streams with non-finite and extreme values fed to ABCOptimizer through ask and tell.

Generators (P-T7-25): every case is drawn from a numpy Generator seeded from the test parameters plus the
named constant SALT_STREAMS. A case holds a random valid space (1..MAX_DIM parameters of the five kinds), a
direction, colony_size in 4..12, limit in 1..5, an optimizer seed and a score stream over POOL: NaN, +inf,
-inf, +-1e308, +-sys.float_info.max and ORDINARY_PER_SIGN ordinary values of each sign, all Python floats.
The stream starts with a non-finite prefix, and every fourth case is non-finite throughout. Each ask's
candidates are told in a random order split into random chunks, so the initial batch is told partially.

Budget and size (P-T7-20): parametrized on the number of cases (20 or 300) and on the number of asks per case
(20 or 60), and asserts that both stream classes (some finite, none finite) occurred.

Harness (P-T7-14, P-T7-15, P-T7-18): one generic _run_case(target, *args) calls the driver and never raises;
_fuzz records (id, input, outcome) per case and only after the loop turns each failing case into a line
'Test #<id> IN=<repr> OUT=<ok|Type: msg>'.

Oracles: Optimizer docstring (scores in the user's direction; a NaN or infinite score becomes the worst value
for the direction: -inf when maximising, +inf when minimising), README Behaviour (NaN or infinity counts as
the worst value), the best_score and best_params properties (the best told score and its parameters) and the
computed spec (the initial batch may be told partially).

Unspecified, therefore not asserted: which of several tied best candidates best_params reports (only
membership among the tied candidates is asserted); best_score and best_params before the first tell; Python
int scores beyond the float range; bit-exact agreement of a log-scaled Float between a candidate's params and
best_params (compared with LOG_REL_TOL).

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: Seeded generated score streams mixing NaN, infinities and extreme finite values.
"""

import math
import sys

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space

HUGE = sys.float_info.max
NON_FINITE = (math.nan, math.inf, -math.inf)
EXTREME = (1e308, -1e308, HUGE, -HUGE)
MAX_DIM = 4
MAX_PREFIX = 7
ORDINARY_PER_SIGN = 3
LOG_REL_TOL = 1e-12
SALT_STREAMS = 21
KINDS = ("float", "float_log", "int", "int_log", "categorical")
SIZES = [
    pytest.param(n_cases, n_asks, id=f"{n_cases}-{n_asks}") for n_cases in (20, 300) for n_asks in (20, 60)
]


def _run_case(target, *args, **kwargs):
    """Calls target(*args, **kwargs); returns ('ok', value) or ('error', type name, message), never raises."""
    try:
        return ("ok", target(*args, **kwargs))
    except Exception as exc:
        return ("error", type(exc).__name__, str(exc))


def _failure_line(record, check):
    """The report line of one record, or '' when its outcome is ok and check finds no violated invariant."""
    case_id, case, outcome = record
    problems = check(case, outcome[1]) if outcome[0] == "ok" else []
    if outcome[0] == "ok" and not problems:
        return ""
    shown = "ok VIOLATED=" + repr(problems)[:400] if outcome[0] == "ok" else f"{outcome[1]}: {outcome[2]}"
    return f"Test #{case_id} IN={repr(case)[:400]} OUT={shown}"


def _fuzz(cases, target, check):
    """Runs target on every case, records (id, input, outcome), and after the loop returns the failure
    lines."""
    records = [(case_id, case, _run_case(target, *case)) for case_id, case in enumerate(cases)]
    lines = [_failure_line(record, check) for record in records]
    return [line for line in lines if line]


def _generate_parameter(rng, kind):
    """One valid parameter of the given kind with random bounds or choices."""
    if kind == "float":
        low = float(rng.uniform(-100.0, 100.0))
        return Float(low, low + float(rng.uniform(1e-3, 100.0)))
    if kind == "float_log":
        low = float(10.0 ** rng.uniform(-5.0, 2.0))
        return Float(low, low * float(10.0 ** rng.uniform(0.01, 4.0)), log=True)
    if kind == "int":
        low = int(rng.integers(-50, 50))
        return Int(low, low + int(rng.integers(0, 31)))
    if kind == "int_log":
        low = int(rng.integers(1, 50))
        return Int(low, low + int(rng.integers(0, 1001)), log=True)
    return Categorical(list(range(int(rng.integers(1, 6)))))


def _generate_mapping(rng, max_dim):
    """A mapping of 1..max_dim parameters of random kinds, named p0, p1, ... in declaration order."""
    kinds = [KINDS[int(i)] for i in rng.integers(0, len(KINDS), size=int(rng.integers(1, max_dim + 1)))]
    return {f"p{j}": _generate_parameter(rng, kind) for j, kind in enumerate(kinds)}


def _score_stream(rng, length, all_non_finite):
    """length Python floats: a non-finite prefix, then draws from POOL (or only non-finite values)."""
    prefix = int(rng.integers(1, MAX_PREFIX + 1))
    ordinary = rng.uniform(1e-3, 1e3, size=ORDINARY_PER_SIGN).tolist()
    pool = list(NON_FINITE) + list(EXTREME) + [-v for v in ordinary] + ordinary
    pool = list(NON_FINITE) if all_non_finite else pool
    stream = [NON_FINITE[int(i)] for i in rng.integers(0, len(NON_FINITE), size=prefix)]
    stream += [pool[int(i)] for i in rng.integers(0, len(pool), size=length)]
    return stream


def _generate_cases(rng, n_cases, n_asks):
    """Cases (mapping, direction, colony_size, limit, seed, stream, chunk_seed, n_asks)."""
    cases = []
    for case_id in range(n_cases):
        colony_size = int(rng.integers(4, 13))
        stream = _score_stream(rng, n_asks + colony_size, case_id % 4 == 3)
        direction = ("min", "max")[int(rng.integers(2))]
        mapping = _generate_mapping(rng, MAX_DIM)
        limit = int(rng.integers(1, 6))
        cases.append((mapping, direction, colony_size, limit, case_id, stream, case_id + 1, n_asks))
    return cases


def _chunks(rng, cands):
    """The candidates in a random order, split into 1..len(cands) consecutive chunks."""
    if len(cands) == 1:
        return [cands]
    order = [cands[int(i)] for i in rng.permutation(len(cands))]
    n_cuts = int(rng.integers(0, len(cands)))
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, len(cands)), size=n_cuts, replace=False))
    return [order[a:b] for a, b in zip([0, *cuts], [*cuts, len(cands)], strict=True)]


def _drive(mapping, direction, colony_size, limit, seed, stream, chunk_seed, n_asks):
    """Target: n_asks ask calls, each answered by chunked tells; returns the told (params, score) pairs and,
    after every tell, (number told so far, best_score, best_params)."""
    space = Space(mapping)
    opt = ABCOptimizer(space, direction, colony_size=colony_size, limit=limit, seed=seed)
    rng = np.random.default_rng(chunk_seed)
    scores = iter(stream)
    told = []
    observed = []
    for _ in range(n_asks):
        for chunk in _chunks(rng, opt.ask()):
            chunk_scores = [next(scores) for _ in chunk]
            opt.tell(chunk, chunk_scores)
            told += [(cand.params, score) for cand, score in zip(chunk, chunk_scores, strict=True)]
            observed.append((len(told), opt.best_score, opt.best_params))
    return told, observed


def _same_params(parameters, a, b):
    """Equality of two params dicts, exact except LOG_REL_TOL for log-scaled Float values."""
    if list(a) != list(b):
        return False
    for parameter, x, y in zip(parameters, a.values(), b.values(), strict=True):
        if isinstance(parameter, Float) and parameter.log:
            if x != pytest.approx(y, rel=LOG_REL_TOL):
                return False
        elif x != y:
            return False
    return True


def _running_best(told, direction):
    """For every prefix length n, (best finite score or None, params told with that score) of told[:n]."""
    better = (lambda a, b: a < b) if direction == "min" else (lambda a, b: a > b)
    best, tied, prefixes = None, [], [(None, [])]
    for params, score in told:
        if math.isfinite(score) and (best is None or better(score, best)):
            best, tied = score, [params]
        elif math.isfinite(score) and score == best:
            tied = [*tied, params]
        prefixes.append((best, tied))
    return prefixes


def _best_problems(case, value):
    """Violations of 'best is the worst value until a finite score is told, then the direction's extreme of
    the finite told scores, with best_params among the candidates told with that score'."""
    mapping, direction = case[0], case[1]
    parameters = list(mapping.values())
    told, observed = value
    worst = math.inf if direction == "min" else -math.inf
    prefixes = _running_best(told, direction)
    problems = []
    for n, best_score, best_params in observed:
        expected, tied = prefixes[n]
        if expected is None and best_score != worst:
            problems.append(f"after {n} tells without a finite score best_score={best_score!r}")
        elif expected is not None and (
            best_score != expected or not any(_same_params(parameters, best_params, p) for p in tied)
        ):
            problems.append(f"after {n} tells best=({best_score!r}, {best_params!r}) expected {expected!r}")
    return problems


def _stream_classes(cases):
    """Which stream classes occurred: 'some finite' and 'none finite'."""
    return {"some finite" if any(math.isfinite(s) for s in case[5]) else "none finite" for case in cases}


@pytest.mark.parametrize(("n_cases", "n_asks"), SIZES)
def test_abc_should_track_the_finite_best_when_told_generated_non_finite_and_extreme_scores(n_cases, n_asks):
    """Given random spaces, directions, colony_size >= 4 and limit >= 1, when n_asks asks are answered with
    chunked tells of a stream mixing NaN, +-inf, +-1e308, +-float max and ordinary values, then nothing
    raises; best_score stays the direction's worst value (+inf for 'min', -inf for 'max') until a finite
    score is told, and afterwards equals the direction's extreme of the finite told scores, with best_params
    equal to the params of a candidate told with that score.

    Oracle: Optimizer docstring and README Behaviour (NaN or infinite -> worst value for the direction);
    best_score / best_params properties. -inf under 'min' and +inf under 'max' are the decisive inputs: a
    plain min/max of the raw scores would report them. limit = 1 and colony_size = 4 make scouts fire early.
    """
    cases = _generate_cases(np.random.default_rng([n_cases, n_asks, SALT_STREAMS]), n_cases, n_asks)

    failures = _fuzz(cases, _drive, _best_problems)

    assert failures == []
    assert _stream_classes(cases) == {"some finite", "none finite"}
