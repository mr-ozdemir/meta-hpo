"""Determinism: the same seed gives the same search, isolated from every outside source of randomness.

Rules: P-T8-14 (a scenario is repeatable: precondition plus steps always give the same postcondition),
P-T1-34 (do not trust one passing run of a behaviour that depends on random draws; run it many times),
P-T3-37 and P-T8-29 (the result must not depend on state outside the code under test: global random
streams, other optimizers, the hash seed of the interpreter), P-T1-33 (run with the same interpreter and
settings: the child processes use sys.executable), P-T7-03 (run the program under test as a subprocess
with its input on stdin), P-T2-02 (tests run in any order), P-T6-20 (cheap tests first), P-T1-31 (the
suite, this file, one class, one test and one parametrize case can each be run alone).

Oracles: README Behaviour 'The same seed reproduces the same search' and 'With cache=True ... a repeated
parameter set reuses its earlier score'; README 'Optimizers search the unit cube [0, 1]^d'; optimize
docstring and the task specification (exactly max_evals trials, numbered in order); Optimizer constructor
signature (seed is a per-optimizer argument); abc.py docstring and Karaboga & Basturk 2007 (a neighbour
differs from its food source in one coordinate, v_ij = x_ij + phi_ij (x_ij - x_kj)); operators
neighbour_value docstring (clipped to [0, 1]); original ABC.py and the task specification (greedy
acceptance needs a strictly greater fitness, a scout replaces the source whose trial reached limit, at
most one per cycle); Study docstrings (best_trial is the min or max of the finite trials).

Isolation is inferred, not stated: if a global random stream, another optimizer or the hash seed could
change the result, the same seed would not reproduce the same search. The original ABC.py is itself not
isolated (it seeds the global stdlib random stream), so the stdlib stream is one of the perturbed ones.

Input space: every seed (int or None), both directions, every Space, every objective, every interleaving
with other optimizers and every global random and hash-seed state, which is unbounded.
Subset: one mixed Space with all four decode kinds (linear Float, log Float, Int, Categorical of strings)
and one pure finite objective; seeds 3, 7 and 11 for the equality tests and range(50) for the
seed-independent invariants; both directions; one intervening run, one interleaved optimizer, two global
stream states and three hash seeds.

Seeds: 7 follows the same-seed test of the existing suite (tests/test_abc.py); 11 is an arbitrary second
seed, chosen only because it differs from 7; 3 is the arbitrary seed of the
optimizer-level tests; range(50) is a choice, because P-T1-34 gives no repetition count. No assertion
compares with a value captured from a seeded run: runs are compared only with each other or checked for
seed-independent properties.

Order (P-T6-20): optimizer-level equality, then optimize() with the same seed, then isolation from other
optimizers, then the 50-seed invariants, then the two subprocess tests, from the cheapest and most basic
to the most expensive. APFD cannot be computed for this order, because no fault history exists. The
verdicts do not depend on the order (P-T2-02): every test builds its own Space, objective and optimizers,
the module holds only constants and pure helpers, and the global random streams are perturbed only inside
child processes, so the test process's global state is never changed.

Unspecified, not asserted:
- whether runs with seed=None differ from each other (no test runs unseeded, because tests must pass a
  seed to anything random);
- whether Space.sample can return exactly 1.0 (the docs say only 'unit cube'); trial x is checked against
  the closed interval [0, 1], and x == 1.0 does occur through clipping;
- whether optimize() or ABCOptimizer may change the global numpy or stdlib random state;
- bit-exact equality across numpy versions or platforms.
"""

import ast
import os
import subprocess
import sys

import numpy as np
import pytest

import metahpo
from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

OPTIMIZER_SEED = 3
SAME_SEEDS = (7, 11)
INTERVENING_SEED = 11
REPEATS = 3
EQUALITY_EVALS = 200
ISOLATION_EVALS = 150
INTERVENING_STEPS = 20
INTERLEAVED_ROUNDS = 40
INVARIANT_SEEDS = range(50)
INVARIANT_EVALS = 67
SMALL_COLONY = 6
SCOUT_LIMIT = 2
NO_SCOUT_LIMIT = 10**9
TARGET_CYCLES = 3
ASK_CAP = 200
FLAT_SCORE = 1.0
DIRECTIONS = ("min", "max")
PICK = {"min": min, "max": max}
CHILD_SEED = 3
CHILD_EVALS = 150
CHILD_TIMEOUT = 10.0
QUIET_GLOBALS = 0
NOISY_GLOBALS = 1
HASH_SEEDS = ("0", "1", "2")
ACT_PENALTY = {"relu": 0.0, "tanh": 0.3, "gelu": 0.6}


def _mixed_space():
    """A fresh Space with one parameter of each decode kind; the small Int and Categorical make cache hits."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "drop": Float(0.0, 0.5),
            "units": Int(1, 4),
            "act": Categorical(["relu", "tanh", "gelu"]),
        }
    )


def stub_objective(params):
    """Pure, finite stub objective with its minimum near lr = 10**-2.5, drop = 0.2, units = 1, act = relu."""
    return (
        (np.log10(params["lr"]) + 2.5) ** 2
        + (params["drop"] - 0.2) ** 2
        + 0.1 * params["units"]
        + ACT_PENALTY[params["act"]]
    )


def flat_objective(params):
    """Stub objective that gives every point the same score, so no neighbour is ever strictly better."""
    return FLAT_SCORE


def _trial_records(study):
    """Every observable field of every trial: (number, params, score, cached, x as a tuple)."""
    return [(t.number, t.params, t.score, t.cached, tuple(t.x.tolist())) for t in study.trials]


def _candidate_records(candidates):
    """Every field of a list of candidates: (id, x as a tuple, params, key)."""
    return [(c.id, tuple(c.x.tolist()), c.params, c.key) for c in candidates]


def _repeat_optimize(times, direction, seed):
    """Trial records of `times` optimize() runs, each with a fresh mixed Space and the same seed."""
    runs = []
    for _ in range(times):
        study = optimize(
            stub_objective, _mixed_space(), direction=direction, max_evals=EQUALITY_EVALS, seed=seed
        )
        runs.append(_trial_records(study))
    return runs


def _step(opt, objective):
    """One ask of `opt`, told at once with `objective`; returns the asked candidates."""
    candidates = opt.ask()
    opt.tell(candidates, [objective(c.params) for c in candidates])
    return candidates


def _drive_pair_to_cycle(first, second, objective, cycles):
    """Steps both optimizers alternately until both reach `cycles`; returns the two candidate streams.

    Stops with an assertion error after ASK_CAP rounds, so a stuck phase cannot hang the test.
    """
    streams = ([], [])
    while first.cycle < cycles or second.cycle < cycles:
        assert len(streams[0]) < ASK_CAP
        streams[0].append(_candidate_records(_step(first, objective)))
        streams[1].append(_candidate_records(_step(second, objective)))
    return streams


def _scout_like_count(stream):
    """Number of single asks after the first batch whose x differs in at least two coordinates from the x
    of every earlier candidate; a neighbour differs from its food source, an earlier x, in one coordinate."""
    seen = [np.array(x) for _, x, _, _ in stream[0]]
    count = 0
    for batch in stream[1:]:
        x = np.array(batch[0][1])
        count += min(int(np.count_nonzero(x != earlier)) for earlier in seen) >= 2
        seen.append(x)
    return count


def _step_times(opt, objective, times):
    """`times` steps of `opt` with `objective`; returns the candidate records of each ask."""
    return [_candidate_records(_step(opt, objective)) for _ in range(times)]


def _interleave(recorded, other, objective, rounds):
    """`rounds` rounds of one step of `recorded` then one step of `other`; returns `recorded`'s asks."""
    stream = []
    for _ in range(rounds):
        stream.append(_candidate_records(_step(recorded, objective)))
        _step(other, objective)
    return stream


def _max_changed_coordinates(study, start):
    """Largest, over trials from index `start` on, of the fewest coordinates in which the trial's x differs
    from the x of any earlier trial."""
    xs = [t.x for t in study.trials]
    return max(min(int(np.count_nonzero(xs[n] != xs[m])) for m in range(n)) for n in range(start, len(xs)))


CHILD_PROGRAM = """
import random
import sys

import numpy as np

from metahpo import Categorical, Float, Int, Space, optimize

seed, globals_seed = (int(v) for v in sys.stdin.read().split())
np.random.seed(globals_seed)
np.random.random(1000)
random.seed(globals_seed)
random.random()
space = Space(
    {
        "lr": Float(1e-4, 1e-1, log=True),
        "drop": Float(0.0, 0.5),
        "units": Int(1, 4),
        "act": Categorical(["relu", "tanh", "gelu"]),
    }
)
penalty = {"relu": 0.0, "tanh": 0.3, "gelu": 0.6}


def objective(params):
    return (
        (np.log10(params["lr"]) + 2.5) ** 2
        + (params["drop"] - 0.2) ** 2
        + 0.1 * params["units"]
        + penalty[params["act"]]
    )


study = optimize(objective, space, max_evals=int(sys.argv[1]), seed=seed)
print([(t.number, t.params, t.score, t.cached, tuple(t.x.tolist())) for t in study.trials])
"""
"""A fresh-process run of optimize(): reads 'seed globals_seed' from stdin, seeds and advances the global
numpy and stdlib random streams with globals_seed, then prints the trial records of a seeded optimize()
over the same mixed Space and objective as _mixed_space and stub_objective."""


def run_program(code, input_string, hash_seed, *args):
    """Runs `code` in a fresh interpreter (sys.executable) with `input_string` on stdin, PYTHONPATH set to
    the directory that holds the metahpo package and PYTHONHASHSEED set to `hash_seed`; returns the
    CompletedProcess with returncode, stdout and stderr."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(metahpo.__file__))
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        input=input_string,
        capture_output=True,
        text=True,
        env=env,
        timeout=CHILD_TIMEOUT,
    )


def _child_input(globals_seed):
    """stdin text of the child program: the optimize() seed and the seed of the global random streams."""
    return f"{CHILD_SEED} {globals_seed}"


def _run_with_hash_seeds(hash_seeds):
    """One child run per hash seed, all with the quiet global streams; returns the CompletedProcesses."""
    return [run_program(CHILD_PROGRAM, _child_input(QUIET_GLOBALS), h, str(CHILD_EVALS)) for h in hash_seeds]


def _in_process_records():
    """Trial records of the child program's optimize() run, computed in this process."""
    study = optimize(stub_objective, _mixed_space(), max_evals=CHILD_EVALS, seed=CHILD_SEED)
    return _trial_records(study)


class TestSameSeed:
    """Equal seeds give equal candidate streams and equal studies."""

    def test_abc_optimizer_should_ask_identical_candidates_including_scouts_when_seeds_are_equal(self):
        """Given two ABCOptimizers with seed 3, colony_size 6 and limit 2 on separately built mixed spaces,
        and a flat objective, when both are stepped alternately until cycle 3, then every ask returns
        candidates with equal id, x, params and key, both best_score and best_params are equal, and each
        stream holds exactly 3 scout candidates.

        Oracle: README 'The same seed reproduces the same search'. The scouts are guaranteed by the ABC rules,
        not by the seed: equal fitness is not strictly greater, so every employed and onlooker visit adds 1 to
        the source's trial; with all probabilities equal the onlookers visit each of the 3 sources once, so
        every trial is 2 >= limit after cycle 0, and every cycle from then on ends with one scout (at most one
        per cycle). So scout draws (Space.sample on the generator) interleaved with the neighbour draws are
        compared too. A scout x differs from every earlier x in all coordinates (probability 1), a neighbour
        in at most one coordinate from its food source, so the scout count is exact.
        """
        first = ABCOptimizer(_mixed_space(), colony_size=SMALL_COLONY, limit=SCOUT_LIMIT, seed=OPTIMIZER_SEED)
        second = ABCOptimizer(
            _mixed_space(), colony_size=SMALL_COLONY, limit=SCOUT_LIMIT, seed=OPTIMIZER_SEED
        )

        stream_first, stream_second = _drive_pair_to_cycle(first, second, flat_objective, TARGET_CYCLES)

        assert stream_second == stream_first
        assert (second.cycle, first.cycle) == (TARGET_CYCLES, TARGET_CYCLES)
        assert second.best_score == first.best_score
        assert second.best_params == first.best_params
        assert _scout_like_count(stream_first) == TARGET_CYCLES

    @pytest.mark.parametrize("direction", [pytest.param(d, id=d) for d in DIRECTIONS])
    @pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed={s}") for s in SAME_SEEDS])
    def test_optimize_should_return_identical_trials_when_run_repeatedly_with_the_same_seed(
        self, seed, direction
    ):
        """Given the mixed space and the stub objective, when optimize() runs 3 times with 200 evaluations and
        the same seed, then the second and third runs equal the first in number, params, score, cached and x
        of every trial.

        Oracle: README 'The same seed reproduces the same search' and the cache sentence. Both directions are
        separate classes ('max' negates scores inside); three runs instead of two catch drift that builds up
        across calls (P-T1-34). The run must hold cached trials, so the cached field is compared on real data:
        Int(1, 4) and a 3-way Categorical make neighbours that keep the decoded parameters frequent.
        """
        runs = _repeat_optimize(REPEATS, direction, seed)

        assert runs[1] == runs[0]
        assert runs[2] == runs[0]
        assert any(cached for _, _, _, cached, _ in runs[0]) is True


class TestIsolation:
    """A seeded search is unaffected by other searches run before it or alongside it."""

    def test_optimize_should_return_identical_trials_when_another_search_runs_in_between(self):
        """Given optimize() with seed 7, when an optimize() run with seed 11 and 20 steps of an ABCOptimizer
        with seed 11 happen before the same seed-7 run is repeated, then both seed-7 runs have equal trials.

        Oracle: README 'The same seed reproduces the same search'; isolation is inferred from it. Guards
        against state left behind by another search (site: any module-level cache, seen dict or shared
        uniform buffer).
        """
        before = _trial_records(optimize(stub_objective, _mixed_space(), max_evals=ISOLATION_EVALS, seed=7))
        optimize(stub_objective, _mixed_space(), max_evals=ISOLATION_EVALS, seed=INTERVENING_SEED)
        _step_times(ABCOptimizer(_mixed_space(), seed=INTERVENING_SEED), stub_objective, INTERVENING_STEPS)

        after = _trial_records(optimize(stub_objective, _mixed_space(), max_evals=ISOLATION_EVALS, seed=7))

        assert after == before

    def test_abc_optimizer_should_ask_the_solo_candidates_when_interleaved_with_another_seed(self):
        """Given ABCOptimizers with seeds 3 and 7 sharing one Space instance, stepped alternately for 40
        rounds, and a solo ABCOptimizer with seed 3 on its own Space stepped 40 times, then the seed-3
        candidates of the interleaved run equal the solo ones at every ask, and the run passes cycle 2.

        Oracle: README 'The same seed reproduces the same search'; seed is a per-optimizer constructor
        argument. Interleaving catches state shared between instances (site: a class-level uniform buffer or
        generator) that runs one after another would miss; the shared Space shows it keeps no per-optimizer
        state. Limit 2 puts scouts into the stream.
        """
        shared = _mixed_space()
        recorded = ABCOptimizer(shared, colony_size=SMALL_COLONY, limit=SCOUT_LIMIT, seed=OPTIMIZER_SEED)
        other = ABCOptimizer(shared, colony_size=SMALL_COLONY, limit=SCOUT_LIMIT, seed=7)
        solo = ABCOptimizer(_mixed_space(), colony_size=SMALL_COLONY, limit=SCOUT_LIMIT, seed=OPTIMIZER_SEED)
        solo_stream = _step_times(solo, stub_objective, INTERLEAVED_ROUNDS)

        interleaved_stream = _interleave(recorded, other, stub_objective, INTERLEAVED_ROUNDS)

        assert interleaved_stream == solo_stream
        assert recorded.cycle >= 2


class TestSeedIndependentInvariants:
    """Properties every seed must show; 50 seeds sample the random paths (P-T1-34)."""

    @pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed={s}") for s in INVARIANT_SEEDS])
    def test_optimize_should_return_numbered_trials_in_the_closed_unit_cube_for_every_seed(self, seed):
        """Given the mixed space, the stub objective and direction min for even and max for odd seeds, when
        optimize() runs 67 evaluations, then there are 67 trials numbered 0..66, every x lies in the closed
        interval [0, 1], and best_score is the min or max of the scores.

        Oracle: optimize docstring and task specification (exactly max_evals trials, in order); README
        'Optimizers search the unit cube [0, 1]^d'; neighbour_value docstring (clipped to [0, 1], so x == 1.0
        is allowed); Study best_trial (min or max of the finite trials; the stub objective is always finite).
        67 = 5 initial trials (default colony_size 10, so 10 // 2 = 5 food sources) + 6 whole cycles of 10
        (5 employed + 5 onlooker asks) + a remainder of 2, so the budget ends inside a phase.
        """
        direction = DIRECTIONS[seed % 2]

        study = optimize(
            stub_objective, _mixed_space(), direction=direction, max_evals=INVARIANT_EVALS, seed=seed
        )

        all_x = np.stack([t.x for t in study.trials])
        assert [t.number for t in study.trials] == list(range(INVARIANT_EVALS))
        assert (float(all_x.min()) >= 0.0, float(all_x.max()) <= 1.0) == (True, True)
        assert study.best_score == PICK[direction](study.scores.tolist())

    @pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed={s}") for s in INVARIANT_SEEDS])
    def test_abc_optimizer_should_ask_only_one_coordinate_neighbours_for_every_seed_when_scouts_are_off(
        self, seed
    ):
        """Given optimize() with an ABCOptimizer of colony_size 6 and limit 10**9 (no source reaches it, so no
        scout), when it runs 67 evaluations, then every trial after the first 3 differs in at most one
        coordinate from some earlier trial.

        Oracle: abc.py docstring and Karaboga & Basturk 2007: a neighbour v_i differs from its food source
        x_i, which is an earlier trial, only in coordinate j. Zero changed coordinates are allowed (phi = 0
        or a clip that keeps the value). sn = colony_size // 2 = 3 initial trials are excluded.
        """
        direction = DIRECTIONS[seed % 2]
        space = _mixed_space()
        opt = ABCOptimizer(
            space, direction=direction, colony_size=SMALL_COLONY, limit=NO_SCOUT_LIMIT, seed=seed
        )

        study = optimize(stub_objective, space, direction=direction, max_evals=INVARIANT_EVALS, optimizer=opt)

        assert _max_changed_coordinates(study, SMALL_COLONY // 2) <= 1


class TestFreshProcess:
    """A seeded run gives the same trials in a fresh interpreter, whatever its global state or hash seed."""

    def test_optimize_should_return_identical_trials_when_the_global_random_streams_differ(self):
        """Given the child program, run with seed 3 once after seeding the global numpy and stdlib random
        streams with 0 and once with 1, when both finish, then both exit with code 0 and empty stderr, print
        the same trial records, and those equal the in-process run with seed 3.

        Oracle: README 'The same seed reproduces the same search'; isolation from the global streams is
        inferred from it. The global streams are moved only inside the child, so this process's global state
        stays untouched (P-T2-02). Discriminates an optimizer that draws from np.random or random instead of
        its own generator (the original ABC.py uses the global stdlib stream).
        """
        quiet = run_program(CHILD_PROGRAM, _child_input(QUIET_GLOBALS), HASH_SEEDS[0], str(CHILD_EVALS))
        noisy = run_program(CHILD_PROGRAM, _child_input(NOISY_GLOBALS), HASH_SEEDS[0], str(CHILD_EVALS))

        warm = _in_process_records()

        assert (quiet.returncode, quiet.stderr) == (0, "")
        assert (noisy.returncode, noisy.stderr) == (0, "")
        assert noisy.stdout == quiet.stdout
        assert ast.literal_eval(quiet.stdout) == warm

    def test_optimize_should_return_identical_trials_when_the_hash_seed_differs(self):
        """Given the child program with seed 3, run with PYTHONHASHSEED 0, 1 and 2, when all finish, then each
        exits with code 0 and empty stderr, all print the same trial records, and those equal the in-process
        run with seed 3 (cold process against warm process).

        Oracle: README 'The same seed reproduces the same search'. The hash seed changes str hashing, so any
        set or dict iteration order or arithmetic that rests on the hashes of the Categorical string choices
        or the parameter names would change; a result that depended on it would differ between the children.
        Three hash seeds instead of two: a hash-dependent value can coincide for two seeds (hash('relu') % 5
        is 0 for both 0 and 1), so one more run lowers that chance (P-T1-34). Defined last: the most
        expensive test (P-T6-20).
        """
        children = _run_with_hash_seeds(HASH_SEEDS)

        warm = _in_process_records()

        assert [(c.returncode, c.stderr) for c in children] == [(0, "")] * len(HASH_SEEDS)
        assert [c.stdout for c in children[1:]] == [children[0].stdout] * (len(HASH_SEEDS) - 1)
        assert ast.literal_eval(children[0].stdout) == warm
