"""Written test plan of the meta-hpo test suite, plus the suite-wide hooks and the hang guard.

Suite
-----
The suite under tests/ checks the metahpo package: the search space, the ABC operators, the ask/tell
optimizer base, the ABC optimizer, the study results and the optimize() facade. It is built top-down: the
system is first split into features, each feature gets one plan, and each plan lists its scenario groups and
the test files that hold them. Every file under tests/ belongs to exactly one plan.

Features
--------
1. search space (metahpo.space)
2. ABC operators (metahpo.operators)
3. optimizer base protocol (metahpo.optimizers.base)
4. ABC optimizer (metahpo.optimizers.abc)
5. study results (metahpo.study)
6. optimize facade (metahpo.facade)
7. cross-cutting (the whole package)

Plans
-----
File rows read 'file | layer | content'. The layer is one of unit, integration, system or
static (outside the triangle, P-T1-48).

Plan: search space
Scenario groups: existing decoding and validation; equivalence classes and strict partitions of Int and
Categorical; spec and code boundaries (u = 0, u = 1.0, share boundaries, log ends); move() checked against
rows(); rows() for every mix of column kinds; invalid parameters and every output value; statements and
branches found unreached; specification gaps; resolved ambiguities; generated special strings, bytes and
mixed spaces; hand-written examples behind the grammar properties; sample() with a real numpy Generator
    test_space.py | unit | existing decoding and validation tests, kept unchanged
    test_space_partition.py | unit | equivalence classes, strict partitions, exhaustive small domains
    test_space_boundaries.py | unit | spec, implicit and code-level boundaries, move() against rows()
    test_space_move.py | unit | one-coordinate move() against rows() for every coordinate
    test_space_rows.py | unit | rows() for no, one and several Int and Categorical columns
    test_space_errors.py | unit | constructor error paths and output coverage of Int and Categorical
    test_space_coverage.py | unit | statement and branch coverage of space.py
    test_space_spec.py | unit | specification gaps that test_space.py leaves open
    test_space_specified.py | unit | resolved ambiguities and unspecified behaviour of space.py
    test_space_fuzz.py | unit | generated special strings, bytes and random valid mixed spaces
    test_space_grammar.py | unit | hand-written Space examples behind test_fuzz_grammar.py
    test_space_sample.py | integration | Space.sample with a real numpy Generator

Plan: ABC operators
Scenario groups: neighbour_value clip formula; fitness and fitness_scalar branches; onlooker_probabilities
including the all-zero case; stdlib mutation harness over operators.py; both sides of every threshold;
equivalence classes of the formulas; generated valid inputs against invariants; branch coverage
    test_operators.py | unit | formulas with discriminating literal inputs
    test_operators_mutation.py | unit | mutation harness and kill matrix for the operator tests
    test_operators_boundaries.py | unit | both sides of every threshold in operators.py
    test_operators_partition.py | unit | equivalence classes of the fitness and onlooker formulas
    test_operators_fuzz.py | unit | generated valid inputs checked against invariants
    test_operators_coverage.py | unit | branch coverage from the control flow graphs

Plan: optimizer base protocol
Scenario groups: direction handling and the worst-score mapping of NaN and inf; best_params and best_score
bookkeeping; constructor errors; reproducible random stream across a refill of the uniform block; branches
found unreached; specification gaps; resolved ambiguities
    test_base.py | unit | Optimizer and Candidate through the public ask/tell interface
    test_base_rng.py | unit | reproducibility of the random stream across block refills
    test_base_coverage.py | unit | direction check and reported best found by the coverage measurement
    test_base_spec.py | unit | specification gaps of the common optimizer interface
    test_base_specified.py | unit | resolved ambiguities and unspecified behaviour of the optimizer contract

Plan: ABC optimizer
Scenario groups: init batch and partial tell; employed sweep; onlooker roulette; scout and limit; greedy
acceptance; protocol errors; FSM model; thresholds of colony, acceptance and scout; equivalence classes of
configuration and scores; statements and branches found unreached; specification gaps; generated score
streams
    test_abc.py | unit | existing phase, scout, determinism, cache and protocol tests, kept unchanged
    test_abc_phases.py | unit | phase schedule against the original ABC.py
    test_abc_acceptance.py | unit | greedy acceptance, trial counters and scout, mutation guided
    test_abc_protocol.py | integration | ask/tell protocol with the real Space, candidate invariants
    test_abc_model.py | unit | Mealy FSM of the ask/tell protocol, W-method sequences
    test_abc_boundaries.py | unit | thresholds of the colony, acceptance, scout and best-tracking rules
    test_abc_partition.py | unit | equivalence classes of configuration, protocol, scores and acceptance
    test_abc_coverage.py | unit | statement and branch coverage of optimizers/abc.py
    test_abc_spec.py | unit | specification gaps beyond test_abc.py
    test_abc_fuzz.py | unit | generated score streams with non-finite and extreme values

Plan: study results
Scenario groups: Trial.failed for non-finite scores; best_trial, best_params and best_score per direction;
n_calls without cached repeats; ValueError without a finite trial; equivalence classes of scores;
specification gaps and resolved ambiguities
    test_study.py | unit | Trial and Study bookkeeping
    test_study_partition.py | unit | equivalence classes of scores, directions and cached flags
    test_study_coverage.py | unit | branch coverage of Study.best_trial
    test_study_spec.py | unit | specification gaps of Trial and Study
    test_study_specified.py | unit | resolved ambiguities and unspecified behaviour of study.py and optimize()

Plan: optimize facade
Scenario groups: exact max_evals budget; cache on and off; optimizer space and direction checks; user
exceptions propagate; scripted optimizer and objective doubles; real ABC chain with edge cases and load;
budget edges, extreme scores and the all-NaN run; equivalence classes of budget and configuration;
branches found unreached; specification gaps; generated spaces and non-finite score streams; the
hand-written cache=False example behind the grammar property; direction 'max' at study, optimizer and
trajectory level
    test_facade.py | unit | optimize() argument checks, budget and cache bookkeeping
    test_facade_budget.py | unit | exact trial and call counts of the evaluation budget
    test_facade_doubles.py | integration | scripted optimizer, objective stubs, spies and mocks
    test_facade_integration.py | integration | facade through the real ABC chain
    test_facade_boundaries.py | integration | budget edges, extreme scores and the all-NaN run
    test_facade_partition.py | integration | equivalence classes of budget and configuration
    test_facade_coverage.py | unit | branch coverage of optimize() and the coverage measurement
    test_facade_spec.py | unit | specification gaps of optimize()
    test_facade_fuzz.py | integration | generated cases run end to end through optimize()
    test_facade_grammar.py | integration | hand-written cache=False example behind test_fuzz_grammar.py
    test_facade_direction.py | integration | direction='max' with an explicit ABCOptimizer

Plan: cross-cutting
Scenario groups: README examples end to end; seeded determinism and order independence; pinned
characterisation snapshots and runtime budget; seeded fuzzing with an exception oracle; grammar-derived
spaces and pairwise configuration coverage; public API contract and configuration smoke test; ask/tell
contract of every optimizer in OPTIMIZER_FACTORIES; static guards on source and test files; suite layout
and this plan; requirement-test traceability matrix
    test_end_to_end.py | system | README examples as acceptance tests
    test_determinism.py | system | same seed gives the same study, isolation from outside randomness
    test_regression.py | system | pinned seeded outcomes and runtime budget
    test_fuzz.py | integration | abnormal inputs against every public entry point
    test_fuzz_grammar.py | system | grammar-derived spaces run through optimize()
    test_api_contract.py | system | metahpo.__all__, signatures, defaults, contract of every optimizer
    test_static.py | static (outside the triangle, P-T1-48) | source and test file guards
    test_suite_layout.py | static (outside the triangle, P-T1-48) | this plan against the files on disk
    test_traceability.py | static (outside the triangle, P-T1-48) | requirement to test matrix

The cross-cutting files are end_to_end, determinism, regression, fuzz, fuzz_grammar, api_contract, static,
suite_layout and traceability. Every other file is named test_<module>.py or test_<module>_<aspect>.py after
its unit (space, operators, base, abc, study, facade).

Layers
------
Unit tests check business rules: the docstring formulas, the ABC.py phase rules and the Space decoding.
Integration tests check module-to-module work and the user's function: ABCOptimizer with the real Space
through ask/tell, and optimize() composing optimizer, space, study and the objective. System tests check the
main optimize() flow as a user sees it. Static files sit outside the test triangle (P-T1-48): they read files
and run no library behaviour. Manual: out of scope, because an automated suite does no exploratory or
qualitative testing; exploratory sessions are left to people reading the reports. Larger tests complement
the unit tests and never replace them: every module has its own unit-layer file test_<module>.py. No ratio
between the layers is stated or required.

V-model
-------
Requirements (README, CHANGELOG) are checked by system tests: test_end_to_end.py, test_api_contract.py,
test_regression.py.
Architecture (the ask/tell Optimizer interface, the facade composing optimizer, space and study) is checked
by integration tests: test_abc_protocol.py, test_facade_doubles.py, test_facade_integration.py.
Detailed design (docstring formulas, ABC.py phase rules, Karaboga & Basturk 2007) is checked by unit tests:
        test_operators.py, test_space.py, test_abc.py, test_abc_phases.py.
The process is iterative, not a straight V: a defect found at any level sends the work back to the
specification and the plan, and the plan is updated whenever a file is added.

Strategy and coverage
---------------------
One strategy and one coverage target per unit:
    space | code-based (white box) | branch coverage
    operators | code-based (white box) | branch coverage
    base | code-based (white box) | branch coverage
    abc | model-based on the phase FSM | every transition of the phase FSM
    study | code-based (white box) | branch coverage
    facade | code-based (white box) | branch coverage
The spec names no strategy for base; code-based (white box) is chosen to match the other small units.
No numeric coverage level is claimed.

Phase FSM of the ABC ask/tell protocol (state -> state: condition):
    init -> init: the initial batch is told partially; the next ask hands out the untold sources
    init -> employed: the last initial source is told
    employed -> employed: an employed candidate is told and sources remain in the sweep
    employed -> onlooker: the candidate of the last source is told
    onlooker -> onlooker: an onlooker candidate is told and fewer than SN onlookers have run
    onlooker -> scout: the SN-th onlooker candidate is told
    scout -> employed (scout candidate): max trial >= limit; the scout candidate is told and cycle += 1
    scout -> employed (skipped): max trial < limit; the next ask starts the employed sweep and cycle += 1

Target criterion: branch coverage. The structural order is statement -> branch -> path: branch subsumes
statement at little extra effort, and path coverage is often intractable. Path coverage is limited to
0/1/many iterations of three loops: the onlooker roulette while loop in ABCOptimizer.ask, the candidate loop
in ABCOptimizer.tell and the optimize while loop. All-Uses is applied only to the ABC state (foods, f, fit,
trial, prob, phase, cycle and the employed and onlooker cursors) and to the facade's `seen` cache.
Measurement: no coverage tool may be added, so coverage is tracked by a per-module branch inventory in
docstrings.

Blind spots
-----------
(1) Compiler: out of scope. The library is pure Python and numpy with no compiled code of its own;
interpreter variation is covered by the CI matrix (Python 3.10 to 3.14 on Ubuntu, 3.14 on Windows and macOS),
and the versions are recorded in the report header of every run.
(2) Parallel programming: out of scope. The library is single-threaded and no thread-safety contract is
specified.
(3) Non-functional: out of scope except for the hang guard below. No performance requirement is specified.
(4) Floating point: in scope. The clip bounds of neighbour_value, fitness at 0, negative, +inf and NaN,
Float log decoding with approximate comparison, and Int and Categorical endpoints at u=1.0.
(5) System level: out of scope. The library uses no I/O, network or OS resources; the platform is recorded
in the report header.
(6) Requirement misunderstanding: in scope through test_traceability.py and test_api_contract.py, because
the source does not list it among the blind spots that may be ignored.

Selection
---------
Step 1: identify the affected module from the changed file (src/metahpo/<module>.py or
src/metahpo/optimizers/<module>.py; the package __init__ files map to __init__).
Step 2: select that module's test files test_<module>*.py, the rows of every module that imports it, and the
files that are always selected.
Import graph of src/metahpo (module <- the modules that import it; both package __init__ files count as
__init__):
    space <- base, abc, facade, __init__
    operators <- abc
    base <- abc, study, facade, __init__
    abc <- facade, __init__
    study <- facade, __init__
    facade <- __init__
Each row below contains the rows of all modules that import it, so the selection is closed over the import
graph: a test that uses a module's public names, or the names of a module that imports it, is selected. The
package __init__ files, which import every module, are not followed as a path from a test to a module,
otherwise every row would be the whole suite. The rows are a module-level superset of the
modification-traversing tests (P-T6-32).
    space: test_space*.py, test_base*.py, test_abc*.py, test_study*.py, test_facade*.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    operators: test_operators*.py, test_abc*.py, test_facade*.py, test_base*.py, test_space_partition.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    base: test_base*.py, test_abc*.py, test_study*.py, test_facade*.py, test_space_partition.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    abc: test_abc*.py, test_facade*.py, test_base*.py, test_space_partition.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    study: test_study*.py, test_facade*.py, test_abc.py, test_abc_acceptance.py, test_space_partition.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    facade: test_facade*.py, test_abc.py, test_abc_acceptance.py, test_space_partition.py,
        test_end_to_end.py, test_determinism.py, test_regression.py, test_fuzz.py, test_fuzz_grammar.py,
        test_api_contract.py
    __init__: test_api_contract.py
Always selected: test_static.py, test_suite_layout.py, test_traceability.py
Statement-level selection of modification-traversing tests (P-T6-32) is out of scope for this plan.

Regression suite
----------------
The whole tests/ directory, run by plain `python -m pytest` through pyproject testpaths = ["tests"], is the
regression suite. test_regression.py holds the pinned seeded outcomes. Nothing depends on manual steps.

Independence
------------
Tests share no mutable module state. Every test builds its own Space, optimizer and seed, and every fixture
is function-scoped. Setting the environment variable METAHPO_TEST_REVERSE=1 runs the collected tests in
reverse order, which shows an order dependence as a failure; any other value, or no value, keeps the order.

Input space convention
----------------------
Every new test module docstring states the size of its input space after the marker 'Input space:' and its
subset strategy after the marker 'Subset:'. Exhaustive testing is not the goal: neighbour_value alone takes
three float64 values, Space takes d unit-cube coordinates per row, and the ABC protocol accepts unbounded
ask/tell sequences, so every module picks a subset by equivalence classes, boundaries and model coverage.
The existing test_space.py and test_abc.py are exempt by name.

Run conditions and hang guard
-----------------------------
pytest_report_header prints the Python, numpy, platform and metahpo versions in the header of every run.
The autouse hang_guard fixture arms faulthandler for HANG_TIMEOUT seconds around each test and exits the run
with a traceback when a test hangs; the traceback is visible with -s, because fd capture swallows it.
"""

import faulthandler
import os
import platform
import sys

import numpy
import pytest

import metahpo

HANG_TIMEOUT = 30.0
"""Seconds one test may run before the hang guard stops the run.

Origin: the 4000 ms per-test timeout of the mutation test template; the source has no general threshold.
The value adds a wide margin for the slowest CI runners (Python 3.10, Windows and macOS) while the whole
suite is meant to finish in well under a minute.
"""

REVERSE_ENV = "METAHPO_TEST_REVERSE"
"""Environment variable that reverses the collection order when it is set to '1'."""


def pytest_report_header(config):
    """Records the run conditions: Python, numpy, platform and metahpo versions."""
    return [
        f"python: {platform.python_version()} ({sys.version.splitlines()[0]})",
        f"numpy: {numpy.__version__}",
        f"platform: {platform.platform()}",
        f"metahpo: {metahpo.__version__}",
    ]


def pytest_collection_modifyitems(config, items):
    """Reverses the collected tests in place when METAHPO_TEST_REVERSE is '1'; otherwise keeps the order."""
    if os.environ.get(REVERSE_ENV) == "1":
        items.reverse()


@pytest.fixture(autouse=True)
def hang_guard():
    """Arms faulthandler for HANG_TIMEOUT seconds around one test and always disarms it afterwards."""
    faulthandler.dump_traceback_later(HANG_TIMEOUT, exit=True, file=sys.__stderr__)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
