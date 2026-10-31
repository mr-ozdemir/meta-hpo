"""Greedy acceptance, trial counters, best tracking and the scout step of ABCOptimizer: mutation-guided tests.

Every test here was derived from a mutant that survived tests/test_abc.py and tests/test_space.py, or from a
DU row of the table below. Oracles: README (Behaviour), CHANGELOG, the docstrings of Optimizer, ABCOptimizer,
operators, Space and Study, the original hyp_optimizer ABC.py and Karaboga & Basturk (2007). The documented
non-underscore counters opt.trial, opt.phase, opt.cycle and opt.food_number are read, never written. Line
numbers: abc = optimizers/abc.py, base = optimizers/base.py, facade = facade.py.

Data-flow criterion: All-Uses. Required rows: 3, 6, 7, 7b, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 23, 24a,
24b, 26, 27, 31, 33, 34. Rows 31 and 33 are executed by the existing test_cache_skips_repeated_parameter_sets
(cross-reference, no new test). A row marked 'same pair' repeats a (definition, use) pair of another row
along a different path; All-Uses needs one path per pair, All-DU-Paths would need every row.

No.  | Variable      | Definition                | Use                         | Definition-Use Path
1    | trial         | abc:43 constructor        | abc:97 failure increment    | INFEASIBLE: every init tell
     |               |                           |                             | redefines trial at abc:126
     |               |                           |                             | before the phase leaves
     |               |                           |                             | 'init'
2    | trial         | abc:43 constructor        | abc:65-66 scout check       | INFEASIBLE: as row 1
3    | trial         | abc:126 via init tell     | abc:97 employed failure     | init tell -> employed ask ->
     |               |                           |                             | abc:94 false (tie) -> abc:97
4    | trial         | abc:126 via init tell     | abc:65-66 scout check       | INFEASIBLE: each source gets
     |               |                           |                             | one employed tell (abc:97 or
     |               |                           |                             | abc:126) before 'scout'
5    | trial         | abc:126 via employed      | abc:97 onlooker failure     | not required: same pair as
     |               | improve                   |                             | row 3 (definition executed by
     |               |                           |                             | the keep-trial-zero test)
6    | trial         | abc:97 employed failure   | abc:97 onlooker failure     | employed fail -> onlooker on
     |               |                           |                             | the same source fails
7    | trial         | abc:126 via onlooker      | abc:65-66 scout check,      | onlooker improves source 1,
     |               | improve                   | fired                       | check reads [2, 0], scouts 0
7b   | trial         | abc:126 via improvements  | abc:66 scout check, false   | every tell improves, check
     |               |                           |                             | reads [0, 0] with limit 1
8    | trial         | abc:97 onlooker failure   | abc:66 true, trial == limit | 2 failures per source, limit
     |               |                           |                             | 2
9    | trial         | abc:97 onlooker failure   | abc:66 false, trial ==      | 2 failures per source, limit
     |               |                           | limit-1                     | 3
10   | fit           | abc:125 via init tell     | abc:94 onlooker greedy test | both employed fail, first
     |               |                           |                             | onlooker compares with init
     |               |                           |                             | fitness (true and false)
11   | trial         | abc:97 / abc:126          | abc:65 worst index (max     | tie [2, 2]; [2, 0]; [2, 4]
     |               |                           | key)                        |
12   | trial         | abc:126 via scout tell    | abc:97 next-cycle employed  | scout resets, cycle 1 fails
     |               |                           | fail                        |
13   | trial         | abc:97 onlooker fail,     | abc:97 employed fail, cycle | counter carried across the
     |               | cycle c                   | c+1                         | cycle boundary
14   | prob          | abc:44 constructor        | abc:71 roulette test        | INFEASIBLE: prob[1.0] at
     |               |                           |                             | abc:44 never reaches abc:71
     |               |                           |                             | because _advance recomputes
     |               |                           |                             | prob at abc:132 before any
     |               |                           |                             | onlooker ask
15   | fit           | abc:125 via init tell     | abc:94 employed, false      | init -> employed tie or NaN
     |               |                           | branch                      |
16   | prob          | abc:132                   | abc:71 for successive       | via the abc:138 sweep advance
     |               |                           | onlookers                   |
17   | fit           | abc:125 via init tell     | abc:132 prob computation    | both employed fail, prob from
     |               |                           |                             | init fitness
18   | fit           | abc:125 via employed      | abc:132 prob computation    | not required: same pair as
     |               | improve                   |                             | row 17
19   | fit           | abc:125 via scout tell    | abc:94 next employed test   | scout point told, then its
     |               |                           |                             | neighbour is judged
20   | fit           | abc:125 via onlooker      | abc:94 next onlooker        | not required: same pair as
     |               | improve                   |                             | rows 10, 15, 19
21   | prob          | abc:132 with unequal      | abc:71 roulette skip branch | not required: same pair as
     |               | values                    |                             | row 16; the skip depends on
     |               |                           |                             | seed-specific uniforms
22   | fit           | abc:42 constructor        | abc:94 / abc:132            | INFEASIBLE: init tells
     |               |                           |                             | redefine every entry at
     |               |                           |                             | abc:125
23   | prob          | abc:132, all entries 1.0  | abc:71 first onlooker,      | equal fitness, first draw
     |               |                           | index 0                     | accepted
24a  | best_f,best_x | base:45 / base:44         | abc:83 first tell, f <      | first init candidate finite
     |               | constructor               | best_f                      |
24b  | best_x        | base:44 None              | abc:83 'best_x is None'     | all init scores NaN
     |               |                           | with f = inf                |
25   | best_f        | abc:84 via                | abc:83                      | not required: same pair as
     |               | employed/onlooker         |                             | row 26; timing differs from
     |               |                           |                             | ABC.py memorize_best_source
26   | best_f        | abc:84 first init         | abc:83 second init          | equal init scores
     |               | candidate                 | candidate                   |
27   | best_f,best_x | abc:84 via init tell      | base:61 best_params,        | read right after the init
     |               |                           | base:65 best_score          | tell
28   | best_x        | abc:84 via                | base:61 best_params         | not required: same pair as
     |               | employed/onlooker         |                             | row 27; unspecified timing
29   | best_x        | abc:84 via scout tell     | base:61 best_params         | not required: as row 28
30   | best_f        | abc:84 via any later tell | base:65 best_score          | not required: as row 28
31   | seen          | facade:46 earlier ask     | facade:43 hit across asks   | existing cache test
32   | seen          | facade:38 / facade:46     | facade:43-44 with           | INFEASIBLE: 'cache and'
     |               |                           | cache=False                 | short-circuits at facade:43
     |               |                           |                             | and facade:45 skips :46
33   | seen          | facade:46 earlier ask     | facade:44 reuse across asks | existing cache test
34   | seen          | facade:46 same batch      | facade:43-44 within one     | one-choice space, 5 equal
     |               |                           | batch                       | keys in the initial batch

Survivors of the existing suite (reach / infect / propagate measured on an instrumented copy; line counts
from python trace under the existing suite: abc:65 489, abc:66 489, abc:83 4768, abc:92 31, abc:94 4699,
abc:105 4700, abc:138 2346, base:69 4768, base:70 16; coverage was high while these mutants lived):

Mutant (site)            | Reached / infected / propagated | Why the existing suite missed | Verdict
                         |                                 | it                            |
M01 abc:94 '>' to '>='   | reached; infected (ties in 5    | only aggregates asserted:     | tie test
                         | tests); not propagated to an    | best score, call count, scout |
                         | asserted value                  | in kinds                      |
M02 abc:66 '>=' to '>'   | reached; infected (trial ==     | weak oracle 'scout in kinds': | at-limit
                         | limit in 5 tests); masked       | the scout only comes a cycle  | test
                         |                                 | later                         |
M05 abc:92 _end_cycle    | reached and infected on every   | 'scout in kinds'; limit=1000  | one-scout
deleted                  | scout tell (phase stays         | in the phase-order test       | test
                         | 'scout'); masked                |                               |
M06 abc:83 best update   | reached; infected (opt.best     | tests read Study.best_*,      | lower-init
deleted                  | never set); not propagated      | never opt.best_*              | test
M07 abc:83 '<' to '<='   | reached; infected (equal score  | opt.best_params never read    | tie-best
                         | with distinct x in 4 tests);    |                               | test
                         | masked                          |                               |
M08 base:70 inf to 0.0   | reached (base:70 count 16);     | NaN only when maximising,     | non-finite
                         | infected only in the max NaN    | where internal 0.0 still      | table, NaN
                         | test; masked                    | loses to -x; Study.best_*     | tests
                         |                                 | ignores optimizer             |
M09 base:69 isfinite to  | reached; never infected         | no +inf or -inf score is told | non-finite
isnan                    |                                 |                               | table
M10 abc:105 '>=' to '>'  | reached; infected (k == i drawn | masked by '<= 1' on the       | partner
                         | in 7 tests); propagated to a    | changed coordinate count      | test
                         | 0-change cand.                  |                               |
M11 abc:65 max to min    | reached; infected (choice       | 'scout in kinds' never says   | unequal
                         | differs in 5 tests); masked     | which source is replaced; [2, | test
                         |                                 | 2] tie                        |
M16 abc:138 advance      | reached and infected wherever   | phase test checks labels      | sweep test
deleted                  | onlookers run; masked           | only; convergence test is     |
                         |                                 | tolerant                      |
M17 abc:83 'or best_x is | reached; never infected         | no all-non-finite prefix is   | all-NaN
None' cut                |                                 | told                          | test
M18 abc:97 '+= 1' to '+= | reached and infected on every   | scouts just come earlier;     | tie test
2'                       | failure (3177); masked          | 'scout in kinds' still holds  |
M19 abc:66 'limit' to    | reached; infected (trial ==     | aggregate oracles only        | below-limit
'limit - 1'              | limit - 1 in 4 tests); masked   |                               | test
M21 abc:91 greedy scout  | reached; infected (scout told   | no test reads the replaced    | scout-worse
(scratch)                | 5.0 against a 1.0 source);      | source                        | test
                         | masked                          |                               |
R7 abc:66 trial[worst]   | reached (489 scout checks);     | 'scout in kinds' holds once   | second-
to trial[0]              | infected in 19 (source 0 below  | source 0 reaches the limit    | source
                         | the limit, another source at    | later; no other test here has | scout test
                         | it); masked                     | source 0 below while another  |
                         |                                 | source is at the limit        |

Equivalent or practically undetectable, no test written:
- M13 abc:71 '>=' to '>': differs only when a uniform equals prob exactly (probability about 2^-53, reached
  thousands of times, never infected); u cannot be chosen through the public API. Practically undetectable,
  not proven equivalent.
- M14 operators fitness_scalar 'f >= 0' to 'f > 0': at f == 0 both branches give 1.0 (1 / (1 + 0) and
  1 + |0|, also for -0.0); reached (f == 0 in the neighbour test), no infection.
- M20 abc:66 '>= limit' to '> limit - 1': identical for integer trial and integer limit; a float limit is
  accepted by the constructor but unspecified.
- F1 facade:46 setdefault to assignment: line 46 meets an existing key only on a cache hit, where score is
  the stored value itself, so the assignment writes the same value.
- C3 HVI (best_f = 0.0 re-set in ABCOptimizer.__init__): the 'best_x is None' disjunct makes the first tell
  overwrite it, so best_f and best_x agree after every tell; only best_score before any tell differs
  (inf vs 0.0), which no spec source states.
Already killed by the existing suite (no new-test credit): M03, M04, M12, M15, C1, C2, S1, S2, F2.

Mutant inventory by level (killing tests are in this module unless noted):
- operand: M15 'trial = 1' (existing scout test; here the sweep test), M18 '+= 2' (tie test), M11 max to
  min (unequal test), S1 move key slice 'key[j:]' (move-key test; existing cache test).
- expression: ROR M01 (tie test), M02 (at-limit test), M07 (tie-best test), M10 (partner test), M13
  (practically undetectable), M14 (equivalent), M20 (equivalent); predicate M09 (non-finite table);
  connector M17 (all-NaN test); arithmetic M19 (below-limit test); S2 stale move key (move-key test);
  wrong index operand R7 'trial[worst]' to 'trial[0]' in the scout check (second-source scout test).
- statement: sdl M03 (tie test), M04 (after-failure reset test), M05 (one-scout test), M06 (lower-init
  test), M12 (skip-when-improves test), M16 (sweep test); F1 setdefault to assignment (equivalent); F2 cache
  filled after tell (one-choice cache test); M21 greedy guard inserted (scout-worse test).
- class level: OMD C1 (ABCOptimizer.tell override deleted, base raises NotImplementedError): KILLED by every
  test that tells. SKR/PCD-like C2 (super().__init__ dropped, AttributeError): KILLED by every test. HVI C3:
  equivalent (see above). HVD n/a: ABCOptimizer declares no attribute that hides a base attribute. OMM and
  OMR n/a: Python has no method overloading. AMC n/a: Python has no access modifiers (a leading underscore is
  a naming convention only). ATC n/a: attribute types are not declared or enforced at runtime, so there is
  no type to change.
Scratch run: 29 mutants, killed 24, survived 0, equivalent 5 (killed + survived + equivalent == total);
APFD of this module in source order uses m = 24 killed mutants.

Unspecified (no correct-value assertion beyond the cited source):
- the optimizer's best_params/best_score when every told score is non-finite: only ABC.py initial() lines
  87-88 state it; Study.best_trial raises ValueError in the same situation;
- best_score before the first tell; float values of limit;
- the exact semantics of opt.trial/phase/cycle/food_number (not in README or CHANGELOG; read-only here);
- best tracking timing: the library updates best on every tell, ABC.py once per phase (DU rows 25, 28-30);
- changed-coordinate counts are exact for every seed except a measure-zero set (phi == 0, equal
  coordinates); ANY_SEED = 0 is fixed for reproducibility only.

Input space: Every ask/tell sequence with any scores, colony_size, limit and seed, which is unbounded.
Subset: Mutation-guided scenarios on small colonies with scores chosen so that each acceptance, trial-counter
and scout decision is visible.
"""

import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

ANY_SEED = 0
NEVER_SCOUT_LIMIT = 1000
TIE_SCORE = 1.0
CONSTANT_SCORE = 1.0
BETTER_SCORE = 0.5
WORSE_SCOUT_SCORE = 9.0


def _tell_initial_batch(opt, scores):
    """Asks the initial batch and tells it `scores` in one call; returns the initial candidates."""
    initial = opt.ask()
    opt.tell(initial, list(scores))
    return initial


def _tell_single_candidates(opt, scores):
    """Asks one candidate per entry of `scores` and tells it that score; returns the asked candidates."""
    asked = []
    for score in scores:
        (cand,) = opt.ask()
        opt.tell([cand], [score])
        asked.append(cand)
    return asked


def _record_key_params_and_rows_decode(opt, space, n_asks, score):
    """Asks and tells `n_asks` single candidates with `score`; returns (key, params) and the rows() decode."""
    recorded, decoded = [], []
    for _ in range(n_asks):
        (cand,) = opt.ask()
        keys, params = space.rows(cand.x[None, :])
        recorded.append((cand.key, cand.params))
        decoded.append((keys[0], params[0]))
        opt.tell([cand], [score])
    return recorded, decoded


def test_abc_should_increment_trial_and_keep_source_when_neighbour_score_ties():
    """Given init scores [1.0, 1.0], when the first employed neighbour is told the same 1.0, then source 0
    fails exactly once: trial == [1, 0].

    Infection condition: the tie fitness_scalar(1.0) == fit[0] == 0.5 at the greedy test (site: abc.py:94);
    only a tie separates '>' from '>=' (M01 gives [0, 0]); exact value 1 also kills M18 ('+= 2' gives
    [2, 0]). Oracle: original ABC.py:113 'fitness_sol > fitness[i]' replaces, else ABC.py:121 'trial + 1';
    hand value 0 + 1 = 1. DU rows 3, 15.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=NEVER_SCOUT_LIMIT, seed=ANY_SEED)
    _tell_initial_batch(opt, [TIE_SCORE, TIE_SCORE])

    _tell_single_candidates(opt, [TIE_SCORE])
    trial = list(opt.trial)

    assert trial == [1, 0]


def test_abc_should_reset_trial_to_zero_when_score_strictly_improves_after_a_failure():
    """Given init [1.0, 1.0] and two failing employed tells (trial [1, 1]), when the first onlooker is told
    the strictly better 0.5, then its source is reset: trial == [0, 1].

    The first onlooker is source 0 because equal fitness gives prob exactly 0.9 * 0.5 / 0.5 + 0.1 == 1.0
    and every uniform is below 1.0. Infection of M04 (site: 'trial[i] = 0' at abc.py:126) needs trial > 0
    before the improvement; an improvement directly after init leaves M04 uninfected because trial is
    already 0. M04 gives [1, 1]. Oracle: ABC.py:140-141 (onlooker improvement sets trial 0), ABC.py:127
    (prob formula). DU rows 3, 10, 17, 23.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=NEVER_SCOUT_LIMIT, seed=ANY_SEED)
    _tell_initial_batch(opt, [TIE_SCORE, TIE_SCORE])
    _tell_single_candidates(opt, [TIE_SCORE, TIE_SCORE])

    _tell_single_candidates(opt, [BETTER_SCORE])
    trial = list(opt.trial)

    assert trial == [0, 1]


def test_abc_should_keep_trial_zero_when_first_employed_score_improves():
    """Given init [1.0, 1.0], when both employed neighbours are told the strictly better 0.5, then trial
    stays [0, 0] and the onlooker phase starts.

    Negative half of the trial-reset pair: the reset at abc.py:126 is executed but trial was already 0,
    so M04 is executed with no infection and gives the same result (non-discriminating by design; the
    discriminating half is the after-failure reset test); it kills only the class-level mutants C1 and C2.
    Oracle: ABC.py:113-114 (improvement sets trial 0) and the phase order employed -> onlooker of ABC.py
    and the abc.py module docstring. Executes the definition of DU row 5 (abc.py:126 via employed improve).
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=NEVER_SCOUT_LIMIT, seed=ANY_SEED)
    _tell_initial_batch(opt, [TIE_SCORE, TIE_SCORE])

    _tell_single_candidates(opt, [BETTER_SCORE, BETTER_SCORE])
    state = (list(opt.trial), opt.phase)

    assert state == ([0, 0], "onlooker")


def test_abc_should_keep_first_source_as_best_when_init_scores_tie():
    """Given two initial candidates with distinct params, when both are told 1.0, then the first one is
    the best: best_params == init[0].params and best_score == 1.0.

    Infection of M07 (site: '<' at abc.py:83 turned into '<=') needs equal scores with distinct decoded
    params; M07 reports init[1].params. Oracle: ABC.py initial() lines 87-88 (best = source 0) and
    memorize_best_source line 64 (strict '<' keeps the first of equal values). DU rows 24a, 26, 27.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, seed=ANY_SEED)
    initial = opt.ask()
    assert initial[0].params != initial[1].params

    opt.tell(initial, [TIE_SCORE, TIE_SCORE])
    best = (opt.best_params, opt.best_score)

    assert best == (initial[0].params, TIE_SCORE)


def test_abc_should_report_lower_init_score_as_best_when_second_init_candidate_is_better():
    """Given direction 'min', when the initial batch is told [3.0, 2.0], then best_score == 2.0 and
    best_params == init[1].params.

    The better score comes second, so the best update (site: abc.py:84) must run on a later candidate;
    M06 (update deleted) reports best_score inf and best_params None. Oracle: README 'searches for the
    parameters that ... minimise'; ABC.py memorize_best_source line 64; hand value min(3.0, 2.0) = 2.0.
    DU rows 24a, 27.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction="min", colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [3.0, 2.0])
    best = (opt.best_params, opt.best_score)

    assert best == (initial[1].params, 2.0)


def test_abc_should_count_nan_employed_score_as_failure_when_minimising():
    """Given direction 'min' and init [1.0, nan], when the first employed neighbour is told nan, then the
    source keeps its point and fails once: trial == [1, 0].

    NaN must lose the greedy test (site: abc.py:94 via base.py:69-70): mapped to +inf it has fitness 0,
    and 0 > 0.5 is false. M08 (non-finite mapped to 0.0) gives fitness 1.0 > 0.5 and [0, 0]. Oracle:
    README 'A score of NaN or infinity counts as the worst possible value'; Optimizer docstring '+inf when
    minimising'; operators.fitness_scalar docstring '+inf gives 0'; ABC.py:121. DU rows 3, 15.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction="min", colony_size=4, limit=NEVER_SCOUT_LIMIT, seed=ANY_SEED)
    _tell_initial_batch(opt, [1.0, math.nan])

    _tell_single_candidates(opt, [math.nan])
    trial = list(opt.trial)

    assert trial == [1, 0]


@pytest.mark.parametrize(
    ("direction", "finite_score", "non_finite_score"),
    [
        pytest.param("max", 1.0, math.inf, id="max-inf-discriminating-M09"),
        pytest.param("min", -1.0, -math.inf, id="min-neginf-discriminating-M09"),
        pytest.param("min", 1.0, math.nan, id="min-nan-discriminating-M08-same-M09"),
        pytest.param("min", -1.0, math.inf, id="min-inf-same-M08-M09"),
        pytest.param("max", 1.0, -math.inf, id="max-neginf-same-M09"),
        pytest.param("max", 1.0, math.nan, id="max-nan-same-M08"),
    ],
)
def test_abc_should_treat_non_finite_score_as_worst(direction, finite_score, non_finite_score):
    """Given one finite and one non-finite initial score, when both are told, then the finite candidate is
    the best: best_score == finite_score and best_params == init[0].params.

    Site: base.py:69-70 (non-finite -> +inf internally). Case roles (in the ids): M09 ('isfinite' ->
    'isnan') is infected only by +inf when maximising and -inf when minimising, where the faulty mapping
    makes the non-finite score win; for min/+inf and max/-inf the faulty predicate yields the same
    internal value (same result). M08 (non-finite -> 0.0) wins only against a finite internal value
    above 0.0, which is why the min/inf rows use -1.0 (same result) and the min/nan row uses 1.0
    (discriminating); max/nan maps to -1.0 internally, below 0.0 (same result). Oracle: README 'NaN or
    infinity counts as the worst possible value for the chosen direction'; Optimizer docstring. DU rows
    24a, 27.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction=direction, colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [finite_score, non_finite_score])
    best = (opt.best_params, opt.best_score)

    assert best == (initial[0].params, finite_score)


def test_abc_should_keep_first_init_candidate_as_best_when_all_init_scores_are_nan():
    """Given direction 'min', when both initial candidates are told nan, then best_params ==
    init[0].params and best_score == +inf.

    Only an all-non-finite prefix infects M17 (site: the 'or best_x is None' disjunct at abc.py:83
    deleted): inf < inf is false, so best_x stays None. Oracle: original ABC.py initial() lines 87-88
    (best = source 0 unconditionally) with the Optimizer docstring mapping NaN to +inf when minimising.
    README and docstrings do not state the optimizer's best for an all-failed run; ABC.py is the only
    source (see the unspecified list). M07 reports init[1].params and M08 best_score 0.0 here as well.
    DU row 24b.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction="min", colony_size=4, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [math.nan, math.nan])
    best_params, best_score = opt.best_params, opt.best_score

    assert best_params == initial[0].params
    assert best_score == math.inf


def test_abc_should_visit_onlookers_in_sweep_order_when_fitness_is_equal():
    """Given init [1.0, 1.0] and limit 2, when 2 employed and 2 onlooker neighbours are all told 1.0, then
    every source has failed twice: phase 'scout' and trial == [2, 2].

    Equal fitness makes prob exactly [1.0, 1.0], so every onlooker is accepted and the visit order is set
    by the sweep advance (site: abc.py:138). M16 (advance deleted) sends both onlookers to source 0 and
    gives [3, 1]. Oracle: ABC.py:149 'i = (i + 1) % FOOD_NUMBER' and ABC.py:134; hand count: one employed
    failure plus one onlooker failure per source, 1 + 1 = 2. DU rows 6, 10, 16, 17, 23.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])

    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn))
    state = (opt.phase, list(opt.trial))

    assert state == ("scout", [2, 2])


def test_abc_should_issue_scout_when_trial_equals_limit():
    """Given limit 2 and trial [2, 2] at phase 'scout', when one candidate is asked, then it is a scout
    of the same cycle: phase 'scout', cycle 0, and a fresh point differing from init[0].x in both
    coordinates.

    On-point of the scout boundary (site: '>=' at abc.py:66): M02 ('>') skips the scout and gives
    phase 'employed', cycle 1 and a one-coordinate neighbour. Expected changed count: hand value 2, the
    two Float coordinates 'a' and 'b' written in the arrange block, both redrawn by a scout. A fresh
    uniform point equals a coordinate of init[0] with probability 0; seed fixed for reproducibility.
    Oracle: ABC.py:152 'np.amax(trial) >= LIMIT' and init() lines 72-74 (all coordinates redrawn). DU rows
    8, 11.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    both_coordinates = 2
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    initial = _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn))
    assert (opt.phase, list(opt.trial)) == ("scout", [2, 2])

    (cand,) = opt.ask()
    observed = (opt.phase, opt.cycle, int(np.count_nonzero(cand.x != initial[0].x)))

    assert observed == ("scout", 0, both_coordinates)


def test_abc_should_skip_scout_when_trial_is_one_below_limit():
    """Given limit 3 and trial [2, 2] at phase 'scout', when one candidate is asked, then no scout is
    issued: phase 'employed', cycle 1, and a neighbour of init[0] differing in exactly 1 coordinate.

    Off-point of the scout boundary (site: abc.py:66). Same result for M02 (non-discriminating); it guards
    the 'limit - 1' mutant M19, which scouts here. Source 0 is still init[0] because every tell tied; the
    neighbour changes exactly one coordinate for every seed except a measure-zero set (phi == 0). Oracle:
    ABC.py:152 (2 >= 3 is false), increase_cycle, and ABC.py:96-99 (one parameter changed). DU row 9.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=3, seed=ANY_SEED)
    sn = opt.food_number
    initial = _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn))
    assert (opt.phase, list(opt.trial)) == ("scout", [2, 2])

    (cand,) = opt.ask()
    observed = (opt.phase, opt.cycle, int(np.count_nonzero(cand.x != initial[0].x)))

    assert observed == ("employed", 1, 1)


def test_abc_should_issue_at_most_one_scout_per_cycle_when_two_sources_are_exhausted():
    """Given limit 2 and trial [2, 2], when the scout is told 1.0 and the next candidate is asked, then the
    next cycle's employed phase runs although source 1 still has trial 2 >= limit: phase 'employed',
    cycle 1, trial == [0, 2].

    Two exhausted sources are the input where 'at most one scout per cycle' matters (site: _end_cycle
    after the scout tell, abc.py:92). M05 (call deleted) stays in phase 'scout' with cycle 0 and scouts
    source 1 in the same cycle. Oracle: ABC.py send_scout_bees lines 151-153 (one init(argmax) per call,
    one call per cycle); abc.py module docstring 'at most one scout per cycle'. DU row 8.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn + 1))

    opt.ask()
    state = (opt.phase, opt.cycle, list(opt.trial))

    assert state == ("employed", 1, [0, 2])


def test_abc_should_scout_first_source_when_trials_tie():
    """Given limit 2 and tied trial counters [2, 2], when the cycle-0 scout is told 1.0, then source 0 is
    the one replaced: trial == [0, 2].

    Tie case of the worst-source choice (site: max over trial at abc.py:65): same result for M11 (tie),
    since min and max agree on [2, 2]; the unequal-trial test is the discriminating half. Oracle: ABC.py:153
    'trial.argmax(axis=0)', numpy returns the first maximum; init() line 81 sets the replaced trial to 0.
    DU rows 8, 11.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn))

    _tell_single_candidates(opt, [CONSTANT_SCORE])
    trial = list(opt.trial)

    assert trial == [0, 2]


def test_abc_should_replace_source_with_most_trials_when_trials_differ():
    """Given limit 2 and constant scores, when the cycle-1 scout tell is made with trial [2, 4], then source
    1 (most trials) is replaced and cycle 2 starts: trial == [2, 0], phase 'employed', cycle 2.

    Unequal trials are the only input where max -> min (M11, site: abc.py:65) chooses differently; M11
    gives [0, 4]. The cycle-0 tie [2, 2] does not infect M11. Hand trace: cycle 0 adds 2 per source and the
    scout resets source 0 ([0, 2]); cycle 1 adds 2 per source ([2, 4]) and the scout resets source 1.
    The cycle-1 employed tell on source 0 is judged at abc.py:94 against the fitness its cycle-0 scout
    tell defined at abc.py:125 (row 19). Oracle: ABC.py:152-153 np.amax / argmax, init() line 81. DU rows
    11, 12, 13, 19.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    tells_per_cycle = 2 * sn + 1
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * tells_per_cycle - 1))

    _tell_single_candidates(opt, [CONSTANT_SCORE])
    state = (list(opt.trial), opt.phase, opt.cycle)

    assert state == ([2, 0], "employed", 2)


def test_abc_should_scout_second_source_when_only_it_reaches_limit():
    """Given limit 1, init [1.0, 2.0] and single tells 1.0, 1.0, 0.5, 1.0 (trial [0, 1] at phase 'scout'),
    when one candidate is asked and told 1.0, then it is the scout of source 1: phase 'scout' after the
    ask, trial == [0, 0] and cycle 1 after the tell.

    Discriminating value: source 0 is below the limit while source 1 is at it, so the scout decision
    depends on a source other than source 0 (site: the scout check reads trial[worst] at abc.py:66). Hand
    trace with fitness(1.0) = 0.5, fitness(2.0) = 1/3, fitness(0.5) = 2/3: the employed tie fails source
    0 (trial [1, 0]); the employed 1.0 improves source 1 (0.5 > 1/3, trial [1, 0]), so both fitness
    values are 0.5 and prob is exactly [1.0, 1.0]; onlooker 0 improves source 0 with 0.5 (trial [0, 0]);
    onlooker 1 ties on source 1 (trial [0, 1]). Every onlooker draw is accepted, so the trace holds for
    every seed. R7 (scout check reads trial[0]) and M11 (worst = min) both see trial 0 < 1, skip the scout
    and report phase 'employed' after the ask. Oracle: ABC.py:152 'np.amax(trial) >= LIMIT', ABC.py:153
    argmax, init() line 81 (trial 0), ABC.py:113/121 and :140-141 (greedy), :149 (sweep). DU rows 8, 11.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=1, seed=ANY_SEED)
    _tell_initial_batch(opt, [TIE_SCORE, 2.0])
    _tell_single_candidates(opt, [TIE_SCORE, TIE_SCORE, BETTER_SCORE, TIE_SCORE])
    assert (opt.phase, list(opt.trial)) == ("scout", [0, 1])

    (cand,) = opt.ask()
    phase_after_ask = opt.phase
    opt.tell([cand], [CONSTANT_SCORE])
    observed = (phase_after_ask, list(opt.trial), opt.cycle)

    assert observed == ("scout", [0, 0], 1)


def test_abc_should_replace_scout_source_even_when_scout_score_is_worse():
    """Given limit 2 and trial [2, 2], when the scout is told 9.0 (worse than the source's 1.0) and the next
    candidate is asked, then the source was abandoned anyway: trial == [0, 2] and the next employed
    neighbour differs from the scout point in exactly 1 coordinate.

    A worse scout score discriminates any greedy-scout variant (scratch mutant M21, site: abc.py:91), which
    keeps the old source (trial [2, 2], neighbour two coordinates away from the scout point). The count is
    exactly 1 for every seed except a measure-zero set (phi == 0). Oracle: ABC.py send_scout_bees calls
    init(), which overwrites foods, f and fitness and sets trial 0 unconditionally (lines 70-81, 151-153);
    Karaboga & Basturk 2007 scout = abandon and replace. Kills M21, M05 and M10. DU rows: none executed
    to their use; the scout tell runs the definitions at abc.py:121-126 shared by rows 12 and 19, but the
    following candidate is only asked, never told, so abc.py:94 and abc.py:97 are not reached (their uses
    are executed by the unequal-trial test).
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    sn = opt.food_number
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, (CONSTANT_SCORE,) * (2 * sn))
    (scout,) = opt.ask()
    assert opt.phase == "scout"

    opt.tell([scout], [WORSE_SCOUT_SCORE])
    trial = list(opt.trial)
    (following,) = opt.ask()

    assert trial == [0, 2]
    assert int(np.count_nonzero(following.x != scout.x)) == 1


def test_abc_should_move_source_zero_relative_to_the_other_source_when_two_sources():
    """Given colony_size 4 (two sources, so the only partner of source 0 is source 1) and init [1.0, 1.0],
    when the first employed candidate is asked, then it differs from init[0].x in exactly 1 coordinate.

    Corner i = 0 with sn = 2 (site: partner index 'k += k >= i' at abc.py:105): M10 ('>') makes k == i,
    so phi * (x_ij - x_ij) = 0 and nothing changes (0 coordinates). The existing suite masked it with
    '<= 1'. Exactly 1 holds for every seed except a measure-zero set (phi == 0 or equal coordinates).
    Oracle: Karaboga & Basturk 2007 (k != i); ABC.py:92-94 (redraw while neighbour == change_index) and
    ABC.py:96-99 (one parameter changed).
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=NEVER_SCOUT_LIMIT, seed=ANY_SEED)
    initial = _tell_initial_batch(opt, [TIE_SCORE, TIE_SCORE])

    (cand,) = opt.ask()
    changed = int(np.count_nonzero(cand.x != initial[0].x))

    assert changed == 1


def test_abc_should_read_onlooker_reset_trial_at_scout_check():
    """Given limit 2, init [1.0, 1.0] and single tells 1.0, 1.0, 1.0, 0.5 (the second onlooker improves
    source 1, trial [2, 0]), when the scout is told 1.0, then source 0 was replaced and nothing is left
    exhausted: trial == [0, 0], phase 'employed', cycle 1.

    The trial reset by an onlooker improvement (site: abc.py:126) is read at the scout check abc.py:65-66.
    M11 gives [3, 0], M02 [3, 0], M04 [2, 1]; M16 already fails the precondition ([0, 1]). Oracle:
    ABC.py:140-141 (onlooker improvement resets trial), ABC.py:149 (sweep), ABC.py:152-153 and init()
    line 81. DU rows 7, 11.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])
    _tell_single_candidates(opt, [CONSTANT_SCORE, CONSTANT_SCORE, CONSTANT_SCORE, BETTER_SCORE])
    assert (opt.phase, list(opt.trial)) == ("scout", [2, 0])

    _tell_single_candidates(opt, [CONSTANT_SCORE])
    state = (list(opt.trial), opt.phase, opt.cycle)

    assert state == ([0, 0], "employed", 1)


def test_abc_should_skip_scout_when_every_tell_improves():
    """Given limit 1 (smallest valid) and strictly decreasing scores 0.9, 0.8, 0.7, 0.6 after init [1.0, 1.0],
    so trial [0, 0] at phase 'scout', when one candidate is asked, then no scout is issued: phase
    'employed', cycle 1.

    Each score beats every source, so the result does not depend on which onlooker is picked or on the
    seed. False branch of the scout check (site: abc.py:66): M19 ('limit - 1') scouts on 0 >= 0 and M12
    (cycle increment deleted) stays at cycle 0. Oracle: ABC.py:152 (0 >= 1 is false) and increase_cycle;
    ABCOptimizer accepts limit >= 1. DU row 7b.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=1, seed=ANY_SEED)
    _tell_initial_batch(opt, [1.0, 1.0])
    _tell_single_candidates(opt, [0.9, 0.8, 0.7, 0.6])
    assert (opt.phase, list(opt.trial)) == ("scout", [0, 0])

    opt.ask()
    state = (opt.phase, opt.cycle)

    assert state == ("employed", 1)


def test_space_move_should_give_same_key_and_params_as_rows_for_every_neighbour():
    """Given a Float/Int/Categorical space, colony_size 4, limit 2 and constant scores, when 40 single
    candidates (employed, onlooker and scout) are asked, then each candidate's (key, params) equals the
    full rows() decode of its x.

    A wrong key from the one-coordinate re-decode (site: Space.move key splice, space.py:222) leaves the
    function inputs correct and surfaces only through caching, so the key itself is compared. Scratch
    mutants S1 (slice off by one) and S2 (stale key) are killed here; the existing suite also kills both
    (S1 via the cache test, S2 via the convergence and NaN tests), so no new-kill credit. Oracle:
    Space.move docstring '(key, params) of a point differing ... only in coordinate j'; Space.rows
    docstring (a key is the hashable raw row); Candidate docstring (key = decoded identity used for
    caching).
    """
    space = Space({"a": Float(0.0, 1.0), "k": Int(0, 9), "c": Categorical(["x", "y", "z"])})
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=ANY_SEED)
    _tell_initial_batch(opt, [CONSTANT_SCORE, CONSTANT_SCORE])

    recorded, decoded = _record_key_params_and_rows_decode(opt, space, n_asks=40, score=CONSTANT_SCORE)

    assert recorded == decoded


def test_optimize_should_call_function_once_when_initial_batch_decodes_to_one_parameter_set():
    """Given a one-choice space and the default colony (5 sources, one initial batch of 5), when optimize
    runs with max_evals=5, then the function is called once and the 4 repeats inside the same batch are
    cached.

    Every decoded key is equal, so cache hits must happen within one batch, before any tell (site:
    facade.py:46 updating 'seen' inside the batch loop); scratch mutant F2 (cache filled after tell) gives
    5 calls, also killed by the existing cache test. Oracle: README 'With cache=True (the default), a
    repeated parameter set reuses its earlier score'; CHANGELOG 'a configuration is never trained twice';
    Study.n_calls docstring. DU row 34.
    """
    space = Space({"c": Categorical(["only"])})
    calls = []

    def spy_objective(params):
        calls.append(params)
        return 1.0

    study = optimize(spy_objective, space, max_evals=5, seed=ANY_SEED)

    assert len(calls) == 1
    assert study.n_calls == 1
    assert [t.cached for t in study.trials] == [False, True, True, True, True]
