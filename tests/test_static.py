"""Static checks on the source and test files of meta-hpo: the code is analysed, never run.

Rules: P-T1-48 (use static tests next to dynamic ones; compiling is one of the static tools), P-T1-22
(production code and test code live in separate directories) and P-T3-39 (question every test with the
four flaky-test cards: day of the week, several threads, a time frame, the latest news items).

Every check is a pure helper that takes source text and a repo-relative path and returns its findings.
Each helper is run over the real files and also over in-memory snippets that must and must not be
reported, so the analysis itself is shown able to fail. File lists come from Path(__file__), are sorted
by their POSIX path and never depend on the working directory. Nothing is written to disk: builtin
compile() is used instead of py_compile.

Oracles: the rule texts above, CONTRIBUTING.md (no hash comment lines in Python files; pass a seed to
anything random), pyproject.toml (wheel packages = src/metahpo, pytest testpaths = tests, no
python_files setting, so pytest's defaults test_*.py and *_test.py apply), the signatures
optimize(function, space, direction, max_evals, optimizer, seed, cache) and
ABCOptimizer(space, direction, colony_size, limit, seed), Space.sample(rng, n), and metahpo.__all__.

Unspecified behaviour, decided here and not asserted as a library requirement:
- The source gives no answers to the question cards, so the module-to-card mapping in FLAKY_MODULES is
  the project's choice.
- Whether a trailing comment counts as a comment line is not stated; every comment token is reported.
- optimizer=None is treated like no optimizer, because optimize() then builds its own ABCOptimizer.
- A call that unpacks *args or **kwargs counts as seeded only when the seed is also passed explicitly.
- A star import from metahpo is accepted, because it binds only the names in metahpo.__all__.
- Direct Optimizer(space, direction, seed) constructions and dynamic imports are not checked.
- The numpy.random constructors listed in SEEDABLE_NUMPY_RANDOM count as seeded when their first
  argument is given and is not None; every other numpy.random function call is treated as use of the
  hidden global state and reported, except Generator, whose bit generator is checked as its own call.

Input space: The source and test files on disk, which are unbounded.
Subset: The real files as they are on disk, plus small synthetic snippets for each checker.
"""

import ast
import fnmatch
import io
import sys
import tokenize
from pathlib import Path, PurePosixPath

import pytest

import metahpo

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "metahpo"
TESTS_DIR = ROOT / "tests"
HASH = chr(35)


def _python_files(directory):
    """Return the sorted POSIX paths, relative to ROOT, of every .py file below directory."""
    paths = (p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(p.relative_to(ROOT).as_posix() for p in paths)


def _is_test_module_name(name):
    """Return True when pytest's default collection would treat the file name as a test module."""
    patterns = ("test_*.py", "*_test.py")
    return name == "conftest.py" or any(fnmatch.fnmatchcase(name, p) for p in patterns)


SRC_FILES = _python_files(PACKAGE_DIR)
TEST_FILES = _python_files(TESTS_DIR)
TEST_MODULES = [p for p in TEST_FILES if _is_test_module_name(PurePosixPath(p).name)]
PUBLIC_NAMES = frozenset(metahpo.__all__) | {"operators"}

ALLOWED_IMPORTS = {
    "metahpo": "exact",
    "metahpo.operators": "exact",
    "numpy": "package",
    "pytest": "package",
    "conftest": "exact",
}
"""Modules a test module may import, besides any top-level name in sys.stdlib_module_names.

Source: the hard constraints of the test suite (pytest, numpy and the standard library only, no new
dependencies such as hypothesis or pytest plugins) and the public-API-only rule (tests import metahpo,
and metahpo.operators only for the operator formulas). "exact" allows only that dotted name; "package"
allows the name and its submodules. "conftest" is the suite's own tests/conftest.py, which
test_suite_layout.py imports to call its hooks; it is part of the suite, not a dependency. Names imported
from metahpo must be in metahpo.__all__ or be
operators.
"""

FLAKY_MODULES = {
    "datetime": "day of the week",
    "calendar": "day of the week",
    "zoneinfo": "day of the week",
    "time": "day of the week; time frame",
    "timeit": "time frame",
    "sched": "time frame",
    "threading": "several threads",
    "_thread": "several threads",
    "concurrent": "several threads",
    "multiprocessing": "several threads",
    "asyncio": "several threads",
    "socket": "latest news items (changing external content)",
    "urllib": "latest news items (changing external content)",
    "http": "latest news items (changing external content)",
    "random": "hidden non-determinism (P-T3-37, P-T3-38)",
    "secrets": "hidden non-determinism (P-T3-37, P-T3-38)",
}
"""Top-level modules a test module must not import, each mapped to the question card it answers.

The four cards of P-T3-39 ask whether a test depends on the day of the week, starts several threads,
needs a task to finish within a time frame, or checks the latest news items. The source gives no
answers to the cards, so this mapping is the project's choice. The standard random and secrets modules
are listed as hidden non-determinism (secrets draws from operating system entropy); numpy.random stays
allowed because its top-level name is numpy, and its calls are checked by the seed checker instead.
"""

SEEDABLE_NUMPY_RANDOM = {
    "default_rng": "seed",
    "RandomState": "seed",
    "SeedSequence": "entropy",
    "PCG64": "seed",
    "PCG64DXSM": "seed",
    "MT19937": "seed",
    "Philox": "seed",
    "SFC64": "seed",
}
"""numpy.random constructors seeded by their first argument, mapped to that argument's keyword name.

Source: CONTRIBUTING.md ("pass a seed to anything random") and the numpy.random signatures. A call of
one of these names is unseeded when its first argument (positional 0 or the keyword) is missing or None.
Every other numpy.random function call (rand, standard_normal, beta, shuffle, seed, ...) uses the hidden
global state and is always reported. Generator(...) is not reported itself, because its bit generator
argument is checked as a call of its own.
"""

ASSERT_EXEMPTIONS: dict[str, str] = {}
"""Test functions allowed to hold no assertion, as {'tests/<file>::<function>': reason}. Empty."""

SEED_EXEMPTIONS = {
    "tests/test_abc.py::test_protocol_errors": (
        "ABCOptimizer(sphere_space(), colony_size=3) raises ValueError before any random number is drawn"
    ),
}
"""Test functions allowed to make an unseeded random call, as {'tests/<file>::<function>': reason}."""


def _scopes(tree):
    """Yield (qualname, node) for each top-level statement; Test class methods get 'Class::method'."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                name = getattr(item, "name", None)
                yield (f"{node.name}::{name}" if name else node.name), item
        else:
            yield "<module>", node


def _import_aliases(tree):
    """Map every name bound by an import to the dotted module or object it stands for."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                aliases[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _dotted(node, aliases):
    """Return the dotted name of a Name or Attribute chain with import aliases resolved, else None."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _compile_error_lines(data, relpath):
    """Compile source bytes without running them and return the line of the SyntaxError, if any."""
    try:
        compile(data, relpath, "exec", dont_inherit=True)
    except SyntaxError as error:
        return [error.lineno]
    return []


def _package_name_violations(relpath):
    """Return [relpath] when a file of the package is named like a pytest test module, else []."""
    return [relpath] if _is_test_module_name(PurePosixPath(relpath).name) else []


def _same_tree(module_file, expected_dir):
    """Return True when the imported module file lies directly in expected_dir."""
    return Path(module_file).resolve().parent == Path(expected_dir).resolve()


def _package_import_violations(source, relpath):
    """Report missing metahpo imports, relative imports, src or tests imports and sys.path use."""
    tree = ast.parse(source, relpath)
    aliases = _import_aliases(tree)
    found = []
    has_metahpo = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops = [alias.name.split(".")[0] for alias in node.names]
            has_metahpo = has_metahpo or "metahpo" in tops
            found += [(node.lineno, "src or tests import") for top in tops if top in ("src", "tests")]
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            found.append((node.lineno, "relative import"))
        elif isinstance(node, ast.ImportFrom):
            top = node.module.split(".")[0]
            has_metahpo = has_metahpo or top == "metahpo"
            if top in ("src", "tests"):
                found.append((node.lineno, "src or tests import"))
            if node.module == "sys" and any(alias.name == "path" for alias in node.names):
                found.append((node.lineno, "sys.path"))
        elif isinstance(node, ast.Attribute) and node.attr == "path":
            if isinstance(node.value, ast.Name) and aliases.get(node.value.id, node.value.id) == "sys":
                found.append((node.lineno, "sys.path"))
    if not has_metahpo:
        found.append((1, "no metahpo import"))
    return sorted(found)


def _module_allowed(name):
    """Return True when ALLOWED_IMPORTS or the standard library allows importing the dotted name."""
    top = name.split(".")[0]
    exact = ALLOWED_IMPORTS.get(name) == "exact"
    package = ALLOWED_IMPORTS.get(top) == "package"
    stdlib = top in sys.stdlib_module_names and top not in ALLOWED_IMPORTS
    return exact or package or stdlib


def _disallowed_imports(source, relpath):
    """Return (line, module) for every import outside ALLOWED_IMPORTS and the standard library."""
    found = []
    for node in ast.walk(ast.parse(source, relpath)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names if not _module_allowed(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if not _module_allowed(node.module):
                found.append((node.lineno, node.module))
            elif node.module == "metahpo":
                names = [a.name for a in node.names if a.name != "*" and a.name not in PUBLIC_NAMES]
                found += [(node.lineno, f"metahpo.{name}") for name in names]
    return sorted(found)


def _comment_lines(data, relpath):
    """Return the line of every comment token; a hash character inside a string is not a comment."""
    tokens = tokenize.tokenize(io.BytesIO(data).readline)
    return [tok.start[0] for tok in tokens if tok.type == tokenize.COMMENT]


def _is_assertion_call(func, aliases):
    """Return True for pytest.raises and numpy.testing.assert_* calls, with aliases resolved."""
    name = _dotted(func, aliases) or ""
    return name == "pytest.raises" or name.startswith("numpy.testing.assert_")


def _asserts_in_own_body(function, aliases):
    """Return True when the function body, without nested defs, lambdas or classes, asserts."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call) and _is_assertion_call(node.func, aliases):
            return True
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    return False


def _functions_without_assert(source, relpath):
    """Return the test functions (pytest defaults: test* functions, Test* classes) that never assert."""
    tree = ast.parse(source, relpath)
    aliases = _import_aliases(tree)
    found = []
    for qualname, node in _scopes(tree):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        in_test_class = "::" not in qualname or qualname.startswith("Test")
        if is_function and node.name.startswith("test") and in_test_class:
            if not _asserts_in_own_body(node, aliases):
                found.append(qualname)
    return found


def _flaky_imports(source, relpath):
    """Return (line, top-level module) for every import whose top-level name is in FLAKY_MODULES."""
    found = []
    for node in ast.walk(ast.parse(source, relpath)):
        if isinstance(node, ast.Import):
            tops = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            tops = [node.module.split(".")[0]]
        else:
            tops = []
        found += [(node.lineno, top) for top in tops if top in FLAKY_MODULES]
    return sorted(found)


def _argument(call, index, keyword):
    """Return the argument passed by keyword or at the positional index, or None when it is absent.

    A positional argument counts only when no *args unpacking comes before it, because the unpacked
    values would shift its position.
    """
    by_keyword = [kw.value for kw in call.keywords if kw.arg == keyword]
    if by_keyword:
        return by_keyword[0]
    before = call.args[: index + 1]
    if len(call.args) <= index or any(isinstance(arg, ast.Starred) for arg in before):
        return None
    return call.args[index]


def _given(node):
    """Return True when an argument is present and is not the constant None."""
    return node is not None and not (isinstance(node, ast.Constant) and node.value is None)


def _is_unseeded_default_rng(node, aliases):
    """Return True for numpy.random.default_rng() called without a seed or with None."""
    if not isinstance(node, ast.Call) or _dotted(node.func, aliases) != "numpy.random.default_rng":
        return False
    return not _given(_argument(node, 0, "seed"))


def _is_unseeded_call(call, aliases):
    """Apply the seeding rules for optimize, ABCOptimizer, sample and numpy.random to one call.

    A numpy.random constructor in SEEDABLE_NUMPY_RANDOM is reported when its first argument is missing
    or None; any other numpy.random call except Generator is reported. A call that unpacks *args or
    **kwargs is reported unless the seed (for optimize: the seed or the optimizer) is passed explicitly,
    because the unpacked values cannot be checked. An explicit seed cannot be replaced by them, since
    Python rejects a second value for the same argument.
    """
    name = _dotted(call.func, aliases) or ""
    short = name.rsplit(".", 1)[-1]
    if name.startswith("numpy.random.") and short in SEEDABLE_NUMPY_RANDOM:
        return not _given(_argument(call, 0, SEEDABLE_NUMPY_RANDOM[short]))
    if name.startswith("numpy.random.") and short != "Generator":
        return True
    if short == "optimize":
        return not (_given(_argument(call, 5, "seed")) or _given(_argument(call, 4, "optimizer")))
    if short == "ABCOptimizer":
        return not _given(_argument(call, 4, "seed"))
    if short == "sample" and isinstance(call.func, ast.Attribute):
        rng = _argument(call, 0, "rng")
        return not _given(rng) or _is_unseeded_default_rng(rng, aliases)
    return False


def _unseeded_calls(source, relpath):
    """Return (qualname, line, callee) for every random call that does not pass a seed."""
    tree = ast.parse(source, relpath)
    aliases = _import_aliases(tree)
    found = []
    for qualname, scope in _scopes(tree):
        calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
        found += [
            (qualname, c.lineno, _dotted(c.func, aliases) or "?")
            for c in calls
            if _is_unseeded_call(c, aliases)
        ]
    return sorted(found)


def _read(relpath):
    """Return the text of a repo file."""
    return (ROOT / relpath).read_text(encoding="utf-8")


def _not_exempted(relpath, qualnames, exemptions):
    """Return the qualnames whose 'relpath::qualname' key is not in the exemption mapping."""
    return [q for q in qualnames if f"{relpath}::{q}" not in exemptions]


def _unexempted_unseeded_calls(relpath):
    """Return 'file:line callee in function' for each unseeded call SEED_EXEMPTIONS does not cover."""
    calls = _unseeded_calls(_read(relpath), relpath)
    kept = [c for c in calls if f"{relpath}::{c[0]}" not in SEED_EXEMPTIONS]
    return [f"{relpath}:{line} {callee} in {qualname}" for qualname, line, callee in kept]


def _violating_functions(checker, relpath):
    """Return the qualnames the named checker reports for a repo file."""
    source = _read(relpath)
    if checker == "assert":
        return set(_functions_without_assert(source, relpath))
    return {qualname for qualname, _, _ in _unseeded_calls(source, relpath)}


def _defined_functions(relpath):
    """Return the qualnames of the functions defined in a repo file, or an empty set if it is missing."""
    path = ROOT / relpath
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_bytes(), relpath)
    return {q for q, node in _scopes(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _unsafe_ids(relpaths):
    """Return the ids that hold a backslash or a __pycache__ part."""
    return [p for p in relpaths if "\\" in p or "__pycache__" in p.split("/")]


FLAKY_EACH_MODULE_CASES = [pytest.param(f"import {m}\n", [(1, m)], id=f"import-{m}") for m in FLAKY_MODULES]
"""One violating snippet per key of FLAKY_MODULES, so every banned module is shown to be caught."""

EXEMPTION_CASES = [pytest.param(k, r, "assert", id=k) for k, r in ASSERT_EXEMPTIONS.items()] + [
    pytest.param(k, r, "seed", id=k) for k, r in SEED_EXEMPTIONS.items()
]


def test_file_lists_should_hold_files_from_every_level_of_both_trees():
    """Given the lists built from Path(__file__), when they are inspected, then both trees are walked
    recursively: the package root, the optimizers subpackage and this module are all present.

    Oracle: P-T1-48 (every file is statically checked) and P-T3-39 (lists do not depend on the cwd).
    optimizers/abc.py is the decisive member: a non-recursive glob would drop it.
    """
    expected_members = {"src/metahpo/__init__.py", "src/metahpo/optimizers/abc.py", "tests/test_static.py"}

    listed = set(SRC_FILES) | set(TEST_MODULES)
    missing = expected_members - listed

    assert missing == set()


def test_file_lists_should_be_sorted_posix_paths_without_cache_entries():
    """Given the file lists, when their order and spelling are inspected, then each list is sorted and
    holds only POSIX paths outside __pycache__, so the parametrized ids are the same on every run and OS.

    Oracle: P-T3-39 (deterministic tests).
    """
    all_ids = SRC_FILES + TEST_FILES

    unsafe = _unsafe_ids(all_ids)

    assert (SRC_FILES, TEST_FILES) == (sorted(SRC_FILES), sorted(TEST_FILES))
    assert unsafe == []


@pytest.mark.parametrize("relpath", SRC_FILES + TEST_FILES, ids=SRC_FILES + TEST_FILES)
def test_source_file_should_compile_without_being_run(relpath):
    """Given a Python file of the package or the tests, when builtin compile() reads it, then no
    SyntaxError is raised; the file is neither imported nor executed and nothing is written to disk.

    Oracle: P-T1-48 ("... COMPILING!" is one of the static test tools; the code is not run).
    """
    data = (ROOT / relpath).read_bytes()

    error_lines = _compile_error_lines(data, relpath)

    assert error_lines == [], f"{relpath} does not compile, SyntaxError on line(s) {error_lines}"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("def (\n", [1], id="plain-syntax-error"),
        pytest.param("x = 1\ndef (\n", [2], id="error-on-second-line"),
        pytest.param("return 1\n", [1], id="return-outside-function-parses-but-does-not-compile"),
        pytest.param("nonlocal x\n", [1], id="module-level-nonlocal-parses-but-does-not-compile"),
        pytest.param("await x\n", [1], id="module-level-await-parses-but-does-not-compile"),
        pytest.param("x = 1\n", [], id="assignment"),
        pytest.param("", [], id="empty-file"),
        pytest.param("def f():\n    return 1\n", [], id="return-inside-function"),
    ],
)
def test_compile_checker_should_report_the_error_line_only_when_source_does_not_compile(snippet, expected):
    """Given an in-memory snippet, when the compile checker runs on it, then it returns the SyntaxError
    line for broken code and [] for valid code.

    Oracle: Python language reference. The return, nonlocal and await cases pass ast.parse but fail
    compile(), so they show that real compilation is done, not just parsing.
    """
    data = snippet.encode()

    error_lines = _compile_error_lines(data, "snippet.py")

    assert error_lines == expected


@pytest.mark.parametrize("relpath", SRC_FILES, ids=SRC_FILES)
def test_package_should_hold_no_test_module(relpath):
    """Given a file of src/metahpo, when its name is checked, then it is neither conftest.py nor named
    like a pytest test module (test_*.py or *_test.py, pytest's defaults since pyproject.toml sets no
    python_files).

    Oracle: P-T1-22 (production code and test code in separate directories): production code is the
    wheel package src/metahpo, tests live in tests/ (pytest testpaths).
    """
    name_violations = _package_name_violations(relpath)

    assert name_violations == [], f"test module inside the package: {name_violations}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("test_x.py", ["src/metahpo/test_x.py"], id="test-prefix"),
        pytest.param("conftest.py", ["src/metahpo/conftest.py"], id="conftest"),
        pytest.param("test_.py", ["src/metahpo/test_.py"], id="test-prefix-with-empty-rest"),
        pytest.param("my_test.py", ["src/metahpo/my_test.py"], id="test-suffix"),
        pytest.param("_test.py", ["src/metahpo/_test.py"], id="test-suffix-with-empty-stem"),
        pytest.param(
            "optimizers/test_abc.py", ["src/metahpo/optimizers/test_abc.py"], id="nested-test-module"
        ),
        pytest.param("space.py", [], id="production-module"),
        pytest.param("test.py", [], id="no-underscore-near-miss"),
        pytest.param("testing.py", [], id="testing-near-miss"),
        pytest.param("contest.py", [], id="conftest-near-miss"),
        pytest.param("attest.py", [], id="suffix-near-miss"),
        pytest.param("test_x.txt", [], id="not-a-python-file"),
    ],
)
def test_package_name_checker_should_report_only_names_pytest_collects(name, expected):
    """Given a file name inside src/metahpo, when the package name checker runs, then it reports exactly
    the names pytest's default python_files patterns or conftest.py would match.

    Oracle: pytest's default python_files ['test_*.py', '*_test.py'] with case-sensitive glob matching,
    and the P-T1-22 requirement (rejects test_x.py and conftest.py, accepts space.py).
    """
    relpath = f"src/metahpo/{name}"

    name_violations = _package_name_violations(relpath)

    assert name_violations == expected


def test_static_checks_should_analyse_the_package_that_the_tests_import():
    """Given the repo root resolved from this file, when the imported metahpo is located, then
    src/metahpo and tests/ exist and metahpo is imported from that same src/metahpo.

    Oracle: P-T1-22. The test fails when metahpo comes from another copy, such as a non-editable
    site-packages install, because the static checks would then analyse a tree the tests do not import.
    """
    imported_here = _same_tree(metahpo.__file__, PACKAGE_DIR)

    assert (PACKAGE_DIR.is_dir(), TESTS_DIR.is_dir()) == (True, True)
    assert imported_here is True


@pytest.mark.parametrize(
    ("module_file", "expected"),
    [
        pytest.param(PACKAGE_DIR / "__init__.py", True, id="same-tree"),
        pytest.param(ROOT.parent / "other-copy" / "metahpo" / "__init__.py", False, id="other-copy"),
        pytest.param(PACKAGE_DIR / "optimizers" / "__init__.py", False, id="subpackage-is-not-the-root"),
    ],
)
def test_same_tree_checker_should_accept_only_the_package_directory(module_file, expected):
    """Given a module path, when the location checker compares it with src/metahpo, then only a file
    directly in src/metahpo matches; paths are compared only, nothing is created.

    Oracle: P-T1-22 requirement (the location check must be able to fail for another copy).
    """
    same = _same_tree(str(module_file), PACKAGE_DIR)

    assert same is expected


@pytest.mark.parametrize("relpath", TEST_MODULES, ids=TEST_MODULES)
def test_test_module_should_import_metahpo_by_its_package_name(relpath):
    """Given a test module, when its imports are analysed, then it imports metahpo, uses no relative
    import, never imports src or tests and never touches sys.path.

    Oracle: P-T1-22. Tests use the package as installed from src/metahpo, not path hacks into the tree.
    """
    source = _read(relpath)

    violations = _package_import_violations(source, relpath)

    assert violations == [], f"{relpath}: {violations}"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("import metahpo\nfrom . import x\n", [(2, "relative import")], id="relative-no-module"),
        pytest.param(
            "import metahpo\nfrom ..helpers import y\n", [(2, "relative import")], id="relative-lv2"
        ),
        pytest.param(
            "from src.metahpo import Space\n",
            [(1, "no metahpo import"), (1, "src or tests import")],
            id="src-layout-import",
        ),
        pytest.param(
            "import metahpo\nimport tests.test_abc\n", [(2, "src or tests import")], id="tests-import"
        ),
        pytest.param(
            'import metahpo\nimport sys\nsys.path.insert(0, "src")\n', [(3, "sys.path")], id="sys-path-insert"
        ),
        pytest.param(
            'import metahpo\nimport sys as s\ns.path.append("src")\n',
            [(3, "sys.path")],
            id="aliased-sys-path",
        ),
        pytest.param("import metahpo\nfrom sys import path\n", [(2, "sys.path")], id="from-sys-import-path"),
        pytest.param("import numpy\n", [(1, "no metahpo import")], id="no-metahpo-import"),
        pytest.param("import metahpo\n", [], id="plain-import"),
        pytest.param("import metahpo as mh\n", [], id="aliased-import"),
        pytest.param("from metahpo.operators import fitness\n", [], id="from-operators"),
        pytest.param(
            "import metahpo\nimport sys\nx = sys.stdlib_module_names\n", [], id="sys-without-path-near-miss"
        ),
    ],
)
def test_package_import_checker_should_report_each_path_hack_with_its_line(snippet, expected):
    """Given an in-memory snippet, when the package import checker runs, then each missing metahpo
    import, relative import, src or tests import and sys.path use is reported with its line, and
    compliant imports give [].

    Oracle: P-T1-22 requirement self-test list.
    """
    source = snippet

    violations = _package_import_violations(source, "tests/test_snippet.py")

    assert violations == expected


@pytest.mark.parametrize("relpath", TEST_MODULES, ids=TEST_MODULES)
def test_test_module_should_import_only_allowed_modules(relpath):
    """Given a test module, when its imports are analysed, then every module is in ALLOWED_IMPORTS or
    the standard library, and every name taken from metahpo is public.

    Oracle: the hard constraints (pytest, numpy and the standard library only) and metahpo.__all__.
    """
    source = _read(relpath)

    disallowed = _disallowed_imports(source, relpath)

    assert disallowed == [], f"{relpath}: {disallowed}"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("import scipy\n", [(1, "scipy")], id="scipy"),
        pytest.param("import hypothesis\n", [(1, "hypothesis")], id="hypothesis"),
        pytest.param("import metahpo.space\n", [(1, "metahpo.space")], id="private-submodule"),
        pytest.param(
            "from metahpo.optimizers.abc import ABCOptimizer\n",
            [(1, "metahpo.optimizers.abc")],
            id="deep-private-submodule",
        ),
        pytest.param("from metahpo import space\n", [(1, "metahpo.space")], id="private-submodule-by-name"),
        pytest.param("import metahpo.operators.x\n", [(1, "metahpo.operators.x")], id="below-operators"),
        pytest.param("import pytest_mock\n", [(1, "pytest_mock")], id="pytest-plugin-prefix"),
        pytest.param("import _pytest\n", [(1, "_pytest")], id="pytest-internals"),
        pytest.param("import metahpo_extra\n", [(1, "metahpo_extra")], id="metahpo-prefix"),
        pytest.param("import numpy_financial\n", [(1, "numpy_financial")], id="numpy-prefix"),
        pytest.param("from test_abc import sphere_space\n", [(1, "test_abc")], id="sibling-test-module"),
        pytest.param("import numpy.random as npr\n", [], id="numpy-submodule"),
        pytest.param("from numpy.testing import assert_allclose\n", [], id="numpy-testing"),
        pytest.param("from unittest.mock import patch\n", [], id="stdlib-mock"),
        pytest.param("from __future__ import annotations\n", [], id="future"),
        pytest.param("import math\n", [], id="stdlib"),
        pytest.param("import metahpo\n", [], id="metahpo"),
        pytest.param("from metahpo import Space, optimize\n", [], id="public-names"),
        pytest.param("from metahpo import operators\n", [], id="operators-by-name"),
        pytest.param("import metahpo.operators as ops\n", [], id="operators-module"),
    ],
)
def test_allowed_imports_checker_should_report_each_disallowed_module(snippet, expected):
    """Given an in-memory snippet, when the allowed imports checker runs, then each module or metahpo
    name outside the allowed set is reported with its line, and allowed imports give [].

    Oracle: P-T1-22 requirement self-test list and ALLOWED_IMPORTS. The prefix cases are decisive:
    the top-level name must equal pytest, numpy or metahpo, not merely start with it.
    """
    source = snippet

    disallowed = _disallowed_imports(source, "tests/test_snippet.py")

    assert disallowed == expected


@pytest.mark.parametrize("relpath", SRC_FILES + TEST_FILES, ids=SRC_FILES + TEST_FILES)
def test_python_file_should_hold_no_comment_token(relpath):
    """Given a Python file of the package or the tests, when it is tokenized, then it holds no comment
    token; explanations belong in docstrings.

    Oracle: CONTRIBUTING.md Style ("No hash comment lines in Python, YAML or TOML files") and P-T1-48
    (linting is a static test tool).
    """
    data = (ROOT / relpath).read_bytes()

    lines = _comment_lines(data, relpath)

    assert lines == [], f"{relpath} has comments on line(s) {lines}"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("x = 1  " + HASH + " trailing\n", [1], id="trailing-comment"),
        pytest.param(HASH + " full line\nx = 1\n", [1], id="full-line-comment"),
        pytest.param("x = 1\n    " + HASH + " indented\ny = 2\n", [2], id="indented-comment"),
        pytest.param("x = 1\n" + HASH + " a\n" + HASH + " b\n", [2, 3], id="every-comment-line-listed"),
        pytest.param("x = 'a' + '" + HASH + "' + 'b'\n", [], id="hash-in-string"),
        pytest.param('x = f"a' + HASH + 'b"\n', [], id="hash-in-f-string"),
        pytest.param('"""doc ' + HASH + ' text"""\n', [], id="hash-in-docstring"),
    ],
)
def test_comment_checker_should_report_comment_lines_but_not_hashes_in_strings(snippet, expected):
    """Given an in-memory snippet built with chr(35), when the comment checker runs, then it lists the
    line of each comment token and ignores the hash character inside strings.

    Oracle: Python tokenize semantics (COMMENT tokens; f-strings tokenize as strings on every version).
    """
    data = snippet.encode()

    lines = _comment_lines(data, "snippet.py")

    assert lines == expected


@pytest.mark.parametrize("relpath", TEST_MODULES, ids=TEST_MODULES)
def test_test_function_should_hold_an_assertion_in_its_own_body(relpath):
    """Given a test module, when its test functions are analysed, then each holds an assert statement,
    a pytest.raises call or a numpy.testing.assert_* call in its own body, unless ASSERT_EXEMPTIONS
    names it; asserts only inside nested helpers do not count.

    Oracle: P-T1-48 (statically analyse the test code itself). numpy.testing.assert_* calls count
    because the suite's style allows them as matchers.
    """
    names = _functions_without_assert(_read(relpath), relpath)

    unexempted = _not_exempted(relpath, names, ASSERT_EXEMPTIONS)

    assert unexempted == [], f"{relpath}: test functions without an assertion: {unexempted}"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("def test_a():\n    x = 1\n", ["test_a"], id="no-assertion"),
        pytest.param(
            "def test_b():\n    def h():\n        assert True\n    h()\n",
            ["test_b"],
            id="assert-in-nested-helper",
        ),
        pytest.param(
            "import pytest\ndef test_l():\n    f = lambda: pytest.raises(ValueError)\n    f()\n",
            ["test_l"],
            id="raises-in-lambda",
        ),
        pytest.param(
            "class TestK:\n    def test_m(self):\n        pass\n", ["TestK::test_m"], id="test-method"
        ),
        pytest.param("async def test_n():\n    pass\n", ["test_n"], id="async-test"),
        pytest.param("import pytest\ndef test_r():\n    pytest.approx(1)\n", ["test_r"], id="approx-alone"),
        pytest.param("def testx():\n    pass\n", ["testx"], id="pytest-default-test-prefix"),
        pytest.param('def test_s():\n    raises(ValueError, int, "x")\n', ["test_s"], id="unresolved-raises"),
        pytest.param(
            "import pytest\ndef test_c():\n    with pytest.raises(ValueError):\n        f()\n",
            [],
            id="with-raises",
        ),
        pytest.param("def test_d():\n    assert 1\n", [], id="assert"),
        pytest.param(
            'import pytest\ndef test_e():\n    pytest.raises(ValueError, int, "x")\n',
            [],
            id="plain-raises-call",
        ),
        pytest.param(
            "import pytest\ndef test_g():\n    with pytest.raises(ValueError) as e:\n        f()\n",
            [],
            id="raises-as",
        ),
        pytest.param(
            'from pytest import raises\ndef test_h():\n    raises(ValueError, int, "x")\n',
            [],
            id="imported-raises",
        ),
        pytest.param(
            "import numpy as np\ndef test_i():\n    np.testing.assert_array_equal(a, b)\n",
            [],
            id="numpy-testing-assert",
        ),
        pytest.param("def helper():\n    pass\n", [], id="not-a-test"),
        pytest.param("class Helper:\n    def test_z(self):\n        pass\n", [], id="not-a-test-class"),
        pytest.param("class TestK:\n    def test_m(self):\n        assert True\n", [], id="asserting-method"),
    ],
)
def test_assert_checker_should_report_test_functions_without_an_own_assertion(snippet, expected):
    """Given an in-memory snippet, when the assert checker runs, then it names each test function whose
    own body has no assertion and accepts the others.

    Oracle: pytest's default collection (test* functions, Test* classes) and the P-T1-48 requirement
    (an assert only inside a nested helper does not count).
    """
    source = snippet

    names = _functions_without_assert(source, "tests/test_snippet.py")

    assert names == expected


@pytest.mark.parametrize("relpath", TEST_MODULES, ids=TEST_MODULES)
def test_test_module_should_not_import_a_flaky_stdlib_module(relpath):
    """Given a test module, when its imports are analysed, then none has a top-level name listed in
    FLAKY_MODULES.

    Oracle: P-T3-39 question cards (day of the week, several threads, time frame, latest news items).
    The source gives no answers to the cards, so the module mapping is the project's choice.
    """
    source = _read(relpath)

    flaky = _flaky_imports(source, relpath)

    assert flaky == [], f"{relpath}: (line, module) pairs {flaky}, cards in FLAKY_MODULES"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    FLAKY_EACH_MODULE_CASES
    + [
        pytest.param("from time import perf_counter\n", [(1, "time")], id="from-time"),
        pytest.param("from datetime import date\n", [(1, "datetime")], id="from-datetime"),
        pytest.param("import datetime as dt\n", [(1, "datetime")], id="aliased-datetime"),
        pytest.param(
            "from concurrent.futures import ThreadPoolExecutor\n", [(1, "concurrent")], id="from-concurrent"
        ),
        pytest.param("import concurrent.futures\n", [(1, "concurrent")], id="dotted-concurrent"),
        pytest.param("import os, time\n", [(1, "time")], id="second-alias"),
        pytest.param("import numpy as np\nrng = np.random.default_rng(0)\n", [], id="numpy-generator"),
        pytest.param("from numpy import random\n", [], id="numpy-random-by-name"),
        pytest.param("import numpy.random\n", [], id="numpy-random-module"),
        pytest.param("import math\n", [], id="math"),
        pytest.param("import timezone_helper\n", [], id="time-prefix-near-miss"),
    ],
)
def test_flaky_import_checker_should_match_banned_top_level_names_only(snippet, expected):
    """Given an in-memory snippet, when the flaky import checker runs, then each import whose top-level
    name is in FLAKY_MODULES is reported with its line, and other imports give [].

    Oracle: P-T3-39 requirement self-test list. Matching on the top-level name keeps numpy.random
    allowed and still catches concurrent.futures.
    """
    source = snippet

    flaky = _flaky_imports(source, "tests/test_snippet.py")

    assert flaky == expected


@pytest.mark.parametrize("relpath", TEST_MODULES, ids=TEST_MODULES)
def test_random_call_should_pass_a_seed(relpath):
    """Given a test module, when its calls are analysed, then every optimize, ABCOptimizer, sample and
    numpy.random call passes a seed, unless SEED_EXEMPTIONS names the enclosing test function.

    Oracle: CONTRIBUTING.md ("pass a seed to anything random"), P-T3-39, and the signatures of
    optimize, ABCOptimizer and Space.sample. The checker cannot prove statically that a value is a fixed
    integer or a numpy Generator; it rejects missing seeds and None, numpy.random constructors from
    SEEDABLE_NUMPY_RANDOM without a seed, and every numpy.random call that uses the global state.
    """
    source_path = ROOT / relpath

    reported = _unexempted_unseeded_calls(relpath)

    assert (source_path.is_file(), reported) == (True, [])


PRELUDE = "import numpy as np\nfrom metahpo import ABCOptimizer, optimize\n"
OPT = "metahpo.optimize"
ABC = "metahpo.ABCOptimizer"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param(
            PRELUDE + "optimize(f, s, max_evals=5)\n", [("<module>", 3, OPT)], id="optimize-no-seed"
        ),
        pytest.param(
            PRELUDE + "optimize(f, s, seed=None)\n", [("<module>", 3, OPT)], id="optimize-seed-none"
        ),
        pytest.param(
            PRELUDE + "optimize(f, s, optimizer=None)\n", [("<module>", 3, OPT)], id="optimizer-none"
        ),
        pytest.param(
            PRELUDE + 'optimize(f, s, "min", 5, None, None)\n', [("<module>", 3, OPT)], id="positional-nones"
        ),
        pytest.param(PRELUDE + "optimize(f, s, **kw)\n", [("<module>", 3, OPT)], id="optimize-kwargs"),
        pytest.param(PRELUDE + "ABCOptimizer(s)\n", [("<module>", 3, ABC)], id="abc-no-seed"),
        pytest.param(PRELUDE + "ABCOptimizer(s, seed=None)\n", [("<module>", 3, ABC)], id="abc-seed-none"),
        pytest.param(
            PRELUDE + 'ABCOptimizer(s, "min", 10, 15)\n', [("<module>", 3, ABC)], id="abc-4-positionals"
        ),
        pytest.param(PRELUDE + "ABCOptimizer(*a)\n", [("<module>", 3, ABC)], id="abc-star-args"),
        pytest.param(
            PRELUDE + 'ABCOptimizer(*a, "min", 10, 15, 0)\n',
            [("<module>", 3, ABC)],
            id="positional-seed-after-star-args",
        ),
        pytest.param(
            PRELUDE + "np.random.default_rng(**kw)\n",
            [("<module>", 3, "numpy.random.default_rng")],
            id="rng-kwargs",
        ),
        pytest.param(PRELUDE + "s.sample(None, 3)\n", [("<module>", 3, "s.sample")], id="sample-none"),
        pytest.param(
            PRELUDE + "s.sample(rng=None, n=3)\n", [("<module>", 3, "s.sample")], id="sample-rng-none"
        ),
        pytest.param(
            PRELUDE + "s.sample(np.random.default_rng(), 3)\n",
            [("<module>", 3, "numpy.random.default_rng"), ("<module>", 3, "s.sample")],
            id="sample-unseeded-generator",
        ),
        pytest.param(
            PRELUDE + "np.random.default_rng()\n",
            [("<module>", 3, "numpy.random.default_rng")],
            id="rng-no-seed",
        ),
        pytest.param(
            PRELUDE + "np.random.default_rng(None)\n",
            [("<module>", 3, "numpy.random.default_rng")],
            id="rng-none",
        ),
        pytest.param(
            PRELUDE + "np.random.default_rng(seed=None)\n",
            [("<module>", 3, "numpy.random.default_rng")],
            id="rng-seed-none",
        ),
        pytest.param(
            PRELUDE + "np.random.rand(3)\n", [("<module>", 3, "numpy.random.rand")], id="legacy-rand"
        ),
        pytest.param(
            PRELUDE + "np.random.seed(0)\n", [("<module>", 3, "numpy.random.seed")], id="global-seed"
        ),
        pytest.param(
            "import numpy\nnumpy.random.shuffle(a)\n",
            [("<module>", 2, "numpy.random.shuffle")],
            id="legacy-shuffle",
        ),
        pytest.param(
            "from numpy.random import default_rng\ndefault_rng()\n",
            [("<module>", 2, "numpy.random.default_rng")],
            id="imported-default-rng",
        ),
        pytest.param(
            "from numpy import random\nrandom.randint(3)\n",
            [("<module>", 2, "numpy.random.randint")],
            id="numpy-random-by-name",
        ),
        pytest.param(
            "import metahpo as mh\nmh.optimize(f, s)\n", [("<module>", 2, OPT)], id="attribute-optimize"
        ),
        pytest.param(
            PRELUDE + "def test_x():\n    ABCOptimizer(s)\n", [("test_x", 4, ABC)], id="inside-test"
        ),
        pytest.param(
            PRELUDE + "class TestK:\n    def test_m(self):\n        optimize(f, s)\n",
            [("TestK::test_m", 5, OPT)],
            id="inside-test-method",
        ),
        pytest.param(
            PRELUDE + "np.random.standard_normal(3)\n",
            [("<module>", 3, "numpy.random.standard_normal")],
            id="global-state-standard-normal",
        ),
        pytest.param(
            PRELUDE + "np.random.RandomState()\n",
            [("<module>", 3, "numpy.random.RandomState")],
            id="unseeded-random-state",
        ),
        pytest.param(
            PRELUDE + "np.random.Generator(np.random.PCG64())\n",
            [("<module>", 3, "numpy.random.PCG64")],
            id="unseeded-bit-generator",
        ),
        pytest.param(
            PRELUDE + "np.random.SeedSequence()\n",
            [("<module>", 3, "numpy.random.SeedSequence")],
            id="unseeded-seed-sequence",
        ),
        pytest.param(PRELUDE + "np.random.RandomState(0)\n", [], id="seeded-random-state"),
        pytest.param(PRELUDE + "np.random.Generator(np.random.PCG64(0))\n", [], id="seeded-bit-generator"),
        pytest.param(PRELUDE + "np.random.SeedSequence(entropy=0)\n", [], id="seeded-seed-sequence"),
        pytest.param(PRELUDE + "optimize(f, s, seed=7)\n", [], id="optimize-seed"),
        pytest.param(PRELUDE + 'optimize(f, s, "min", 5, None, 7)\n', [], id="optimize-6th-positional-seed"),
        pytest.param(PRELUDE + "optimize(f, s, optimizer=opt)\n", [], id="optimize-with-optimizer"),
        pytest.param(PRELUDE + 'optimize(f, s, "min", 5, opt)\n', [], id="optimize-5th-positional-optimizer"),
        pytest.param(PRELUDE + 'ABCOptimizer(s, "min", 10, 15, 0)\n', [], id="abc-5th-positional-seed"),
        pytest.param(PRELUDE + "ABCOptimizer(s, seed=1)\n", [], id="abc-seed"),
        pytest.param(PRELUDE + "ABCOptimizer(s, seed=SEED)\n", [], id="abc-seed-constant"),
        pytest.param(PRELUDE + "ABCOptimizer(s, seed=1, **kw)\n", [], id="abc-explicit-seed-with-kwargs"),
        pytest.param(PRELUDE + "optimize(f, s, seed=1, **kw)\n", [], id="optimize-explicit-seed-with-kwargs"),
        pytest.param(PRELUDE + "s.sample(np.random.default_rng(0), 3)\n", [], id="sample-seeded-generator"),
        pytest.param(PRELUDE + "s.sample(opt.rng, 3)\n", [], id="sample-existing-generator"),
        pytest.param(
            PRELUDE + "rng = np.random.default_rng(0)\nrng.random(3)\nrng.permutation(4)\n",
            [],
            id="generator-methods-are-not-legacy-calls",
        ),
    ],
)
def test_seed_checker_should_report_each_unseeded_random_call(snippet, expected):
    """Given an in-memory snippet, when the seed checker runs, then each unseeded call is reported with
    its enclosing test function, line and resolved callee, and seeded calls give [].

    Oracle: P-T3-39 requirement self-test list and CONTRIBUTING.md ("pass a seed to anything random");
    positional seeds follow the signatures optimize(function, space, direction, max_evals, optimizer,
    seed, cache) and ABCOptimizer(space, direction, colony_size, limit, seed); numpy.random constructors
    take their seed as the first argument (SEEDABLE_NUMPY_RANDOM).
    """
    source = snippet

    calls = _unseeded_calls(source, "tests/test_snippet.py")

    assert calls == expected


@pytest.mark.parametrize(("key", "reason", "checker"), EXEMPTION_CASES)
def test_exemption_should_name_an_existing_test_function_and_a_reason(key, reason, checker):
    """Given an exemption entry, when its key is resolved, then the file and the function exist and the
    reason is not blank.

    Oracle: P-T1-48 requirement (exemptions stay justified and point at real tests).
    """
    relpath, qualname = key.split("::", 1)

    defined = _defined_functions(relpath)

    assert qualname in defined, f"exemption {key} names no existing function, remove the entry"
    assert reason.strip() != ""


@pytest.mark.parametrize(("key", "reason", "checker"), EXEMPTION_CASES)
def test_exemption_should_still_be_needed_by_its_checker(key, reason, checker):
    """Given an exemption entry, when its checker runs on the named file, then the named function still
    breaks that check, so a stale entry is caught.

    Oracle: P-T1-48 requirement (stale exemptions are removed).
    """
    relpath, qualname = key.split("::", 1)

    violating = _violating_functions(checker, relpath)

    assert qualname in violating, f"stale exemption, remove the entry {key}"


def test_static_module_should_hold_no_hash_character_at_all():
    """Given this module's own bytes, when they are searched for the hash character, then none is found,
    so no comment line can hide here and snippets build the character with chr(35).

    Oracle: the hard constraint of the suite (no hash comment lines) and CONTRIBUTING.md Style.
    """
    data = Path(__file__).read_bytes()

    found = HASH.encode() in data

    assert found is False
