# Contributing to meta-hpo

Thank you for helping. This guide covers setting up, checking your work and opening a pull request.

## Setup

```console
git clone https://github.com/mr-ozdemir/meta-hpo.git
cd meta-hpo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Checks

Every commit must pass these locally; CI runs the same commands:

```console
ruff check .
ruff format --check .
mypy
pytest
```

## Style

- Formatting and linting are done by Ruff; types are checked by mypy in strict mode.
- No `#` comment lines in Python, YAML or TOML files. Explanations belong in docstrings or in the text.
- Line endings are LF; `.gitattributes` takes care of it.
- Inner loops work on numpy arrays; single values use plain Python, because a numpy call on one number
  costs more than the arithmetic itself. The exception is a single-value path that must agree bit for bit
  with its array path, such as the log decode of `Space.move`.

## Tests

- Every change in behaviour comes with a test.
- Tests are deterministic: pass a `seed` to anything random.
- A new optimizer needs at least a convergence test on a benchmark function, a same-seed-same-result test
  and a test of its ask/tell protocol.
- Each test file must also pass on its own, and the suite must pass in reverse order:
  `METAHPO_TEST_REVERSE=1 pytest`.
- Coverage is measured with `coverage run -m pytest` and `coverage report -m --branch`; every statement and
  branch of `src/metahpo` outside abstract methods is covered.
- `docs/test-izlenebilirlik.md` maps each test-writing rule to the tests that satisfy it; update it when a
  test is added for a new rule.

## Branches and pull requests

- `master` holds released code and `develop` collects finished work.
  Neither receives direct pushes, and force pushes are never used.
- Every task starts with an issue and has one owner: one issue, one branch, one pull request.
- The branch starts from `develop` and is named after the kind of work: `feat/<name>`, `fix/<name>` or
  `docs/<name>`. It is deleted when its pull request is merged.
- The owner opens the pull request against `develop`, describes what changed and ends the description
  with `Part of #N`, the number of the issue.
- A pull request is merged when every CI check is green and the other maintainer has approved it.
  The owner merges it, with a merge commit only; squash and rebase merges are turned off.
- Changes asked for in a review are new commits on the same branch; commits are never amended.
- CI runs on every pull request and on pushes to `develop`. A pull request from `develop` to `master` skips
  the CI steps, because the same commit was already tested on `develop`.

## Commits

- Short, English, imperative subject line: "Add log scale to Int", "Speed up neighbour moves".
- No `Co-authored-by` lines and no tool names in the message.

## Changelog

User-facing changes get a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), in the section
that fits: Added, Changed, Deprecated, Removed, Fixed or Security.

## Releases

- Releases are made from `master` by the repository owner.
- A `release/x.y.z` branch sets the version in `pyproject.toml` and `metahpo.__version__` and gives
  `## [Unreleased]` in the changelog a dated version heading; it reaches `develop` through a pull request
  like any other task.
- The pull request from `develop` to `master` lists the pull requests it covers and closes their issues
  with `Closes #N`.
- Pushing a tag `vX.Y.Z` on `master` runs the release workflow: it checks that the tag matches the package
  version, builds the sdist and wheel, and publishes them to PyPI with trusted publishing, without tokens.

## Reporting bugs

Open an issue with a minimal script that reproduces the problem, the expected result, and your Python,
numpy and meta-hpo versions.
