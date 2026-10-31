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
  costs more than the arithmetic itself.

## Tests

- Every change in behaviour comes with a test.
- Tests are deterministic: pass a `seed` to anything random.
- A new optimizer needs at least a convergence test on a benchmark function, a same-seed-same-result test
  and a test of its ask/tell protocol.

## Branches and pull requests

- `master` holds released code and `develop` collects finished work.
  Neither receives direct pushes, and force pushes are never used.
- Work happens on a personal branch, kept up to date with `git merge --ff-only origin/develop`.
- Open the pull request against `develop` and ask the other maintainer for review.
  Describe what changed, which tests were added, and the related decisions.
- Pull requests are merged with a merge commit only, by a maintainer other than the author.
- One topic per pull request.

## Commits

- Short, English, imperative subject line: "Add log scale to Int", "Speed up neighbour moves".
- No `Co-authored-by` lines and no tool names in the message.

## Changelog

User-facing changes get a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), in the section
that fits: Added, Changed, Deprecated, Removed, Fixed or Security.

## Reporting bugs

Open an issue with a minimal script that reproduces the problem, the expected result, and your Python,
numpy and meta-hpo versions.
