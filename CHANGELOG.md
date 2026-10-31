# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-10-31

### Fixed

- `ABCOptimizer` accepts a neighbour when its raw score is strictly lower; comparing the fitness
  1 / (1 + f) rejected tiny improvements, such as 0 against 1e-18.
- `tell()` accepts only candidates the optimizer handed out, checks the whole batch (pending candidates,
  no repeats, scores that convert with `float()`) before applying it, and keeps its own copy of the
  handed-out params.
- `optimize()` withdraws the candidates that `max_evals` cuts off with the new `Optimizer.cancel()`, so a
  passed optimizer stays usable, and hands the objective a copy of the params.
- `Categorical` rejects duplicate choices, which made the cache evaluate the same value twice.
- The release workflow runs lint, type checks and tests before building.

## [0.2.0] - 2026-10-31

### Changed

- `max_evals` counts real calls of the objective: cached repeats no longer use up the budget, and a search
  over a space without `Float` stops once every parameter set has been evaluated.
- Optimizers avoid handing out a parameter set twice; `ABCOptimizer` takes `max_tries` (default 100).

## [0.1.0] - 2026-10-31

### Added

- `Space` with `Float`, `Int` and `Categorical` parameters, decoded from the unit cube in vectorised numpy.
- `ABCOptimizer`: Artificial Bee Colony behind an ask/tell interface, following the phase order of the
  original implementation.
- `optimize()` for one-call optimisation, returning a `Study` with every trial and the best one.
- Caching of repeated parameter sets, so a configuration is never trained twice.
- NaN and infinite scores are treated as the worst value for the chosen direction.
