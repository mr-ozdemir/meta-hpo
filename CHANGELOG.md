# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-11-09

### Added

- `optimize()` prints one line per generation and writes every run to a time-stamped folder below `output`
  (default `hpo_results`): all trials, the best so far, generation and population statistics, a parameter
  analysis, `report.md` and plots.
- A Ctrl+C returns the study so far and an error raised by the objective is re-raised after the results are
  written; `Study.stop_reason` says why the search ended.
- `Callback` objects passed with `callbacks` receive the run, generation and trial events, and every trial
  records its phase, generation, start time and duration.

### Changed

- matplotlib is a dependency; it draws the plots of the results folder.
- The optimizer settings, algorithm notes and references moved from the README to `docs/optimizers.md`.

## [0.4.0] - 2026-11-06

### Added

- `BBPSOOptimizer`: enhanced bare-bones PSO of Zhang, Lim & Liu (2023) with five search operations, a
  second remote leader and a neighbouring elite guide per particle.
- `BBPSOOptimizer` refines the leader and four other particles after every iteration with a root-finding
  step that Q-learning picks among inverse quadratic interpolation, secant, Steffensen and ITP.
- `BBPSOOptimizer` takes `target`, the best score the objective can return, which the root-finding steps aim at.

## [0.3.1] - 2026-11-05

### Fixed

- `ABCOptimizer.cancel()` keeps a parameter set reserved when another candidate with the same key
  was already told, so an evaluated set is not handed out again as new.
- The PyPI badge in the README refreshes within an hour instead of showing an old version.

## [0.3.0] - 2026-11-04

### Added

- `PSOOptimizer`: Particle Swarm Optimization with the inertia-weight update, the Clerc & Kennedy
  constants as defaults, a global-best topology and velocity clamping in unit-cube widths.
- `PSOOptimizer` hands out the whole swarm as one batch per iteration, so its candidates can be evaluated in
  parallel.
- A `PSOOptimizer` coordinate that leaves the unit cube stops on the wall with zero velocity.
- `operators.constriction()` computes the Clerc & Kennedy constriction coefficient for any phi above 4.

### Fixed

- README links point to the GitHub repository, so they also work on the PyPI page.

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
