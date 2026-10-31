# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Space` with `Float`, `Int` and `Categorical` parameters, decoded from the unit cube in vectorised numpy.
- `ABCOptimizer`: Artificial Bee Colony behind an ask/tell interface, following the phase order of the
  original implementation.
- `optimize()` for one-call optimisation, returning a `Study` with every trial and the best one.
- Caching of repeated parameter sets, so a configuration is never trained twice.
- NaN and infinite scores are treated as the worst value for the chosen direction.
