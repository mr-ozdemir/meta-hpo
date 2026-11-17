# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- The results folder is resolved to an absolute path when the search starts (an `output` argument,
  `METAHPO_OUTPUT`, the default folder and `resume(folder)` alike), so an objective that changes the
  working directory no longer breaks the run with `FileNotFoundError`; config.json and summary.json record
  the absolute path.
- On Windows a monitor or editor that holds a JSON file of the results folder open no longer kills the
  search: the replace is retried for about a second, `best.json` is skipped while the run goes on and
  written at the next update and the end, and only `config.json`, `state.json` and the end-of-run files
  raise if they stay held, after every other result was written; no temporary file is left behind.
- An exception or a KeyboardInterrupt raised by a callback ends the run like one of the objective: the
  library's listeners still finish it (stop reason `error` or `interrupted`, results written, files
  closed, the folder resumable), and an error in a callback's own `on_run_end` no longer hides the first
  error or stops the results writer.
- Searches that start in the same second with the same `output`, in threads or processes, each get their
  own folder (`..._abc_2`, `_3`, ...) instead of failing with `FileExistsError`.
- Scores that differ only in their last bits (0.3 and 0.1 + 0.2), or equal scores of 1e300, are drawn like
  equal scores instead of losing every plot to "Too many bins for data range".
- Every plot is drawn on its own: a plot that fails no longer takes the later ones with it, and
  summary.json and report.md name each failed plot with its error.
- A huge score (a penalty of 1e160, 1e300 or the largest float) no longer loses the plots: scores beyond
  about 1e301 are drawn divided by a power of 2, and the statistics files are written without numpy
  warnings, with inf or NaN where a mean, spread or correlation does not fit a float.
- The plots of a large run are drawn in linear time (a 50,000-trial run: about 13 s instead of 68 s).
- report.md names the compressed `trials.csv.gz` and `population.csv.gz` of a large run, leaves out a
  population file an optimizer without population never wrote, and always ends its generation table with
  the last generation.
- The generation event of `ABCOptimizer` and `BBPSOOptimizer` fires right after the last trial of the
  generation, as for PSO, instead of after the first trial of the next one: the console's `evals`
  equals the `calls` column of generations.csv, and the population columns of generations.csv and the rows
  of population.csv show the population at the end of the generation. A run that ends with a complete
  generation counts it in summary.json's `generations`. The trials of a search do not change.
- A console whose encoding cannot show a parameter name, a choice or a value (cp1254, ascii) prints the
  character as a backslash escape and keeps printing, instead of going silent for the rest of the search;
  a stream whose `encoding` is missing, None or not usable gets the lines unchanged.

- `Int` and `Float` bounds given as numpy scalars or 0-d arrays are stored as the Python numbers they hold:
  `Int(np.uint8(0), np.uint8(255))` no longer overflows into an empty, "exhausted" run, and `float32`
  bounds give the same search as the same Python floats.
- `Int(0, 10**400)` and `Float` bounds too large for a float raise `ValueError` instead of `OverflowError`.
- One-coordinate decoding agrees with batch decoding for `Int` ranges wider than 2**53, and a unit value
  below 0 gives the first choice of a `Categorical` instead of one from the end.
- A `Categorical` of numpy arrays, or of choices equal across types such as `1`, `True` and `1.0`, works
  in the results files and plots, which find a trial's choice from its position, not by `==`; two equal
  arrays are rejected as duplicate choices.
- The `Categorical` docstring and the README say that a choice is handed to the objective as it is and
  must not be changed in place.
- `Trial == Trial` and `Study == Study` return a bool instead of raising `ValueError`: `x` and numpy arrays
  in the params are compared element by element, and two failed trials with a NaN score are equal.
- Integer arguments are checked before the search starts and before a results folder is written:
  `max_evals`, `runs`, `log_every`, `verbose` and `seed` of `optimize()` and `resume()`, the integer
  settings of the three optimizers and `Patience(evals)` accept Python and numpy integers and store them as
  `int`, and reject floats (also `20.0`), booleans and strings with `ValueError`; `verbose` still takes
  `False` and `True`, `seed` any iterable of seeds but a string (a numpy array, a range), `log_every=0` is
  rejected also without console output, and a negative seed of a later run before the first run starts.
  A folder that recorded an integer setting as a whole float (`limit=10.0`) still resumes.
- BBPSO no longer stops with `ZeroDivisionError` when all scores are tiny (around 1e-165 and below): the
  inverse quadratic interpolation step works on the scores scaled by a power of 2, and searches with scores
  of ordinary scale are unchanged.
- numpy values in the `meta` of a candidate (from an optimizer of your own) and numpy arrays among the
  parameters are written as the plain numbers and lists they hold in `trials.jsonl`, the CSV files and the
  summaries, instead of failing with `TypeError` or writing `array(...)` text; `config.json` records a
  `Categorical` as before.
- `config.json` records the module of the optimizer class, and `resume()` builds `ABCOptimizer`,
  `PSOOptimizer` or `BBPSOOptimizer` again only when the folder was written by the library's class; an
  optimizer of your own with one of these names now asks for `optimizer=` instead of being replaced.

## [0.7.1] - 2026-11-16

### Fixed

- A search whose budget ends inside a batch now tries the first trials of the same search with a larger
  budget, so `resume(max_evals=...)` extends every run; a run whose budget ended inside a batch after a
  cached repeat (mostly PSO flights) ends with other trials than in 0.7.0, and such a folder written by
  0.7.0 or older resumes only with its own budget.
- `resume()` accepts a space whose Categorical holds functions, lambdas, methods or other objects in a new
  process: the results files show such a choice without its memory address, and folders of older versions
  that recorded the address still resume.
- `resume()` builds a stopping rule again only when it is the library's `Patience`, `MaxTime` or
  `TargetScore`; a rule of your own with one of these names now asks for `stop=` instead of being replaced.
- A run that `MaxTime` ended always resumes without a call: the rule decides from where the trial ended,
  not from the clock after the results of the trial were written.
- A damaged `config.json` or `state.json` makes `resume()` raise `ValueError` naming the file instead of a
  bare `KeyError` or `JSONDecodeError`; a `state.json` without the generator state, start time or (for a run
  never resumed) pauses is rebuilt from the other records.

## [0.7.0] - 2026-11-14

### Added

- `optimize(stop=...)` ends a search early with stopping rules: `Patience(evals, min_delta)` after `evals`
  calls without improvement, `MaxTime(seconds, minutes, hours)` after that much search time and
  `TargetScore(score)` once the best score reaches `score`, recorded as the stop reasons `patience`, `time`
  and `target`.
- `Stopper` is the base class of a stopping rule of your own: it sets `reason` and overrides
  `should_stop(state)`, which is asked after every trial; the first rule that says stop ends the run, and
  every run works on its own copy of the rules.
- `config.json` records the stopping rules of a search; `resume()` continues with them, so a run a rule
  ended stays finished, or with the rules given as `resume(stop=...)`.

## [0.6.0] - 2026-11-11

### Added

- `resume(function, folder)` continues a search that stopped (Ctrl+C, an error, a killed process or a power
  cut) from its results folder and gives exactly the search an uninterrupted run would have given, also
  without a seed.
- `resume()` restores the finished runs of a folder without calling the objective, continues the unfinished
  one, runs the ones that never started, and takes a larger `max_evals` to extend the search.
- Every run folder holds `state.json` with the start time and the random generator state of the run,
  `config.json` records every optimizer setting (`Optimizer.settings()`) and the cache flag, and
  `trials.jsonl` is synced to disk at least once a second.

### Changed

- `config.json` stores Categorical choices that JSON can hold (strings, numbers, booleans, None) as values
  instead of their `repr()`, and `trials.jsonl` writes a NaN or infinite score as `"nan"`, `"inf"` or
  `"-inf"` instead of `null` and records the exact start time as `start_epoch`.

## [0.5.1] - 2026-11-10

### Added

- `optimize(runs=...)` repeats the search, every run with a fresh optimizer, its own cache and the full
  `max_evals`, and returns one `Study` per run; `seed` takes one seed per run or one integer for every run.
- With several runs, `optimizer` is a function that builds an optimizer from a seed, and the results
  folder holds every run, a summary over the runs and plots that compare them.

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
