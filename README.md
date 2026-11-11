# meta-hpo

[![CI](https://github.com/mr-ozdemir/meta-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/mr-ozdemir/meta-hpo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/meta-hpo?label=PyPI&cacheSeconds=3600)](https://pypi.org/project/meta-hpo/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mr-ozdemir/meta-hpo/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/mr-ozdemir/meta-hpo/blob/master/pyproject.toml)

Metaheuristic hyperparameter optimization for Python.

You describe a search space and a function that trains your model and returns a score.
meta-hpo searches for the parameters that maximise or minimise that score.
Several metaheuristics share one interface ([Optimizers](#optimizers)), and every run is reported on the
console and in a results folder ([Results](#results)).
A run that stops half way continues from that folder as if it had never stopped
([Resuming a run](#resuming-a-run)).

## Installation

```console
pip install meta-hpo
```

meta-hpo needs Python 3.10+; pip installs numpy and matplotlib with it.

## Quick start

```python
import numpy as np

import metahpo as mh

space = mh.Space(
    {
        "lr": mh.Float(1e-4, 1e-2, log=True),
        "kernel": mh.Categorical([3, 5, 7]),
        "units": mh.Int(16, 256),
        "dropout": mh.Float(0.0, 0.5),
    }
)


def train(params):
    lr_term = 0.05 * (np.log10(params["lr"]) + 3) ** 2
    units_term = ((params["units"] - 128) / 256) ** 2
    dropout_term = (params["dropout"] - 0.2) ** 2
    kernel_bonus = {3: 0.0, 5: 0.02, 7: 0.01}[params["kernel"]]
    return 0.9 - lr_term - units_term - dropout_term + kernel_bonus


study = mh.optimize(train, space, direction="max", max_evals=300, seed=1)
print(study.best_params, study.best_score)
```

Here `train` is a closed-form score with its peak at `lr=1e-3`, `kernel=5`, `units=128` and
`dropout=0.2`, so the example runs as it is. In your code `train` builds and trains the model with these
parameters and returns its score: a Keras or PyTorch training run, cross-validation, an F1 score.
Its training budget is entirely yours; meta-hpo only counts evaluations.

If the run stops before it ends (Ctrl+C, a crash, a power cut, the computer switched off for the night),
no finished evaluation is lost. Pass `train` and the results folder of the run to `mh.resume` and the
search goes on as if it had never stopped ([Resuming a run](#resuming-a-run)):

```python
study = mh.resume(train, "hpo_results/2026-11-09_18-26-15_abc")
print(study.best_params, study.best_score)
```

## Results

`optimize()` prints one line per generation and, while it runs, writes everything to `hpo_results/` in the
working directory:

```text
[meta-hpo] run 1 | ABCOptimizer | direction=max | max_evals=300 | seed=1 | 4 parameters
gen    0 | evals     16/300 | best 0.884134 (trial 13) | gen best 0.884134 mean 0.807182 std 0.0518115 | ...
gen   29 | evals    300/300 | best 0.92 (trial 288) | gen best 0.919993 mean 0.913963 std 0.00801146 | ...
[meta-hpo] run 1 finished (budget) after 300 calls in 00:00:00 | best 0.92 at trial 288 | {...}
```

```text
hpo_results/2026-11-09_18-26-15_abc/    one folder per optimize() call: start time and optimizer
├── config.json                         search space, optimizer settings, budget and seed
├── summary.json                        best result, stop reason, calls and duration
├── report.md                           readable summary with the plots
└── run_001/
    ├── best.json                       the best so far, replaced on every improvement
    ├── trials.csv, trials.jsonl        every trial: params, score, phase, generation, time, duration
    ├── state.json                      start time and random generator state, for resuming
    ├── best_history.csv                every improvement of the best
    ├── generations.csv                 score and population statistics per generation
    ├── population.csv                  the population after every generation
    ├── parameters.csv, categories.csv  how each parameter relates to the score
    └── plots/                          convergence, diversity, distributions, correlations, ...
```

- `best.json` is replaced on every improvement, so the best so far survives a crash.
- Ctrl+C stops the search and returns the study so far.
- An error raised by your function is re-raised after the results are written.
- A stopped run continues from this folder with `mh.resume` ([Resuming a run](#resuming-a-run)).

| Setting | Effect |
|---|---|
| `output="hpo_results"` | folder for the results; `None` writes nothing |
| `verbose=1` | `0` silent, `1` one line per generation, `2` one line per trial as well |
| `log_every=1` | print every n-th generation |
| `callbacks=None` | your own `mh.Callback` objects receive the same events |

The environment variables `METAHPO_OUTPUT` and `METAHPO_VERBOSE` override `output` and `verbose`.

## Repeated runs

```python
studies = mh.optimize(train, space, direction="max", max_evals=300, runs=5, seed=[1, 2, 3, 4, 5])
```

- `runs` repeats the search; every run gets a fresh optimizer, its own cache and the full `max_evals`.
  The result is a list with one `Study` per run (a single `Study` when `runs=1`).
- `seed` is one seed per run, one integer that every run uses, or `None` for random runs.
- With several runs, pass your own optimizer as a function of the seed:
  `optimizer=lambda seed: mh.PSOOptimizer(space, direction="max", seed=seed)`.
- All runs go into one results folder (`run_001`, `run_002`, ...) with a summary and plots that
  compare the runs.

## Resuming a run

A run that stopped (Ctrl+C, a crash, a power cut) continues from its results folder and gives exactly the
search an uninterrupted run would have given, also without a seed.

```python
study = mh.resume(train, "hpo_results/2026-11-09_18-26-15_abc")
print(study.best_params, study.best_score)
```

Stopped after 120 of its 300 evaluations and resumed, the [Quick start](#quick-start) run restores those
120 trials without training again and ends on the same lines as the uninterrupted run in
[Results](#results):

```text
[meta-hpo] run 1 | ABCOptimizer | direction=max | max_evals=300 | seed=1 | 4 parameters
[meta-hpo] run 1 | resumed | 120 trials restored (120 calls) | best 0.919322
gen   11 | evals    126/300 | best 0.919326 (trial 122) | gen best 0.919326 mean 0.900067 std 0.0231047 | ...
gen   29 | evals    300/300 | best 0.92 (trial 288) | gen best 0.919993 mean 0.913963 std 0.00801146 | ...
[meta-hpo] run 1 finished (budget) after 300 calls in 00:00:00 | best 0.92 at trial 288 | {...}
```

- The folder provides the space, the optimizer and its settings, the budget, the seeds and every finished
  trial; only `train` is passed again.
- The trial that was running when the run stopped is evaluated again; finished trials are not.
- `max_evals=` extends a run to a larger budget.
- `optimizer=` is needed for an optimizer of your own and `space=` for Categorical choices that JSON
  cannot store, such as classes or functions.
- With several runs, the finished runs are restored, the unfinished run continues and the runs that never
  started run with their recorded seeds.
- A folder whose runs finished returns its studies without calling `train`.
- The results go on in the same folder.

## Examples

Runnable scripts that tune a small convolutional network on CIFAR-10 are in
[examples/](https://github.com/mr-ozdemir/meta-hpo/tree/master/examples):

- [PyTorch](https://github.com/mr-ozdemir/meta-hpo/blob/master/examples/cifar10_pytorch.py)
- [TensorFlow](https://github.com/mr-ozdemir/meta-hpo/blob/master/examples/cifar10_tensorflow.py)

## Ask and tell

For full control over where and how candidates are evaluated, ask for candidates and tell their scores.
`ABCOptimizer`'s first `ask()` returns the initial population (`colony_size // 2` candidates) and every later
`ask()` one candidate, so count evaluations, not rounds, to set a budget:

```python
opt = mh.ABCOptimizer(space, direction="max", colony_size=10, limit=15, seed=1)

evals = 0
while evals < 300:
    candidates = opt.ask()
    opt.tell(candidates, [train(c.params) for c in candidates])
    evals += len(candidates)

print(opt.best_params, opt.best_score)
```

This loop makes exactly 300 evaluations; 300 rounds of one `ask()` and one `tell()` would make 304.

## Optimizers

| Optimizer | Description |
|---|---|
| `ABCOptimizer` | Artificial Bee Colony, the default |
| `PSOOptimizer` | classic particle swarm optimization |
| `BBPSOOptimizer` | enhanced bare-bones PSO with Q-learning refinement |

```python
opt = mh.PSOOptimizer(space, direction="max", seed=1)
study = mh.optimize(train, space, direction="max", max_evals=300, optimizer=opt)
```

Settings, algorithm details and references: [docs/optimizers.md](https://github.com/mr-ozdemir/meta-hpo/blob/master/docs/optimizers.md).

## Search space

| Parameter | Values | Example |
|---|---|---|
| `Float(low, high, log=False)` | real numbers in `[low, high]` | `Float(1e-4, 1e-2, log=True)` |
| `Int(low, high, log=False)` | integers `low..high`, inclusive | `Int(16, 256)` |
| `Categorical(choices)` | one of `choices` | `Categorical(["relu", "tanh"])` |

Optimizers search the unit cube `[0, 1]^d`; the space maps each point to typed parameters.

## Behaviour

- A score of NaN or infinity counts as the worst possible value for the chosen direction.
- With `cache=True` (the default), a repeated parameter set reuses its earlier score instead of training again
  and does not count against `max_evals`.
- The same `seed` reproduces the same search.

`max_evals` counts real calls of your function. The same seed and the same scores give the same
candidates; training itself is reproducible only if you also seed your framework and make it
deterministic, and different scores for the same parameters change the rest of the search.

Optimizers do not hand out a parameter set twice: a repeat is drawn again, up to `max_tries` times (100
by default), before it may come back. So candidates follow a different distribution than in the textbook
algorithms, most visibly in small discrete spaces.

## Contributing

See [CONTRIBUTING.md](https://github.com/mr-ozdemir/meta-hpo/blob/master/CONTRIBUTING.md); changes are
listed in [CHANGELOG.md](https://github.com/mr-ozdemir/meta-hpo/blob/master/CHANGELOG.md).

## License

[MIT](https://github.com/mr-ozdemir/meta-hpo/blob/master/LICENSE)
