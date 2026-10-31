# meta-hpo

[![CI](https://github.com/mr-ozdemir/meta-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/mr-ozdemir/meta-hpo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/meta-hpo?label=PyPI)](https://pypi.org/project/meta-hpo/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)

Metaheuristic hyperparameter optimization for Python, with a small numpy core.

You describe a search space and a function that trains your model and returns a score.
meta-hpo searches for the parameters that maximise or minimise that score.
The first optimizer is the Artificial Bee Colony (ABC); more metaheuristics will follow behind the same interface.

## Installation

```console
pip install meta-hpo
```

meta-hpo needs Python 3.10+ and numpy.

## Quick start

```python
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
    model = build_model(params["kernel"], params["units"], params["dropout"])
    return fit_and_score(model, lr=params["lr"])


study = mh.optimize(train, space, direction="max", max_evals=300, seed=1)
print(study.best_params, study.best_score)
```

`build_model` and `fit_and_score` stand for your own model code; they are not part of meta-hpo.
`train` can do anything: a Keras or PyTorch training run, cross-validation, an F1 score.
Its training budget is entirely yours; meta-hpo only counts evaluations.

### Runnable example

This version needs only numpy. A closed-form score with its peak at `lr=1e-3`, `kernel=5`, `units=128`
and `dropout=0.2` stands in for training:

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


def score(params):
    lr_term = 0.05 * (np.log10(params["lr"]) + 3) ** 2
    units_term = ((params["units"] - 128) / 256) ** 2
    dropout_term = (params["dropout"] - 0.2) ** 2
    kernel_bonus = {3: 0.0, 5: 0.02, 7: 0.01}[params["kernel"]]
    return 0.9 - lr_term - units_term - dropout_term + kernel_bonus


study = mh.optimize(score, space, direction="max", max_evals=300, seed=1)
print(study.best_params, study.best_score)
```

## Ask and tell

For full control over where and how candidates are evaluated, ask for candidates and tell their scores.

A round is one `ask()` and one `tell()`; an evaluation is one call of `train`.
The first `ask()` returns the whole initial population, `colony_size // 2` candidates.
Every later `ask()` returns one candidate.
So 300 rounds with `colony_size=10` make 304 evaluations.
To set a budget, count evaluations instead of rounds:

```python
opt = mh.ABCOptimizer(space, direction="max", colony_size=10, limit=15, seed=1)

evals = 0
while evals < 300:
    candidates = opt.ask()
    opt.tell(candidates, [train(c.params) for c in candidates])
    evals += len(candidates)

print(opt.best_params, opt.best_score)
```

This loop makes exactly 300 evaluations.

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

`seed` makes the search reproducible: the same seed and the same scores give the same candidates.
Training itself is reproducible only if you also seed your framework and configure it to run deterministically.
If `train` returns different scores for the same parameters, the rest of the search differs too.

Optimizers do not hand out a parameter set twice.
When a new candidate repeats an earlier one, the optimizer draws it again, up to `max_tries` attempts
(100 by default); only then may a repeat come back.
So the proposed candidates follow a different distribution than in the textbook ABC,
most visibly in small discrete spaces.
`max_evals` counts real calls of your function.

## References

- Karaboga D., Basturk B. (2007). A powerful and efficient algorithm for numerical function optimization:
  artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459–471.
  [doi:10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x)
- The official ABC page, with reference implementations: <https://abc.erciyes.edu.tr/>

meta-hpo follows the phase order of the reference ABC: employed bees, then onlooker bees, then at most one
scout per cycle. It deviates on purpose in four points:

- It searches the unit cube `[0, 1]^d` and decodes each point to typed parameters.
- It accepts a neighbour by comparing raw minimised scores, not fitness values.
- It never hands out a parameter set again while it can still find a new one.
- It treats NaN and infinity as the worst score.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
