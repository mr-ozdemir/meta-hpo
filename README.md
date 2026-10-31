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

`train` can do anything: a Keras or PyTorch training run, cross-validation, an F1 score.
Its training budget is entirely yours; meta-hpo only counts evaluations.

## Ask and tell

For full control over where and how candidates are evaluated:

```python
opt = mh.ABCOptimizer(space, direction="max", colony_size=10, limit=15, seed=1)

for _ in range(300):
    candidates = opt.ask()
    opt.tell(candidates, [train(c.params) for c in candidates])

print(opt.best_params, opt.best_score)
```

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
