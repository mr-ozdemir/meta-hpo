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

## CIFAR-10 with PyTorch

This example tunes a small convolutional network on CIFAR-10.
It needs `pip install torch torchvision`.
Each evaluation trains for three epochs on 45,000 training images.
It scores accuracy on the other 5,000, so the test set stays untouched for the final model:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import metahpo as mh

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
)
full = datasets.CIFAR10("data", train=True, download=True, transform=transform)
train_set, val_set = random_split(full, [45000, 5000], generator=torch.Generator().manual_seed(0))

space = mh.Space(
    {
        "lr": mh.Float(1e-4, 1e-2, log=True),
        "batch_size": mh.Categorical([64, 128, 256]),
        "channels": mh.Int(16, 64),
        "dropout": mh.Float(0.0, 0.5),
        "optimizer": mh.Categorical(["adam", "sgd"]),
    }
)


def build_model(channels, dropout):
    return nn.Sequential(
        nn.Conv2d(3, channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(channels, 2 * channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(2 * channels * 8 * 8, 10),
    )


def train(params):
    torch.manual_seed(0)
    model = build_model(params["channels"], params["dropout"]).to(device)
    if params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    model.train()
    for _ in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in DataLoader(val_set, batch_size=1000):
            correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
    return correct / len(val_set)


study = mh.optimize(train, space, direction="max", max_evals=30, seed=1)
print(study.best_params, study.best_score)
```

Thirty evaluations take minutes on a GPU and much longer on a CPU.
After the search, train the best configuration for longer and report its accuracy on the test set.

## CIFAR-10 with TensorFlow

The same search with TensorFlow and Keras needs `pip install tensorflow`.
It uses the same split: the last 5,000 training images are held out for validation:

```python
import tensorflow as tf

import metahpo as mh

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_fit, y_fit = x_train[:45000], y_train[:45000]
x_val, y_val = x_train[45000:], y_train[45000:]

space = mh.Space(
    {
        "lr": mh.Float(1e-4, 1e-2, log=True),
        "batch_size": mh.Categorical([64, 128, 256]),
        "filters": mh.Int(16, 64),
        "dropout": mh.Float(0.0, 0.5),
        "optimizer": mh.Categorical(["adam", "sgd"]),
    }
)


def build_model(filters, dropout):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(2 * filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(10),
        ]
    )


def train(params):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
    model = build_model(params["filters"], params["dropout"])
    if params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x_fit, y_fit, batch_size=params["batch_size"], epochs=3, verbose=0)
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy


study = mh.optimize(train, space, direction="max", max_evals=30, seed=1)
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
