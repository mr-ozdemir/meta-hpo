"""Tune a small convolutional network on CIFAR-10 with PyTorch.

Each evaluation trains for three epochs on 45,000 training images and scores accuracy on the other 5,000,
so the test set stays untouched for the final model. Needs `pip install meta-hpo torch torchvision`.
"""

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


def main():
    study = mh.optimize(train, space, direction="max", max_evals=30, seed=1)
    print(study.best_params, study.best_score)


if __name__ == "__main__":
    main()
