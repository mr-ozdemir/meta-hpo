# Examples

Each script tunes a small convolutional network on CIFAR-10 with `optimize()`.

| Script | Framework | Install |
|---|---|---|
| [cifar10_pytorch.py](cifar10_pytorch.py) | PyTorch | `pip install meta-hpo torch torchvision` |

Run a script from this folder:

```console
python cifar10_pytorch.py
```

The dataset is downloaded on the first run.
Each evaluation trains for three epochs on 45,000 training images and scores accuracy on the other 5,000,
so the test set stays untouched for the final model.
Thirty evaluations take minutes on a GPU and much longer on a CPU.
After the search, train the best configuration for longer and report its accuracy on the test set.
