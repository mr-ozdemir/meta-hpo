import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space


def test_float_linear_and_log_bounds():
    u = np.array([0.0, 0.5, 1.0])
    assert Float(2.0, 4.0).decode(u).tolist() == [2.0, 3.0, 4.0]
    log = Float(1e-4, 1e-2, log=True).decode(u)
    assert log[0] == pytest.approx(1e-4) and log[1] == pytest.approx(1e-3) and log[2] == pytest.approx(1e-2)


def test_int_gives_every_value_an_equal_share():
    u = (np.arange(10000) + 0.5) / 10000
    counts = np.bincount(Int(3, 7).decode(u) - 3)
    assert counts.tolist() == [2000] * 5
    assert Int(3, 7).decode(np.array([1.0])).tolist() == [7]


def test_categorical_maps_to_choices():
    space = Space({"k": Categorical([3, 5, 7])})
    params = space.to_dicts(np.array([[0.1], [0.5], [1.0]]))
    assert [p["k"] for p in params] == [3, 5, 7]


def test_to_dicts_returns_python_types():
    space = Space(
        {"lr": Float(1e-4, 1e-2, log=True), "units": Int(16, 256), "act": Categorical(["relu", "tanh"])}
    )
    (params,) = space.to_dicts(np.array([[0.3, 0.6, 0.9]]))
    assert isinstance(params["lr"], float) and isinstance(params["units"], int) and params["act"] == "tanh"


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        Float(1.0, 1.0)
    with pytest.raises(ValueError):
        Float(0.0, 1.0, log=True)
    with pytest.raises(ValueError):
        Categorical([])
    with pytest.raises(TypeError):
        Space({"x": (0, 1)})
