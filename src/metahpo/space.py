"""Search space: users declare typed parameters, optimizers search the unit cube [0, 1]^d.

Parameters of the same kind are decoded together: one numpy expression per kind (linear float, log float,
integer, log integer, categorical) covers all of its columns, so decoding cost does not grow with the number
of Python-level parameter objects. Integer, categorical and log-scaled handling lives only here; the
optimizers never see it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _decode_float(u: FloatArray, low: Any, high: Any, log: bool) -> FloatArray:
    if log:
        lo, hi = np.log(low), np.log(high)
        return np.asarray(np.clip(np.exp(lo + u * (hi - lo)), low, high), dtype=np.float64)
    return np.asarray(np.clip(low + u * (high - low), low, high), dtype=np.float64)


def _decode_int(u: FloatArray, low: Any, high: Any, log: bool) -> FloatArray:
    if log:
        lo, hi = np.log(np.asarray(low) - 0.5), np.log(np.asarray(high) + 0.5)
        values = np.rint(np.exp(lo + u * (hi - lo)))
    else:
        values = low + np.floor(u * (np.asarray(high) - low + 1))
    return np.asarray(np.clip(values, low, high), dtype=np.float64)


def _decode_categorical(u: FloatArray, n: Any) -> FloatArray:
    return np.asarray(np.minimum(np.floor(u * n), np.asarray(n) - 1), dtype=np.float64)


def _exact_integer(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer() and abs(number) <= 2**53 and number == value


class Parameter:
    """Base class of the parameter types."""

    @property
    def kind(self) -> str:
        """Parameters of the same kind are decoded together."""
        raise NotImplementedError

    def decode(self, u: FloatArray) -> NDArray[Any]:
        """Unit values in [0, 1] to raw values (categories as indices)."""
        raise NotImplementedError

    def decode_one(self, u: float) -> float:
        """`decode` for a single value, bit-identical to `decode`, for one-coordinate moves."""
        raise NotImplementedError

    def to_python(self, value: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class Float(Parameter):
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"Float needs low < high, got {self.low} and {self.high}")
        if not math.isfinite(self.high - self.low):
            raise ValueError(f"Float needs finite bounds with a finite width, got {self.low} and {self.high}")
        if self.log and self.low <= 0:
            raise ValueError("a log-scaled Float needs low > 0")

    @property
    def kind(self) -> str:
        return "float_log" if self.log else "float"

    def decode(self, u: FloatArray) -> FloatArray:
        return _decode_float(u, self.low, self.high, self.log)

    def decode_one(self, u: float) -> float:
        if self.log:
            lo, hi = np.log(self.low), np.log(self.high)
            value = float(np.exp(lo + u * (hi - lo)))
        else:
            value = self.low + u * (self.high - self.low)
        return min(max(value, self.low), self.high)

    def to_python(self, value: Any) -> float:
        return float(value)


@dataclass(frozen=True)
class Int(Parameter):
    """Integers low..high inclusive.

    With `log=False` every integer gets an equal share of the unit interval; with `log=True` the shares
    are equal on a log scale, so small integers get larger shares.
    """

    low: int
    high: int
    log: bool = False

    def __post_init__(self) -> None:
        for bound in (self.low, self.high):
            if not _exact_integer(bound):
                raise ValueError(f"Int bounds must be integers with magnitude at most 2**53, got {bound!r}")
        if not self.low <= self.high:
            raise ValueError(f"Int needs low <= high, got {self.low} and {self.high}")
        if self.log and self.low <= 0:
            raise ValueError("a log-scaled Int needs low > 0")

    @property
    def kind(self) -> str:
        return "int_log" if self.log else "int"

    def decode(self, u: FloatArray) -> NDArray[np.int64]:
        return _decode_int(u, self.low, self.high, self.log).astype(np.int64)

    def decode_one(self, u: float) -> float:
        if self.log:
            lo, hi = np.log(self.low - 0.5), np.log(self.high + 0.5)
            value = float(np.rint(np.exp(lo + u * (hi - lo))))
        else:
            value = self.low + math.floor(u * (self.high - self.low + 1))
        return float(min(max(value, self.low), self.high))

    def to_python(self, value: Any) -> int:
        return int(value)


@dataclass(frozen=True)
class Categorical(Parameter):
    """One of `choices`; every choice gets an equal share of the unit interval."""

    choices: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.choices) == 0:
            raise ValueError("Categorical needs at least one choice")
        object.__setattr__(self, "choices", tuple(self.choices))

    @property
    def kind(self) -> str:
        return "categorical"

    def decode(self, u: FloatArray) -> NDArray[np.int64]:
        return _decode_categorical(u, len(self.choices)).astype(np.int64)

    def decode_one(self, u: float) -> float:
        n = len(self.choices)
        return float(min(math.floor(u * n), n - 1))

    def to_python(self, value: Any) -> Any:
        return self.choices[int(value)]


class Space:
    """An ordered set of named parameters; dimension d = number of parameters."""

    def __init__(self, parameters: Mapping[str, Parameter]) -> None:
        if not isinstance(parameters, Mapping):
            raise TypeError(
                f"a Space needs a mapping of names to parameters, got {type(parameters).__name__}"
            )
        if not parameters:
            raise ValueError("a Space needs at least one parameter")
        for name, p in parameters.items():
            if not isinstance(p, Parameter):
                raise TypeError(f"parameter {name!r} is not a metahpo parameter: {p!r}")
        self.names: tuple[str, ...] = tuple(parameters)
        self.parameters: tuple[Parameter, ...] = tuple(parameters.values())
        groups: dict[str, list[int]] = {}
        for j, p in enumerate(self.parameters):
            groups.setdefault(p.kind, []).append(j)
        self._groups = []
        for kind, cols in groups.items():
            ps = [self.parameters[j] for j in cols]
            low = np.array([getattr(p, "low", 0) for p in ps], dtype=np.float64)
            highs = [getattr(p, "high", len(getattr(p, "choices", ()))) for p in ps]
            high = np.array(highs, dtype=np.float64)
            self._groups.append((kind, np.array(cols), low, high))
        self._int_cols = [j for j, p in enumerate(self.parameters) if isinstance(p, Int)]
        self._cat_cols = [(j, p.choices) for j, p in enumerate(self.parameters) if isinstance(p, Categorical)]
        self._plain = not self._int_cols and not self._cat_cols

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        """n uniform points of the unit cube, shape (n, d)."""
        return rng.random((n, self.dim))

    def decode(self, u: FloatArray) -> FloatArray:
        """Unit-cube points (n, d) to raw values (n, d): integers as whole numbers, categories as indices."""
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        raw = np.empty_like(u)
        for kind, cols, low, high in self._groups:
            block = u[:, cols]
            if kind in ("float", "float_log"):
                raw[:, cols] = _decode_float(block, low, high, kind == "float_log")
            elif kind in ("int", "int_log"):
                raw[:, cols] = _decode_int(block, low, high, kind == "int_log")
            else:
                raw[:, cols] = _decode_categorical(block, high)
        return raw

    def rows(self, u: FloatArray) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
        """Unit-cube points (n, d) to (keys, params) with one decode.

        A key is the hashable raw row: equal keys mean the user's function sees equal parameters.
        params are the dictionaries the user's function receives, with Python types.
        """
        raw = self.decode(u).tolist()
        keys = [tuple(r) for r in raw]
        if not self._plain:
            for r in raw:
                for j in self._int_cols:
                    r[j] = int(r[j])
                for j, choices in self._cat_cols:
                    r[j] = choices[int(r[j])]
        names = self.names
        return keys, [dict(zip(names, r, strict=True)) for r in raw]

    def move(
        self, key: tuple[Any, ...], params: dict[str, Any], j: int, u_j: float
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """(key, params) of a point differing from a decoded point only in coordinate j: O(1) decoding."""
        p = self.parameters[j]
        raw = p.decode_one(u_j)
        new_params = dict(params)
        new_params[self.names[j]] = p.to_python(raw)
        return key[:j] + (raw,) + key[j + 1 :], new_params

    def to_dicts(self, u: FloatArray) -> list[dict[str, Any]]:
        """Unit-cube points (n, d) to the parameter dictionaries that the user's function receives."""
        return self.rows(u)[1]
