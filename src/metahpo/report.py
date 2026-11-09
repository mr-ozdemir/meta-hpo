"""Reporting for optimize(): console progress and an `hpo_results` folder with every detail of a search.

optimize() emits events to callbacks: on_run_start, on_trial, on_generation and on_run_end. Two callbacks
ship with the library and are switched on by default:

- ConsoleReporter prints one line per generation (`verbose=1`, every `log_every` generations) or one line
  per trial as well (`verbose=2`): evaluations used, the best score so far, the generation's best, mean and
  standard deviation, failed and cached trials, elapsed time and an estimate of the remaining time.
- ResultWriter writes a time-stamped folder below `output` (default `hpo_results` in the working directory).
  Trials, the best result, generations and population snapshots are appended while the search runs and
  flushed after every write, so the best result is on disk even if the process is killed. Summaries,
  analyses and plots are written when the run ends, also after KeyboardInterrupt or an error in the
  objective.

The environment variables METAHPO_OUTPUT and METAHPO_VERBOSE replace the defaults of `output` and `verbose`
(METAHPO_OUTPUT=none switches the folder off), which is useful in test suites and CI.

Folder layout (one `run_NNN` folder per run; `optimize(runs=...)` writes all runs into the same folder):

    <output>/<YYYY-mm-dd_HH-MM-SS>_<optimizer>/
        config.json, summary.json, report.md
        plots/runs_convergence.png, plots/runs_best.png      (only with several runs)
        run_001/
            trials.csv, trials.jsonl, best.json, best_history.csv, generations.csv, population.csv,
            parameters.csv, categories.csv, plots/*.png
            summary.json                                      (only with several runs)

With several runs, summary.json holds the summary of the best run plus `runs` (one entry per run),
`best_run` and `run_scores` (best, mean, median, std and worst of the runs' best scores), and report.md
starts with a table of the runs. Both are rewritten after every run, so they cover the runs so far.

Large searches (max_evals >= COMPRESS_FROM) write trials.csv and population.csv gzip-compressed
(.csv.gz), which pandas and most tools read directly.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import platform
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from metahpo.optimizers.base import Optimizer
from metahpo.space import Categorical, Float, Int, Space
from metahpo.study import Study, Trial

DEFAULT_OUTPUT = "hpo_results"
COMPRESS_FROM = 50_000


@dataclass
class RunState:
    """What a callback sees: the configuration, the live study and the running best.

    `best` is the best finite trial so far (None before one exists), `calls` the real calls made, `started`
    the wall-clock start in seconds since the epoch. `run` is the number of this run and `runs` the number
    of runs optimize() was asked for.
    """

    space: Space
    optimizer: Optimizer
    direction: str
    max_evals: int
    seed: int | None
    study: Study
    run: int = 1
    started: float = field(default_factory=time.time)
    calls: int = 0
    best: Trial | None = None
    runs: int = 1

    @property
    def optimizer_name(self) -> str:
        return type(self.optimizer).__name__

    def better(self, score: float, than: float) -> bool:
        if not math.isfinite(score):
            return False
        if not math.isfinite(than):
            return True
        return score < than if self.direction == "min" else score > than


class Callback:
    """Receives the events of optimize(); every method does nothing unless overridden."""

    def on_run_start(self, state: RunState) -> None:
        """Before the first candidate is asked."""

    def on_trial(self, state: RunState, trial: Trial, new_best: bool) -> None:
        """After every trial, cached repeats included; `new_best` is True when it improved the best."""

    def on_generation(self, state: RunState, generation: int, trials: list[Trial], population: Any) -> None:
        """When the optimizer's generation counter moves past `generation`, and once more for the last,
        possibly unfinished generation when the run ends. `trials` are the trials handed out in that
        generation; `population` is the optimizer's snapshot (see Optimizer.population) or None."""

    def on_run_end(self, state: RunState) -> None:
        """After the search stopped, for every stop reason; state.study.stop_reason says why."""


def resolve_output(output: str | os.PathLike[str] | None) -> Path | None:
    """The folder for ResultWriter: METAHPO_OUTPUT replaces the default; 'none' or None switch it off."""
    if output is not None and os.fspath(output) == DEFAULT_OUTPUT:
        output = os.environ.get("METAHPO_OUTPUT", DEFAULT_OUTPUT)
    if output is None or os.fspath(output).strip().lower() in ("", "none", "off", "0"):
        return None
    return Path(output)


def resolve_verbose(verbose: int) -> int:
    """The console level: METAHPO_VERBOSE replaces the default level 1."""
    if verbose == 1 and os.environ.get("METAHPO_VERBOSE", "").strip().isdigit():
        return int(os.environ["METAHPO_VERBOSE"])
    return verbose


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _iso(epoch: float) -> str:
    if not math.isfinite(epoch):
        return ""
    return dt.datetime.fromtimestamp(epoch).isoformat(timespec="milliseconds")


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6g}"


def _finite(values: Sequence[float]) -> np.ndarray[Any, np.dtype[np.float64]]:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: Sequence[float], direction: str) -> dict[str, float]:
    """best, mean, median, std and worst of the finite values (NaN when there is none)."""
    arr = _finite(values)
    if arr.size == 0:
        return {k: math.nan for k in ("best", "mean", "median", "std", "worst")}
    best, worst = (np.min(arr), np.max(arr)) if direction == "min" else (np.max(arr), np.min(arr))
    return {
        "best": float(best),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "worst": float(worst),
    }


class ConsoleReporter(Callback):
    """Epoch-style progress on the console: one line per generation, or per trial with verbose=2.

    When the console cannot be written (a closed stream or a broken pipe), it stops printing and the search
    and the other listeners go on.
    """

    def __init__(self, verbose: int = 1, log_every: int = 1, stream: IO[str] | None = None) -> None:
        if log_every < 1:
            raise ValueError("log_every must be at least 1")
        self.verbose = verbose
        self.log_every = log_every
        self.stream = stream
        self._broken = False
        self._run_bests: list[tuple[int, float]] = []

    def _print(self, text: str) -> None:
        if self._broken:
            return
        try:
            print(text, file=self.stream if self.stream is not None else sys.stdout, flush=True)
        except (OSError, ValueError):
            self._broken = True

    def on_run_start(self, state: RunState) -> None:
        seed = "random" if state.seed is None else state.seed
        self._print(
            f"[meta-hpo] run {_run_label(state)} | {state.optimizer_name} | direction={state.direction} | "
            f"max_evals={state.max_evals} | seed={seed} | {state.space.dim} parameters"
        )

    def on_trial(self, state: RunState, trial: Trial, new_best: bool) -> None:
        if self.verbose < 2:
            return
        mark = " *best*" if new_best else ""
        cached = " (cached)" if trial.cached else ""
        meta = trial.meta
        self._print(
            f"  trial {trial.number:5d} | gen {meta.get('generation', '-')} {meta.get('phase', '')} "
            f"#{meta.get('individual', '-')} | score {_fmt(trial.score)}{cached} | "
            f"{trial.duration:.3f}s{mark} | {trial.params}"
        )

    def on_generation(self, state: RunState, generation: int, trials: list[Trial], population: Any) -> None:
        if (generation + 1) % self.log_every and state.study.stop_reason == "":
            return
        gen = _stats([t.score for t in trials], state.direction)
        elapsed = time.time() - state.started
        eta = elapsed / state.calls * (state.max_evals - state.calls) if state.calls else math.nan
        eta_txt = _clock(eta) if math.isfinite(eta) else "--"
        best = state.best
        best_txt = "none" if best is None else f"{_fmt(best.score)} (trial {best.number})"
        failed = sum(t.failed for t in trials)
        cached = sum(t.cached for t in trials)
        self._print(
            f"gen {generation:4d} | evals {state.calls:6d}/{state.max_evals} | best {best_txt} | "
            f"gen best {_fmt(gen['best'])} mean {_fmt(gen['mean'])} std {_fmt(gen['std'])} | "
            f"failed {failed} cached {cached} | {_clock(elapsed)} ETA {eta_txt}"
        )

    def on_run_end(self, state: RunState) -> None:
        best = state.best
        elapsed = time.time() - state.started
        if best is None:
            result = "no finite score"
        else:
            result = f"best {_fmt(best.score)} at trial {best.number} | {best.params}"
        self._print(
            f"[meta-hpo] run {_run_label(state)} finished ({state.study.stop_reason}) after {state.calls} "
            f"calls in {_clock(elapsed)} | {result}"
        )
        if state.runs == 1:
            return
        self._run_bests.append((state.run, best.score if best is not None else math.nan))
        if state.run == state.runs or state.study.stop_reason == "interrupted":
            self._print(_runs_line(self._run_bests, state.direction))


def _run_label(state: RunState) -> str:
    return str(state.run) if state.runs == 1 else f"{state.run}/{state.runs}"


def _best_run(entries: Sequence[tuple[int, float]], direction: str) -> int | None:
    """The run number with the best finite score, the first one on a tie; None when no run has one."""
    finite = [(run, score) for run, score in entries if math.isfinite(score)]
    if not finite:
        return None
    pick = min if direction == "min" else max
    return pick(finite, key=lambda e: e[1])[0]


def _runs_line(entries: Sequence[tuple[int, float]], direction: str) -> str:
    stats = _stats([score for _, score in entries], direction)
    best_run = _best_run(entries, direction)
    if best_run is None:
        return f"[meta-hpo] {len(entries)} runs | no finite score"
    return (
        f"[meta-hpo] {len(entries)} runs | best {_fmt(stats['best'])} (run {best_run}) | "
        f"mean {_fmt(stats['mean'])} std {_fmt(stats['std'])} | worst {_fmt(stats['worst'])}"
    )


def _describe_space(space: Space) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, p in zip(space.names, space.parameters, strict=True):
        if isinstance(p, Categorical):
            out.append({"name": name, "type": "Categorical", "choices": [repr(c) for c in p.choices]})
        elif isinstance(p, Int):
            out.append({"name": name, "type": "Int", "low": p.low, "high": p.high, "log": p.log})
        else:
            assert isinstance(p, Float)
            out.append({"name": name, "type": "Float", "low": p.low, "high": p.high, "log": p.log})
    return out


def _optimizer_settings(opt: Optimizer) -> dict[str, Any]:
    names = (
        "colony_size",
        "limit",
        "swarm_size",
        "inertia",
        "cognitive",
        "social",
        "max_velocity",
        "target",
        "crossover",
        "q_alpha",
        "q_gamma",
        "max_tries",
    )
    return {n: getattr(opt, n) for n in names if isinstance(getattr(opt, n, None), (int, float))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, data: Any) -> None:
    """Writes JSON atomically: a temporary file replaced in one step, so a reader never sees half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class _CsvFile:
    """A CSV file written row by row and flushed after every row; gzip-compressed when `compressed`."""

    def __init__(self, path: Path, header: list[str], compressed: bool = False) -> None:
        self.path = path.with_suffix(path.suffix + ".gz") if compressed else path
        raw: IO[str]
        if compressed:
            raw = io.TextIOWrapper(gzip.open(self.path, "wb"), encoding="utf-8", newline="")
        else:
            raw = self.path.open("w", encoding="utf-8", newline="")
        self._fh = raw
        self._writer = csv.writer(raw)
        self._writer.writerow(header)
        self._fh.flush()

    def row(self, values: list[Any]) -> None:
        self._writer.writerow(["" if v is None else v for v in values])
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else repr(value)
    return value


class ResultWriter(Callback):
    """Writes the `hpo_results` folder; see the module docstring for the layout and the files."""

    def __init__(self, output: str | os.PathLike[str] = DEFAULT_OUTPUT, plots: bool = True) -> None:
        self.root = Path(output)
        self.plots = plots
        self.folder: Path | None = None
        self.run_dir: Path | None = None
        self._files: dict[str, _CsvFile] = {}
        self._jsonl: IO[str] | None = None
        self._pop_header: list[str] | None = None
        self._generation_rows: list[dict[str, Any]] = []
        self._best_history: list[dict[str, Any]] = []
        self._notes: list[str] = []
        self._calls_at: dict[int, int] = {}
        self._best_at: dict[int, float] = {}
        self._seeds: list[Any] = []
        self._runs: list[dict[str, Any]] = []

    def _make_folder(self, state: RunState) -> Path:
        stamp = dt.datetime.fromtimestamp(state.started).strftime("%Y-%m-%d_%H-%M-%S")
        base = self.root / f"{stamp}_{state.optimizer_name.replace('Optimizer', '').lower()}"
        folder, n = base, 1
        while folder.exists():
            n += 1
            folder = base.with_name(f"{base.name}_{n}")
        folder.mkdir(parents=True)
        return folder

    def on_run_start(self, state: RunState) -> None:
        if self.folder is None:
            self.folder = self._make_folder(state)
        self._files = {}
        self._jsonl = None
        self._pop_header = None
        self._generation_rows = []
        self._best_history = []
        self._notes = []
        self._calls_at = {}
        self._best_at = {}
        self._seeds.append(state.seed)
        self.run_dir = self.folder / f"run_{state.run:03d}"
        self.run_dir.mkdir()
        (self.run_dir / "plots").mkdir()
        compressed = state.max_evals >= COMPRESS_FROM
        names = list(state.space.names)
        self._files["trials"] = _CsvFile(
            self.run_dir / "trials.csv",
            [
                "run",
                "trial",
                "generation",
                "phase",
                "individual",
                "detail",
                *names,
                *[f"x_{n}" for n in names],
                "score",
                "failed",
                "cached",
                "best_so_far",
                "is_new_best",
                "start_time",
                "end_time",
                "duration_s",
                "elapsed_s",
            ],
            compressed,
        )
        self._compressed = compressed
        self._jsonl = (self.run_dir / "trials.jsonl").open("w", encoding="utf-8")
        self._files["best_history"] = _CsvFile(
            self.run_dir / "best_history.csv",
            ["run", "trial", "generation", "phase", "score", "time", "elapsed_s", "calls", *names],
        )
        param_cols = [
            c
            for n, p in zip(names, state.space.parameters, strict=True)
            if not isinstance(p, Categorical)
            for c in (f"mean_{n}", f"std_{n}")
        ]
        self._gen_header = [
            "run",
            "generation",
            "calls",
            "trials",
            "best_so_far",
            "gen_best",
            "gen_mean",
            "gen_median",
            "gen_std",
            "gen_worst",
            "failed",
            "cached",
            "pop_best",
            "pop_mean",
            "pop_std",
            "pop_worst",
            "diversity_std",
            "diversity_mean_distance",
            *param_cols,
            "improved",
            "end_time",
            "duration_s",
        ]
        self._files["generations"] = _CsvFile(self.run_dir / "generations.csv", self._gen_header)
        self._gen_started = state.started
        self._last_best = math.nan
        config = {
            "library": {"name": "meta-hpo", "version": _version()},
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "optimizer": {"name": state.optimizer_name, "settings": _optimizer_settings(state.optimizer)},
            "direction": state.direction,
            "max_evals": state.max_evals,
            "runs": state.runs,
            "seed": self._seeds[0],
            "seeds": self._seeds,
            "space": _describe_space(state.space),
            "started": _iso(state.started),
            "folder": str(self.folder),
        }
        _write_json(self.folder / "config.json", config)
        _write_json(self.run_dir / "best.json", {"status": "running", "best": None})

    def on_trial(self, state: RunState, trial: Trial, new_best: bool) -> None:
        assert self.run_dir is not None and self._jsonl is not None
        names = list(state.space.names)
        meta = dict(trial.meta)
        generation, phase, individual = (
            meta.pop("generation", ""),
            meta.pop("phase", ""),
            meta.pop("individual", ""),
        )
        detail = json.dumps(meta) if meta else ""
        end = trial.start_time + trial.duration if math.isfinite(trial.start_time) else math.nan
        best_score = state.best.score if state.best is not None else math.nan
        self._calls_at[trial.number] = state.calls
        self._best_at[trial.number] = best_score
        params = [trial.params[n] for n in names]
        xs = [float(v) for v in trial.x]
        self._files["trials"].row(
            [
                state.run,
                trial.number,
                generation,
                phase,
                individual,
                detail,
                *params,
                *[_cell(v) for v in xs],
                _cell(trial.score),
                int(trial.failed),
                int(trial.cached),
                _cell(best_score),
                int(new_best),
                _iso(trial.start_time),
                _iso(end),
                f"{trial.duration:.6f}",
                f"{(end if math.isfinite(end) else time.time()) - state.started:.6f}",
            ]
        )
        record = {
            "run": state.run,
            "trial": trial.number,
            "generation": generation,
            "phase": phase,
            "individual": individual,
            "detail": meta,
            "params": trial.params,
            "x": xs,
            "score": trial.score,
            "failed": trial.failed,
            "cached": trial.cached,
            "best_so_far": best_score,
            "is_new_best": new_best,
            "start_time": _iso(trial.start_time),
            "end_time": _iso(end),
            "duration_s": trial.duration,
        }
        self._jsonl.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._jsonl.flush()
        if new_best:
            elapsed = (end if math.isfinite(end) else time.time()) - state.started
            entry = {
                "run": state.run,
                "trial": trial.number,
                "generation": generation,
                "phase": phase,
                "score": trial.score,
                "time": _iso(end),
                "elapsed_s": elapsed,
                "calls": state.calls,
                "params": trial.params,
            }
            self._best_history.append(entry)
            self._files["best_history"].row(
                [
                    state.run,
                    trial.number,
                    generation,
                    phase,
                    _cell(trial.score),
                    _iso(end),
                    f"{elapsed:.6f}",
                    state.calls,
                    *params,
                ]
            )
            _write_json(self.run_dir / "best.json", {"status": "running", "best": entry})

    def on_generation(self, state: RunState, generation: int, trials: list[Trial], population: Any) -> None:
        assert self.run_dir is not None
        now = time.time()
        gen = _stats([t.score for t in trials], state.direction)
        last = trials[-1].number if trials else -1
        row: dict[str, Any] = {
            "run": state.run,
            "generation": generation,
            "calls": self._calls_at.get(last, state.calls),
            "trials": len(trials),
            "best_so_far": self._best_at.get(last, state.best.score if state.best is not None else math.nan),
            "gen_best": gen["best"],
            "gen_mean": gen["mean"],
            "gen_median": gen["median"],
            "gen_std": gen["std"],
            "gen_worst": gen["worst"],
            "failed": sum(t.failed for t in trials),
            "cached": sum(t.cached for t in trials),
        }
        if population is not None:
            xs = np.asarray(population["x"], dtype=np.float64)
            pop = _stats(list(population["score"]), state.direction)
            row.update(
                {
                    "pop_best": pop["best"],
                    "pop_mean": pop["mean"],
                    "pop_std": pop["std"],
                    "pop_worst": pop["worst"],
                }
            )
            row["diversity_std"] = float(np.mean(np.std(xs, axis=0))) if len(xs) else math.nan
            diffs = xs[:, None, :] - xs[None, :, :]
            dist = np.sqrt((diffs**2).sum(-1))
            n = len(xs)
            row["diversity_mean_distance"] = float(dist.sum() / (n * (n - 1))) if n > 1 else 0.0
            decoded = state.space.decode(xs) if len(xs) else np.zeros((0, state.space.dim))
            for j, (name, p) in enumerate(zip(state.space.names, state.space.parameters, strict=True)):
                if not isinstance(p, Categorical):
                    row[f"mean_{name}"] = float(np.mean(decoded[:, j]))
                    row[f"std_{name}"] = float(np.std(decoded[:, j]))
            self._write_population(state, generation, population)
        best_now = row["best_so_far"]
        row["improved"] = int(state.better(best_now, self._last_best))
        self._last_best = best_now
        row["end_time"] = _iso(now)
        row["duration_s"] = round(now - self._gen_started, 6)
        self._gen_started = now
        self._generation_rows.append(row)
        self._files["generations"].row([_cell(row.get(c, math.nan)) for c in self._gen_header])

    def _write_population(self, state: RunState, generation: int, population: dict[str, Any]) -> None:
        assert self.run_dir is not None
        xs = np.asarray(population["x"], dtype=np.float64)
        extras = [k for k in population if k not in ("x", "score")]
        names = list(state.space.names)
        if "population" not in self._files:
            header = ["run", "generation", "individual", "score", *names, *[f"x_{n}" for n in names], *extras]
            self._files["population"] = _CsvFile(self.run_dir / "population.csv", header, self._compressed)
        rows = state.space.to_dicts(xs) if len(xs) else []
        for i, params in enumerate(rows):
            self._files["population"].row(
                [
                    state.run,
                    generation,
                    i,
                    _cell(float(population["score"][i])),
                    *[params[n] for n in names],
                    *[_cell(float(v)) for v in xs[i]],
                    *[_cell(population[k][i]) for k in extras],
                ]
            )

    def close(self) -> None:
        """Closes the files that are written while the search runs; on_run_end calls it."""
        for f in self._files.values():
            f.close()
        if self._jsonl is not None and not self._jsonl.closed:
            self._jsonl.close()

    def on_run_end(self, state: RunState) -> None:
        assert self.folder is not None and self.run_dir is not None
        self.close()
        study = state.study
        ended = time.time()
        summary = _run_summary(state, ended)
        _write_json(
            self.run_dir / "best.json",
            {"status": "finished", "stop_reason": study.stop_reason, "best": summary["best"]},
        )
        _write_parameter_analysis(state, self.run_dir)
        plot_files: list[str] = []
        if self.plots:
            try:
                from metahpo import plots

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plot_files = plots.make_plots(
                        state, self.run_dir, self._generation_rows, self._best_history
                    )
            except Exception as exc:
                self._notes.append(f"Plots failed: {type(exc).__name__}: {exc}")
        summary["plots"] = plot_files
        summary["notes"] = self._notes
        summary["folder"] = str(self.folder)
        if state.runs == 1:
            _write_json(self.folder / "summary.json", summary)
            _write_report(self.folder, self.run_dir, state, summary, self._generation_rows)
            return
        summary["run"] = state.run
        _write_json(self.run_dir / "summary.json", summary)
        score = summary["best"]["score"] if summary["best"] is not None else math.nan
        self._runs.append(
            {
                "summary": summary,
                "run_dir": self.run_dir,
                "generations": self._generation_rows,
                "best_history": self._best_history,
                "score": score,
            }
        )
        self._write_runs(state)

    def _write_runs(self, state: RunState) -> None:
        """summary.json, report.md and the run comparison plots of all runs so far."""
        assert self.folder is not None
        entries = [(r["summary"]["run"], r["score"]) for r in self._runs]
        best_run = _best_run(entries, state.direction)
        shown = next((r for r in self._runs if r["summary"]["run"] == best_run), self._runs[-1])
        notes: list[str] = []
        run_plots: list[str] = []
        if self.plots:
            try:
                from metahpo import plots

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_plots = plots.make_run_plots(self._runs, self.folder, state.direction)
            except Exception as exc:
                notes.append(f"Run plots failed: {type(exc).__name__}: {exc}")
        runs = [
            {
                "run": r["summary"]["run"],
                "seed": r["summary"]["seed"],
                "best_score": r["score"],
                "stop_reason": r["summary"]["stop_reason"],
                "calls": r["summary"]["calls"],
                "duration_s": r["summary"]["duration_s"],
                "folder": r["run_dir"].name,
            }
            for r in self._runs
        ]
        summary = {
            **shown["summary"],
            "runs": runs,
            "best_run": best_run,
            "run_scores": _stats([score for _, score in entries], state.direction),
            "run_plots": run_plots,
            "notes": [*shown["summary"]["notes"], *notes],
        }
        _write_json(self.folder / "summary.json", summary)
        _write_report(self.folder, shown["run_dir"], state, summary, shown["generations"])


def _version() -> str:
    """The version of the loaded metahpo package, read without importing the package from this module."""
    return str(getattr(sys.modules.get("metahpo"), "__version__", "unknown"))


def _run_summary(state: RunState, ended: float) -> dict[str, Any]:
    study = state.study
    best = state.best
    durations = _finite([t.duration for t in study.trials if not t.cached])
    best_entry = None
    if best is not None:
        best_entry = {
            "score": best.score,
            "params": best.params,
            "trial": best.number,
            "generation": best.meta.get("generation"),
            "phase": best.meta.get("phase"),
            "individual": best.meta.get("individual"),
            "found_at": _iso(best.start_time + best.duration) if math.isfinite(best.start_time) else None,
            "found_after_s": (best.start_time + best.duration - state.started)
            if math.isfinite(best.start_time)
            else None,
        }
    return {
        "status": "finished",
        "stop_reason": study.stop_reason,
        "optimizer": state.optimizer_name,
        "direction": state.direction,
        "seed": state.seed,
        "max_evals": state.max_evals,
        "calls": state.calls,
        "trials": len(study.trials),
        "cached_trials": sum(t.cached for t in study.trials),
        "failed_trials": sum(t.failed for t in study.trials),
        "generations": state.optimizer.generation,
        "started": _iso(state.started),
        "ended": _iso(ended),
        "duration_s": ended - state.started,
        "trial_duration_s": {
            "mean": float(durations.mean()) if durations.size else None,
            "median": float(np.median(durations)) if durations.size else None,
            "max": float(np.max(durations)) if durations.size else None,
            "total": float(durations.sum()) if durations.size else 0.0,
        },
        "best": best_entry,
    }


def _rank(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def _corr(a: np.ndarray[Any, np.dtype[np.float64]], b: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _write_parameter_analysis(state: RunState, run_dir: Path) -> None:
    """parameters.csv (one row per parameter) and categories.csv (one row per category of a Categorical)."""
    real = [t for t in state.study.trials if not t.cached and not t.failed]
    scores = np.array([t.score for t in real], dtype=np.float64)
    best = state.best
    params_file = _CsvFile(
        run_dir / "parameters.csv",
        [
            "parameter",
            "type",
            "low",
            "high",
            "log",
            "best_value",
            "distinct_values",
            "mean",
            "std",
            "min",
            "max",
            "pearson_with_score",
            "spearman_with_score",
        ],
    )
    cats_file = _CsvFile(
        run_dir / "categories.csv",
        ["parameter", "choice", "trials", "mean_score", "best_score", "worst_score"],
    )
    for name, p in zip(state.space.names, state.space.parameters, strict=True):
        values = [t.params[name] for t in real]
        best_value = best.params[name] if best is not None else ""
        if isinstance(p, Categorical):
            params_file.row(
                [
                    name,
                    "Categorical",
                    "",
                    "",
                    "",
                    best_value,
                    len({repr(v) for v in values}),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
            for choice in p.choices:
                chosen = np.array([s for v, s in zip(values, scores, strict=True) if v == choice])
                st = _stats(chosen.tolist(), state.direction)
                cats_file.row(
                    [name, choice, len(chosen), _cell(st["mean"]), _cell(st["best"]), _cell(st["worst"])]
                )
            continue
        arr = np.array(values, dtype=np.float64)
        kind = "Int" if isinstance(p, Int) else "Float"
        low, high, log = getattr(p, "low", ""), getattr(p, "high", ""), getattr(p, "log", "")
        if arr.size == 0:
            params_file.row([name, kind, low, high, log, best_value, 0, "", "", "", "", "", ""])
            continue
        params_file.row(
            [
                name,
                kind,
                low,
                high,
                log,
                best_value,
                len(set(arr.tolist())),
                _cell(float(arr.mean())),
                _cell(float(arr.std())),
                _cell(float(np.min(arr))),
                _cell(float(np.max(arr))),
                _cell(_corr(arr, scores)),
                _cell(_corr(_rank(arr), _rank(scores))),
            ]
        )
    params_file.close()
    cats_file.close()


def _write_report(
    folder: Path, run_dir: Path, state: RunState, summary: dict[str, Any], generations: list[dict[str, Any]]
) -> None:
    """report.md: a human-readable summary with links to the data files and plots."""
    best = summary["best"]
    rel = run_dir.name
    lines = [
        f"# meta-hpo report: {state.optimizer_name}",
        "",
        f"- Started: {summary['started']}, ended: {summary['ended']} ({_clock(summary['duration_s'])})",
        f"- Stop reason: **{summary['stop_reason']}**",
        f"- Real calls: {summary['calls']} of max_evals={summary['max_evals']}; trials {summary['trials']} "
        f"(cached {summary['cached_trials']}, failed {summary['failed_trials']}); generations "
        f"{summary['generations']}",
        f"- Direction: {summary['direction']}; seed: {summary['seed']}",
        "",
    ]
    if "runs" in summary:
        stats = summary["run_scores"]
        lines += [
            "## Runs",
            "",
            f"Best scores of {len(summary['runs'])} runs: best {_fmt(stats['best'])} (run "
            f"{summary['best_run']}), mean {_fmt(stats['mean'])}, std {_fmt(stats['std'])}, worst "
            f"{_fmt(stats['worst'])}. The sections below describe run {summary['run']}.",
            "",
            "| Run | Seed | Best score | Stop reason | Calls | Duration | Folder |",
            "|---|---|---|---|---|---|---|",
            *[
                f"| {r['run']} | {r['seed']} | {_fmt(r['best_score'])} | {r['stop_reason']} | "
                f"{r['calls']} | {_clock(r['duration_s'])} | `{r['folder']}` |"
                for r in summary["runs"]
            ],
            "",
            *[f"![{Path(p).stem}](plots/{Path(p).name})" for p in summary["run_plots"]],
            "",
        ]
    lines += ["## Best result", ""]
    if best is None:
        lines.append("No trial returned a finite score.")
    else:
        lines += [
            f"- Score: **{best['score']}** at trial {best['trial']} (generation {best['generation']}, "
            f"phase {best['phase']}, individual {best['individual']})",
            f"- Found after {best['found_after_s']:.1f} s" if best["found_after_s"] is not None else "",
            "",
            "| Parameter | Value |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in best["params"].items()],
        ]
    lines += [
        "",
        "## Progress by generation",
        "",
        "| Generation | Calls | Best so far | Gen best | Gen mean | Gen std | Diversity |",
        "|---|---|---|---|---|---|---|",
    ]
    step = max(1, len(generations) // 25)
    for row in generations[::step]:
        lines.append(
            f"| {row['generation']} | {row['calls']} | {_fmt(row['best_so_far'])} | "
            f"{_fmt(row['gen_best'])} | {_fmt(row['gen_mean'])} | {_fmt(row['gen_std'])} | "
            f"{_fmt(row.get('diversity_std', math.nan))} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- `{rel}/trials.csv`: every trial (phase, generation, individual, parameters, unit-cube x, score, "
        "start and end time, duration)",
        f"- `{rel}/generations.csv`: statistics per generation (scores, population, diversity, parameter "
        "means)",
        f"- `{rel}/population.csv`: the population at the end of every generation",
        f"- `{rel}/best_history.csv`, `{rel}/best.json`: every improvement and the best result",
        f"- `{rel}/parameters.csv`, `{rel}/categories.csv`: parameter analysis",
        "- `config.json`, `summary.json`: configuration and summary",
        "",
    ]
    if summary["plots"]:
        lines += ["## Plots", ""]
        lines += [f"![{Path(p).stem}]({rel}/plots/{Path(p).name})" for p in summary["plots"]]
        lines.append("")
    for note in summary["notes"]:
        lines.append(f"> {note}")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
