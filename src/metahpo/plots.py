"""Plots of an optimize() run for the `hpo_results` folder, drawn with matplotlib.

matplotlib is a dependency of meta-hpo. It is imported only here, inside make_plots, so `import metahpo`
stays fast. The figures are drawn without pyplot and styled on their axes, so the backend is not switched
and the open figures and the rcParams of the user's session are left untouched; a figure is not registered
anywhere, and it is cleared after it was saved or failed. Every plot is drawn on its own: a plot that fails
is skipped, the others are still written, and ResultWriter notes each failed plot with its error in
summary.json and report.md.

Degenerate data is drawn, not refused: scores that differ only in their last bits are drawn like equal
scores, and scores beyond about 1e301 are drawn divided by a power of 2, with the axis labelled so. The
cost of every plot grows linearly with the number of trials.

A pruned trial's score is its last intermediate value, not a final score: the score plots leave it out as
they leave out a NaN score, while it still counts as an evaluation on the axes that count them. When trials
reported intermediate values, 12_learning_curves draws them by step, completed trials in colour and pruned
trials in grey with a cross where they stopped.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from metahpo.space import Categorical, _choice_index, _choice_text

HISTOGRAM_BINS = 40
"""Bins of the score and duration histograms."""

DENSITY_BINS = 256
"""Grid points of the kernel density of the scores."""


def _style(ax: Any) -> None:
    """A light grid behind the data and no top or right spine, set on the axes, not in rcParams."""
    ax.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")


def _resolvable(low: float, high: float, bins: int) -> bool:
    """True when [low, high] splits into `bins` bins of finite, positive width in float64: a span too small
    for them (values that differ only in their last bits) or too large (an infinite width) is not."""
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.linspace(low, high, bins + 1)
    return bool(np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0))


def _kde(values: np.ndarray[Any, np.dtype[np.float64]]) -> tuple[Any, Any] | None:
    """A Gaussian kernel density estimate on a grid (Scott's bandwidth), or None for fewer than two
    distinct values, for values whose spread is too small or too large for the grid, and for a spread that
    does not fit a float. The values are binned first, so the cost does not grow with their number squared.
    """
    n = len(values)
    with np.errstate(over="ignore", invalid="ignore"):
        spread = float(np.std(values))
    if n < 2 or spread == 0.0 or not math.isfinite(spread):
        return None
    h = 1.06 * spread * n ** (-0.2)
    low, high = float(np.min(values)) - 3 * h, float(np.max(values)) + 3 * h
    if not _resolvable(low, high, DENSITY_BINS):
        return None
    counts, edges = np.histogram(values, bins=DENSITY_BINS, range=(low, high))
    grid = (edges[:-1] + edges[1:]) / 2
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / h) ** 2)
    return grid, kernel @ counts / (n * h * math.sqrt(2 * math.pi))


def _histogram(ax: Any, values: np.ndarray[Any, np.dtype[np.float64]], kde: bool = False) -> None:
    """A HISTOGRAM_BINS-bin histogram, with its kernel density scaled to the counts when `kde`. Values whose
    span is too small for the bins are drawn like equal values: all in the middle bin of a range around
    them."""
    low, high = float(np.min(values)), float(np.max(values))
    if low == high or not _resolvable(low, high, HISTOGRAM_BINS):
        middle = low + (high - low) / 2
        width = max(0.5, abs(middle) * 2.0**-40)
        _, edges, _ = ax.hist(
            np.full(len(values), middle),
            bins=HISTOGRAM_BINS,
            range=(middle - width, middle + width),
            color="C0",
            alpha=0.6,
            edgecolor="white",
        )
        return
    _, edges, _ = ax.hist(values, bins=HISTOGRAM_BINS, color="C0", alpha=0.6, edgecolor="white")
    density = _kde(values) if kde else None
    if density is not None:
        grid, dens = density
        ax.plot(grid, dens * len(values) * (edges[1] - edges[0]), color="C0", lw=1.5)


LARGEST_DRAWN = 2.0**1000
"""Scores beyond this size (about 1e301) are drawn scaled down: matplotlib computes axis ranges, ticks and
colour scales from differences of the values, which overflow near the largest float."""


def _score_scale(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """1.0, or the power of 2 that brings the largest finite |value| below 2**990 when it is beyond
    LARGEST_DRAWN, so the scaled values can be drawn; a power of 2 scales every value exactly."""
    finite = values[np.isfinite(values)]
    largest = float(np.max(np.abs(finite))) if finite.size else 0.0
    if largest <= LARGEST_DRAWN:
        return 1.0
    return math.ldexp(1.0, 990 - math.frexp(largest)[1])


def _scaled(label: str, scale: float) -> str:
    """The axis label of values multiplied by `scale`, for example 'score / 2^34'."""
    return label if scale == 1.0 else f"{label} / 2^{-math.frexp(scale)[1] + 1}"


def _parallel_lines(ax: Any, xs: np.ndarray[Any, np.dtype[np.float64]], top: np.ndarray[Any, Any]) -> None:
    """The unit-cube positions of every trial as polylines, the trials in `top` in red over the others, drawn
    as two line collections instead of one line per trial."""
    from matplotlib.collections import LineCollection

    columns = np.arange(xs.shape[1], dtype=np.float64)
    segments = np.stack([np.broadcast_to(columns, xs.shape), xs], axis=-1)
    ax.add_collection(LineCollection(segments[~top], colors="0.75", alpha=0.25, linewidths=0.6, zorder=1))
    ax.add_collection(LineCollection(segments[top], colors="C3", alpha=0.9, linewidths=1.4, zorder=3))
    ax.set_xlim(-0.05 * max(1, len(columns) - 1), (len(columns) - 1) * 1.05 or 1.0)
    ax.set_ylim(-0.05, 1.05)


def _phase_stairs(ax: Any, kinds: list[Any], gens: list[int], tally: Counter[Any]) -> None:
    """The trials per phase and generation as stacked bars of width 1, one filled step outline per phase
    instead of one rectangle per generation, so the cost does not grow with the number of patches."""
    span = np.arange(gens[0], gens[-1] + 1)
    edges = np.arange(gens[0] - 0.5, gens[-1] + 1.0)
    bottom = np.zeros(len(span))
    for k in kinds:
        heights = np.array([tally[(gen, k)] for gen in span.tolist()], dtype=np.float64)
        ax.stairs(bottom + heights, edges, baseline=bottom, fill=True, label=k)
        bottom = bottom + heights


def _draw(
    name: str,
    plot: Callable[[], Any],
    out: Path,
    written: list[str],
    notes: list[str] | None,
    made: list[Any],
) -> None:
    """Draws one plot with `plot`, which returns its figure (None when the plot does not apply), and saves
    it as out/<name>.png. An exception skips the plot and is noted in `notes`. Every figure made for the plot
    (`made`) is cleared afterwards, whether it was saved or not, so none outlives its plot."""
    try:
        with np.errstate(all="ignore"):
            figure = plot()
            if figure is not None:
                path = out / f"{name}.png"
                figure.savefig(path, dpi=130, bbox_inches="tight")
                written.append(str(path))
    except Exception as exc:
        if notes is not None:
            notes.append(f"Plot {name} failed: {type(exc).__name__}: {exc}")
    finally:
        for fig in made:
            fig.clear()
        made.clear()


def make_plots(
    state: Any,
    run_dir: Path,
    generations: list[dict[str, Any]],
    best_history: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> list[str]:
    """Draws every plot of one run into run_dir/plots and returns the file paths. Every plot is drawn on its
    own; one that fails is skipped and noted in `notes` as 'Plot <name> failed: <error>'."""
    from matplotlib.figure import Figure

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    trials = state.study.trials
    real = [t for t in trials if not t.cached]
    direction = state.direction
    written: list[str] = []

    made: list[Any] = []

    def subplots(*args: Any, **kwargs: Any) -> Any:
        fig = Figure(figsize=kwargs.pop("figsize", None), layout="constrained")
        made.append(fig)
        axes = fig.subplots(*args, **kwargs)
        for ax in np.atleast_1d(axes).flat:
            _style(ax)
        return fig, axes

    def draw(name: str, plot: Callable[[], Any]) -> None:
        _draw(name, plot, out, written, notes, made)

    raw_scores = np.array([math.nan if t.pruned else t.score for t in real], dtype=np.float64)
    scale = _score_scale(raw_scores)
    score_label = _scaled("score", scale)
    scores = raw_scores * scale
    finite = np.isfinite(scores)
    calls = np.arange(1, len(real) + 1)
    pick = np.fmin if direction == "min" else np.fmax
    best_curve = pick.accumulate(np.where(finite, scores, np.nan)) if len(real) else np.array([])
    elapsed = np.array([state.elapsed(t.start_time + t.duration, t.number) for t in real], dtype=np.float64)

    def convergence() -> Any:
        fig, axes = subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot(calls, best_curve, color="C0", lw=2, label="best so far")
        axes[0].scatter(calls[finite], scores[finite], s=8, alpha=0.35, color="C1", label="trial score")
        axes[0].set(xlabel="real evaluations", ylabel=score_label, title="Convergence by evaluations")
        axes[0].legend()
        axes[1].plot(elapsed, best_curve, color="C0", lw=2)
        axes[1].set(xlabel="seconds since start", ylabel=f"best {score_label}", title="Convergence by time")
        return fig

    draw("01_convergence", convergence)

    if generations:
        g = np.array([r["generation"] for r in generations])
        col = {
            k: np.array([r.get(k, math.nan) for r in generations], dtype=np.float64)
            for k in (
                "best_so_far",
                "gen_best",
                "gen_mean",
                "gen_std",
                "pop_mean",
                "pop_std",
                "diversity_std",
                "diversity_mean_distance",
            )
        }
        for k in ("best_so_far", "gen_best", "gen_mean", "gen_std", "pop_mean", "pop_std"):
            col[k] = col[k] * scale

        def per_generation() -> Any:
            fig, ax = subplots(figsize=(10, 4.5))
            ax.plot(g, col["best_so_far"], lw=2, label="best so far")
            ax.plot(g, col["gen_best"], lw=1, label="generation best")
            ax.plot(g, col["gen_mean"], lw=1, label="generation mean")
            ax.fill_between(
                g,
                col["gen_mean"] - col["gen_std"],
                col["gen_mean"] + col["gen_std"],
                alpha=0.2,
                label="generation mean ± std",
            )
            if np.isfinite(col["pop_mean"]).any():
                ax.plot(g, col["pop_mean"], ls="--", lw=1, label="population mean")
            ax.set(xlabel="generation", ylabel=score_label, title="Scores per generation")
            ax.legend(fontsize=8)
            return fig

        draw("02_generations", per_generation)

        def diversity() -> Any:
            if not np.isfinite(col["diversity_std"]).any():
                return None
            fig, ax = subplots(figsize=(10, 4))
            ax.plot(g, col["diversity_std"], label="mean std per dimension")
            ax.plot(g, col["diversity_mean_distance"], label="mean pairwise distance")
            ax.set(xlabel="generation", ylabel="unit-cube distance", title="Population diversity")
            ax.legend()
            return fig

        draw("03_diversity", diversity)

    def score_distribution() -> Any:
        if not finite.any():
            return None
        fig, ax = subplots(figsize=(8, 4))
        _histogram(ax, scores[finite], kde=True)
        ax.set(xlabel=score_label, title="Score distribution of real evaluations")
        return fig

    draw("04_score_distribution", score_distribution)

    names = list(state.space.names)
    params = state.space.parameters
    ok = [t for t in real if t.state == "complete"]
    ok_scores = np.array([t.score for t in ok], dtype=np.float64) * scale
    n = len(names)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    def parameter_vs_score() -> Any:
        fig, axes = subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
        for j, (ax, name, p) in enumerate(zip(axes.flat, names, params, strict=False)):
            values = [t.params[name] for t in ok]
            if isinstance(p, Categorical):
                where = np.array([_choice_index(p, float(t.x[j])) for t in ok], dtype=np.int64)
                used = sorted(set(where.tolist()))
                ax.boxplot(
                    [ok_scores[where == i] for i in used], positions=used, widths=0.6, showfliers=False
                )
                jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(where))
                ax.scatter(where + jitter, ok_scores, s=4, color="k", alpha=0.4)
                ax.set_xticks(range(len(p.choices)), [_choice_text(c) for c in p.choices])
            else:
                ax.scatter(values, ok_scores, s=10, alpha=0.5, c=np.arange(len(ok)), cmap="viridis")
                if getattr(p, "log", False):
                    ax.set_xscale("log")
            ax.set(xlabel=name, ylabel=score_label)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle("Parameter vs score (colour = trial order)")
        return fig

    def parameter_evolution() -> Any:
        fig, axes = subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
        numbers = [t.number for t in ok]
        for j, (ax, name, p) in enumerate(zip(axes.flat, names, params, strict=False)):
            values = [t.params[name] for t in ok]
            if isinstance(p, Categorical):
                values = [_choice_index(p, float(t.x[j])) for t in ok]
                ax.set_yticks(range(len(p.choices)), [_choice_text(c) for c in p.choices])
            sc = ax.scatter(numbers, values, s=8, c=ok_scores, cmap="viridis")
            if getattr(p, "log", False):
                ax.set_yscale("log")
            ax.set(xlabel="trial", ylabel=name)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.colorbar(sc, ax=list(axes.flat)[:n], label=score_label)
        fig.suptitle("Parameter evolution over trials (colour = score)")
        return fig

    def parallel_coordinates() -> Any:
        xs = np.array([t.x for t in ok], dtype=np.float64).reshape(len(ok), n)
        ranked = np.argsort(ok_scores if direction == "min" else -ok_scores)
        top = np.zeros(len(ok), dtype=bool)
        top[ranked[: max(1, len(ok) // 10)]] = True
        fig, ax = subplots(figsize=(max(6, 1.6 * n), 4.5))
        _parallel_lines(ax, xs, top)
        ax.set_xticks(range(n), names, rotation=20)
        ax.set(ylabel="unit-cube position", title="Parallel coordinates (red = best 10%)")
        return fig

    numeric = [(name, p) for name, p in zip(names, params, strict=True) if not isinstance(p, Categorical)]

    def correlation() -> Any:
        if not numeric or len(ok) <= 2:
            return None
        data = np.column_stack(
            [np.array([t.params[nm] for t in ok], dtype=np.float64) for nm, _ in numeric] + [ok_scores]
        )
        labels = [nm for nm, _ in numeric] + ["score"]
        corr = np.corrcoef(data, rowvar=False)
        fig, ax = subplots(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(False)
        for (i, j), value in np.ndenumerate(corr):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Correlation of numeric parameters and score")
        return fig

    if ok:
        draw("05_parameter_vs_score", parameter_vs_score)
        draw("06_parameter_evolution", parameter_evolution)
        draw("07_parallel_coordinates", parallel_coordinates)
        draw("08_correlation", correlation)

    def durations_plot() -> Any:
        if not real:
            return None
        durations = np.array([t.duration for t in real], dtype=np.float64)
        fig, axes = subplots(1, 2, figsize=(12, 4))
        axes[0].plot(calls, durations, lw=0.8)
        axes[0].set(xlabel="real evaluation", ylabel="seconds", title="Duration per evaluation")
        _histogram(axes[1], durations)
        axes[1].set(xlabel="seconds", title="Duration distribution")
        return fig

    draw("09_durations", durations_plot)

    def phases_plot() -> Any:
        phases = [(t.meta.get("generation"), t.meta.get("phase")) for t in trials if t.meta.get("phase")]
        if not phases:
            return None
        kinds = sorted({p for _, p in phases})
        gens = sorted({g for g, _ in phases if g is not None})
        tally = Counter(phases)
        fig, ax = subplots(figsize=(10, 4))
        if all(isinstance(gen, (int, np.integer)) for gen in gens):
            _phase_stairs(ax, kinds, [int(gen) for gen in gens], tally)
        else:
            bottom = np.zeros(len(gens))
            for k in kinds:
                counts = [tally[(gen, k)] for gen in gens]
                ax.bar(gens, counts, bottom=bottom, label=k, width=1.0)
                bottom += np.array(counts)
        ax.set(xlabel="generation", ylabel="trials", title=f"Trials per phase ({state.optimizer_name})")
        ax.legend(fontsize=8)
        return fig

    draw("10_phases", phases_plot)

    def improvements() -> Any:
        if not best_history:
            return None
        steps = [b["calls"] for b in best_history]
        values = [b["score"] * scale for b in best_history]
        fig, ax = subplots(figsize=(10, 4))
        ax.step(steps, values, where="post")
        ax.scatter(steps, values, s=14)
        ax.set(
            xlabel="real evaluations",
            ylabel=f"best {score_label}",
            title=f"Improvements ({len(best_history)})",
        )
        return fig

    draw("11_improvements", improvements)

    curves = [t for t in real if t.intermediate_values]

    def learning_curves() -> Any:
        if not curves:
            return None
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        values = np.array([v for t in curves for v in t.intermediate_values.values()], dtype=np.float64)
        factor = _score_scale(values)
        lines: dict[bool, list[Any]] = {False: [], True: []}
        ends: list[Any] = []
        for t in curves:
            steps = sorted(t.intermediate_values)
            points = np.array([[s, t.intermediate_values[s] * factor] for s in steps], dtype=np.float64)
            lines[t.pruned].append(points)
            if t.pruned:
                ends.append(points[-1])
        fig, ax = subplots(figsize=(10, 4.5))
        ax.add_collection(LineCollection(lines[False], colors="C0", alpha=0.35, linewidths=1.0))
        ax.add_collection(LineCollection(lines[True], colors="0.55", alpha=0.35, linewidths=1.0))
        if ends:
            stops = np.array(ends)
            ax.scatter(stops[:, 0], stops[:, 1], marker="x", s=16, color="0.35")
        ax.autoscale()
        handles = [
            Line2D([], [], color="C0", label=f"completed ({len(lines[False])})"),
            Line2D([], [], color="0.55", marker="x", label=f"pruned ({len(lines[True])})"),
        ]
        ax.legend(handles=handles)
        ax.set(xlabel="step", ylabel=_scaled("intermediate value", factor), title="Learning curves")
        return fig

    draw("12_learning_curves", learning_curves)
    return written


def make_run_plots(
    runs: list[dict[str, Any]], folder: Path, direction: str, notes: list[str] | None = None
) -> list[str]:
    """Draws the comparison of several runs into folder/plots and returns the file paths: the best score
    over the evaluations of every run, and the best score of every run. Each is drawn on its own; one that
    fails is skipped and noted in `notes`."""
    from matplotlib.figure import Figure

    out = folder / "plots"
    out.mkdir(exist_ok=True)
    written: list[str] = []
    scores = np.array([r["score"] for r in runs], dtype=np.float64)
    history_scores = np.array([b["score"] for r in runs for b in r["best_history"]], dtype=np.float64)
    scale = min(_score_scale(scores), _score_scale(history_scores))
    label = _scaled("best score", scale)

    made: list[Any] = []

    def new_figure(**kwargs: Any) -> Any:
        fig = Figure(layout="constrained", **kwargs)
        made.append(fig)
        return fig

    def draw(name: str, plot: Callable[[], Any]) -> None:
        _draw(name, plot, out, written, notes, made)

    def convergence() -> Any:
        fig = new_figure(figsize=(10, 4.5))
        ax = fig.subplots()
        _style(ax)
        for r in runs:
            history = r["best_history"]
            if history:
                calls = [b["calls"] for b in history] + [r["summary"]["calls"]]
                curve = [b["score"] * scale for b in history] + [history[-1]["score"] * scale]
                ax.step(calls, curve, where="post", lw=1.4, label=f"run {r['summary']['run']}")
        ax.set(xlabel="real evaluations", ylabel=label, title=f"Best score of {len(runs)} runs")
        if any(r["best_history"] for r in runs):
            ax.legend(fontsize=8, ncol=max(1, len(runs) // 10))
        return fig

    def best_per_run() -> Any:
        numbers = [r["summary"]["run"] for r in runs]
        values = scores * scale
        finite = np.isfinite(values)
        fig = new_figure(figsize=(max(6, 0.5 * len(runs) + 3), 4))
        ax = fig.subplots()
        _style(ax)
        ax.scatter(np.array(numbers)[finite], values[finite], s=50, color="C0", zorder=3)
        if finite.any():
            mean = float(np.sum(values[finite] / np.count_nonzero(finite)))
            ax.axhline(mean, color="C3", lw=1.2, ls="--", label="mean")
            ax.legend(fontsize=8)
            low, high = float(np.min(values[finite])), float(np.max(values[finite]))
            pad = (high - low) * 0.2 or abs(high) * 0.05 or 1.0
            ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(numbers)
        better = "lower" if direction == "min" else "higher"
        ax.set(xlabel="run", ylabel=label, title=f"Best score per run ({better} is better)")
        return fig

    draw("runs_convergence", convergence)
    draw("runs_best", best_per_run)
    return written
