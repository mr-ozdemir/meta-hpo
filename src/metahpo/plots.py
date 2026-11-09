"""Plots of an optimize() run for the `hpo_results` folder, drawn with matplotlib.

matplotlib is a dependency of meta-hpo. It is imported only here, inside make_plots, so `import metahpo`
stays fast. The figures are drawn without pyplot and styled on their axes, so the backend is not switched
and the open figures and the rcParams of the user's session are left untouched. When a plot fails,
ResultWriter notes the error in report.md and the other result files are still written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from metahpo.space import Categorical


def _style(ax: Any) -> None:
    """A light grid behind the data and no top or right spine, set on the axes, not in rcParams."""
    ax.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")


def _kde(values: np.ndarray[Any, np.dtype[np.float64]]) -> tuple[Any, Any] | None:
    """A Gaussian kernel density estimate on a grid (Scott's bandwidth), or None for fewer than two
    distinct values. The values are binned first, so the cost does not grow with their number squared.
    """
    n, spread = len(values), float(np.std(values))
    if n < 2 or spread == 0.0:
        return None
    h = 1.06 * spread * n ** (-0.2)
    counts, edges = np.histogram(values, bins=256, range=(np.min(values) - 3 * h, np.max(values) + 3 * h))
    grid = (edges[:-1] + edges[1:]) / 2
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / h) ** 2)
    return grid, kernel @ counts / (n * h * math.sqrt(2 * math.pi))


def _histogram(ax: Any, values: np.ndarray[Any, np.dtype[np.float64]], kde: bool = False) -> None:
    """A 40-bin histogram, with its kernel density scaled to the counts when `kde`."""
    _, edges, _ = ax.hist(values, bins=40, color="C0", alpha=0.6, edgecolor="white")
    density = _kde(values) if kde else None
    if density is not None:
        grid, dens = density
        ax.plot(grid, dens * len(values) * (edges[1] - edges[0]), color="C0", lw=1.5)


def make_plots(
    state: Any, run_dir: Path, generations: list[dict[str, Any]], best_history: list[dict[str, Any]]
) -> list[str]:
    """Draws every plot of one run into run_dir/plots and returns the file paths."""
    from matplotlib.figure import Figure

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    trials = state.study.trials
    real = [t for t in trials if not t.cached]
    direction = state.direction
    written: list[str] = []

    def subplots(*args: Any, **kwargs: Any) -> Any:
        fig = Figure(figsize=kwargs.pop("figsize", None), layout="constrained")
        axes = fig.subplots(*args, **kwargs)
        for ax in np.atleast_1d(axes).flat:
            _style(ax)
        return fig, axes

    def save(fig: Any, name: str) -> None:
        path = out / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        written.append(str(path))

    scores = np.array([t.score for t in real], dtype=np.float64)
    finite = np.isfinite(scores)
    calls = np.arange(1, len(real) + 1)
    pick = np.fmin if direction == "min" else np.fmax
    best_curve = pick.accumulate(np.where(finite, scores, np.nan)) if len(real) else np.array([])
    elapsed = np.array([t.start_time + t.duration - state.started for t in real], dtype=np.float64)

    fig, axes = subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(calls, best_curve, color="C0", lw=2, label="best so far")
    axes[0].scatter(calls[finite], scores[finite], s=8, alpha=0.35, color="C1", label="trial score")
    axes[0].set(xlabel="real evaluations", ylabel="score", title="Convergence by evaluations")
    axes[0].legend()
    axes[1].plot(elapsed, best_curve, color="C0", lw=2)
    axes[1].set(xlabel="seconds since start", ylabel="best score", title="Convergence by time")
    save(fig, "01_convergence")

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
        ax.set(xlabel="generation", ylabel="score", title="Scores per generation")
        ax.legend(fontsize=8)
        save(fig, "02_generations")

        if np.isfinite(col["diversity_std"]).any():
            fig, ax = subplots(figsize=(10, 4))
            ax.plot(g, col["diversity_std"], label="mean std per dimension")
            ax.plot(g, col["diversity_mean_distance"], label="mean pairwise distance")
            ax.set(xlabel="generation", ylabel="unit-cube distance", title="Population diversity")
            ax.legend()
            save(fig, "03_diversity")

    if finite.any():
        fig, ax = subplots(figsize=(8, 4))
        _histogram(ax, scores[finite], kde=True)
        ax.set(xlabel="score", title="Score distribution of real evaluations")
        save(fig, "04_score_distribution")

    names = list(state.space.names)
    params = state.space.parameters
    ok = [t for t in real if not t.failed]
    ok_scores = np.array([t.score for t in ok], dtype=np.float64)
    if ok:
        n = len(names)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
        for ax, name, p in zip(axes.flat, names, params, strict=False):
            values = [t.params[name] for t in ok]
            if isinstance(p, Categorical):
                labels = [repr(v) for v in values]
                order = [repr(c) for c in p.choices]
                where = np.array([order.index(label) for label in labels])
                used = sorted(set(where.tolist()))
                ax.boxplot(
                    [ok_scores[where == i] for i in used], positions=used, widths=0.6, showfliers=False
                )
                jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(where))
                ax.scatter(where + jitter, ok_scores, s=4, color="k", alpha=0.4)
                ax.set_xticks(range(len(order)), order)
            else:
                ax.scatter(values, ok_scores, s=10, alpha=0.5, c=np.arange(len(ok)), cmap="viridis")
                if getattr(p, "log", False):
                    ax.set_xscale("log")
            ax.set(xlabel=name, ylabel="score")
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle("Parameter vs score (colour = trial order)")
        save(fig, "05_parameter_vs_score")

        fig, axes = subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
        numbers = [t.number for t in ok]
        for ax, name, p in zip(axes.flat, names, params, strict=False):
            values = [t.params[name] for t in ok]
            if isinstance(p, Categorical):
                values = [list(p.choices).index(v) for v in values]
                ax.set_yticks(range(len(p.choices)), [repr(c) for c in p.choices])
            sc = ax.scatter(numbers, values, s=8, c=ok_scores, cmap="viridis")
            if getattr(p, "log", False):
                ax.set_yscale("log")
            ax.set(xlabel="trial", ylabel=name)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.colorbar(sc, ax=list(axes.flat)[:n], label="score")
        fig.suptitle("Parameter evolution over trials (colour = score)")
        save(fig, "06_parameter_evolution")

        xs = np.array([t.x for t in ok], dtype=np.float64)
        ranked = np.argsort(ok_scores if direction == "min" else -ok_scores)
        top = set(ranked[: max(1, len(ok) // 10)].tolist())
        fig, ax = subplots(figsize=(max(6, 1.6 * n), 4.5))
        for i in range(len(ok)):
            ax.plot(
                range(n),
                xs[i],
                color="C3" if i in top else "0.75",
                alpha=0.9 if i in top else 0.25,
                lw=1.4 if i in top else 0.6,
                zorder=3 if i in top else 1,
            )
        ax.set_xticks(range(n), names, rotation=20)
        ax.set(ylabel="unit-cube position", title="Parallel coordinates (red = best 10%)")
        save(fig, "07_parallel_coordinates")

        numeric = [(name, p) for name, p in zip(names, params, strict=True) if not isinstance(p, Categorical)]
        if numeric and len(ok) > 2:
            data = np.column_stack(
                [np.array([t.params[nm] for t in ok], dtype=np.float64) for nm, _ in numeric] + [ok_scores]
            )
            labels = [nm for nm, _ in numeric] + ["score"]
            with np.errstate(invalid="ignore", divide="ignore"):
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
            save(fig, "08_correlation")

    durations = np.array([t.duration for t in real], dtype=np.float64)
    if len(real):
        fig, axes = subplots(1, 2, figsize=(12, 4))
        axes[0].plot(calls, durations, lw=0.8)
        axes[0].set(xlabel="real evaluation", ylabel="seconds", title="Duration per evaluation")
        _histogram(axes[1], durations)
        axes[1].set(xlabel="seconds", title="Duration distribution")
        save(fig, "09_durations")

    phases = [(t.meta.get("generation"), t.meta.get("phase")) for t in trials if t.meta.get("phase")]
    if phases:
        kinds = sorted({p for _, p in phases})
        gens = sorted({g for g, _ in phases if g is not None})
        counts = {k: [sum(1 for g, p in phases if g == gen and p == k) for gen in gens] for k in kinds}
        fig, ax = subplots(figsize=(10, 4))
        bottom = np.zeros(len(gens))
        for k in kinds:
            ax.bar(gens, counts[k], bottom=bottom, label=k, width=1.0)
            bottom += np.array(counts[k])
        ax.set(xlabel="generation", ylabel="trials", title=f"Trials per phase ({state.optimizer_name})")
        ax.legend(fontsize=8)
        save(fig, "10_phases")

    if best_history:
        fig, ax = subplots(figsize=(10, 4))
        ax.step([b["calls"] for b in best_history], [b["score"] for b in best_history], where="post")
        ax.scatter([b["calls"] for b in best_history], [b["score"] for b in best_history], s=14)
        ax.set(xlabel="real evaluations", ylabel="best score", title=f"Improvements ({len(best_history)})")
        save(fig, "11_improvements")
    return written


def make_run_plots(runs: list[dict[str, Any]], folder: Path, direction: str) -> list[str]:
    """Draws the comparison of several runs into folder/plots and returns the file paths: the best score
    over the evaluations of every run, and the best score of every run."""
    from matplotlib.figure import Figure

    out = folder / "plots"
    out.mkdir(exist_ok=True)
    written: list[str] = []

    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax = fig.subplots()
    _style(ax)
    for r in runs:
        history = r["best_history"]
        if history:
            calls = [b["calls"] for b in history] + [r["summary"]["calls"]]
            curve = [b["score"] for b in history] + [history[-1]["score"]]
            ax.step(calls, curve, where="post", lw=1.4, label=f"run {r['summary']['run']}")
    ax.set(xlabel="real evaluations", ylabel="best score", title=f"Best score of {len(runs)} runs")
    if any(r["best_history"] for r in runs):
        ax.legend(fontsize=8, ncol=max(1, len(runs) // 10))
    path = out / "runs_convergence.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    written.append(str(path))

    numbers = [r["summary"]["run"] for r in runs]
    scores = np.array([r["score"] for r in runs], dtype=np.float64)
    finite = np.isfinite(scores)
    fig = Figure(figsize=(max(6, 0.5 * len(runs) + 3), 4), layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.scatter(np.array(numbers)[finite], scores[finite], s=50, color="C0", zorder=3)
    if finite.any():
        ax.axhline(float(np.mean(scores[finite])), color="C3", lw=1.2, ls="--", label="mean")
        ax.legend(fontsize=8)
        low, high = float(np.min(scores[finite])), float(np.max(scores[finite]))
        pad = (high - low) * 0.2 or abs(high) * 0.05 or 1.0
        ax.set_ylim(low - pad, high + pad)
    ax.set_xticks(numbers)
    better = "lower" if direction == "min" else "higher"
    ax.set(xlabel="run", ylabel="best score", title=f"Best score per run ({better} is better)")
    path = out / "runs_best.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    written.append(str(path))
    return written
