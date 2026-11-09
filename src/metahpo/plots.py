"""Plots of an optimize() run for the `hpo_results` folder, drawn with matplotlib and seaborn.

matplotlib and seaborn are optional (`pip install meta-hpo[report]`); they are imported only here, inside
make_plots, so the rest of the library needs numpy alone. make_plots raises ImportError when they are
missing, and ResultWriter then notes in report.md that plots were skipped.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from metahpo.space import Categorical


def make_plots(
    state: Any, run_dir: Path, generations: list[dict[str, Any]], best_history: list[dict[str, Any]]
) -> list[str]:
    """Draws every plot of one run into run_dir/plots and returns the file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    trials = state.study.trials
    real = [t for t in trials if not t.cached]
    direction = state.direction
    written: list[str] = []

    def subplots(*args: Any, **kwargs: Any) -> Any:
        return plt.subplots(*args, layout="constrained", **kwargs)

    def save(fig: Any, name: str) -> None:
        path = out / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
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
        sns.histplot(scores[finite], bins=40, kde=True, ax=ax)
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
                sns.boxplot(x=labels, y=ok_scores, order=order, ax=ax, color="C0")
                sns.stripplot(x=labels, y=ok_scores, order=order, ax=ax, size=2, color="k", alpha=0.4)
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
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                xticklabels=labels,
                yticklabels=labels,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                ax=ax,
            )
            ax.set_title("Correlation of numeric parameters and score")
            save(fig, "08_correlation")

    durations = np.array([t.duration for t in real], dtype=np.float64)
    if len(real):
        fig, axes = subplots(1, 2, figsize=(12, 4))
        axes[0].plot(calls, durations, lw=0.8)
        axes[0].set(xlabel="real evaluation", ylabel="seconds", title="Duration per evaluation")
        sns.histplot(durations, bins=40, ax=axes[1])
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
