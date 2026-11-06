"""Figures generated from recorded CSV/JSONL outputs only.

Historical figures in ``figures/`` are never overwritten; regenerated and new figures go
to ``figures/derived/``. Colours: validated categorical palette slots 1-4 (seed lines are
also direct-labelled, so identity never relies on colour alone).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .artifacts import expected_raw_loss, write_derived

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def _style(ax: Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def historical_figures(repo: Path) -> list[Path]:
    res = repo / "results"
    out_dir = repo / "figures" / "derived"
    mg = pd.read_csv(res / "merged.csv")
    derived = write_derived(repo)
    paths = []

    # 1. Per-seed trajectories: accuracy, legacy patching statistic, raw loss
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.4), sharex=True)
    ideal = derived["ideal_raw_loss"]
    for i, (seed, g) in enumerate(mg.groupby("seed")):
        g = g.sort_values("step")
        c = SERIES[i]
        axes[0].plot(g.step, g.induction_eval_score, color=c, lw=1.6, label=f"seed {seed}")
        axes[1].plot(g.step, g.patch_effect_size, color=c, lw=1.6, marker="o", ms=2.5)
        axes[2].plot(g.step, g.raw_loss, color=c, lw=1.6)
    axes[0].set_ylabel("Accuracy on repeated half\n(clean sequences)")
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="center right")
    axes[1].set_ylabel("Historical patching statistic\n(best layer, 8 pairs)")
    for k in range(9):
        axes[1].axhline(k / 8, color=GRID, lw=0.8, zorder=0)
    axes[1].text(
        8000, 0.5, "grid: multiples of 1/8", ha="right", va="bottom", fontsize=7, color=INK2
    )
    axes[2].set_ylabel("Raw training loss (nats, EMA)")
    for key, lab in (
        ("ungated_positional_copy", "ideal: ungated copy"),
        ("gated_copy", "ideal: gated copy"),
    ):
        axes[2].axhline(ideal[key], color=INK2, ls="--", lw=0.9)
        axes[2].text(8000, ideal[key] + 0.02, lab, ha="right", va="bottom", fontsize=7, color=INK2)
    axes[2].set_xlabel("Training step")
    axes[2].set_xscale("symlog", linthresh=500)
    axes[2].set_xlim(0, 8300)
    axes[2].set_xticks([0, 250, 500, 1000, 1500, 2000, 4000, 8000])
    axes[2].set_xticklabels(["0", "250", "500", "1k", "1.5k", "2k", "4k", "8k"])
    for ax in axes:
        _style(ax)
    fig.suptitle("Historical run (results/merged.csv): 4 seeds", fontsize=10, color=INK)
    paths.append(_save(fig, out_dir / "historical_trajectories.png"))

    # 2. Quantisation of the patching statistic
    m = mg[mg.step >= 300].patch_effect_size.to_numpy() * 8
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(
        m,
        bins=[float(x) for x in np.arange(-0.0625, 8.2, 0.125)],
        color=SERIES[0],
        edgecolor="white",
        lw=0.5,
    )
    ax.set_xlabel("8 x historical patching statistic (checkpoints at step >= 300, 4 seeds)")
    ax.set_ylabel("Checkpoints")
    ax.set_xticks(range(9))
    _style(ax)
    paths.append(_save(fig, out_dir / "patch_statistic_quantisation.png"))

    # 3. Threshold sweep
    sw = pd.DataFrame(derived["sweep"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(sw.theta, sw.T_b, color=SERIES[0], marker="o", ms=4, lw=1.6, label="T_b (accuracy)")
    ax.plot(
        sw.theta,
        sw.T_m,
        color=SERIES[1],
        marker="s",
        ms=4,
        lw=1.6,
        label="T_m (patching statistic)",
    )
    ax.set_xlabel("Threshold theta (all seeds >= theta for 3 consecutive checkpoints)")
    ax.set_ylabel("First stable checkpoint")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    paths.append(_save(fig, out_dir / "sustained_ceiling_sweep.png"))
    return paths


def load_diagnostics(repo: Path) -> pd.DataFrame:
    rows = []
    for run in sorted((repo / "results" / "diagnostics").glob("*/checkpoints.jsonl")):
        manifest = json.loads((run.parent / "manifest.json").read_text())
        for line in run.read_text().splitlines():
            r = json.loads(line)
            r["run"] = run.parent.name
            r["run_status"] = manifest.get("status")
            r["d_model"] = manifest["config"]["d_model"]
            r["n_layers"] = manifest["config"]["n_layers"]
            r["frac_induction_sequences"] = manifest["config"]["frac_induction_sequences"]
            rows.append(r)
    return pd.DataFrame(rows)


def diagnostics_report(repo: Path) -> list[Path]:
    df = load_diagnostics(repo)
    if df.empty:
        raise FileNotFoundError("no results/diagnostics/*/checkpoints.jsonl found")
    out = repo / "results" / "diagnostics" / "summary.csv"
    keep = [
        "run",
        "run_status",
        "seed",
        "step",
        "d_model",
        "n_layers",
        "frac_induction_sequences",
        "induction_eval_score",
        "raw_loss",
        "hist_patch_effect_size",
        "hist_n_defined",
        "large_n_pairs",
        "large_n_defined",
        "large_defined_fraction",
        "large_patch_effect_size",
        "large_conditional_best",
        "large_conditional_L0",
        "large_patch_best_layer",
        "large_gap_median",
        "head_best",
        "head_best_effect",
        "random_copy_rate",
        "offset8_acc",
        "offset16_acc",
        "offset24_acc",
        "offset32_acc",
    ]
    df[[c for c in keep if c in df.columns]].to_csv(out, index=False)
    paths = [out]

    runs = sorted(df.run.unique())
    panels = [
        ("induction_eval_score", "Accuracy, clean\n(historical metric)"),
        ("hist_patch_effect_size", "Patching statistic\n(historical, 8 pairs)"),
        ("large_defined_fraction", "Defined fraction\n(64 pairs, |gap| >= 0.05)"),
        ("large_conditional_best", "Recovery | defined\n(best layer, 64 pairs)"),
        ("random_copy_logprob", "log p(half-back token)\non non-repeating seqs"),
        ("offset16_acc", "Accuracy, repeat at\noffset 16 (never trained)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.0 * len(panels)), sharex=True)
    for i, run in enumerate(runs):
        g = df[df.run == run].sort_values("step")
        for ax, (col, _) in zip(axes, panels, strict=True):
            if col in g:
                ax.plot(
                    g.step,
                    g[col],
                    color=SERIES[i % len(SERIES)],
                    lw=1.5,
                    marker="o",
                    ms=2,
                    label=run,
                )
    for ax, (col, lab) in zip(axes, panels, strict=True):
        ax.set_ylabel(lab, fontsize=8)
        if col == "random_copy_logprob":
            ax.set_ylim(-5.2, 0.2)
            ax.axhline(-np.log(100), color=INK2, ls="--", lw=0.8)
        else:
            ax.set_ylim(-0.05, 1.08)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="center right")
    axes[-1].set_xlabel("Training step")
    fig.suptitle("CPU diagnostic re-runs (results/diagnostics/)", fontsize=10, color=INK)
    paths.append(_save(fig, repo / "figures" / "derived" / "diagnostics_trajectories.png"))
    return paths


__all__ = ["diagnostics_report", "expected_raw_loss", "historical_figures"]
