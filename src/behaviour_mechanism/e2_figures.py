"""Stage 2 figures and summary tables, generated only from saved results in
``results/stage2/``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .figures import INK, INK2, SERIES, _save, _style

CORRUPTION_LABEL = {
    "earlier_occurrence": "earlier occurrence",
    "copy_source": "copy source",
    "query_token": "query token",
    "random_repeat": "random repeat (legacy analogue)",
}


def _runs(stage2: Path) -> list[Path]:
    return sorted(p.parent for p in stage2.glob("*/behaviour.jsonl"))


def _behaviour(run: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in (run / "behaviour.jsonl").read_text().splitlines()]
    return pd.DataFrame(rows).sort_values("step")


def first_sustained(
    steps: list[int], values: list[float], thr: float, window: int = 2
) -> int | None:
    """First step from which ``values >= thr`` holds for ``window`` consecutive checkpoints."""
    for i in range(len(steps) - window + 1):
        if all(v is not None and v == v and v >= thr for v in values[i : i + window]):
            return int(steps[i])
    return None


def settled_step(steps: list[int], values: list[float], tol: float = 0.05) -> int | None:
    """First checkpoint from which every later value stays within ``tol`` of the final
    value. Descriptive only: it depends on ``tol`` and on the checkpoint grid."""
    final = values[-1]
    for i in range(len(steps)):
        if all(v == v and abs(v - final) <= tol for v in values[i:]):
            return int(steps[i])
    return None


def settling_table(run: Path) -> pd.DataFrame:
    b = _behaviour(run)
    mech = pd.read_csv(run / "mechanism.csv")
    leg = pd.read_csv(run / "legacy_statistic.csv")
    steps = b.step.tolist()
    rows: list[dict[str, object]] = []

    def add(kind: str, name: str, st: list[int], vals: list[float]) -> None:
        rows.append(
            {
                "kind": kind,
                "quantity": name,
                "final_value": round(vals[-1], 3),
                "settled_step": settled_step(st, vals),
            }
        )

    for col in ("in_dist_acc", "heldout_acc", "extrapolation_acc", "gap8_acc"):
        add("behaviour", col, steps, b[col].tolist())
    for d in ("1", "2-3"):
        add(
            "behaviour",
            f"in_dist_acc_depth_{d}",
            steps,
            [x[d] for x in b.in_dist_acc_by_depth.tolist()],
        )
    for corr in CORRUPTION_LABEL:
        g = leg[leg.corruption == corr].sort_values("step")
        add("legacy", f"legacy|{corr}", g.step.tolist(), g.legacy_patch_effect_size_v1.tolist())
        add(
            "defined_fraction",
            f"defined|{corr}",
            g.step.tolist(),
            (g.n_defined / g.n_attempted).tolist(),
        )
    m = mech[mech.component.isin(["attn_out", "mlp_out"])]
    for key, g in m.groupby(["corruption", "site", "direction"]):
        corr, site, direction = (str(k) for k in key)
        g = g.sort_values("step")
        vals = g.conditional_effect_mean.tolist()
        if any(v != v for v in vals):
            continue
        add(f"conditional_{direction}", f"{corr}|{site}", g.step.tolist(), vals)
    return pd.DataFrame(rows)


def e2_report(repo: Path) -> list[Path]:
    stage2 = repo / "results" / "stage2"
    runs = _runs(stage2)
    if not runs:
        raise FileNotFoundError("no results/stage2/*/behaviour.jsonl")
    base_df = pd.read_csv(stage2 / "baselines" / "baselines.csv")
    base = base_df[base_df.test_set == "in_dist"].set_index("label").accuracy
    out: list[Path] = []
    fig_dir = repo / "figures" / "stage2"

    # ---- Figure 1: behaviour per run
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 3.1 * len(runs)), squeeze=False)
    series = [
        ("in_dist_acc", "in-distribution lengths"),
        ("heldout_acc", "held-out lengths"),
        ("extrapolation_acc", "extrapolation lengths"),
        ("gap8_acc", "non-adjacent (gap 8)"),
    ]
    for ax, run in zip(axes[:, 0], runs, strict=True):
        b = _behaviour(run)
        for i, (col, lab) in enumerate(series):
            ax.plot(b.step, b[col], color=SERIES[i], lw=1.6, marker="o", ms=2.5, label=lab)
        for key, lab in (
            ("fixed_offset_best", "best fixed offset"),
            ("reference_a", "reference A"),
            ("reference_b", "reference B"),
        ):
            ax.axhline(base[key], color=INK2, ls="--", lw=0.8)
            ax.text(b.step.max(), base[key] + 0.012, lab, ha="right", fontsize=7, color=INK2)
        ax.set_ylim(-0.03, 1.06)
        ax.set_ylabel("Accuracy on scored positions")
        ax.set_title(run.name, fontsize=9, color=INK, loc="left")
        _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="center right")
    axes[-1, 0].set_xlabel("Training step")
    fig.suptitle("E2 (exploratory): behaviour on fixed test batches", fontsize=10, color=INK)
    out.append(_save(fig, fig_dir / "e2_behaviour.png"))

    # ---- Figure 2: mechanism per run
    rows = []
    for run in runs:
        if not (run / "mechanism.csv").exists():
            continue
        b = _behaviour(run)
        mech = pd.read_csv(run / "mechanism.csv")
        leg = pd.read_csv(run / "legacy_statistic.csv")
        n_layers = int(mech.layer.max()) + 1
        fig, axes = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
        axes[0].plot(
            b.step,
            b.in_dist_acc,
            color=SERIES[0],
            lw=1.6,
            marker="o",
            ms=2.5,
            label="in-distribution",
        )
        axes[0].plot(
            b.step,
            b.heldout_acc,
            color=SERIES[1],
            lw=1.6,
            marker="o",
            ms=2.5,
            label="held-out lengths",
        )
        axes[0].set_ylabel("Accuracy")
        for i, corr in enumerate(CORRUPTION_LABEL):
            g = leg[leg.corruption == corr]
            axes[1].plot(
                g.step,
                g.legacy_patch_effect_size_v1,
                color=SERIES[i],
                lw=1.5,
                marker="o",
                ms=2.5,
                label=CORRUPTION_LABEL[corr],
            )
            axes[2].plot(
                g.step,
                g.n_defined / g.n_attempted,
                color=SERIES[i],
                lw=1.5,
                marker="o",
                ms=2.5,
                label=CORRUPTION_LABEL[corr],
            )
        axes[1].set_ylabel("Legacy statistic\n(zero-filled, max over layers)")
        axes[2].set_ylabel("Defined fraction")
        for ax, corr in ((axes[3], "earlier_occurrence"), (axes[4], "copy_source")):
            m = mech[
                (mech.corruption == corr)
                & (mech.component == "attn_out")
                & (mech.direction == "denoise")
            ]
            for l in range(n_layers):
                g = m[m.layer == l]
                ax.plot(
                    g.step,
                    g.conditional_effect_mean,
                    color=SERIES[l],
                    lw=1.5,
                    marker="o",
                    ms=2.5,
                    label=f"attention layer {l}",
                )
            ax.set_ylabel(f"Denoising effect | defined\n({CORRUPTION_LABEL[corr]})")
        for ax in axes:
            ax.set_ylim(-0.3, 1.3) if ax in (axes[3], axes[4]) else ax.set_ylim(-0.05, 1.08)
            ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0))
            _style(ax)
        axes[-1].set_xlabel("Training step")
        fig.suptitle(
            f"E2 (exploratory): {run.name}, 128 fixed pairs per corruption", fontsize=10, color=INK
        )
        out.append(_save(fig, fig_dir / f"e2_mechanism_{run.name}.png"))

        sp = stage2 / f"settling_{run.name}.csv"
        settling_table(run).to_csv(sp, index=False)
        out.append(sp)
        steps = b.step.tolist()
        row = {
            "run": run.name,
            "in_dist_acc>=0.9": first_sustained(steps, b.in_dist_acc.tolist(), 0.9),
            "heldout_acc>=0.9": first_sustained(steps, b.heldout_acc.tolist(), 0.9),
        }
        for corr in CORRUPTION_LABEL:
            g = leg[leg.corruption == corr].sort_values("step")
            row[f"legacy[{corr}]>=0.8"] = first_sustained(
                g.step.tolist(), g.legacy_patch_effect_size_v1.tolist(), 0.8
            )
            row[f"defined[{corr}]>=0.95"] = first_sustained(
                g.step.tolist(), (g.n_defined / g.n_attempted).tolist(), 0.95
            )
        rows.append(row)
    if rows:
        p = stage2 / "transitions.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        out.append(p)
    return out
