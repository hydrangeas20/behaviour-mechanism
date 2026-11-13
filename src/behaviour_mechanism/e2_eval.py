"""Stage 2 evaluation of saved E2 checkpoints, and positive-control calibration.

The same protocol is applied to every saved checkpoint: fixed test batches for behaviour,
and one fixed set of clean/corrupted pairs per corruption for the mechanism measurement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import constructed_induction as ci
from .checkpoints import load_params
from .config import ExperimentConfig
from .e2 import E2_CORRUPTIONS, E2Config, make_e2_pairs, split_rng
from .e2_run import eval_sets, evaluate_behaviour
from .measurement import (
    LEGACY_NAME,
    METRIC_VERSION,
    PairSet,
    legacy_statistic,
    measure,
    standard_sites,
)
from .provenance import build_manifest, sha256_file, write_manifest

PAIR_STREAM = 100  # test-split stream index offset for mechanism pairs


def _configs(run_dir: Path) -> tuple[ExperimentConfig, E2Config, dict[str, Any]]:
    man = json.loads((run_dir / "manifest.json").read_text())
    mc = dict(man["model_config"])
    mc["seeds"] = tuple(mc["seeds"])
    ec = {k: v for k, v in man["e2_config"].items() if k != "generator_version"}
    for k in ("heldout_lengths", "extrapolation_lengths"):
        ec[k] = tuple(ec[k])
    return ExperimentConfig(**mc), E2Config(**ec), man


def fixed_pairs(e2: E2Config, n: int, distinct: bool = False) -> dict[str, PairSet]:
    out = {}
    for i, corr in enumerate(E2_CORRUPTIONS):
        d = make_e2_pairs(split_rng(e2, "test", PAIR_STREAM + i), e2, n, corr, distinct=distinct)
        out[corr] = PairSet(d["clean"], d["corrupt"], d["target"], d["read_pos"], corr)
    return out


def evaluate_run(run_dir: Path, out: Path, n_pairs: int = 128, n_test: int = 512) -> Path:
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds an evaluation; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    cfg, e2, train_manifest = _configs(run_dir)
    files = sorted(
        (run_dir / "checkpoints").glob("step_*.npz"),
        key=lambda p: int(re.findall(r"\d+", p.stem)[0]),
    )
    if not files:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    sets = eval_sets(e2, "test", n_test)
    pairs = fixed_pairs(e2, n_pairs)
    sites = standard_sites(cfg, heads=True)
    beh_rows, mech_rows, leg_rows = [], [], []
    raw_gap = np.full((len(files), len(E2_CORRUPTIONS), n_pairs), np.nan, dtype=np.float32)
    raw_eff = np.full(
        (len(files), len(E2_CORRUPTIONS), len(sites), 2, n_pairs), np.nan, dtype=np.float32
    )
    steps = []
    for fi, f in enumerate(files):
        step = int(re.findall(r"\d+", f.stem)[0])
        steps.append(step)
        params = load_params(f)
        beh_rows.append({"step": step, **evaluate_behaviour(params, cfg, e2, sets)})
        for ci_, corr in enumerate(E2_CORRUPTIONS):
            res = measure(params, cfg, pairs[corr], sites)
            leg = legacy_statistic(res, cfg.n_layers)
            leg_rows.append(
                {
                    "step": step,
                    "corruption": corr,
                    LEGACY_NAME: leg["value"],
                    "best_layer": leg["best_layer"],
                    **{f"layer_{l}": v for l, v in enumerate(leg["per_layer"])},
                    "n_attempted": res[0].n_attempted,
                    "n_defined": res[0].n_defined,
                }
            )
            raw_gap[fi, ci_] = res[0].per_pair_gap
            for r in res:
                mech_rows.append({"step": step, **r.to_dict(per_pair=False)})
                si = next(i for i, s in enumerate(sites) if s.name == r.site)
                di = 0 if r.direction == "denoise" else 1
                raw_eff[fi, ci_, si, di] = [np.nan if e is None else e for e in r.per_pair_effect]
        print(f"[e2-eval] step {step} done", flush=True)
    with (out / "behaviour.jsonl").open("w") as fh:
        for brow in beh_rows:
            fh.write(json.dumps(brow) + "\n")
    pd.DataFrame(mech_rows).to_csv(out / "mechanism.csv", index=False)
    pd.DataFrame(leg_rows).to_csv(out / "legacy_statistic.csv", index=False)
    np.savez_compressed(
        out / "per_pair.npz",
        steps=np.asarray(steps),
        corruptions=np.asarray(E2_CORRUPTIONS),
        sites=np.asarray([s.name for s in sites]),
        directions=np.asarray(["denoise", "noise"]),
        gap=raw_gap,
        effect=raw_eff,  # NaN marks an undefined pair (|gap| below the threshold)
    )
    write_manifest(
        out / "manifest.json",
        build_manifest(
            "e2_evaluation",
            metric_version=METRIC_VERSION,
            run_dir=run_dir.name,
            training_experiment_id=train_manifest.get("experiment_id"),
            training_git=train_manifest.get("git"),
            training_seconds=train_manifest.get("seconds"),
            n_pairs=n_pairs,
            n_test_sequences=n_test,
            pair_stream=f"test split, index {PAIR_STREAM}+corruption",
            checkpoints={f.name: sha256_file(f) for f in files},
            status="complete",
        ),
    )
    return out


def calibrate(out: Path, n_pairs: int = 128) -> Path:
    """Run the E2 measurement on the constructed induction model and compare each cell
    with its analytic prediction (distinct pairs), and record how much natural pairs, which
    contain chance duplicate tokens, deviate."""
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a calibration; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    e2 = E2Config()
    params = ci.build()
    sites = standard_sites(ci.CFG, heads=True)
    rows, leg_rows = [], []
    for kind, distinct in (("distinct", True), ("natural", False)):
        for corr, pairs in fixed_pairs(e2, n_pairs, distinct=distinct).items():
            res = measure(params, ci.CFG, pairs, sites)
            leg = legacy_statistic(res, ci.CFG.n_layers)
            leg_rows.append(
                {
                    "pairs": kind,
                    "corruption": corr,
                    LEGACY_NAME: leg["value"],
                    "layer_0": leg["per_layer"][0],
                    "layer_1": leg["per_layer"][1],
                }
            )
            for r in res:
                pred = ci.predicted_effect(corr, r.site)
                obs = r.conditional_effect_mean
                rows.append(
                    {
                        "pairs": kind,
                        **r.to_dict(per_pair=False),
                        "predicted_effect": pred,
                        "abs_error": None if obs is None else abs(obs - pred),
                        "pass": bool(
                            obs is not None and abs(obs - pred) <= 0.02 and r.n_undefined == 0
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(leg_rows).to_csv(out / "legacy_statistic.csv", index=False)
    sets = eval_sets(e2, "test", 512)
    beh = evaluate_behaviour(params, ci.CFG, e2, sets)
    summary = {
        "metric_version": METRIC_VERSION,
        "model": "constructed two-layer induction model",
        "behaviour": beh,
        **{
            kind: {
                "cells": int((df.pairs == kind).sum()),
                "passed": int(df[df.pairs == kind]["pass"].sum()),
                "max_abs_error": float(df[df.pairs == kind].abs_error.max()),
                "min_defined_fraction": float(df[df.pairs == kind].defined_fraction.min()),
            }
            for kind in ("distinct", "natural")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        out / "manifest.json",
        build_manifest(
            "e2_calibration",
            metric_version=METRIC_VERSION,
            n_pairs=n_pairs,
            e2_config=e2.to_dict(),
            status="complete",
        ),
    )
    return out
