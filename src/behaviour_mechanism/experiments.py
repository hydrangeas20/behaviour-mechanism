"""Runners for the E3/E4 controls. Outputs are CSV + manifest under ``results/controls/``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from jax import random

from . import constructed
from .checkpoints import load_params
from .config import ExperimentConfig
from .controls import run_controls
from .patching import layer_pair_records, summarise
from .probes import offset_accuracy, positional_copy_rate
from .provenance import build_manifest, sha256_file, write_manifest
from .training import induction_eval_score


def _fresh(out: Path) -> None:
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a run; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)


def run_constructed(out: Path, n_pairs: int = 64) -> Path:
    """E4: historical probe, behavioural probes and E3 controls on each hand-set model."""
    import numpy as np

    _fresh(out)
    cfg = constructed.CFG
    probe_rows, control_rows = [], []
    for model in constructed.MODELS:
        params = constructed.build(model)
        s = summarise(layer_pair_records(params, cfg, random.PRNGKey(5), n_pairs), cfg.n_layers)
        hist8 = summarise(layer_pair_records(params, cfg, random.PRNGKey(10_000), 8), cfg.n_layers)
        rng = np.random.default_rng(0)
        from .model import build_model

        row: dict[str, Any] = {
            "model": model,
            "behaviour_accuracy": induction_eval_score(
                build_model(cfg), params, cfg, random.PRNGKey(0)
            ),
            "hist8_patch_effect_size": hist8["patch_effect_size"],
            **{f"probe_{k}": v for k, v in s.items()},
            **positional_copy_rate(params, cfg, 64, rng),
            **offset_accuracy(params, cfg, 16, 64, rng),
            **{f"expected_{k}": v for k, v in constructed.expected(model).items()},
        }
        probe_rows.append(row)
        for r in run_controls(params, cfg, n_pairs, 0):
            control_rows.append({"model": model, **r})
    pd.DataFrame(probe_rows).to_csv(out / "probe.csv", index=False)
    pd.DataFrame(control_rows).to_csv(out / "controls.csv", index=False)
    write_manifest(
        out / "manifest.json",
        build_manifest(
            "e4_constructed",
            n_pairs=n_pairs,
            config=cfg.to_dict(),
            constants={
                "BETA": constructed.BETA,
                "K_GATE": constructed.K_GATE,
                "GAMMA": constructed.GAMMA,
            },
            status="complete",
        ),
    )
    return out


def run_checkpoint_controls(
    cfg: ExperimentConfig, ckpt_dir: Path, out: Path, n_pairs: int, pair_seed: int
) -> Path:
    """E3 on saved checkpoints. The same pairs (``pair_seed``) are used at every step."""
    _fresh(out)
    files = sorted(ckpt_dir.glob("step_*.npz"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not files:
        raise FileNotFoundError(f"no step_*.npz in {ckpt_dir}")
    rows = []
    for f in files:
        step = int(re.findall(r"\d+", f.stem)[0])
        params = load_params(f)
        for r in run_controls(params, cfg, n_pairs, pair_seed):
            rows.append({"step": step, **r})
        print(f"[controls] {f.name} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "controls.csv", index=False)
    write_manifest(
        out / "manifest.json",
        build_manifest(
            "e3_checkpoint_controls",
            config_hash=cfg.config_hash(),
            n_pairs=n_pairs,
            pair_seed=pair_seed,
            checkpoint_dir=str(ckpt_dir),
            checkpoints={f.name: sha256_file(f) for f in files},
            checkpoint_manifest=str(ckpt_dir / "manifest.json"),
            status="complete",
        ),
    )
    return out


def run_stage1_positive_controls(out: Path, n_pairs: int = 64, seed: int = 0) -> Path:
    """Stage 1: analytic predictions vs the corrected measurement on constructed copiers."""
    import json

    from .measurement import METRIC_VERSION
    from .positive_controls import CORRUPTIONS, SITES, TOLERANCE, run

    _fresh(out)
    rows, legacy = run(n_pairs, seed)
    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(legacy).to_csv(out / "legacy_statistic.csv", index=False)
    summary = {
        "metric_version": METRIC_VERSION,
        "cells": len(df),
        "passed": int(df["pass"].sum()),
        "failed": int((~df["pass"]).sum()),
        "failures": df[~df["pass"]].to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_manifest(
        out / "manifest.json",
        build_manifest(
            "stage1_positive_controls",
            metric_version=METRIC_VERSION,
            n_pairs=n_pairs,
            pair_seed=seed,
            corruptions=list(CORRUPTIONS),
            sites=[s.name for s in SITES],
            tolerance=TOLERANCE,
            models=list(constructed.MODELS),
            status="complete",
        ),
    )
    return out
