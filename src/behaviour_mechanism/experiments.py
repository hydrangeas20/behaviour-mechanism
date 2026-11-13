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


def run_e2_baselines(out: Path, n_test: int = 2000, train_steps: int = 8000) -> Path:
    """Stage 2: shortcut baselines, reference solutions and leakage checks for E2."""
    import hashlib
    import json

    import numpy as np

    from . import e2 as e2mod

    _fresh(out)
    e2 = e2mod.E2Config()
    tables = e2mod.fit_tables(e2)
    sets = {
        "in_dist": e2mod.make_batch(e2mod.split_rng(e2, "test", 0), e2, n_test, frac_repeat=1.0),
        "heldout": e2mod.make_batch(
            e2mod.split_rng(e2, "test", 1), e2, n_test, e2.heldout_lengths, 1.0
        ),
        "extrapolation": e2mod.make_batch(
            e2mod.split_rng(e2, "test", 2), e2, n_test, e2.extrapolation_lengths, 1.0
        ),
    }
    rows = []
    for name, batch in sets.items():
        for r in e2mod.baseline_table(e2, batch, tables):
            rows.append({"test_set": name, "n_sequences": n_test, **r})
    pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False)

    # Leakage: exact sequences and exact chunks, evaluation sets vs every training batch.
    def chunk_hashes(b: e2mod.Batch) -> set[bytes]:
        hs = set()
        for i in np.flatnonzero(b.is_repeat):
            s, l = int(b.start[i]), int(b.length[i])
            hs.add(hashlib.blake2b(b.tokens[i, s : s + l].tobytes(), digest_size=12).digest())
        return hs

    eval_seq: set[bytes] = set()
    eval_chunks: set[bytes] = set()
    n_eval = 0
    for split in ("val", "test"):
        for idx in (0, 1, 2):
            lengths = (None, e2.heldout_lengths, e2.extrapolation_lengths)[idx]
            b = e2mod.make_batch(e2mod.split_rng(e2, split, idx), e2, n_test, lengths, 1.0)
            eval_seq |= e2mod.sequence_hashes(b.tokens)
            eval_chunks |= chunk_hashes(b)
            n_eval += n_test
    train_seq: set[bytes] = set()
    train_chunks: set[bytes] = set()
    train_lengths: set[int] = set()
    n_train = 0
    for step in range(train_steps):
        b = e2mod.make_batch(e2mod.split_rng(e2, "train", step), e2, 64)
        train_seq |= e2mod.sequence_hashes(b.tokens)
        train_chunks |= chunk_hashes(b)
        train_lengths |= set(b.length[b.is_repeat].tolist())
        n_train += 64
    leakage = {
        "generator_version": e2mod.GENERATOR_VERSION,
        "n_eval_sequences": n_eval,
        "n_train_sequences": n_train,
        "train_steps_checked": train_steps,
        "duplicate_train_sequences": n_train - len(train_seq),
        "eval_sequences_also_in_train": len(eval_seq & train_seq),
        "eval_chunks_also_in_train": len(eval_chunks & train_chunks),
        "train_lengths_seen": sorted(int(x) for x in train_lengths),
        "heldout_lengths_in_train": sorted(int(x) for x in train_lengths & set(e2.heldout_lengths)),
    }
    (out / "leakage.json").write_text(json.dumps(leakage, indent=2) + "\n")
    write_manifest(
        out / "manifest.json",
        build_manifest("e2_baselines", e2_config=e2.to_dict(), n_test=n_test, status="complete"),
    )
    return out
