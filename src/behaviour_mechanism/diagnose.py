"""Diagnostic re-run: train with the historical code and measure more at each checkpoint.

At every checkpoint this records

* the historical behavioural score and the historical patching statistic, computed with
  the historical RNG scheme (``PRNGKey(seed + 10_000)``, 8 pairs) -- directly comparable
  with ``results/mechanistic.csv``;
* the same statistic decomposed (defined fraction, conditional recovery, gaps) on a
  larger independent sample of pairs, so estimator noise can be separated from change;
* added behavioural probes: positional copy rate on non-repeating sequences, and
  induction accuracy when the chunk repeats at offsets other than the training offset.

Rows are appended to ``<out>/checkpoints.jsonl`` as they are produced, so an
interrupted run keeps everything measured so far.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from jax import random

from .config import ExperimentConfig
from .model import count_params, init_params
from .patching import head_pair_records, layer_pair_records, legacy_head_effects, summarise
from .probes import offset_accuracy, positional_copy_rate
from .provenance import build_manifest, write_manifest
from .training import run_training

PROBE_OFFSETS = (8, 16, 24, 32)


def make_checkpoint_fn(
    cfg: ExperimentConfig, seed: int, n_large: int, n_probe_seq: int, n_head_pairs: int
) -> Callable[[int, Any], dict[str, Any]]:
    legacy_rng = [random.PRNGKey(seed + 10_000)]

    def fn(step: int, params: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        legacy_rng[0], patch_rng, _lens_rng = random.split(legacy_rng[0], 3)
        legacy = summarise(layer_pair_records(params, cfg, patch_rng, 8), cfg.n_layers)
        rec["hist_patch_effect_size"] = legacy["patch_effect_size"]
        rec["hist_patch_best_layer"] = legacy["patch_best_layer"]
        rec["hist_n_defined"] = legacy["n_defined"]

        big_key = random.fold_in(random.PRNGKey(seed + 70_000), step)
        big = summarise(layer_pair_records(params, cfg, big_key, n_large), cfg.n_layers)
        rec.update({f"large_{k}": v for k, v in big.items()})

        if n_head_pairs:
            hk = random.fold_in(random.PRNGKey(seed + 80_000), step)
            hrecs = head_pair_records(params, cfg, hk, n_head_pairs)
            heads = legacy_head_effects(hrecs, cfg.n_layers, cfg.n_heads)
            li, hi = np.unravel_index(int(np.argmax(heads)), heads.shape)
            rec["head_best"] = f"L{li}H{hi}"
            rec["head_best_effect"] = float(heads[li, hi])
            rec["head_n_defined"] = sum(r.defined for r in hrecs)

        rng_np = np.random.default_rng([seed, step, 1])
        rec.update(positional_copy_rate(params, cfg, n_probe_seq, rng_np))
        for off in PROBE_OFFSETS:
            if off <= cfg.seq_len // 2:
                rec.update(offset_accuracy(params, cfg, off, n_probe_seq, rng_np))
        return rec

    return fn


def run_diagnostic(
    cfg: ExperimentConfig,
    seed: int,
    out: Path,
    stop_at: int | None,
    n_large: int = 64,
    n_probe_seq: int = 128,
    n_head_pairs: int = 16,
    label: str = "",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "checkpoints.jsonl"
    if rows_path.exists():
        raise FileExistsError(f"{rows_path} exists; choose a new --out to keep runs immutable")
    manifest = build_manifest(
        "diagnostic_run",
        label=label,
        experiment_id=f"diag-{cfg.config_hash()}-s{seed}-to{stop_at or cfg.total_steps}",
        config=cfg.to_dict(),
        config_hash=cfg.config_hash(),
        seed=seed,
        stop_at=stop_at,
        n_params=count_params(init_params(cfg, random.PRNGKey(0))),
        measurement={
            "historical_patch_pairs": 8,
            "large_patch_pairs": n_large,
            "probe_sequences": n_probe_seq,
            "head_pairs": n_head_pairs,
            "probe_offsets": [o for o in PROBE_OFFSETS if o <= cfg.seq_len // 2],
        },
        status="running",
    )
    write_manifest(out / "manifest.json", manifest)
    fn = make_checkpoint_fn(cfg, seed, n_large, n_probe_seq, n_head_pairs)
    t0 = time.time()
    with rows_path.open("a") as fh:

        def write(rec: dict[str, Any]) -> None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        res = run_training(
            cfg,
            seed,
            on_checkpoint=fn,
            on_record=write,
            log=lambda msg: print(msg, flush=True),
            log_every=250,
            stop_at=stop_at,
        )
    manifest.update(status="complete", seconds=round(time.time() - t0, 1), n_rows=len(res.records))
    write_manifest(out / "manifest.json", manifest)
    return rows_path
