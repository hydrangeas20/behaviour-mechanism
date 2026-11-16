"""E2 training and behavioural evaluation (exploratory; see docs/STAGE2_REPORT.md)."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .checkpoints import save_params
from .config import INDUCTION_LOSS_WEIGHT, ExperimentConfig
from .e2 import (
    GENERATOR_VERSION,
    Batch,
    E2Config,
    make_batch,
    make_gap_probe_batch,
    make_shuffled_repeat_batch,
    split_rng,
)
from .model import build_model, count_params, forward
from .provenance import build_manifest, sha256_file, write_manifest
from .training import make_optimizer, train_step

Params = Any
GAP_PROBES = (4, 8)


@partial(jax.jit, static_argnums=(0,))
def _logprobs(cfg: ExperimentConfig, params: Params, tokens: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.log_softmax(forward(params, cfg, tokens[:, :-1]), axis=-1)


def score_batch(params: Params, cfg: ExperimentConfig, batch: Batch) -> dict[str, np.ndarray]:
    """Per-position correctness and target log-probability, shape (B, T-1)."""
    lp = np.asarray(_logprobs(cfg, params, jnp.asarray(batch.tokens)))
    tgt = batch.tokens[:, 1:]
    return {
        "correct": lp.argmax(-1) == tgt,
        "logprob": np.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0],
        "pred": lp.argmax(-1),
    }


def _acc(sc: dict[str, np.ndarray], sel: np.ndarray) -> tuple[float, float]:
    if not sel.any():
        return float("nan"), float("nan")
    return float(sc["correct"][sel].mean()), float(sc["logprob"][sel].mean())


def eval_sets(e2: E2Config, split: str, n: int) -> dict[str, Batch]:
    """Fixed evaluation batches for ``split`` (``val`` or ``test``); same for every
    checkpoint. Index 0 = in-distribution, then held-out, extrapolation, probes."""
    return {
        "in_dist": make_batch(split_rng(e2, split, 0), e2, n, frac_repeat=1.0),
        "heldout": make_batch(split_rng(e2, split, 1), e2, n, e2.heldout_lengths, 1.0),
        "extrapolation": make_batch(split_rng(e2, split, 2), e2, n, e2.extrapolation_lengths, 1.0),
        "random": make_batch(split_rng(e2, split, 3), e2, n, frac_repeat=0.0),
        "shuffled": make_shuffled_repeat_batch(split_rng(e2, split, 4), e2, n),
        **{
            f"gap{g}": make_gap_probe_batch(split_rng(e2, split, 10 + g), e2, n, g)
            for g in GAP_PROBES
        },
    }


def evaluate_behaviour(
    params: Params, cfg: ExperimentConfig, e2: E2Config, sets: dict[str, Batch]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, b in sets.items():
        sc = score_batch(params, cfg, b)
        sel = b.scored[:, 1:]
        if name == "random":
            # no scored positions: report the loss and how often the model "copies" the
            # successor of the last earlier occurrence of the current token.
            out["random_nll"] = float(-sc["logprob"].mean())
            continue
        acc, lp = _acc(sc, sel)
        out[f"{name}_acc"], out[f"{name}_logprob"] = acc, lp
        if name == "shuffled":
            # how often the model outputs the token that an in-order repeat would have had
            b_ord = b.tokens.copy()
            for i in range(b.tokens.shape[0]):
                s, l = int(b.start[i]), int(b.length[i])
                b_ord[i, s + l : s + 2 * l] = b.tokens[i, s : s + l]
            out["shuffled_inorder_copy_rate"] = float((sc["pred"] == b_ord[:, 1:])[sel].mean())
        if name in ("in_dist", "heldout", "extrapolation"):
            by_len = {}
            for l in sorted(set(b.length.tolist())):
                m = sel & (b.length == l)[:, None]
                by_len[int(l)] = _acc(sc, m)[0]
            out[f"{name}_acc_by_length"] = by_len
        if name == "in_dist":
            # depth = index of the target inside the repeat (1 .. L-1)
            pos = np.arange(1, b.tokens.shape[1])[None, :]
            depth = pos - (b.start + b.length)[:, None]
            out["in_dist_acc_by_depth"] = {
                lab: _acc(sc, sel & (depth >= lo) & (depth <= hi))[0]
                for lab, lo, hi in (("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 99))
            }
            # all-position raw loss on a mixed batch is recorded by the training loop
    return out


def train_e2(
    cfg: ExperimentConfig,
    e2: E2Config,
    seed: int,
    out: Path,
    ckpt_every: int = 250,
    n_val: int = 512,
    label: str = "",
) -> Path:
    """Train one model on E2. Writes ``metrics.jsonl``, ``manifest.json`` and checkpoints
    under ``out`` (checkpoints in ``out/checkpoints``)."""
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a run; choose a new --out")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    params = model.init(init_rng, jnp.ones((1, cfg.seq_len), dtype=jnp.int32))["params"]
    tx = make_optimizer(cfg)
    opt_state = tx.init(params)
    sets = eval_sets(e2, "val", n_val)
    mixed_val = make_batch(split_rng(e2, "val", 99), e2, n_val)
    manifest = build_manifest(
        "e2_training",
        label=label,
        experiment_id=f"e2-{cfg.config_hash()}-d{e2.data_seed}-s{seed}",
        model_config=cfg.to_dict(),
        model_config_hash=cfg.config_hash(),
        e2_config=e2.to_dict(),
        generator_version=GENERATOR_VERSION,
        seed=seed,
        n_params=count_params(params),
        optimizer={
            "name": "adamw",
            "peak_lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_steps": cfg.warmup_steps,
            "schedule": "linear warmup, cosine decay to 0 over total_steps",
            "scored_position_loss_weight": INDUCTION_LOSS_WEIGHT,
        },
        batch_size=cfg.batch_size,
        total_steps=cfg.total_steps,
        checkpoint_every=ckpt_every,
        n_val_sequences=n_val,
        status="running",
    )
    write_manifest(out / "manifest.json", manifest)
    t0 = time.time()
    hashes: dict[str, str] = {}
    ema: float | None = None
    with (out / "metrics.jsonl").open("a") as fh:
        for step in range(cfg.total_steps + 1):
            if step % ckpt_every == 0 or step == cfg.total_steps:
                rec: dict[str, Any] = {
                    "step": step,
                    "seed": seed,
                    "train_loss_ema": ema,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                rec.update(evaluate_behaviour(params, cfg, e2, sets))
                sc = score_batch(params, cfg, mixed_val)
                rec["val_nll_all_positions"] = float(-sc["logprob"].mean())
                path = out / "checkpoints" / f"step_{step}.npz"
                save_params(path, params)
                hashes[path.name] = sha256_file(path)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"[e2 seed {seed}] step {step:5d} in_dist={rec['in_dist_acc']:.3f} "
                    f"heldout={rec['heldout_acc']:.3f} nll={rec['val_nll_all_positions']:.3f}",
                    flush=True,
                )
            if step == cfg.total_steps:
                break
            b = make_batch(split_rng(e2, "train", step), e2, cfg.batch_size)
            params, opt_state, loss, _ = train_step(
                model, params, opt_state, tx, jnp.asarray(b.tokens), jnp.asarray(b.scored)
            )
            ema = float(loss) if ema is None else 0.95 * ema + 0.05 * float(loss)
    manifest.update(status="complete", seconds=round(time.time() - t0, 1), checkpoint_sha256=hashes)
    write_manifest(out / "manifest.json", manifest)
    return out
