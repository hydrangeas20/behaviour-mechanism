"""Training loop and behavioural metric (faithful to the notebook).

RNG consumption matches the notebook's ``run_training`` exactly: ``PRNGKey(seed)`` ->
split for init; at each checkpoint split an eval key *before* the training step; one
split per training batch. Re-running a seed therefore draws the same data, although
float results differ across hardware (the historical run used a Colab T4 GPU).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from .config import INDUCTION_LOSS_WEIGHT, ExperimentConfig
from .data import make_induction_batch
from .model import TinyGPT2, build_model

Params = Any
Array = jnp.ndarray


def make_optimizer(cfg: ExperimentConfig) -> optax.GradientTransformation:
    warmup = min(cfg.warmup_steps, max(cfg.total_steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=warmup,
        decay_steps=max(cfg.total_steps, warmup + 1),
    )
    return optax.adamw(schedule, weight_decay=cfg.weight_decay)


def weighted_nll(params: Params, model: TinyGPT2, tokens: Array, target_positions: Array) -> Array:
    """Training objective: next-token NLL with induction positions up-weighted x3."""
    logits: Array = model.apply({"params": params}, tokens[:, :-1])  # type: ignore[assignment]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, tokens[:, 1:, None], axis=-1).squeeze(-1)
    weights = jnp.where(target_positions[:, 1:], INDUCTION_LOSS_WEIGHT, 1.0)
    return (nll * weights).sum() / weights.sum()


def raw_nll(params: Params, model: TinyGPT2, tokens: Array) -> Array:
    logits: Array = model.apply({"params": params}, tokens[:, :-1])  # type: ignore[assignment]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, tokens[:, 1:, None], axis=-1).squeeze(-1).mean()


@partial(jax.jit, static_argnums=(0, 3))
def train_step(
    model: TinyGPT2,
    params: Params,
    opt_state: Any,
    tx: optax.GradientTransformation,
    tokens: Array,
    target_positions: Array,
) -> tuple[Params, Any, Array, Array]:
    loss, grads = jax.value_and_grad(weighted_nll)(params, model, tokens, target_positions)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # NOTE (historical behaviour, kept): raw loss is evaluated with the *updated* params on
    # the batch just trained on, while the weighted loss uses the pre-update params.
    return params, opt_state, loss, raw_nll(params, model, tokens)


@partial(jax.jit, static_argnums=(0,))
def _argmax_preds(model: TinyGPT2, params: Params, tokens: Array) -> Array:
    logits: Array = model.apply({"params": params}, tokens[:, :-1])  # type: ignore[assignment]
    return jnp.argmax(logits, axis=-1)


def induction_eval_score(
    model: TinyGPT2, params: Params, cfg: ExperimentConfig, rng: Array
) -> float:
    """Behavioural metric (vectorised; numerically identical to the notebook loop).

    Argmax next-token accuracy over every position of the repeated half of induction
    sequences, *including* the first repeated position. Only clean induction sequences
    are scored: behaviour on non-repeating sequences is never measured.
    """
    correct, total = 0, 0
    for _ in range(cfg.eval_batches):
        rng, sub = random.split(rng)
        tokens, mask, tpos = make_induction_batch(sub, cfg, cfg.eval_batch_size)
        if int(np.asarray(mask).sum()) == 0:
            continue
        preds = np.asarray(_argmax_preds(model, params, tokens))
        tok = np.asarray(tokens)
        sel = np.asarray(tpos)[:, 1:] & np.asarray(mask)[:, None]
        correct += int((preds == tok[:, 1:])[sel].sum())
        total += int(sel.sum())
    return correct / max(total, 1)


CheckpointCallback = Callable[[int, Params], dict[str, Any]]


@dataclass
class TrainResult:
    records: list[dict[str, Any]]
    saved_params: dict[int, Params] = field(default_factory=dict)
    seconds: float = 0.0


def run_training(
    cfg: ExperimentConfig,
    seed: int,
    on_checkpoint: CheckpointCallback | None = None,
    keep_params: bool = False,
    log: Callable[[str], None] | None = None,
    log_every: int = 1000,
    stop_at: int | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train one seed. ``on_checkpoint(step, params)`` may return extra record fields.

    ``stop_at`` ends training early *without* changing the learning-rate schedule (which
    is always defined by ``cfg.total_steps``), so a truncated run follows the same
    trajectory as the full run up to ``stop_at``.
    """
    last = cfg.total_steps if stop_at is None else min(stop_at, cfg.total_steps)
    t0 = time.time()
    model = build_model(cfg)
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    params = model.init(init_rng, jnp.ones((1, cfg.seq_len), dtype=jnp.int32))["params"]
    tx = make_optimizer(cfg)
    opt_state = tx.init(params)

    ckpt_steps = set(cfg.checkpoint_steps())
    records: list[dict[str, Any]] = []
    saved: dict[int, Params] = {}
    running_loss: float | None = None
    running_raw: float | None = None

    for step in range(cfg.total_steps + 1):
        if step in ckpt_steps:
            rng, eval_rng = random.split(rng)
            rec: dict[str, Any] = {
                "seed": seed,
                "step": step,
                "induction_eval_score": induction_eval_score(model, params, cfg, eval_rng),
                "train_loss": running_loss,
                "raw_loss": running_raw,
            }
            if on_checkpoint is not None:
                rec.update(on_checkpoint(step, params))
            records.append(rec)
            if on_record is not None:
                on_record(rec)
            if keep_params:
                saved[step] = jax.tree_util.tree_map(np.asarray, params)
            if log is not None and (step % log_every == 0 or step == last):
                log(f"[seed {seed}] step {step:5d} behaviour={rec['induction_eval_score']:.3f}")
        if step >= last:
            break
        rng, batch_rng = random.split(rng)
        tokens, _, tpos = make_induction_batch(batch_rng, cfg, cfg.batch_size)
        params, opt_state, loss, raw = train_step(model, params, opt_state, tx, tokens, tpos)
        running_loss = (
            float(loss) if running_loss is None else 0.95 * running_loss + 0.05 * float(loss)
        )
        running_raw = float(raw) if running_raw is None else 0.95 * running_raw + 0.05 * float(raw)

    return TrainResult(records=records, saved_params=saved, seconds=time.time() - t0)
