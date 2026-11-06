"""Added behavioural diagnostics (not part of the historical run).

These separate two strategies that the historical behavioural metric cannot tell apart:

* **token-matching induction** -- find an earlier occurrence of the current token and
  copy its successor; works at any repeat offset;
* **fixed-offset positional copying** -- copy the token ``seq_len // 2 - 1`` positions
  back; works only at the training offset and also fires on non-repeating sequences.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .data import make_corrupt_copy_batch, make_offset_probe_batch
from .model import forward

Params = Any


@partial(jax.jit, static_argnums=(0,))
def _logprobs(cfg: ExperimentConfig, params: Params, tokens: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.log_softmax(forward(params, cfg, tokens[:, :-1]), axis=-1)


def _score(
    cfg: ExperimentConfig, params: Params, tokens: np.ndarray, scored: np.ndarray, source_shift: int
) -> tuple[float, float]:
    """Accuracy and mean log-prob of ``tokens[p - source_shift]`` as the prediction for ``p``."""
    lp = np.asarray(_logprobs(cfg, params, jnp.asarray(tokens)))  # (B, T-1, V)
    t = tokens.shape[1]
    pos = np.arange(1, t)
    src = np.clip(pos - source_shift, 0, t - 1)
    want = tokens[:, src]  # (B, T-1)
    sel = scored[:, 1:]
    preds = lp.argmax(-1)
    acc = float((preds == want)[sel].mean())
    mlp = float(np.take_along_axis(lp, want[..., None], axis=-1)[..., 0][sel].mean())
    return acc, mlp


def offset_accuracy(
    params: Params, cfg: ExperimentConfig, offset: int, n_seq: int, rng_np: np.random.Generator
) -> dict[str, float]:
    """Induction accuracy when the chunk repeats at ``offset`` (training offset = T // 2)."""
    tokens, scored = make_offset_probe_batch(rng_np, cfg, offset, n_seq)
    acc, mlp = _score(cfg, params, tokens, scored, source_shift=offset)
    return {f"offset{offset}_acc": acc, f"offset{offset}_logprob": mlp}


def positional_copy_rate(
    params: Params, cfg: ExperimentConfig, n_seq: int, rng_np: np.random.Generator
) -> dict[str, float]:
    """On fully random sequences (no repeat), how often the model predicts the token half a
    sequence back. Chance = 1 / vocab_size. A high value means copying is not gated on a
    token match."""
    tokens, scored = make_corrupt_copy_batch(rng_np, cfg, n_seq)
    acc, mlp = _score(cfg, params, tokens, scored, source_shift=cfg.seq_len // 2)
    return {"random_copy_rate": acc, "random_copy_logprob": mlp}
