"""Synthetic data generators.

``make_induction_batch`` and ``make_clean_corrupt_pair`` reproduce the notebook
functions exactly, including their RNG consumption, so that a re-run with the same
seed draws the same tokens.

Important property of the historical task (audit P0-1): every structured sequence is
``[context, context]`` with ``len(context) == seq_len // 2``. The repeat therefore
always sits at a *fixed offset* of ``seq_len // 2``. A model can reach perfect
behavioural accuracy by copying the token at position ``p - seq_len // 2 + 1``
without ever matching tokens (positional copying). ``make_offset_probe_batch`` and
``make_corrupt_copy_batch`` are added diagnostics that separate the two strategies.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from .config import ExperimentConfig

Array = jnp.ndarray


def _np_rng_from_key(key: Array) -> np.random.Generator:
    """Notebook idiom: derive a NumPy generator from a JAX key."""
    return np.random.default_rng(int(random.randint(key, (), 0, 2**31 - 1)))


def make_induction_batch(
    rng: Array, cfg: ExperimentConfig, batch_size: int
) -> tuple[Array, Array, Array]:
    """Faithful copy of the notebook generator.

    Returns ``tokens (B, T)``, ``induction_mask (B,)`` and ``target_positions (B, T)``.
    ``target_positions[i, p]`` is True for every ``p`` in the repeated half; the token at
    ``p`` is predicted from position ``p - 1``. Note that ``p == half`` (the first token of
    the repeat) is included: it cannot be predicted by token-matching induction, only by
    position (or by a prior that the sequence repeats).
    """
    rng, sub = random.split(rng)
    tokens = np.array(random.randint(sub, (batch_size, cfg.seq_len), 0, cfg.vocab_size))
    induction_mask = np.zeros(batch_size, dtype=bool)
    target_positions = np.zeros((batch_size, cfg.seq_len), dtype=bool)

    half = cfg.seq_len // 2
    rng_np = _np_rng_from_key(rng)
    for i in range(batch_size):
        if rng_np.random() < cfg.frac_induction_sequences:
            context = rng_np.integers(0, cfg.vocab_size, size=half)
            tokens[i, :half] = context
            tokens[i, half : half * 2] = context
            induction_mask[i] = True
            target_positions[i, half : half * 2] = True

    return jnp.array(tokens), jnp.array(induction_mask), jnp.array(target_positions)


def make_clean_corrupt_pair(cfg: ExperimentConfig, rng: Array) -> tuple[Array, Array, int, int]:
    """Faithful copy of the notebook probe pair.

    clean = ``[context, context]``; corrupted = ``[context, other]``. The two sequences
    differ at *every* position of the second half, including position ``half`` -- the
    position whose output is read (it predicts ``target_pos = half + 1``). The corruption
    therefore changes the query token itself, not only the "is this a repeat" cue.
    """
    rng_np = _np_rng_from_key(rng)
    half = cfg.seq_len // 2
    context = rng_np.integers(0, cfg.vocab_size, size=half)
    other = rng_np.integers(0, cfg.vocab_size, size=half)
    clean = np.concatenate([context, context])
    corrupted = np.concatenate([context, other])
    target_pos = half + 1
    target_tok = int(context[1])
    return jnp.array(clean)[None, :], jnp.array(corrupted)[None, :], target_pos, target_tok


# ---------------------------------------------------------------------------
# Added diagnostics (not part of the historical run)
# ---------------------------------------------------------------------------


def make_offset_probe_batch(
    rng_np: np.random.Generator, cfg: ExperimentConfig, offset: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sequences ``[chunk, chunk, filler]`` with ``len(chunk) == offset``.

    Returns ``tokens (B, T)`` and a boolean ``scored (B, T)`` marking positions
    ``offset + 1 .. 2 * offset - 1`` (token-matching induction can predict these: the
    current token has an earlier occurrence followed by the target). The first repeated
    position ``offset`` is excluded because only position/prior can predict it.

    With ``offset == seq_len // 2`` this is the training distribution. With any other
    offset, a fixed-offset positional copier predicts the wrong token while a
    token-matching induction mechanism is unaffected.
    """
    if not 2 <= offset <= cfg.seq_len // 2:
        raise ValueError("offset must be in [2, seq_len // 2]")
    tokens = rng_np.integers(0, cfg.vocab_size, size=(batch_size, cfg.seq_len))
    tokens[:, offset : 2 * offset] = tokens[:, :offset]
    scored = np.zeros((batch_size, cfg.seq_len), dtype=bool)
    scored[:, offset + 1 : 2 * offset] = True
    return tokens, scored


def make_corrupt_copy_batch(
    rng_np: np.random.Generator, cfg: ExperimentConfig, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``[context, other]`` sequences (the probe's corrupted distribution).

    Returns tokens and ``scored`` for positions ``half + 1 .. seq_len - 1``. Used to
    measure the *positional copy rate*: how often the argmax prediction at ``p - 1`` equals
    the token at ``p - half`` even though the second half is not a repeat.
    """
    half = cfg.seq_len // 2
    tokens = rng_np.integers(0, cfg.vocab_size, size=(batch_size, cfg.seq_len))
    scored = np.zeros((batch_size, cfg.seq_len), dtype=bool)
    scored[:, half + 1 :] = True
    return tokens, scored
