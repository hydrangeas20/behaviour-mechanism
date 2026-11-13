"""Activation patching with per-pair records.

Definitions (historical, unchanged)
-----------------------------------
For a clean/corrupted pair, read the log-probability of ``target_tok`` at position
``target_pos - 1``:

* ``gap = lp_clean - lp_corrupt``
* ``recovered_l = (lp_patched_l - lp_corrupt) / gap`` where ``lp_patched_l`` is the
  corrupted run with layer ``l``'s attention output replaced by the clean one.

Pairs with ``|gap| < 0.05`` are skipped. The historical *layer-level* aggregate
(``legacy_layer_effects``) clips each recovery to [-5, 5], **sums over the non-skipped
pairs and divides by the total number of pairs, including skipped ones**. A skipped pair
therefore contributes exactly 0. When all pairs are skipped the aggregate is exactly 0
-- an undefined measurement is recorded as "no effect". The historical *head-level*
aggregate divides by the number of non-skipped pairs instead. Both are reproduced here
unchanged; ``summarise`` additionally reports the defined fraction and the recovery
conditional on a defined pair, so the two components can be told apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .config import ExperimentConfig
from .data import make_clean_corrupt_pair
from .model import Cache, Patch, forward

Params = Any
Array = jnp.ndarray

GAP_THRESHOLD = 0.05
CLIP = 5.0


@dataclass(frozen=True)
class PairRecord:
    gap: float
    lp_clean: float
    lp_corrupt: float
    recovered: tuple[float, ...]  # unclipped; per layer, or flattened (layer, head)
    defined: bool


def _lp(logits: Array, pos: Array, tok: Array) -> Array:
    return jax.nn.log_softmax(logits[0, pos])[tok]


@partial(jax.jit, static_argnums=(0,))
def _layer_pair(
    cfg: ExperimentConfig, params: Params, clean: Array, corrupt: Array, read_pos: Array, tok: Array
) -> tuple[Array, Array, Array]:
    cache = Cache()
    lp_clean = _lp(forward(params, cfg, clean, cache=cache), read_pos, tok)
    lp_corrupt = _lp(forward(params, cfg, corrupt), read_pos, tok)
    patched = [
        _lp(
            forward(params, cfg, corrupt, patch=Patch(layer_attn_out={l: cache.layer_attn_out[l]})),
            read_pos,
            tok,
        )
        for l in range(cfg.n_layers)
    ]
    return lp_clean, lp_corrupt, jnp.stack(patched)


@partial(jax.jit, static_argnums=(0,))
def _head_pair(
    cfg: ExperimentConfig, params: Params, clean: Array, corrupt: Array, read_pos: Array, tok: Array
) -> tuple[Array, Array, Array]:
    cache = Cache()
    lp_clean = _lp(forward(params, cfg, clean, cache=cache), read_pos, tok)
    lp_corrupt = _lp(forward(params, cfg, corrupt), read_pos, tok)
    patched = []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            p = Patch(head_out={(l, h): cache.head_out[l][:, h]})
            patched.append(_lp(forward(params, cfg, corrupt, patch=p), read_pos, tok))
    return lp_clean, lp_corrupt, jnp.stack(patched)


def _records(
    fn: Any, params: Params, cfg: ExperimentConfig, rng: Array, n_pairs: int
) -> list[PairRecord]:
    out: list[PairRecord] = []
    for _ in range(n_pairs):
        rng, sub = random.split(rng)
        clean, corrupt, target_pos, target_tok = make_clean_corrupt_pair(cfg, sub)
        lp_c, lp_x, lp_p = fn(
            cfg, params, clean, corrupt, jnp.asarray(target_pos - 1), jnp.asarray(target_tok)
        )
        lp_c, lp_x = float(lp_c), float(lp_x)
        gap = lp_c - lp_x
        defined = abs(gap) >= GAP_THRESHOLD
        rec = tuple(float((p - lp_x) / gap) if defined else float("nan") for p in np.asarray(lp_p))
        out.append(PairRecord(gap, lp_c, lp_x, rec, defined))
    return out


def layer_pair_records(
    params: Params, cfg: ExperimentConfig, rng: Array, n_pairs: int = 8
) -> list[PairRecord]:
    """Per-pair records using the historical RNG sequence of ``activation_patching_effect``."""
    return _records(_layer_pair, params, cfg, rng, n_pairs)


def head_pair_records(
    params: Params, cfg: ExperimentConfig, rng: Array, n_pairs: int = 4
) -> list[PairRecord]:
    return _records(_head_pair, params, cfg, rng, n_pairs)


def legacy_layer_effects(records: list[PairRecord], n_layers: int) -> np.ndarray:
    """Exact historical layer-level aggregate (skipped pairs count as 0)."""
    total = np.zeros(n_layers)
    for r in records:
        if r.defined:
            total += np.clip(np.asarray(r.recovered), -CLIP, CLIP)
    return total / len(records)


def legacy_head_effects(records: list[PairRecord], n_layers: int, n_heads: int) -> np.ndarray:
    """Exact historical head-level aggregate (divides by the number of defined pairs)."""
    total = np.zeros(n_layers * n_heads)
    n_used = 0
    for r in records:
        if r.defined:
            total += np.clip(np.asarray(r.recovered), -CLIP, CLIP)
            n_used += 1
    return (total / max(n_used, 1)).reshape(n_layers, n_heads)


def summarise(records: list[PairRecord], n_layers: int) -> dict[str, Any]:
    """Decompose the historical statistic into its components.

    Returns the legacy best-layer value (``patch_effect_size``), the defined fraction,
    the per-layer recovery conditional on defined pairs (clipped mean, NaN if none), the
    median clean/corrupt gap, and the mean clean/corrupt log-probabilities.
    """
    legacy = legacy_layer_effects(records, n_layers)
    best = int(np.argmax(legacy))
    defined = [r for r in records if r.defined]
    if defined:
        cond = np.mean([np.clip(r.recovered, -CLIP, CLIP) for r in defined], axis=0)
    else:
        cond = np.full(n_layers, np.nan)
    gaps = np.array([r.gap for r in records])
    out: dict[str, Any] = {
        "patch_effect_size": float(legacy[best]),
        "patch_best_layer": best,
        "n_pairs": len(records),
        "n_defined": len(defined),
        "defined_fraction": len(defined) / len(records),
        "gap_median": float(np.median(gaps)),
        "lp_clean_mean": float(np.mean([r.lp_clean for r in records])),
        "lp_corrupt_mean": float(np.mean([r.lp_corrupt for r in records])),
    }
    for l in range(n_layers):
        out[f"legacy_L{l}"] = float(legacy[l])
        out[f"conditional_L{l}"] = float(cond[l])
    out["conditional_best"] = float(np.nanmax(cond)) if defined else float("nan")
    return out
