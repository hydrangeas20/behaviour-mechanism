"""E3: corruption and intervention controls (added; not part of the historical protocol).

Every clean input is a historical repeat sequence ``[c, c]``. The read position is
``half`` (it predicts ``c[1]`` at ``half + 1``). Four corruptions isolate different inputs
to that prediction:

``random_second_half``  historical: every token from ``half`` on is resampled.
``query_token``         only the token at ``half`` changes, so the "does the current token
                        match the token half a sequence back" cue breaks; the copy source
                        (position 1) is intact.
``earlier_occurrence``  only the token at position 0 changes, so the match cue breaks while
                        the query token and the copy source are intact.
``copy_source``         only the token at position 1 changes, so the copied token changes
                        and the match cue is intact. A copier, gated or not, should change its
                        output.

Resampled tokens are drawn to differ from the original token.

Interventions are run in both directions on the same pairs:

* *denoising*: corrupted run with one clean activation restored;
  ``recovery = (lp - lp_corrupt) / gap``;
* *noising*: clean run with one corrupted activation inserted;
  ``disruption = (lp_clean - lp) / gap``.

The sites are each layer's attention output (all positions, as in the historical probe),
each layer's MLP output, the residual stream after each layer at the read position only,
and all attention outputs jointly. Summaries use only pairs with ``|gap| >= 0.05`` and
report how many those are.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .model import Cache, Patch, forward
from .patching import CLIP, GAP_THRESHOLD

Params = Any

CORRUPTIONS = ("random_second_half", "query_token", "earlier_occurrence", "copy_source")


def _resample_different(rng: np.random.Generator, orig: np.ndarray, vocab: int) -> np.ndarray:
    new = (orig + rng.integers(1, vocab, size=orig.shape)) % vocab
    assert (new != orig).all()
    return new


def make_control_pairs(
    rng: np.random.Generator, cfg: ExperimentConfig, n: int, corruption: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``clean (n, T)``, ``corrupt (n, T)`` and ``target (n,)`` (= clean ``c[1]``)."""
    half = cfg.seq_len // 2
    ctx = rng.integers(0, cfg.vocab_size, size=(n, half))
    clean = np.concatenate([ctx, ctx], axis=1)
    corrupt = clean.copy()
    if corruption == "random_second_half":
        corrupt[:, half:] = rng.integers(0, cfg.vocab_size, size=(n, half))
    elif corruption == "query_token":
        corrupt[:, half] = _resample_different(rng, clean[:, half], cfg.vocab_size)
    elif corruption == "earlier_occurrence":
        corrupt[:, 0] = _resample_different(rng, clean[:, 0], cfg.vocab_size)
    elif corruption == "copy_source":
        corrupt[:, 1] = _resample_different(rng, clean[:, 1], cfg.vocab_size)
    else:
        raise ValueError(f"unknown corruption {corruption!r}")
    return clean, corrupt, clean[:, 1].copy()


def intervention_names(n_layers: int) -> list[str]:
    names = [f"attn_L{l}" for l in range(n_layers)]
    names += [f"mlp_L{l}" for l in range(n_layers)]
    names += [f"resid_L{l}" for l in range(n_layers)]
    return [*names, "attn_all"]


def _patch_for(name: str, src: Cache, read_pos: int, n_layers: int) -> Patch:
    kind, _, layer = name.partition("_L")
    if name == "attn_all":
        return Patch(layer_attn_out={l: src.layer_attn_out[l] for l in range(n_layers)})
    l = int(layer)
    if kind == "attn":
        return Patch(layer_attn_out={l: src.layer_attn_out[l]})
    if kind == "mlp":
        return Patch(mlp_out={l: src.mlp_out[l]})
    if kind == "resid":
        return Patch(resid_post={l: (read_pos, src.resid_post[l][:, read_pos])})
    raise ValueError(name)


@partial(jax.jit, static_argnums=(0, 1))
def _run(
    cfg: ExperimentConfig,
    read_pos: int,
    params: Params,
    clean: jnp.ndarray,
    corrupt: jnp.ndarray,
    target: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    def lp(logits: jnp.ndarray) -> jnp.ndarray:
        lps = jax.nn.log_softmax(logits[:, read_pos], axis=-1)
        return jnp.take_along_axis(lps, target[:, None], axis=-1)[:, 0]

    cc, xc = Cache(), Cache()
    out = {
        "lp_clean": lp(forward(params, cfg, clean, cache=cc)),
        "lp_corrupt": lp(forward(params, cfg, corrupt, cache=xc)),
    }
    for name in intervention_names(cfg.n_layers):
        out[f"den_{name}"] = lp(
            forward(params, cfg, corrupt, patch=_patch_for(name, cc, read_pos, cfg.n_layers))
        )
        out[f"noi_{name}"] = lp(
            forward(params, cfg, clean, patch=_patch_for(name, xc, read_pos, cfg.n_layers))
        )
    return out


def run_controls(
    params: Params,
    cfg: ExperimentConfig,
    n_pairs: int,
    seed: int,
    corruptions: tuple[str, ...] = CORRUPTIONS,
    forward_fn: Callable[..., dict[str, jnp.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """One summary row per (corruption, intervention). Pairs are drawn from ``seed``."""
    read_pos = cfg.seq_len // 2
    fn = forward_fn or _run
    rows: list[dict[str, Any]] = []
    for ci, corr in enumerate(corruptions):
        rng = np.random.default_rng([seed, ci])
        clean, corrupt, target = make_control_pairs(rng, cfg, n_pairs, corr)
        res = {
            k: np.asarray(v)
            for k, v in fn(
                cfg, read_pos, params, jnp.asarray(clean), jnp.asarray(corrupt), jnp.asarray(target)
            ).items()
        }
        gap = res["lp_clean"] - res["lp_corrupt"]
        defined = np.abs(gap) >= GAP_THRESHOLD
        base = {
            "corruption": corr,
            "n_pairs": n_pairs,
            "n_defined": int(defined.sum()),
            "gap_median": float(np.median(gap)),
            "lp_clean_mean": float(res["lp_clean"].mean()),
            "lp_corrupt_mean": float(res["lp_corrupt"].mean()),
        }
        for name in intervention_names(cfg.n_layers):
            row = dict(base, intervention=name)
            if defined.any():
                g = gap[defined]
                rec = (res[f"den_{name}"][defined] - res["lp_corrupt"][defined]) / g
                dis = (res["lp_clean"][defined] - res[f"noi_{name}"][defined]) / g
                row.update(
                    recovery_mean=float(np.clip(rec, -CLIP, CLIP).mean()),
                    recovery_median=float(np.median(rec)),
                    disruption_mean=float(np.clip(dis, -CLIP, CLIP).mean()),
                    disruption_median=float(np.median(dis)),
                )
            else:
                row.update(
                    recovery_mean=np.nan,
                    recovery_median=np.nan,
                    disruption_mean=np.nan,
                    disruption_median=np.nan,
                )
            rows.append(row)
    return rows


def fixed_pair_records(
    params: Params,
    cfg: ExperimentConfig,
    clean: np.ndarray,
    corrupt: np.ndarray,
    target: np.ndarray,
) -> list[Any]:
    """Historical layer-level patching on an explicit, reusable pair set (E1 fixed probes).

    Computes exactly the historical per-pair quantities (layer attention output, all
    positions, read at ``half``) in one batched pass, and returns ``PairRecord`` objects so
    the historical aggregates and ``patching.summarise`` apply unchanged.
    """
    from .patching import PairRecord

    read_pos = cfg.seq_len // 2
    res = {
        k: np.asarray(v)
        for k, v in _run(
            cfg, read_pos, params, jnp.asarray(clean), jnp.asarray(corrupt), jnp.asarray(target)
        ).items()
    }
    out = []
    for i in range(clean.shape[0]):
        lc, lx = float(res["lp_clean"][i]), float(res["lp_corrupt"][i])
        gap = lc - lx
        defined = abs(gap) >= GAP_THRESHOLD
        rec = tuple(
            float((res[f"den_attn_L{l}"][i] - lx) / gap) if defined else float("nan")
            for l in range(cfg.n_layers)
        )
        out.append(PairRecord(gap, lc, lx, rec, defined))
    return out
