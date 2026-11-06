"""Model definition (faithful copy of the notebook) plus a functional forward pass.

``TinyGPT2`` and its submodules keep the notebook's parameter names, so parameters
saved by either implementation load in the other.

``forward`` re-implements the same computation as a pure function of the parameter
tree. It supports caching each layer's attention output (whole layer, after the output
projection) and each head's pre-projection output, and substituting either of them --
which is all activation patching needs. ``tests/test_model.py`` checks that ``forward``
matches ``TinyGPT2.apply`` to float32 tolerance, so patching runs the same network that
was trained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ExperimentConfig

Params = Any
Array = jnp.ndarray

LN_EPS = 1e-6  # flax.linen.LayerNorm default


class CausalSelfAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: Array, return_attn: bool = False) -> Any:
        b, t, _ = x.shape
        head_dim = self.d_model // self.n_heads
        qkv = nn.Dense(3 * self.d_model, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(z: Array) -> Array:
            return z.reshape(b, t, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(head_dim)
        causal_mask = jnp.tril(jnp.ones((t, t), dtype=bool))
        attn_logits = jnp.where(causal_mask[None, None], attn_logits, -1e9)
        attn = jax.nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.d_model)
        out = nn.Dense(self.d_model, name="proj")(out)
        if return_attn:
            return out, attn
        return out


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x: Array, return_attn: bool = False) -> Any:
        ln1 = nn.LayerNorm(name="ln1")(x)
        attn = None
        if return_attn:
            attn_out, attn = CausalSelfAttention(self.d_model, self.n_heads, name="attn")(
                ln1, return_attn=True
            )
        else:
            attn_out = CausalSelfAttention(self.d_model, self.n_heads, name="attn")(ln1)
        x = x + attn_out
        ln2 = nn.LayerNorm(name="ln2")(x)
        ff = nn.Dense(self.d_ff, name="ff1")(ln2)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.d_model, name="ff2")(ff)
        x = x + ff
        if return_attn:
            return x, attn
        return x


class TinyGPT2(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int

    @nn.compact
    def __call__(self, tokens: Array, return_activations: bool = False) -> Any:
        _, t = tokens.shape
        x = nn.Embed(self.vocab_size, self.d_model, name="tok_emb")(tokens)
        x = x + nn.Embed(self.max_seq_len, self.d_model, name="pos_emb")(jnp.arange(t)[None, :])
        activations: dict[str, Any] = {"resid_pre_0": x}
        attns = []
        for i in range(self.n_layers):
            block = TransformerBlock(self.d_model, self.n_heads, self.d_ff, name=f"block_{i}")
            if return_activations:
                x, attn = block(x, return_attn=True)
                attns.append(attn)
                activations[f"resid_post_{i}"] = x
            else:
                x = block(x)
        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(self.vocab_size, name="unembed")(x)
        if return_activations:
            activations["attn_patterns"] = attns
            activations["final_resid"] = x
            return logits, activations
        return logits


def build_model(cfg: ExperimentConfig) -> TinyGPT2:
    return TinyGPT2(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
    )


def init_params(cfg: ExperimentConfig, rng: Array) -> Params:
    dummy = jnp.ones((1, cfg.seq_len), dtype=jnp.int32)
    return build_model(cfg).init(rng, dummy)["params"]


# ---------------------------------------------------------------------------
# Functional forward pass with caching and patching
# ---------------------------------------------------------------------------


def _layer_norm(p: Params, x: Array) -> Array:
    mean = x.mean(-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    out: Array = (x - mean) * jax.lax.rsqrt(var + LN_EPS) * p["scale"] + p["bias"]
    return out


def _dense(p: Params, x: Array) -> Array:
    out: Array = x @ p["kernel"] + p["bias"]
    return out


@dataclass
class Patch:
    """Substitute activations during ``forward``.

    ``layer_attn_out[l]`` replaces layer ``l``'s attention output (post projection,
    pre residual-add), shape ``(B, T, d_model)``. ``head_out[(l, h)]`` replaces head
    ``h`` of layer ``l`` before the output projection, shape ``(B, T, head_dim)``.
    Replacement is applied at every position, as in the historical notebook; because
    attention is causal and clean/corrupted sequences share their first half, only
    positions ``>= half`` can differ.

    Added for the E3 controls (not used by the historical statistic):
    ``mlp_out[l]`` replaces layer ``l``'s MLP output (all positions, ``(B, T, d_model)``);
    ``resid_post[l] = (pos, value)`` replaces the residual stream after layer ``l`` at one
    position only, ``value`` of shape ``(B, d_model)``.
    """

    layer_attn_out: dict[int, Array] = field(default_factory=dict)
    head_out: dict[tuple[int, int], Array] = field(default_factory=dict)
    mlp_out: dict[int, Array] = field(default_factory=dict)
    resid_post: dict[int, tuple[int, Array]] = field(default_factory=dict)


@dataclass
class Cache:
    layer_attn_out: dict[int, Array] = field(default_factory=dict)
    head_out: dict[int, Array] = field(default_factory=dict)  # (B, H, T, head_dim)
    resid_post: dict[int, Array] = field(default_factory=dict)
    mlp_out: dict[int, Array] = field(default_factory=dict)


def forward(
    params: Params,
    cfg: ExperimentConfig,
    tokens: Array,
    patch: Patch | None = None,
    cache: Cache | None = None,
) -> Array:
    """Pure-function forward pass equivalent to ``TinyGPT2.apply``."""
    _, t = tokens.shape
    head_dim = cfg.d_model // cfg.n_heads
    x = params["tok_emb"]["embedding"][tokens] + params["pos_emb"]["embedding"][jnp.arange(t)][None]
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))
    for i in range(cfg.n_layers):
        bp = params[f"block_{i}"]
        h_in = _layer_norm(bp["ln1"], x)
        qkv = _dense(bp["attn"]["qkv"], h_in)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        b = q.shape[0]

        def split(z: Array, b: int = b) -> Array:
            return z.reshape(b, t, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(head_dim)
        logits = jnp.where(causal[None, None], logits, -1e9)
        heads = jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(logits, axis=-1), v)
        if patch is not None:
            for (pl, ph), val in patch.head_out.items():
                if pl == i:
                    heads = heads.at[:, ph].set(val)
        if cache is not None:
            cache.head_out[i] = heads
        attn_out = _dense(
            bp["attn"]["proj"], heads.transpose(0, 2, 1, 3).reshape(b, t, cfg.d_model)
        )
        if patch is not None and i in patch.layer_attn_out:
            attn_out = patch.layer_attn_out[i]
        if cache is not None:
            cache.layer_attn_out[i] = attn_out
        x = x + attn_out
        ff = _dense(bp["ff2"], nn.gelu(_dense(bp["ff1"], _layer_norm(bp["ln2"], x))))
        if patch is not None and i in patch.mlp_out:
            ff = patch.mlp_out[i]
        if cache is not None:
            cache.mlp_out[i] = ff
        x = x + ff
        if patch is not None and i in patch.resid_post:
            rpos, rval = patch.resid_post[i]
            x = x.at[:, rpos].set(rval)
        if cache is not None:
            cache.resid_post[i] = x
    x = _layer_norm(params["ln_f"], x)
    return _dense(params["unembed"], x)


def count_params(params: Params) -> int:
    return int(sum(a.size for a in jax.tree_util.tree_leaves(params)))
