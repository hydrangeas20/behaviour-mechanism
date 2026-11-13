"""E4: hand-set ("constructed-weights") copiers whose mechanism is known by construction.

All three models use the historical architecture class (``TinyGPT2``, pre-LN, GELU MLPs)
with ``vocab 100``, ``seq_len 64``, ``d_model 512``, 2 layers, 2 heads and ``d_ff 256``.
That is wider than the trained model so that every feature has its own coordinates. The
historical probe code runs on them unchanged.

Residual layout (one-hot blocks of value 1)::

    T  0..99     current token               (embedding)
    P  100..163  position                    (embedding)
    A  164..263  layer-0 head 0 output       (model-specific, below)
    C  264..363  token at q - 31             (layer-0 head 1: the copy candidate)
    M  364, 365  match yes / match no        (layer-0 MLP, or a constant)
    O  366..465  output token, 466 = "none"  (layer-1 MLP; read by the unembedding)

LayerNorm is made the identity: every position carries the same number of one-hot blocks
at each depth, so the mean and variance of the LayerNorm input are known constants. Setting
``scale = std`` and ``bias = mean`` undoes the normalisation exactly. The attention logit
margins are at least 30 and the MLP gates are saturated GELUs, so all blocks are one-hot to
about 1e-12.

Models
------
``ungated``     copies ``C`` on every sequence (M is a constant "yes"). Head 0 attends to
                ``q - 32`` and writes that token into A (unused).
``mlp_gate``    head 0 attends to ``q - 32`` *by position only* and writes that token into
                A; the layer-0 MLP compares A with the current token T. The match signal is
                computed from the residual stream, so it is not in any attention output.
``attn_gate``   head 0 attends to ``q - 32`` only if the token there equals the current
                token (otherwise to itself) and writes the *source position*; the layer-0
                MLP checks "source position = q - 32". The match signal is carried by the
                layer-0 attention output.

In every model, layer 1 has zero attention, and its MLP writes ``O = C`` if match-yes,
else "none". The unembedding gives logit 12 to the output token.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .model import LN_EPS, init_params

Params = Any

CFG = ExperimentConfig(
    vocab_size=100,
    d_model=512,
    n_layers=2,
    n_heads=2,
    d_ff=256,
    max_seq_len=64,
    seq_len=64,
    seeds=(0,),
)
MODELS = ("ungated", "mlp_gate", "attn_gate")

V, T_, P_, A_, C_, M_YES, M_NO, O_, O_NONE = 100, 0, 100, 164, 264, 364, 365, 366, 466
BETA = 60.0  # attention logit unit (before the 1/sqrt(head_dim) scale is undone)
K_GATE = 20.0  # GELU gate steepness: pre-activation +10 (on) or <= -10 (off)
GAMMA = 12.0  # output logit


def _gelu(x: float) -> float:
    return float(jax.nn.gelu(jnp.asarray(x)))


def _ln_identity(n_blocks: int, d: int) -> dict[str, jnp.ndarray]:
    mean = n_blocks / d
    var = n_blocks / d - mean**2
    std = math.sqrt(var + LN_EPS)
    return {"scale": jnp.full((d,), std), "bias": jnp.full((d,), mean)}


def build(model: str, cfg: ExperimentConfig = CFG) -> Params:
    if model not in MODELS:
        raise ValueError(model)
    d, hd, half = cfg.d_model, cfg.d_model // cfg.n_heads, cfg.seq_len // 2
    zeros = jax.tree_util.tree_map(np.zeros_like, init_params(cfg, jax.random.PRNGKey(0)))
    p = jax.tree_util.tree_map(np.asarray, zeros)
    scale_qk = math.sqrt(hd)

    p["tok_emb"]["embedding"][:, :] = 0
    p["tok_emb"]["embedding"][np.arange(V), T_ + np.arange(V)] = 1.0
    p["pos_emb"]["embedding"][np.arange(cfg.seq_len), P_ + np.arange(cfg.seq_len)] = 1.0

    b0 = p["block_0"]
    qkv = b0["attn"]["qkv"]["kernel"]  # (d, 3d): q | k | v, each split into heads of size hd
    q0, k0, v0 = 0, d, 2 * d  # head 0 offsets
    q1, k1, v1 = hd, d + hd, 2 * d + hd  # head 1 offsets
    pos = np.arange(cfg.seq_len)

    # Head 1: attend to q - 31 by position; value = token -> C.
    for q in pos:
        qkv[P_ + q, q1 + q] = BETA * scale_qk
    for k in pos:
        if k + 31 < cfg.seq_len:
            qkv[P_ + k, k1 + k + 31] = 1.0
    qkv[T_ + np.arange(V), v1 + np.arange(V)] = 1.0

    if model in ("ungated", "mlp_gate"):
        # Head 0: attend to q - 32 by position; value = token -> A.
        for q in pos:
            qkv[P_ + q, q0 + q] = BETA * scale_qk
        for k in pos:
            if k + half < cfg.seq_len:
                qkv[P_ + k, k0 + k + half] = 1.0
        qkv[T_ + np.arange(V), v0 + np.arange(V)] = 1.0
    else:
        # Head 0 (attn_gate): logit = BETA*([k == q-32] + [T_k == T_q]) + 0.5*BETA*[k == q];
        # value = source position -> A[0:64].
        o_pos, o_tok, o_self = 0, cfg.seq_len, cfg.seq_len + V  # sub-blocks of the head
        for q in pos:
            qkv[P_ + q, q0 + o_pos + q] = BETA * scale_qk
            qkv[P_ + q, q0 + o_self + q] = 0.5 * BETA * scale_qk
        for k in pos:
            if k + half < cfg.seq_len:
                qkv[P_ + k, k0 + o_pos + k + half] = 1.0
            qkv[P_ + k, k0 + o_self + k] = 1.0
        qkv[T_ + np.arange(V), q0 + o_tok + np.arange(V)] = BETA * scale_qk
        qkv[T_ + np.arange(V), k0 + o_tok + np.arange(V)] = 1.0
        qkv[P_ + pos, v0 + pos] = 1.0

    proj = b0["attn"]["proj"]["kernel"]  # (d, d): concatenated heads -> residual
    proj[np.arange(V), A_ + np.arange(V)] = 1.0  # head 0 -> A
    proj[hd + np.arange(V), C_ + np.arange(V)] = 1.0  # head 1 -> C

    b0["ln1"].update(_ln_identity(2, d))
    b0["ln2"].update(_ln_identity(4, d))
    ff1, ff1b = b0["ff1"]["kernel"], b0["ff1"]["bias"]
    ff2, ff2b = b0["ff2"]["kernel"], b0["ff2"]["bias"]
    on = _gelu(0.5 * K_GATE)
    if model == "ungated":
        ff2b[M_YES] = 1.0
    else:
        if model == "mlp_gate":
            pairs = [(T_ + v, A_ + v) for v in range(V)]  # current token == token at q-32
        else:
            pairs = [(P_ + q, A_ + q - half) for q in range(half, cfg.seq_len)]  # src == q-32
        for u, (i, j) in enumerate(pairs):
            ff1[i, u] = K_GATE
            ff1[j, u] = K_GATE
            ff1b[u] = -1.5 * K_GATE
            ff2[u, M_YES] = 1.0 / on
            ff2[u, M_NO] = -1.0 / on
        ff2b[M_NO] = 1.0

    b1 = p["block_1"]
    b1["ln1"].update(_ln_identity(5, d))
    b1["ln2"].update(_ln_identity(5, d))
    for v in range(V):
        b1["ff1"]["kernel"][C_ + v, v] = K_GATE
        b1["ff1"]["kernel"][M_YES, v] = K_GATE
        b1["ff1"]["bias"][v] = -1.5 * K_GATE
        b1["ff2"]["kernel"][v, O_ + v] = 1.0 / on
        b1["ff2"]["kernel"][v, O_NONE] = -1.0 / on
    b1["ff2"]["bias"][O_NONE] = 1.0

    p["ln_f"].update(_ln_identity(6, d))
    p["unembed"]["kernel"][O_ + np.arange(V), np.arange(V)] = GAMMA
    return jax.tree_util.tree_map(jnp.asarray, p)


def expected(model: str) -> dict[str, Any]:
    """What the historical probe should report on each model, derived by hand."""
    return {
        "ungated": {"defined_fraction": 0.0, "patch_effect_size": 0.0},
        "mlp_gate": {"defined_fraction": 1.0, "conditional_L0": 0.0, "patch_effect_size": 0.0},
        "attn_gate": {"defined_fraction": 1.0, "conditional_L0": 1.0, "patch_effect_size": 1.0},
    }[model]
