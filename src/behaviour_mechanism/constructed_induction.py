"""Hand-set two-layer induction model: the positive control for E2 measurements.

Architecture class: the historical ``TinyGPT2`` with vocab 100, seq_len 64, d_model 512,
2 layers, 2 heads. Both MLPs and the second head of each layer are zero, so the model is
attention-only in effect.

Residual layout (one-hot blocks)::

    T  0..99     current token                    (embedding)
    P  100..163  position                         (embedding)
    A  164..263  previous token                   (layer-0 head 0)
    C  264..363  successor of the matched token   (layer-1 head 0; read by the unembedding)

* **Layer 0, head 0 (previous-token head).** Position ``q`` attends to ``q - 1`` by
  position and writes that token into A. Position 0 can only attend to itself.
* **Layer 1, head 0 (induction head).** The query is the current token (T); the key is the
  previous token (A). Position ``q`` therefore attends to every ``k <= q`` whose previous
  token equals the current token, and writes the token *at* ``k`` into C. Position 0 is
  excluded with a large negative key, because its A block holds its own token.
* **Unembedding.** Logit ``GAMMA`` for the token in C.

With a unique match, the output is the successor of the earlier occurrence of the current
token: token-matching induction at any offset. With several matches, attention splits
evenly; with none, it spreads over all earlier positions and the output is diffuse.

LayerNorm is the identity wherever the residual holds a fixed number of one-hot blocks
(layer-0 and layer-1 inputs, and the final norm when the match is unique). Elsewhere it
only rescales the logits.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .constructed import _ln_identity
from .model import init_params

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
V, T_, P_, A_, C_ = 100, 0, 100, 164, 264
BETA = 60.0
GAMMA = 12.0


def build(cfg: ExperimentConfig = CFG) -> Any:
    d, hd, t = cfg.d_model, cfg.d_model // cfg.n_heads, cfg.seq_len
    zeros = jax.tree_util.tree_map(np.zeros_like, init_params(cfg, jax.random.PRNGKey(0)))
    p = jax.tree_util.tree_map(np.asarray, zeros)
    sq = math.sqrt(hd)
    p["tok_emb"]["embedding"][np.arange(V), T_ + np.arange(V)] = 1.0
    p["pos_emb"]["embedding"][np.arange(t), P_ + np.arange(t)] = 1.0
    q0, k0, v0 = 0, d, 2 * d  # head 0 offsets inside the qkv kernel

    b0 = p["block_0"]
    qkv = b0["attn"]["qkv"]["kernel"]
    for q in range(t):
        qkv[P_ + q, q0 + q] = BETA * sq  # query slot q
        if q + 1 < t:
            qkv[P_ + q, k0 + q + 1] = 1.0  # position q is the key for query q + 1
    qkv[T_ + np.arange(V), v0 + np.arange(V)] = 1.0
    b0["attn"]["proj"]["kernel"][np.arange(V), A_ + np.arange(V)] = 1.0
    b0["ln1"].update(_ln_identity(2, d))
    b0["ln2"].update(_ln_identity(3, d))

    b1 = p["block_1"]
    qkv = b1["attn"]["qkv"]["kernel"]
    qkv[T_ + np.arange(V), q0 + np.arange(V)] = BETA * sq  # query: current token
    qkv[A_ + np.arange(V), k0 + np.arange(V)] = 1.0  # key: previous token
    qkv[P_ + np.arange(t), q0 + V] = BETA * sq  # constant-1 query dimension
    qkv[P_ + 0, k0 + V] = -2.0  # position 0 is never a valid key
    qkv[T_ + np.arange(V), v0 + np.arange(V)] = 1.0
    b1["attn"]["proj"]["kernel"][np.arange(V), C_ + np.arange(V)] = 1.0
    b1["ln1"].update(_ln_identity(3, d))
    b1["ln2"].update(_ln_identity(4, d))

    p["ln_f"].update(_ln_identity(4, d))
    p["unembed"]["kernel"][C_ + np.arange(V), np.arange(V)] = GAMMA
    return jax.tree_util.tree_map(jnp.asarray, p)


SITES_2L = (
    "attn_out.L0",
    "attn_out.L1",
    "head_out.L0H0",
    "head_out.L0H1",
    "head_out.L1H0",
    "head_out.L1H1",
    "mlp_out.L0",
    "mlp_out.L1",
    "resid_post.L0",
    "resid_post.L1",
)


def predicted_effect(corruption: str, site: str) -> float:
    """Expected ``bm-patch-2`` conditional effect (both directions) on *distinct* E2 pairs.

    Derivation (read position ``q``; earlier occurrence at ``q - L``; copy source at
    ``q - L + 1``):

    * Layer-1 attention / head L1H0 write C, the only input to the logits: 1 always, and so
      is the residual stream after layer 1 at the read position.
    * Zero components (head 1 of each layer, both MLPs): 0 always.
    * Layer-0 attention / head L0H0 write the previous-token blocks A. Only the
      ``earlier_occurrence`` corruption changes an A block that the induction head uses
      (the key at ``q - L + 1``): 1 there, 0 for the other corruptions. ``copy_source``
      changes the *value* read at ``q - L + 1``, which comes from the token embedding;
      ``query_token`` and ``random_repeat`` change the *query*, also an embedding.
    * Residual after layer 0 at the read position holds the query token: 1 for
      ``query_token`` and ``random_repeat``, 0 for ``earlier_occurrence`` and
      ``copy_source`` (that position's residual is identical in both runs).
    """
    if site in ("attn_out.L1", "head_out.L1H0", "resid_post.L1"):
        return 1.0
    if site in ("head_out.L0H1", "head_out.L1H1", "mlp_out.L0", "mlp_out.L1"):
        return 0.0
    if site in ("attn_out.L0", "head_out.L0H0"):
        return 1.0 if corruption == "earlier_occurrence" else 0.0
    if site == "resid_post.L0":
        return 1.0 if corruption in ("query_token", "random_repeat") else 0.0
    raise ValueError(site)
