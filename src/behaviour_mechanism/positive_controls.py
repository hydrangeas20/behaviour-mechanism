"""Stage 1 positive controls: analytic predictions vs ``measurement.measure``.

The models are the hand-set copiers in :mod:`constructed`. Their mechanisms are known by
construction, so the effect of each (corruption, site, direction) cell can be derived on
paper before running anything. The derivations below use the module layout of
:mod:`constructed`: layer-0 head 0 = A, head 1 = C (copy candidate), layer-0 MLP = match
flag M, layer-1 MLP = output O, and layer-1 attention identically zero.

Reading position ``half`` (it predicts ``c[1]``). Corruptions (all from
:func:`controls.make_control_pairs`):

* ``query_token`` changes the token at ``half``.
* ``earlier_occurrence`` changes the token at 0.
* ``copy_source`` changes the token at 1.
* ``random_second_half`` changes everything from ``half`` on; it equals ``query_token``
  except when the resampled token coincides (probability 1/100 per pair).

Per-cell reasoning, in short:

* **Layer-1 attention is identically zero.** Patching it changes nothing: effect 0 in
  every cell where a gap exists. This is the zero-effect check.
* **``resid_post`` at the read position** carries the whole state of that position. Every
  defined cell has effect 1 for both layers, because layer-1 attention is zero, so position
  ``half`` reads nothing from other positions after layer 0.
* **``mlp_out.L1``** writes O, which is the only input to the logits: effect 1 whenever a
  gap exists.
* **Copy source** changes C only, so head 1 (and the layer-0 attention that contains it)
  gives 1, while head 0 and the match MLP give 0 because the match is intact.
* **Match cue** (query token or earlier occurrence):
  - In the **attention-gated** copier, head 0's output depends on whether the tokens match,
    so attention L0 and head 0 give 1. The MLP-0 flag gives 1.
  - In the **MLP-gated** copier, head 0 fetches the token at position 0 by position. Under
    ``query_token`` that token is unchanged, so every attention output is identical in the
    clean and corrupted runs: attention effects are 0, and only MLP-0 gives 1. Under
    ``earlier_occurrence`` the fetched token itself changes, so head 0 and attention L0
    give 1.
* **Ungated copier.** The match cue does not affect the output: the gap is 0 for every
  match corruption, so all pairs are undefined and the effect is ``None`` (not 0).
  ``copy_source`` behaves as in the gated copiers.

Every value holds for ``denoise`` and ``noise`` alike, because each mechanism has a single
path per cue. Tolerances cover the coincidence rate of ``random_second_half``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import constructed
from .controls import make_control_pairs
from .measurement import (
    METRIC_VERSION,
    MeasurementResult,
    PairSet,
    Site,
    legacy_statistic,
    measure,
)

CORRUPTIONS = ("query_token", "earlier_occurrence", "copy_source", "random_second_half")
SITES = (
    Site("attn_out", 0),
    Site("attn_out", 1),
    Site("head_out", 0, 0),
    Site("head_out", 0, 1),
    Site("mlp_out", 0),
    Site("mlp_out", 1),
    Site("resid_post", 0),
    Site("resid_post", 1),
)


def _match_cell(model: str, corruption: str, site: str) -> float | None:
    if model == "ungated":
        return None  # gap 0: undefined, not zero
    if site in ("attn_out.L1",):
        return 0.0
    if site.startswith(("resid_post", "mlp_out.L1")):
        return 1.0
    if site == "mlp_out.L0":
        return 1.0
    if site == "head_out.L0H1":
        return 0.0  # copy candidate unchanged
    # attn_out.L0 and head_out.L0H0 carry the match evidence?
    if model == "attn_gate":
        return 1.0
    return 1.0 if corruption == "earlier_occurrence" else 0.0  # mlp_gate


def _copy_cell(site: str) -> float:
    if site == "attn_out.L1":
        return 0.0
    if site in ("head_out.L0H0", "mlp_out.L0"):
        return 0.0
    return 1.0  # attn_out.L0, head_out.L0H1, mlp_out.L1, resid_post.*


def predicted_effect(model: str, corruption: str, site: str) -> float | None:
    if corruption == "copy_source":
        return _copy_cell(site)
    return _match_cell(model, corruption, site)


def predicted_defined_fraction(model: str, corruption: str) -> float:
    if model == "ungated" and corruption != "copy_source":
        return 0.0
    return 1.0


TOLERANCE = {"effect": 0.02, "defined_fraction": 0.05}


def run(n_pairs: int = 64, seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (comparison rows, legacy rows). Pairs are the same across models."""
    cfg = constructed.CFG
    rows, legacy_rows = [], []
    for ci, corr in enumerate(CORRUPTIONS):
        clean, corrupt, target = make_control_pairs(
            np.random.default_rng([seed, ci]), cfg, n_pairs, corr
        )
        pairs = PairSet(clean, corrupt, target, np.full(n_pairs, cfg.seq_len // 2), corr)
        for model in constructed.MODELS:
            results = measure(constructed.build(model), cfg, pairs, list(SITES))
            leg = legacy_statistic(results, cfg.n_layers)
            legacy_rows.append(
                {"model": model, "corruption": corr, **leg, "per_layer": leg["per_layer"]}
            )
            for r in results:
                rows.append(_compare(model, r))
    return rows, legacy_rows


def _compare(model: str, r: MeasurementResult) -> dict[str, Any]:
    pe = predicted_effect(model, r.corruption, r.site)
    pdf = predicted_defined_fraction(model, r.corruption)
    df_ok = abs(r.defined_fraction - pdf) <= TOLERANCE["defined_fraction"]
    if pe is None:
        eff_ok = r.conditional_effect_mean is None
    else:
        eff_ok = (
            r.conditional_effect_mean is not None
            and abs(r.conditional_effect_mean - pe) <= TOLERANCE["effect"]
        )
    return {
        "metric_version": METRIC_VERSION,
        "model": model,
        "corruption": r.corruption,
        "site": r.site,
        "direction": r.direction,
        "predicted_defined_fraction": pdf,
        "observed_defined_fraction": r.defined_fraction,
        "n_attempted": r.n_attempted,
        "n_defined": r.n_defined,
        "n_undefined": r.n_undefined,
        "predicted_effect": pe,
        "observed_effect": r.conditional_effect_mean,
        "activation_delta_mean": r.activation_delta_mean,
        "pass": bool(df_ok and eff_ok),
    }
