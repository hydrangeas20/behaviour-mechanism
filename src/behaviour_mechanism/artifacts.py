"""Checks and derived analyses on the historical CSVs in ``results/``.

Everything here reads the committed artifacts only; nothing is retrained. Outputs go to
``results/derived/`` and never overwrite the historical files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import INDUCTION_LOSS_WEIGHT as LOSS_W
from .config import load_config
from .stability import default_thetas, threshold_sweep, variance_stable_step


def _i(x: Any) -> int:
    return int(x)


HISTORICAL_CONFIG_HASH = "57c45749eb6d"  # printed by the notebook run that wrote results/
N_PAIRS_HISTORICAL = 8  # n_pairs used for every main-run / negative-control checkpoint


def load(results: Path) -> dict[str, pd.DataFrame]:
    names = [
        "behaviour",
        "mechanistic",
        "merged",
        "variance_by_step",
        "negative_control",
        "per_layer_patching",
        "scale_scan",
    ]
    return {n: pd.read_csv(results / f"{n}.csv") for n in names}


def quantisation(values: np.ndarray, n_pairs: int = N_PAIRS_HISTORICAL) -> dict[str, float]:
    """Distance of each value to the nearest multiple of ``1 / n_pairs``.

    If every probe pair yields recovery of either ~1 or exactly 0 (skipped), the legacy
    aggregate is ``k / n_pairs``. Under a continuous (uniform) alternative the median
    distance is ``1 / (4 n_pairs)`` and P(distance < 0.01) is ``0.02 * n_pairs``.
    """
    d = np.abs(values * n_pairs - np.round(values * n_pairs)) / n_pairs
    return {
        "n": len(values),
        "median_distance": float(np.median(d)),
        "frac_within_0.01": float((d < 0.01).mean()),
        "uniform_null_median_distance": 1 / (4 * n_pairs),
        "uniform_null_frac_within_0.01": 0.02 * n_pairs,
    }


def expected_raw_loss(
    vocab: int, frac: float, seq_len: int, gated: bool, loss_weight: float = 1.0
) -> float:
    """Raw (unweighted) next-token loss of two idealised strategies.

    Positions: ``seq_len - 1`` predictions; targets ``1 .. half-1`` are random; target
    ``half`` (first repeated token) cannot be inferred from the first half; targets
    ``half+1 .. seq_len-1`` are copyable in repeat sequences.

    Where the model cannot tell a repeat from a non-repeat, it puts mass ``q`` on the
    half-back token. ``q`` minimises the *training* objective, in which repeat positions
    are up-weighted by ``loss_weight``: ``q = w f / (w f + 1 - f)`` (0.875 for the
    historical w = 3, f = 0.7).

    * ungated: that mixture applies at every target ``>= half``;
    * gated: the same at target ``half``; afterwards the model knows whether the sequence
      repeats (token at ``half`` vs token at 0), so copyable targets cost 0 and non-repeat
      targets cost ``ln V``.
    1/V chance collisions are ignored (second-order).
    """
    half = seq_len // 2
    ln_v = math.log(vocab)
    q = loss_weight * frac / (loss_weight * frac + 1 - frac) if frac > 0 else 0.0
    hit = -math.log(q + (1 - q) / vocab)
    miss = -math.log((1 - q) / vocab)
    mix = frac * hit + (1 - frac) * miss
    n_late = seq_len - 1 - half
    late = (1 - frac) * ln_v if gated else mix
    return ((half - 1) * ln_v + mix + n_late * late) / (seq_len - 1)


def verify(repo: Path) -> dict[str, Any]:
    res = repo / "results"
    d = load(res)
    out: dict[str, Any] = {}

    cfg = load_config(repo / "configs" / "config.json")
    out["config_hash"] = cfg.config_hash()
    out["config_hash_matches_notebook_output"] = cfg.config_hash() == HISTORICAL_CONFIG_HASH

    b, m, mg = d["behaviour"], d["mechanistic"], d["merged"]
    joined = b.merge(m, on=["seed", "step"])
    num = joined.select_dtypes("number").columns
    out["merged_equals_join"] = bool(
        np.allclose(joined[num].fillna(-999).to_numpy(), mg[num].fillna(-999).to_numpy())
    )
    vb = (
        mg.groupby("step")
        .agg(
            behavior_mean=("induction_eval_score", "mean"),
            behavior_std=("induction_eval_score", "std"),
            mech_mean=("patch_effect_size", "mean"),
            mech_std=("patch_effect_size", "std"),
        )
        .reset_index()
    )
    out["variance_by_step_recomputes"] = bool(
        np.allclose(vb.to_numpy(), d["variance_by_step"].to_numpy())
    )
    out["seeds"] = sorted(int(s) for s in mg.seed.unique())
    out["checkpoints_per_seed"] = {_i(k): _i(v) for k, v in mg.groupby("seed").size().items()}

    # Original variance rule (notebook cell 25)
    v = d["variance_by_step"]
    out["variance_rule"] = {
        "T_b": variance_stable_step(v.step.tolist(), v.behavior_std.tolist(), 0.05),
        "T_m": variance_stable_step(v.step.tolist(), v.mech_std.tolist(), 0.10),
    }

    # Preprint sustained-ceiling sweep
    sweep = threshold_sweep(
        mg, {"T_b": "induction_eval_score", "T_m": "patch_effect_size"}, default_thetas()
    )
    sweep["gap"] = sweep["T_m"] - sweep["T_b"]
    out["sweep"] = sweep.to_dict("records")

    # Undefined measurements recorded as exactly 0 (all probe pairs skipped)
    out["exact_zero_patch"] = {
        "main_rows": int((m.patch_effect_size == 0).sum()),
        "main_rows_after_step_300": int(((m.patch_effect_size == 0) & (m.step >= 300)).sum()),
        "main_total_rows": len(m),
        "negative_control_rows": int((d["negative_control"].patch_effect_size == 0).sum()),
        "negative_control_total": len(d["negative_control"]),
        "negative_control_zero_from_step": int(
            d["negative_control"]
            .groupby("step")
            .patch_effect_size.apply(lambda x: (x == 0).all())
            .pipe(lambda s: s[s].index.to_series().loc[lambda i: i >= 1100].min())
        ),
        "scale_scan_rows": d["scale_scan"]
        .loc[d["scale_scan"].final_patch_effect_size == 0, ["d_model", "n_layers", "seed"]]
        .to_dict("records"),
    }
    post = m[m.step >= 300].patch_effect_size.to_numpy()
    pre = m[m.step < 300].patch_effect_size.to_numpy()
    out["quantisation_after_step_300"] = quantisation(post)
    out["quantisation_before_step_300"] = quantisation(pre)

    # Best-layer identity
    out["best_layer_counts_after_300"] = {
        _i(k): _i(v) for k, v in m[m.step >= 300].patch_best_layer.value_counts().items()
    }

    # Behavioural score > 31/32 means the *first* repeated token is also predicted
    half = cfg.seq_len // 2
    bound = (half - 1) / half
    out["behaviour_exceeds_token_matching_bound"] = {
        "bound": bound,
        "n_rows_above_bound": int((b.induction_eval_score > bound + 1e-9).sum()),
        "n_rows_step_ge_350": int((b.step >= 350).sum()),
        "n_rows_step_ge_350_above_bound": int(
            ((b.step >= 350) & (b.induction_eval_score > bound + 1e-9)).sum()
        ),
    }

    # Raw loss plateau vs idealised strategies
    raw = b.groupby("step").raw_loss.mean()
    out["raw_loss_seed_mean"] = {_i(k): round(float(v), 4) for k, v in raw.dropna().items()}
    out["ideal_raw_loss"] = {
        "ungated_positional_copy": expected_raw_loss(
            cfg.vocab_size, cfg.frac_induction_sequences, cfg.seq_len, False, LOSS_W
        ),
        "gated_copy": expected_raw_loss(
            cfg.vocab_size, cfg.frac_induction_sequences, cfg.seq_len, True, LOSS_W
        ),
        "copy_mass_q": LOSS_W
        * cfg.frac_induction_sequences
        / (LOSS_W * cfg.frac_induction_sequences + 1 - cfg.frac_induction_sequences),
        "no_copy": math.log(cfg.vocab_size),
    }
    out["per_seed_transitions"] = per_seed_transitions(mg)
    return out


def _first_sustained(g: pd.DataFrame, col: str, pred: Any, window: int = 3) -> int | None:
    g = g.sort_values("step")
    ok = pred(g[col].to_numpy())
    steps = g.step.to_numpy()
    for i in range(len(steps) - window + 1):
        if ok[i : i + window].all():
            return int(steps[i])
    return None


def per_seed_transitions(mg: pd.DataFrame) -> list[dict[str, Any]]:
    """Per seed: when accuracy, the patching statistic and the raw loss each settle.

    * accuracy: >= 0.99 for 3 consecutive checkpoints;
    * patching statistic: >= 0.8 for 3 consecutive checkpoints;
    * raw loss: <= 3.03 nats (about 0.05 above the gated-copy ideal) for 3 checkpoints.
    The loss criterion is exploratory (chosen after seeing the data) and is reported to
    show alignment, not as a confirmatory test.
    """
    rows = []
    for seed, g in mg.groupby("seed"):
        rows.append(
            {
                "seed": _i(seed),
                "accuracy_ge_0.99": _first_sustained(
                    g, "induction_eval_score", lambda x: x >= 0.99
                ),
                "patch_ge_0.8": _first_sustained(g, "patch_effect_size", lambda x: x >= 0.8),
                "raw_loss_le_3.03": _first_sustained(g, "raw_loss", lambda x: x <= 3.03),
            }
        )
    return rows


def write_derived(repo: Path) -> dict[str, Any]:
    out = verify(repo)
    dest = repo / "results" / "derived"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out["sweep"]).to_csv(dest / "sustained_ceiling_sweep.csv", index=False)
    (dest / "artifact_verification.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out
