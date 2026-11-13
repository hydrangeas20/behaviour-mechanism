# Mechanism measurement: methodology (`bm-patch-2`)

Code: `src/behaviour_mechanism/measurement.py`. Positive controls:
`positive_controls.py` (fixed-offset copiers) and `constructed_induction.py` (induction).
Saved outputs: `results/stage1/positive_controls/` and `results/stage2/calibration/`.
Tests: `tests/test_measurement.py`, `tests/test_e2.py`.

## 1. The legacy statistic and why it failed

`legacy_patch_effect_size_v1` is the statistic of the original experiment, kept unchanged
(`patching.legacy_layer_effects`; recomputed from per-pair outputs by
`measurement.legacy_statistic`, and tested equal to the historical code). For 8 pairs
`clean = [c, c]`, `corrupt = [c, random]`:

1. gap = log p_clean(target) − log p_corrupt(target), read at one position;
2. for each layer, copy the clean **attention output** into the corrupted run and compute
   recovery = (log p_patched − log p_corrupt) / gap;
3. skip pairs with |gap| < 0.05, clip recovery to [−5, 5], **sum over the remaining
   pairs, divide by all 8**, and report the **maximum over layers**.

On the MLP-gated constructed copier it returns 0.00 with every pair defined. I checked each
candidate cause against the saved control outputs
(`results/stage1/positive_controls/comparison.csv`):

| Candidate cause | Verdict | Evidence |
|---|---|---|
| Implementation bug | **No** | The same code returns the hand-derived value in all 192 control cells, and `test_legacy_statistic_equals_historical_code` shows the recomputation equals the historical code. |
| Recovery formula | **No** | With a defined gap and the right site, the formula returns 1.00 (MLP-0 output, residual stream). |
| Denominator / undefined-pair handling | **Not for this failure** | All 64 pairs are defined, so no zero-filling occurs. It *is* the cause of a different failure: the ungated copier and the negative control read 0 because every pair is undefined. |
| Target activation | **Yes** | The clean and corrupted runs have *identical* attention outputs (activation difference 0.000 at both layers). Restoring something that did not change cannot recover anything. |
| Intervention design (corruption) | **Yes, jointly** | The historical corruption changes the query token. In the MLP-gated copier the query token reaches the match computation through the token embedding and the residual stream, never through attention. With the `earlier_occurrence` corruption, the same model and the same attention site give 1.00. |
| Causal pathway measured | **Yes: this is the underlying issue** | Attention-output denoising measures "does the clean/corrupted difference pass through this attention output". It is silent about pathways through embeddings, the residual stream and MLPs. |

So zero-filling is one problem, not the only one. A legacy value of 0 can mean
"undefined" (no gap), "the pathway bypasses attention", or "no effect". The statistic
cannot tell these apart, and its maximum over layers adds upward selection noise when
effects are small.

## 2. What `bm-patch-2` reports

`measure(params, cfg, pairs, sites, directions)` returns one `MeasurementResult` per
(site, direction), each with:

| Field | Meaning |
|---|---|
| `metric_version` | `"bm-patch-2"` |
| `site`, `component`, `layer`, `head`, `positions` | target activation: `attn_out`, `head_out`, `mlp_out` (all positions) or `resid_post` (read position only) |
| `direction` | `denoise`: corrupted run + clean activation, effect = (lp − lp_corrupt) / gap. `noise`: clean run + corrupted activation, effect = (lp_clean − lp) / gap |
| `corruption` | which input difference defines the pair |
| `n_attempted`, `n_defined`, `n_undefined`, `defined_fraction` | a pair is defined when \|gap\| ≥ `gap_threshold` (0.05) |
| `conditional_effect_mean` / `_median` / `_unclipped_mean` | over **defined pairs only**; `None` when no pair is defined, never 0 |
| `activation_delta_mean` | mean norm of the clean − corrupted activation at the site and read position; 0 means the site cannot carry the difference |
| `gap_median`, `gap_threshold`, `clip`, `per_pair_effect`, `per_pair_gap` | raw material for re-analysis; undefined pairs are `None` |

Results serialise to strict JSON (`to_json`; `NaN` is rejected, undefined is `null`).
Nothing is aggregated across sites or layers, and no maximum is taken.

The legacy value is available beside these under its legacy name.

## 3. How to read the results

- **Defined fraction** is a statement about the model's *outputs* on the probe inputs:
  does this corruption change the prediction at all? It is behavioural.
- **Conditional effect** is a statement about *mediation*: given that the corruption
  changes the prediction, how much of that change passes through the site?
- A denoising effect near 1 shows the site is **sufficient** to carry the difference
  under this corruption. A noising effect near 1 shows it is **necessary**. Neither shows
  that the site is the unique or complete implementation of a mechanism: several sites
  can each be sufficient (the residual stream always is).
- Compare `activation_delta_mean` before reading a 0: a 0 with zero activation difference
  means "this corruption does not route through this site", not "this site is unimportant".
- No single site or corruption detects every mechanism. Use a set of corruptions that
  isolate different inputs, and a set of sites that covers attention, MLPs and the
  residual stream.

## 4. Positive controls

### 4.1 Fixed-offset copiers (historical task)

Three hand-set models (`constructed.py`): ungated, MLP-gated and attention-gated copiers.
Predictions for every cell (3 models × 4 corruptions × 8 sites × 2 directions = 192) are
derived in the docstring of `positive_controls.py` before running the measurement.

Result (`bm stage1-controls`, 64 pairs): **192 / 192 cells match** within 0.02, including
the cells that are predicted to be undefined (`None`) and the cells predicted to be 0.

| Model | Mechanism-appropriate intervention | Predicted | Observed |
|---|---|---|---|
| Attention-gated | denoise layer-0 attention (or head 0) under `query_token` | 1 | 1.00 |
| MLP-gated | denoise layer-0 **MLP output** under `query_token` | 1 | 1.00 |
| MLP-gated | denoise layer-0 attention under `query_token` | 0 (activation difference 0) | 0.00 |
| MLP-gated | denoise layer-0 attention under `earlier_occurrence` | 1 | 1.00 |
| Ungated | any site under a match-cue corruption | undefined | `None`, 0 / 64 defined |
| All three | layer-1 attention (identically zero) | 0 | 0.00 |

Legacy statistic on the same pairs, for comparison (`legacy_statistic.csv`): attention-gated
1.00 under every corruption; MLP-gated 0.00 under `query_token` and the historical
corruption, 1.00 under `earlier_occurrence` and `copy_source`; ungated 0.00 under the
match-cue corruptions (undefined, zero-filled).

### 4.2 Induction model (E2 task)

A hand-set two-layer model with a previous-token head and an induction head
(`constructed_induction.py`). Predictions per (corruption, site) are in
`predicted_effect`. See `results/stage2/calibration/` and the Stage 2 report for the
outcome.

## 5. Limitations

- The measurement detects mediation of a specified input difference at specified sites.
  It does not identify a mechanism, and it cannot rule out mechanisms that use sites or
  corruptions outside the grid.
- `attn_out`, `head_out` and `mlp_out` are patched at all positions. A positive effect
  therefore does not say *which position's* activation matters.
- Effects are ratios. When gaps are small (just above the threshold) the ratio is noisy;
  the clipped mean, median and unclipped mean are all reported so this can be seen.
- The gap threshold (0.05 nats) is inherited from the original experiment and is
  arbitrary. Per-pair gaps are saved so other thresholds can be applied afterwards.
- The positive controls are idealised: one-hot features and a single path per cue. Passing
  them shows the code computes what it claims. It does not show that trained models are
  as cleanly decomposable.
- No uncertainty intervals are computed here. Per-pair values are saved for bootstrapping.

## 6. Reproduction

```sh
pip install -e ".[dev]"
pytest -q tests/test_measurement.py tests/test_e2.py
bm stage1-controls --out results/stage1/my_positive_controls   # about 1 min
bm e2-calibrate --out results/stage2/my_calibration            # about 1 min
```
