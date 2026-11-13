# Evidence inventory (2026-11-13)

A one-page index of what this repository's evidence does and does not establish. Details
are in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) (historical experiment, E3, E4),
[METRIC_METHODOLOGY.md](METRIC_METHODOLOGY.md) (Stage 1) and
[STAGE2_REPORT.md](STAGE2_REPORT.md) (Stage 2).

"Verified" means recomputed from a committed artifact, or measured by a run whose outputs
and manifest are committed, and pinned by a test where practical.

## Audit corrections already on `main` (merged PR #1, not repeated here)

- Tested package extracted from the notebook; CI; run manifests.
- Per-pair patching records with the legacy aggregate kept unchanged.
- E3 corruption × intervention controls (2 seeds) and E4 constructed copiers.
- E1 fixed-probe infrastructure and draft pre-registration (not run).
- README revision notice; proposed correction (draft, unpublished).

## 1. Verified results

| Result | Evidence | Test |
|---|---|---|
| Historical trajectories reproduce on CPU (seed 0 within 0.001 to step 1,400; seed 1 within 4e-6 to step 2,500) | `results/diagnostics/main_s*_to3000/` | step-0 exact regressions in `tests/test_model_and_patching.py` |
| The historical task is solved by fixed-offset copying: accuracy above the 31/32 token-matching bound at 148/148 checkpoints; other offsets ≤ 0.067 | `results/derived/artifact_verification.json`, `results/diagnostics/summary.csv` | `test_historical_artifacts_are_consistent` |
| Legacy statistic zero-fills undefined pairs: 41/176 main and 56/88 negative-control values are exactly 0; 74% of later values lie on a 1/8 grid | `results/derived/` | same |
| Legacy statistic's rise coincides with the second loss drop in 4/4 historical seeds (exploratory criterion) | `results/derived/artifact_verification.json` | — |
| Constructed copiers: the legacy statistic reads 0.00 on an MLP-gated copier with every pair defined, 1.00 on an attention-gated copier, undefined on an ungated one | `results/controls/e4_constructed/`, re-verified in `results/stage1/positive_controls/` | `test_recorded_e3_e4_results`, `test_legacy_metric_misses_the_mlp_gate_but_corrected_sites_detect_it` |
| **Stage 1.** The MLP-gate failure is a site and corruption coverage problem, not arithmetic: attention outputs are identical in the clean and corrupted runs; MLP-output and residual sites give 1.00 | `results/stage1/positive_controls/comparison.csv` | `tests/test_measurement.py` |
| **Stage 1.** `bm-patch-2` matches hand-derived predictions in 192/192 cells on the fixed-offset copiers | `results/stage1/positive_controls/summary.json` | `test_full_positive_control_grid_passes` |
| **Stage 2.** `bm-patch-2` matches hand-derived predictions in 80/80 cells on a constructed induction model (distinct-token pairs) | `results/stage2/calibration/` | `test_corrected_metric_matches_analytic_predictions_on_induction_model` |
| **Stage 2.** On E2, the historical fixed-offset shortcut is at chance (0.0085); position-only, frequency, bigram and previous-token baselines are at chance; a rule-based reference solves it (0.992) | `results/stage2/baselines/baselines.csv` | `test_shortcut_baselines_fail_and_reference_solves` |
| **Stage 2.** E2 leaves a positional-mixture shortcut worth 0.136 accuracy on trained lengths (0.010 on held-out lengths) | same | `test_positional_mixture_shortcut_is_bounded_and_fails_on_unseen_lengths` |
| **Stage 2.** No exact sequence or chunk is shared between evaluation sets and the 512,000 training sequences; held-out lengths never occur in training | `results/stage2/baselines/leakage.json` | `test_splits_do_not_share_sequences_or_streams` |
| **Stage 2.** Exploratory E2 runs (one seed each): see the Stage 2 report | `results/stage2/e2_*/` | `test_recorded_stage2_results` |

## 2. Previously reported, not verified

| Claim | Source | Why unverified |
|---|---|---|
| Head-level localisation (3/4 seeds concentrate in one layer-0 head) | preprint §4.3, `figures/per_head_final.png` | 4 pairs, no saved numbers |
| Logit-lens trajectories | `figures/logit_lens.png` | 1 sequence per seed, no saved output |
| Real-text track: 93.7% of positions improve; loss 4.626 → 2.443 | preprint §5.2 | no saved output; "before" is the untrained model |
| Scale scan: 256/4 and 128/4 presets | `results/scale_scan.csv` | 2 seeds, final checkpoint only; only the 64/2 preset was re-run |

## 3. Invalidated conclusions

| Conclusion | Status |
|---|---|
| "Behavioural stability precedes mechanistic stability" (original headline) | Not supported: the late rise of the statistic tracks a second behavioural transition |
| The task measures induction | Invalid: fixed-offset copying |
| The legacy statistic measures mechanism strength | Invalid: it mixes the defined fraction, site coverage and a max over layers |
| The negative control "decays to zero" | Invalid: the measurement is undefined there |
| Seed bootstrap P(T_b < T_m) = 0.98 | Invalid as stated: computed with the stability rule the preprint rejects |
| T_b = 300 for all thresholds | Wrong for θ ≥ 0.98 (350; 450 at θ = 1.00) |

## 4. Missing evidence

- Historical checkpoints (never published).
- Uncertainty intervals for any patching quantity (per-pair values are now saved, but no
  bootstrap has been run).
- Where the trained fixed-offset models compute their gate; whether post-gating changes
  are reorganisation or sharpening.
- More than one seed for any E2 result.
- E2 negative control (no repeats) for the trained models.

## 5. Unexecuted experiments

- **E1:** pre-registered 8-seed re-measurement of the historical setup (infrastructure
  ready; awaiting approval).
- **E2 confirmatory:** 8 seeds with pre-registered criteria. Stage 2 is exploratory only.
- **E2 two-layer and attention-only variants:** one two-layer run was started as a
  contingency and stopped at step ~5,000 (incomplete).
- **E3 extensions:** seeds 2–7, per-head sites on the historical task.
- **E5** (width × depth factorial) and **E6** (real-text patching).
- Any GPU work; architecture generalisation.
