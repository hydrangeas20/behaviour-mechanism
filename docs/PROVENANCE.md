# Artifact provenance

## Historical artifacts (commit `a5032fc`, 2026-09-20)

These files are unchanged on this branch. Fields marked **unknown** were not recorded at
the time and are not reconstructed.

| Artifact | Produced by (notebook cell) | Config hash | Verified on this branch |
|---|---|---|---|
| `configs/config.json` | cell 9 (`config_<hash>.json`) | `57c45749eb6d` | hash recomputed and equal to the notebook output |
| `results/behaviour.csv` | cell 21 (`behavior_<hash>.csv`, renamed) | `57c45749eb6d` | step-0 rows reproduced exactly on CPU (`tests/test_model_and_patching.py`) |
| `results/mechanistic.csv` | cell 22 (**no saved output** in the notebook) | `57c45749eb6d` | step-0 seed-0 value reproduced to 1e-6; a CPU re-run matches seed 0 to at least step 400 |
| `results/merged.csv` | cell 24 | `57c45749eb6d` | equals the join of the two files above |
| `results/variance_by_step.csv` | cell 24 | `57c45749eb6d` | recomputed exactly from `merged.csv` |
| `results/per_layer_patching.csv` | cell 28 (independent RNG `seed + 20000`) | `57c45749eb6d` | internally consistent; not re-derived |
| `results/negative_control.csv` | cell 35 | `584a4b4b2a1c` | hash recomputed and equal |
| `results/scale_scan.csv` | cell 37 (2 seeds, 4,000 steps, 6 pairs) | not saved | partly re-run on CPU (64/2 only) |
| `figures/behaviour-mechanism.png` | cell 26 (saved as `behavior_vs_mechanism_<hash>.png`) | `57c45749eb6d` | visually consistent with `merged.csv` |
| `figures/variance_bands.png` | cell 27 | `57c45749eb6d` | dotted lines show the **rejected** variance rule (T_b = 0, T_m = 2000) |
| `figures/per_layer_patching.png` | cell 28 | `57c45749eb6d` | consistent with the CSV |
| `figures/per_head_final.png` | cell 33 (4 pairs; **no CSV saved**) | `57c45749eb6d` | cannot be checked numerically |
| `figures/logit_lens.png` | cell 29 (1 sequence per seed; **no saved output**) | `57c45749eb6d` | cannot be checked |
| `figures/negative_control.png` | cell 35 | `584a4b4b2a1c` | consistent with the CSV |
| `figures/real_text_*.png`, `figures/induction_schematic.png` | cells 43–53 | n/a | real-text numbers in the README and preprint (93.7%, 4.626 → 2.443) have **no saved output** in the repository |
| `paper/behaviour_mechanism_preprint.pdf` | written separately; the sustained-ceiling analysis code was not committed | n/a | sweep recomputed; see audit §C |

The following were not recorded for the historical run and are **unknown**:
- exact package versions beyond `jax 0.11.1` and `flax 0.12.9` (printed in cell 3);
- the optax version;
- the CUDA/driver version;
- the GPU model beyond the notebook metadata `gpuType: T4`;
- wall-clock times;
- the git commit at run time (the notebook ran in Colab, outside git).

Model checkpoints and `training_state_<hash>.pkl` were kept on the author's Google Drive
and are not in the repository.

## New artifacts (this branch)

| Artifact | Command | Manifest |
|---|---|---|
| `results/derived/artifact_verification.json`, `results/derived/sustained_ceiling_sweep.csv` | `bm verify-artifacts` | deterministic function of the CSVs |
| `figures/derived/historical_*.png`, `patch_statistic_quantisation.png`, `sustained_ceiling_sweep.png` | `bm figures` | deterministic function of the CSVs |
| `results/diagnostics/<run>/checkpoints.jsonl` | `bm diagnose …` | `results/diagnostics/<run>/manifest.json` (git commit, dirty flag, versions, hardware, config, seed, timestamps, command) |
| `results/diagnostics/summary.csv`, `figures/derived/diagnostics_trajectories.png` | `bm diagnostics-report` | derived from the run directories |
| `results/controls/e4_constructed/` | `bm constructed` | `manifest.json` (commit `9c3c69e`, clean tree; construction constants) |
| `results/controls/e3_main_s{0,1}/` | `bm controls --checkpoints artifacts/checkpoints/main_s{0,1} --n-pairs 128` | `manifest.json` (commit `9c3c69e`, clean tree; sha256 of every input checkpoint); `checkpoints_manifest.json` (the training run that wrote the checkpoints, with per-checkpoint records) |
| `artifacts/checkpoints/main_s{0,1}/step_*.npz` (**not committed**: 3.3 MB each; regenerate with `scripts/train_and_save_checkpoints.py`) | see README | sha256s recorded in the E3 manifests |
| `figures/derived/e3_controls.png`, `results/controls/*/summary_wide.csv` | `bm controls-report` | derived from the E3 CSVs |

The diagnostic runs were produced from a working tree with uncommitted changes, so their
manifests record `dirty: true` and the parent commit `3fbc2bf`. The committed package
differs from the code that ran only by formatting, type annotations and one semantically
identical edit (see UPGRADE_HANDOFF §5).

The checkpoint-training runs for E3 started from commit `cb094ee` with uncommitted package
code (`dirty: true`). Their records equal the committed diagnostic runs value for value
(behavioural score and raw loss at every saved step), so they follow the same trajectory.
