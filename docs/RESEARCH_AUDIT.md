# Research audit: Behaviour–Mechanism

- **Audit date:** 2026-11-05.
- **Baseline:** `main` @ `3fbc2bf`.
- **Branch:** `research/rigor-and-reproducibility`.
- **Scope:** every tracked file: the notebook (56 cells, including its saved outputs), 7
  CSVs, 10 figures, the preprint PDF, the config, the README and `requirements.txt`.

The README and the preprint were treated as claims to check, not as authority. "Verified"
means recomputed from a committed artifact or measured by a run recorded in
`results/diagnostics/`, with its manifest.

---

## A. Executive assessment

**Scientific state.** The historical numbers are genuine and reproducible:
- the extracted pipeline reproduces the step-0 metrics exactly;
- CPU re-runs match the GPU patching statistic checkpoint by checkpoint: seed 0 to within
  0.001 for steps 0–1,400 (28 checkpoints), and seed 1 to within 4e-6 for steps 0–2,500
  (33 checkpoints).

The *interpretation* does not survive the audit. Three findings, each verified, change
what the experiment shows:

1. **The task does not require induction.**
   - The repeat is always at offset 32.
   - The model learns fixed-offset copying: it predicts the first repeated token, and it
     fails at every other offset.
2. **The "mechanistic" statistic is dominated by a skip rule.**
   - Pairs whose clean/corrupted gap is under 0.05 count as 0.
   - The statistic is mostly *the fraction of 8 pairs with a gap*, not the size of the
     recovery.
3. **That fraction tracks a behavioural change the behavioural metric does not cover.**
   - The model first copies half-back tokens on *all* sequences.
   - About 1,000–1,700 steps later it learns to stop copying on non-repeating ones. Those
     are exactly the probe's corrupted inputs.
   - The historical training-loss curves show this second transition in every seed, at
     the same checkpoints as the patching statistic's rise.

The defensible conclusion is narrower and different from the published one:

> Argmax accuracy on clean repeats saturates early. The model's behaviour on
> non-repeating sequences changes later. A patching statistic built on clean-vs-random
> pairs registers only the second change.

The data do not show that behaviour stabilises before "the mechanism". Where a gap exists,
layer-0 patching recovers most of it from the first checkpoints after behaviour emerges.

**Follow-up controls (§J, added 2026-11-06)** support this account without settling it:

- **E4.** On hand-set copiers, the historical probe returns the values expected by
  construction. It also returns **0 for an MLP-gated copier**, a demonstrated blind spot.
- **E3, copy-source corruption.** This corruption is defined throughout training. Its
  layer-0 attention recovery and disruption are 1.00 at every checkpoint from step 300
  (2 seeds).
- **E3, noising.** Reliance on layer-2 components for the match cue keeps growing after
  gating appears. Some internal change continues after both behavioural transitions; the
  historical statistic cannot see it.

**Engineering state (before):**
- a single Colab notebook that cannot run outside Colab;
- unpinned dependencies;
- no tests, no CI, no package;
- two analyses in the preprint (the sustained-ceiling sweep and the real-text summary
  numbers) whose code or outputs were never committed.

**After this branch:**
- a tested package with regression tests against notebook outputs;
- a CLI;
- CI;
- run manifests;
- CPU diagnostic re-runs;
- documentation that separates historical, derived and new results.

## B. Research reconstruction

1. **Question.** Do behavioural and mechanistic measurements of one capability stabilise
   at the same training step? (notebook §0; preprint §1)
2. **Hypotheses.** H0: T_b ≈ T_m. H1: T_b ≠ T_m. Stability was first defined by a
   cross-seed std rule (cell 25), which gives T_b = 0, T_m = 2000. Preprint §3.5 rejects
   that rule and replaces it with a sustained-ceiling rule over θ ∈ [0.80, 1.00]. The
   revision is outcome-informed, and it is disclosed.
3. **Competing hypotheses named by the author.**
   - The circuit reorganises.
   - Multiple implementations exist.
   - Probe sensitivity changes.
   Not named: the task admits a non-induction solution; the statistic measures definedness.
4. **Behavioural measure.** Argmax accuracy at every position of the repeated half of
   clean repeat sequences, including the first repeated position
   (`training.induction_eval_score`).
5. **Mechanistic measure.** For 8 pairs `clean = [c, c]` and `corrupt = [c, r]`, read
   log p(c₁) at position 32. The recovery from patching layer *l*'s clean attention output
   into the corrupted run is (lp_patched − lp_corrupt) / gap. Pairs with |gap| < 0.05
   are skipped. Each recovery is clipped to [−5, 5], summed, and divided by 8. The
   best layer is reported (`patching.legacy_layer_effects`).
6. **Behavioural convergence.** All seeds ≥ θ for 3 consecutive checkpoints.
7. **Mechanistic change.** The same rule applied to the patching statistic.
8. **What would support H1.** T_b < T_m robustly, and the patching statistic tracking
   something other than behaviour.
9. **What would falsify the "mechanism" reading.** The patching statistic's change is
   explained by a change in model outputs on the probe's own inputs. §E shows that this
   is what happened.
10. **Boundaries.** One synthetic task, one small model family, 4 seeds, a single GPU run
    per seed.

**Proposed precise reformulation** (tested in roadmap E1). Let T_acc, T_gate and T_patch
be the per-seed steps at which, respectively:
- clean-repeat accuracy reaches 0.99;
- the log-probability of the half-back token on non-repeating sequences falls below
  ln(1/V) + 1;
- the patching statistic reaches 0.8.

H-align: T_patch = T_gate (within one checkpoint) and T_acc < T_gate for every seed.

## C. Verified evidence inventory

Commands: `bm verify-artifacts` writes `results/derived/artifact_verification.json`;
`bm diagnostics-report` writes `results/diagnostics/summary.csv`.

| # | Result | Artifact | Code path (historical → package) | Provenance | Verified? |
|---|---|---|---|---|---|
| 1 | The committed config is the run config | `configs/config.json` | cell 9 → `config.py` | hash `57c45749eb6d`, equal to the notebook output | **Yes** |
| 2 | Accuracy reaches ≥ 0.99 at step 300 (seed 3: 350) | `results/behaviour.csv` | cell 15 → `training.induction_eval_score` | GPU T4; commit `a5032fc` | **Yes** (recomputed); step-0 values reproduced exactly |
| 3 | Patching trajectories per seed | `results/mechanistic.csv` | cells 17, 22 → `patching.py` | cell 22 has no saved output | **Yes**: CPU equals GPU within 0.001 (seed 0, to step 1,400) and within 4e-6 (seed 1, to step 2,500) (`results/diagnostics/main_s*_to3000`) |
| 4 | Variance rule: T_b = 0, T_m = 2000 | `variance_by_step.csv` | cell 25 → `stability.variance_stable_step` | notebook output | **Yes** |
| 5 | Sustained ceiling: T_b = 300, T_m ∈ {2000, 2500}, "20 of 21 θ" | preprint §4.1; code not committed | → `stability.threshold_sweep` | — | **Partly.** T_m is exactly as claimed. T_b is 300 for θ ≤ 0.97 but **350** for θ ∈ {0.98, 0.99} and **450** at θ = 1.00. The gap range 1,700–2,200 still holds (2,150 at θ 0.98–0.99). `results/derived/sustained_ceiling_sweep.csv` |
| 6 | Seed bootstrap: P(T_b < T_m) = 0.98 | cell 32 output | cell 32 | uses the **rejected** variance rule | Reproducible, but **not valid** for the revised criterion (the preprint says so; the README does not) |
| 7 | Negative control "decays to approximately zero by 1,500" | `negative_control.csv` | cell 35 | hash `584a4b4b2a1c` | **Reinterpreted.** 56/88 rows are exactly 0, i.e. all 8 pairs undefined, and every row from step 1,100 on is 0. The measurement is undefined there, not 0 |
| 8 | Layer 0 carries the effect | `per_layer_patching.csv`, `mechanistic.csv` | cell 28 | — | **Yes**: best layer = 0 in 150/152 rows after step 300 |
| 9 | 3/4 seeds concentrate in one layer-0 head | `figures/per_head_final.png` only (4 pairs) | cell 33 | no CSV | **Not verifiable** numerically. The CPU re-run's best head (16 pairs) is L0H1 for seed 0 from step 1,200 |
| 10 | 64/2 config: patching 0.088 with accuracy 1.0 | `scale_scan.csv` (2 seeds) | cell 37 | config not saved | **Reinterpreted.** The value is the mean of 0.176 and 0.000, and the 0.000 is an undefined measurement. CPU re-runs: the 64/2 model never stops copying on non-repeating sequences within 4,000 steps, and only 9–33% of pairs are defined |
| 11 | Real-text: 93.7% of positions improve; loss 4.626 → 2.443 | preprint §5.2 | cell 53 | **no saved output** | **Not verifiable.** The "before" model is the untrained step-0 model |
| 12 | Real-text depth diagnostic null | `figures/real_text_depth.png` | cells 49, 51 | final losses 2.459 / 2.457 / 2.459 in cell 49 output | **Yes** for the losses (1 seed) |
| 13 | Logit lens | `figures/logit_lens.png` | cell 29 | 1 sequence per seed, no output | Not verifiable; not used for any claim |

New evidence produced in this audit (CPU; `results/diagnostics/*/manifest.json`):

| Run | Config | Steps | Key observations |
|---|---|---|---|
| `main_s0_to3000` | historical, seed 0 | 0–3,000 (schedule of 8,000) | historical statistic reproduced to step 1,400; defined fraction (64 pairs) 24/64 at step 300 → 1/64 at 1,000 → 59/64 at 1,300 → 64/64 at 2,000; log p(half-back \| random sequence) −0.15 until 1,200, then −3.9 at 2,000; offset-8/16/24 accuracy ≤ 0.061 throughout |
| `main_s1_to3000` | historical, seed 1 | 0–3,000 | historical statistic and raw loss reproduced to step 2,500; log p(half-back \| random) ≈ −0.15 from step 350 to 1,400, −0.27 at 1,500, −3.44 at 2,000; defined fraction 3–26/64 until 1,400, 43/64 at 1,500, 63/64 at 2,000; offset accuracy ≤ 0.055 |
| `scale64x2_s0`, `scale64x2_s1` | scale-scan 64/2 preset | 0–4,000 | accuracy 1.0 from step 798; copying on random sequences never gated (log p ≈ −0.15 to −0.23); defined fraction 9–33% |
| `negctrl_s0_to2000` | frac_induction = 0, seed 0 | 0–2,000 | accuracy 0; log p(half-back \| random) = −4.61 (uniform) from step 50; defined pairs fall from 60/64 at init to 0/64 from step 1,250; conditional recovery undefined from then on. Uses patching RNG `seed + 10000`, not the historical `seed + 50000`, so its 8-pair column is not a reproduction of `negative_control.csv` |

## D. Scientific strengths

- **The measurement pipeline is correct as code.** Patching hooks re-run downstream
  computation; patched and functional forward passes agree with the Flax model (tests).
  Clean and corrupted inputs share their first half exactly, and patching all positions
  is equivalent to patching the read position, given causal masking.
- **Provenance is good for a notebook project.** The config hash links the config to the
  outputs, and checkpoint-level CSVs are released. The RNG scheme is deterministic: the
  extracted code reproduces the historical batches and metrics.
- **Self-correction is disclosed.** The flawed variance rule is reported and replaced in
  the preprint.
- **Null results are retained.** Examples: the real-text depth diagnostic, and the
  per-head heterogeneity.
- **The limitations sections are mostly careful.** For example, they say the result does
  not establish an unstable mechanism, and that width and depth are confounded.
- **A negative control and a scale probe exist.** Most portfolio projects have neither.

## E. Scientific weaknesses

### E.1 The task is solved by fixed-offset copying (P0-1)

- `make_induction_batch` always writes `tokens[half:] = tokens[:half]`.
  `induction_gap_min/max` are recorded in the config but never read (test
  `test_gap_config_fields_are_not_used`).
- **Evidence the model uses position.**
  - Token-matching induction cannot predict the first repeated token (position 32), so
    it is bounded at 31/32 = 0.969 accuracy on this metric.
  - Accuracy exceeds that bound at **148/148** checkpoints from step 350 on.
  - On repeats at offsets 8, 16 and 24 (`probes.offset_accuracy`), accuracy stays near
    chance (≤ 0.067) in every CPU run at every checkpoint.
- **Mechanistic plausibility.**
  - A canonical induction circuit needs a previous-token head composing into a later
    induction head (at least 2 layers).
  - The effect localises to **layer 0**, consistent with a single positional head that
    attends 31 back.
- **Impact.**
  - "Induction" in the README, preprint and figure titles should read "fixed-offset
    copying".
  - Comparisons to Olsson et al. do not apply.
  - The head-level "heterogeneous mechanistic organisation" finding concerns a
    positional copier.

### E.2 The patching statistic conflates "undefined" with "zero" (P0-2)

- `activation_patching_effect` divides by `n_pairs` even when pairs were skipped. The
  per-head version divides by the number used: the two are inconsistent.
- **Evidence.**
  - 41/176 main values and 56/88 negative-control values are *exactly* 0.0 at every
    layer. That can only happen if all 8 pairs were skipped.
  - After step 300, the values are quantised at k/8. The median distance to the nearest
    k/8 is 0.0001; 74% lie within 0.01, against 16% for a continuous quantity.
    `figures/derived/patch_statistic_quantisation.png`.
  - The CPU re-run confirms the decomposition directly: historical value ≈ (defined/8) ×
    (conditional recovery).
- **Impact.**
  - Between steps 300 and about 1,200 the statistic mostly measures **how many probe
    pairs have a clean/corrupted gap at all**.
  - The negative control's "decay to zero" means the probe became undefined. That is
    expected for a model that never copies, and it is not evidence about a plateau.
- **Fix (implemented).** `patching.summarise` reports `defined_fraction`, the recovery
  conditional on defined pairs, gaps, and log-probabilities. The legacy value is kept
  unchanged for comparability.

### E.3 The late rise is a behavioural transition on the probe's corrupted inputs (P0-3)

The probe's corrupted sequence `[c, r]` is a uniformly random sequence: the training
"background" distribution, which the behavioural metric never scores.

- **Ungated phase.**
  - After the first transition the model puts p ≈ 0.86 on the half-back token *in random
    sequences*: log p = −0.15 in the CPU re-run, steps 300–1,200, seed 0.
  - That matches the optimum for a copier that cannot tell repeats from non-repeats under
    the ×3 loss weight: q = 3·0.7 / (3·0.7 + 0.3) = 0.875.
  - Clean and corrupted outputs are therefore nearly identical, gap ≈ 0, and pairs are
    skipped.
- **Gated phase.** Later the model learns to copy only when the token at position 32
  matches position 0. In seed 0, log p on random sequences falls to −3.9 by step 2,000,
  the gap opens to about 4 nats, and every pair becomes defined.
- **The historical raw training loss shows both phases** (no re-run needed):
  - plateau near the ungated ideal of **3.33** nats (observed about 3.34–3.38);
  - then a drop to the gated ideal of **2.98** (observed 2.96–3.00);
  - ideals from `artifacts.expected_raw_loss`.
- **Per seed, the second drop and the patching rise coincide.** Exploratory criteria:
  loss ≤ 3.03 and patching ≥ 0.8, each for 3 checkpoints.

| Seed | accuracy ≥ 0.99 | patching ≥ 0.8 | raw loss ≤ 3.03 |
|---|---|---|---|
| 0 | 300 | 1,350 | 1,350 |
| 1 | 300 | 2,000 | 2,000 |
| 2 | 300 | 1,100 | 1,300 |
| 3 | 350 | 2,000 | 2,000 |

- **Seed 1 CPU re-run.** Copying on random sequences stays ungated (log p ≈ −0.15) until
  step 1,400. Over 1,450–2,000 it gates (to −3.44), and the defined fraction rises from
  29/64 to 63/64 over the same checkpoints.
- **Impact.**
  - H1 as stated ("behavioural evaluation is not a trustworthy real-time proxy for
    mechanism") is **not supported**.
  - A loss-based behavioural measure, or accuracy on non-repeating inputs, stabilises
    with the patching statistic.
  - What the data show is that *argmax accuracy on one distribution* is a poor proxy for
    *behaviour on another distribution*.
  - Conditional on a gap existing, layer-0 patching recovers it. Conditional recovery is
    noisy (0.2–1.1) while only 1–26 of 64 pairs are defined, then 0.98–1.0 once the gap
    opens (seed 0 from step 1,200; seed 1 from step 1,500).
  - The historical statistic shows no internal change beyond this behavioural one. The E3
    noising controls (§J) do show further change: downstream necessity grows after
    gating, while the gap grows only modestly. "No internal change" is therefore *not*
    a conclusion of this audit.
- **Alternative explanations still open.**
  - The gating computation may be implemented in a different component (an MLP, or later
    layers) that single-layer attention patching cannot see.
  - The noisy conditional recovery in the ungated phase (0.2–1.0 on 1–24 pairs) may hide
    a real change.
  - Roadmap E3 and E4 address both; see §J for what they found.

### E.4 Other methodological issues (P1)

- **Corruption design.** The corruption replaces the *query token itself* (position 32)
  along with everything after it, so the probe tests "repeat vs not", not "which earlier
  token is attended". There is one read position per pair. Only attention outputs are
  patched, one layer at a time, so redundancy across layers would be invisible.
- **Best-of-layers selection.** Taking the maximum over 4 noisy layer estimates biases the
  statistic upward, which matters early. At initialisation the main run reads
  0.77 ± 0.44 across seeds.
- **No fixed probe set.** New random pairs are drawn at every checkpoint, so cross-seed
  std mixes estimator noise with real differences.
- **Real-text track.**
  - The qualitative "phase change" compares the *untrained* model with step 3,000. Any
    language model improves on most positions relative to initialisation.
  - The notebook markdown attributes red tokens to induction copying without evidence
    (the README does not).
  - One seed, and no causal measurement.

## F. Statistical limitations

- **Replication.**
  - 4 seeds, one run each, on one GPU. Checkpoints from one run are not independent
    replicates, and no analysis here treats them as such.
  - Scale scan: 2 seeds per configuration, 6 probe pairs, final checkpoint only. The README
    table reports means without n.
- **Uncertainty.**
  - No within-checkpoint uncertainty exists in the historical data.
  - The seed bootstrap (cell 32) used the rejected rule, and with 4 seeds a seed
    bootstrap has at most 35 distinct resamples.
  - No new intervals were computed for the historical data. With n = 4, per-seed values
    are reported instead.
- **Criterion sensitivity.**
  - The sustained-ceiling criterion was introduced after the variance rule failed.
  - The 21 thresholds are nested, not independent tests: "20 of 21" is a robustness
    description, not a count of replications.
  - T_m is interval-censored by the checkpoint grid: (1,500, 2,000] for seeds 1 and 3.
- **Multiple comparisons.** Several stability rules and thresholds were examined. No
  claim here depends on a p-value.
- **Missing values.** Undefined patching measurements were coded 0 rather than missing
  (§E.2).

## G. Engineering weaknesses (as found)

- **The notebook is not runnable outside Colab.**
  - Cell 7's `except ImportError` branch references `RESULTS_DIR` before assignment
    (a NameError).
  - The Drive path is the placeholder `[Filepath]_results`.
  - Saved outputs show it actually ran as `sentinelbench_gpt2_results`.
- **Unpinned dependencies.** `requirements.txt` lists 6 names with no versions; JAX and
  Flax versions are known only from a cell output.
- **Missing artifacts and code.**
  - Checkpoints and the training-state pickle are not published.
  - Cells 22, 29 and 53 have no saved outputs.
  - The sustained-ceiling sweep code was never committed.
- **Stale or wrong statements.**
  - Comments say "vocab 512, 3000 steps".
  - The "position-targeted" patching docstring is inaccurate, though equivalent in effect.
  - Raw loss is computed with the *post-update* parameters.
  - `dropout_rate` is unused.
- **README errors.**
  - The file tree lists `behavior.csv`, `behavior_vs_mechanism.png` and
    `behavior_mechanism_research_memo.pdf`, none of which exists.
  - Notebook cell 30 embeds a missing image.
- **No tests, CI, package, logging or provenance capture.**
- **Figures.**
  - `variance_bands.png` draws ±1 std bands outside [0, 1].
  - It marks T_b = 0 and T_m = 2000 from the rejected rule.
  - It linearly interpolates across the unobserved 1,500–2,000 interval.
  - The default matplotlib colours were not checked for colour-vision deficiency.

## H. Prioritised issues

| ID | Sev. | Description | Evidence / location | Impact | Fix | Effort | New experiments? | Status |
|---|---|---|---|---|---|---|---|---|
| P0-1 | P0 | Task solvable by fixed-offset copying; gap config unused | `data.py` (notebook cell 11); accuracy > 31/32; offset probes at chance | "induction" claims and comparisons invalid | relabel; offset-randomised protocol v2 | S (docs) / M (E2) | E2 | Documented; probes implemented; E2 proposed |
| P0-2 | P0 | Skipped pairs counted as 0; statistic ≈ defined fraction | `patching.legacy_layer_effects` (cell 17); exact zeros; k/8 quantisation | T_m and the negative-control reading are not about recovery | report defined fraction and conditional recovery | S | no | **Fixed** (additive; legacy preserved) |
| P0-3 | P0 | Patching rise = behavioural gating transition on the corrupted distribution | raw loss vs ideals; per-seed alignment; CPU log-probs | central conclusion reversed | reinterpret; pre-register E1 | S (docs) / M (E1) | E1 | Documented; E1 proposed |
| P1-1 | P1 | 8 fresh pairs per checkpoint; max over layers; no estimator SE | cell 22 | noise is read as instability | larger fixed pair set; per-layer reporting | S | E1 | `--n-large` implemented; fixed pairs proposed |
| P1-2 | P1 | Outcome-informed criterion; T_b misreported for θ ≥ 0.98; checkpoint gap 1,500–2,000 | preprint §4.1; `sustained_ceiling_sweep.csv` | over-precise claims | correct the numbers; denser checkpoints | S | E1 | Numbers corrected in docs |
| P1-3 | P1 | Bootstrap uses the rejected rule | cell 32 | invalid uncertainty claim | removed from README claims | S | no | Done |
| P1-4 | P1 | 4 seeds; scale scan 2 seeds; no n in tables | CSVs | weak generalisation | 8 seeds | M | E1, E5 | Proposed |
| P1-5 | P1 | Corruption changes the query token; attention-only, single-layer patching | cell 17 | narrow causal claim | alternative corruptions and interventions | M | E3 | **Partly done**: E3 on 2 seeds (§J); more seeds via E1 |
| P1-8 | P1 | Probe can return 0 for a gated mechanism (MLP-computed gate) | E4, `results/controls/e4_constructed/` | historical zeros are uninterpretable as "no mechanism" | report defined fraction; use several corruptions and sites | S | E4 | **Found by E4**; documented |
| P1-6 | P1 | Head-level, logit-lens and real-text numbers lack artifacts | cells 29, 33, 53 | unverifiable | regenerate with saved outputs | M | partly | Marked unverified |
| P1-7 | P1 | Not runnable outside Colab; no pins; no tests/CI | cells 2, 7; `requirements.txt` | no independent reproduction | package, tests, CI, verified environment | M | no | **Fixed** |
| P2-1 | P2 | README file tree and links wrong; emoji title | README | confusion | rewrite | S | no | **Fixed** |
| P2-2 | P2 | Figure issues (bands, rejected-rule markers, palette) | `figures/variance_bands.png` | mild misreading | regenerated figures in `figures/derived/` | S | no | **Fixed** (originals kept) |
| P2-3 | P2 | Stale comments; post-update raw loss; unused dropout | notebook | minor | documented in package docstrings | S | no | Documented |

## I. Research roadmap (summary)

See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md). In order of information gain:

1. **E1.** Pre-registered re-measurement of the historical setup with decomposed metrics,
   8 seeds and dense checkpoints between 1,500 and 2,500. Tests H-align.
2. **E2.** An offset-randomised task that only token matching solves. Tests the original
   question on a task that fits its framing.
3. **E3.** Alternative corruptions and interventions. **Done for 2 seeds** (§J). Extend
   to 8 seeds inside E1.
4. **E4.** A constructed-weights positive control. **Done** (§J).

A larger model is **not** recommended until E1 and E2 are done.

## J. Follow-up controls: E3 and E4 (2026-11-06)

Code: `src/behaviour_mechanism/controls.py` and `constructed.py`.
Runs: `results/controls/*/` (CSV + manifest; commit `9c3c69e`, clean tree).
Regression tests: `tests/test_controls.py` and `test_recorded_e3_e4_results`.
The historical protocol and data are unchanged; these are new measurements.

### J.1 E4: constructed-weights positive control

The three copiers are hand-set in the historical architecture class (2 layers, 2 heads,
d_model 512, LayerNorm made exactly the identity). In all three, layer-0 head 1 copies the
token at q − 31, and layer 1's MLP outputs it when a "match" flag is set.

- **Ungated:** the flag is constant.
- **MLP-gated:** head 0 fetches the token at q − 32 by position, and the layer-0 MLP
  compares it with the current token.
- **Attention-gated:** head 0 attends to q − 32 only if the token there matches the
  current token, so the match is carried by the attention output.

Historical probe, 64 pairs (`probe.csv`):

| Model | Behavioural accuracy | Defined fraction | Patching statistic | Recovery \| defined, L0 | Expected by construction |
|---|---|---|---|---|---|
| ungated | 1.000 | 0.00 | 0.00 (undefined) | — | 0 defined, 0 |
| MLP-gated | 0.969 (= 31/32) | 1.00 | **0.00** | 0.00 | 1 defined, 0 |
| attention-gated | 0.969 (= 31/32) | 1.00 | 1.00 | 1.00 | 1 defined, 1 |

What this shows:

1. The probe pipeline computes what it is supposed to compute. Every value equals the
   hand derivation.
2. A historical 0 can mean "undefined" (ungated) *or* "gated outside attention"
   (MLP-gated). The statistic alone cannot tell these apart from "no mechanism", so zeros
   are uninterpretable.
3. The 31/32 accuracy bound behaves as argued in §E.1. Token-gated copiers sit exactly on
   it; the trained models exceed it, which means they also use position.

The E3 controls separate the two gated designs (`controls.csv`). Under the
**query-token** corruption, layer-0 attention denoising gives 0.00 (MLP-gated) against
1.00 (attention-gated); MLP-0 denoising gives 1.00 for both.

### J.2 E3: corruption × intervention controls on re-trained checkpoints

- **Checkpoints.** From CPU re-runs of seeds 0 and 1, saved by
  `scripts/train_and_save_checkpoints.py`. Their records equal the earlier diagnostic runs,
  which themselves reproduce the GPU history.
- **Pairs.** 128 per corruption, the same pairs at every checkpoint.
- **Sites.**
  - each layer's attention output;
  - each layer's MLP output;
  - the residual stream at the read position after each layer;
  - all attention outputs jointly.
- **Directions.** Denoising (recovery) and noising (disruption).
- **Summaries.** Means of clipped per-pair values over defined pairs. No uncertainty
  intervals were computed.

| Observation | Seed 0 (steps 300–3,000) | Seed 1 (steps 300–2,500) |
|---|---|---|
| Copy-source: defined pairs | 128/128 at every step | 128/128 at every step |
| Copy-source: L0 attention recovery / disruption | 1.00 / 1.00 at every step | 1.00 / 1.00 at every step |
| Query-token: defined pairs | 57 → 22 → 5 (300–1,000), 66 (1,200), 116 (1,300), ≥ 124 after | 103 (300), 10 (1,000), 56–74 (1,400–1,500), ≥ 127 after |
| Query-token: median gap | ≈ 0 until 1,200; 3.4 (1,300); 4.35 (3,000) | ≈ 0 until 1,500; 4.19 (2,000); 4.34 (2,500) |
| Query-token: L0 attention recovery | 0.73–0.96 while few pairs are defined; 0.99–1.00 from 1,300 | 0.74–0.97; 1.00 from 2,000 |
| Query-token: L2 attention disruption | ≤ 0.17 until 1,500; 0.50 (2,000); 0.62 (3,000) | ≤ 0.15 until 1,500; 0.93 (2,000); 0.91 (2,500) |
| Query-token: L2 MLP disruption | 0.15–0.45 until 1,500; 0.74; 0.81 | 0.20–0.26 until 1,500; 0.50; 0.69 |
| Query-token: L0 MLP disruption | 0.62–0.85 until 1,300; 0.95–0.99 after | 0.56–0.67 until 1,500; 0.84–0.96 after |

Earlier-occurrence and historical random-second-half corruptions give the same pattern
as query-token (`summary_wide.csv`).

### J.3 What E3/E4 change in the interpretation

- **Consistent with the revised account (§E.3).**
  - The copying pathway is present and causally carried by layer-0 attention from step
    300 on, through the period in which the historical statistic was 0 or varied.
  - The match cue only produces a measurable gap once gating appears.
  - In the gated phase, the match information passes through layer-0 attention output:
    the attention-gated signature, not the MLP-gated one.
- **Refines it.** "Layer 0 carries the effect" holds only in the *sufficiency* sense. In
  the gated phase, *necessity* is spread over layer-0 MLP, layer-2 attention and layer-2/3
  MLPs.
- **Partly qualifies it.** After gating, the necessity of layer-2 components keeps rising
  (seed 0: L2 attention 0.13 → 0.50 → 0.62 between steps 1,500, 2,000 and 3,000) while the
  query-token gap rises only from 4.05 to 4.35 nats. This is internal change after both
  behavioural transitions, invisible to the historical statistic. It resembles the
  original hypothesis in a weaker form. It could be reorganisation, or sharpening of one
  computation as confidence grows (the copy-source gap also grows, 8.7 → 10.7 nats).
  E3 cannot tell which.
- **Did anything contradict the revised account?** No result contradicts the claim that
  the historical statistic's late rise tracks gating. The post-gating noising change is
  the main finding that the revised account, as first written, did not anticipate. The
  wording in §A and §E.3 has been adjusted.
- **Inconclusive or negative.**
  - The ungated-phase recoveries for match-cue corruptions rest on 3–103 small-gap pairs
    and are noisy.
  - The residual-stream patch for the earlier-occurrence corruption recovers only
    0.02–0.25 in the ungated phase (seed 0, 3–35 defined pairs). This is unexplained.
  - Only 2 seeds were run, and there are no intervals.
