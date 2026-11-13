# Research roadmap

**Status (2026-11-06).**

- **E4:** done.
- **E3:** done for 2 seeds (see audit §J).
- **E1:** infrastructure implemented and tested. The pre-registration is drafted in
  [E1_PREREGISTRATION.md](E1_PREREGISTRATION.md). The 8-seed run has **not** been run and
  awaits approval.
- **E2:** task, baselines and an **exploratory** run are done (Stage 2; one seed per
  configuration; see [STAGE2_REPORT.md](STAGE2_REPORT.md)). The 8-seed confirmatory study,
  the two-layer attention-only variant and the no-repeat control have **not** been run.
- **E5, E6:** not started.

None of the experiments below has been run unless it says so. "CPU diagnostic" means the
small re-runs in `results/diagnostics/` (see [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) §C).

The audit changes which question comes next. The historical result is a gap between
"argmax accuracy on clean repeats saturates" (about step 300) and "the patching statistic
saturates" (steps 1,100–2,000). The audit finds that:

1. the task can be solved by fixed-offset positional copying, and the model does so;
2. the patching statistic is mostly the fraction of 8 probe pairs whose clean/corrupted
   gap is at least 0.05; and
3. its rise coincides, seed by seed, with a second drop in training loss. That drop is
   what you would expect when the model learns to stop copying on non-repeating
   sequences.

The most informative next experiments test that account directly. Scaling up the model is
not among them.

Resource figures are measured on this workspace (2-vCPU x86-64, JAX CPU), where a training
step of the main model takes about 0.33 s and a full diagnostic checkpoint about 1–2 s.
GPU estimates are **unverified**.

| Priority | Experiment | Cost (CPU, measured basis) | Changes the conclusion if… |
|---|---|---|---|
| 1 | E1: pre-registered re-measurement, 8 seeds (now includes E3 controls at every checkpoint) | about 22–25 min per seed (measured components), about 3–3.5 h total | patching transitions do **not** line up with the gating transitions |
| 2 | E2: offset-randomised task | ~8 × 45 min (8,000 steps) | separation survives when only token matching works |
| — | E3: corruption and intervention controls | **done, 2 seeds**: about 20 min to train checkpoints, about 2 min of controls per seed | — |
| — | E4: constructed-weights positive control | **done**: under 1 min | — |
| 5 | E5: 64/2 factorial (width × depth) | ~4 × 2 × 6 min | only after E1, using decomposed metrics |
| 6 | E6: patching on the real-text models | GPU recommended | low priority until E1–E4 are done |

---

## E1. Pre-registered re-measurement of the historical setup

1. **Question.** Does the patching statistic's late rise track a change in *recovery* or a
   change in *behaviour on non-repeating sequences*?
2. **Hypotheses.**
   - H-align: for each seed, the step at which the defined fraction reaches 1 equals,
     within one checkpoint, the step at which the half-back copy rate on random sequences
     falls below 0.1.
   - H-recovery: conditional recovery (best layer) stays at least 0.9 from the first
     checkpoint with 8 or more defined pairs onward.
3. **Competing explanations.** A genuine reorganisation of the copying computation after
   behaviour saturates. If so, conditional recovery changes while the defined fraction is
   already 1.
4. **Design.** The historical config, unchanged. Exact commands and decision rules are in
   [E1_PREREGISTRATION.md](E1_PREREGISTRATION.md) (draft).
   - `--fixed-pairs 128`: one fixed pair set reused at every checkpoint. Implemented;
     `test_fixed_pair_records_match_historical_per_pair_code`.
   - `--control-pairs 64`: E3 at every checkpoint. Implemented.
   - `--extra-steps 1550:2500:50`: closes the checkpoint gap without changing the training
     RNG. Implemented; `test_extra_checkpoints_do_not_change_the_trajectory`, plus a
     main-config smoke run that reproduced the historical step 0/50/100 values.
   - **Still to add before freezing:** a pair bootstrap for within-checkpoint
     uncertainty.
5. **Model and task.** As historical (4L/4H, d = 128, fixed-offset repeats).
6. **Baselines.** The historical statistic, recomputed with the historical RNG. The CPU
   diagnostic reproduces it exactly while CPU and GPU trajectories agree.
7. **Controls.** The negative control (`frac_induction_sequences = 0`), with undefined
   checkpoints reported as missing, not as 0.
8. **Measurements.**
   - accuracy;
   - raw loss on a fixed held-out batch, not the training EMA;
   - copy rate;
   - defined fraction;
   - conditional recovery per layer and per head;
   - the gap distribution.
9. **Statistics.**
   - Treat the seed as the unit of analysis.
   - Compute the per-seed transition-step differences (patching minus copy-rate) and
     report all of them.
   - Report the median and a seed-bootstrap interval, labelled as descriptive given
     n = 8.
   - Measure within-checkpoint estimator error by bootstrapping over pairs.
10. **Resources.** CPU, about 25 min per seed to step 3,000; about 3.5 h for 8 seeds on 2
    vCPUs. Seconds per seed on a GPU (unverified).
11. **Success / failure.**
    - H-align is supported if at least 7 of 8 seeds align within one checkpoint.
    - H-align is refuted if the median misalignment is more than 2 checkpoints.
12. **Information gain.** High. It decides whether the headline is about mechanism or
    about the choice of behavioural metric.
13. **Limitations.** Same small model and synthetic task.

## E2. Offset-randomised task — exploratory run DONE; confirmatory study NOT RUN

Exploratory findings (one seed; [STAGE2_REPORT.md](STAGE2_REPORT.md)):
- The historical shortcut is at chance on E2; a positional-mixture shortcut worth 0.136
  remains.
- The lr 3e-4 configuration did not learn the task in 8,000 steps; lr 1e-3 did.
- Under the legacy statistic, the apparent lag ranges from −4,250 to +2,750 steps
  depending on the corruption.
- No behaviour–mechanism lag is resolved at a 250-step checkpoint grid.

The confirmatory design in §10 of the report supersedes points 9–11 below. The original
design notes follow.

1. **Question.** When positional copying cannot solve the task, do accuracy and patching
   still saturate at different times?
2. **Hypothesis.** The original H1 (T_b ≠ T_m), with a token-matching mechanism required.
3. **Competing explanations.**
   - Separation is specific to the positional shortcut.
   - The patching statistic still tracks behaviour on the corrupted distribution.
4. **Design.**
   - Draw the chunk length and start position per sequence (for example, chunk length in
     [8, 24] and start in [0, T − 2·len]).
   - Score only positions that token matching can predict.
   - Use a corruption that keeps the query token and breaks only its earlier occurrence
     (see E3).
   - This is a **new protocol** (v2). Results must not be pooled with the historical ones.
5. **Model and task.** The historical architecture, plus a 2-layer attention-only variant
   in which a canonical two-head induction circuit is expected.
6. **Baselines.** The historical task, trained with the same seeds.
7. **Controls.**
   - Negative control: no repeats.
   - Shuffled-repeat control: repeat with permuted order.
8. **Measurements.**
   - E1 metrics;
   - offset-generalisation accuracy;
   - previous-token-head and induction-head attention scores.
9. **Statistics.** As E1, with 8 seeds.
10. **Resources.** CPU, about 45 min per seed for 8,000 steps; about 6 h for 8 seeds.
11. **Success / failure.**
    - Supported: T_b < T_m in at least 7 of 8 seeds under each pre-registered criterion,
      including a loss-based behavioural criterion.
    - Otherwise not supported.
12. **Information gain.** High. It tests the original research question on a task that
    matches its "induction" framing.
13. **Limitations.** Synthetic task; offset distribution chosen by the experimenter.

## E3. Corruption and intervention controls — DONE for seeds 0 and 1 (audit §J.2)

Results, in brief:
- The copy-source corruption is defined throughout. Its layer-0 attention recovery and
  disruption are 1.00 at every checkpoint.
- Match-cue corruptions become defined only at gating. They then show the
  attention-gated signature.
- Downstream necessity (layer-2 attention and MLPs) keeps rising after gating.

What remains: more seeds and denser checkpoints (inside E1), and per-head sites. The
original design notes follow.

- **Corruptions.**
  - (a) the historical one: random second half;
  - (b) change only the token at position `half`;
  - (c) keep `half` and change position 0, so the earlier occurrence no longer matches;
  - (d) keep both and change position 1, so the copy source changes.
- **Interventions.**
  - layer attention output (historical);
  - per head;
  - MLP output;
  - residual stream;
  - all layers jointly, to detect redundancy.
- Run **denoising and noising** in both directions.
- **Report.** Recovery under each combination, at the checkpoints where the historical
  statistic was 0, k/8 and 1.
- **Success criterion.** A causal claim about "layer 0 implements the behaviour" is kept
  only if it holds under at least 3 corruptions.

## E4. Constructed-weights positive control — DONE (audit §J.1)

Result: the probe returns the hand-derived values on all three copiers, and it reports
**0 for an MLP-gated copier**, a blind spot. The original design notes follow.

Hand-set two models and check what the probe reports on each:
- a pure positional copier (layer-0 head attending `half − 1` back);
- a gated copier.

Expected results:
- the defined fraction is 0 for the positional copier;
- conditional recovery for the gated copier is 1 at layer 0.

If the probe does not return these known values, every historical patching number is
uninterpretable. Cost: under 1 h, no training.

## E5. Width × depth factorial for the 64/2 discrepancy

Run it only after E1. Use d ∈ {64, 128} × layers ∈ {2, 4}, 4 seeds each, with the
decomposed metrics. The historical 64/2 seed-1 value (0.0) is an all-pairs-skipped
measurement. Whether 64/2 models stay ungated copiers is the first thing to check (the CPU
diagnostic in `results/diagnostics/scale64x2_*` is a start).

## E6. Real-text patching

The real-text track has no causal measurements, one seed, and compares step 0 with step
3,000. Defer until E1–E4 settle the synthetic case.

## Not recommended now

- **Larger models.** Scaling adds cost without addressing the measurement-validity issues.
- **More thresholds for the sustained-ceiling sweep.** The 21 thresholds are not
  independent tests, and the result is already bounded by the checkpoint grid.
