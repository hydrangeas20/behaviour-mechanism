# E1 pre-registration (DRAFT — not frozen, not run)

**Status:** draft for owner review. Once approved, freeze it by committing it with the tag
`e1-prereg-v1` *before* launching any E1 seed. Any later change becomes v1.1 and must be
disclosed.

## Question

In the historical setup, does the historical patching statistic's late rise coincide,
seed by seed, with the model learning to stop copying on non-repeating sequences? And is
the copying pathway's causal profile stable while this happens?

## Fixed design (no changes to the historical protocol)

- Config: `configs/config.json` (hash `57c45749eb6d`), seeds **0–7**, training truncated at
  step 3,000. The learning-rate schedule is still defined by 8,000 steps.
- Historical checkpoints, plus extra measurement checkpoints every 50 steps from 1,550 to
  2,500. Extra checkpoints leave the training RNG untouched: their behavioural evaluation
  uses its own key. This is verified by `test_extra_checkpoints_do_not_change_the_trajectory`
  and by a main-config smoke run, which reproduced the historical step 0/50/100 patching
  values with extra steps at 25 and 75.
- Measurements at every checkpoint:
  - the historical 8-pair statistic (historical RNG, historical steps only);
  - **128 fixed probe pairs** reused at every checkpoint (`--fixed-pairs 128`, pair RNG
    `[seed, 424242]`): defined fraction, conditional recovery per layer, gap;
  - E3 controls on 64 fixed pairs per corruption (`--control-pairs 64`);
  - half-back copy log-probability on non-repeating sequences;
  - offset-8/16/24 accuracy;
  - accuracy;
  - raw loss.

```sh
for s in 0 1 2 3 4 5 6 7; do
  bm diagnose --seed $s --stop-at 3000 --out results/e1/seed_$s \
    --fixed-pairs 128 --control-pairs 64 --extra-steps 1550:2500:50 \
    --label "E1 prereg v1, seed $s"
done
```

**Estimated cost:** about 22–25 min per seed on the 2-vCPU workspace, about 3–3.5 h for 8
seeds run sequentially. The estimate comes from measured components: 0.35 s per training
step, about 5 s per checkpoint with controls. GPU cost is not measured.

## Per-seed transition steps (defined in advance)

All use "first checkpoint at which the condition holds for 3 consecutive checkpoints".

- `T_acc`: accuracy ≥ 0.99.
- `T_gate`: copy log-probability on non-repeating sequences ≤ ln(1/100) + 1 = −3.61.
- `T_def`: fixed-pair defined fraction ≥ 0.95.
- `T_patch`: historical 8-pair statistic ≥ 0.8, at historical checkpoints only.

## Hypotheses and decision rules

- **H-align.** |T_def − T_gate| ≤ 100 steps (2 dense checkpoints).
  - Supported if this holds in ≥ 7 of 8 seeds.
  - Refuted if the median |T_def − T_gate| is > 250 steps.
  - Otherwise inconclusive.
- **H-order.** T_acc < T_gate in every seed. A descriptive check; expected to hold.
- **H-copy-stable.** For the copy-source corruption, layer-0 attention recovery and
  disruption are ≥ 0.95 at every checkpoint from T_acc onward.
  - Supported if this holds in ≥ 7 of 8 seeds.
  - Refuted if it holds in ≤ 4 of 8.
- **H-downstream (exploratory, no decision rule).** Report the trajectory of layer-2
  attention and MLP disruption under the query-token corruption after T_gate, together
  with the query-token gap over the same steps.

## Analysis

- **Unit of analysis:** the seed. Report every seed's T values and the differences
  between them.
- **Descriptive summaries:** the median and range. A seed-bootstrap interval (10,000
  resamples) is labelled descriptive, given n = 8.
- **Within-checkpoint uncertainty:** a pair bootstrap of the fixed-pair defined fraction
  and of conditional recovery (to be added to `bm diagnostics-report` before freezing).
- **Exclusions:** none planned. A seed that fails to train is reported, not dropped.

## What E1 will not show

E1 cannot show that the gate is implemented in any particular component; noising and
denoising measure sufficiency and necessity of the patched sites only. It also cannot
show that the results generalise beyond this task and architecture (E2 addresses the
task).
