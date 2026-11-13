# Stage 2 protocol: E2 exploratory run (written before any E2 training)

**Status:** exploratory. This is not the pre-registered eight-seed study, and nothing here
is a confirmatory test. Written and committed before the first E2 model was trained.

## Task

Generator `e2-v1` (`src/behaviour_mechanism/e2.py`): chunk length uniform on 8–24 except
the held-out lengths 11, 17 and 22; start uniform; adjacent repeat; 70% repeat sequences;
vocabulary 100; length 64. The held-out lengths are an addition to the roadmap's E2 sketch,
made so that generalisation to unseen offsets can be tested.

## Runs declared in advance

| Run | Model | Optimiser | Steps | Seed | Role |
|---|---|---|---|---|---|
| `e2_main_lr3e-4_s0` | historical 4-layer, 4-head, d_model 128 | historical: AdamW, peak lr 3e-4, warmup 100, cosine to 0 | 8,000 | 0 | **primary** |
| `e2_lr1e-3_s0` | same | same but peak lr 1e-3 | 8,000 | 0 | secondary, in case the primary does not learn the task in 8,000 steps |

Both runs save a checkpoint every 250 steps (33 checkpoints). If the primary run learns the
task, mechanism analysis is done on it; the secondary is then reported for behaviour only
unless time allows more. If neither learns the task, that is the result, and no mechanism
conclusions are drawn.

"Learns the task" is defined in advance as validation accuracy on in-distribution scored
positions of at least 0.90 at the final checkpoint. The rule-based reference solutions in
`baseline_table` give the achievable ceiling.

## Evaluation at every saved checkpoint

- **Behaviour** (fixed validation batches, 512 sequences each): accuracy on scored
  positions for in-distribution, held-out and extrapolation lengths; by length and by depth
  inside the repeat; non-adjacent gap probes; shuffled-repeat control; loss on random
  sequences.
- **Mechanism** (128 fixed pairs per corruption, drawn from the test stream): the legacy
  statistic `legacy_patch_effect_size_v1`, and `bm-patch-2` results (attempted, defined,
  undefined, defined fraction, conditional effect) for attention, head, MLP and
  residual sites, denoising and noising, under four corruptions.
- **Calibration:** the same measurement on a hand-built two-layer induction model.

## What would count as an apparent lag, and what would not

A later rise in one curve than another is not by itself evidence of a developmental lag.
For each mechanism curve the report must say whether the change is in the defined fraction
(the model's outputs on the probe inputs) or in the conditional effect (the causal profile
given a defined gap), and whether it survives a change of corruption.

## Amendment 1 (outcome-informed; written at step ~3,500 of the declared runs)

Both declared runs sit on a plateau at steps 2,000–3,500: in-distribution accuracy
0.15–0.18 and held-out-length accuracy 0.02–0.05. That pattern is what a mixture of
positional copiers would produce, not token matching. The declared runs continue to
8,000 steps unchanged.

Because the declared runs may end without learning the task, one contingency run is added
**after seeing these interim results**:

| Run | Model | Optimiser | Steps | Seed | Role |
|---|---|---|---|---|---|
| `e2_2layer_lr1e-3_s0` | 2 layers, otherwise historical (4 heads, d_model 128, MLPs kept) | AdamW, peak lr 1e-3, warmup 100, cosine to 0 | 12,000 | 0 | contingency, outcome-informed |

The roadmap's E2 sketch names a two-layer variant; choosing it, the learning rate and the
longer schedule now is a decision made with knowledge of the interim results. Anything
found with this run is exploratory in a stronger sense than the declared runs, and the
report must say so. The "learns the task" rule (final in-distribution accuracy of at least
0.90) is unchanged.
