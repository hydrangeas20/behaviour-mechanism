# Upgrade handoff (2026-11-05)

## 1. Baseline

| | |
|---|---|
| Original commit | `3fbc2bfab6fd1a1f0d4cc7642581a637ce0a36da` (`main`; parent `a5032fc` "Initial release") |
| Original branch | `main` (the only branch) |
| Working branch | `research/rigor-and-reproducibility` |
| Initial state | clean working tree, 22 tracked files, 2 commits |

The initial repository contained:
- a notebook;
- 7 CSVs;
- 10 PNGs;
- a preprint PDF;
- a config;
- an unpinned `requirements.txt`.

It had no package, tests or CI. Nothing was merged, force-pushed or deleted. Every
historical file is byte-identical on this branch, except `README.md` (rewritten) and
`requirements.txt` (comment header added; package list unchanged).

## 2. Audit findings

Full detail: [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md).

**Strengths**
- a deterministic pipeline;
- a config hash linking config to outputs;
- checkpoint-level data;
- a negative control and a scale probe;
- a disclosed self-correction of the stability rule;
- null results retained.

**Scientific weaknesses (P0)**
1. The task is solvable by fixed-offset copying, and the model does solve it that way.
2. The patching statistic counts undefined pairs as 0, so it mostly measures the
   *defined fraction*.
3. The statistic's late rise is a behavioural transition, the model learning to stop
   copying on non-repeating sequences. It is visible in the historical loss curves.

**Engineering weaknesses**
- The notebook cannot run outside Colab.
- Dependencies are unpinned.
- There were no tests or CI.
- The sweep code and the real-text outputs were never committed.
- The README's file tree is wrong.

**Reproducibility gaps**
- No checkpoints are published.
- Head-level, logit-lens and real-text numbers have no saved outputs.
- The GPU run is not bit-reproducible on CPU beyond step 1,400 (seed 0) or 2,500 (seed 1).

## 3. Implemented improvements

| Area | Files |
|---|---|
| Package | `src/behaviour_mechanism/`: `config`, `data`, `model` (Flax model plus functional forward with cache/patch), `training`, `patching` (per-pair records; legacy and decomposed aggregates), `probes`, `stability`, `artifacts`, `figures`, `diagnose`, `provenance`, `cli` |
| CLI | `bm verify-artifacts`, `bm figures`, `bm diagnose`, `bm diagnostics-report`, `bm env` |
| Tests | `tests/`: 23 tests. These include exact regressions against notebook outputs: the config hash, the first printed batch, the step-0 behavioural score (exact) and the step-0 patching value. They also check that functional and Flax forward passes are equal, identity-patch invariance, head/layer patch equivalence, downstream-only patch effects, the legacy aggregate semantics (skipped pairs = 0, all-skipped = exact 0, clipping, head normalisation), the stability rules, the artifact consistency checks, the probes at init, and a CLI smoke run with manifest and immutability checks |
| CI | `.github/workflows/ci.yml`: Python 3.12 and 3.13; ruff, ruff format, mypy --strict, pytest, `bm verify-artifacts --strict` |
| Environment | `pyproject.toml`; `requirements/verified-cpu.txt` (the exact versions the suite passed with) |
| Derived analyses | `results/derived/artifact_verification.json`, `sustained_ceiling_sweep.csv` |
| New experiments | `results/diagnostics/` (5 runs, with manifests), `results/diagnostics/summary.csv` |
| Figures | `figures/derived/`: historical trajectories with loss ideals, quantisation histogram, threshold sweep, diagnostics trajectories (validated palette, direct labels). The original figures are unchanged |
| Docs | `README.md` (rewritten, banner at `assets/behaviour-mechanism-header.png`), `docs/RESEARCH_AUDIT.md`, `docs/RESEARCH_ROADMAP.md`, `docs/PROVENANCE.md`, this file |

**Bugs fixed:** none in the historical code path. Its behaviour is reproduced on purpose.
The inconsistent normalisation, the zero-for-undefined coding and the fixed-offset data are
documented and decomposed, not silently changed. In the package code, a patching
recompilation per pair was fixed: the target token and position were static jit arguments.

## 4. Scientific corrections

- **Metric definitions.** Unchanged. The historical patching statistic is reproduced
  exactly and reported beside its two components: the defined fraction and the
  conditional recovery.
- **Evaluation procedures.**
  - The behavioural metric includes the first repeated position, which token matching
    cannot predict. This is documented.
  - Added probes: offset-shifted repeats, and the half-back copy log-probability on
    non-repeating sequences.
- **Experimental assumptions.** "Induction" is not established. The model is a
  fixed-offset copier.
- **Result interpretation.**
  - "Behaviour stabilises before mechanism" is withdrawn.
  - Replacement: argmax accuracy on clean repeats saturates before the model's behaviour
    on non-repeating sequences changes, and the patching statistic tracks the latter.
- **Statistics.**
  - T_b is 350, not 300, for θ ∈ {0.98, 0.99}, and 450 at θ = 1.00.
  - The seed bootstrap's P = 0.98 used the rejected rule and is not a valid uncertainty
    statement.
  - The negative control's zeros are undefined measurements.
- **Do the historical results remain valid?** As *numbers*, yes: they are reproduced. As
  evidence for the README and preprint *interpretation*, no. The preprint is unchanged and
  predates the audit.

## 5. Verification

Executed in this workspace (Linux x86-64, 2 vCPU, Python 3.13.18, JAX 0.12.0.1 CPU,
Flax 0.12.12, Optax 0.2.8):

| Command | Result |
|---|---|
| `pip install -e ".[dev]"` | ok |
| `python -m pytest -q` | **23 passed**, 0 failed, 0 skipped (about 25 s) |
| `ruff check .` / `ruff format --check .` | pass |
| `mypy` (strict, `src/`) | pass, 13 files |
| `bm verify-artifacts --strict` | all 3 consistency checks true; variance rule reproduces T_b = 0, T_m = 2000 |
| `bm figures`, `bm diagnostics-report` | ok |
| `bm diagnose …` × 5 (see `artifacts/logs/diagnostics.log`, not committed) | all complete: main s0 1,092 s, main s1 1,139 s, 64/2 s0 306 s, 64/2 s1 304 s, negative control s0 721 s |
| GitHub Actions | run 42151304796 (pull_request, head `ff89007`): `cpu (3.12)` and `cpu (3.13)` both succeeded. Draft PR #1 |

**Skipped / not run**
- GPU runs.
- Full 8,000-step runs.
- Seeds 2–3 re-runs.
- The 128/4 and 256/4 scale presets.
- The real-text track; it needs Project Gutenberg network access, which this workspace
  does not have.
- The notebook itself, which requires Colab and Drive.

**Environment limitations**
- The diagnostic manifests record `dirty: true` at parent commit `3fbc2bf`, because the
  runs started before the package was first committed. While they ran, the
  diagnose/patching/probes/training code was changed only by `ruff format`, type
  annotations, and one semantically identical edit in `probes._score` (an unused variable
  removed). No snapshot of the exact pre-format files was kept.

**Remaining defects**
- The notebook is still non-runnable outside Colab. It is kept unchanged as the historical
  record.
- `diagnose` draws new pairs per checkpoint (fixed pairs are proposed in E1).

## 6. Scientific conclusions

**Established**
- The historical trajectories, reproduced.
- Fixed-offset copying.
- The k/8 structure of the patching statistic and its undefined zeros.
- Per-seed alignment of the patching rise with the second loss drop (4/4 historical seeds,
  exploratory criterion).
- In 2/2 CPU re-runs, alignment with gating on non-repeating sequences.
- The 64/2 model does not gate within 4,000 steps.

**Uncertain**
- Whether the gating computation is implemented in a component that single-layer attention
  patching misses.
- Whether conditional recovery changes in the ungated phase (too few defined pairs to tell).
- Generality beyond 2 re-run seeds.

**Plausible alternatives**
- Gating may be realised partly in MLPs or later layers.
- The 64/2 result could be slower training rather than a capacity limit: 4,000 steps may
  simply be too few.

**Do not conclude**
- Anything about induction heads.
- That behavioural evaluation lags or leads "mechanism".
- Anything about larger models or safety-relevant capabilities.
- That the negative control "decays to zero".

## 7. Remaining work

- **P0:** none left undocumented. The P0 *experiments* (E1, E2) are proposed, not run.
- **P1:**
  - fixed probe set and denser checkpoints (E1);
  - alternative corruptions and interventions (E3);
  - constructed-weights control (E4);
  - 8 seeds;
  - head-level, logit-lens and real-text results with saved outputs.
- **P2:**
  - optionally rename "induction" throughout the notebook markdown (historical file, left
    unchanged);
  - decide whether to post an erratum to the Zenodo preprint;
  - choose a licence (none present).

## 8. Research roadmap

[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md). Highest value:
1. **E1:** pre-registered re-measurement, 8 seeds; about 3.5 h on CPU to step 3,000.
2. **E2:** offset-randomised task; about 6 h on CPU.

## 9. Reproducibility assessment

An independent researcher can currently:
- install the package and run all tests;
- recompute every number in `results/derived/`;
- re-run any seed of the main, negative-control or scale configs on CPU. This matches the
  historical GPU values until floating-point divergence, at about step 1,400–2,500.

Not available:
- historical checkpoints;
- head-level, logit-lens and real-text numbers;
- the preprint's original sweep code (re-implemented here from its formula).

## 10. Exact next steps

```sh
git clone https://github.com/hydrangeas20/behaviour-mechanism && cd behaviour-mechanism
git checkout research/rigor-and-reproducibility
pip install -e ".[dev]" && pytest -q && bm verify-artifacts --strict
# reproduce one diagnostic (about 20 min CPU):
bm diagnose --seed 0 --stop-at 3000 --out results/diagnostics/repro_s0 && bm diagnostics-report
```

**Decisions for the owner**
1. Review the draft PR. Merging is yours.
2. Decide whether the README's revised headline is acceptable. It reverses the preprint's
   interpretation.
3. Decide whether to publish a Zenodo erratum or new version.
4. Approve compute for E1, the 8-seed re-measurement. It is feasible on CPU (about 3.5 h)
   or a GPU (minutes; unverified).
5. Choose a licence.
