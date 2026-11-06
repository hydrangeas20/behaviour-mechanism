# Proposed correction to the September 2026 preprint (DRAFT for author review)

> **Status:** proposal only. It has **not** been published. Zenodo record
> 10.5281/zenodo.22602752 and `paper/behaviour_mechanism_preprint.pdf` are unchanged.
> The author decides whether to issue it, and in what form: an erratum note, a new Zenodo
> version, or both. Evidence for each item is in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md);
> commands are in the README.

---

**Correction to:** H. Aden, *Behavioural and Mechanistic Measurements Diverge During
Small-Model Training* (preprint, September 2026; doi:10.5281/zenodo.22602752).

**Summary.** The reported numbers are reproducible, and an independent re-implementation
re-derives them. Re-analysis and follow-up controls show that the paper's central
interpretation is not supported. The patching measurement used as "mechanistic evidence"
mostly reflects whether the model's *output* differs between the clean and corrupted
probe inputs. Its late rise coincides with a second behavioural transition, which the
behavioural metric did not measure. The separation of 1,700–2,200 steps is therefore a
separation between two behavioural properties on two input distributions. It is not a
separation between behavioural and mechanistic stabilisation. This revised account is
better supported than the original, but it is not definitively established (see
"Remaining uncertainty").

## Corrections

1. **Task (§3.1).**
   - Every structured sequence repeats its first half at a fixed offset of 32 tokens.
     The configured gap range (2–40) is never used.
   - The model reaches accuracy above the 31/32 bound that token-matching induction
     allows on this metric.
   - It performs at chance on repeats at any other offset.
   - The learned behaviour is fixed-offset copying. "Induction" and comparisons to
     induction-head work (§2.1, §5) do not apply.
2. **Patching statistic (§3.3–3.4).**
   - Pairs whose clean/corrupted log-probability gap is below 0.05 are skipped but
     counted as zero in the average.
   - Undefined measurements are therefore reported as 0: 41 of 176 main-run values and
     56 of 88 negative-control values.
   - Values after step 300 cluster at multiples of 1/8 (8 pairs). During the variable
     phase the statistic mostly measures the fraction of probe pairs that are defined.
3. **Interpretation of the separation (§4.1, §7.1, §9).**
   - After accuracy saturates, the model copies the token from half a sequence back on
     *all* sequences.
   - About 800–1,700 steps later it learns to stop copying on non-repeating
     sequences. Non-repeating sequences are exactly the probe's corrupted inputs.
   - The historical training loss shows both phases in every seed. They match the
     idealised levels (3.33 and 2.98 nats).
   - The patching statistic's rise coincides with the second loss drop in all 4 seeds.
   - The sentence "Behavioural stability and mechanistic stability are distinct empirical
     properties" (§9) is not supported by this experiment.
4. **Negative control (§4.2, Figure 3).** The control's patching values do not "decay
   to zero". From step 1,100 on, the measurement is undefined: no probe pair has a gap. The
   control is consistent with the revised account (the model never copies), but it does not
   show a noise floor.
5. **Stability values (§4.1).**
   - Under the sustained-ceiling criterion, T_b = 300 for θ ≤ 0.97, but 350 for
     θ ∈ {0.98, 0.99} and 450 at θ = 1.00.
   - T_m = 2,000 for θ 0.80–0.88 and 2,500 for θ 0.89–0.99, as stated.
   - T_m is interval-censored by the checkpoint grid (no checkpoints between 1,500 and
     2,000).
   - The criterion was adopted after the variance rule failed, and the 21 thresholds are
     nested, not independent.
6. **Head-level localisation (§4.3).**
   - This concerns a fixed-offset copier, measured with 4 probe pairs; no numeric output
     was saved.
   - Follow-up controls show that layer-0 attention is *sufficient* to restore the
     effect, but not the only *necessary* component (item 8).
7. **Real-text track (§5.2).**
   - The "before" model in the 93.7% comparison is the untrained initialisation.
   - The summary numbers have no saved output in the repository.
8. **New controls that qualify the interpretation (not in the preprint).**
   - **Constructed-weights control.** On hand-set copiers the probe returns the values
     derived by hand: ungated undefined, attention-gated 1.0. For an **MLP-gated copier
     it returns 0.0**, even though every pair is defined. A zero from this probe is
     therefore not evidence of an absent mechanism.
   - **Copy-source control** (2 re-trained seeds). A corruption of the copied token is
     defined at every checkpoint. Layer-0 attention fully restores and fully disrupts it
     from step 300 on: the copying pathway is present throughout the period in which the
     original statistic varied.
   - **Noising controls.** The model's reliance on layer-2 attention and MLPs for the
     match cue keeps growing after gating appears, while behaviour changes only modestly.
     Some internal change continues after both behavioural transitions. The original
     statistic does not measure it, and whether it is reorganisation or sharpening is
     unresolved.

## Remaining uncertainty

- The revised account rests on 4 historical seeds, plus 2 re-trained seeds for the
  controls. Its alignment criteria were chosen after inspecting the data.
- A pre-registered 8-seed re-measurement (E1) is designed but has not been run.
- Whether a separation between behavioural and *mechanistic* stabilisation exists on a
  task that requires token-matching induction is untested (E2).

## What remains valid

- The released data are internally consistent.
- The headline numbers reproduce exactly where CPU and GPU arithmetic agree.
- The paper's methodological point stands: agreement between behavioural and mechanistic
  measurements should be tested, not assumed. This experiment, read correctly, shows
  *why* the test must check what each measurement conditions on.

---

*Suggested distribution (author's choice):* a short erratum PDF attached as a new Zenodo
version, with the original kept as version 1, and a link from the README.
