# Research Blueprint — Embryo Development as a Latent Dynamical System

This is the crystallized research programme: everything from the design conversation that survived scrutiny, organized as a document to make research decisions against for the life of the project. Companion files: `HANDOFF.md` (session-continuity technical detail), `BLUEPRINT_SUMMARY.md` (the short, daily-reference version of this document — Part IX below).

Every section answers one question: *why am I implementing this?*

---

# Part I — Scientific Foundation

**Motivation.** Time-lapse embryo imaging is now standard in IVF. Every existing ML approach to it — including this project's own starting pipeline — treats it as frame or window classification. That discards exactly the structure a clinician actually reasons about: not "what stage is this frame," but "how is this embryo *developing*, and is something going wrong." Morphokinetic timing (not just sequence) is already known in the clinical literature to carry prognostic signal that a per-frame classifier structurally cannot use.

**Knowledge gap.** No existing method treats embryo development explicitly as a latent dynamical system with a principled, minimal state — and, more specifically, no one has tested whether such a treatment actually adds value here, or whether the natural mathematical coordinates for that system match the discrete clinical taxonomy already in use.

**Central scientific question.** Is human embryo development, as captured by time-lapse imaging, governed by a shared, low-dimensional dynamical law — and do the discrete clinical categories (phase, arrest) correspond to that law's structure, or to something else?

**Irreducible question underneath it.** Does a *shared* dynamical law exist across individuals at all — i.e., is between-embryo variability explainable as different initial conditions and fixed parameters of one generative process, rather than requiring a fundamentally different process per individual? Everything else (dimensionality, unification of categories) is only meaningful if this holds.

**Core hypotheses** (pruned to what is actually testable with the resources this project has — see Part VI for what was deliberately excluded):

- **H1**: explicit dynamics modeling improves prediction/explanation of embryo image sequences beyond frame/window-independent classification, on tasks a classifier structurally cannot solve well.
- **H2**: a discrete regime variable is necessary — some trajectories exhibit qualitatively distinct *future* dynamics that continuous state alone does not explain.
- **H3**: biologically-derived constraints (chronology, monotonicity) improve sample efficiency or robustness beyond generic regularization of equal magnitude.
- **H4** (aspirational, deferred — not in the near-term budget): the mathematically natural coordinate system for this process is more fundamental than the hand-designed biological state; specifically, "regime" may reduce to a local dynamical property (velocity collapse, multistability) of a single continuous coordinate rather than being an independent axis.

**Expected discoveries**: whether dynamics modeling has any measurable value here at all (informative either way); whether regime is a real independent structure or an emergent property of continuous dynamics (the latter would be a *more* elegant, more publishable outcome than the former, not a lesser one); whether prior biological knowledge is load-bearing or decorative.

**Alternative possible discoveries**: no benefit anywhere (a valid negative finding if rigorously controlled); benefit confined to a narrow task subset (a scoped, still-real claim); constraints actively harming rare-but-legitimate deviations (a specific, useful negative finding about over-regularization).

**Falsification conditions**: H1 is falsified by no measurable advantage on any temporal-coherence task after a frame-shuffle sanity check rules out a trivial task-design artifact. H2 is falsified by no improvement of a regime-aware model over a matched-capacity continuous-only model. H3 is falsified by no advantage over a placebo/scrambled-constraint control at equal regularization strength.

**Why this matters scientifically**: it's a direct, real-world test — in a genuinely underexplored imaging domain — of whether the increasingly assumed value of "treat medical time series as latent dynamical systems" actually holds, or is unnecessary machinery for this specific problem.

**Why a negative result on H1 would still be valuable**: it is arguably the single most useful possible outcome of the whole project. A rigorously controlled "no" settles, for this domain, a question that is currently just assumed rather than tested — and prevents the field from investing further effort in a direction that doesn't pay off here.

---

# Part II — Mathematical Formulation

**Objects.**

| Symbol | Object | Observable? | Identifiable? |
|---|---|---|---|
| `t∈[0,T]`, `t_k` | Continuous time / discrete, possibly irregular sampling times | `t_k` known; `t` inferred | — |
| `x(t)∈𝒳⊆ℝᵈ` | Hidden biological state | No | Only up to a diffeomorphism unless externally anchored |
| `θ∈Θ` | Static, individual-specific parameters (`dθ/dt≈0`) | No (from images) | No (from images alone — needs an auxiliary channel) |
| `r(t)∈ℛ={1,…,K}` | Discrete regime, càdlàg | Indirectly, via qualitative shifts in `h` | Only given a sufficient observation horizon |
| `τ(t)` | Sojourn time since last regime jump — **derived**, not primitive | — | Fully determined by `r(·)` |
| `f_r(x,t;θ,u)` | Regime-dependent drift | No | Up to the same equivalence class as `x` |
| `σ(x,t;θ)` | Diffusion coefficient | No | More identifiable than `f` (quadratic-variation estimators are local; drift estimation is not) |
| `Q(x,τ;θ)` | Regime transition generator | No | Conditional on `r` being identifiable at all |
| `h(x,r,t)` | Observation/image-formation map | This *is* what's observed | Generically non-injective — components of `x` in `ker(dh)` are structurally unlearnable, not just data-limited |
| `ε_k~𝒩(0,Σ_obs(x))` | Observation noise, heteroscedastic | — | Confounded with `σ` without an independent noise channel |
| `g(x)`, `ℓ_k=g(x(t_k))+η_k` | Readout map / observed labels | Labels are observed, noisily | As identifiable as `x`, further limited by annotation noise |
| Uncertainty | The posterior `p(x,r,θ∣y_{1:N})` — not a primitive | — | Derived; a property of the observer, not the system |

**Generative model.**
```
θ ~ p(θ)
x(0) ~ π₀(·;θ),  r(0) = r₀

dx(t) = f_{r(t)}(x(t);θ,u(t)) dt + σ(x(t);θ) dW(t) + Σᵢ Δᵢ dNᵢ(t)

ℙ(r(t+dt)=r′ | r(t)=r, x(t), τ(t)) = Q_{r,r′}(x(t),τ(t);θ) dt + o(dt)
τ(t) = t − sup{s≤t : r(s⁻)≠r(s)}

y_k = h(x(t_k), r(t_k), t_k) + ε_k
ℓ_k = g(x(t_k)) + η_k

𝒟 = {(t_k, y_k, ℓ_k)}   ← the only thing ever actually observed
```

**State-space simplification chain** (each step's assumption, and which experiment tests it):
1. General stochastic hybrid system (above).
2. Remove regime-switching (`K=1`) → plain nonlinear SDE. *Assumes* one smooth field suffices — tested by H2/E2.
3. Remove `u(t)`/time-inhomogeneity → autonomous SDE. *Assumes* culture-condition drift is negligible or absorbed into `θ` — untested, a stated limitation (Part VI).
4. Remove stochasticity (`σ→0`) → deterministic ODE (the Neural-ODE reading). *Assumes* cross-individual variability is entirely `θ`/`x(0)` variation, not path-wise noise — a secondary diagnostic within E1/E2, not a standalone experiment given current resourcing.
5. Linearize `f` → linear state-space model. *Assumes* no strong nonlinear/multistable structure in the tested operating range — tested directly by whether Linear SSM (E1) already suffices.
6. *Separate branch*: discard continuous `x` entirely, model only `r(t)` semi-Markov — the HSMM, deliberately deferred (Part V) until the cheaper E2 approximation justifies it.

**Continuous vs. discrete time**: resolved by elimination in favor of a hybrid jump-diffusion, Markov-modulated system. Pure-continuous cannot represent genuine discontinuities (division count); pure-discrete-time conflates the imaging *sampling schedule* with the biological process's own timescale (real elapsed-time data exists in the raw dataset — see `HANDOFF.md` §14 — but is currently unused); pure-event-driven cannot capture genuine continuous within-regime variation.

**Constraints** — hard (state-space membership, regime-transition sparsity for irreversibility) vs. soft (monotonicity, bounded velocity, minimum duration — imposed as penalties on estimation, never hard restrictions on the searched model class, because real biology can violate them in informative ways that a hard constraint would make the model incapable of explaining).

**What is assumed vs. what must be discovered.**

| Assumed (part of the modeling ontology, not being tested) | To be discovered (the actual experimental targets) |
|---|---|
| The general hybrid jump-diffusion generative form exists | Whether `K=1` or `K>1` fits better (H2) |
| The state/parameter split (ploidy, competence → static `θ`; progression/regime/quality → dynamic `x(t)`) | Whether `σ≈0` (near-deterministic) suffices, or genuine stochasticity is needed |
| `h` exists and is at least partially informative | The actual functional form of `f` — linear vs. nonlinear (H1) |
| — | Whether the biological or a data-derived coordinate system is closer to fundamental (H4, deferred) |
| — | The dimensionality of the quality residual `q` |

---

# Part III — Research Questions

**RQ1 (from H1).** *Does explicit dynamics modeling improve prediction/explanation over frame-independent classification?*
Why it matters: gates the entire remaining programme. Evidence required: statistically significant improvement (multiple seeds, confidence intervals) on temporal-coherence tasks, surviving a frame-shuffle sanity check. Experiment: Identity/Persistence/Linear SSM vs. classifier on early-transition prediction and missing-frame imputation. Conclusions →
- Clear positive → proceed to RQ2/RQ3.
- Clear negative (post-sanity-check) → do not build RQ2/RQ3; execute the pivot directions (Part VI); write up as a rigorous negative benchmark.
- Positive on a subset of tasks only → narrow the project's scope to that subset; still proceed, with a narrower claim.

**RQ2 (from H2).** *Is a discrete regime necessary, or does continuous state already explain apparent regime-like behavior?*
Why it matters: this is the practically-testable core of the most interesting long-term question in the whole programme. Evidence required: a matched-capacity likelihood/prediction gap concentrated on anomalous windows specifically. Experiment: simplified 2-regime switching model vs. continuous-only SSM. Conclusions →
- Regime clearly helps, stably trained → centerpiece finding; consider a later upgrade to a full duration-aware HSMM.
- No measurable benefit → simplifies the project to continuous-only; check whether "arrest" still manifests as a local-velocity collapse of the continuous state — a soft positive signal for the long-term unification narrative even without a formal regime variable.
- Training instability/inconclusive → report honestly and stop; leave regime-existence open rather than force a conclusion.

**RQ3 (from H3).** *Do biologically-derived constraints improve sample efficiency or robustness beyond generic regularization?*
Why it matters: tests whether the project's "biology-informed" framing is substantive or decorative. Evidence required: benefit over a placebo/scrambled-constraint control at matched regularization strength, ideally across multiple data-budget levels. Experiment: constrained vs. unconstrained vs. placebo-constrained variants of the RQ1/RQ2 model. Conclusions →
- Clear, placebo-robust benefit → real, publishable finding.
- No benefit beyond placebo → data already implicitly encodes the constraint at this sample size; drop the machinery, still informative.
- Constraints hurt on rare legitimate deviations → a specific, worth-reporting negative finding about over-regularization.

**RQ4 (from H4, aspirational, deferred).** *Does a data-derived coordinate system reveal that the hand-designed state is a suboptimal projection of something more fundamental?*
Why it matters: the highest novelty ceiling in the whole programme. Evidence required: reproducible (across seeds and across ≥2 independent unsupervised methods) improvement in Markov-gap/smoothness diagnostics, generalizing to held-out patients. Experiment: unsupervised slow-mode discovery (diffusion-map/TICA/Koopman-style) vs. the RQ1–RQ3-validated biological state. Conclusions →
- Data-derived coordinate wins clearly → potentially the strongest possible paper this project can produce; requires substantial additional robustness work before publication.
- No difference → a genuinely positive result for the interpretable-first design — still worth reporting, less spectacular.
- Inconclusive/unstable unsupervised method → time-box, report as a limitation, do not let it consume the remaining budget.

RQ4 is explicitly **not** part of the near-term plan (Part VIII). It is contingent on RQ1–RQ3 succeeding and on additional time/resources becoming available.

---

# Part IV — Experimental Roadmap (organized by scientific dependency)

```
M0: baseline reproduction + embedding cache
        │
        ▼
E1 (RQ1)  ◄── the unlocking experiment; nothing below is scientifically meaningful without it
   ┌────┴────┐
   ▼         ▼
E3 (RQ3)   E2 (RQ2)     ◄── both depend only on E1, not on each other; can run in either order.
                              (The M1–M3 week-ordering in Part VIII does E3 before E2 for
                              engineering-cost reasons — that is an implementation choice,
                              not a scientific dependency.)
   └────┬────┘
        ▼
E4 (RQ4, deferred)  ◄── needs a validated biological state (post E1–E3) to compare against
```

**E1** — *Objective*: test H1/RQ1. *Required data*: existing labeled cohort, video-level splits, existing model's cached embeddings. *Required implementation*: Persistence, Identity Dynamics, Linear SSM; frame-shuffle ablation. *Evaluation*: early-transition prediction, missing-frame imputation, held-out patients, multiple seeds. *Possible outcomes*: clear win / clear loss / partial win. *Consequences*: gates all downstream work (see Part III).

**E3** — *Objective*: test H3/RQ3, built on E1's model. *Required data*: same, plus the phase-ontology logic already present in the dataset code. *Required implementation*: chronology/monotonicity auxiliary loss terms; a scrambled-constraint placebo control. *Evaluation*: sample-efficiency curve at multiple data-budget levels. *Possible outcomes/consequences*: as in RQ3 above.

**E2** — *Objective*: test H2/RQ2, built on E1's model. *Required data*: same, ideally with some anomalous/arrested examples identifiable (even informally curated). *Required implementation*: a simplified 2-regime switching augmentation (not full HSMM). *Evaluation*: matched-capacity likelihood gap, concentrated on anomalous windows. *Possible outcomes/consequences*: as in RQ2 above.

**E4** (deferred) — *Objective*: test H4/RQ4. *Required data*: same cohort, no labels used in the unsupervised fitting step. *Required implementation*: an unsupervised slow-mode discovery method (≥2 independent methods, e.g. diffusion map and TICA, for cross-validation of the finding itself). *Evaluation*: correlation with the validated biological state, Markov-gap/smoothness comparison, held-out generalization. *Possible outcomes/consequences*: as in RQ4 above.

---

# Part V — Minimal Architecture (derived from the science)

| Module | Status | Why it exists / which RQ | If removed |
|---|---|---|---|
| Embedding cache (reusing the existing encoder, no new encoder) | **ESSENTIAL** | Feeds every experiment | Nothing can run within available compute |
| Held-out-*patient* evaluation harness, multiple seeds, confidence intervals | **ESSENTIAL**, cross-cutting | Required for *any* of RQ1–RQ3's findings to be credible (peer-review binding requirement) | Every result becomes untrustworthy, regardless of which RQ it supports |
| Persistence / Identity Dynamics / Linear SSM comparison harness | **ESSENTIAL** | RQ1 | RQ1 — the unlocking experiment — cannot run |
| Frame-shuffle sanity ablation | **ESSENTIAL**, cheap | Validity of RQ1 itself | Any RQ1 result is untrustworthy, not just weaker |
| Chronology/monotonicity auxiliary loss terms | **USEFUL** | RQ3 | Project still delivers a complete, valid finding on RQ1+RQ2 alone |
| Scrambled-constraint placebo control | **ESSENTIAL if RQ3 attempted at all** | Validity of RQ3 | RQ3's result becomes uninterpretable — same logic as the shuffle ablation for RQ1 |
| Simplified 2-regime switching model | **USEFUL** (becomes essential only once RQ1 passes) | RQ2 | Project can still deliver a valid RQ1(+RQ3) finding without it |
| Unsupervised slow-mode discovery (diffusion-map/TICA/Koopman) | **OPTIONAL**, deferred | RQ4 | The most scientifically novel question stays untested — acceptable for the near-term budget, not acceptable long-term |
| Full Constraint Engine hierarchy | **NOT NOW** | Only justified once one constraint category (RQ3) proves independently useful | Nothing lost near-term; explicitly documented so it isn't rebuilt from scratch later |
| Trajectory Analyzer, Knowledge Grounding / RAG | **NOT NOW** | Depend on a validated trajectory existing, which it doesn't yet | Nothing lost; premature until RQ1–RQ2 land |
| Full HSMM with duration modeling, full Switching Dynamical System, Neural ODE, Koopman modeling proper | **NOT NOW** | Each is a justified *upgrade path* only if its cheaper approximation (E2, or E1's linear model) succeeds and shows a specific limitation that motivates it | Building any of these now would be premature generalization — a model family with no evidence yet that it's needed |
| Dedicated uncertainty-calibration subsystem | **NOT NOW** | Use whatever posterior variance the Linear SSM/regime model already gives for free | A standalone calibration research thread is over-engineering at this stage |
| Curated anomaly-detection data pipeline | **BLOCKED, OPTIONAL** | Would strengthen RQ2's statistical power | Not buildable without a domain-expert collaborator; use synthetic anomalies as a documented fallback if pursued at all |

---

# Part VI — Research Risks

| Risk | Probability | Impact | Mitigation | Pivot |
|---|---|---|---|---|
| No shared dynamics across individuals (the irreducible question fails) | Low–Med | Catastrophic if true | Already partly tested as a byproduct of RQ1 — if there's no shared law, Identity/Linear SSM fails to generalize across held-out patients at all | Fall back to a purely descriptive/statistical study; abandon the dynamical-systems framing |
| Latent state not identifiable (even qualitatively, not just for the known-excluded variables) | Med–High | Medium — limits claim strength, doesn't kill the project | Cross-seed/cross-method reproducibility check built into evaluation from the start | Report at the level of qualitative structure (dimensionality, regime existence) rather than a canonical numeric coordinate |
| Progression not low-dimensional | Low–Med | Medium | RQ1 already tests 1D sufficiency implicitly against a higher-dim learned embedding | Expand state dimensionality — still compatible with the same experimental machinery |
| Encoder bias (embeddings already presuppose low dimensionality, since the encoder was trained for classification) | Medium | Med–High for RQ4 specifically, lower for RQ1–3 (bias affects compared models symmetrically there) | For RQ4, check against a non-classification-trained embedding if feasible | Scope claims explicitly to "given this encoder," not to raw images |
| Insufficient observability (weak signal for `q` specifically) | Med–High for `q` | Low–Med — affects a side-analysis, not the core RQ1/RQ2 findings | The persistence-vs-noise test already designed for exactly this | Drop `q` from the state entirely, simplifying to `(p,r,τ)` — a legitimate, even desirable, outcome |
| Dataset bias — single population, single imaging protocol | High (essentially certain given resources) | Medium — limits generalization claims, doesn't invalidate internal findings | Scope every claim explicitly to this population/protocol | None needed within this project; flag as required future work (external validation), not something this phase must solve |
| Anomalous/regime data scarcity | High (current pipeline actively filters these out; no curation resource confirmed) | Medium — limits RQ2's statistical power on the anomaly-concentrated evidence | Begin informal curation opportunistically, early | Fall back to synthetic anomalies |
| Confounded/weak outcome-adjacent labels (health, ploidy, live birth) if ever leaned on | High if used naively | High if it happened | **Already mitigated by design** — these are excluded from the core state (Part II); this risk only re-emerges if someone later reintroduces them carelessly | Watch for regression against this design decision, don't relitigate it |

---

# Part VII — Publication Strategy

**If the project fully succeeds** (RQ1, RQ2, and RQ4 all land — a 2-year-horizon scenario, not the near-term deliverable):

- *Title*: "A single hidden coordinate governs human embryo development and predicts arrest before it is clinically visible."
- *Abstract claim*: embryo development has been described using a fixed vocabulary of discrete stages and a separately-diagnosed arrest category; unsupervised analysis of time-lapse imaging, unconstrained by that vocabulary, shows both are readouts of one continuous process — the classical stages are ordered segments of one coordinate, and arrest is a local dynamical stall on that same coordinate, detected measurably earlier than current classification-based screening.
- *Main figure* (4 panels): (a) trajectories collapsing onto one shared curve in the discovered coordinate; (b) classical phase labels as ordered segments on it; (c) arrested/degenerating trajectories shown as velocity-collapse points on the *same* curve, not a separate axis; (d) a timing comparison — how much earlier the coordinate detects impending arrest than the current classifier.
- *Key table*: RQ1/RQ2/RQ4 results side by side — classifier vs. dynamics-aware models on temporal-coherence tasks; continuous-only vs. regime-aware likelihood gap on anomalous windows; biological vs. data-derived coordinate on Markov-gap/generalization diagnostics.
- *Experimental story*: existence of a shared dynamical law → sufficiency of a low-dimensional continuous state → unification of the discrete clinical taxonomy and arrest into that one coordinate → practical, measurable clinical benefit.
- *Reviewer's main criticism*: single-dataset, single-population, and insufficient engagement with adjacent literature (pseudotime/RNA-velocity trajectory inference, disease-progression state-space models, existing embryo-timelapse ML work) — see `HANDOFF.md` §12 for the full binding list. *Answer*: explicit population/protocol scoping in every claim; a dedicated related-work section positioning against each of those literatures; the smoothed-classifier baseline included from the start, not added under review pressure; empirical (not just theoretical) identifiability checks reported in the main text, not relegated to supplement.

**If the project only partially succeeds** (RQ1 and RQ3 land cleanly, RQ2/RQ4 don't): the best publishable paper is a rigorous, well-controlled benchmark and validation study — "does explicit dynamics modeling help for embryo time-lapse imaging: a systematic evaluation" — reporting the classifier-vs-dynamics and constraint-vs-placebo comparisons honestly, whichever direction they point. This is a solid, domain-venue-appropriate contribution (medical-imaging tier, not flagship), not a discovery paper, and should be described to collaborators and committees as such from the outset — not oversold.

---

# Part VIII — Implementation Plan

**Milestones** (repo must stay working after every one):

- **M0 (wk 1–2)**: reproduce the existing classifier's baseline exactly; cache embeddings once; begin (slow, opportunistic) curation of anomalous trajectories for later possible use.
- **M1 (wk 3–5)**: E1 (RQ1) — the go/no-go gate.
- **M2 (wk 6–9)**: E3 (RQ3), on top of M1's model.
- **M3 (wk 10–14)**: E2 (RQ2), time-boxed.
- **M4 (wk 15–16)**: cheap opportunistic side-analyses — `q` persistence check, uncertainty-calibration byproduct check.
- **M5 (wk 17–18)**: consolidation, one honest write-up.

**Implementation order**: embedding cache → evaluation harness (with held-out-patient splits verified) → E1's comparison models → E1's frame-shuffle ablation → E3's constraint terms and placebo control → E2's regime model.

**Validation order**: baseline reproduction matches documented numbers → sanity ablations pass → main experiments run with multiple seeds → cross-seed/cross-method reproducibility checks on any claimed identifiable structure.

**Ablation order** (cheapest and most fundamental first): frame-shuffle order (validates RQ1's task design itself) → scrambled-constraint placebo (validates RQ3) → capacity-matching checks between continuous-only and regime-aware models (validates RQ2).

**Baselines** (see `HANDOFF.md` for the full table): random/majority, population base-rate, domain heuristic, the existing classifier, **classifier + post-hoc temporal smoothing** (the cheapest real competitor — must be beaten, not skipped), Identity Dynamics, Linear SSM, plain HMM (isolates discreteness from duration-awareness), simplified 2-regime switching.

**Sanity checks**: frame-shuffle ablation; held-out-*patient* (never held-out-window-only) split verification, checked explicitly, not assumed; baseline reproduction matching documented numbers before any new model is trusted.

**Computational requirements**: qualitatively low relative to the original classifier's training run. No new large-network training is required for E1–E3 — they operate on cached embeddings from the existing, already-trained encoder. The one meaningfully expensive step is the one-time embedding extraction pass, which should be cached once and reused across every experiment. E4 (deferred) would add the cost of an unsupervised discovery method's hyperparameter/robustness sweep, non-trivial but still far cheaper than training a new encoder.

---

*Part IX (the daily-reference summary) lives in `BLUEPRINT_SUMMARY.md` — short by design, meant to be kept open, not read start to finish.*
