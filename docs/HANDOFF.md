# HANDOFF.md — SciML Extension for Embryo Developmental Dynamics

**Purpose of this document**: complete context transfer for a fresh Claude Code session with zero memory of the design conversation that produced it. Read this top to bottom before touching anything. It captures a long (13-stage) design conversation that was **entirely conceptual** — see §3 for what that means in practice.

---

## 1. Project Identity & Status

This repository (`embryonicDevelopmentSciMLExtension`) currently contains a **working, unmodified** pipeline:
- `Training/` — PyTorch training pipeline for ResNet18/TimeSformer models detecting developmental-stage transitions in human embryo time-lapse image sequences (Gomez et al. Zenodo dataset).
- `WebApplication/` — a Flask app for clinicians to manage embryo records and run predictions using the trained models.

See `CLAUDE.md` (already exists in this repo) for the architecture of that existing pipeline — file-level orientation, common commands, known quirks. This handoff document is about the **SciML extension** layered on top of it conceptually — a long-term research programme to reframe embryo development as a hidden dynamical system rather than a frame classifier.

**Critical fact, stated once and clearly: nothing described below has been implemented.** No new code, no new files, no experiments, no results exist yet. Everything in this document is the output of a pure design/theory conversation. Do not assume any model, dataset split, or evaluation harness described here exists in the repository — it does not, until §16's next task is executed.

---

## 2. Current Objective

The **operative** objective (superseding the earlier, more ambitious versions described in §10) is the realistic, resource-constrained plan from the "lead researcher" pass (§11): a single researcher, limited compute, an existing working classifier that must keep working throughout, publication desirable but not mandatory. The immediate next action is §16.

The **long-term aspirational objective**, if the realistic plan succeeds, is described in §7 — a specific, falsifiable scientific narrative this whole programme is aimed at, not yet supported by any evidence.

---

## 3. Repository State — What's Real vs. What's Conceptual

| Item | Status |
|---|---|
| `Training/`, `WebApplication/` pipeline | Real, working, existing code. Described in `CLAUDE.md`. |
| Any `sciml/` package, `LatentEncoder`, `DynamicsModel`, `Trajectory` class, etc. | **Does not exist.** Purely conceptual, discussed across §7 of the design conversation and superseded by the leaner plan in §11 below. |
| The 9-hypothesis research programme (H1–H9), 5-stage 12-month roadmap | Conceptual, superseded — kept in §10 as background/context, not the active plan. |
| The 3-idea realistic plan and M0–M5 milestones | **This is the active plan.** See §11. Not yet started. |
| Any experiment, model, or reported number | **Does not exist.** Zero results have been produced. |
| The mathematical formulation in §4 | Conceptual — a precise theoretical specification, not yet fit to data. |

---

## 4. Mathematical Formulation (crystallized theory)

This is the most precise, load-bearing artifact of the whole conversation — the pure applied-math specification of what the whole project is implicitly trying to discover, deliberately independent of any ML framework.

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| `t ∈ [0,T]` | Continuous developmental time |
| `t_k` | Discrete, possibly irregular sampling (imaging) times |
| `x(t) ∈ 𝒳 ⊆ ℝᵈ` (or a manifold `ℳ`) | Hidden biological state |
| `θ ∈ Θ` | Static, individual-specific parameters (`dθ/dt ≈ 0` over `[0,T]`) |
| `r(t) ∈ ℛ = {1,…,K}` | Discrete regime (càdlàg, piecewise-constant) |
| `τ(t)` | Sojourn time since last regime jump — **derived**, not primitive: `τ(t) = t − sup{s≤t : r(s⁻)≠r(s)}` |
| `f_r(x,t;θ,u)` | Regime-dependent drift (vector field) |
| `σ(x,t;θ)` | Diffusion coefficient |
| `W(t)` | Standard Wiener process |
| `N_i(t)`, `λ_i(x,t;θ)` | Jump/counting processes and their intensities (division, compaction, blastulation events) |
| `Q(x,τ;θ)` | Regime transition-rate generator (state- and sojourn-time-dependent) |
| `u(t)` | External/exogenous perturbation (culture conditions, etc.) |
| `h(x,r,t)` | Observation / image-formation map, `𝒳→𝒴` |
| `ε_k ~ 𝒩(0,Σ_obs(x(t_k)))` | Observation (sensor) noise, heteroscedastic |
| `g(x)` | Readout map (task-specific: phase, grade — NOT part of the physical generative mechanism) |
| `ℓ_k`, `η_k` | Observed labels and annotation noise: `ℓ_k = g(x(t_k)) + η_k` |
| `X_{[0,T]}` | Full trajectory, `(x(·),r(·)) ∈ C([0,T],𝒳)×D([0,T],ℛ)` — a point in path space |
| `ψ(x)` | A scalar functional of state used to state constraints (e.g. a progression-like coordinate) |
| `𝒳_adm` | Admissible domain (hard constraint set) |
| `𝒟 = {(t_k,y_k,ℓ_k)}` | Everything actually observed |

### 4.2 Mathematical objects — role / observable / identifiable

See the full table from the design conversation; key points:
- `x(t)`: not observable; identifiable only up to a diffeomorphism unless externally anchored.
- `θ`: not observable or identifiable from images alone (needs an auxiliary channel — e.g. genetic testing for ploidy).
- `r(t)`: indirectly observable through qualitative shifts in `h`; identifiable only given a sufficient observation horizon.
- `h`: generically non-injective — components of `x` in `ker(dh)` are **structurally** unlearnable, regardless of data volume (this is not a data-scarcity problem).
- `σ` is generally easier to identify than `f` from finite-time data (classical SDE-estimation fact: volatility is locally recoverable from quadratic variation, drift is not).
- Uncertainty is not a primitive object — it is the derived posterior `p(x,r,θ | y_{1:N})`, i.e. a property of the observer/estimator, not of the biological system.

### 4.3 Generative model

```
θ ~ p(θ)
x(0) ~ π₀(·;θ),   r(0) = r₀

dx(t) = f_{r(t)}(x(t);θ,u(t)) dt + σ(x(t);θ) dW(t) + Σᵢ Δᵢ dNᵢ(t)

ℙ(r(t+dt)=r′ | r(t)=r, x(t), τ(t)) = Q_{r,r′}(x(t),τ(t);θ) dt + o(dt),  r′≠r
τ(t) = t − sup{s≤t : r(s⁻)≠r(s)}

y_k = h(x(t_k), r(t_k), t_k) + ε_k,   ε_k ~ 𝒩(0, Σ_obs(x(t_k)))
ℓ_k = g(x(t_k)) + η_k

𝒟 = {(t_k, y_k, ℓ_k)}_{k=1}^N     ← the only thing ever actually observed
```

### 4.4 State-space simplification chain (each step's assumption)

1. **General stochastic hybrid system** (above).
2. Remove regime-switching (`K=1`) → plain nonlinear SDE. *Assumes* one smooth vector field suffices everywhere (does NOT rule out multistability of that single field).
3. Remove `u(t)` and time-inhomogeneity → autonomous SDE. *Assumes* exogenous drift is negligible or absorbed into `θ` — risk: real culture-condition drift gets misattributed to intrinsic biology.
4. Remove stochasticity (`σ→0`) → deterministic ODE, **the Neural-ODE interpretation**. *Assumes* all cross-individual variability comes from `θ`/`x(0)`, not path-wise noise.
5. Linearize `f` → linear state-space model. *Assumes* the operating range has no strong nonlinear/multistable structure — i.e. assumes away exactly what would justify a regime variable.
6. **Separate branch** (not on this chain): discard continuous `x` entirely, model only `r(t)` as a semi-Markov process with duration-dependent hazards — the **HSMM**.

*Where one stops in this chain is the actual scientific hypothesis under test — not an implementation detail.*

### 4.5 Continuous vs. discrete time — resolved by elimination

A **hybrid jump-diffusion, Markov-modulated system** is the minimal adequate class: pure-continuous fails to represent genuine discontinuities (division count) without distortion; pure-discrete-time conflates the imaging *sampling schedule* with the biological process's own timescale (real inter-frame times are irregular — see §14, `embryo_dataset_time_elapsed`); pure-event-driven fails to capture genuine continuous within-regime variation (blastocoel expansion).

### 4.6 Constraints — hard vs. soft, and where they live

| Constraint | Formalization | Belongs to |
|---|---|---|
| Admissible values | `x(t) ∈ 𝒳_adm` | **Hard** — state-space definition itself |
| Monotonicity | `⟨∇ψ(x), f_r(x)⟩ ≥ 0` | Dynamics, but **soft in practice** (see below) |
| Irreversibility | `Q_{r→r′}=0` for `r′` earlier in a partial order | **Hard**, sparsity on `Q` |
| Bounded velocity | `‖f_r‖ ≤ V_max` | Dynamics, soft in practice |
| Minimum duration | `Q_{r→r′}(τ)=0` for `τ<τ_min(r)` | Dynamics, soft in practice |
| Phase ordering | `g(x)=ℓ_i` iff `ψ(x)∈[c_{i-1},c_i)` | **Readout map** (`g`), not the dynamics directly |
| Trajectory continuity | `x(·)∈C([0,T],𝒳)` a.s. | Built into the choice of stochastic-process class |

Rule: constraints that are logically certain (domain membership, irreversibility sparsity) are hard. Constraints that are well-motivated *beliefs* the biology could occasionally violate (monotonicity, min-duration, bounded velocity — real fragmentation/arrest can break these) belong as **soft penalties on the estimation objective**, not hard restrictions on the searched model class — otherwise the model becomes structurally incapable of explaining genuine exceptions.

### 4.7 Unknowns and learnability

`f_r, σ, Q, h, g, Σ_obs, π₀(x(0);θ), p(θ)` are the unknown functions/distributions. Individual `x(t), r(t), θᵢ` are latent variables with a well-defined posterior but inherit every identifiability limit above. `K` (regime count) and the model class itself (linear/nonlinear, Markov/semi-Markov) are a **different kind** of unknown — subject to model comparison, never pointwise-identified.

---

## 5. The Minimal Biological Latent State (design result)

After a full system-identification pass (necessity, observability, identifiability, Markovianity analysis), the minimal state settled on was:

```
S(t) = ( p(t), r(t), q(t), τ(t) )
```
- `p` — progression, continuous, ordered, the one irreplaceable coordinate.
- `r` — regime, discrete, small cardinality (normal / arrested / degenerating), justified by a Markov-sufficiency argument (two trajectories can coincide in `p,q` and still diverge in future behavior).
- `q` — quality/shape residual, continuous, low-dim, justified only if it is *persistent* within a trajectory (autocorrelated), not just noise — this is a stated, testable, unresolved condition, not a settled fact.
- `τ` — sojourn time, mathematically necessary only under Markov-first-order-compatible families (SSM, Neural ODE, Koopman); **not needed at all under HSMM**, where it's built into the family. Its necessity is *diagnostic* of an incomplete coordinate system (Mori–Zwanzig framing), not a biological fact in its own right.

Explicitly **excluded** from the core dynamical state (demoted to static parameters or downstream outputs): ploidy (θ, structurally unidentifiable from images — morphology is a documented weak proxy), health/viability (an output, confounded by unobserved maternal/clinical factors, not a state coordinate), fine ICM/TE lineage structure (excluded given the current 2D imaging modality — a sensor limitation, not necessarily a biological one).

Was reused from real dataset structure: `chronological_phases` ontology already present in `Training/DataSet.py` (see §14); `blastocyst_grade`/`pgt_a_grade`/`live_birth` in the WebApp DB schema as *candidate future* weak-supervision signals (population linkage to the training cohort is unverified — flag this before relying on it).

---

## 6. The Coordinate-System Question (the single most important open thread)

The minimal state in §5 was then itself challenged: are `(p,r,q,τ)` the *mathematically natural* coordinates, or a human-imposed projection of something deeper? Key unresolved hypothesis, current best guess:

- `p` (progression) is plausibly a legitimate **order parameter** — a macroscopic slow mode of a much higher-dimensional true generative process, in the same sense temperature is a legitimate macroscopic variable despite not being a microscopic degree of freedom. Development being directional/irreversible (not oscillatory or genuinely multi-attractor in a complex way) gives a priori reason to expect a dominant ~1D slow mode.
- `r` (regime) is plausibly **not an independent coordinate at all**, but a human categorization of continuous multistable dynamics — what looks like discrete "arrest" may just be the same single slow coordinate's local velocity collapsing near zero (a region of its own vector field), not a second hidden variable.
- `q` (quality) is the weakest-grounded of the four — defined by residual/exclusion, not by mechanism. Could be a real second slow direction, could merge into `p`, could split into multiple weak directions. Genuinely open.
- `τ` should disappear entirely if the truly fundamental coordinates are found — its presence in the current design is evidence the design is not yet fundamental, not evidence sojourn-time itself is biologically meaningful.

This is the single question with the highest scientific-novelty ceiling in the whole programme (see §7, §11) and is currently the **least resourced** — deliberately deprioritized for practical reasons (§11), not because it's unimportant.

---

## 7. The Target Scientific Narrative (aspirational — NOT a claimed result)

If the realistic plan (§11) and, further, the coordinate-system question (§6) both succeed in the most scientifically interesting way, the resulting paper's shape was reverse-engineered as follows. **This is a hypothesized best-case outcome, explicitly not a finding.**

- **Central question**: does human embryo development, as captured by time-lapse imaging, admit a single, reproducible, shared dynamical description — and are the discrete clinical categories (phase, arrest) readouts of that one process rather than separate phenomena?
- **Hypothesized discovery**: a single data-derived coordinate, found without using phase labels, recovers the entire classical phase taxonomy as ordered intervals along it — and clinically-flagged "arrest" is not a separate category but a local velocity-collapse on the *same* coordinate, detectable earlier than current classification-based screening.
- **The one figure** (4 panels): (a) many trajectories collapsing onto one shared curve in the discovered coordinate; (b) classical phase labels overlaid as ordered segments on that curve; (c) arrested/degenerating trajectories shown as velocity-collapse points on the *same* curve, not a separate axis; (d) a timing comparison — how much earlier the coordinate detects impending arrest than the current classifier.
- **One-sentence reduction** (no "framework/architecture/pipeline/system"): *"Embryo development runs on a single hidden clock, and developmental arrest is that clock stalling — not a separate biological event."*
- **Fully honest, defensible claim** (reduced from the overreaching version — see conversation for the full reduction trace): *"In this dataset, the image-observable dynamics of human preimplantation development are well-described by a single, reproducible, data-derived continuous coordinate that recovers the ordering of the standard morphokinetic phase taxonomy without being given it, and along which clinically-flagged developmental arrest corresponds to a characteristic, measurable drop in local dynamical velocity — enabling earlier detection of arrest than the current classification-based approach, within the population and imaging protocol studied."*

Do not let this narrative bias data analysis. It is a target to test, not a conclusion to confirm.

---

## 8. Assumptions (explicitly flagged across the conversation, not yet verified)

- The 15 discrete phase labels are a discretization of a single underlying progression clock, carrying no information beyond ordering.
- Regime dynamics are adequately captured by a small (~3) discrete set with genuinely different forward dynamics per regime.
- The quality residual is effectively low-dimensional.
- Progression's ordering is recoverable from images; its absolute scale requires external anchoring (phase labels / real elapsed time).
- Ploidy and health have genuinely negligible causal influence on the image-generating process (inherited from the outside literature, not re-derived from this dataset).
- The existing phase labels are themselves reliable ground truth (not verified — inter-observer variability in embryology grading is real and documented).
- Classification-trained embeddings retain the fine continuous structure a richer dynamical analysis needs (an information-bottleneck risk, never tested).
- The WebApp DB population (with `blastocyst_grade`/`pgt_a_grade`/`live_birth`) and the Training CSV population (Zenodo Gomez et al. dataset) may be **different cohorts** — linkage unverified.

---

## 9. Open Research Questions

1. Is progression genuinely 1D, or does it need ≥2 axes (e.g. structural vs. biochemical maturation that can desynchronize)?
2. Does the quality residual decompose into a static baseline + dynamic residual, or is it closer to pure noise (test: within-trajectory vs. across-trajectory variance)?
3. Is 3 the right regime cardinality, or does data support finer distinctions (test: does a 4th regime reduce prediction error or just overfit)?
4. Does previous-regime identity (not just current regime + `τ`) carry additional predictive value (resumption-from-arrest vs. first-arrival)?
5. Would richer imaging (multi-focal-plane/3D) make ICM/TE structure identifiable, and would it then belong in the core state or remain a separate output layer?
6. Is progression's scale actually well-anchored by existing phase labels, or does non-uniform label spacing leave residual non-identifiability (test: do independently trained progression estimates agree only up to a monotone, not affine, transform across runs)?
7. (From the coordinate-system challenge) Does a data-derived slow coordinate (diffusion-map/TICA/Koopman-style) actually outperform the hand-built `p` on Markov-gap and smoothness diagnostics, or are they statistically indistinguishable (a genuinely positive result for the interpretable-first design either way)?

---

## 10. Completed Work (theoretical only — see §3)

- Full audit of the existing repository and its architecture (captured in `CLAUDE.md`).
- A long-term conceptual SciML architecture (Visual Encoder → Visual Embedding → Observation Model → Biological State → Dynamics Model → Trajectory → Readout/Constraints/Analyzer → Knowledge Grounding/RAG) — **superseded as the near-term plan by §11**, retained only as long-term context.
- A file-level migration plan mapping the target architecture onto the actual existing code (composition over modification; the one identified minimal-diff seam is `Training/train_val_test_pipline.py`'s loss line, see §14).
- The biological state design (§5), its system-identification critique, and the coordinate-system challenge (§6).
- A full 9-hypothesis research programme (H1–H9) with experiment designs, prioritization, staging, baselines, ablations, evaluation criteria, publication strategy, and risk analysis — **superseded as the active plan by §11**, retained as background/reference (useful if scope later expands).
- A ruthless, resource-constrained re-scoping into 3 chosen ideas and an 18-week adaptive milestone plan (§11) — **this is the active plan**.
- An adversarial peer-review self-critique (§12) — treat its demands as binding requirements for any future paper, not just historical commentary.
- A reverse-engineered "what would the finished paper actually say" narrative (§7).
- A first-principles interrogation of that narrative's central question, including necessary conditions, alternative explanations with falsification experiments, and counterfactual geometries (kept in the source conversation; summarized in §6–§7).
- The pure mathematical formulation (§4) — the most precise artifact produced, deliberately independent of any ML framework.

**Zero code has been written. Zero experiments have been run. Zero numbers exist.**

---

## 11. The Active Plan — 3 Chosen Ideas + Adaptive Roadmap

Given one researcher, limited compute, and an existing pipeline that must keep working:

**The three ideas** (in priority order):
1. **Dynamics vs. classifier** — a minimal Identity/Persistence/Linear-SSM comparison harness against the existing classifier, on temporal-coherence tasks a per-frame classifier structurally can't do well (early-transition prediction, missing-frame imputation). This is the existential gate everything else depends on.
2. **Simplified 2-regime switching augmentation** — a deliberately de-scoped version of full HSMM (no explicit duration modeling yet), testing whether discreteness adds value beyond continuous state.
3. **Biological constraints as an auxiliary loss** — reusing the phase-ontology logic already present in `Training/DataSet.py` (see §14), cheap to implement, tests whether prior knowledge helps.

Explicitly rejected as unrealistic *for now* (not forever): full HSMM with duration modeling, the coordinate-system comparison (§6) as a full rigorous study, curated-anomaly-data-dependent experiments (no confirmed embryologist collaborator), the 3-paper publication strategy (scaled down to "one honest write-up, expanded only if warranted").

**Milestones** (~18 weeks total, each ending in an explicit decision, repo must stay working throughout):

- **M0 (wk 1–2)**: Reproduce the existing classifier's baseline numbers exactly. Cache visual embeddings once for reuse. Begin (slow, don't wait) curation of anomalous/arrested trajectory examples for possible later use. *Decision*: if iteration is too slow given available compute, cut dataset size/resolution now, not later.
- **M1 (wk 3–5)**: Idea 1. Include a shuffle-frame-order sanity ablation from the start. *This is the real go/no-go gate.* Clear negative (even after the sanity check) → jump to the pivot directions below. Positive/promising → continue.
- **M2 (wk 6–9)**: Idea 3, on top of M1's model. Include a placebo/scrambled-constraint control to distinguish "this biological structure helps" from "any extra regularization helps."
- **M3 (wk 10–14)**: Idea 2, time-boxed explicitly. Watch for discrete-latent training instability (a known real risk) and report honestly if it occurs rather than iterating indefinitely.
- **M4 (wk 15–16)**: Cheap opportunistic side-analyses — quality persistence check (open question #2 above), uncertainty calibration check — both free byproducts of M1–M3's models.
- **M5 (wk 17–18)**: Consolidation and one honest write-up, whatever was found.

**Fallback directions if M1 shows no benefit** (design at least one before declaring failure):
1. Narrow the claim to robustness/imputation rather than accuracy.
2. Pivot to interpretability of the *existing* classifier (probe its embeddings for a progression-like signal, no predictive-improvement claim).
3. Pivot to a data-centric error analysis of the existing classifier (does it fail specifically post-compaction, as observability theory would predict — a testable, concrete hypothesis).
4. Substitute synthetic anomalies for the (likely unavailable) curated real ones, to test anomaly-sensitivity cheaply.

**What NOT to build until evidence justifies it** (with triggers): full Constraint Engine hierarchy (only after one category proves useful), Trajectory Analyzer (only once a validated trajectory exists), Knowledge Grounding/RAG (not this project), full HSMM/Switching Systems (only if the simplified version in Idea 2 succeeds), Koopman/diffusion-map analysis (deferred, see §6), Neural ODE (only if linear SSM's failure is specifically attributable to nonlinearity, not just "more expressive is probably better"), any dedicated uncertainty-calibration subsystem (use only what a probabilistic model gives for free).

---

## 12. Known Weaknesses / Binding Requirements for Future Work

From an adversarial peer-review pass — treat these as requirements, not just critique:

- **Missing baseline that must be included**: classifier + post-hoc temporal smoothing. This is the cheapest plausible competitor to any dynamics claim and was not originally in the plan.
- **Missing citations to engage with before claiming novelty**: pseudotime/trajectory-inference literature (Monocle, Slingshot, RNA velocity/scVelo, Waddington-OT), disease-progression state-space models in clinical ML, existing embryo time-lapse ML/outcome-prediction literature (this is not a green field), PINN literature (for constraint framing), Koopman/DMD/EDMD and reaction-coordinate discovery (TICA, VAMPnets) from molecular dynamics.
- **Necessary before any identifiability claim**: an empirical cross-seed/cross-method consistency check — the theoretical identifiability analysis (§4.2, §6) has never been tested against real data.
- **Necessary for any clinical claim**: held-out-*patient* splits (never held-out-window only — leakage risk is real in this per-video dataset), and explicit discussion of the WebApp/Training population-linkage uncertainty (§8).
- **Overclaims to avoid**: calling anything "biologically-informed" before empirical validation; presenting constrained/monotonic losses as novel (they're standard); continuing to gesture at "health"/"viability" modeling when §5 already concluded these are likely unidentifiable from images; using "Scientific Machine Learning framework" language before a genuine methodological or empirical finding exists.

---

## 13. Important Design Decisions

- **Composition over modification, everywhere.** `Training/ModelBuilder.py`, `Training/DataSet.py`'s core logic, `Training/Load_data.py`, and all of `WebApplication/` remain untouched by design. The one legitimate exception, if/when reached, is a small, `None`-default optional-argument extension to `Training/train_val_test_pipline.py`'s training loop (see §14) — never a rewrite.
- **State vs. parameter, strictly separated.** Ploidy and health/viability are excluded from the dynamical state entirely (θ or output, never a state coordinate) — this was a deliberate, argued decision (§5), not an oversight.
- **`τ` is diagnostic, not biological.** Treat any future need for an explicit sojourn-time coordinate as evidence the current coordinate system is incomplete, not as a finding about biology.
- **Constraints are soft by default.** Only domain-membership and irreversibility-sparsity are hard; everything else (monotonicity, duration, velocity bounds) is a penalty on estimation, not a restriction on the model class (§4.6) — because real biology can violate them in informative ways.
- **The coordinate-system question (§6) is deliberately deprioritized for practical reasons, not because it's unimportant** — it is the single highest-novelty-ceiling item in the whole programme and should be revisited first if the realistic plan (§11) succeeds and there is appetite/resources for a follow-up.
- **Publication ambition is scaled down.** One honest write-up, expanded to a paper only if results earn it — not the earlier 3-paper strategy.

---

## 14. Files Concerned

| File | Relevance |
|---|---|
| `Training/config_args.py` | `ConfigArgs` — hierarchical config (CLI > `config.ini` > hardcoded defaults). Instantiated at import time in several modules — a pattern to *not* propagate into new code. |
| `Training/DataSet.py` | `Embryo_Transition_Dataset`. `chronological_phases` list (~lines 151–167) is the phase ontology — reuse target for Idea 3's constraint. `_create_sequences` (~188–221) does window+adjacency filtering — literally an embedded, unreusable Chronology/Transition constraint; extracting it (behavior-preserving) is the natural first touch to this file. `__getitem__` (~266–317) returns `(images_seq, consistency_flag, first_frame_phase, last_frame_phase)` — the existing supervision signal to reuse for Idea 1/3, no new labeling needed. |
| `Training/Load_data.py` | `get_dataloaders()` factory — reuse as-is. |
| `Training/ModelBuilder.py` | `get_model()` (~99–143). ResNet18 branch returns a raw `timm` model exposing public `forward_features`/`forward_head` — reachable by composition, zero edits needed. `TimeSformerWrapper` (~43–96) wraps the HF model as `self.model`; `forward()` returns only `.logits`, discarding hidden states — but `wrapper.model` is a public attribute reachable directly, so hidden-state access is also possible without editing this file. |
| `Training/train_val_test_pipline.py` | `train_model()` (~80–183) — the loss line (~129, `loss = criterion(outputs, targets)`) is the single identified minimal-diff seam for any future constraint/telemetry hook, via an optional `None`-default argument. `evaluate()` (~186–339) already builds phase-transition confusion matrices (~238–334) — a proto-analysis worth reusing as design inspiration, not as code. Hardcoded `device="cuda"`, no CPU fallback. |
| `Training/train.py` | Orchestration script — must keep behaving identically, unmodified invocation, throughout every milestone. |
| `Training/preProcess.py` | One-shot data prep, orthogonal to the runtime pipeline. |
| `Configs/config.ini` | `[data]`/`[model]`/`[training]` sections — any future dynamics-selection config should follow this exact pattern additively (new section, never edit existing keys). |
| `WebApplication/Classes/Doctor.py` | `predictTransitions` (~523–675) has a **pre-existing bug**: `torch.load(model_path)` loads a full model, but training saves only `state_dict()` — inconsistent, likely broken except for the "no checkpoint found → random prediction" fallback path (`is_random` flag, ~604–614). Also duplicates `DataSet._create_sequences`' windowing logic independently. **Out of scope for the current plan**, but must be fixed before any future work reconnects trajectories/explanations to the web app. |
| `WebApplication/Dataset_shema.sql` | `embryo` table: `blastocyst_grade`, `pgt_a_grade` (Euploid/Aneuploid), `live_birth` — candidate future weak-supervision signals; population linkage to the training cohort is **unverified**. |
| `Data/embryo_dataset_time_elapsed/` | Real inter-frame elapsed-time annotations exist in the raw dataset but are **not currently joined** into the training CSVs (`preProcess.py`'s `process_annotations()` never references `time_csv_dir`) — recoverable asset, needed for any genuinely continuous-time modeling. |
| `Data/embryo_dataset_grades.csv` | Grading information referenced in `preProcess.py`; exact columns not yet inspected. |
| `CLAUDE.md` | Existing architecture orientation for the current pipeline — read this first for anything not covered here. |
| `HANDOFF.md` (this file) | This document. |

No new files exist yet anywhere in the repository as a result of this conversation.

---

## 15. Next Immediate Task

Execute **M0** (§11): reproduce the existing classifier's baseline numbers exactly, cache visual embeddings once, and confirm the iteration loop (embedding extraction → evaluation) is fast enough given available compute. Do not proceed to M1 until M0's numbers are trusted and verified against whatever is currently documented as the pipeline's reported performance. If a supervisor or new session is picking this up cold: start here, not with any of the more elaborate architecture discussed in §10.
