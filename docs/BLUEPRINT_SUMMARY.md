# Blueprint Summary — Keep This Open

Full detail: `RESEARCH_BLUEPRINT.md`. Session/technical continuity: `HANDOFF.md`. This file only.

## Scientific Question
Is human embryo development, as seen in time-lapse imaging, governed by a shared, low-dimensional dynamical law — and do the discrete clinical categories (phase, arrest) reflect that law's real structure, or something else? Irreducible version: does a *shared* law exist across individuals at all, with variation explained by different initial conditions/parameters of one process?

## Mathematical Model (one line each)
- Hidden state `x(t)`, static parameters `θ` (ploidy, competence — do **not** evolve), regime `r(t)` (discrete, small), sojourn time `τ(t)` (derived, not primitive).
- `dx = f_r(x;θ) dt + σ(x;θ) dW + jump terms`; regime jumps via generator `Q(x,τ;θ)`.
- Observation: `y_k = h(x(t_k),r(t_k)) + ε_k`; label: `ℓ_k = g(x(t_k)) + η_k`.
- `h` is generically non-injective — some components of `x` are **structurally** unobservable, not just data-limited.
- Hard constraints: state-space membership, irreversibility (transition-generator sparsity). Soft constraints (penalties, not hard rules): monotonicity, bounded velocity, minimum duration.

## Core Hypotheses
- **H1**: explicit dynamics beats frame-independent classification on temporal-coherence tasks. *(Gates everything — test first.)*
- **H2**: a discrete regime is necessary, beyond continuous state.
- **H3**: biological constraints beat a placebo/scrambled control.
- **H4** (deferred, 2-year horizon): the natural mathematical coordinates differ from the hand-designed biological ones — possibly "regime" is just a velocity-collapse of one continuous coordinate, not an independent axis.

## Experiments — in dependency order
```
M0 baseline repro + embedding cache
   → E1 (H1) ← THE unlocking experiment, must pass first
        → E3 (H3)   and   → E2 (H2)   [independent of each other, both need E1]
             → E4 (H4) [deferred, needs E1–E3 done first]
```
Every experiment needs: held-out-*patient* splits, multiple seeds, and its own sanity/placebo control (frame-shuffle for E1, scrambled-constraint for E3, matched-capacity for E2) — without the control, the result is not trustworthy, full stop.

## Architecture — build only what's justified now
**ESSENTIAL**: embedding cache, held-out-patient eval harness w/ CIs, Persistence/Identity/Linear-SSM harness, frame-shuffle sanity check.
**USEFUL**: constraint auxiliary loss + placebo control, simplified 2-regime switching model.
**NOT NOW**: full Constraint Engine, Trajectory Analyzer, RAG, full HSMM, Switching Systems, Neural ODE, Koopman analysis, dedicated uncertainty subsystem, curated-anomaly pipeline. Build these only when their cheaper approximation's result specifically demands it — not before.

## Milestones (18 weeks, repo stays working throughout)
M0 (wk1–2) repro+cache → M1 (wk3–5) E1, **go/no-go gate** → M2 (wk6–9) E3 → M3 (wk10–14) E2, time-boxed → M4 (wk15–16) cheap side-checks → M5 (wk17–18) write-up.

**If M1 fails clearly**: don't build M2/M3. Pivot to (a) robustness/imputation framing, (b) interpretability of the *existing* classifier, or (c) data-centric error analysis of where/why it fails.

## Top Risks
No shared dynamics (low prob., catastrophic) · state not identifiable (med-high prob., limits claim strength — mitigate with cross-seed checks) · dataset/population bias (high prob., certain — scope every claim to this cohort/protocol) · anomaly data scarcity (high prob. — curate opportunistically from week 1, or fall back to synthetic anomalies).

## Publication Goal
**Full success** (2-yr horizon): *"A single hidden coordinate governs human embryo development and predicts arrest before it is clinically visible."* One-sentence version: *embryo development runs on one hidden clock, and arrest is that clock stalling — not a separate event.*
**Partial success** (near-term, realistic floor): a rigorous benchmark/validation paper — does explicit dynamics modeling help here at all — reported honestly regardless of which way it points. Domain-venue tier, not flagship. Say so, don't oversell it.

## The One Rule
Every module, every experiment, every constraint exists to answer a specific research question above. If you can't say which one, don't build it yet.
