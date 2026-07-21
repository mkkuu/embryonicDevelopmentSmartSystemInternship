# evaluation.models

Real, registered `Model` implementations for the shared evaluation
framework (`evaluation.model.Model` — `fit`/`predict`/`save`/`load`).
Not to be confused with the retired `examples/` package, which existed
only to demonstrate the interface.

Four models, forming an ablation ladder — each adds exactly one piece of
information (or one piece of learning) the previous one lacked:

| Model | Uses |
|---|---|
| `base_rate` | nothing at all |
| `identity_dynamics` | window $i$'s own embedding content |
| `persistence` | the *change* from window $i-1$ to window $i$, assuming no dynamics ($A=I$, fixed) |
| `linear_ssm` | the *learned* transition $A$ between window $i-1$ and window $i$ |

`identity_dynamics` beating `base_rate` is evidence the embedding's
**content** is informative. `persistence` beating `identity_dynamics` is
evidence the embedding's **change over time** carries information content
alone does not. `linear_ssm` beating `persistence` is evidence that
**learning** the transition, rather than fixing it to the identity, adds
value beyond the simplest possible use of temporal structure. Together
these are the direct empirical test of `RESEARCH_BLUEPRINT.md` Part I,
H1. Anything built later (a regime model, Neural ODE, ...) that fails to
beat `linear_ssm` has not demonstrated it is doing more than a learned
linear dynamics operator already does.

## `base_rate` — `BaseRateModel`

Predicts the fixed empirical transition rate from training data, for
every window, ignoring the embedding entirely. The floor every other
model in this comparison must beat to justify using the embedding at
all — the "population base-rate baseline" in `RESEARCH_BLUEPRINT.md`'s
baseline table.

## `identity_dynamics` — `IdentityDynamicsModel`

Scientific assumption: `x(t+Δ) = x(t)`, read as a statement about the
dynamics-processing step itself — the trajectory equals the raw,
unprocessed observation sequence, unchanged. Predicts from window $i$'s
own embedding alone, via logistic regression, with zero reference to any
other window. Formalizes the true `dx/dt = 0` / no-processing floor from
`HANDOFF.md` Section 4.4. See `identity_dynamics.py`'s module docstring
for the full formulation.

## `persistence` — `PersistenceModel`

Scientific assumption: the future embedding equals the current embedding,
applied as a one-step-back forecast — $\hat e_i = e_{i-1}$ — with the
prediction driven by the residual $d_i = \lVert e_i - e_{i-1}\rVert_2$.
See `persistence.py`'s module docstring, including an explicit note on
how this differs from `identity_dynamics`.

## `linear_ssm` — `LinearStateSpaceModel`

Model: $z_{i+1} = Az_i + b + w_i$, $e_i = Cz_i + \mu + v_i$. Learns the
transition matrix $A$ (ridge-regularized least squares on
within-trajectory consecutive pairs) and, by default, an observation
matrix $C$ via PCA — reducing to a `latent_dim`-dimensional state rather
than operating on the full embedding, since a full $D \times D$ $A$ is
wildly overparameterized for this project's realistic dataset size (see
`linear_ssm.py`'s module docstring, design decision 1). $C$ is optional —
`latent_dim=None` operates directly in embedding space.

Two capabilities beyond the required interface:
- `forecast(trajectory, steps)` — future prediction, rolling $A$ forward.
- `impute_missing_window(trajectory, missing_index)` — missing-frame
  prediction, via forward/backward extrapolation (backward only when $A$
  is safely invertible).

Evaluated separately from the classification comparison — see
`evaluate_linear_ssm_dynamics.py` — since forecast/imputation quality is
a regression (embedding reconstruction error) problem, not a
classification one.

## Running the classification comparison

```bash
cd Training
python -m evaluation.run_e1_baselines \
    --cache_root ../Embeddings \
    --embedding_model_name resnet18 \
    --seeds 0 1 2
```

Writes `Results/evaluation/e1_baseline_comparison/` — per-seed results,
aggregated confidence intervals, comparison plots (including
`comparison_<metric>.png`, which puts all four models side by side), and
`REPORT.md`.

## Running the Linear SSM's dynamics evaluation

```bash
cd Training
python -m evaluation.evaluate_linear_ssm_dynamics \
    --cache_root ../Embeddings --embedding_model_name resnet18 \
    --latent_dim 16 --max_steps 5
```

Reports $k$-step forecast MSE and single-window imputation MSE, each
against a naive (unchanged-embedding) baseline for scale.

## Tests

`Tests/evaluation/models/` — synthetic `Trajectory` objects only, no real
cache or GPU required. Run from the repo root:

```bash
pytest Tests/
```
