# Training/experiments — experiment and benchmark runners (not production code)

Nothing in this tree is imported by the application (`webapp_api`, `reporting`, `orchestrator`
runtime) — verified by the static import graph on 2026-09-15. Everything here reads the
embeddings and the frozen models and writes under `../Results/evaluation/` (server only).
Run from inside `Training/`, e.g. `python -m experiments.e1_ladder.run_e1_full_analysis`.
Each script keeps its original CLI and default paths (`../Embeddings`, `../Results/...`); only
its location and its import lines changed when it was moved here (2026-09-15 restructure).

The frozen benchmark tooling stays in `orchestrator/` on purpose (`run_fixed_question_benchmark.py`,
`run_validator_experiment.py`, `capture_run_identity.py`, `summarize_fixed_question_runs.py`):
their paths are part of the run identity of the pending Validator v1.3 rebenchmark.

| Directory | Experiment | Runners (from the repo) | Salvaged one-off scripts | Results directories (server) | Report |
|---|---|---|---|---|---|
| `e1_ladder/` | E1 ablation ladder (H1), Test split, pre-registered | `run_e1_full_analysis.py` (current entry point), `run_e1_baselines.py` (older seed-loop CI, superseded), `run_frame_shuffle_sanity_check.py` (validity control), `evaluate_linear_ssm_dynamics.py` (linear SSM forecast/imputation) | — | `e1_full_analysis/`, `e1_frame_shuffle/`, `linear_ssm_dynamics_report.json` | `docs/SCIENTIFIC_BACKGROUND.md` §3, `docs/SCIENTIFIC_REPORT.md` |
| `gru/` | GRU non-linear dynamics (Val, the one authorised Test evaluation, frame-shuffle control) | `run_gru_evaluation.py`, `run_gru_test_evaluation.py` (one-time Test exception, not to be re-run), `run_gru_frame_shuffle_sanity_check.py` | — | `e1_gru_step6a_full/`, `e1_gru_step6b_test_evaluation/`, `e1_gru_step6c_frame_shuffle/` | `docs/SCIENTIFIC_BACKGROUND.md` §3, `docs/GRU_IDENTITY_ANALYSIS.md` |
| `hmm_semi_hmm/` | HMM alpha/rho sweep (k = 7) and the k-step alignment diagnostic | `run_hmm_evaluation.py`, `run_hmm_k_step_alignment_check.py` | `archive/hmm_diagnostics/`, `archive/semi_hmm_construction/` (incl. `phase_f_full_train_val.py`, the fit of the served Semi-HMM), `archive/emission_reweighting/`, `archive/model_comparison/` | `e1_hmm_*`, `semi_hmm_*`, `emission_*`, `global_model_comparison/` | `docs/SCIENTIFIC_BACKGROUND.md` §4, `docs/HANDOFF_SEMI_HMM.md` |
| `embeddings_exploration/` | ad hoc PCA / t-SNE / UMAP look at the embedding cache | `explore.py` | — | none kept | — |
| `rag_retrieval_smoke/` | hand-labelled retrieval smoke benchmark (precision/recall@k) | `eval_benchmark.py` | — | `Training/rag_benchmark_report.json` (server, gitignored) | `docs/archive/reports/RAG_PHASE_3_REPORT.md` |
| `llm_regression/` | the three LLM-layer regressions: routing with a null LLM, router + template (18 questions), 28-question answer quality; plus the temperature A/B | `run_orchestration_evaluation.py`, `run_llm_benchmark.py`, `run_rag_llm_quality_eval.py`, `run_temperature_ab_check.py`; data modules `eval_questions.py` (18 router-validation questions, also used by `Tests/orchestrator/test_router_*`), `rag_llm_quality_questions.py` | `archive/llm_benchmark_probes/` | `orchestration_foundation_eval/`, `llm_benchmark_*/`, `rag_llm_quality/` | `docs/BENCHMARK.md` §4, `docs/reference/RAG_LLM_QUALITY_REPORT.md`, `docs/archive/reports/LLM_EVALUATION.md` |
| `context_ablations/` | the paired context experiments on Q1–Q15: FULL vs OBSERVED_ONLY, PREDICTION_ONLY, context v2 vs compact_v1 | `run_observed_only_experiment.py`, `run_prediction_only_experiment.py`, `run_compact_context_experiment.py`; filters `observed_only.py`, `prediction_only.py` (pinned by `Tests/orchestrator/test_observed_only.py`); companions `verify_observed_only_pairing.py`, `summarize_observed_only_experiment.py`, `summarize_compact_context_experiment.py` | `archive/validator_benchmark/` (validator replay, compact_v2 probe, v1.3 preflight and gate) | `observed_only_experiment/`, `prediction_only_experiment/`, `compact_context_experiment/`, `compact_v2_context_probe/` | `docs/BENCHMARK.md` §5, analyses next to each artefact |
| `rag_llm_diagnostics/` | one-off diagnostics and A/B checks on the RAG/LLM chain (Aug–Sep 2026) | `run_rag_context_failure_diagnostic.py`, `run_llm_model_ab_test.py`, `run_event_transition_validation.py`, `run_context_representation_audit.py`, `validate_grounding_change.py`, `measure_retrieval_rank.py` | — | `event_anomaly_rag_inventory/`, `grounding_numeric_normalization_ab*/` | `docs/archive/reports/RAG_*`, `EVENT_*`, `GROUNDING_NUMERIC_NORMALIZATION_REPORT.md` |
| `archive/` | scripts that only existed in the server's `/tmp`, copied bit-for-bit (37) | — | see `archive/README.md` | — | — |

Two strings inside `context_ablations/observed_only.py` and `run_prediction_only_experiment.py`
still say `orchestrator.observed_only` / `orchestrator.prediction_only`: they are provenance
labels written into the artefacts and were left unchanged so that new artefacts stay
byte-comparable with the existing ones.
