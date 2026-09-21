# Training/experiments/archive — scripts d'expériences rapatriés (historique, hors production)

Ces 37 scripts existaient uniquement dans `/tmp` du serveur GPU du laboratoire
jusqu'au 2026-09-15. Ils ont été copiés **bit à bit** (SHA256 vérifiés contre le manifeste du
snapshot du 2026-09-15) sans aucune modification de contenu, avec leur nom d'origine.

Seule exception, pour la publication du dépôt (2026-09-21) : dans six scripts de
`llm_benchmark_probes/` (`analyze_bench.py`, `check_context.py`, `independent_check.py`,
`probe2.py`, `probe_enrichment.py`, `side_effects.py`), le chemin absolu du dépôt sur le serveur
a été remplacé par le placeholder `/path/to/embryonicDevelopmentSciMLExtension`. Aucune autre
ligne n'a changé ; ces six fichiers ne correspondent donc plus aux SHA256 du manifeste du
snapshot, qui reste la référence bit à bit.

- **Aucun de ces scripts n'est importé ni exécuté par l'application** (`Training/webapp_api`,
  `Training/reporting`, `Training/orchestrator`) ni par les tests. Ils ne font partie d'aucun
  chemin d'exécution de production.
- Ils ont été écrits et lancés depuis `Training/` (chemins relatifs `../Results/...`) ou depuis
  `/tmp` avec `PYTHONPATH=Training`. Relancés depuis ce dossier, les chemins relatifs et
  `sys.path` peuvent nécessiter une adaptation — à faire dans une copie, jamais ici.
- Le script `summarize_compact_context_experiment.py` de `/tmp` était identique à
  `Training/experiments/context_ablations/summarize_compact_context_experiment.py` et n'a pas été dupliqué.
- Le script producteur de `Results/evaluation/gru_identity_threshold_analysis/` n'a été retrouvé
  ni dans le dépôt ni dans `/tmp` (statut UNKNOWN).

Colonnes : résultat associé = répertoire sous `Results/evaluation/` (serveur, copie dans le
snapshot) que le script lit ou écrit ; « rapport » = document `docs/` qui en rend compte.

## semi_hmm_construction/ — construction du Semi-HMM (août 2026)

| Script | Date /tmp | Résultat associé | Statut |
|---|---|---|---|
| `phase_f_full_train_val.py` | 2026-08-23 | **`semi_hmm_weekend_phaseF/`** — ajuste et évalue le Semi-HMM de référence (dmax=268, negative_binomial) servi par `Training/reporting/model_loader.py` ; lit `e1_hmm_k7_sweep/best_model` | **critique pour la reproductibilité** |
| `phase_e_dmax_diagnostic.py` | 2026-08-23 | `semi_hmm_weekend_phaseE/` — diagnostic du choix de dmax | terminé |
| `phase_d_benchmark.py` | 2026-08-21 | `semi_hmm_weekend_phaseD/` | terminé |
| `phase_a2_censoring_check.py` | 2026-08-21 | `semi_hmm_phase_a2_censoring/` — censure structurelle tPB2/tEB | terminé |
| `semi_hmm_next_step_analysis.py` | 2026-08-23 | `semi_hmm_next_step_analysis/` (dont `DECISION.md`) ; lit `semi_hmm_weekend_phaseE/…report.json` | terminé |
| `semi_hmm_smoke_test.py` | 2026-08-21 | `semi_hmm_smoke/` — test de fumée (8 Train / 4 Val), cité par `docs/HANDOFF_SEMI_HMM.md` | fumée |
| `semi_hmm_nan_check.py` | 2026-08-21 | aucun artefact conservé | diagnostic |
| `duration_robustness_check.py` | 2026-08-21 | aucun artefact conservé | diagnostic |
| `time_next_phase.py` | 2026-08-24 | lit `semi_hmm_weekend_phaseF/model` — chronométrage de `next_phase_distribution` (rapport `docs/SEMI_HMM_PHASE_1_6_REPORT.md`) | diagnostic |
| `diag_cost.py` | 2026-08-24 | lit `semi_hmm_weekend_phaseF/model` — coût d'inférence | diagnostic |

## hmm_diagnostics/ — diagnostics HMM ayant motivé le Semi-HMM (2026-08-21)

| Script | Résultat associé | Statut |
|---|---|---|
| `hmm_duration_heterogeneity_diagnostic.py` | `e1_hmm_duration_heterogeneity_diagnostic/` ; lit `e1_hmm_k7_sweep/best_model`, `e1_hmm_phase2_analysis/report.json` (rapport `docs/HANDOFF_SEMI_HMM.md`) | terminé |
| `hmm_segment_count_control.py` | `e1_hmm_segment_count_control/` | terminé |
| `hmm_k7_winner_diagnostic.py` | lit `e1_hmm_k7_sweep/best_model` | diagnostic |
| `dump_phase2_durations.py` | lit `e1_hmm_phase2_analysis/report.json` | diagnostic |

## emission_reweighting/ — branche rééquilibrage des émissions (close, résultat négatif ; 2026-08-23/24)

| Script | Résultat associé | Rapport |
|---|---|---|
| `emission_balanced_experiment.py` | `emission_balanced/` | `docs/HANDOFF_EMISSION_BALANCED.md`, `docs/SCIENTIFIC_RESULTS.md` |
| `emission_weighted_sweep_experiment.py` | `emission_weighted_sweep/` | `docs/HANDOFF_EMISSION_WEIGHTED.md` |
| `emission_alpha025_experiment.py` | `emission_alpha025/` (clôture de la branche) | idem |
| `emission_diagnostic.py` | `semi_hmm_next_step_analysis/emission_diagnostic/` | idem |

## model_comparison/

| Script | Résultat associé |
|---|---|
| `prauc_evaluation.py` (2026-08-24) | `global_model_comparison/PRAUC_REPORT.md` ; lit `e1_hmm_k7_sweep/best_model` et `semi_hmm_weekend_phaseF/model` |

## validator_benchmark/ — Validator, contextes compacts, rebenchmark v1.3 (septembre 2026)

| Script | Rôle | Résultat associé |
|---|---|---|
| `replay_validator_compact.py` (2026-09-12) | rejeu read-only du Validator v1.1 sur les 90 réponses compact_v1 | `compact_context_experiment/validator_replay_v1_1_compact_v1.json` |
| `probe_compact_v2_contexts.py` (2026-09-12) | sonde des contextes compact_v1 vs compact_v2, sans LLM | `compact_v2_context_probe/` |
| `ctxdiff.py` (2026-09-14) | diff des contextes entre le run v1.2 et le preflight v1.3 | lit `validator_experiment/…validator_v1.json` |
| `preflight_v1_3.py` (2026-09-14) | preflight sans LLM du rebenchmark v1.3 (hashes, contextes, provider) | lit `validator_experiment/…validator_v1.json` |
| `launch_v1_3_gated.sh` (2026-09-14) | lanceur conditionnel du rebenchmark v1.3 (refuse si GPU0 occupé ou `gpu_fraction != 1.0`) ; **la copie active reste `/tmp/launch_v1_3_gated.sh` sur le serveur** | `validator_experiment/` |

## llm_benchmark_probes/ — sondes de session autour du benchmark Q1–Q15 (2026-08-26 → 09-07)

`analyze_bench.py`, `show_bench.py`, `check_context.py`, `independent_check.py`, `diag_retrieval.py`,
`rag_diag.py`, `probe2.py`, `probe_enrichment.py`, `side_effects.py`, `bench_chain.sh`, `bench_nemo_v2.sh`,
`replicate_nemo_v2.sh` — lectures et enchaînements de runs dans `event_anomaly_rag_inventory/` et
`llm_benchmark_ollama_llama3.2-latest/`. Les rapports correspondants (`docs/RAG_*`, `docs/PROJECT_CHECKPOINT.md`
2026-09-02 → 09-07e) contiennent les résultats ; ces scripts ne sont conservés que comme trace.

## reporting_cache/

| Script | Rôle |
|---|---|
| `cache_validation_benchmark.py` (2026-08-24) | mesure de l'accélération du cache d'inférence (`docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md`) |
