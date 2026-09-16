# BENCHMARK — the 15 fixed questions, the protocol, the results so far

Consolidated on 2026-09-15 from `reference/RAG_FIXED_QUESTION_BENCHMARK.md` (the frozen
specification), `Training/orchestrator/fixed_question_benchmark.py` (the code that pins the
questions), the analyses stored next to each artefact under `Results/evaluation/` on the GPU
server, and the session log (`archive/sessions/PROJECT_CHECKPOINT.md`, cited as "checkpoint").
No score is recomputed here; nothing was re-run.

## 1. What is being measured

Not the scientific models (they are measured in `SCIENTIFIC_BACKGROUND.md`) but the **answer
layer**: given one embryo, one window and one question, does the local LLM, fed with the
Reporting API's outputs and the retrieved documents, produce an answer that is correct,
honest about gaps, and never turns a model prediction into an observation?

Anchor for all 15 questions: patient `Patient_319`, split `val`, window **156** — a real,
verified annotated `t4 → t6` **skip** transition (`window_start = 156`, `window_end = 157`);
the previous annotated transition is `t3 → t4` at windows 119/120; the annotation history holds
12 transitions. At that window the Semi-HMM predicts `t7` (second phase `t8`), which is exactly
what makes the anchor discriminating: the observed truth (`t4`) and the model's belief (`t7`)
disagree, and a good answer must say which is which.

## 2. The questions (frozen since 2026-09-01, wording verbatim, French)

| Id | Question | Category | Route | Support status as coded |
|---|---|---|---|---|
| Q1 | Quelle a été la dernière transition détectée et à quel moment a-t-elle eu lieu ? | history / time | DYNAMIC_DATA | SUPPORTED |
| Q2 | Depuis combien de temps l'embryon se trouve-t-il dans la phase actuelle ? | history / time | DYNAMIC_DATA | SUPPORTED in the spec; **real data gap** in practice (§7) |
| Q3 | Quelle était la phase dominante juste avant la transition ? | history | DYNAMIC_DATA | SUPPORTED |
| Q4 | Quelle est la deuxième phase la plus probable autour de cette transition ? | per-phase probabilities | DYNAMIC_DATA | SUPPORTED |
| Q5 | À quel moment les probabilités des deux phases commencent-elles à se rapprocher ? | probabilities | DYNAMIC_DATA | SUPPORTED (series reading) |
| Q6 | Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ? | probabilities | DYNAMIC_DATA | SUPPORTED (series reading) |
| Q7 | Le modèle détecte-t-il la transition avant ou après l'annotation, et avec quel décalage ? | model vs annotation | HYBRID | PARTIAL (gap test: the compared events are not the same event → the honest answer is "non calculable") |
| Q8 | La prédiction est-elle stable après la transition ou observe-t-on des retours vers la phase précédente ? | stability | DYNAMIC_DATA | SUPPORTED |
| Q9 | Y a-t-il un saut de phase ou une régression dans la séquence prédite ? | model vs annotation | DYNAMIC_DATA | SUPPORTED (since 2026-09-02g) |
| Q10 | Le niveau d'incertitude est-il plus élevé autour de cette transition que pendant les périodes stables ? | uncertainty | HYBRID | PARTIAL (local band only, no aggregate) |
| Q11 | Cette transition est-elle compatible avec l'ordre attendu du développement embryonnaire ? | biology / RAG | HYBRID | PARTIAL (phase order is definitional) |
| Q12 | Quelle est la prochaine étape développementale attendue selon le référentiel ? | biology / RAG | HYBRID | PARTIAL |
| Q13 | Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques disponibles ? | biology / RAG | HYBRID | **KNOWN_GAP** (no morphokinetic reference times in the corpus; time unit unverified) |
| Q14 | Le décalage observé doit-il être considéré comme atypique ou peut-il relever de la variabilité biologique connue ? | biology / RAG | HYBRID | **KNOWN_GAP** (lag itself a gap; no variability ranges in the corpus) |
| Q15 | Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division chaotique ? | biology / RAG | HYBRID | **UNAVAILABLE** (terms neither annotated nor defined in the corpus) |

For Q5, Q6, Q8, Q10, Q13, Q14 and Q15 an explicit "je ne peux pas répondre" with the reason is
a **correct** answer when the information is genuinely unavailable; a plausible answer with
invented content is a hallucination. The success criteria per question are in the code
(`success_criterion`, `expected_information`, `expected_sources`) and in the specification.

## 3. Protocol

- One LLM generation per question and replication; grading is **manual**, PASS = 1,
  PARTIAL = 0.5, FAIL = 0, summed over 15 questions (or 45 answers for 3 replications).
  Grading conventions in force since 2026-09-09: Q7 "non calculable" = PASS at this anchor;
  a Q9 answer that only covers the model side = PARTIAL.
- Provider: local Ollama, `mistral-nemo:12b` (digest `e7e06d107c6c…`, Q4_0, Ollama 0.7.0),
  `num_ctx = 16384`, `seed = None`, `temperature = None`, timeout 240 s for benchmark runs.
  The production default model is `llama3.2:latest` (see `ARCHITECTURE.md`); the benchmark
  showed it scores 1.0/15 on this task, which is why `mistral-nemo` is the benchmark model.
- Run identity: `python -m orchestrator.capture_run_identity --label <run>` records the model
  digest, num_ctx, timeout, seed/temperature, Ollama version, system-prompt and question hashes,
  RAG index identity, module hashes and the GPU placement of the model. Two runs are comparable
  only if these match.
- **Replication is mandatory.** With byte-identical contexts, the run-to-run standard deviation
  of the score is 0.75/15 (n = 4, 2026-09-07c). Detecting a 0.6-point effect at that noise
  level would need ≈ 25 runs per arm. Single runs are therefore read for *invariant* failures,
  not for score differences; paired designs (same raw answer, two treatments) are preferred.
- GPU rule: the model must be entirely in GPU0 VRAM (`gpu_fraction = 1.0`). A run under CPU
  offload is not comparable and is invalidated, never scored (§6).
- `split = test` is refused with HTTP 403 at every layer; the benchmark lives on `val`.

## 4. Score history (mistral-nemo unless stated; sources in the checkpoint)

| Date | Condition | Score /15 | Note |
|---|---|---|---|
| 2026-09-02d | `llama3.2:latest`, context v2 | **1.0** (10 refusals) | production model on this task |
| 2026-09-02d | `mistral-nemo:12b`, single-variable model swap | **5.5** | the reference baseline |
| 2026-09-03 | after the temporal-context restructuring | 4.5 (corrected Q9 text) / 3.5 (frozen text) | regression within the ±0.5 noise floor |
| 2026-09-07b | `llama3.1:8b` | 4.0 | nemo still ahead |
| 2026-09-07b/c | prompt v2 "reading method", n = 4 replications | mean **5.62**, σ 0.75, range 4.5–6.0 | central methodological finding: replication needed |
| 2026-09-09→11 | FULL vs OBSERVED_ONLY, paired, n = 3 | FULL **6.83** (σ 2.52) vs OBSERVED_ONLY **4.50** (σ 1.00) | Δ −2.33, t(2) = −1.32, n.s. |
| 2026-09-11 | PREDICTION_ONLY, n = 3 (unpaired) | **4.67** (σ 1.44) | see §5 |
| 2026-09-12 | context v2 vs compact_v1, paired, n = 3 | V2 **5.00** vs COMPACT **6.17** | Δ +1.17, t(2) = 1.07, n.s.; not adopted (Q3 regressed 3/3 → 1/3) |
| 2026-09-12 | Validator v1.2 OFF vs ON, compact_v2, n = 3 | OFF **18.5/45**, ON **19.0/45** | see §6; not comparable with the v2-context history |

Per-question OFF scores of the last valid run (sum over 3 replications): Q1 2, Q2 2.5, Q3 3,
Q4 1, Q5 0, Q6 1.5, Q7 0.5, Q8 1, Q9 1.5, Q10 2, Q11 0, Q12 2, Q13 0, Q14 1, Q15 0.5.

## 5. The three-condition experiment (FULL / OBSERVED_ONLY / PREDICTION_ONLY)

Same 15 questions, same model, the context filtered to keep only observed data, or only
model-derived data. Artefacts: `observed_only_experiment/paired_mistral-nemo_12b_paired_v1.json`
(SHA-256 `595c960c…62d3`) and `prediction_only_experiment/paired_mistral-nemo_12b_prediction_only_v1.json`
(`a06d5da8…6b72e0`), analyses `analysis_paired_v1.md` and `analysis_three_conditions_v1.md`.

Finding that motivated the Validator: on the questions whose answer is model-derived, removing
the annotations costs nothing (FULL 9.5/18 = PREDICTION_ONLY 9.5/18); on the questions whose
answer is observed, PREDICTION_ONLY collapses (0.0/6 vs 5.5/6) while remaining "grounded" — in
Q3 the model's `t7` is stated as *the observed* phase in 3/3 replications, with zero invented
tokens. In the project's words: **the LLM turns a model output into an observation**. The
grounding check cannot see this, because it verifies presence, not provenance or correctness.

## 6. The Scientific Validator

`Training/validator/` + `Training/orchestrator/scientific_validation.py`, on by default
(`EMBRYO_SCIENTIFIC_VALIDATOR=on|off`, or `answer_question(..., validate=...)`).

- The raw answer is **never rewritten**. The service returns `response` (raw) and
  `final_answer` = raw + a "QUALIFICATION SCIENTIFIQUE" block, plus a `validation` report
  (claims, provenance, verdicts). A Validator failure degrades to "raw answer + error".
- Verdicts: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, NOT_ASSESSABLE (an honest limit,
  not a violation), CONTRADICTED (no emitter yet).
- Rules: R1 observation ≠ interpretation · **R2** prediction ≠ observation (the reason the
  layer exists) · R2b transition provenance · R3 annotated skip ≠ predicted skip · R4 absence
  of evidence · R5 compatible ≠ proven · R6 association ≠ causation · R7 corpus silent →
  NOT_ASSESSABLE · R8 definition absent → NOT_ASSESSABLE · R9/R9b timing needs a sourced unit
  and origin, median ≠ range · **R9c** (v1.3) morphokinetic-compatibility verdict with an
  unverified time unit → NOT_ASSESSABLE · R10 no IVF/ICSI assumption · **R11** offset ≠ lag ·
  **R12** (v1.3) asserted return/regression against `n_phase_returns = 0` → NOT_SUPPORTED ·
  R13 clinical caution · **R14** the Consensus never validates a prediction.
- The Istanbul Consensus corpus is read directly (`docs/corpus/`), lexically; it is not in the
  vector index. Also injected as a `[SCIENTIFIC]` context block for the biology questions.

**Paired OFF vs ON design** (the only valid way to measure it): one generation per question and
replication; OFF = the raw answer, ON = the same raw answer + qualification. Never two
generations, never a re-run of a single question.

**v1.2 result** (2026-09-12, `validator_experiment/paired_validator_off_vs_on_mistral-nemo_12b_validator_v1.json`,
SHA-256 `f8b0364c…25ae9`, complete, 45 raw answers × 2 arms, 0 errors): OFF 18.5/45, ON 19.0/45
(one PARTIAL → PASS). 21/45 raw answers hallucinated; v1.2 targeted 12 correctly, 4 wrongly,
and produced **5 false positives** on non-hallucinated answers (two of them PASS). Conclusion of
the audit: valid benchmark, gain of the Validator **not demonstrated** (n = 3, one grader, no
blinding possible).

**v1.3** (2026-09-14, precision-first, no score target): replayed read-only on the 45 stored
answers, v1.2 reproduced bit-for-bit before the change; after it, the 5 false positives
disappear, no NOT_SUPPORTED remains on a non-hallucinated answer, every correct detection is
kept; new rules R9c and R12; qualification added on 16/45 answers instead of 27/45. Tests: 184
validator/integration, full suite 1 218 passed.

**v1.3 rebenchmark status (2026-09-15): WAITING_FOR_VALID_GPU.** The first attempt
(2026-09-14, label `validator_v1_3`) was **invalidated**: an external `torchrun`
hyper-parameter search held 6.3 GB of GPU0, Ollama placed only 48 % of the model in VRAM,
Q1 took 97 s instead of 15.8 s and Q5 hit the 240 s timeout. It was stopped after Q1–Q5;
the partial artefact (`…validator_v1_3.json`, SHA-256 `927327f7…1776`, `complete = false`) is
kept next to a `.INVALID.txt` stating `GPU_CONTENTION_CPU_OFFLOAD` and **must never be
scored**. A gated launcher (`Training/experiments/archive/validator_benchmark/launch_v1_3_gated.sh`,
active copy in the server's `/tmp`) refuses to start unless GPU0 carries no foreign process
and `gpu_fraction == 1.0`; it will write the label `validator_v1_3b`. The external job was at
trial 31/50 on 2026-09-15 with an estimated end around 2026-09-16 20:40 CEST.

## 7. Known structural limits of the benchmark

- **Q2 is a real data gap**: `get_trajectory.transition_chain` carries phase, observed
  duration and next-phase distribution but no segment start instant, so "how long in the
  current phase" is not derivable from the context (checkpoint 2026-09-07c, C4). The FAIL is
  kept for comparability.
- **Q5/Q6** (reading trends in a probability series) are outside what a lexical validator
  can check; only manual grading catches them.
- **Q13/Q14/Q15** are gap tests by construction; their PASS is an honest limitation.
- The **time unit** is `unknown/unverified` in every artefact (R9c enforces it); the owner's
  statement that it is hours is not recorded in the pipeline.
- Grounding rate is **not** a quality score: after one fix the grounded count rose 8 → 11/15
  while the score fell.
- One grader, no blinding, n = 3: differences below ≈ 1.5/15 are not interpretable.

## 8. Artefact registry (server `Results/evaluation/`, copies in the 2026-09-15 snapshot)

| Artefact | Status |
|---|---|
| `event_anomaly_rag_inventory/fixed_question_benchmark_*.json` (llama3.2 vs nemo, post-restructuring, prompt_v2 + 3 replications, `run_identity_*.json`) | valid, context v2 |
| `observed_only_experiment/paired_mistral-nemo_12b_paired_v1.json` — `595c960c…` | valid, 90/90 |
| `prediction_only_experiment/paired_mistral-nemo_12b_prediction_only_v1.json` — `a06d5da8…` | valid, 45/45 |
| `compact_context_experiment/paired_mistral-nemo_12b_compact_v1.json` — `06c74438…`, `grades_compact_v1.json`, `analysis_compact_v1.md` | valid, 90/90, manually graded |
| `validator_experiment/paired_validator_off_vs_on_mistral-nemo_12b_validator_v1.json` — `f8b0364c…` | valid, the v1.2 reference |
| `validator_experiment/…validator_v1_3.json` — `927327f7…` + `.INVALID.txt` | **INVALID**, partial, never to be scored |

Runners: `Training/orchestrator/run_fixed_question_benchmark.py` (context formats v2 / compact
/ compact_v2), `run_validator_experiment.py` (frozen on compact_v2, `--run-label`,
`--replications`, `--timeout`), `summarize_fixed_question_runs.py` (mechanical side-by-side).
