"""
The RAG Fixed-Question Functional Benchmark
(`docs/RAG_FIXED_QUESTION_BENCHMARK.md`) -- the project's durable primary
functional benchmark, replacing free-text-only Web App chat testing.

**DEFINITIVE 15-QUESTION LIST (2026-09-01).** This list is the functional
reference for RAG development. It fully REPLACES every earlier list (the
`H1-H7`+`Q1-Q8` set and the intermediate 15-question set are gone -- not
kept in parallel). Do NOT reword, add, or remove a question without an
explicit request.

The 15 questions, in order:
  Q1  last detected transition + when
  Q2  time already elapsed in the current phase
  Q3  dominant phase just before the transition
  Q4  second most probable phase around the transition
  Q5  when the two phases' probabilities start to converge
  Q6  how the probabilities evolve in the windows before/after the transition
  Q7  does the model detect the transition before/after the annotation, with what lag
  Q8  is the prediction stable after the transition, or are there returns to the previous phase
  Q9  is there a phase skip or a regression in the predicted sequence
  Q10 is uncertainty higher around this transition than during stable periods
  Q11 is this transition compatible with the expected developmental order
  Q12 what is the next expected developmental step per the reference
  Q13 is the observed timing compatible with the available morphokinetic references
  Q14 should the observed lag be considered atypical or within known biological variability
  Q15 could the observed pattern be a direct cleavage / reverse cleavage / chaotic division

Q11-Q15 are the genuinely RAG/documentary/biological half of the
benchmark. They are NOT designed as refusal tests -- but the ingested
Chroma corpus (30 docs, frozen 2026-08-25) was verified this session and
contains NO morphokinetic reference times, NO biological-variability
ranges, and NO definitions of direct/reverse cleavage or chaotic
division. It DOES contain the canonical phase ORDER (as the HMM's
left-to-right / forbidden-transition constraint, HANDOFF.md §4.6 /
HMM_RESEARCH_PLAN.md §4). So Q11/Q12 are partially answerable from the
corpus; Q13/Q14/Q15 currently can only be answered with an honest,
sourced "not in the available corpus" -- see
`docs/RAG_FIXED_QUESTION_BENCHMARK.md` §6 and its proposed future
morphokinetic-reference document.

Every question is typed OBSERVED / MODEL-DERIVED / DOCUMENTARY / HYBRID /
UNAVAILABLE and carries a `current_support` verdict (SUPPORTED / PARTIAL
/ KNOWN_GAP / UNAVAILABLE) plus a free-text `gaps` note. The doc is the
authoritative source for wording / typing / rationale / the capability
audit -- this module is its executable form.

`is_gap_test=True` marks questions whose CORRECT answer today still
includes an honest statement of a limitation (Q7: model timing is a
current-phase-flip proxy, not a calibrated detector; Q10: only a LOCAL
entropy band, not the whole-trajectory aggregate the question asks for;
Q13/Q14: morphokinetic / biological-variability knowledge absent from
the corpus; Q15: those events are neither annotated nor defined
anywhere). A low score on those must never be conflated with a
regression elsewhere.

CAPABILITY AUDIT SUMMARY (updated 2026-09-01c -- `get_inference_history`
multi-window tool + Q2 `get_trajectory` co-planning; see the doc §3/§4):
  SUPPORTED  : Q1, Q2 (G2 fixed), Q3, Q4, Q5 (G1), Q6 (G1), Q8 (G1),
               Q9 (G1-for-Q9 fixed 2026-09-02g -- the PREDICTED sequence,
                   its skips and its regressions now reach the context)
  PARTIAL    : Q7 (lag now estimable as a phase-flip proxy -- but see
                   docs/TEMPORAL_CONTEXT_RESTRUCTURING.md sec 5.3: at the
                   real anchor the lag is NOT computable and the correct
                   answer is to say so),
               Q10 (local entropy band, not the aggregate),
               Q11, Q12 (corpus has the phase order only)
  KNOWN_GAP  : Q13, Q14  (corpus -- morphokinetic / variability knowledge absent)
  UNAVAILABLE: Q15  (direct/reverse cleavage / chaotic division are neither
               annotated in the data nor defined in the corpus)
No question is removed for being unsupported -- the benchmark's job is to
reveal these limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

TYPE_OBSERVED = "OBSERVED"
TYPE_MODEL_DERIVED = "MODEL-DERIVED"
TYPE_DOCUMENTARY = "DOCUMENTARY"
TYPE_HYBRID = "HYBRID"
TYPE_UNAVAILABLE = "UNAVAILABLE"

CATEGORY_HISTORY = "A. Historique / Temporalité"
CATEGORY_PROBABILITIES = "B. Modèle — probabilités par phase"
CATEGORY_MODEL_VS_ANNOTATION = "C. Modèle vs annotation"
CATEGORY_UNCERTAINTY = "D. Stabilité / Incertitude"
CATEGORY_BIOLOGICAL_RAG = "E. RAG biologique / consensus / limites"

SUPPORT_SUPPORTED = "SUPPORTED"      # needed tool(s) planned, data present, correct answer achievable
SUPPORT_PARTIAL = "PARTIAL"          # some needed data reachable, some not
SUPPORT_KNOWN_GAP = "KNOWN_GAP"      # a needed capability is missing (no multi-window tool, corpus gap)
SUPPORT_UNAVAILABLE = "UNAVAILABLE"  # underlying data was never collected -- honest refusal is the answer


@dataclass(frozen=True)
class FixedQuestion:
    question_id: str  # "Q1".."Q15"
    question: str
    category: str  # one of the CATEGORY_* constants above
    type: str  # one of the TYPE_* constants above
    expected_tools: List[str] = field(default_factory=list)  # IDEAL tools a
    # correct answer needs -- NOT necessarily what the deterministic router
    # currently plans (see `gaps`). [] for a question with no supporting tool.
    expected_sources: List[str] = field(default_factory=list)
    expected_information: str = ""
    expected_behavior: str = ""
    success_criterion: str = ""
    behavior_if_absent: str = ""
    current_support: str = SUPPORT_SUPPORTED  # one of the SUPPORT_* constants
    gaps: str = ""
    video_id: Optional[str] = None
    window: Optional[int] = None
    split: str = "val"
    is_historical: bool = False  # retained for run_fixed_question_benchmark.py compatibility; always False now
    is_gap_test: bool = False


_PATIENT = "Patient_319"
_WINDOW = 156  # real, verified t4 -> t6 SKIP transition (window_start=156, window_end=157);
# immediately-preceding real transition t3 -> t4 at window_start=119 / window_end=120
# (docs/RAG_HISTORY_SCOPE_FIX.md §7 -- n_transitions=12 in the full annotation history).

FIXED_QUESTIONS = (
    FixedQuestion(
        "Q1", "Quelle a été la dernière transition détectée et à quel moment a-t-elle eu lieu ?",
        CATEGORY_HISTORY, TYPE_OBSERVED,
        expected_tools=["get_transition_events"],
        expected_sources=["Reporting API — get_transition_events (full history, window=None)"],
        expected_information="The most recent observed transition at/before window 156: t4->t6 "
                              "(window_start=156, window_end=157), a skip. Its 'moment' = window 156/157 "
                              "(and, if time data is surfaced, the corresponding window_end_time).",
        expected_behavior="Names the transition as a phase pair AND gives its window (or time); frames "
                           "it as an OBSERVED annotation transition, not a model prediction.",
        success_criterion="SUCCESS if it states t4->t6 (or t3->t4 if interpreting 'last completed before "
                           "now') AND a window/time; PARTIAL if the pair is right but 'à quel moment' is "
                           "missing; FAIL if it declines or invents a transition.",
        behavior_if_absent="If a video had no transition at all, the correct answer is an explicit "
                            "'aucune transition détectée', never a fabricated one.",
        current_support=SUPPORT_SUPPORTED,
        gaps="Routes DYNAMIC_DATA ('derniere transition'); get_transition_events planned with "
             "scope=full_history ('derniere transition' is a _TRANSITION_HISTORY_TRIGGERS entry), so "
             "the whole annotation transition list + windows reaches context. Residual risk is Layer-F "
             "only (llama3.2 previously mishandled a similar history question).",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q2", "Depuis combien de temps l’embryon se trouve-t-il dans la phase actuelle ?",
        CATEGORY_HISTORY, TYPE_MODEL_DERIVED,
        expected_tools=["get_trajectory", "get_current_inference"],
        expected_sources=["Reporting API — get_trajectory.transition_chain (segment the current window "
                          "belongs to + its observed start)",
                          "Reporting API — get_current_inference.window (window_end_time) and .duration "
                          "(model expected_duration — a DIFFERENT quantity)"],
        expected_information="ELAPSED time already spent in the current phase = current window time minus "
                              "the start time of the phase segment the window falls in. This needs the "
                              "trajectory's segment boundaries, NOT just the single window.",
        expected_behavior="Distinguishes explicitly between (a) elapsed-so-far in the current phase and "
                           "(b) the model's expected TOTAL/remaining duration for that phase; gives (a) "
                           "if reachable, else says so.",
        success_criterion="SUCCESS if it reports elapsed-in-phase grounded in the segment start time; "
                           "PARTIAL if it only reports the model expected_duration (the wrong quantity) "
                           "while noting the distinction; FAIL if it conflates the two or invents a number.",
        behavior_if_absent="Correct answer today: 'je dispose de la durée attendue de la phase par le "
                            "modèle, mais pas de la durée déjà écoulée, qui nécessiterait le segment de "
                            "trajectoire courant'.",
        current_support=SUPPORT_SUPPORTED,
        gaps="G2 FIXED (2026-09-01c): '_ELAPSED_DURATION_TRIGGERS' ('depuis combien de temps ...') now "
             "co-plans get_trajectory ALONGSIDE get_current_inference (they were mutually exclusive). "
             "Context now carries the current phase segment (get_trajectory.transition_chain: phase + "
             "observed_duration_windows) AND the current window time (get_current_inference.window). "
             "Elapsed-in-phase = current window time - segment start. Residual: the LLM must do the "
             "subtraction and not confuse it with the model's prospective expected_duration.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q3", "Quelle était la phase dominante juste avant la transition ?",
        CATEGORY_HISTORY, TYPE_OBSERVED,
        expected_tools=["get_transition_events"],
        expected_sources=["Reporting API — get_transition_events (from_phase of the bracketing transition)"],
        expected_information="from_phase of the transition bracketing window 156 = t4 (the t4->t6 skip). "
                              "'juste avant la transition' = the phase the sequence was in immediately "
                              "before the change.",
        expected_behavior="Names t4 as the phase immediately preceding the transition; OBSERVED framing.",
        success_criterion="SUCCESS if it states t4; PARTIAL if it gestures at 'the earlier phase' without "
                           "naming it; FAIL if it declines or names the post-transition phase.",
        behavior_if_absent="If no transition brackets the window, correct behaviour is 'aucune transition "
                            "à cette fenêtre, donc pas de phase 'juste avant'.'",
        current_support=SUPPORT_SUPPORTED,
        gaps="Routes DYNAMIC_DATA ('juste avant la transition', 'phase dominante'); get_transition_events "
             "planned (scope=current_window is correct here -- the bracketing transition's from_phase IS "
             "the answer). No history/G1 need.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q4", "Quelle est la deuxième phase la plus probable autour de cette transition ?",
        CATEGORY_PROBABILITIES, TYPE_MODEL_DERIVED,
        expected_tools=["get_current_inference"],
        expected_sources=["Reporting API — current_state.phase_probabilities (all 15 entries)"],
        expected_information="The SECOND-highest entry of phase_probabilities at window 156 (name + value).",
        expected_behavior="Names the runner-up phase and its real probability; frames it as the model's "
                           "second choice at this window.",
        success_criterion="SUCCESS if the 2nd-place phase and its probability are correct; PARTIAL if the "
                           "phase is right but the value is wrong/missing; FAIL if it declines.",
        behavior_if_absent="N/A — phase_probabilities is always present when get_current_inference succeeds.",
        current_support=SUPPORT_SUPPORTED,
        gaps="Routes DYNAMIC_DATA ('phase la plus probable' + 'cette transition'); get_current_inference "
             "planned, full 15-entry phase_probabilities present. Reasoning step: argmax#2. 'autour de "
             "cette transition' is answered at the single anchored window (a strict multi-window reading "
             "would need G1, but the single-window runner-up is the expected answer). Known pre-existing "
             "phase_probabilities dict-KEY grounding blind spot (phase NAME still grounded via the looser "
             "path).",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q5", "À quel moment les probabilités des deux phases commencent-elles à se rapprocher ?",
        CATEGORY_PROBABILITIES, TYPE_MODEL_DERIVED,
        expected_tools=["get_inference_history"],
        expected_sources=["Reporting API — get_inference_history: per-window phase_probabilities over the "
                          "band [center-before, center+after]"],
        expected_information="The per-window phase_probabilities series in a band around window 156; the "
                              "window where the top-2 probabilities' gap starts shrinking is read off "
                              "that series (offset_from_center of the first entry where |p1 - p2| drops).",
        expected_behavior="Uses the get_inference_history entries to locate where the two competing "
                           "phases' probabilities start converging, expressed as a window / offset; does "
                           "NOT fabricate a value beyond the returned band.",
        success_criterion="SUCCESS if it identifies a plausible convergence-onset window grounded in the "
                           "per-window series; PARTIAL if it describes the trend without pinning a window; "
                           "FAIL if it declines despite the series being present; HALLUCINATION = a window "
                           "outside the returned band or a fabricated probability.",
        behavior_if_absent="If get_inference_history is missing/failed, correct behaviour is to say the "
                            "multi-window probability series is unavailable for this window.",
        current_support=SUPPORT_SUPPORTED,
        gaps="G1 ADDRESSED (2026-09-01c): '_INFERENCE_HISTORY_TRIGGERS' ('se rapprocher') co-plans "
             "get_inference_history (default band 5 before / 5 after). Context now carries the per-window "
             "phase_probabilities series. Residual is Layer-F (the LLM must read the convergence onset "
             "off the series). The band is local -- a convergence starting >5 windows before the centre "
             "would need a wider `before`.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q6", "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?",
        CATEGORY_PROBABILITIES, TYPE_MODEL_DERIVED,
        expected_tools=["get_inference_history"],
        expected_sources=["Reporting API — get_inference_history: per-window phase_probabilities + entropy "
                          "for windows before AND after the centre"],
        expected_information="The per-window phase_probabilities / entropy trajectory across the band "
                              "around window 156 — how the dominant phase's probability rises/falls and "
                              "how entropy moves before vs after the transition.",
        expected_behavior="Describes the actual evolution read from the get_inference_history entries "
                           "(e.g. 'p(t6) rises from X at offset -5 to Y at the centre, then …'); stays "
                           "within the returned band; separates model probabilities from the observed "
                           "ground_truth per entry.",
        success_criterion="SUCCESS if the described trend matches the returned series; PARTIAL if vague; "
                           "FAIL if it declines despite the series; HALLUCINATION = a trend not supported "
                           "by the entries.",
        behavior_if_absent="If get_inference_history failed, say the multi-window series is unavailable.",
        current_support=SUPPORT_SUPPORTED,
        gaps="G1 ADDRESSED (2026-09-01c): 'dans les fenetres precedant' co-plans get_inference_history "
             "(band 5/5, symmetric -> covers 'precedant et suivant'). get_transition_events is also "
             "planned (via 'transition') for the observed anchor. Residual is Layer-F.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q7", "Le modèle détecte-t-il la transition avant ou après l’annotation, et avec quel décalage ?",
        CATEGORY_MODEL_VS_ANNOTATION, TYPE_HYBRID,
        expected_tools=["get_transition_events", "get_inference_history"],
        expected_sources=["Reporting API — get_transition_events (OBSERVED annotation window: 156/157)",
                          "Reporting API — get_inference_history: the window where the model's own "
                          "current_phase flips, read from the per-window entries (a proxy for the model "
                          "'detecting' the transition — NOT a calibrated consistency_flag_prob detector)"],
        expected_information="OBSERVED: the annotation transition window (t4->t6 at 156/157, from "
                              "get_transition_events). MODEL-DERIVED: the offset_from_center of the first "
                              "get_inference_history entry whose current_phase is the post-transition "
                              "phase. The 'décalage' = the signed difference. This is a current-phase-flip "
                              "proxy; a true calibrated detection series still only exists offline "
                              "(run_hmm_k_step_alignment_check.py).",
        expected_behavior="Gives the observed annotation window; gives the model current-phase-flip window "
                           "from get_inference_history; states the signed lag between them; keeps OBSERVED "
                           "and MODEL-DERIVED explicitly separate; frames the model side as a current-"
                           "phase-flip proxy, not a calibrated detector.",
        success_criterion="SUCCESS = both windows + a signed lag, correctly labelled and provenance-"
                           "separated; PARTIAL = one side only, or the lag without the proxy caveat; "
                           "HALLUCINATION = a lag not supported by the entries, or a calibrated-detection "
                           "claim.",
        behavior_if_absent="If get_inference_history failed, give the observed side + say the model-flip "
                            "window is unavailable.",
        current_support=SUPPORT_PARTIAL,
        gaps="Routes HYBRID (HYBRID_PATTERNS 'avant ou apres l'annotation', matched via the U+2019 fold). "
             "get_current_inference + get_transition_events + get_inference_history + retrieve_documents "
             "planned. 2026-09-01c: the model-side timing is now ESTIMABLE as a current_phase-flip offset "
             "from get_inference_history -- so the 'décalage' is answerable, but it is a phase-flip proxy, "
             "NOT the calibrated consistency_flag_prob / k-step series (still offline only). Kept "
             "is_gap_test because the honest framing of that distinction is part of a correct answer.",
        video_id=_PATIENT, window=_WINDOW, is_gap_test=True,
    ),
    FixedQuestion(
        "Q8", "La prédiction est-elle stable après la transition ou observe-t-on des retours vers la phase précédente ?",
        CATEGORY_UNCERTAINTY, TYPE_MODEL_DERIVED,
        expected_tools=["get_inference_history"],
        expected_sources=["Reporting API — get_inference_history: the model's current_phase (and "
                          "phase_probability / entropy) for each window AFTER the centre"],
        expected_information="The model's predicted current_phase for each post-transition entry of "
                              "get_inference_history: does it stay on the post-transition phase, or does "
                              "current_phase (or a rising p of an earlier phase) swing back toward t4/t5?",
        expected_behavior="Reads the post-centre entries and states whether the predicted phase is stable "
                           "or shows returns to an earlier phase, grounded in the per-window series; does "
                           "not conflate this with the OBSERVED transitions (which have no backward case).",
        success_criterion="SUCCESS if the stability/return verdict matches the returned post-centre "
                           "entries; PARTIAL if hedged; FAIL if it declines despite the series; "
                           "HALLUCINATION = a pattern not in the entries.",
        behavior_if_absent="If get_inference_history failed, say the post-transition predicted-phase "
                            "sequence is unavailable.",
        current_support=SUPPORT_SUPPORTED,
        gaps="G1 ADDRESSED (2026-09-01c): 'stable apres la transition' / 'retours vers la phase "
             "precedente' co-plan get_inference_history (band 5/5 -> 5 post-centre entries). The model's "
             "post-transition current_phase sequence is now in context. get_transition_events is also "
             "planned (scope=full_history via 'precedente') for the observed anchor. Residual is Layer-F.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q9", "Y a-t-il un saut de phase ou une régression dans la séquence prédite ?",
        CATEGORY_MODEL_VS_ANNOTATION, TYPE_HYBRID,
        expected_tools=["get_transition_events", "get_inference_history"],
        expected_sources=["Reporting API — get_transition_events (is_skip, phase_distance, signed) "
                          "-- the OBSERVED half",
                          "Reporting API — get_inference_history + the derived series_analysis "
                          "(model_derived_phase_sequence, phase_skips, phase_regressions) "
                          "-- the PREDICTED half (wired 2026-09-02g)"],
        expected_information="is_skip=True, phase_distance=+2 for the bracketing t4->t6 (a forward SKIP, "
                              "not a regression). A 'régression' would be a negative phase_distance / "
                              "backward transition — this project's data has 0/0/0 backward transitions "
                              "across 704 videos (hmm.py docstring). 'Séquence prédite' regression would "
                              "need the model's per-window predicted phase sequence (G1).",
        expected_behavior="Confirms the skip (t4->t6, t5 skipped) from is_skip; states plainly that no "
                           "regression is present in the OBSERVED transitions; flags that a regression in "
                           "the MODEL's predicted sequence cannot be checked window-by-window today.",
        success_criterion="SUCCESS if it identifies the skip AND correctly says no regression is observed; "
                           "PARTIAL if it only answers the skip half; FAIL if it declines or claims a "
                           "regression.",
        behavior_if_absent="For a non-skip bracketing transition, correct behaviour is 'aucun saut'.",
        current_support=SUPPORT_SUPPORTED,
        gaps="G1-for-Q9 FIXED (2026-09-02g): 'sequence predite' added to "
             "orchestrator._INFERENCE_HISTORY_TRIGGERS (tool planning only, router.classify() "
             "untouched, collision-checked -> matches Q9 and nothing else). Q9 now co-plans "
             "get_inference_history, so the context carries the model's per-window predicted phase "
             "sequence AND the deterministic series_analysis "
             "(model_derived_phase_sequence vs observed_annotation_phase_sequence, "
             "n_phase_skips_in_predicted_sequence, n_phase_regressions_in_predicted_sequence). "
             "Before this, the ONLY skip in Q9's context was get_transition_events' OBSERVED one, "
             "so the context invited answering the annotation's question instead of the model's. "
             "NOTE FOR THE GRADER: `expected_behavior` above still says a model-sequence regression "
             "'cannot be checked window-by-window today' -- that sentence is now STALE and was "
             "deliberately left unedited (expected_behavior/success_criterion are grading text, "
             "frozen by instruction). On real data the model sequence over W151-W161 is "
             "t7 x6 then t8 x5: 0 skips, 0 regressions.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q10", "Le niveau d’incertitude est-il plus élevé autour de cette transition que pendant les périodes stables ?",
        CATEGORY_UNCERTAINTY, TYPE_MODEL_DERIVED,
        expected_tools=["get_inference_history", "retrieve_documents"],
        expected_sources=["Reporting API — get_inference_history: entropy per window in a LOCAL band "
                          "around the transition (compare centre vs band edges)",
                          "RAG documentary corpus (calibration / entropy findings) — presence UNVERIFIED"],
        expected_information="Entropy per window in the band around window 156: is entropy higher near "
                              "the centre (the transition) than at the band edges? A LOCAL contrast is now "
                              "computable. A whole-trajectory transition-vs-stable AGGREGATE is still not "
                              "(the band is 5/5, not the full trajectory or dataset).",
        expected_behavior="Compares entropy at/near the centre against the band edges from "
                           "get_inference_history and states the local contrast; is explicit that this is "
                           "a LOCAL comparison, not the whole-trajectory/dataset aggregate the question "
                           "literally asks for (unless a documented aggregate finding is retrieved).",
        success_criterion="SUCCESS = a correct local entropy contrast, correctly scoped as local (not "
                           "over-claimed as an aggregate) OR a retrieved documented aggregate finding; "
                           "PARTIAL = local contrast without the scope caveat; HALLUCINATION = an invented "
                           "dataset-wide claim.",
        behavior_if_absent="If get_inference_history failed and no documented finding is retrieved, honest "
                            "'je ne peux comparer que localement / la tendance agrégée n'est pas exposée'.",
        current_support=SUPPORT_PARTIAL,
        gaps="G1 PARTIALLY ADDRESSED (2026-09-01c): 'periodes stables' (∩ 'cette transition' -> HYBRID) "
             "co-plans get_inference_history -> entropy per window in a LOCAL band is now in context, so a "
             "centre-vs-edges contrast is computable. The question literally asks for a transition-vs-"
             "STABLE-PERIODS aggregate, which a 5/5 band does not provide -- that still needs a wider "
             "tool or a documented finding. Kept is_gap_test for that scope gap.",
        video_id=_PATIENT, window=_WINDOW, is_gap_test=True,
    ),
    FixedQuestion(
        "Q11", "Cette transition est-elle compatible avec l’ordre attendu du développement embryonnaire ?",
        CATEGORY_BIOLOGICAL_RAG, TYPE_HYBRID,
        expected_tools=["get_transition_events", "retrieve_documents"],
        expected_sources=["Reporting API — get_transition_events (from_phase=t4, to_phase=t6, is_skip, "
                          "phase_distance=+2)",
                          "RAG corpus — canonical phase order as the HMM left-to-right / forbidden-"
                          "transition constraint (HANDOFF.md §4.6, HMM_RESEARCH_PLAN.md §4)"],
        expected_information="t4->t6 is a FORWARD move in the canonical order tPB2<tPNa<tPNf<t2<...<tEB, "
                              "skipping t5. Forward + skip = compatible with the ordering (no backward "
                              "step) but is a documented skip.",
        expected_behavior="Combines the observed transition (forward, skips t5) with the corpus's phase "
                           "ordering; concludes 'compatible (sens du développement respecté), mais avec un "
                           "saut de t5'. Does not claim a biological-consensus source the corpus lacks.",
        success_criterion="SUCCESS if it correctly judges forward-compatibility AND notes the skip, "
                           "grounded in the phase order from the corpus; PARTIAL if it judges "
                           "compatibility without the skip nuance; FAIL if it declines or gets the "
                           "direction wrong.",
        behavior_if_absent="If the phase order is not retrieved, honest 'je ne peux pas vérifier l'ordre "
                            "attendu sans le référentiel des phases'.",
        current_support=SUPPORT_PARTIAL,
        gaps="Routes HYBRID ('cette transition' + 'ordre attendu du developpement'); all 3 tools planned. "
             "The corpus HAS the phase ordering (as a modelling constraint, not a clinical reference), so "
             "forward/backward + skip are judgeable. It does NOT have an external biological-consensus "
             "statement of the 'expected order' — answer must stay scoped to the project's own ontology.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q12", "Quelle est la prochaine étape développementale attendue selon le référentiel ?",
        CATEGORY_BIOLOGICAL_RAG, TYPE_HYBRID,
        expected_tools=["get_current_inference", "retrieve_documents"],
        expected_sources=["Reporting API — get_current_inference.next_phase (model's duration-aware "
                          "one-step-ahead distribution)",
                          "RAG corpus — canonical phase order (what comes after t6)"],
        expected_information="Per the canonical order, the step after t6 is t7 (then t8, t9+, tM, ...). "
                              "The model's own next_phase distribution may or may not agree — the two "
                              "should be reported separately.",
        expected_behavior="States the next step per the reference order (t7), and separately the model's "
                           "predicted next phase; does not conflate 'référentiel' (the project's phase "
                           "ontology) with an external clinical guideline the corpus does not contain.",
        success_criterion="SUCCESS if it names t7 as the next reference step, grounded in the corpus phase "
                           "order; PARTIAL if it only gives the model's next_phase without the reference; "
                           "FAIL if it declines or invents a clinical reference.",
        behavior_if_absent="Honest 'le seul référentiel disponible est l'ordre chronologique des phases du "
                            "projet'.",
        current_support=SUPPORT_PARTIAL,
        gaps="Routes HYBRID ('prochaine etape' + 'referentiel'); get_current_inference + retrieve_documents "
             "planned (no get_transition_events — 'prochaine etape' names no transition). 'Référentiel' in "
             "the corpus = the CHRONOLOGICAL_PHASES ontology only; no external morphokinetic/ESHRE "
             "reference is ingested.",
        video_id=_PATIENT, window=_WINDOW,
    ),
    FixedQuestion(
        "Q13", "Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques disponibles ?",
        CATEGORY_BIOLOGICAL_RAG, TYPE_HYBRID,
        expected_tools=["get_transition_events", "get_current_inference"],  # + a morphokinetic reference the corpus lacks
        expected_sources=["Reporting API — get_transition_events + window time (observed transition timing)",
                          "RAG corpus — morphokinetic reference times (hpi) for t4/t6 — NOT PRESENT"],
        expected_information="The observed transition time (window 156/157, window_end_time in the "
                              "dataset's time unit) compared against reference morphokinetic times for "
                              "the t4/t6 stages. The corpus contains NO such reference table (verified "
                              "2026-09-01).",
        expected_behavior="Reports the observed timing if reachable; then honestly states that no "
                           "morphokinetic reference (hpi ranges for t4/t6) exists in the available "
                           "corpus, so compatibility cannot be assessed.",
        success_criterion="SUCCESS = observed timing given (or its unavailability noted) + explicit "
                           "'aucun repère morphocinétique dans le corpus'; HALLUCINATION = citing a "
                           "specific reference hpi value or declaring compatibility.",
        behavior_if_absent="This IS the tested case — the honest corpus-gap statement is the correct "
                            "answer.",
        current_support=SUPPORT_KNOWN_GAP,
        gaps="Corpus gap. Routes HYBRID ('cette transition' + 'reperes morphocinetiques'); all 3 tools "
             "planned. retrieve_documents runs but the ingested 30-doc corpus has NO morphokinetic "
             "reference times. Proposed fix: ingest a dedicated morphokinetic-reference document "
             "(see docs/RAG_FIXED_QUESTION_BENCHMARK.md §6).",
        video_id=_PATIENT, window=_WINDOW, is_gap_test=True,
    ),
    FixedQuestion(
        "Q14", "Le décalage observé doit-il être considéré comme atypique ou peut-il relever de la variabilité biologique connue ?",
        CATEGORY_BIOLOGICAL_RAG, TYPE_HYBRID,
        expected_tools=["get_transition_events", "get_current_inference"],  # + variability ranges the corpus lacks
        expected_sources=["Reporting API — the model-vs-annotation lag (itself a G1 gap, see Q7)",
                          "RAG corpus — known biological-variability ranges for developmental timing — NOT PRESENT"],
        expected_information="Whether the 'décalage' (model-vs-annotation lag from Q7) falls inside "
                              "documented biological variability for these transitions. BOTH inputs are "
                              "missing: the lag itself needs G1, and the corpus has no variability ranges.",
        expected_behavior="Explains that the lag cannot be quantified with current tools AND that the "
                           "corpus has no biological-variability reference to compare against; declines to "
                           "label it 'atypique' or 'normal'.",
        success_criterion="SUCCESS = both limitations stated clearly, no verdict invented; HALLUCINATION = "
                           "declaring the lag typical/atypical, or citing a variability range.",
        behavior_if_absent="This IS the tested case — the double honest limitation is the correct answer.",
        current_support=SUPPORT_KNOWN_GAP,
        gaps="G1 (the lag) + corpus gap (variability ranges). Routes HYBRID ('decalage observe' + "
             "'variabilite biologique'); get_current_inference + retrieve_documents planned (no "
             "get_transition_events — 'decalage observe' names no transition; a minor limitation, "
             "documented). Neither input to the question exists today.",
        video_id=_PATIENT, window=_WINDOW, is_gap_test=True,
    ),
    FixedQuestion(
        "Q15", "Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division chaotique ?",
        CATEGORY_BIOLOGICAL_RAG, TYPE_UNAVAILABLE,
        expected_tools=["get_transition_events"],  # only is_skip is derivable; the named events are not
        expected_sources=["Reporting API — get_transition_events (is_skip is the only structurally-"
                          "derivable atypical feature)",
                          "RAG corpus — definitions of direct/reverse cleavage / chaotic division — NOT PRESENT"],
        expected_information="Direct cleavage / reverse cleavage / chaotic division are (a) NOT annotated "
                              "anywhere in this project's data and (b) NOT defined anywhere in the ingested "
                              "corpus (both verified 2026-09-01). The only atypical structural feature "
                              "derivable is the skip transition t4->t6.",
        expected_behavior="Distinguishes 'no data source / no corpus definition exists for these events' "
                           "from 'no such event occurred'. May mention the observed skip as the one "
                           "structurally-detectable atypical feature. Never asserts one of the three "
                           "patterns is or is not present.",
        success_criterion="SUCCESS = honest 'ces motifs ne sont ni annotés dans les données ni définis "
                           "dans le corpus disponible' (optionally + the skip); HALLUCINATION = affirming "
                           "or ruling out a direct/reverse cleavage / chaotic division.",
        behavior_if_absent="This IS the tested case — the honest data+corpus absence statement is the "
                            "correct answer.",
        current_support=SUPPORT_UNAVAILABLE,
        gaps="Routes HYBRID ('motif observe' + 'direct cleavage'/'reverse cleavage'/'division chaotique'); "
             "get_current_inference + retrieve_documents planned (no get_transition_events — 'motif "
             "observe' names no transition; the skip is therefore not in context, a documented minor "
             "limitation). The three events are absent from BOTH the annotations and the corpus. Proposed "
             "fix: a morphokinetic-reference document (defines the terms) — but the per-embryo detection "
             "of these events would still require raw-frame analysis this project has never done.",
        video_id=_PATIENT, window=_WINDOW, is_gap_test=True,
    ),
)

assert len(FIXED_QUESTIONS) == 15, "expected exactly 15 questions in the fixed benchmark"
assert len({q.question_id for q in FIXED_QUESTIONS}) == 15, "question_id must be unique"
assert [q.question_id for q in FIXED_QUESTIONS] == [f"Q{i}" for i in range(1, 16)], \
    "questions must be Q1..Q15 in order"
assert all(q.type in {TYPE_OBSERVED, TYPE_MODEL_DERIVED, TYPE_DOCUMENTARY, TYPE_HYBRID, TYPE_UNAVAILABLE}
           for q in FIXED_QUESTIONS), "every question must carry a valid type"
assert all(q.current_support in {SUPPORT_SUPPORTED, SUPPORT_PARTIAL, SUPPORT_KNOWN_GAP, SUPPORT_UNAVAILABLE}
           for q in FIXED_QUESTIONS), "every question must carry a valid current_support verdict"
assert all(q.video_id == _PATIENT and q.window == _WINDOW and q.split == "val" for q in FIXED_QUESTIONS), \
    "every question is anchored to Patient_319 / val / window 156 (test split stays LOCKED)"
