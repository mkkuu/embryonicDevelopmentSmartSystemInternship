"""
Integrity tests for the FULL vs OBSERVED-ONLY context-ablation experiment
(`Training/experiments/context_ablations/observed_only.py`).

Four guarantees, each asserted independently of the code that is supposed to
provide it:

  1. OBSERVED-ONLY carries NO model-derived data, at any depth.
  2. The OBSERVED data is strictly identical between the two conditions.
  3. The frozen 15-question benchmark is unchanged by this experiment.
  4. No scientific model / RAG corpus definition was modified.

Synthetic fixtures only -- no GPU, no cached embeddings, no Chroma index, no
Ollama. The fixture mirrors the real payload shapes
(`Training/reporting/schemas.py`'s `to_dict()` outputs) rather than importing
them, so these tests run in an environment with no torch installed.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import temporal_context
from experiments.context_ablations import observed_only
from orchestrator.fixed_question_benchmark import (
    FIXED_QUESTIONS,
    TYPE_HYBRID,
    TYPE_MODEL_DERIVED,
    TYPE_OBSERVED,
)


# ---------------------------------------------------------------------------
# fixtures -- the real payload shapes, with obviously-identifiable values
# ---------------------------------------------------------------------------

def _history_entry(window: int, offset: int, is_center: bool, model_phase: str,
                   observed_phase: str) -> dict:
    return {
        "window_start": window,
        "offset_from_center": offset,
        "is_center": is_center,
        "window_start_time": 40.0 + window / 10,
        "window_end_time": 41.0 + window / 10,
        "time_available": True,
        "model_derived": {
            "current_phase": model_phase,
            "current_phase_index": 5,
            "phase_probability": 0.777,
            "phase_probabilities": {model_phase: 0.777, "t5": 0.223},
            "entropy": 0.4242,
            "most_likely_next_phase": "t7",
            "next_phase_probability": 0.61,
            "next_phase_distribution": {"t7": 0.61, "t8": 0.39},
        },
        "observed": {"ground_truth_phase": observed_phase, "consistency_flag": 0},
    }


@pytest.fixture
def full_context() -> dict:
    history = {
        "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
                   "patient_id": "Patient_319", "split": "val"},
        "center_window": 156,
        "requested_before": 5, "requested_after": 5,
        "n_windows_before": 2, "n_windows_after": 1, "n_windows": 4,
        "window_starts": [154, 155, 156, 157],
        "entries": [
            _history_entry(154, -2, False, "t4", "t4"),
            _history_entry(155, -1, False, "t4", "t4"),
            _history_entry(156, 0, True, "t6", "t4"),
            _history_entry(157, 1, False, "t6", "t6"),
        ],
        "model": {"model_name": "semi_hmm", "model_version": "dmax=268,negative_binomial",
                  "model_configuration": {"dmax": 268}, "model_source": "Results/...",
                  "inference_timestamp": "2026-09-09T10:00:00"},
        "warnings": ["band clamped"],
    }
    dynamic = {
        "get_current_inference": {
            "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
                       "patient_id": "Patient_319", "split": "val"},
            "window": {"window_start": 156, "window_start_time": 55.6, "window_end_time": 56.6,
                       "window_mid_time": 56.1, "window_duration": 1.0, "time_available": True,
                       "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": 0},
            "current_state": {"current_phase": "t6", "current_phase_index": 5,
                              "phase_probability": 0.777,
                              "phase_probabilities": {"t6": 0.777, "t5": 0.223},
                              "entropy": 0.4242},
            "next_phase": {"most_likely_next_phase": "t7", "next_phase_probability": 0.61,
                           "next_phase_distribution": {"t7": 0.61}},
            "duration": {"duration_available": True, "expected_duration": 12.5,
                         "duration_quantiles": {"0.5": 12}, "duration_unit": "windows"},
            "model": {"model_name": "semi_hmm", "model_version": "dmax=268",
                      "model_configuration": {"dmax": 268}, "model_source": "Results/...",
                      "inference_timestamp": "2026-09-09T10:00:00"},
            "ground_truth": {"ground_truth_phase": "t4", "consistency_flag": 1},
            "warnings": ["duration not estimable (structural censoring)"],
        },
        "get_trajectory": {
            "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
                       "patient_id": "Patient_319", "split": "val"},
            "n_windows": 527,
            "window_starts": [0, 1, 2],
            "transition_chain": [
                {"phase": "t3", "observed_duration_windows": 37,
                 "duration_probability_at_observed": 0.0123,
                 "next_phase_distribution": {"t4": 0.9}},
                {"phase": "t4", "observed_duration_windows": 37,
                 "duration_probability_at_observed": 0.0456,
                 "next_phase_distribution": {"t6": 0.7}},
            ],
            "model": {"model_name": "semi_hmm", "model_version": "dmax=268",
                      "model_configuration": {}, "model_source": "x",
                      "inference_timestamp": "2026-09-09T10:00:00"},
        },
        "get_transition_events": {
            "video_id": "Patient_319", "split": "val", "window": None, "n_transitions": 1,
            "transitions": [{
                "from_phase": "t4", "to_phase": "t6", "from_phase_index": 3, "to_phase_index": 5,
                "observed": True, "provenance": "observed_annotation",
                "window_start": 156, "window_end": 157,
                "frame_start": 167, "frame_end": 175, "frame_mapping_available": True,
                "window_start_time": 43.1, "window_end_time": 43.4, "time_available": True,
                "phase_distance": 2, "is_skip": True,
            }],
        },
        "get_inference_history": history,
        "series_analysis": temporal_context.derive_series_analysis(history),
    }
    return {
        "question": "Q?",
        "route": {"category": "DYNAMIC_DATA"},
        "document_context": [{"tool": "retrieve_documents", "source_type": "document",
                              "content": "L'ordre canonique des phases ...",
                              "source": "docs/HANDOFF.md", "section": "4.6",
                              "status": "authoritative", "authoritative": True, "score": 0.68}],
        "dynamic_context": dynamic,
        "model_metadata": None,
        "warnings": [],
        "provenance": [],
        "tools_called": list(dynamic.keys()),
    }


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# 1. OBSERVED-ONLY contains no model-derived data
# ---------------------------------------------------------------------------

def test_no_forbidden_key_survives_anywhere(full_context):
    filtered = observed_only.filter_context(full_context)
    assert observed_only.find_model_derived_keys(filtered["dynamic_context"]) == []
    observed_only.assert_no_model_derived(filtered)


def test_the_full_context_would_fail_the_same_check(full_context):
    """Control: the checker is not vacuous -- it must flag the FULL context."""
    offenders = observed_only.find_model_derived_keys(full_context["dynamic_context"])
    assert offenders, "the model-derived detector found nothing in the FULL context"
    with pytest.raises(AssertionError):
        observed_only.assert_no_model_derived(full_context)


def test_model_derived_values_are_absent_from_the_rendered_text(full_context):
    """Value-level, not just key-level: the numbers themselves must be gone
    from what the LLM actually receives."""
    from orchestrator import prompt as prompt_module

    filtered = observed_only.filter_context(full_context)
    text = prompt_module.build_user_prompt(filtered)
    for value in ("0.777", "0.4242", "0.61", "12.5", "0.0123", "0.0456", "semi_hmm", "dmax=268"):
        assert value not in text, f"model-derived value {value!r} leaked into the prompt"


def test_the_rendered_prompt_shows_no_model_derived_section_or_field_name(full_context):
    """Not just values: the OBSERVED-ONLY prompt must not carry the model's
    field NAMES either (the `W151.phase_probability = null` shape)."""
    from orchestrator import prompt as prompt_module

    text = prompt_module.build_user_prompt(observed_only.filter_context(full_context))
    assert "MODEL-DERIVED" not in text
    for name in ("phase_probability", "model_phase", "second_phase", "entropy",
                  "next_phase_distribution", "probability_gap", "expected_duration"):
        assert name not in text, f"model field name {name!r} still rendered in the prompt"
    # and the annotation IS there, window-addressed
    assert "observed_phase" in text
    assert "156" in text


def test_no_experimental_condition_marker_is_injected(full_context):
    """The filter SUBTRACTS only. An `experimental_condition` block existed in
    an earlier draft and was removed: an instruction given to arm B and not to
    arm A would make the two arms differ by more than the ablation."""
    filtered = observed_only.filter_context(full_context)
    assert "experimental_condition" not in filtered["dynamic_context"]
    assert not hasattr(observed_only, "condition_marker")
    # Every key of the filtered context comes from the full one, except the
    # observed-only recomputation and the bookkeeping fields.
    added = set(filtered["dynamic_context"]) - set(full_context["dynamic_context"])
    assert added <= {"observed_series_analysis"}, f"the filter ADDED {added}"


def test_no_arm_specific_instruction_reaches_the_llm(full_context):
    """No imperative addressed to the model may appear in arm B's prompt but
    not in arm A's. Checked on the real rendered difference, not on intent."""
    from orchestrator import prompt as prompt_module

    text_full = prompt_module.build_user_prompt(full_context)
    text_observed = prompt_module.build_user_prompt(observed_only.filter_context(full_context))

    # French second-person imperatives / directive phrasings an instruction
    # would use. None may be present in B unless it is also present in A.
    directives = ("reponds", "réponds", "ne reconstruis", "tu dois", "il faut",
                  "dis-le", "n'invente", "explicitement, sans", "condition experimentale",
                  "condition expérimentale")
    for directive in directives:
        in_b = directive.lower() in text_observed.lower()
        in_a = directive.lower() in text_full.lower()
        assert not (in_b and not in_a), (
            f"arm-B-only directive {directive!r} found in the OBSERVED-ONLY prompt"
        )


def test_the_system_prompt_is_identical_for_both_arms(full_context):
    """The ablation is in the user prompt only; the system prompt is one shared
    object neither arm can alter."""
    from orchestrator import prompt as prompt_module

    assert prompt_module.build_system_prompt() == prompt_module.build_system_prompt()
    # and nothing in this experiment touches it
    assert "SYSTEM_PROMPT" not in _code_only(observed_only)
    assert "build_system_prompt" not in _code_only(observed_only)


def test_specific_model_blocks_are_dropped(full_context):
    filtered = observed_only.filter_context(full_context)["dynamic_context"]
    assert "series_analysis" not in filtered
    assert "current_state" not in filtered["get_current_inference"]
    assert "next_phase" not in filtered["get_current_inference"]
    assert "duration" not in filtered["get_current_inference"]
    assert "model" not in filtered["get_current_inference"]
    for segment in filtered["get_trajectory"]["transition_chain"]:
        assert set(segment) == {"phase", "observed_duration_windows"}
    # The history key itself is gone: its renderer always prints a MODEL-DERIVED
    # section, so keeping the key (even emptied) would show the model's field
    # names with null values. Its annotation lives in observed_series_analysis.
    assert "get_inference_history" not in filtered
    assert "observed_series_analysis" in filtered


def test_get_model_metadata_is_dropped_entirely(full_context):
    full_context["dynamic_context"]["get_model_metadata"] = {"model_version": "dmax=268"}
    filtered = observed_only.filter_context(full_context)
    assert "get_model_metadata" not in filtered["dynamic_context"]


def test_an_unknown_future_tool_is_dropped_by_default(full_context):
    """Whitelist discipline: a tool nobody taught the filter about must be
    excluded, never passed through."""
    full_context["dynamic_context"]["get_some_future_model_tool"] = {"prediction": 0.9}
    filtered = observed_only.filter_context(full_context)
    assert "get_some_future_model_tool" not in filtered["dynamic_context"]


def test_model_derived_fields_are_removed_not_overwritten_with_annotations(full_context):
    """The experiment's own explicit rule: never substitute the annotation into
    a model-derived field."""
    filtered = observed_only.filter_context(full_context)["dynamic_context"]
    analysis = filtered["observed_series_analysis"]
    # The annotation is still under its own name, with its own value -- and the
    # window it belongs to is the DICT KEY, never an array index.
    assert analysis["per_window"]["156"]["observed_phase"] == "t4"
    assert "model_phase" not in analysis["per_window"]["156"]
    assert "current_phase" not in analysis["per_window"]["156"]
    assert "model_derived_phase_sequence" not in analysis
    assert "model_phase" not in _canonical(analysis)
    assert analysis["observed_annotation_phase_sequence"]["156"] == "t4"


# ---------------------------------------------------------------------------
# 2. OBSERVED data is identical between conditions
# ---------------------------------------------------------------------------

def test_observed_payload_is_byte_identical_between_conditions(full_context):
    filtered = observed_only.filter_context(full_context)
    assert _canonical(observed_only.observed_payload(full_context)) == \
           _canonical(observed_only.observed_payload(filtered))


def test_every_observed_field_survives_the_filter(full_context):
    filtered = observed_only.filter_context(full_context)["dynamic_context"]
    assert filtered["get_current_inference"]["ground_truth"] == \
           full_context["dynamic_context"]["get_current_inference"]["ground_truth"]
    assert filtered["get_transition_events"] == \
           full_context["dynamic_context"]["get_transition_events"]
    per_window = filtered["observed_series_analysis"]["per_window"]
    for before in full_context["dynamic_context"]["get_inference_history"]["entries"]:
        after = per_window[str(before["window_start"])]
        assert after["observed_phase"] == before["observed"]["ground_truth_phase"]
        assert after["consistency_flag"] == before["observed"]["consistency_flag"]
        assert after["window_start_time"] == before["window_start_time"]
        assert after["offset_from_center"] == before["offset_from_center"]


def test_rag_document_context_is_untouched(full_context):
    filtered = observed_only.filter_context(full_context)
    assert filtered["document_context"] == full_context["document_context"]
    assert _canonical(filtered["document_context"]) == _canonical(full_context["document_context"])


def test_the_filter_never_mutates_its_input(full_context):
    before = _canonical(full_context)
    observed_only.filter_context(full_context)
    assert _canonical(full_context) == before, "filter_context() mutated the FULL context"


def test_observed_derived_analysis_uses_only_annotations(full_context):
    analysis = observed_only.filter_context(full_context)["dynamic_context"]["observed_series_analysis"]
    assert analysis["provenance"] == observed_only.PROVENANCE_OBSERVED_DERIVED
    # The annotation changes t4 -> t6 at window 157 in the fixture; the MODEL
    # changes at 156. The observed-derived block must report the ANNOTATION's.
    assert analysis["all_observed_phase_changes"] == [
        {"from_window": 156, "to_window": 157, "from_phase": "t4", "to_phase": "t6"}
    ]
    assert analysis["n_observed_phase_changes"] == 1
    assert analysis["observed_phase_at_center_window"] == "t4"
    for forbidden in ("probability_gap", "convergence", "entropy", "stability",
                      "model_vs_observed_lag_windows"):
        assert forbidden not in analysis


def test_observed_derived_analysis_is_empty_for_an_empty_band():
    assert observed_only.derive_observed_series_analysis({"entries": []}) == {}
    assert observed_only.derive_observed_series_analysis(None) == {}


def test_removal_report_lists_what_was_dropped(full_context):
    filtered = observed_only.filter_context(full_context)
    removed = filtered["observed_only_removed_paths"]
    assert "get_current_inference.current_state" in removed
    assert "get_current_inference.duration" in removed
    assert "get_trajectory.transition_chain[0].duration_probability_at_observed" in removed
    assert "get_inference_history.entries[0].model_derived" in removed
    assert any(path.startswith("get_inference_history (whole block") for path in removed)
    assert any(path.startswith("series_analysis") for path in removed)


# ---------------------------------------------------------------------------
# 3. the frozen benchmark is unchanged
# ---------------------------------------------------------------------------

FROZEN_QUESTION_TEXTS = (
    "Quelle a été la dernière transition détectée et à quel moment a-t-elle eu lieu ?",
    "Depuis combien de temps l’embryon se trouve-t-il dans la phase actuelle ?",
    "Quelle était la phase dominante juste avant la transition ?",
    "Quelle est la deuxième phase la plus probable autour de cette transition ?",
    "À quel moment les probabilités des deux phases commencent-elles à se rapprocher ?",
    "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?",
    "Le modèle détecte-t-il la transition avant ou après l’annotation, et avec quel décalage ?",
    "La prédiction est-elle stable après la transition ou observe-t-on des retours vers la phase "
    "précédente ?",
    "Y a-t-il un saut de phase ou une régression dans la séquence prédite ?",
    "Le niveau d’incertitude est-il plus élevé autour de cette transition que pendant les périodes "
    "stables ?",
    "Cette transition est-elle compatible avec l’ordre attendu du développement embryonnaire ?",
    "Quelle est la prochaine étape développementale attendue selon le référentiel ?",
    "Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques "
    "disponibles ?",
    "Le décalage observé doit-il être considéré comme atypique ou peut-il relever de la variabilité "
    "biologique connue ?",
    "Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division "
    "chaotique ?",
)


def test_the_fifteen_questions_are_unchanged_in_wording_and_order():
    assert len(FIXED_QUESTIONS) == 15
    assert tuple(q.question for q in FIXED_QUESTIONS) == FROZEN_QUESTION_TEXTS
    assert tuple(q.question_id for q in FIXED_QUESTIONS) == tuple(f"Q{i}" for i in range(1, 16))


def test_the_anchor_and_split_are_unchanged():
    for q in FIXED_QUESTIONS:
        assert q.video_id == "Patient_319"
        assert q.window == 156
        assert q.split == "val"


def test_success_criteria_are_non_empty_and_not_rewritten_by_this_experiment():
    """The rubric is the frozen `success_criterion`; this experiment adds no
    criterion of its own and rewrites none."""
    for q in FIXED_QUESTIONS:
        assert q.success_criterion.strip()
    assert "success_criterion" not in _code_only(observed_only), (
        "observed_only.py must not define or alter any scoring criterion"
    )


def test_computability_is_derived_from_the_frozen_type_only():
    assert observed_only.computability_of(TYPE_MODEL_DERIVED) == observed_only.NOT_COMPUTABLE
    assert observed_only.computability_of(TYPE_HYBRID) == observed_only.PARTIALLY_COMPUTABLE
    assert observed_only.computability_of(TYPE_OBSERVED) == observed_only.COMPUTABLE
    # Applied to the real benchmark: the model-derived questions are the ones
    # pre-declared not computable, and nothing else.
    not_computable = {q.question_id for q in FIXED_QUESTIONS
                      if observed_only.computability_of(q.type) == observed_only.NOT_COMPUTABLE}
    assert not_computable == {q.question_id for q in FIXED_QUESTIONS
                              if q.type == TYPE_MODEL_DERIVED}


# ---------------------------------------------------------------------------
# 4. nothing scientific / RAG-side was modified
# ---------------------------------------------------------------------------

def _imported_module_names(module) -> set:
    """Every module name the module actually imports, read from its AST --
    never from its prose, so a docstring naming a module it deliberately does
    NOT use (e.g. explaining where a value originally came from) cannot fail
    the check."""
    import ast
    import inspect

    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _code_only(module) -> str:
    """Module source with comments and every docstring removed -- what the
    module DOES, not what it says about the rest of the system."""
    import ast
    import inspect
    import io
    import tokenize

    source = inspect.getsource(module)
    without_comments = tokenize.untokenize(
        token for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    )
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    for doc in docstrings:
        without_comments = without_comments.replace(doc, "")
    return without_comments


def test_experiment_modules_do_not_import_or_touch_the_scientific_models():
    from experiments.context_ablations import run_observed_only_experiment

    for module in (observed_only, run_observed_only_experiment):
        imported = _imported_module_names(module)
        for forbidden in ("evaluation", "evaluation.models", "evaluation.models.semi_hmm",
                          "evaluation.models.hmm", "reporting.model_loader"):
            assert forbidden not in imported, (
                f"{module.__name__} imports {forbidden!r} -- this experiment must never "
                f"touch a scientific model"
            )
        code = _code_only(module)
        for forbidden in ("SemiHMMModel", "HMMModel", ".fit(", ".predict(", "model_loader"):
            assert forbidden not in code, (
                f"{module.__name__} calls/uses {forbidden!r} in code -- this experiment must "
                f"never touch a scientific model"
            )


def test_experiment_modules_do_not_modify_the_rag_corpus_or_index():
    from experiments.context_ablations import run_observed_only_experiment

    for module in (observed_only, run_observed_only_experiment):
        imported = _imported_module_names(module)
        for forbidden in ("rag", "rag.ingest", "rag.inventory", "rag.vectorstore",
                          "rag.embeddings"):
            assert forbidden not in imported, (
                f"{module.__name__} imports {forbidden!r} -- the RAG corpus/index is frozen"
            )
        code = _code_only(module)
        for forbidden in ("INCLUDED", "vectorstore", "persist_dir", "ingest("):
            assert forbidden not in code, (
                f"{module.__name__} uses {forbidden!r} in code -- the RAG corpus/index is frozen"
            )


def test_the_rag_corpus_inventory_still_declares_thirty_documents():
    """Guards the corpus definition itself against an incidental edit."""
    from rag import inventory

    assert len(inventory.INCLUDED) == 30
    assert all(entry.source.startswith("docs/") for entry in inventory.INCLUDED)


def test_observed_only_never_calls_a_tool_or_the_reporting_api():
    code = _code_only(observed_only)
    for forbidden in ("tools.get_", "requests.", "urllib", "open("):
        assert forbidden not in code, (
            f"observed_only.py must be a pure function over an already-built context "
            f"(found {forbidden!r})"
        )


def test_the_production_entry_point_does_not_import_this_experiment():
    """`answer_question()` must be unaffected: the FULL arm is production."""
    from orchestrator import orchestrator as orchestrator_module

    assert "observed_only" not in _imported_module_names(orchestrator_module)
    assert "observed_only" not in _code_only(orchestrator_module)
