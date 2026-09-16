"""Shared contexts for the Validator tests.

The three fixtures mirror the REAL context shapes of the three benchmark
conditions, at the frozen anchor Patient_319 / window 156 / val:

    model prediction : t7 (151-156) -> t8 (157-161)
    annotation       : t4 (151-156) -> t6 (157-161), skip of t5

The model is WRONG on phase identity and RIGHT on the change boundary. That is
the real state of the frozen Semi-HMM on this window, and it is what makes
these fixtures worth using rather than invented ones.
"""

import pytest

_WINDOW = {"window_start": 156, "window_start_time": 43.1, "window_end_time": 44.8,
           "time_available": True, "time_unit": "unknown/unverified"}
_CURRENT_STATE = {"current_phase": "t7", "current_phase_index": 8,
                  "phase_probability": 0.7609993694101455,
                  "phase_probabilities": {"t7": 0.7609993694101455, "t8": 0.23679880956344662},
                  "entropy": 0.5638288292962345}
_NEXT_PHASE = {"most_likely_next_phase": "t7", "next_phase_probability": 0.7313823792655003}
_GROUND_TRUTH = {"ground_truth_phase": "t4", "consistency_flag": 0}
_OBSERVED_TRANSITION = {"from_phase": "t4", "to_phase": "t6", "observed": True,
                        "provenance": "observed_annotation", "window_start": 156,
                        "window_end": 157, "phase_distance": 2, "is_skip": True}
_MODEL_CHANGE = {"from_window": 156, "to_window": 157, "from_phase": "t7", "to_phase": "t8"}


@pytest.fixture
def prediction_only_context():
    """MODEL-DERIVED + DERIVED-from-model + RAG. No annotation anywhere."""
    return {
        "question": "Quelle etait la phase dominante juste avant la transition ?",
        "dynamic_context": {
            "get_current_inference": {"window": dict(_WINDOW),
                                      "current_state": dict(_CURRENT_STATE),
                                      "next_phase": dict(_NEXT_PHASE)},
            "model_series_analysis": {
                "per_window": {"156": {"model_phase": "t7", "phase_probability": 0.761}},
                "sequences": {"model_derived_phase_sequence": {"156": "t7", "157": "t8"}},
                "convergence": {"all_model_phase_changes": [dict(_MODEL_CHANGE)]},
                "stability": {"n_phase_returns": 0, "phase_skips": [], "phase_regressions": []},
            },
        },
        "document_context": [], "warnings": [],
    }


@pytest.fixture
def observed_only_context():
    """OBSERVED + DERIVED-from-observed + RAG. No model output anywhere."""
    return {
        "question": "Quelle etait la phase dominante juste avant la transition ?",
        "dynamic_context": {
            "get_current_inference": {"window": dict(_WINDOW),
                                      "ground_truth": dict(_GROUND_TRUTH)},
            "get_transition_events": {"n_transitions": 1,
                                      "transitions": [dict(_OBSERVED_TRANSITION)]},
        },
        "document_context": [], "warnings": [],
    }


@pytest.fixture
def full_context():
    """Both classes present -- the production-of-record shape."""
    return {
        "question": "Quelle etait la phase dominante juste avant la transition ?",
        "dynamic_context": {
            "get_current_inference": {"window": dict(_WINDOW),
                                      "current_state": dict(_CURRENT_STATE),
                                      "next_phase": dict(_NEXT_PHASE),
                                      "ground_truth": dict(_GROUND_TRUTH)},
            "get_transition_events": {"n_transitions": 1,
                                      "transitions": [dict(_OBSERVED_TRANSITION)]},
            "series_analysis": {
                "per_window": {"156": {"model_phase": "t7", "observed_phase": "t4"}},
                "sequences": {"model_derived_phase_sequence": {"156": "t7", "157": "t8"},
                              "observed_annotation_phase_sequence": {"156": "t4", "157": "t6"}},
                "convergence": {"all_model_phase_changes": [dict(_MODEL_CHANGE)]},
                "timing": {"all_observed_phase_changes": [
                    {"from_window": 156, "to_window": 157, "from_phase": "t4", "to_phase": "t6"}],
                    "model_vs_observed_lag_windows": None},
                "stability": {"n_phase_returns": 0, "phase_skips": [], "phase_regressions": []},
            },
        },
        "document_context": [], "warnings": [],
    }


@pytest.fixture(scope="session")
def corpus():
    from validator.corpus import Corpus
    return Corpus()
