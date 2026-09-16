"""Pure dataclass shape/serialization tests -- no model, no filesystem."""

from reporting.schemas import (
    CurrentState, DurationInfo, GroundTruthInfo, InferenceRecord, ModelInfo,
    NextPhaseInfo, SampleInfo, TrajectorySummary, WindowInfo,
)


def _sample():
    return SampleInfo(sample_id="Patient_1#val", video_name="Patient_1", patient_id="Patient_1", split="val")


def _window():
    return WindowInfo(window_start=5, window_start_time=3.1, window_end_time=4.8, window_mid_time=3.95,
                       window_duration=1.7, time_available=True, time_unit="unknown/unverified",
                       n_zero_time_diffs_in_window=0)


def _current_state():
    probs = {"tPB2": 0.1, "t9+": 0.9}
    return CurrentState(current_phase="t9+", current_phase_index=10, phase_probability=0.9,
                         phase_probabilities=probs, entropy=0.325)


def _next_phase():
    return NextPhaseInfo(next_phase_distribution={"t9+": 0.8, "tM": 0.2},
                          most_likely_next_phase="t9+", next_phase_probability=0.8)


def _duration(available=True):
    if available:
        return DurationInfo(duration_available=True, expected_duration=12.4,
                             duration_quantiles={"0.5": 12}, duration_unit="windows")
    return DurationInfo(duration_available=False, expected_duration=None, duration_quantiles=None,
                         duration_unit="windows")


def _model_info():
    return ModelInfo(model_name="semi_hmm", model_version="v1", model_configuration={"dmax": 268},
                      model_source="Results/evaluation/semi_hmm_weekend_phaseF/model/",
                      inference_timestamp="2026-08-24T00:00:00+00:00")


def test_sample_info_to_dict_has_all_fields():
    d = _sample().to_dict()
    assert set(d.keys()) == {"sample_id", "video_name", "patient_id", "split"}


def test_window_info_to_dict_roundtrips_values():
    d = _window().to_dict()
    assert d["window_start"] == 5
    assert d["time_unit"] == "unknown/unverified"


def test_current_state_to_dict():
    d = _current_state().to_dict()
    assert d["current_phase"] == "t9+"
    assert d["phase_probabilities"]["t9+"] == 0.9


def test_duration_info_unavailable_has_none_expected_duration_not_zero():
    d = _duration(available=False).to_dict()
    assert d["duration_available"] is False
    assert d["expected_duration"] is None  # never a fabricated 0.0


def test_duration_info_available_has_numeric_expected_duration():
    d = _duration(available=True).to_dict()
    assert d["duration_available"] is True
    assert d["expected_duration"] == 12.4


def test_inference_record_to_dict_composes_all_sections():
    record = InferenceRecord(
        sample=_sample(), window=_window(), current_state=_current_state(), next_phase=_next_phase(),
        duration=_duration(), model=_model_info(),
        ground_truth=GroundTruthInfo(ground_truth_phase="t9+", consistency_flag=0),
        warnings=["example warning"],
    )
    d = record.to_dict()
    assert set(d.keys()) == {"sample", "window", "current_state", "next_phase", "duration", "model",
                              "ground_truth", "warnings"}
    assert d["warnings"] == ["example warning"]
    assert d["ground_truth"]["consistency_flag"] == 0


def test_inference_record_warnings_default_empty_list():
    record = InferenceRecord(
        sample=_sample(), window=_window(), current_state=_current_state(), next_phase=_next_phase(),
        duration=_duration(), model=_model_info(),
        ground_truth=GroundTruthInfo(ground_truth_phase=None, consistency_flag=None),
    )
    assert record.warnings == []


def test_trajectory_summary_to_dict():
    summary = TrajectorySummary(sample=_sample(), n_windows=3, window_starts=[0, 1, 2],
                                 transition_chain=[{"phase": "t9+", "observed_duration_windows": 3}],
                                 model=_model_info())
    d = summary.to_dict()
    assert d["n_windows"] == 3
    assert d["transition_chain"][0]["phase"] == "t9+"
