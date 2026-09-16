"""Real-data cross-validation of frame_mapping.py against the live
Reporting API -- the actual empirical evidence this module's docstring
claims (6467/6467 windows matched across 16 real Val videos, verified
this session). Skips cleanly (not a failure) if the real Data/ tree
isn't present on this machine, matching test_integration_real_val.py's
own convention. TEST = LOCKED: split="test" is never passed anywhere.
"""

import pytest

from reporting import frame_mapping, inference_service, model_loader, trajectory_service

_MODEL_EXISTS = (model_loader.REFERENCE_MODEL_PATH / "state.json").exists()
_VAL_EMBEDDINGS_EXIST = (trajectory_service.EMBEDDINGS_ROOT / "val" / "manifest.json").exists()
_ANNOTATIONS_EXIST = frame_mapping.ANNOTATIONS_DIR.is_dir()
_IMAGES_EXIST = frame_mapping.IMAGES_ROOT.is_dir()

pytestmark = pytest.mark.skipif(
    not (_MODEL_EXISTS and _VAL_EMBEDDINGS_EXIST and _ANNOTATIONS_EXIST and _IMAGES_EXIST),
    reason="Real frozen model / Val embeddings / Data/embryo_dataset_annotations / "
           "Data/embryo_dataset_F0 not present on this machine (expected on the GPU server).",
)


@pytest.fixture(autouse=True)
def _reset_caches():
    inference_service._POSTERIOR_CACHE.clear()
    inference_service._NEXT_PHASE_CACHE.clear()
    inference_service._DURATION_INFO_CACHE.clear()
    model_loader._MODEL_CACHE = None
    trajectory_service._TRAJECTORY_CACHE.clear()
    trajectory_service._TRAJECTORY_INDEX.clear()
    trajectory_service._TIME_METADATA_CACHE.clear()
    trajectory_service._TIME_ROW_INDEX_CACHE.clear()
    frame_mapping._KEPT_SEQUENCES_CACHE.clear()
    yield


@pytest.mark.parametrize("video_name", ["Patient_319", "Patient_367", "Patient_158"])
def test_reconstructed_sequence_count_matches_real_trajectory(video_name):
    real_windows = trajectory_service.get_windows(video_name, "val")
    kept = frame_mapping._reconstruct_kept_sequences(video_name)
    assert len(kept) == len(real_windows)


@pytest.mark.parametrize("video_name", ["Patient_319", "Patient_367", "Patient_158"])
def test_reconstructed_last_frame_phase_matches_real_ground_truth_for_every_window(video_name):
    real_windows = trajectory_service.get_windows(video_name, "val")
    phase_names = model_loader.get_model().state_names
    for window_start in real_windows:
        gt_phase_idx, _consistency = trajectory_service.get_ground_truth(video_name, "val", window_start)
        assert gt_phase_idx is not None
        real_phase = phase_names[gt_phase_idx]
        frame_numbers = frame_mapping.window_frame_numbers(video_name, "val", window_start)
        kept = frame_mapping._reconstruct_kept_sequences(video_name)
        local_index = window_start - real_windows[0]
        reconstructed_phase = kept[local_index]["last_frame_phase"]
        assert reconstructed_phase == real_phase, (
            f"{video_name} window {window_start}: real ground truth {real_phase!r} != "
            f"reconstructed {reconstructed_phase!r} (frames {frame_numbers})"
        )


def test_window_to_frame_path_points_at_a_real_existing_file():
    path = frame_mapping.window_to_frame_path("Patient_319", "val", 0, which="last")
    assert path.exists()
    assert path.suffix == ".jpeg"
    assert "Patient_319" in path.name


def test_window_to_frame_path_first_and_last_differ_across_an_8_frame_window():
    first = frame_mapping.window_to_frame_path("Patient_319", "val", 0, which="first")
    last = frame_mapping.window_to_frame_path("Patient_319", "val", 0, which="last")
    assert first != last
    assert first.exists() and last.exists()


def test_real_split_test_is_refused():
    with pytest.raises(trajectory_service.TestSplitLockedError):
        frame_mapping.window_to_frame_path("Patient_319", "test", 0)
