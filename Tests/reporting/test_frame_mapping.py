"""Synthetic tests for frame_mapping.py's reconstruction logic -- no real
Data/ files needed, monkeypatches _real_frame_numbers_with_phase directly
with hand-built (frame_number, phase) sequences. Real-data cross-validation
against the live Reporting API lives in test_frame_mapping_real_val.py
(skip-guarded, GPU server only)."""

import pytest

from reporting import frame_mapping, trajectory_service


@pytest.fixture(autouse=True)
def _reset_caches():
    frame_mapping._KEPT_SEQUENCES_CACHE.clear()
    yield
    frame_mapping._KEPT_SEQUENCES_CACHE.clear()


def _wire(monkeypatch, video_name, frames_with_phase, video_windows):
    monkeypatch.setattr(frame_mapping, "_real_frame_numbers_with_phase", lambda v: frames_with_phase)
    monkeypatch.setattr(trajectory_service, "get_windows", lambda v, s: video_windows)


def test_split_test_is_rejected(monkeypatch):
    with pytest.raises(trajectory_service.TestSplitLockedError):
        frame_mapping.window_frame_numbers("Patient_1", "test", 0)
    with pytest.raises(trajectory_service.TestSplitLockedError):
        frame_mapping.window_to_frame_path("Patient_1", "test", 0)


def test_reconstructs_a_single_kept_window_for_uniform_phase(monkeypatch):
    frames = [(i, "t2") for i in range(100, 108)]  # exactly 8 frames, all t2 -> exactly 1 window
    _wire(monkeypatch, "Patient_1", frames, [0])
    result = frame_mapping.window_frame_numbers("Patient_1", "val", 0)
    assert result == list(range(100, 108))


def test_window_spanning_two_adjacent_phases_is_kept(monkeypatch):
    # t2 (index 3) and t3 (index 4) are adjacent in CHRONOLOGICAL_PHASES.
    frames = [(100 + i, "t2" if i < 4 else "t3") for i in range(8)]
    _wire(monkeypatch, "Patient_1", frames, [0])
    result = frame_mapping.window_frame_numbers("Patient_1", "val", 0)
    assert result == list(range(100, 108))


def test_window_spanning_two_non_adjacent_phases_is_dropped(monkeypatch):
    # t2 (index 3) and t5 (index 6) are NOT adjacent -- this window must be filtered out,
    # exactly like Embryo_Transition_Dataset._create_sequences() would.
    frames = [(100 + i, "t2" if i < 4 else "t5") for i in range(8)]
    _wire(monkeypatch, "Patient_1", frames, [])  # no window kept -> no real window_start to query
    kept = frame_mapping._reconstruct_kept_sequences("Patient_1")
    assert kept == []


def test_window_spanning_three_phases_is_dropped(monkeypatch):
    frames = [(100, "t2"), (101, "t2"), (102, "t3"), (103, "t3"),
              (104, "t4"), (105, "t4"), (106, "t4"), (107, "t4")]
    _wire(monkeypatch, "Patient_1", frames, [])
    kept = frame_mapping._reconstruct_kept_sequences("Patient_1")
    assert kept == []


def test_local_offset_is_relative_to_this_videos_first_window_start(monkeypatch):
    # Mirrors the real, empirically-confirmed convention: window_start is a
    # GLOBAL index across the whole split (Patient_367 -> 527-962 in
    # practice), never assumed to be 0-based per video.
    frames = [(100 + i, "t2") for i in range(16)]  # 9 kept windows (16-8+1)
    _wire(monkeypatch, "Patient_1", frames, list(range(500, 509)))  # global window_starts 500..508
    result = frame_mapping.window_frame_numbers("Patient_1", "val", 500)
    assert result == list(range(100, 108))
    result_last = frame_mapping.window_frame_numbers("Patient_1", "val", 508)
    assert result_last == list(range(108, 116))


def test_window_start_not_belonging_to_this_video_raises(monkeypatch):
    frames = [(i, "t2") for i in range(100, 108)]
    _wire(monkeypatch, "Patient_1", frames, [0])
    with pytest.raises(frame_mapping.FrameMappingError):
        frame_mapping.window_frame_numbers("Patient_1", "val", 999)


def test_reconstruction_disagreeing_with_real_trajectory_length_raises(monkeypatch):
    # The real trajectory claims a window exists (window_start=5) but the
    # reconstruction only produced 1 kept sequence -- must raise, never
    # silently index out of range or guess.
    frames = [(i, "t2") for i in range(100, 108)]
    _wire(monkeypatch, "Patient_1", frames, [0, 1, 2, 3, 4, 5])
    with pytest.raises(frame_mapping.FrameMappingError):
        frame_mapping.window_frame_numbers("Patient_1", "val", 5)


def test_window_to_frame_path_last_vs_first(monkeypatch, tmp_path):
    frames = [(100 + i, "t2") for i in range(8)]
    _wire(monkeypatch, "Patient_1", frames, [0])
    monkeypatch.setattr(frame_mapping, "IMAGES_ROOT", tmp_path)
    video_dir = tmp_path / "Patient_1"
    video_dir.mkdir()
    (video_dir / "Patient_1_Image_100.jpeg").write_bytes(b"fake")
    (video_dir / "Patient_1_Image_107.jpeg").write_bytes(b"fake")

    last_path = frame_mapping.window_to_frame_path("Patient_1", "val", 0, which="last")
    assert last_path.name == "Patient_1_Image_107.jpeg"
    first_path = frame_mapping.window_to_frame_path("Patient_1", "val", 0, which="first")
    assert first_path.name == "Patient_1_Image_100.jpeg"


def test_window_to_frame_path_rejects_invalid_which(monkeypatch):
    frames = [(100 + i, "t2") for i in range(8)]
    _wire(monkeypatch, "Patient_1", frames, [0])
    with pytest.raises(ValueError):
        frame_mapping.window_to_frame_path("Patient_1", "val", 0, which="middle")


def test_window_to_frame_path_raises_when_file_missing_on_disk(monkeypatch, tmp_path):
    frames = [(100 + i, "t2") for i in range(8)]
    _wire(monkeypatch, "Patient_1", frames, [0])
    monkeypatch.setattr(frame_mapping, "IMAGES_ROOT", tmp_path)  # empty dir -- no JPEGs at all
    with pytest.raises(frame_mapping.FrameFileNotFoundError):
        frame_mapping.window_to_frame_path("Patient_1", "val", 0, which="last")


def test_missing_annotations_file_raises_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mapping, "ANNOTATIONS_DIR", tmp_path)  # empty -- no *_phases.csv here
    with pytest.raises(frame_mapping.AnnotationsNotFoundError):
        frame_mapping._real_frame_numbers_with_phase("DoesNotExist")


def test_kept_sequences_are_cached_per_video(monkeypatch):
    calls = {"n": 0}

    def counting(video_name):
        calls["n"] += 1
        return [(100 + i, "t2") for i in range(8)]

    monkeypatch.setattr(frame_mapping, "_real_frame_numbers_with_phase", counting)
    frame_mapping._reconstruct_kept_sequences("Patient_1")
    frame_mapping._reconstruct_kept_sequences("Patient_1")
    assert calls["n"] == 1
