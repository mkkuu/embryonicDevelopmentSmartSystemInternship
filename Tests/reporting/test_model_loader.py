"""
Synthetic-data tests for model_loader's validation logic (n_states=15,
dmax=268, duration_family=negative_binomial, class_weight=None,
embedding_dim=512). Deliberately uses small, fast synthetic models to
trigger EACH mismatch independently -- the real frozen checkpoint is only
exercised by the integration test (test_integration_real_val.py), which
needs the actual server-side artifacts.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import Trajectory
from reporting import model_loader


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """model_loader._MODEL_CACHE is a module-level global -- without this,
    a real model loaded by a DIFFERENT test file (e.g.
    test_integration_real_val.py, which runs alphabetically before this
    file and legitimately populates the cache) leaks into these tests,
    silently defeating get_model()'s own file-existence check. Caught the
    hard way: test_get_model_raises_file_not_found_for_missing_checkpoint
    failed exactly this way in a real pytest run on the server before this
    fixture was added."""
    model_loader._MODEL_CACHE = None
    yield
    model_loader._MODEL_CACHE = None


def make_trajectory(video_name, embeddings, phases):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    phases_t = torch.as_tensor(phases, dtype=torch.long)
    return Trajectory(video_name=video_name, window_starts=list(range(len(phases))),
                       embeddings=embeddings_t, consistency_flag=(phases_t != phases_t).long(),
                       first_frame_phase=phases_t.clone(), last_frame_phase=phases_t)


def _fit_tiny_model(n_states, dmax, duration_family, embedding_dim, class_weight=None):
    rng = np.random.RandomState(0)
    trajectories = []
    for v in range(6):
        seq = list(range(n_states)) if n_states > 1 else [0, 0, 0]
        reps = []
        for s in seq:
            reps.extend([s] * rng.randint(2, 4))
        embeddings = rng.normal(size=(len(reps), embedding_dim)) + np.array(reps)[:, None]
        trajectories.append(make_trajectory(f"v{v}", embeddings, reps))
    model = SemiHMMModel(n_states=n_states, dmax=dmax, duration_family=duration_family,
                          logreg_class_weight=class_weight)
    model.fit(trajectories)
    return model


def test_get_model_raises_file_not_found_for_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_loader.get_model(path=tmp_path / "does_not_exist")


def test_validate_rejects_wrong_n_states(tmp_path):
    model = _fit_tiny_model(n_states=4, dmax=20, duration_family="negative_binomial", embedding_dim=8)
    with pytest.raises(model_loader.ModelIncompatibleError, match="n_states"):
        model_loader._validate(model)


def test_validate_rejects_wrong_dmax(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=20, duration_family="negative_binomial", embedding_dim=8)
    with pytest.raises(model_loader.ModelIncompatibleError, match="dmax"):
        model_loader._validate(model)


def test_validate_rejects_wrong_duration_family(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="empirical", embedding_dim=8)
    with pytest.raises(model_loader.ModelIncompatibleError, match="duration_family"):
        model_loader._validate(model)


def test_validate_rejects_wrong_class_weight(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=8,
                             class_weight="balanced")
    with pytest.raises(model_loader.ModelIncompatibleError, match="class_weight"):
        model_loader._validate(model)


def test_validate_rejects_wrong_embedding_dim(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=8)
    with pytest.raises(model_loader.ModelIncompatibleError, match="embedding_dim"):
        model_loader._validate(model)


def test_validate_accepts_matching_configuration_shape(tmp_path):
    # n_states/dmax/duration_family/class_weight all correct; embedding_dim
    # deliberately still wrong (512-dim synthetic data would be slow/pointless)
    # -- isolates that ONLY the embedding_dim check fires, not a false
    # positive on the others, confirming _validate reports every real
    # mismatch independently rather than short-circuiting on the first one.
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=8)
    with pytest.raises(model_loader.ModelIncompatibleError) as exc_info:
        model_loader._validate(model)
    msg = str(exc_info.value)
    assert "n_states" not in msg
    assert "dmax" not in msg
    assert "duration_family" not in msg
    assert "class_weight" not in msg
    assert "embedding_dim" in msg


def test_get_model_caches_after_first_load(tmp_path, monkeypatch):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=512)
    save_path = tmp_path / "model"
    model.save(save_path)
    monkeypatch.setattr(model_loader, "_validate", lambda m: None)  # bypass real-reference checks for this cache test
    first = model_loader.get_model(path=save_path, force_reload=True)
    second = model_loader.get_model(path=save_path)
    assert first is second  # same object, not reloaded


def test_model_configuration_contains_expected_keys(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=8)
    cfg = model_loader.model_configuration(model)
    assert cfg["dmax"] == 268
    assert cfg["duration_family"] == "negative_binomial"
    assert cfg["n_states"] == 15


def test_model_version_string_is_stable_and_readable(tmp_path):
    model = _fit_tiny_model(n_states=15, dmax=268, duration_family="negative_binomial", embedding_dim=8)
    version = model_loader.model_version_string(model)
    assert "dmax=268" in version
    assert "negative_binomial" in version
