"""
Flask test-client tests. split='test' rejection is tested on every
endpoint without needing the real model/embeddings (the guard fires in
_split_param() before any filesystem access). Endpoints needing real data
(/model, /videos, /videos/<id>/inference/<window>) are exercised by
test_integration_real_val.py instead, against the real frozen checkpoint.
"""

import pytest

from reporting.api import app
from reporting.trajectory_service import TestSplitLockedError


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


@pytest.mark.parametrize("path", [
    "/videos",
    "/videos/Patient_1",
    "/videos/Patient_1/windows",
    "/videos/Patient_1/inference/0",
    "/videos/Patient_1/trajectory",
])
def test_every_endpoint_rejects_test_split_with_403(client, path):
    resp = client.get(path, query_string={"split": "test"})
    assert resp.status_code == 403
    body = resp.get_json()
    assert body["error"] == "test_locked"


def test_split_defaults_to_val_not_test(monkeypatch, client):
    """No split param at all -> must default to 'val', never 'test'."""
    captured = {}

    def fake_list_videos(split):
        captured["split"] = split
        return []

    monkeypatch.setattr("reporting.api.trajectory_service.list_videos", fake_list_videos)
    resp = client.get("/videos")
    assert resp.status_code == 200
    assert captured["split"] == "val"


def test_unknown_video_returns_404(monkeypatch, client):
    def raise_not_found(video_name, split):
        raise KeyError(f"video_name={video_name!r} not found")

    monkeypatch.setattr("reporting.api.trajectory_service.get_windows", raise_not_found)
    resp = client.get("/videos/DoesNotExist/windows")
    assert resp.status_code == 404


def test_bad_split_value_returns_400(client):
    resp = client.get("/videos", query_string={"split": "not_a_real_split"})
    assert resp.status_code == 400
