"""
Unit tests for the k=7-aligned HMM hyperparameter-selection protocol added
to run_hmm_evaluation.py (docs/PROJECT_STATE.md's "HMM HYPERPARAMETER
SELECTION PROTOCOL" checkpoint). Two things are covered:

  1. select_best_config() — a PURE function (no I/O, no model fitting),
     tested here directly against hand-built metric dicts so the decision
     rule itself is verified without ever training an HMM or touching the
     real embedding cache.
  2. _k7_scores() actually calls HMMModel.predict_k_step(..., k=k), never
     HMMModel.predict() — the one thing this whole pipeline revision
     exists to guarantee (the old, k=1, misaligned event must not leak
     back into the new selection metrics).

All synthetic data, no real cache, no Test split loaded anywhere in this
file (grepped and confirmed as part of this session's own audit — see
docs/PROJECT_STATE.md).
"""

import numpy as np
import pytest
import torch

from evaluation.models.hmm import HMMModel
from experiments.hmm_semi_hmm.run_hmm_evaluation import (
    BRIER_TIE_DECIMALS,
    MIN_K7_AUROC,
    _k7_scores,
    select_best_config,
)
from evaluation.trajectory import Trajectory


def make_trajectory(video_name, embeddings, phases, window_starts=None, first_phases=None):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    phases_t = torch.as_tensor(phases, dtype=torch.long)
    T = embeddings_t.shape[0]
    first_t = torch.as_tensor(first_phases, dtype=torch.long) if first_phases is not None else phases_t.clone()
    consistency = (first_t != phases_t).long()
    return Trajectory(
        video_name=video_name,
        window_starts=window_starts if window_starts is not None else list(range(T)),
        embeddings=embeddings_t,
        consistency_flag=consistency,
        first_frame_phase=first_t,
        last_frame_phase=phases_t,
    )


def make_separable_dataset(n_states=4, n_videos=20, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[10.0 * i, 0.0] for i in range(n_states)])
    trajectories = []
    for v in range(n_videos):
        seq = [0]
        while seq[-1] < n_states - 1:
            step = 1 if rng.rand() < 0.85 else min(2, n_states - 1 - seq[-1])
            seq.append(min(seq[-1] + step, n_states - 1))
        full_seq = []
        for s in seq:
            full_seq.extend([s] * rng.randint(2, 5))
        embeddings = centers[full_seq] + rng.normal(scale=0.5, size=(len(full_seq), 2))
        trajectories.append(make_trajectory(f"v{v}", embeddings, full_seq))
    return trajectories


def _base_row(alpha, rho, auroc_k7, brier_k7, phase_acc):
    return {
        "transition_smoothing_alpha": alpha,
        "transition_decay_rho": rho,
        "val_auroc_k7": auroc_k7,
        "val_brier_k7": brier_k7,
        "val_phase_decoding_accuracy": phase_acc,
    }


# --------------------------------------------------------------------------
# _k7_scores: must use predict_k_step(k=k), never predict()
# --------------------------------------------------------------------------

def test_k7_scores_uses_predict_k_step_not_predict(monkeypatch):
    trajectories = make_separable_dataset(n_states=5, n_videos=10, seed=1)
    model = HMMModel(n_states=5)
    model.fit(trajectories)

    called = {"predict": False, "predict_k_step": False}
    real_predict = HMMModel.predict
    real_predict_k_step = HMMModel.predict_k_step

    def spy_predict(self, trajs):
        called["predict"] = True
        return real_predict(self, trajs)

    def spy_predict_k_step(self, trajs, k=1):
        called["predict_k_step"] = True
        assert k == 3, "expected k7_scores to forward the exact k it was given"
        return real_predict_k_step(self, trajs, k=k)

    monkeypatch.setattr(HMMModel, "predict", spy_predict)
    monkeypatch.setattr(HMMModel, "predict_k_step", spy_predict_k_step)

    _k7_scores(model, trajectories, k=3, n_bins=5)

    assert called["predict_k_step"] is True
    assert called["predict"] is False, (
        "_k7_scores must compute its metrics exclusively from "
        "predict_k_step() -- calling predict() here would silently "
        "reintroduce the k=1 misalignment this pipeline revision exists "
        "to eliminate."
    )


def test_k7_scores_matches_manual_predict_k_step_computation():
    trajectories = make_separable_dataset(n_states=4, n_videos=8, seed=2)
    model = HMMModel(n_states=4)
    model.fit(trajectories)

    result = _k7_scores(model, trajectories, k=2, n_bins=10)

    y_true, y_prob = [], []
    for traj in trajectories:
        y_true.extend(traj.consistency_flag.tolist())
        y_prob.extend(model.predict_k_step_transition_probability(traj, k=2).tolist())
    from sklearn.metrics import brier_score_loss
    # abs=1e-6, not 1e-9: _k7_scores goes through predict_k_step(), whose
    # TrajectoryPrediction.consistency_flag_prob is float32 (torch tensor
    # contract), while this manual reference stays in float64 numpy
    # throughout -- the two differ at float32 precision, not because of a
    # computation bug.
    assert result["val_brier_k7"] == pytest.approx(brier_score_loss(y_true, y_prob), abs=1e-6)
    assert result["val_n_windows_k7"] == len(y_true)


# --------------------------------------------------------------------------
# select_best_config: the pure decision rule
# --------------------------------------------------------------------------

def test_select_best_config_rejects_low_auroc_even_with_better_brier():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=MIN_K7_AUROC - 0.01, brier_k7=0.05, phase_acc=0.5),  # better Brier, ineligible
        _base_row(1.0, 0.3, auroc_k7=MIN_K7_AUROC + 0.01, brier_k7=0.10, phase_acc=0.5),  # worse Brier, eligible
    ]
    result = select_best_config(rows, min_k7_auroc=MIN_K7_AUROC)
    assert result["selection_status"] == "selected"
    assert result["selected"]["transition_smoothing_alpha"] == 1.0
    assert result["selected"]["transition_decay_rho"] == 0.3
    assert result["rows"][0]["eligible"] is False
    assert result["rows"][1]["eligible"] is True


def test_select_best_config_min_brier_wins_among_eligible():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.70, brier_k7=0.15, phase_acc=0.4),
        _base_row(1.0, 0.2, auroc_k7=0.70, brier_k7=0.10, phase_acc=0.4),  # best Brier
        _base_row(2.0, 0.2, auroc_k7=0.70, brier_k7=0.20, phase_acc=0.4),
    ]
    result = select_best_config(rows, min_k7_auroc=0.65)
    assert result["selected"]["transition_smoothing_alpha"] == 1.0
    assert result["n_eligible"] == 3


def test_select_best_config_brier_tie_broken_by_auroc():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.70, brier_k7=0.123444, phase_acc=0.4),
        _base_row(1.0, 0.2, auroc_k7=0.80, brier_k7=0.123401, phase_acc=0.4),  # same at 4 decimals, higher AUROC
    ]
    assert round(rows[0]["val_brier_k7"], BRIER_TIE_DECIMALS) == round(rows[1]["val_brier_k7"], BRIER_TIE_DECIMALS)
    result = select_best_config(rows, min_k7_auroc=0.65)
    assert result["selected"]["transition_smoothing_alpha"] == 1.0
    assert "tied at val_brier_k7" in result["selection_reason"]


def test_select_best_config_double_tie_broken_by_phase_decoding_accuracy():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.75, brier_k7=0.1200, phase_acc=0.40),
        _base_row(1.0, 0.2, auroc_k7=0.75, brier_k7=0.1200, phase_acc=0.55),  # same brier+auroc, higher phase acc
    ]
    result = select_best_config(rows, min_k7_auroc=0.65)
    assert result["selected"]["transition_smoothing_alpha"] == 1.0
    assert "phase_decoding_accuracy" in result["selection_reason"]


def test_select_best_config_all_configurations_preserved_in_output():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.50, brier_k7=0.20, phase_acc=0.4),
        _base_row(1.0, 0.3, auroc_k7=0.70, brier_k7=0.10, phase_acc=0.4),
        _base_row(2.0, 0.5, auroc_k7=0.72, brier_k7=0.11, phase_acc=0.4),
    ]
    result = select_best_config(rows, min_k7_auroc=0.65)
    assert len(result["rows"]) == len(rows)
    configs_in = {(r["transition_smoothing_alpha"], r["transition_decay_rho"]) for r in rows}
    configs_out = {(r["transition_smoothing_alpha"], r["transition_decay_rho"]) for r in result["rows"]}
    assert configs_in == configs_out


def test_select_best_config_no_eligible_configuration_does_not_invent_a_winner():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.40, brier_k7=0.05, phase_acc=0.4),
        _base_row(1.0, 0.3, auroc_k7=0.50, brier_k7=0.06, phase_acc=0.4),
    ]
    result = select_best_config(rows, min_k7_auroc=MIN_K7_AUROC)
    assert result["selected"] is None
    assert result["selection_status"] == "no_eligible_configuration"
    assert result["n_eligible"] == 0


def test_select_best_config_matches_rule_on_a_9_config_grid_by_hand():
    # Mirrors the real 9-config alpha x rho grid shape, with hand-picked
    # values whose winner is unambiguous by direct inspection.
    rows = []
    values = {
        (0.5, 0.2): (0.60, 0.140), (0.5, 0.3): (0.62, 0.100), (0.5, 0.5): (0.70, 0.121),
        (1.0, 0.2): (0.71, 0.119), (1.0, 0.3): (0.72, 0.1185), (1.0, 0.5): (0.69, 0.125),
        (2.0, 0.2): (0.68, 0.128), (2.0, 0.3): (0.73, 0.115),  # <- global min brier AMONG eligible configs
        (2.0, 0.5): (0.74, 0.116),
    }
    # Note: (0.5, 0.3) has the single lowest raw Brier (0.100) of the whole
    # grid, but its AUROC (0.62) is below MIN_K7_AUROC -- it must be
    # excluded despite that, which is exactly what this test checks.
    for (alpha, rho), (auroc, brier) in values.items():
        rows.append(_base_row(alpha, rho, auroc_k7=auroc, brier_k7=brier, phase_acc=0.46))
    result = select_best_config(rows, min_k7_auroc=MIN_K7_AUROC)
    assert result["selected"]["transition_smoothing_alpha"] == 2.0
    assert result["selected"]["transition_decay_rho"] == 0.3
    # Sanity: the two configs below MIN_K7_AUROC=0.6373 must be excluded
    # even though one of them (0.5, 0.2) has a lower raw Brier than several
    # eligible configs.
    ineligible = {(r["transition_smoothing_alpha"], r["transition_decay_rho"]) for r in result["rows"] if not r["eligible"]}
    assert (0.5, 0.2) in ineligible
    assert (0.5, 0.3) in ineligible


def test_select_best_config_is_deterministic_and_does_not_mutate_input():
    rows = [
        _base_row(0.5, 0.2, auroc_k7=0.70, brier_k7=0.12, phase_acc=0.4),
        _base_row(1.0, 0.3, auroc_k7=0.71, brier_k7=0.11, phase_acc=0.4),
    ]
    original_keys = [set(r.keys()) for r in rows]
    result_a = select_best_config(rows, min_k7_auroc=0.65)
    result_b = select_best_config(rows, min_k7_auroc=0.65)
    assert result_a["selected"] == result_b["selected"]
    assert result_a["selection_reason"] == result_b["selection_reason"]
    # Input rows must not have been mutated in place (no 'eligible' key leaked
    # back into the caller's own list).
    for row, keys_before in zip(rows, original_keys):
        assert set(row.keys()) == keys_before


def test_select_best_config_raises_on_empty_rows():
    with pytest.raises(ValueError):
        select_best_config([], min_k7_auroc=MIN_K7_AUROC)


def test_no_test_split_referenced_anywhere_in_run_hmm_evaluation():
    import inspect

    from experiments.hmm_semi_hmm import run_hmm_evaluation as module

    source = inspect.getsource(module)
    # The only two split names this script may ever construct an
    # EmbeddingDataset/_load_split call with are "train" and "val" — no
    # code path here is even capable of loading Test (mirrors the original
    # script's own structural guard, preserved by this revision).
    assert '"test"' not in source
    assert "'test'" not in source
