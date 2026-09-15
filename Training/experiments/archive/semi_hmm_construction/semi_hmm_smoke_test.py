"""
STEP 2 smoke test -- Semi-HMM on a small REAL Train/Val subset, CPU only,
no Test, no sweep, no scientific comparison to the HMM k=7 result. Writes
its own report to Results/evaluation/semi_hmm_smoke/ (new, additive dir),
never touching any existing Results/evaluation/e1_* directory.
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import flatten_predictions, score_probabilistic
from evaluation.metrics import compute_metrics
from evaluation.models.hmm import DEFAULT_PHASE_NAMES
from evaluation.models.semi_hmm import (
    EmpiricalDurationModel,
    NegativeBinomialDurationModel,
    SemiHMMModel,
    determine_dmax,
    extract_phase_durations,
)
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/semi_hmm_smoke")
OUT_DIR.mkdir(parents=True, exist_ok=True)
report = {}

N_TRAIN = 8
N_VAL = 4

t0 = time.time()
cache_root = Path("../Embeddings/resnet18")
train_full = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
val_full = group_into_trajectories(EmbeddingDataset(cache_root, "val"))
t_load = time.time() - t0

train_traj = train_full[:N_TRAIN]
val_traj = val_full[:N_VAL]

# ---------------------------------------------------------------------------
# Section 2: data
# ---------------------------------------------------------------------------
n_train_windows = sum(len(t) for t in train_traj)
n_val_windows = sum(len(t) for t in val_traj)
Ts_train = [len(t) for t in train_traj]
Ts_val = [len(t) for t in val_traj]
all_Ts = Ts_train + Ts_val
emb_dim = train_traj[0].embeddings.shape[1]

data_report = {
    "n_train_trajectories": len(train_traj), "n_val_trajectories": len(val_traj),
    "n_train_windows": n_train_windows, "n_val_windows": n_val_windows,
    "min_T": int(min(all_Ts)), "mean_T": float(np.mean(all_Ts)), "max_T": int(max(all_Ts)),
    "embedding_dim": int(emb_dim),
    "load_time_seconds": t_load,
}
print("=== DATA ===")
print(json.dumps(data_report, indent=2))

# sanity: temporal order preserved (window_starts strictly increasing per trajectory)
order_ok = all(all(a < b for a, b in zip(t.window_starts[:-1], t.window_starts[1:])) for t in train_traj + val_traj)
phases_valid = all(
    int(t.last_frame_phase.min()) >= 0 and int(t.last_frame_phase.max()) < 15
    for t in train_traj + val_traj if len(t) > 0
)
data_report["window_order_strictly_increasing"] = order_ok
data_report["phases_within_valid_range"] = phases_valid
print("window order OK:", order_ok, "| phases valid:", phases_valid)

report["data"] = data_report

# ---------------------------------------------------------------------------
# Section 5: fit (config A = negative_binomial)
# ---------------------------------------------------------------------------
durations_by_phase = extract_phase_durations(train_traj)
dmax_observed = determine_dmax(durations_by_phase, strategy="observed_max")
dmax = min(dmax_observed, 40)  # smoke-test-only cap for speed, documented, not a scientific choice
print(f"\ndmax (observed_max on this tiny Train subset) = {dmax_observed}, capped to {dmax} for smoke-test speed")

t0 = time.time()
model_a = SemiHMMModel(n_states=15, dmax=dmax, duration_family="negative_binomial")
model_a.fit(train_traj)
t_fit_a = time.time() - t0
print(f"fit (negative_binomial) took {t_fit_a:.2f}s")

t0 = time.time()
model_b = SemiHMMModel(n_states=15, dmax=dmax, duration_family="empirical")
model_b.fit(train_traj)
t_fit_b = time.time() - t0
print(f"fit (empirical) took {t_fit_b:.2f}s")

report["config"] = {"n_states": 15, "dmax": dmax, "dmax_observed_uncapped": dmax_observed,
                     "n_train_used": N_TRAIN, "n_val_used": N_VAL}
report["fit_timing"] = {"negative_binomial_seconds": t_fit_a, "empirical_seconds": t_fit_b}

# ---------------------------------------------------------------------------
# Section 6: immediate checks (transition, duration, emission)
# ---------------------------------------------------------------------------
def check_transition(model, label):
    A = np.exp(model._log_A)
    K = model.n_states
    no_self = all(A[i, i] == 0.0 for i in range(K))
    j_gt_i = all(A[i, j] == 0.0 for i in range(K) for j in range(K) if j <= i)
    finite = np.all(np.isfinite(A))
    row_sums = [A[i, :].sum() for i in range(K - 1)]
    rows_ok = all(abs(s - 1.0) < 1e-4 for s in row_sums)
    print(f"[{label}] transition: no_self={no_self} j>i_only={j_gt_i} finite={finite} "
          f"non_terminal_rows_sum_to_1={rows_ok}")
    return {"no_self_transition": no_self, "j_greater_i_only": j_gt_i, "finite": bool(finite),
            "non_terminal_rows_sum_to_one": rows_ok}


def check_durations(model, label):
    ok = True
    details = []
    for i, name in enumerate(model.state_names):
        dm = model.duration_models[i]
        probs = [dm.probability(d) for d in range(1, dm.dmax + 1)]
        total = sum(probs)
        exp_d = dm.expected_duration()
        finite = np.isfinite(total) and np.isfinite(exp_d) and all(np.isfinite(p) for p in probs)
        nonneg = all(p >= 0.0 for p in probs)
        sum_ok = abs(total - 1.0) < 1e-4
        details.append({"phase": name, "sum": total, "expected_duration": exp_d,
                         "finite": bool(finite), "nonneg": bool(nonneg), "sum_ok": bool(sum_ok)})
        ok = ok and finite and nonneg and sum_ok
    print(f"[{label}] duration models all OK: {ok}")
    return {"all_ok": ok, "details": details}


def check_emission(model, label):
    hmm = model._emission_hmm
    ok = True
    if hmm._scaler is not None:
        ok = ok and np.all(np.isfinite(hmm._scaler.mean_)) and np.all(np.isfinite(hmm._scaler.scale_))
        assert hmm._scaler.mean_.shape[0] == 512, f"expected 512-d embeddings, got {hmm._scaler.mean_.shape[0]}"
    if hmm._logreg is not None:
        ok = ok and np.all(np.isfinite(hmm._logreg.coef_)) and np.all(np.isfinite(hmm._logreg.intercept_))
    print(f"[{label}] emission finite & 512-d: {ok}")
    return {"finite": bool(ok), "embedding_dim_check": 512}


checks = {}
for model, label in [(model_a, "negative_binomial"), (model_b, "empirical")]:
    checks[label] = {
        "transition": check_transition(model, label),
        "duration": check_durations(model, label),
        "emission": check_emission(model, label),
    }
report["immediate_checks"] = checks

# ---------------------------------------------------------------------------
# Section 7: forward/filtering on Val
# ---------------------------------------------------------------------------
model = model_a  # primary model for the remaining sections
t0 = time.time()
filtering_examples = []
row_sum_violations = 0
for traj in val_traj:
    filt = model.filtering(traj)
    row_sums = filt.sum(axis=1)
    row_sum_violations += int(np.sum(np.abs(row_sums - 1.0) > 1e-3))
    for t in [0, len(traj) // 2, len(traj) - 1] if len(traj) > 2 else range(len(traj)):
        argmax_j = int(np.argmax(filt[t]))
        prob = float(filt[t, argmax_j])
        entropy = float(-(np.clip(filt[t], 1e-300, 1.0) * np.log(np.clip(filt[t], 1e-300, 1.0))).sum())
        filtering_examples.append({
            "video": traj.video_name, "t": t, "phase_argmax": model.state_names[argmax_j],
            "probability": prob, "entropy": entropy,
        })
t_filter = time.time() - t0
print(f"\nfiltering() on {len(val_traj)} Val trajectories took {t_filter:.2f}s, row-sum violations: {row_sum_violations}")
for ex in filtering_examples[:8]:
    print(" ", ex)
report["filtering"] = {
    "row_sum_violations": row_sum_violations, "timing_seconds": t_filter,
    "examples": filtering_examples[:12],
}

# ---------------------------------------------------------------------------
# Section 8: next_phase_distribution
# ---------------------------------------------------------------------------
next_phase_examples = []
next_phase_violations = 0
for traj in val_traj[:2]:
    nxt = model.next_phase_distribution(traj)
    A = np.exp(model._log_A)
    row_sums = nxt.sum(axis=1)
    next_phase_violations += int(np.sum(np.abs(row_sums - 1.0) > 1e-3))
    if not np.all(np.isfinite(nxt)):
        next_phase_violations += 1
    for t in [0, len(traj) // 2]:
        top = int(np.argmax(nxt[t]))
        next_phase_examples.append({
            "video": traj.video_name, "t": t,
            "top_next_phase": model.state_names[top], "prob": float(nxt[t, top]),
        })
print(f"\nnext_phase_distribution violations: {next_phase_violations}")
for ex in next_phase_examples:
    print(" ", ex)
report["next_phase_distribution"] = {"violations": next_phase_violations, "examples": next_phase_examples}

# ---------------------------------------------------------------------------
# Section 9: duration comparison (qualitative, this tiny sample only)
# ---------------------------------------------------------------------------
duration_compare = []
for i, name in enumerate(model.state_names):
    durs = [d for d, _ in durations_by_phase.get(i, [])]
    if not durs:
        continue
    dm = model.duration_models[i]
    qs = dm.quantiles([0.1, 0.5, 0.9])
    duration_compare.append({
        "phase": name, "n_observed_this_smoke_sample": len(durs),
        "observed_mean": float(np.mean(durs)), "model_expected_duration": dm.expected_duration(),
        "model_median": qs[0.5], "model_q10": qs[0.1], "model_q90": qs[0.9],
    })
print("\n=== DURATION COMPARISON (tiny sample, qualitative only) ===")
for row in duration_compare:
    print(" ", row)
report["duration_comparison"] = duration_compare

# ---------------------------------------------------------------------------
# Section 10: serialization round-trip
# ---------------------------------------------------------------------------
save_dir = OUT_DIR / "model_negative_binomial"
model.save(save_dir)
reloaded = SemiHMMModel.load(save_dir)
serialization_ok = True
for traj in val_traj:
    before = model.predict([traj])[0].consistency_flag_prob.numpy()
    after = reloaded.predict([traj])[0].consistency_flag_prob.numpy()
    if not np.allclose(before, after, atol=1e-5):
        serialization_ok = False
print(f"\nserialization round-trip (negative_binomial): predictions identical = {serialization_ok}")

save_dir_b = OUT_DIR / "model_empirical"
model_b.save(save_dir_b)
reloaded_b = SemiHMMModel.load(save_dir_b)
serialization_ok_b = True
for traj in val_traj:
    before = model_b.predict([traj])[0].consistency_flag_prob.numpy()
    after = reloaded_b.predict([traj])[0].consistency_flag_prob.numpy()
    if not np.allclose(before, after, atol=1e-5):
        serialization_ok_b = False
print(f"serialization round-trip (empirical): predictions identical = {serialization_ok_b}")
report["serialization"] = {"negative_binomial_ok": serialization_ok, "empirical_ok": serialization_ok_b}

# ---------------------------------------------------------------------------
# Section 11: determinism (two independent fits)
# ---------------------------------------------------------------------------
model_repeat = SemiHMMModel(n_states=15, dmax=dmax, duration_family="negative_binomial")
model_repeat.fit(train_traj)
det_A = bool(np.allclose(np.exp(model._log_A), np.exp(model_repeat._log_A), atol=1e-9))
det_pi = bool(np.allclose(np.exp(model._log_pi), np.exp(model_repeat._log_pi), atol=1e-9))
det_dur = all(
    abs(model.duration_models[i].expected_duration() - model_repeat.duration_models[i].expected_duration()) < 1e-6
    for i in range(15)
)
det_predictions = True
for traj in val_traj:
    p1 = model.predict([traj])[0].consistency_flag_prob.numpy()
    p2 = model_repeat.predict([traj])[0].consistency_flag_prob.numpy()
    if not np.allclose(p1, p2, atol=1e-6):
        det_predictions = False
print(f"\ndeterminism: A={det_A} pi={det_pi} durations={det_dur} predictions={det_predictions}")
report["determinism"] = {"A_identical": det_A, "pi_identical": det_pi,
                          "durations_identical": det_dur, "predictions_identical": det_predictions}

# ---------------------------------------------------------------------------
# Section 12: metrics (smoke-test only, NOT a scientific comparison)
# ---------------------------------------------------------------------------
predictions = model.predict(val_traj)
metrics = compute_metrics(val_traj, predictions)
y_true, y_prob = flatten_predictions(val_traj, predictions)
scored = score_probabilistic(y_true, y_prob, n_bins=5)
print("\n=== SMOKE-TEST METRICS (pipeline check only, NOT scientific) ===")
print(json.dumps({"accuracy": metrics["accuracy"], "auroc": metrics.get("auroc"),
                   "brier": scored["brier"], "ece": scored["ece"], "n_windows": scored["n_windows"]}, indent=2))
report["metrics_smoke_test_only"] = {
    "accuracy": metrics["accuracy"], "auroc": metrics.get("auroc"),
    "brier": scored["brier"], "ece": scored["ece"], "n_windows": scored["n_windows"],
    "disclaimer": "Ces resultats verifient le fonctionnement du pipeline, pas la performance scientifique du modele.",
}

# ---------------------------------------------------------------------------
# Section 13: causality
# ---------------------------------------------------------------------------
base_traj = val_traj[0]
prefix_len = min(5, len(base_traj) - 1) if len(base_traj) > 1 else len(base_traj)
causality_ok = True
if prefix_len >= 1 and len(base_traj) > prefix_len:
    from evaluation.trajectory import Trajectory
    modified_embeddings = base_traj.embeddings.clone()
    modified_last_phase = base_traj.last_frame_phase.clone()
    modified_first_phase = base_traj.first_frame_phase.clone()
    modified_flag = base_traj.consistency_flag.clone()
    # corrupt the suffix (after prefix_len) with random content
    rng = torch.Generator().manual_seed(0)
    modified_embeddings[prefix_len:] = torch.randn(
        modified_embeddings[prefix_len:].shape, generator=rng
    ) * 50.0
    modified_last_phase[prefix_len:] = 14  # force to last phase, arbitrary corruption
    modified_traj = Trajectory(
        video_name=base_traj.video_name, window_starts=base_traj.window_starts,
        embeddings=modified_embeddings, consistency_flag=modified_flag,
        first_frame_phase=modified_first_phase, last_frame_phase=modified_last_phase,
    )
    filt_orig = model.filtering(base_traj)
    filt_mod = model.filtering(modified_traj)
    causality_ok = bool(np.allclose(filt_orig[:prefix_len], filt_mod[:prefix_len], atol=1e-9))
print(f"\ncausality check (prefix unaffected by corrupted suffix): {causality_ok}")
report["causality"] = {"ok": causality_ok, "prefix_len": prefix_len}

# ---------------------------------------------------------------------------
# Section 14: performance
# ---------------------------------------------------------------------------
report["performance"] = {
    "load_time_seconds": t_load,
    "fit_negative_binomial_seconds": t_fit_a,
    "fit_empirical_seconds": t_fit_b,
    "filtering_seconds_total": t_filter,
    "filtering_seconds_per_window_approx": t_filter / max(n_val_windows, 1),
}
print("\n=== PERFORMANCE ===")
print(json.dumps(report["performance"], indent=2))

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
(OUT_DIR / "smoke_test_report.json").write_text(json.dumps(report, indent=2, default=str))
print("\nReport written to", OUT_DIR / "smoke_test_report.json")
print("\nALL_SECTIONS_COMPLETE")
