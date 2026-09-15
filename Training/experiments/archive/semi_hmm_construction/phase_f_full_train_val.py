"""
Phase F -- Semi-HMM FULL real Train/Val evaluation (the first real
comparison against the frozen HMM k=7 benchmark). One-off script (not
committed, not part of the production package -- same convention as
Phase A-E's /tmp scripts).

Protocol, fixed by explicit user decision (2026-08-23, after Phase E):
  - dmax = 268 (observed_max over full real Train, Phase E diagnostic;
    zero truncation, no negative-binomial tail-extrapolation edge case).
  - duration_family = "negative_binomial" (empirical fallback already
    forced internally for any state with zero observed durations --
    tPB2/tEB -- regardless of this setting, see semi_hmm.py Phase A2).
  - n_states = 15, same embeddings/labels/emission as the classic HMM.
  - Train = 492 trajectories (196,934 windows), Val = 106 trajectories
    (43,339 windows). Test split NEVER loaded, never referenced anywhere
    in this file.

Evaluation methodology: predict_k_step(val, k=7) -- NOT predict()'s raw
k=1 -- because consistency_flag is an intra-window event (first vs last
frame of the SAME 8-frame window) and k=window_size-1=7 is the quantity
that actually matches it (see semi_hmm.py's predict_k_step_transition_probability
docstring; this is the exact same alignment fix already diagnosed and
applied for the classic HMM on 2026-08-21 -- reusing it here, not
inventing a new methodology). k=1 is also reported for transparency/
diagnostic purposes but is NOT the headline comparison number.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import flatten_predictions, score_probabilistic
from evaluation.metrics import compute_metrics
from evaluation.models.hmm import DEFAULT_PHASE_NAMES
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

DMAX = 268
DURATION_FAMILY = "negative_binomial"
N_STATES = 15
K_HEADLINE = 7  # window_size - 1, matches consistency_flag's own definition

OUT_DIR = Path("../Results/evaluation/semi_hmm_weekend_phaseF")
OUT_DIR.mkdir(parents=True, exist_ok=True)
report = {
    "protocol": {
        "dmax": DMAX, "duration_family": DURATION_FAMILY, "n_states": N_STATES,
        "k_headline": K_HEADLINE,
        "dmax_decision": "observed_max over full real Train (Phase E diagnostic), "
                          "explicit user decision 2026-08-23, zero truncation",
    }
}

t0 = time.time()
train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_load = time.time() - t0
n_train_windows = sum(len(t) for t in train_traj)
n_val_windows = sum(len(t) for t in val_traj)
print(f"load: {t_load:.2f}s, n_train={len(train_traj)} ({n_train_windows} windows), "
      f"n_val={len(val_traj)} ({n_val_windows} windows)")
report["load_time_seconds"] = t_load
report["n_train_trajectories"] = len(train_traj)
report["n_train_windows"] = n_train_windows
report["n_val_trajectories"] = len(val_traj)
report["n_val_windows"] = n_val_windows

# ---------------------------------------------------------------------------
# 1. Fit on FULL real Train
# ---------------------------------------------------------------------------
t0 = time.time()
model = SemiHMMModel(n_states=N_STATES, dmax=DMAX, duration_family=DURATION_FAMILY)
model.fit(train_traj)
t_fit = time.time() - t0
print(f"fit on FULL Train (492 trajectories, dmax={DMAX}): {t_fit:.2f}s")
report["fit_time_seconds"] = t_fit

# ---------------------------------------------------------------------------
# 2. Predict on FULL real Val -- headline k=7, plus k=1 for transparency
# ---------------------------------------------------------------------------
t0 = time.time()
preds_k7 = model.predict_k_step(val_traj, k=K_HEADLINE)
t_predict_k7 = time.time() - t0
print(f"predict_k_step(Val, k={K_HEADLINE}): {t_predict_k7:.2f}s "
      f"({t_predict_k7 / n_val_windows:.5f}s/window)")
report["predict_k7_time_seconds"] = t_predict_k7

t0 = time.time()
preds_k1 = model.predict(val_traj)  # equivalently predict_k_step(val, k=1)
t_predict_k1 = time.time() - t0
print(f"predict(Val) [k=1]: {t_predict_k1:.2f}s")
report["predict_k1_time_seconds"] = t_predict_k1

# ---------------------------------------------------------------------------
# 3. Metrics -- k=7 is the HEADLINE, directly comparable to the frozen HMM
#    k=7 benchmark (alpha=2.0/rho=0.5, e1_hmm_k7_sweep): AUROC=0.63735,
#    Brier=0.14072, ECE=0.12312, baseline Brier=0.11483 (read-only
#    reference values, hardcoded here for the comparison table -- NOT
#    recomputed, that artifact is untouched).
# ---------------------------------------------------------------------------
HMM_K7_REFERENCE = {
    "auroc": 0.63735, "brier": 0.14072, "ece": 0.12312,
    "brier_baseline_constant_rate": 0.11483,
    "source": "Results/evaluation/e1_hmm_k7_sweep (alpha=2.0/rho=0.5), read-only reference, not recomputed",
}
report["hmm_k7_reference"] = HMM_K7_REFERENCE

y_true_k7, y_prob_k7 = flatten_predictions(val_traj, preds_k7)
score_k7 = score_probabilistic(y_true_k7, y_prob_k7, n_bins=10)
classif_k7 = compute_metrics(val_traj, preds_k7)

y_true_k1, y_prob_k1 = flatten_predictions(val_traj, preds_k1)
score_k1 = score_probabilistic(y_true_k1, y_prob_k1, n_bins=10)
classif_k1 = compute_metrics(val_traj, preds_k1)

print()
print(f"=== Semi-HMM k={K_HEADLINE} (HEADLINE, comparable to HMM k=7) ===")
print(f"  AUROC={score_k7['auroc']:.5f}  Brier={score_k7['brier']:.5f}  "
      f"ECE={score_k7['ece']:.5f}  baseline_brier={score_k7['brier_baseline_constant_rate']:.5f}")
print(f"  accuracy={classif_k7['accuracy']:.4f} precision={classif_k7['precision']:.4f} "
      f"recall={classif_k7['recall']:.4f} f1={classif_k7['f1']:.4f}")
print()
print(f"=== Semi-HMM k=1 (diagnostic only, NOT the headline comparison) ===")
print(f"  AUROC={score_k1['auroc']:.5f}  Brier={score_k1['brier']:.5f}  "
      f"ECE={score_k1['ece']:.5f}  baseline_brier={score_k1['brier_baseline_constant_rate']:.5f}")

report["semi_hmm_k7"] = {**score_k7, "classification": classif_k7}
report["semi_hmm_k1"] = {**score_k1, "classification": classif_k1}

print()
print("=== Comparison vs frozen HMM k=7 (Val, read-only reference) ===")
for metric in ("auroc", "brier", "ece"):
    hmm_val = HMM_K7_REFERENCE[metric]
    semi_val = score_k7[metric]
    delta = semi_val - hmm_val
    better = (delta > 0) if metric == "auroc" else (delta < 0)
    print(f"  {metric:6s} HMM={hmm_val:.5f}  Semi-HMM={semi_val:.5f}  "
          f"delta={delta:+.5f}  {'Semi-HMM better' if better else 'HMM better'}")

# ---------------------------------------------------------------------------
# 4. Per-phase breakdown (k=7), and predicted vs observed duration
# ---------------------------------------------------------------------------
phase_names = list(DEFAULT_PHASE_NAMES)
per_phase_true, per_phase_prob, per_phase_name = [], [], []
for traj, pred in zip(val_traj, preds_k7):
    lf = traj.last_frame_phase.numpy()
    yt = traj.consistency_flag.numpy()
    yp = pred.consistency_flag_prob.numpy()
    per_phase_true.extend(yt.tolist())
    per_phase_prob.extend(yp.tolist())
    per_phase_name.extend([phase_names[i] for i in lf.tolist()])
per_phase_true = np.array(per_phase_true)
per_phase_prob = np.array(per_phase_prob)
per_phase_name = np.array(per_phase_name)

print()
print("=== Per-phase (k=7): n, observed, predicted, brier, expected_duration(model) ===")
per_phase_report = {}
for i, name in enumerate(phase_names):
    mask = per_phase_name == name
    n = int(mask.sum())
    if n == 0:
        continue
    obs = float(per_phase_true[mask].mean())
    pred_mean = float(per_phase_prob[mask].mean())
    brier = float(np.mean((per_phase_prob[mask] - per_phase_true[mask]) ** 2))
    expected_dur = model.expected_duration(i)
    dm = model.duration_models[i]
    row = {
        "n": n, "observed": obs, "predicted": pred_mean, "brier": brier,
        "model_expected_duration_windows": expected_dur,
        "duration_model_type": type(dm).__name__,
    }
    per_phase_report[name] = row
    print(f"  {name:6s} n={n:5d} obs={obs:.4f} pred={pred_mean:.4f} brier={brier:.4f} "
          f"E[D]_model={expected_dur:7.2f}  family={row['duration_model_type']}")
report["per_phase_k7"] = per_phase_report

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
(OUT_DIR / "phaseF_report.json").write_text(json.dumps(report, indent=2))
model.save(OUT_DIR / "model")
print()
print("Report written to", OUT_DIR / "phaseF_report.json")
print("Model saved to", OUT_DIR / "model")
print("PHASE_F_DONE")
