import sys
sys.path.insert(0, ".")
import numpy as np
from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import score_probabilistic, flatten_predictions
from evaluation.models.hmm import HMMModel
from evaluation.trajectory import group_into_trajectories

val = group_into_trajectories(EmbeddingDataset("../Embeddings/resnet18", "val"))
from pathlib import Path
model = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
preds = model.predict_k_step(val, k=7)
y_true, y_prob = flatten_predictions(val, preds)
scored = score_probabilistic(y_true, y_prob, n_bins=10)
print("AUROC:", scored["auroc"], "Brier:", scored["brier"], "baseline:", scored["brier_baseline_constant_rate"], "ECE:", scored["ece"])
print("N windows:", scored["n_windows"], "observed positive rate:", scored["observed_positive_rate"])
print()
print("Calibration bins (best model, alpha=2.0/rho=0.5, k=7):")
for b in scored["calibration_bins"]:
    lo = b["bin_lo"]
    hi = b["bin_hi"]
    n = b["n"]
    mp = b["mean_predicted"]
    obs = b["observed_frequency"]
    print("  [{:.1f},{:.1f}) n={:6d} mean_pred={} observed={}".format(lo, hi, n, mp, obs))

phase_names = model.state_names
per_phase_true, per_phase_prob, per_phase_name = [], [], []
for traj, pred in zip(val, preds):
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
print("Per-phase (S_t = last_frame_phase) breakdown:")
for name in phase_names:
    mask = per_phase_name == name
    n = int(mask.sum())
    if n == 0:
        continue
    obs = per_phase_true[mask].mean()
    pred_mean = per_phase_prob[mask].mean()
    brier = float(np.mean((per_phase_prob[mask] - per_phase_true[mask]) ** 2))
    print("  {:6s} n={:6d} observed={:.4f} mean_pred={:.4f} brier={:.4f}".format(name, n, obs, pred_mean, brier))
