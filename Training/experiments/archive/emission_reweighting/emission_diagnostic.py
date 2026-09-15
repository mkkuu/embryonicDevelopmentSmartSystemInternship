"""
Option B -- targeted emission/calibration diagnostic for the worst-
calibrated Semi-HMM phases (tPNf, t3, t5, t6, tB) vs. reference phases
(t9+, t8). One-off, READ-ONLY script (not committed, same convention as
prior phase scripts). NO training, NO new model, NO code change to any
scientific module. Only the two already-frozen models are loaded:
HMM (Results/evaluation/e1_hmm_k7_sweep/best_model) and Semi-HMM
(Results/evaluation/semi_hmm_weekend_phaseF/model). Test split is NEVER
loaded, never referenced. Reuses the prior next-step analysis's
per-phase table (analysis.json) for calibration means rather than
re-running the expensive k=7 predict pass a second time.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from scipy.special import logsumexp

from embeddings.dataset import EmbeddingDataset
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/semi_hmm_next_step_analysis/emission_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE_NAMES = list(DEFAULT_PHASE_NAMES)
TARGET_PHASES = ["tPNf", "t3", "t5", "t6", "tB"]
REFERENCE_PHASES = ["t9+", "t8"]
FOCUS_PHASES = TARGET_PHASES + REFERENCE_PHASES
K7 = 7

report = {"protocol": {"target_phases": TARGET_PHASES, "reference_phases": REFERENCE_PHASES,
                        "note": "read-only, no training, no Test, frozen models only"}}

t0 = time.time()
train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
hmm = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
semi = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")
report["load_time_seconds"] = t_load

prior_analysis = json.loads(
    Path("../Results/evaluation/semi_hmm_next_step_analysis/analysis.json").read_text()
)
phase_e = json.loads(
    Path("../Results/evaluation/semi_hmm_weekend_phaseE/dmax_duration_diagnostic_report.json").read_text()
)

# ---------------------------------------------------------------------------
# 0. Emission identity check: is Semi-HMM's composed _emission_hmm's
#    scaler/logreg numerically the same as the standalone frozen HMM's?
# ---------------------------------------------------------------------------
same_scaler_mean = np.allclose(hmm._scaler.mean_, semi._emission_hmm._scaler.mean_)
same_logreg_coef = np.allclose(hmm._logreg.coef_, semi._emission_hmm._logreg.coef_)
emission_identity = {"same_scaler_mean": bool(same_scaler_mean), "same_logreg_coef": bool(same_logreg_coef)}
print(f"emission identity check: same_scaler_mean={same_scaler_mean} same_logreg_coef={same_logreg_coef}")
report["emission_identity_check"] = emission_identity

# ---------------------------------------------------------------------------
# 1. Embedding separability on TRAIN (all 15 phases computed, focus phases highlighted)
# ---------------------------------------------------------------------------
all_emb = torch.cat([t.embeddings for t in train_traj if len(t) > 0], dim=0).numpy()
all_lbl = torch.cat([t.last_frame_phase for t in train_traj if len(t) > 0], dim=0).numpy()
dim = all_emb.shape[1]
print(f"Train embeddings: {all_emb.shape[0]} windows, dim={dim}")

centroids = {}
for i, name in enumerate(PHASE_NAMES):
    mask = all_lbl == i
    if mask.sum() == 0:
        continue
    centroids[name] = all_emb[mask].mean(axis=0)

separability = {}
for i, name in enumerate(PHASE_NAMES):
    mask = all_lbl == i
    n = int(mask.sum())
    if n == 0 or name not in centroids:
        continue
    pts = all_emb[mask]
    c = centroids[name]
    intra = float(np.mean(np.linalg.norm(pts - c[None, :], axis=1)))
    other_dists = {other: float(np.linalg.norm(c - oc)) for other, oc in centroids.items() if other != name}
    nearest_name = min(other_dists, key=other_dists.get)
    nearest_dist = other_dists[nearest_name]
    mean_inter = float(np.mean(list(other_dists.values())))
    videos = set(t.video_name for t in train_traj if len(t) > 0 and (t.last_frame_phase.numpy() == i).any())
    n_segments = phase_e["train_per_phase"].get(name, {}).get("n_segments_total")
    separability[name] = {
        "n_windows": n, "n_videos": len(videos), "n_segments": n_segments, "dim": dim,
        "intra_class_dispersion": intra,
        "inter_class_nearest_phase": nearest_name, "inter_class_nearest_distance": nearest_dist,
        "inter_class_mean_distance": mean_inter,
        "ratio_nearest_inter_over_intra": nearest_dist / intra if intra > 0 else float("nan"),
        "ratio_mean_inter_over_intra": mean_inter / intra if intra > 0 else float("nan"),
    }

print()
print("=== Embedding separability (Train), focus phases ===")
for name in FOCUS_PHASES:
    s = separability.get(name)
    if s is None:
        continue
    print(f"  {name:6s} n={s['n_windows']:5d} n_vid={s['n_videos']:3d} intra={s['intra_class_dispersion']:.3f} "
          f"nearest={s['inter_class_nearest_phase']:6s}@{s['inter_class_nearest_distance']:.3f} "
          f"mean_inter={s['inter_class_mean_distance']:.3f} "
          f"ratio_nearest={s['ratio_nearest_inter_over_intra']:.3f} ratio_mean={s['ratio_mean_inter_over_intra']:.3f}")
report["embedding_separability_train"] = separability

# ---------------------------------------------------------------------------
# 2. Emission-only phase confusion on VAL (no temporal information at all --
#    raw discriminative posterior of the underlying LogisticRegression)
# ---------------------------------------------------------------------------
val_emb = torch.cat([t.embeddings for t in val_traj if len(t) > 0], dim=0).numpy()
val_lbl = torch.cat([t.last_frame_phase for t in val_traj if len(t) > 0], dim=0).numpy()
X_scaled = hmm._scaler.transform(val_emb)
logreg_logproba = hmm._logreg.predict_log_proba(X_scaled)  # (N, n_classes_present)
logreg_classes = hmm._logreg.classes_
full_logproba = np.full((val_emb.shape[0], len(PHASE_NAMES)), -1e10)
for col, cls in enumerate(logreg_classes):
    full_logproba[:, cls] = logreg_logproba[:, col]
emission_pred = np.argmax(full_logproba, axis=1)
emission_confidence = np.exp(np.max(full_logproba, axis=1))
emission_true_class_conf = np.exp(full_logproba[np.arange(len(val_lbl)), val_lbl])

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm_emission = confusion_matrix(val_lbl, emission_pred, len(PHASE_NAMES))

def summarize_confusion(cm, names, focus, confidence_true=None, confidence_pred=None, labels=None):
    out = {}
    for name in focus:
        i = names.index(name)
        row = cm[i]
        n = row.sum()
        if n == 0:
            continue
        acc = row[i] / n
        others = [(names[j], int(row[j])) for j in range(len(names)) if j != i and row[j] > 0]
        others.sort(key=lambda x: -x[1])
        entry = {
            "n": int(n), "accuracy": float(acc),
            "top_confusions": [{"confused_with": nm, "n": c, "pct": 100.0 * c / n} for nm, c in others[:4]],
        }
        if confidence_true is not None and labels is not None:
            mask = labels == i
            entry["mean_confidence_true_class"] = float(confidence_true[mask].mean())
            entry["mean_confidence_predicted_class"] = float(confidence_pred[mask].mean())
        out[name] = entry
    return out

emission_summary = summarize_confusion(cm_emission, PHASE_NAMES, FOCUS_PHASES,
                                        emission_true_class_conf, emission_confidence, val_lbl)
print()
print("=== Emission-only (no temporal info) confusion, focus phases ===")
for name, e in emission_summary.items():
    conf_str = ", ".join(f"{c['confused_with']}({c['pct']:.1f}%)" for c in e["top_confusions"])
    print(f"  {name:6s} n={e['n']:5d} acc={e['accuracy']:.3f} conf_true={e['mean_confidence_true_class']:.3f} "
          f"conf_pred={e['mean_confidence_predicted_class']:.3f}  confused_with: {conf_str}")
report["emission_only_confusion_val"] = {
    "overall_accuracy": float((emission_pred == val_lbl).mean()),
    "per_phase": emission_summary,
}

# ---------------------------------------------------------------------------
# 3. HMM filtered phase confusion (causal argmax of filtering(), VAL)
# ---------------------------------------------------------------------------
t0 = time.time()
hmm_true, hmm_pred, hmm_conf_true, hmm_conf_pred = [], [], [], []
for traj in val_traj:
    if len(traj) == 0:
        continue
    filt = hmm.filtering(traj)  # (T,K)
    yt = traj.last_frame_phase.numpy()
    yp = np.argmax(filt, axis=1)
    hmm_true.extend(yt.tolist()); hmm_pred.extend(yp.tolist())
    hmm_conf_true.extend(filt[np.arange(len(yt)), yt].tolist())
    hmm_conf_pred.extend(np.max(filt, axis=1).tolist())
t_hmm_filter = time.time() - t0
hmm_true, hmm_pred = np.array(hmm_true), np.array(hmm_pred)
hmm_conf_true, hmm_conf_pred = np.array(hmm_conf_true), np.array(hmm_conf_pred)
cm_hmm = confusion_matrix(hmm_true, hmm_pred, len(PHASE_NAMES))
hmm_filtered_summary = summarize_confusion(cm_hmm, PHASE_NAMES, FOCUS_PHASES, hmm_conf_true, hmm_conf_pred, hmm_true)
print(f"\nHMM filtering() on Val: {t_hmm_filter:.2f}s, overall phase accuracy={float((hmm_pred==hmm_true).mean()):.4f}")
for name, e in hmm_filtered_summary.items():
    conf_str = ", ".join(f"{c['confused_with']}({c['pct']:.1f}%)" for c in e["top_confusions"])
    print(f"  {name:6s} acc={e['accuracy']:.3f} conf_true={e['mean_confidence_true_class']:.3f}  confused_with: {conf_str}")
report["hmm_filtered_confusion_val"] = {"overall_accuracy": float((hmm_pred == hmm_true).mean()),
                                         "per_phase": hmm_filtered_summary, "time_seconds": t_hmm_filter}

# ---------------------------------------------------------------------------
# 4. Semi-HMM filtered phase confusion AND k=7 consistency prob, SINGLE
#    forward pass per trajectory (avoids paying for _explicit_duration_forward
#    twice -- filtering() and predict_k_step() would each recompute it).
# ---------------------------------------------------------------------------
t0 = time.time()
semi_true, semi_pred, semi_conf_true, semi_conf_pred = [], [], [], []
_NEG_INF = -1e10
for traj in val_traj:
    T = len(traj)
    if T == 0:
        continue
    _, log_age_posterior = semi._explicit_duration_forward(traj)
    log_marginal = logsumexp(log_age_posterior, axis=2)  # (T,K)
    filt = np.exp(log_marginal - logsumexp(log_marginal, axis=1, keepdims=True))
    yt = traj.last_frame_phase.numpy()
    yp = np.argmax(filt, axis=1)
    semi_true.extend(yt.tolist()); semi_pred.extend(yp.tolist())
    semi_conf_true.extend(filt[np.arange(T), yt].tolist())
    semi_conf_pred.extend(np.max(filt, axis=1).tolist())
t_semi_filter = time.time() - t0
semi_true, semi_pred = np.array(semi_true), np.array(semi_pred)
semi_conf_true, semi_conf_pred = np.array(semi_conf_true), np.array(semi_conf_pred)
cm_semi = confusion_matrix(semi_true, semi_pred, len(PHASE_NAMES))
semi_filtered_summary = summarize_confusion(cm_semi, PHASE_NAMES, FOCUS_PHASES, semi_conf_true, semi_conf_pred, semi_true)
print(f"\nSemi-HMM filtering() [combined pass] on Val: {t_semi_filter:.2f}s, "
      f"overall phase accuracy={float((semi_pred==semi_true).mean()):.4f}")
for name, e in semi_filtered_summary.items():
    conf_str = ", ".join(f"{c['confused_with']}({c['pct']:.1f}%)" for c in e["top_confusions"])
    print(f"  {name:6s} acc={e['accuracy']:.3f} conf_true={e['mean_confidence_true_class']:.3f}  confused_with: {conf_str}")
report["semi_hmm_filtered_confusion_val"] = {"overall_accuracy": float((semi_pred == semi_true).mean()),
                                              "per_phase": semi_filtered_summary, "time_seconds": t_semi_filter}

# ---------------------------------------------------------------------------
# 5. Error overlap: does the temporal model fix or inherit emission errors?
#    (per-window, matched by position across the 3 pipelines -- all iterate
#    val_traj in the same order, so array position i means the same window
#    everywhere)
# ---------------------------------------------------------------------------
emission_correct = (emission_pred == val_lbl)
hmm_correct = (hmm_pred == hmm_true)
semi_correct = (semi_pred == semi_true)
assert np.array_equal(val_lbl, hmm_true) and np.array_equal(val_lbl, semi_true), "window order must match across pipelines"

overlap = {}
for name in FOCUS_PHASES:
    i = PHASE_NAMES.index(name)
    mask = val_lbl == i
    n = int(mask.sum())
    if n == 0:
        continue
    em_err = ~emission_correct[mask]
    hmm_err = ~hmm_correct[mask]
    semi_err = ~semi_correct[mask]
    n_em_err = int(em_err.sum())
    overlap[name] = {
        "n": n,
        "emission_error_rate": float(em_err.mean()),
        "hmm_error_rate": float(hmm_err.mean()),
        "semi_hmm_error_rate": float(semi_err.mean()),
        "hmm_fixed_pct_of_emission_errors": float((em_err & ~hmm_err).sum() / n_em_err * 100) if n_em_err else float("nan"),
        "hmm_introduced_new_errors_pct": float((~em_err & hmm_err).sum() / (n - n_em_err) * 100) if (n - n_em_err) else float("nan"),
        "semi_hmm_fixed_pct_of_emission_errors": float((em_err & ~semi_err).sum() / n_em_err * 100) if n_em_err else float("nan"),
        "semi_hmm_introduced_new_errors_pct": float((~em_err & semi_err).sum() / (n - n_em_err) * 100) if (n - n_em_err) else float("nan"),
        "hmm_semi_hmm_same_error_pct": float((hmm_err == semi_err).mean() * 100),
    }
print()
print("=== Error overlap: does temporal modeling fix or inherit emission errors? ===")
for name, o in overlap.items():
    print(f"  {name:6s} emission_err={o['emission_error_rate']:.3f} hmm_err={o['hmm_error_rate']:.3f} "
          f"semi_err={o['semi_hmm_error_rate']:.3f}  HMM fixed {o['hmm_fixed_pct_of_emission_errors']:.1f}% of emission errors, "
          f"Semi-HMM fixed {o['semi_hmm_fixed_pct_of_emission_errors']:.1f}%  (HMM/Semi-HMM agree on error pattern {o['hmm_semi_hmm_same_error_pct']:.1f}%)")
report["error_overlap_analysis"] = overlap

# ---------------------------------------------------------------------------
# 6. Balance/data table for focus phases (Train), reusing Phase E where possible
# ---------------------------------------------------------------------------
balance_table = {}
for name in FOCUS_PHASES:
    i = PHASE_NAMES.index(name)
    per_video_counts = []
    total_consistency_flag = []
    for t in train_traj:
        if len(t) == 0:
            continue
        mask = (t.last_frame_phase.numpy() == i)
        c = int(mask.sum())
        if c > 0:
            per_video_counts.append(c)
            total_consistency_flag.extend(t.consistency_flag.numpy()[mask].tolist())
    pe = phase_e["train_per_phase"].get(name, {})
    balance_table[name] = {
        "n_windows": int(sum(per_video_counts)),
        "n_videos_with_phase": len(per_video_counts),
        "mean_windows_per_video": float(np.mean(per_video_counts)) if per_video_counts else float("nan"),
        "std_windows_per_video": float(np.std(per_video_counts, ddof=1)) if len(per_video_counts) > 1 else 0.0,
        "consistency_flag_rate_train": float(np.mean(total_consistency_flag)) if total_consistency_flag else float("nan"),
        "n_segments_total": pe.get("n_segments_total"),
        "pct_censored": pe.get("pct_censored"),
        "mean_duration_windows": pe.get("all_durations_incl_censored", {}).get("mean"),
        "cv_duration_windows": pe.get("all_durations_incl_censored", {}).get("cv_std_over_mean"),
    }
print()
print("=== Balance/data table (Train), focus phases ===")
for name, b in balance_table.items():
    print(f"  {name:6s} n_windows={b['n_windows']:5d} n_videos={b['n_videos_with_phase']:3d} "
          f"mean_win/vid={b['mean_windows_per_video']:.1f} consistency_rate={b['consistency_flag_rate_train']:.3f} "
          f"mean_dur={b['mean_duration_windows']:.1f} cv={b['cv_duration_windows']:.3f}")
report["balance_data_table"] = balance_table

# ---------------------------------------------------------------------------
# 7. Calibration (k=7), reusing the prior analysis's per-phase table (NOT
#    recomputed -- same Val, same frozen models, avoids a 3rd expensive pass)
# ---------------------------------------------------------------------------
prior_per_phase = prior_analysis["per_phase_table"]
calibration_focus = {name: prior_per_phase[name] for name in FOCUS_PHASES if name in prior_per_phase}
report["calibration_k7_focus_phases_reused_from_prior_analysis"] = calibration_focus
print()
print("=== Calibration k=7 (reused from prior analysis.json), focus phases ===")
for name, c in calibration_focus.items():
    print(f"  {name:6s} obs={c['observed']:.3f} pred_hmm={c['predicted_hmm']:.3f} pred_semi={c['predicted_semi_hmm']:.3f} "
          f"brier_hmm={c['brier_hmm']:.3f} brier_semi={c['brier_semi_hmm']:.3f} "
          f"bias_hmm={c['calibration_bias_hmm']:+.3f} bias_semi={c['calibration_bias_semi_hmm']:+.3f}")

# ---------------------------------------------------------------------------
# 8. Integrity checks
# ---------------------------------------------------------------------------
integrity = {
    "test_split_loaded": False,
    "any_model_refit": False,
    "embeddings_modified": False,
    "n_val_windows_processed": int(len(val_lbl)),
    "n_train_windows_processed": int(len(all_lbl)),
}
report["integrity"] = integrity

(OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2))
print()
print("Report written to", OUT_DIR / "analysis.json")
print("EMISSION_DIAGNOSTIC_DONE")
