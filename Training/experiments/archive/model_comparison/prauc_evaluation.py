"""
PR-AUC (one-vs-rest) addition to the global model comparison, explicitly
requested (2026-08-24) to fill a gap already flagged in
docs/GLOBAL_MODEL_COMPARISON.md sec 3 (Balanced Accuracy / Macro-F1 were
marked N/A there for lack of a script; this run also computes them
alongside PR-AUC, on the SAME predictions used throughout that report).

Read-only: loads the two ALREADY FROZEN models (e1_hmm_k7_sweep/best_model,
semi_hmm_weekend_phaseF/model) via .load(), never refits either. Computes
15-way phase-probability predictions on Val for three sources that
genuinely produce a full phase posterior (E1/GRU never do -- confirmed
repeatedly this session, not re-litigated here):
  1. emission-only  (HMM's own _scaler/_logreg, no temporal information)
  2. HMM filtering  (HMMModel.filtering(), causal, temporally smoothed)
  3. Semi-HMM filtering (SemiHMMModel.filtering(), causal, duration+
     transition-aware)
All three are evaluated on the IDENTICAL Val trajectories/windows/labels
(same order, same 43,339-window set) used by every other metric in this
project's HMM/Semi-HMM branch -- never a different split or subsample.

Definitions used (documented explicitly, per the task's own requirement):
  - PR-AUC per phase: sklearn.metrics.average_precision_score(y_true_binary,
    y_score_class), one-vs-rest (phase c positive, all others negative).
  - AUROC per phase: sklearn.metrics.roc_auc_score, same one-vs-rest binarization.
  - PR-AUC macro / AUROC macro: unweighted mean across the 15 per-phase values.
  - PR-AUC weighted: mean weighted by each phase's support (n_true positives) in Val.
  - Per-phase F1/Precision/Recall: computed on the ARGMAX hard decision
    (predicted phase = argmax of the probability vector), NOT a
    thresholded one-vs-rest binary classifier -- documented explicitly
    because this is a different (and more standard, for a genuinely
    multiclass problem) decision rule than the 0.5-threshold binary
    consistency_flag metrics used elsewhere in this project.
  - Balanced Accuracy / Macro-F1: sklearn.metrics.balanced_accuracy_score /
    f1_score(average='macro'), on the same argmax decisions.
  - Multiclass Brier: sum-of-squares convention (sum_c (onehot_c-p_c)^2,
    averaged over samples) -- IDENTICAL formula/convention already used in
    emission_weighted_sweep_experiment.py and emission_alpha025_experiment.py,
    reused verbatim here for direct comparability.
  - Log Loss: sklearn.metrics.log_loss, multiclass, labels=range(15) pinned
    explicitly (never inferred from y_true, so a class absent from one
    source's argmax predictions still gets a defined column).
  - ECE: top-label (argmax confidence vs. argmax correctness), reusing
    evaluation.calibration.calibration_bins/expected_calibration_error
    verbatim -- NOT reimplemented.
Test split: never loaded, never referenced.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, log_loss,
    precision_recall_fscore_support, roc_auc_score,
)

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import calibration_bins, expected_calibration_error
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/global_model_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE_NAMES = list(DEFAULT_PHASE_NAMES)
K = len(PHASE_NAMES)
TARGET_PHASES = ["tPNf", "t3", "t5", "t6", "tB"]
REFERENCE_PHASES = ["t9+", "t8"]

results = {"protocol": {
    "definition": {
        "pr_auc_per_phase": "sklearn.metrics.average_precision_score, one-vs-rest binarization "
                             "(this phase=1, all 14 others=0)",
        "auroc_per_phase": "sklearn.metrics.roc_auc_score, same one-vs-rest binarization",
        "pr_auc_macro": "unweighted mean of the 15 per-phase PR-AUC values",
        "pr_auc_weighted": "mean of the 15 per-phase PR-AUC values weighted by each phase's Val support",
        "auroc_macro": "unweighted mean of the 15 per-phase AUROC values",
        "per_phase_f1_precision_recall": "computed on the ARGMAX hard decision (predicted phase = "
                                          "argmax of the probability vector), NOT 0.5-threshold binary",
        "balanced_accuracy_macro_f1": "sklearn balanced_accuracy_score / f1_score(average=macro), "
                                       "same argmax decisions",
        "multiclass_brier": "sum_c (onehot_c - p_c)^2, averaged over samples -- same convention as "
                             "emission_weighted_sweep_experiment.py / emission_alpha025_experiment.py",
        "log_loss": "sklearn.metrics.log_loss, multiclass, labels=range(15) pinned explicitly",
        "ece": "top-label (argmax confidence vs argmax correctness), evaluation.calibration verbatim",
    },
    "models_evaluated": ["emission_only_unweighted", "hmm_k7_filtering", "semi_hmm_filtering"],
    "note": "All three scored on the IDENTICAL Val set (43339 windows, 106 trajectories), same order, "
            "same labels -- directly comparable to each other and to every other metric already reported "
            "for these frozen models elsewhere in this project. Test never loaded. Models loaded via "
            ".load(), never refit.",
}}

t0 = time.time()
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_val_traj={len(val_traj)}")
results["load_time_seconds"] = t_load

hmm = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
semi_hmm = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
assert hmm.logreg_class_weight is None
assert semi_hmm.logreg_class_weight is None
print(f"Loaded frozen HMM (class_weight={hmm.logreg_class_weight}) and Semi-HMM "
      f"(class_weight={semi_hmm.logreg_class_weight}), never refit.")

y_true = torch.cat([t.last_frame_phase for t in val_traj if len(t) > 0]).numpy()
print(f"n_windows={len(y_true)}")
results["n_windows"] = int(len(y_true))

# ---------------------------------------------------------------------------
# 1. emission-only probabilities (HMM's own _scaler/_logreg, no temporal info)
# ---------------------------------------------------------------------------
val_emb = torch.cat([t.embeddings for t in val_traj if len(t) > 0], dim=0).numpy()
X_scaled = hmm._scaler.transform(val_emb)
logproba = hmm._logreg.predict_log_proba(X_scaled)
emission_probs = np.full((val_emb.shape[0], K), 1e-300)
for col, cls in enumerate(hmm._logreg.classes_):
    emission_probs[:, cls] = np.exp(logproba[:, col])
emission_probs = emission_probs / emission_probs.sum(axis=1, keepdims=True)
t0 = time.time()
print(f"emission-only probs computed: {time.time()-t0:.2f}s, shape={emission_probs.shape}")

# ---------------------------------------------------------------------------
# 2. HMM filtering (causal, temporally smoothed)
# ---------------------------------------------------------------------------
t0 = time.time()
hmm_probs = np.concatenate([hmm.filtering(t) for t in val_traj if len(t) > 0], axis=0)
t_hmm = time.time() - t0
print(f"HMM filtering computed: {t_hmm:.2f}s, shape={hmm_probs.shape}")
results["hmm_filtering_time_seconds"] = t_hmm

# ---------------------------------------------------------------------------
# 3. Semi-HMM filtering (causal, duration+transition-aware)
# ---------------------------------------------------------------------------
t0 = time.time()
semi_probs = np.concatenate([semi_hmm.filtering(t) for t in val_traj if len(t) > 0], axis=0)
t_semi = time.time() - t0
print(f"Semi-HMM filtering computed: {t_semi:.2f}s, shape={semi_probs.shape}")
results["semi_hmm_filtering_time_seconds"] = t_semi

assert emission_probs.shape == hmm_probs.shape == semi_probs.shape == (len(y_true), K)
for name, p in (("emission", emission_probs), ("hmm", hmm_probs), ("semi_hmm", semi_probs)):
    assert np.all(np.isfinite(p)), f"{name} has non-finite probabilities"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4), f"{name} probabilities don't sum to 1"
print("\nAll three probability matrices: finite, sum-to-1 verified, identical shape/order.")


def evaluate(name, probs, y_true):
    y_pred = np.argmax(probs, axis=1)
    onehot = np.eye(K)[y_true]

    per_phase = {}
    support = np.bincount(y_true, minlength=K)
    pr_auc_vals, auroc_vals = np.full(K, np.nan), np.full(K, np.nan)
    for c in range(K):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            pr_auc = float("nan")
            auroc = float("nan")
        else:
            pr_auc = float(average_precision_score(y_bin, probs[:, c]))
            auroc = float(roc_auc_score(y_bin, probs[:, c]))
        pr_auc_vals[c] = pr_auc
        auroc_vals[c] = auroc
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(K)), average=None, zero_division=0
    )
    for c in range(K):
        per_phase[PHASE_NAMES[c]] = {
            "support": int(support[c]), "auroc_ovr": auroc_vals[c], "pr_auc_ovr": pr_auc_vals[c],
            "f1_argmax": float(f1[c]), "precision_argmax": float(precision[c]), "recall_argmax": float(recall[c]),
        }

    valid = ~np.isnan(pr_auc_vals)
    pr_auc_macro = float(np.nanmean(pr_auc_vals))
    pr_auc_weighted = float(np.average(pr_auc_vals[valid], weights=support[valid]))
    auroc_macro = float(np.nanmean(auroc_vals))
    auroc_weighted = float(np.average(auroc_vals[valid], weights=support[valid]))

    multiclass_brier = float(np.mean(np.sum((onehot - probs) ** 2, axis=1)))
    probs_clipped = np.clip(probs, 1e-12, 1.0)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    logloss = float(log_loss(y_true, probs_clipped, labels=list(range(K))))

    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(int)
    ece_bins = calibration_bins(correct, conf, n_bins=10)
    ece = expected_calibration_error(ece_bins, len(correct))

    accuracy = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    summary = {
        "accuracy": accuracy, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "auroc_macro": auroc_macro, "auroc_weighted": auroc_weighted,
        "pr_auc_macro": pr_auc_macro, "pr_auc_weighted": pr_auc_weighted,
        "multiclass_brier": multiclass_brier, "log_loss": logloss, "ece": ece,
    }
    print(f"\n=== {name} ===")
    print(f"  accuracy={accuracy:.4f} balanced_acc={bal_acc:.4f} macro_f1={macro_f1:.4f}")
    print(f"  auroc_macro={auroc_macro:.4f} pr_auc_macro={pr_auc_macro:.4f} pr_auc_weighted={pr_auc_weighted:.4f}")
    print(f"  brier={multiclass_brier:.4f} log_loss={logloss:.4f} ece={ece:.4f}")
    print("  focus phases:")
    for ph in TARGET_PHASES + REFERENCE_PHASES:
        pp = per_phase[ph]
        print(f"    {ph:6s} support={pp['support']:6d} auroc={pp['auroc_ovr']:.3f} pr_auc={pp['pr_auc_ovr']:.3f} "
              f"f1={pp['f1_argmax']:.3f} precision={pp['precision_argmax']:.3f} recall={pp['recall_argmax']:.3f}")
    return summary, per_phase


emission_summary, emission_per_phase = evaluate("emission_only_unweighted", emission_probs, y_true)
hmm_summary, hmm_per_phase = evaluate("hmm_k7_filtering", hmm_probs, y_true)
semi_summary, semi_per_phase = evaluate("semi_hmm_filtering", semi_probs, y_true)

results["summary"] = {
    "emission_only_unweighted": emission_summary,
    "hmm_k7_filtering": hmm_summary,
    "semi_hmm_filtering": semi_summary,
}
results["per_phase"] = {
    "emission_only_unweighted": emission_per_phase,
    "hmm_k7_filtering": hmm_per_phase,
    "semi_hmm_filtering": semi_per_phase,
}

out_path = OUT_DIR / "prauc_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults written to {out_path}")
print("PRAUC_EVALUATION_DONE")
print("EXITCODE:0")
