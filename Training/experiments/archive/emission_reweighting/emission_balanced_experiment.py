"""
Emission-rebalancing experiment (explicitly authorized 2026-08-23).
Tests whether class_weight='balanced' on the shared emission's
LogisticRegression fixes the catastrophic per-window accuracy on
short/underrepresented phases (t3/t5/t6: 3-4% emission-only accuracy,
Results/evaluation/semi_hmm_next_step_analysis/emission_diagnostic/).

STRICT CONTROLS (only class_weight varies):
  - embeddings: Embeddings/resnet18/{train,val}, untouched
  - Train (492 traj) / Val (106 traj): identical splits
  - n_states=15, DEFAULT_PHASE_NAMES mapping: identical
  - HMM transition hyperparameters: alpha=2.0, rho=0.5 (the FROZEN,
    Val-selected e1_hmm_k7_sweep winner) -- matched exactly, not re-swept
  - Semi-HMM transition/duration hyperparameters: class DEFAULTS
    (alpha=1.0, rho=0.3) + dmax=268, duration_family=negative_binomial --
    matches Phase F's own actual call exactly (SemiHMMModel(n_states=15,
    dmax=268, duration_family="negative_binomial")), not re-derived
  - logreg_C=1.0, logreg_max_iter=1000: unchanged defaults
  - ONLY logreg_class_weight varies: None (baseline, loaded from the
    existing FROZEN models, never refit) vs 'balanced' (new fit)

Baseline models are LOADED, never refit -- e1_hmm_k7_sweep/best_model and
semi_hmm_weekend_phaseF/model are historical artifacts, untouched by this
script (verified by checksum before/after, separately from this script).
Test split is NEVER loaded, never referenced.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from scipy.special import logsumexp
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, brier_score_loss

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import calibration_bins, expected_calibration_error, flatten_predictions
from evaluation.metrics import compute_metrics, paired_bootstrap_comparison
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/emission_balanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE_NAMES = list(DEFAULT_PHASE_NAMES)
TARGET_PHASES = ["tPNf", "t3", "t5", "t6", "tB"]
REFERENCE_PHASES = ["t9+", "t8"]
FOCUS_PHASES = TARGET_PHASES + REFERENCE_PHASES
K7 = 7
HMM_ALPHA, HMM_RHO = 2.0, 0.5  # frozen e1_hmm_k7_sweep winner, matched exactly
SEMI_DMAX = 268  # frozen Phase F decision

results = {"protocol": {
    "target_phases": TARGET_PHASES, "reference_phases": REFERENCE_PHASES,
    "hmm_alpha": HMM_ALPHA, "hmm_rho": HMM_RHO, "semi_dmax": SEMI_DMAX,
    "semi_duration_family": "negative_binomial",
    "note": "only logreg_class_weight varies (None baseline, loaded not refit, vs 'balanced', newly fit); "
            "Test never loaded",
}}

t0 = time.time()
train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")
results["load_time_seconds"] = t_load

# ---------------------------------------------------------------------------
# 0. Load FROZEN baselines (never refit)
# ---------------------------------------------------------------------------
hmm_baseline = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
semi_baseline = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
assert hmm_baseline.logreg_class_weight is None
assert semi_baseline.logreg_class_weight is None
print(f"baseline HMM loaded (class_weight={hmm_baseline.logreg_class_weight}), "
      f"baseline Semi-HMM loaded (class_weight={semi_baseline.logreg_class_weight})")

# ---------------------------------------------------------------------------
# 1. Fit BALANCED HMM (only class_weight differs from baseline)
# ---------------------------------------------------------------------------
t0 = time.time()
hmm_balanced = HMMModel(
    n_states=15, transition_smoothing_alpha=HMM_ALPHA, transition_decay_rho=HMM_RHO,
    logreg_class_weight="balanced",
)
hmm_balanced.fit(train_traj)
t_fit_hmm = time.time() - t0
print(f"HMM balanced fit: {t_fit_hmm:.2f}s")
results["hmm_balanced_fit_time_seconds"] = t_fit_hmm

# ---------------------------------------------------------------------------
# 2. Emission-only evaluation on Val: baseline vs balanced
# ---------------------------------------------------------------------------
val_emb = torch.cat([t.embeddings for t in val_traj if len(t) > 0], dim=0).numpy()
val_lbl = torch.cat([t.last_frame_phase for t in val_traj if len(t) > 0], dim=0).numpy()

def emission_only_eval(model, emb, lbl):
    X_scaled = model._scaler.transform(emb)
    logproba = model._logreg.predict_log_proba(X_scaled)
    full = np.full((emb.shape[0], len(PHASE_NAMES)), -1e10)
    for col, cls in enumerate(model._logreg.classes_):
        full[:, cls] = logproba[:, col]
    pred = np.argmax(full, axis=1)
    conf = np.exp(np.max(full, axis=1))
    conf_true = np.exp(full[np.arange(len(lbl)), lbl])
    cm = np.zeros((len(PHASE_NAMES), len(PHASE_NAMES)), dtype=int)
    for t_, p_ in zip(lbl, pred):
        cm[t_, p_] += 1
    per_phase = {}
    for name in FOCUS_PHASES:
        i = PHASE_NAMES.index(name)
        row = cm[i]
        n = int(row.sum())
        if n == 0:
            continue
        mask = lbl == i
        others = [(PHASE_NAMES[j], int(row[j])) for j in range(len(PHASE_NAMES)) if j != i and row[j] > 0]
        others.sort(key=lambda x: -x[1])
        per_phase[name] = {
            "n": n, "accuracy": float(row[i] / n),
            "mean_confidence_true_class": float(conf_true[mask].mean()),
            "mean_confidence_predicted_class": float(conf[mask].mean()),
            "top_confusions": [{"confused_with": nm, "n": c, "pct": 100.0 * c / n} for nm, c in others[:4]],
        }
    return {"overall_accuracy": float((pred == lbl).mean()), "per_phase": per_phase}

emission_baseline_eval = emission_only_eval(hmm_baseline, val_emb, val_lbl)
emission_balanced_eval = emission_only_eval(hmm_balanced, val_emb, val_lbl)
print()
print(f"=== Emission-only: overall accuracy baseline={emission_baseline_eval['overall_accuracy']:.4f} "
      f"balanced={emission_balanced_eval['overall_accuracy']:.4f} ===")
for name in FOCUS_PHASES:
    b = emission_baseline_eval["per_phase"].get(name)
    w = emission_balanced_eval["per_phase"].get(name)
    if b is None or w is None:
        continue
    print(f"  {name:6s} baseline_acc={b['accuracy']:.3f} balanced_acc={w['accuracy']:.3f} "
          f"delta={w['accuracy']-b['accuracy']:+.3f}  conf_true baseline={b['mean_confidence_true_class']:.3f} "
          f"balanced={w['mean_confidence_true_class']:.3f}")
results["emission_only"] = {"baseline": emission_baseline_eval, "balanced": emission_balanced_eval}

target_acc_baseline = np.mean([emission_baseline_eval["per_phase"][n]["accuracy"] for n in TARGET_PHASES])
target_acc_balanced = np.mean([emission_balanced_eval["per_phase"][n]["accuracy"] for n in TARGET_PHASES])
gate_improvement = target_acc_balanced - target_acc_baseline
print(f"\nGate check: mean target-phase emission accuracy baseline={target_acc_baseline:.4f} "
      f"balanced={target_acc_balanced:.4f} delta={gate_improvement:+.4f}")
results["gate_check"] = {
    "mean_target_accuracy_baseline": float(target_acc_baseline),
    "mean_target_accuracy_balanced": float(target_acc_balanced),
    "delta": float(gate_improvement),
    "threshold": 0.05,
    "proceed_to_temporal_models": bool(gate_improvement > 0.05),
}
PROCEED = gate_improvement > 0.05
print(f"PROCEED to HMM/Semi-HMM k=7 evaluation: {PROCEED}")

if not PROCEED:
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nGate not passed (delta <= 0.05) -- stopping after emission-only evaluation as instructed.")
    print("EMISSION_BALANCED_DONE")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 3. HMM k=7 evaluation: baseline (loaded, frozen) vs balanced (freshly fit)
# ---------------------------------------------------------------------------
def score(y_true, y_prob, n_bins=10):
    bins = calibration_bins(y_true, y_prob, n_bins)
    n = len(y_true)
    return {
        "n_windows": n,
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "brier_baseline_constant_rate": float(brier_score_loss(y_true, np.full(n, y_true.mean()))),
        "ece": expected_calibration_error(bins, n),
    }

t0 = time.time()
hmm_baseline_preds = hmm_baseline.predict_k_step(val_traj, k=K7)
hmm_balanced_preds = hmm_balanced.predict_k_step(val_traj, k=K7)
t_hmm_predict = time.time() - t0
print(f"\nHMM predict_k_step(k=7) baseline+balanced: {t_hmm_predict:.2f}s")

y_true_hb, y_prob_hb = flatten_predictions(val_traj, hmm_baseline_preds)
y_true_hw, y_prob_hw = flatten_predictions(val_traj, hmm_balanced_preds)
assert np.array_equal(y_true_hb, y_true_hw)
score_hmm_baseline = score(y_true_hb, y_prob_hb)
score_hmm_balanced = score(y_true_hw, y_prob_hw)
print(f"HMM k=7 baseline: AUROC={score_hmm_baseline['auroc']:.5f} Brier={score_hmm_baseline['brier']:.5f} "
      f"ECE={score_hmm_baseline['ece']:.5f}")
print(f"HMM k=7 balanced: AUROC={score_hmm_balanced['auroc']:.5f} Brier={score_hmm_balanced['brier']:.5f} "
      f"ECE={score_hmm_balanced['ece']:.5f}")
results["hmm_k7"] = {"baseline": score_hmm_baseline, "balanced": score_hmm_balanced,
                      "predict_time_seconds": t_hmm_predict}

# per-phase HMM k=7
def per_phase_k7(preds, val_traj):
    out = {name: {"true": [], "prob": []} for name in PHASE_NAMES}
    for traj, pred in zip(val_traj, preds):
        lf = traj.last_frame_phase.numpy()
        yt = traj.consistency_flag.numpy()
        yp = pred.consistency_flag_prob.numpy()
        for i, idx in enumerate(lf.tolist()):
            out[PHASE_NAMES[idx]]["true"].append(yt[i])
            out[PHASE_NAMES[idx]]["prob"].append(yp[i])
    summary = {}
    for name, d in out.items():
        if not d["true"]:
            continue
        yt = np.array(d["true"]); yp = np.array(d["prob"])
        summary[name] = {"n": len(yt), "observed": float(yt.mean()), "predicted": float(yp.mean()),
                          "brier": float(np.mean((yp - yt) ** 2)), "bias": float(yp.mean() - yt.mean())}
    return summary

hmm_baseline_phase = per_phase_k7(hmm_baseline_preds, val_traj)
hmm_balanced_phase = per_phase_k7(hmm_balanced_preds, val_traj)
print("\n=== HMM k=7 per-phase (focus phases): baseline vs balanced ===")
for name in FOCUS_PHASES:
    b = hmm_baseline_phase.get(name); w = hmm_balanced_phase.get(name)
    if b is None or w is None: continue
    print(f"  {name:6s} obs={b['observed']:.3f} brier_baseline={b['brier']:.3f} brier_balanced={w['brier']:.3f} "
          f"bias_baseline={b['bias']:+.3f} bias_balanced={w['bias']:+.3f}")
results["hmm_k7_per_phase"] = {"baseline": hmm_baseline_phase, "balanced": hmm_balanced_phase}

# trajectory-level + bootstrap comparison (HMM)
def trajectory_level_and_bootstrap(preds_a, preds_b, val_traj, n_bootstrap=1000, seed=0):
    per_traj_yt = [t.consistency_flag.numpy() for t in val_traj]
    per_traj_a = [p.consistency_flag_prob.numpy() for p in preds_a]
    per_traj_b = [p.consistency_flag_prob.numpy() for p in preds_b]
    traj_delta_brier = []
    for yt, pa, pb in zip(per_traj_yt, per_traj_a, per_traj_b):
        if len(yt) == 0: continue
        ba = float(np.mean((pa - yt) ** 2)); bb = float(np.mean((pb - yt) ** 2))
        traj_delta_brier.append(bb - ba)
    traj_delta_brier = np.array(traj_delta_brier)
    wilcoxon_p = float(scipy_stats.wilcoxon(traj_delta_brier).pvalue) if not np.allclose(traj_delta_brier, 0) else 1.0
    n = len(val_traj)
    rng = np.random.RandomState(seed)
    raw_a, raw_b = {"auroc": [], "brier": [], "ece": []}, {"auroc": [], "brier": [], "ece": []}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = np.concatenate([per_traj_yt[i] for i in idx])
        pa = np.concatenate([per_traj_a[i] for i in idx])
        pb = np.concatenate([per_traj_b[i] for i in idx])
        if len(np.unique(yt)) == 2:
            raw_a["auroc"].append(roc_auc_score(yt, pa)); raw_b["auroc"].append(roc_auc_score(yt, pb))
        raw_a["brier"].append(brier_score_loss(yt, pa)); raw_b["brier"].append(brier_score_loss(yt, pb))
        ba_bins = calibration_bins(yt, pa, 10); bb_bins = calibration_bins(yt, pb, 10)
        raw_a["ece"].append(expected_calibration_error(ba_bins, len(yt)))
        raw_b["ece"].append(expected_calibration_error(bb_bins, len(yt)))
    raw_a = {k: np.asarray(v) for k, v in raw_a.items()}
    raw_b = {k: np.asarray(v) for k, v in raw_b.items()}
    bootstrap_cmp = {}
    for metric in ("auroc", "brier", "ece"):
        if len(raw_a[metric]) == len(raw_b[metric]) and len(raw_a[metric]) > 0:
            bootstrap_cmp[metric] = paired_bootstrap_comparison(raw_a, raw_b, metric)
    return {
        "trajectory_level": {
            "n_trajectories": len(traj_delta_brier), "mean_delta_brier": float(traj_delta_brier.mean()),
            "wilcoxon_p": wilcoxon_p,
            "n_balanced_better": int((traj_delta_brier < 0).sum()),
            "n_baseline_better": int((traj_delta_brier > 0).sum()),
        },
        "bootstrap": bootstrap_cmp,
    }

hmm_stats = trajectory_level_and_bootstrap(hmm_baseline_preds, hmm_balanced_preds, val_traj, seed=100)
print(f"\nHMM trajectory-level: mean_delta_brier={hmm_stats['trajectory_level']['mean_delta_brier']:+.5f} "
      f"wilcoxon_p={hmm_stats['trajectory_level']['wilcoxon_p']:.4f} "
      f"balanced_better={hmm_stats['trajectory_level']['n_balanced_better']}/106")
for metric, c in hmm_stats["bootstrap"].items():
    print(f"  bootstrap {metric}: mean_diff(balanced-baseline)={c['mean_difference']:+.5f} "
          f"CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
results["hmm_statistical_comparison"] = hmm_stats

# ---------------------------------------------------------------------------
# 4. Semi-HMM: baseline (loaded, frozen) vs balanced (freshly fit)
# ---------------------------------------------------------------------------
t0 = time.time()
semi_balanced = SemiHMMModel(n_states=15, dmax=SEMI_DMAX, duration_family="negative_binomial",
                              logreg_class_weight="balanced")
semi_balanced.fit(train_traj)
t_fit_semi = time.time() - t0
print(f"\nSemi-HMM balanced fit: {t_fit_semi:.2f}s")
results["semi_hmm_balanced_fit_time_seconds"] = t_fit_semi

t0 = time.time()
semi_baseline_preds = semi_baseline.predict_k_step(val_traj, k=K7)
semi_balanced_preds = semi_balanced.predict_k_step(val_traj, k=K7)
t_semi_predict = time.time() - t0
print(f"Semi-HMM predict_k_step(k=7) baseline+balanced: {t_semi_predict:.2f}s")

y_true_sb, y_prob_sb = flatten_predictions(val_traj, semi_baseline_preds)
y_true_sw, y_prob_sw = flatten_predictions(val_traj, semi_balanced_preds)
assert np.array_equal(y_true_sb, y_true_sw)
score_semi_baseline = score(y_true_sb, y_prob_sb)
score_semi_balanced = score(y_true_sw, y_prob_sw)
print(f"Semi-HMM k=7 baseline: AUROC={score_semi_baseline['auroc']:.5f} Brier={score_semi_baseline['brier']:.5f} "
      f"ECE={score_semi_baseline['ece']:.5f}")
print(f"Semi-HMM k=7 balanced: AUROC={score_semi_balanced['auroc']:.5f} Brier={score_semi_balanced['brier']:.5f} "
      f"ECE={score_semi_balanced['ece']:.5f}")
results["semi_hmm_k7"] = {"baseline": score_semi_baseline, "balanced": score_semi_balanced,
                           "predict_time_seconds": t_semi_predict}

semi_baseline_phase = per_phase_k7(semi_baseline_preds, val_traj)
semi_balanced_phase = per_phase_k7(semi_balanced_preds, val_traj)
print("\n=== Semi-HMM k=7 per-phase (focus phases): baseline vs balanced ===")
for name in FOCUS_PHASES:
    b = semi_baseline_phase.get(name); w = semi_balanced_phase.get(name)
    if b is None or w is None: continue
    print(f"  {name:6s} obs={b['observed']:.3f} brier_baseline={b['brier']:.3f} brier_balanced={w['brier']:.3f} "
          f"bias_baseline={b['bias']:+.3f} bias_balanced={w['bias']:+.3f}")
results["semi_hmm_k7_per_phase"] = {"baseline": semi_baseline_phase, "balanced": semi_balanced_phase}

semi_stats = trajectory_level_and_bootstrap(semi_baseline_preds, semi_balanced_preds, val_traj, seed=200)
print(f"\nSemi-HMM trajectory-level: mean_delta_brier={semi_stats['trajectory_level']['mean_delta_brier']:+.5f} "
      f"wilcoxon_p={semi_stats['trajectory_level']['wilcoxon_p']:.4f} "
      f"balanced_better={semi_stats['trajectory_level']['n_balanced_better']}/106")
for metric, c in semi_stats["bootstrap"].items():
    print(f"  bootstrap {metric}: mean_diff(balanced-baseline)={c['mean_difference']:+.5f} "
          f"CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
results["semi_hmm_statistical_comparison"] = semi_stats

# Does Semi-HMM retain an edge over HMM AFTER the emission fix? (Q5)
hmm_balanced_vs_semi_balanced = trajectory_level_and_bootstrap(hmm_balanced_preds, semi_balanced_preds, val_traj, seed=300)
print(f"\n=== Semi-HMM balanced vs HMM balanced (does Semi-HMM still add value after the emission fix?) ===")
print(f"  trajectory-level mean_delta_brier(semi-hmm - hmm)={hmm_balanced_vs_semi_balanced['trajectory_level']['mean_delta_brier']:+.5f} "
      f"wilcoxon_p={hmm_balanced_vs_semi_balanced['trajectory_level']['wilcoxon_p']:.4f}")
for metric, c in hmm_balanced_vs_semi_balanced["bootstrap"].items():
    print(f"  bootstrap {metric}: mean_diff(semi_hmm_balanced - hmm_balanced)={c['mean_difference']:+.5f} "
          f"CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
results["hmm_balanced_vs_semi_hmm_balanced"] = hmm_balanced_vs_semi_balanced

# ---------------------------------------------------------------------------
# 5. Persist models and results
# ---------------------------------------------------------------------------
hmm_balanced.save(OUT_DIR / "hmm_balanced_model")
semi_balanced.save(OUT_DIR / "semi_hmm_balanced_model")
(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
print("\nResults written to", OUT_DIR / "results.json")
print("Models saved to", OUT_DIR / "hmm_balanced_model", "and", OUT_DIR / "semi_hmm_balanced_model")
print("EMISSION_BALANCED_DONE")
