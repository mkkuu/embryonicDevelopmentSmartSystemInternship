"""
Single-config follow-up experiment (explicitly authorized 2026-08-24), closing
the one open question left by emission_weighted_sweep: alpha=0.25 showed the
best emission-level top-label ECE (0.114-0.116 vs 0.121 baseline) but never
reached HMM/Semi-HMM k=7 because it failed the pre-registered target-phase
accuracy gate (+0.0295 < +0.05). This script propagates alpha=0.25 to k=7
unconditionally (no gate this time -- explicitly user-authorized regardless
of outcome) to answer: does the emission-level calibration gain survive?

w_c(alpha=0.25) = (N / (K * N_c)) ** 0.25 -- identical formula/normalization
as emission_weighted_sweep, so alpha=0.25 here is bit-for-bit the same
weighting that run's emission-only table already measured (recomputed here
only because that run never fit an HMM/Semi-HMM at this specific alpha).

STRICT CONTROLS (only alpha=0.25's weight dict is new; everything else
identical to every prior experiment in this branch):
  - embeddings: Embeddings/resnet18/{train,val}, untouched, read-only.
  - Train (492 traj) / Val (106 traj): identical splits.
  - HMM transition: alpha=2.0, rho=0.5 (frozen e1_hmm_k7_sweep winner).
  - Semi-HMM transition/duration: class defaults (alpha=1.0, rho=0.3),
    dmax=268, duration_family='negative_binomial' (Phase F's exact call).
  - logreg_C=1.0, logreg_max_iter=1000: unchanged.
  - k=7 throughout.
alpha=0 reference (HMM/Semi-HMM) reused from the ALREADY frozen models --
never refit -- but its k=7 predictions ARE freshly recomputed here (predict
only, no fit) so this run's alpha=0 vs alpha=0.25 comparison is fully
self-consistent under one code path, matching emission_weighted_sweep's own
methodology.
Test split: never loaded, never referenced.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, brier_score_loss

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import calibration_bins, expected_calibration_error, flatten_predictions
from evaluation.metrics import compute_metrics, paired_bootstrap_comparison
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/emission_alpha025")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = OUT_DIR / "status.txt"
STATUS_FILE.write_text(f"RUNNING {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

PHASE_NAMES = list(DEFAULT_PHASE_NAMES)
K = len(PHASE_NAMES)
TARGET_PHASES = ["tPNf", "t3", "t5", "t6", "tB"]
REFERENCE_PHASES = ["t9+", "t8"]
FOCUS_PHASES = TARGET_PHASES + REFERENCE_PHASES
K7 = 7
HMM_ALPHA, HMM_RHO = 2.0, 0.5
SEMI_DMAX = 268
ALPHA = 0.25

results = {"protocol": {
    "target_phases": TARGET_PHASES, "reference_phases": REFERENCE_PHASES,
    "hmm_alpha": HMM_ALPHA, "hmm_rho": HMM_RHO, "semi_dmax": SEMI_DMAX,
    "semi_duration_family": "negative_binomial", "test_alpha": ALPHA,
    "weighting_formula": "w_c(alpha) = (N / (K * N_c)) ** alpha",
    "note": "single-config follow-up to emission_weighted_sweep, closing the alpha=0.25 gap "
            "(best emission ECE, never reached k=7 due to gate G1); no gate applied here, "
            "propagation to k=7 is unconditional per explicit user authorization; "
            "alpha=0 reused from frozen models, k=7 predictions freshly recomputed (predict-only); "
            "Test never loaded",
}}

try:
    t0 = time.time()
    train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
    val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
    t_load = time.time() - t0
    print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")
    results["load_time_seconds"] = t_load

    all_train_labels = torch.cat([t.last_frame_phase for t in train_traj if len(t) > 0]).numpy()
    N_total = len(all_train_labels)
    counts = np.array([(all_train_labels == c).sum() for c in range(K)], dtype=np.float64)
    base = N_total / (K * counts)
    w = base ** ALPHA
    weight_dict = {c: float(w[c]) for c in range(K)}
    print(f"alpha=0.25 weights: min={min(weight_dict.values()):.4f} max={max(weight_dict.values()):.4f} "
          f"w[t9+]={weight_dict[PHASE_NAMES.index('t9+')]:.4f} w[t3]={weight_dict[PHASE_NAMES.index('t3')]:.4f}")
    results["weight_dict"] = {PHASE_NAMES[c]: weight_dict[c] for c in range(K)}

    hmm_0 = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
    semi_0 = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
    assert hmm_0.logreg_class_weight is None
    assert semi_0.logreg_class_weight is None
    print("Loaded frozen alpha=0 HMM/Semi-HMM (never refit).")

    t0 = time.time()
    hmm_025 = HMMModel(n_states=K, transition_smoothing_alpha=HMM_ALPHA, transition_decay_rho=HMM_RHO,
                        logreg_class_weight=weight_dict)
    hmm_025.fit(train_traj)
    t_fit_hmm = time.time() - t0
    print(f"HMM alpha=0.25 fit: {t_fit_hmm:.2f}s")
    results["hmm_fit_time_seconds"] = t_fit_hmm

    # -----------------------------------------------------------------
    # emission-only evaluation
    # -----------------------------------------------------------------
    val_emb = torch.cat([t.embeddings for t in val_traj if len(t) > 0], dim=0).numpy()
    val_lbl = torch.cat([t.last_frame_phase for t in val_traj if len(t) > 0], dim=0).numpy()

    def emission_only_eval(model, emb, lbl):
        X_scaled = model._scaler.transform(emb)
        logproba = model._logreg.predict_log_proba(X_scaled)
        full = np.full((emb.shape[0], K), -1e10)
        for col, cls in enumerate(model._logreg.classes_):
            full[:, cls] = logproba[:, col]
        probs = np.exp(full)
        pred = np.argmax(full, axis=1)
        conf = np.max(probs, axis=1)
        conf_true = probs[np.arange(len(lbl)), lbl]
        correct = (pred == lbl).astype(int)
        onehot = np.eye(K)[lbl]
        multiclass_brier = float(np.mean(np.sum((onehot - probs) ** 2, axis=1)))
        ece_bins = calibration_bins(correct, conf, n_bins=10)
        ece = expected_calibration_error(ece_bins, len(correct))
        cm = np.zeros((K, K), dtype=int)
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
            others = [(PHASE_NAMES[j], int(row[j])) for j in range(K) if j != i and row[j] > 0]
            others.sort(key=lambda x: -x[1])
            per_phase[name] = {
                "n": n, "accuracy": float(row[i] / n),
                "mean_confidence_true_class": float(conf_true[mask].mean()),
                "mean_confidence_predicted_class": float(conf[mask].mean()),
                "top_confusions": [{"confused_with": nm, "n": c, "pct": 100.0 * c / n} for nm, c in others[:4]],
            }
        return {
            "overall_accuracy": float((pred == lbl).mean()),
            "multiclass_brier": multiclass_brier, "ece": ece,
            "mean_confidence_predicted_class": float(conf.mean()),
            "per_phase": per_phase,
        }

    emission_0 = emission_only_eval(hmm_0, val_emb, val_lbl)
    emission_025 = emission_only_eval(hmm_025, val_emb, val_lbl)
    results["emission_only"] = {"alpha_0": emission_0, "alpha_025": emission_025}
    print(f"\n=== Emission-only ===")
    print(f"  alpha=0:    overall_acc={emission_0['overall_accuracy']:.4f} brier={emission_0['multiclass_brier']:.4f} "
          f"ece={emission_0['ece']:.4f} mean_conf={emission_0['mean_confidence_predicted_class']:.4f}")
    print(f"  alpha=0.25: overall_acc={emission_025['overall_accuracy']:.4f} brier={emission_025['multiclass_brier']:.4f} "
          f"ece={emission_025['ece']:.4f} mean_conf={emission_025['mean_confidence_predicted_class']:.4f}")
    for name in FOCUS_PHASES:
        a = emission_0["per_phase"][name]; b = emission_025["per_phase"][name]
        print(f"  {name:6s} alpha0_acc={a['accuracy']:.3f} alpha025_acc={b['accuracy']:.3f} delta={b['accuracy']-a['accuracy']:+.3f}")

    target_acc_0 = np.mean([emission_0["per_phase"][n]["accuracy"] for n in TARGET_PHASES])
    target_acc_025 = np.mean([emission_025["per_phase"][n]["accuracy"] for n in TARGET_PHASES])
    print(f"\nTarget-phase mean accuracy: alpha=0 {target_acc_0:.4f} alpha=0.25 {target_acc_025:.4f} "
          f"delta={target_acc_025-target_acc_0:+.4f} (informational only -- no gate applied this run)")
    results["target_phase_accuracy"] = {"alpha_0": float(target_acc_0), "alpha_025": float(target_acc_025)}

    # -----------------------------------------------------------------
    # k=7 propagation, unconditional
    # -----------------------------------------------------------------
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
        raw_a = {k_: np.asarray(v) for k_, v in raw_a.items()}
        raw_b = {k_: np.asarray(v) for k_, v in raw_b.items()}
        bootstrap_cmp = {}
        for metric in ("auroc", "brier", "ece"):
            if len(raw_a[metric]) == len(raw_b[metric]) and len(raw_a[metric]) > 0:
                bootstrap_cmp[metric] = paired_bootstrap_comparison(raw_a, raw_b, metric)
        return {
            "trajectory_level": {
                "n_trajectories": len(traj_delta_brier), "mean_delta_brier": float(traj_delta_brier.mean()),
                "wilcoxon_p": wilcoxon_p,
                "n_b_better": int((traj_delta_brier < 0).sum()), "n_a_better": int((traj_delta_brier > 0).sum()),
            },
            "bootstrap": bootstrap_cmp,
        }

    t0 = time.time()
    hmm_0_preds = hmm_0.predict_k_step(val_traj, k=K7)
    hmm_025_preds = hmm_025.predict_k_step(val_traj, k=K7)
    t_hmm_predict = time.time() - t0
    print(f"\nHMM predict_k_step(k=7), alpha=0 & alpha=0.25: {t_hmm_predict:.2f}s")

    yt0, yp0 = flatten_predictions(val_traj, hmm_0_preds)
    yt025, yp025 = flatten_predictions(val_traj, hmm_025_preds)
    assert np.array_equal(yt0, yt025)
    hmm_score_0 = score(yt0, yp0)
    hmm_score_025 = score(yt025, yp025)
    hmm_acc_0 = compute_metrics(val_traj, hmm_0_preds)["accuracy"]
    hmm_acc_025 = compute_metrics(val_traj, hmm_025_preds)["accuracy"]
    print(f"HMM alpha=0:    AUROC={hmm_score_0['auroc']:.5f} Brier={hmm_score_0['brier']:.5f} ECE={hmm_score_0['ece']:.5f} Acc={hmm_acc_0:.4f}")
    print(f"HMM alpha=0.25: AUROC={hmm_score_025['auroc']:.5f} Brier={hmm_score_025['brier']:.5f} ECE={hmm_score_025['ece']:.5f} Acc={hmm_acc_025:.4f}")
    results["hmm_k7"] = {"alpha_0": hmm_score_0, "alpha_025": hmm_score_025,
                          "accuracy": {"alpha_0": hmm_acc_0, "alpha_025": hmm_acc_025},
                          "predict_time_seconds": t_hmm_predict}

    hmm_phase_0 = per_phase_k7(hmm_0_preds, val_traj)
    hmm_phase_025 = per_phase_k7(hmm_025_preds, val_traj)
    print("\n=== HMM k=7 per-phase (focus): alpha=0 vs alpha=0.25 ===")
    for name in FOCUS_PHASES:
        a = hmm_phase_0.get(name); b = hmm_phase_025.get(name)
        if a is None or b is None: continue
        print(f"  {name:6s} obs={a['observed']:.3f} brier0={a['brier']:.3f} brier025={b['brier']:.3f} bias0={a['bias']:+.3f} bias025={b['bias']:+.3f}")
    results["hmm_k7_per_phase"] = {"alpha_0": hmm_phase_0, "alpha_025": hmm_phase_025}

    hmm_stats = trajectory_level_and_bootstrap(hmm_0_preds, hmm_025_preds, val_traj, seed=400)
    print(f"\nHMM alpha=0.25 vs alpha=0: mean_delta_brier={hmm_stats['trajectory_level']['mean_delta_brier']:+.5f} "
          f"wilcoxon_p={hmm_stats['trajectory_level']['wilcoxon_p']:.4f}")
    for metric, c in hmm_stats["bootstrap"].items():
        print(f"  bootstrap {metric}: mean_diff={c['mean_difference']:+.5f} CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
    results["hmm_statistical_comparison"] = hmm_stats

    # -----------------------------------------------------------------
    # Semi-HMM
    # -----------------------------------------------------------------
    t0 = time.time()
    semi_025 = SemiHMMModel(n_states=K, dmax=SEMI_DMAX, duration_family="negative_binomial",
                             logreg_class_weight=weight_dict)
    semi_025.fit(train_traj)
    t_fit_semi = time.time() - t0
    print(f"\nSemi-HMM alpha=0.25 fit: {t_fit_semi:.2f}s")
    results["semi_hmm_fit_time_seconds"] = t_fit_semi

    t0 = time.time()
    semi_0_preds = semi_0.predict_k_step(val_traj, k=K7)
    semi_025_preds = semi_025.predict_k_step(val_traj, k=K7)
    t_semi_predict = time.time() - t0
    print(f"Semi-HMM predict_k_step(k=7), alpha=0 & alpha=0.25: {t_semi_predict:.2f}s")

    yts0, yps0 = flatten_predictions(val_traj, semi_0_preds)
    yts025, yps025 = flatten_predictions(val_traj, semi_025_preds)
    assert np.array_equal(yts0, yts025)
    semi_score_0 = score(yts0, yps0)
    semi_score_025 = score(yts025, yps025)
    semi_acc_0 = compute_metrics(val_traj, semi_0_preds)["accuracy"]
    semi_acc_025 = compute_metrics(val_traj, semi_025_preds)["accuracy"]
    print(f"Semi-HMM alpha=0:    AUROC={semi_score_0['auroc']:.5f} Brier={semi_score_0['brier']:.5f} ECE={semi_score_0['ece']:.5f} Acc={semi_acc_0:.4f}")
    print(f"Semi-HMM alpha=0.25: AUROC={semi_score_025['auroc']:.5f} Brier={semi_score_025['brier']:.5f} ECE={semi_score_025['ece']:.5f} Acc={semi_acc_025:.4f}")
    results["semi_hmm_k7"] = {"alpha_0": semi_score_0, "alpha_025": semi_score_025,
                               "accuracy": {"alpha_0": semi_acc_0, "alpha_025": semi_acc_025},
                               "predict_time_seconds": t_semi_predict}

    semi_phase_0 = per_phase_k7(semi_0_preds, val_traj)
    semi_phase_025 = per_phase_k7(semi_025_preds, val_traj)
    print("\n=== Semi-HMM k=7 per-phase (focus): alpha=0 vs alpha=0.25 ===")
    for name in FOCUS_PHASES:
        a = semi_phase_0.get(name); b = semi_phase_025.get(name)
        if a is None or b is None: continue
        print(f"  {name:6s} obs={a['observed']:.3f} brier0={a['brier']:.3f} brier025={b['brier']:.3f} bias0={a['bias']:+.3f} bias025={b['bias']:+.3f}")
    results["semi_hmm_k7_per_phase"] = {"alpha_0": semi_phase_0, "alpha_025": semi_phase_025}

    semi_stats = trajectory_level_and_bootstrap(semi_0_preds, semi_025_preds, val_traj, seed=500)
    print(f"\nSemi-HMM alpha=0.25 vs alpha=0: mean_delta_brier={semi_stats['trajectory_level']['mean_delta_brier']:+.5f} "
          f"wilcoxon_p={semi_stats['trajectory_level']['wilcoxon_p']:.4f}")
    for metric, c in semi_stats["bootstrap"].items():
        print(f"  bootstrap {metric}: mean_diff={c['mean_difference']:+.5f} CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
    results["semi_hmm_statistical_comparison"] = semi_stats

    # Semi-HMM vs HMM at alpha=0.25
    semi_vs_hmm_025 = trajectory_level_and_bootstrap(hmm_025_preds, semi_025_preds, val_traj, seed=600)
    print(f"\n=== Semi-HMM vs HMM at alpha=0.25 ===")
    print(f"  mean_delta_brier(semi-hmm - hmm)={semi_vs_hmm_025['trajectory_level']['mean_delta_brier']:+.5f} "
          f"wilcoxon_p={semi_vs_hmm_025['trajectory_level']['wilcoxon_p']:.4f}")
    for metric, c in semi_vs_hmm_025["bootstrap"].items():
        print(f"  bootstrap {metric}: mean_diff={c['mean_difference']:+.5f} CI=[{c['ci_low']:+.5f},{c['ci_high']:+.5f}] wilcoxon_p={c['wilcoxon_p']:.4f}")
    results["semi_hmm_vs_hmm_alpha025"] = semi_vs_hmm_025

    # -----------------------------------------------------------------
    # persist
    # -----------------------------------------------------------------
    hmm_025.save(OUT_DIR / "hmm_alpha025")
    semi_025.save(OUT_DIR / "semi_hmm_alpha025")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nResults written to", OUT_DIR / "results.json")
    print("EMISSION_ALPHA025_DONE")
    STATUS_FILE.write_text(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("EXITCODE:0")

except Exception as e:
    import traceback
    traceback.print_exc()
    STATUS_FILE.write_text(f"FAILED {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {e}")
    print("EXITCODE:1")
    sys.exit(1)
