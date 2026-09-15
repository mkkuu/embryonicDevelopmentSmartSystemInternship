"""
Emission-weighted sweep experiment (explicitly authorized 2026-08-24).

Follows up on emission_balanced (Results/evaluation/emission_balanced/),
which found class_weight='balanced' is Case D: it fixes minority-phase
emission accuracy (gate passed, +0.136 mean target-phase accuracy) but
significantly DEGRADES HMM/Semi-HMM k=7 calibration and ranking (AUROC
-0.08/-0.09, Brier +0.014/+0.015, ECE +0.019/+0.021, all p<1e-4). This
experiment tests whether a MODERATE, partial rebalancing recovers some of
the minority-phase gain without the calibration collapse.

WEIGHTING FORMULATION
----------------------
w_c(alpha) = (N / (K * N_c)) ** alpha,  alpha in {0, 0.25, 0.5, 0.75, 1.0}

N = total Train windows, K = 15 (n_states), N_c = Train window count of
class c. This is NOT the naive (N/N_c)^alpha the plan sketch proposed --
it includes sklearn's own 1/K normalization factor, chosen deliberately so
that:
  - alpha=0  -> w_c=1 for all c  -> EXACTLY class_weight=None (the
    already-frozen original models, reused via .load(), never refit).
  - alpha=1  -> w_c = N/(K*N_c) for all c -> EXACTLY sklearn's own
    class_weight='balanced' formula (sklearn.utils.class_weight.
    compute_class_weight) -- bit-for-bit the already-frozen
    emission_balanced experiment, reused via .load(), never refit.
  - alpha in (0,1) -> geometric interpolation between the two, in
    log-weight space (log w_c is linear in alpha).
This is the "equivalent, code-compatible alternative" required before
launching: a literal (N/N_c)^alpha would NOT reduce to the two
already-established endpoints, breaking direct comparability and forcing
a wasteful re-run of the two already-known extremes. With this
1/K-normalized form, only the 3 genuinely new intermediate alphas
(0.25, 0.5, 0.75) require a fresh LogisticRegression fit -- alpha=0 and
alpha=1 are pulled from already-frozen artifacts (read-only .load()),
consistent with "ne pas tester une multitude de parametres inutilement."
`class_weight` is passed to sklearn's LogisticRegression as a plain
{int: float} dict, which HMMModel/SemiHMMModel already forward as-is
(logreg_class_weight typed Optional[str] only as documentation; not
runtime-enforced) -- verified by reading Training/evaluation/models/
{hmm,semi_hmm}.py before writing this script. No source code changes
needed.

PRE-REGISTERED GATE (decided BEFORE seeing any of this run's results)
------------------------------------------------------------------------
For each alpha in {0.25, 0.5, 0.75, 1.0}, computed on the EMISSION-ONLY
Val evaluation, relative to the alpha=0 baseline:
  G1 (signal):      mean target-phase accuracy improves by >= +0.05
                     absolute (tPNf/t3/t5/t6/tB) -- same threshold as the
                     emission_balanced experiment's own gate, established
                     precedent, not re-tuned here.
  G2 (calibration):  emission Brier(alpha) <= 1.15 * Brier(0)
                     AND emission ECE(alpha) <= 1.15 * ECE(0)
                     -- chosen to be TIGHTER than the ~11-17% relative
                     degradation already observed at the DOWNSTREAM HMM/
                     Semi-HMM k=7 level for alpha=1 in emission_balanced
                     (Brier +10.9%, ECE +16.6% relative), so a candidate
                     passing G2 at the emission level is a priori less
                     likely to reproduce that same failure mode once
                     propagated through the temporal models.
  G3 (no collapse):  overall accuracy(alpha) >= overall accuracy(0) - 0.15
                     absolute -- catastrophic-collapse-only threshold;
                     deliberately loose because emission_balanced already
                     showed overall accuracy dropping without being the
                     dominant failure mode (calibration was) -- G3 is not
                     meant to be the binding constraint.
A candidate proceeds to HMM/Semi-HMM k=7 evaluation ONLY if G1 AND G2 AND
G3 all hold. alpha=1's k=7 numbers are always recomputed here too (cheap:
models already frozen/fitted, only a fresh predict_k_step pass, no refit)
so the final comparison table is self-consistent (identical code path)
across all 5 alpha points, regardless of alpha=1's gate outcome.

STRICT CONTROLS (only class_weight/alpha varies)
---------------------------------------------------
  - embeddings: Embeddings/resnet18/{train,val}, untouched, read-only.
  - Train (492 traj) / Val (106 traj): identical splits.
  - HMM transition: alpha=2.0, rho=0.5 (frozen e1_hmm_k7_sweep winner).
  - Semi-HMM transition/duration: class defaults (alpha=1.0, rho=0.3),
    dmax=268, duration_family='negative_binomial' (Phase F's exact call).
  - logreg_C=1.0, logreg_max_iter=1000: unchanged.
  - k=7 throughout (window_size-1), matching every prior experiment.
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

OUT_DIR = Path("../Results/evaluation/emission_weighted_sweep")
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
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GATE_TARGET_DELTA = 0.05
GATE_CALIBRATION_RATIO = 1.15
GATE_COLLAPSE_ABS = 0.15

results = {"protocol": {
    "target_phases": TARGET_PHASES, "reference_phases": REFERENCE_PHASES,
    "hmm_alpha": HMM_ALPHA, "hmm_rho": HMM_RHO, "semi_dmax": SEMI_DMAX,
    "semi_duration_family": "negative_binomial", "alphas": ALPHAS,
    "weighting_formula": "w_c(alpha) = (N / (K * N_c)) ** alpha",
    "gate": {
        "target_delta_threshold": GATE_TARGET_DELTA,
        "calibration_ratio_threshold": GATE_CALIBRATION_RATIO,
        "collapse_abs_threshold": GATE_COLLAPSE_ABS,
    },
    "note": "alpha=0 and alpha=1 reuse frozen models (.load(), never refit); "
            "only alpha in {0.25,0.5,0.75} fit fresh; Test never loaded",
}}

try:
    t0 = time.time()
    train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
    val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
    t_load = time.time() - t0
    print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")
    results["load_time_seconds"] = t_load

    # -----------------------------------------------------------------
    # class weight dicts w_c(alpha) from TRAIN window counts
    # -----------------------------------------------------------------
    all_train_labels = torch.cat([t.last_frame_phase for t in train_traj if len(t) > 0]).numpy()
    N_total = len(all_train_labels)
    counts = np.array([(all_train_labels == c).sum() for c in range(K)], dtype=np.float64)
    print(f"\nTrain window counts per phase (N_total={N_total}):")
    for name, c in zip(PHASE_NAMES, counts):
        print(f"  {name:6s} n={int(c)}")
    results["train_class_counts"] = {name: int(c) for name, c in zip(PHASE_NAMES, counts)}

    def weight_dict(alpha):
        if alpha == 0.0:
            return None
        base = N_total / (K * counts)
        w = base ** alpha
        return {c: float(w[c]) for c in range(K)}

    for a in ALPHAS:
        wd = weight_dict(a)
        if wd is not None:
            print(f"alpha={a}: min_w={min(wd.values()):.4f} max_w={max(wd.values()):.4f} "
                  f"w[t9+]={wd[PHASE_NAMES.index('t9+')]:.4f} w[t3]={wd[PHASE_NAMES.index('t3')]:.4f}")

    # -----------------------------------------------------------------
    # load FROZEN alpha=0 / alpha=1 models (never refit)
    # -----------------------------------------------------------------
    hmm_models = {}
    semi_models = {}
    hmm_models[0.0] = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
    semi_models[0.0] = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
    hmm_models[1.0] = HMMModel.load(Path("../Results/evaluation/emission_balanced/hmm_balanced_model"))
    semi_models[1.0] = SemiHMMModel.load(Path("../Results/evaluation/emission_balanced/semi_hmm_balanced_model"))
    assert hmm_models[0.0].logreg_class_weight is None
    assert semi_models[0.0].logreg_class_weight is None
    assert hmm_models[1.0].logreg_class_weight == "balanced"
    assert semi_models[1.0].logreg_class_weight == "balanced"
    print("\nLoaded frozen alpha=0 (None) and alpha=1 ('balanced') HMM/Semi-HMM models.")

    # -----------------------------------------------------------------
    # fit fresh HMM/Semi-HMM for intermediate alphas only
    # -----------------------------------------------------------------
    fit_times = {}
    for a in (0.25, 0.5, 0.75):
        wd = weight_dict(a)
        t0 = time.time()
        m = HMMModel(n_states=K, transition_smoothing_alpha=HMM_ALPHA, transition_decay_rho=HMM_RHO,
                     logreg_class_weight=wd)
        m.fit(train_traj)
        dt = time.time() - t0
        hmm_models[a] = m
        fit_times[f"hmm_alpha_{a}_fit_seconds"] = dt
        print(f"HMM alpha={a} fit: {dt:.2f}s")
    results["fit_times"] = fit_times

    # -----------------------------------------------------------------
    # emission-only evaluation on Val, all 5 alphas, identical function
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
            "multiclass_brier": multiclass_brier,
            "ece": ece,
            "mean_confidence_predicted_class": float(conf.mean()),
            "per_phase": per_phase,
        }

    emission_eval = {a: emission_only_eval(hmm_models[a], val_emb, val_lbl) for a in ALPHAS}
    results["emission_only"] = {str(a): emission_eval[a] for a in ALPHAS}

    print("\n=== Emission-only, all alphas ===")
    for a in ALPHAS:
        e = emission_eval[a]
        print(f"  alpha={a:<4} overall_acc={e['overall_accuracy']:.4f} brier={e['multiclass_brier']:.4f} "
              f"ece={e['ece']:.4f} mean_conf={e['mean_confidence_predicted_class']:.4f}")
    print("\n=== Per-phase emission accuracy, all alphas ===")
    for name in FOCUS_PHASES:
        row = " ".join(f"a={a}:{emission_eval[a]['per_phase'][name]['accuracy']:.3f}" for a in ALPHAS)
        print(f"  {name:6s} {row}")

    # -----------------------------------------------------------------
    # gate check, per pre-registered thresholds
    # -----------------------------------------------------------------
    target_acc = {a: np.mean([emission_eval[a]["per_phase"][n]["accuracy"] for n in TARGET_PHASES]) for a in ALPHAS}
    gate = {}
    for a in (0.25, 0.5, 0.75, 1.0):
        delta = target_acc[a] - target_acc[0.0]
        g1 = bool(delta >= GATE_TARGET_DELTA)
        g2 = bool(
            emission_eval[a]["multiclass_brier"] <= GATE_CALIBRATION_RATIO * emission_eval[0.0]["multiclass_brier"]
            and emission_eval[a]["ece"] <= GATE_CALIBRATION_RATIO * emission_eval[0.0]["ece"]
        )
        g3 = bool(emission_eval[a]["overall_accuracy"] >= emission_eval[0.0]["overall_accuracy"] - GATE_COLLAPSE_ABS)
        gate[a] = {
            "target_acc_delta": float(delta), "g1_signal": g1,
            "brier_ratio": float(emission_eval[a]["multiclass_brier"] / emission_eval[0.0]["multiclass_brier"]),
            "ece_ratio": float(emission_eval[a]["ece"] / emission_eval[0.0]["ece"]),
            "g2_calibration": g2,
            "overall_acc_drop": float(emission_eval[0.0]["overall_accuracy"] - emission_eval[a]["overall_accuracy"]),
            "g3_no_collapse": g3,
            "gate_pass": bool(g1 and g2 and g3),
        }
    results["gate_check"] = {str(a): gate[a] for a in gate}
    print("\n=== Gate check ===")
    for a in (0.25, 0.5, 0.75, 1.0):
        g = gate[a]
        print(f"  alpha={a}: delta_target={g['target_acc_delta']:+.4f}(G1={g['g1_signal']}) "
              f"brier_ratio={g['brier_ratio']:.3f} ece_ratio={g['ece_ratio']:.3f}(G2={g['g2_calibration']}) "
              f"acc_drop={g['overall_acc_drop']:+.4f}(G3={g['g3_no_collapse']}) -> PASS={g['gate_pass']}")

    candidates_for_k7 = [a for a in (0.25, 0.5, 0.75) if gate[a]["gate_pass"]]
    print(f"\nIntermediate alphas proceeding to HMM/Semi-HMM k=7: {candidates_for_k7}")
    results["candidates_for_k7"] = candidates_for_k7

    # -----------------------------------------------------------------
    # k=7 evaluation: alpha=0 and alpha=1 ALWAYS (frozen, predict-only,
    # no refit), gate-passing intermediates additionally
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
                "n_b_better": int((traj_delta_brier < 0).sum()),
                "n_a_better": int((traj_delta_brier > 0).sum()),
            },
            "bootstrap": bootstrap_cmp,
        }

    k7_alphas = [0.0, 1.0] + candidates_for_k7
    hmm_preds = {}
    semi_preds = {}
    k7_scores = {"hmm": {}, "semi_hmm": {}}
    k7_per_phase = {"hmm": {}, "semi_hmm": {}}
    k7_accuracy = {"hmm": {}, "semi_hmm": {}}

    for a in k7_alphas:
        t0 = time.time()
        hp = hmm_models[a].predict_k_step(val_traj, k=K7)
        dt_hmm = time.time() - t0
        hmm_preds[a] = hp
        yt, yp = flatten_predictions(val_traj, hp)
        k7_scores["hmm"][a] = score(yt, yp)
        k7_per_phase["hmm"][a] = per_phase_k7(hp, val_traj)
        k7_accuracy["hmm"][a] = compute_metrics(val_traj, hp)["accuracy"]
        print(f"HMM alpha={a} k=7 predict: {dt_hmm:.2f}s -> AUROC={k7_scores['hmm'][a]['auroc']:.5f} "
              f"Brier={k7_scores['hmm'][a]['brier']:.5f} ECE={k7_scores['hmm'][a]['ece']:.5f}")

        # Semi-HMM: alpha=0/1 already fitted (frozen, loaded) -> predict only.
        # Intermediate alphas: needs its own fit (composes its own emission).
        if a not in semi_models:
            t0 = time.time()
            wd = weight_dict(a)
            sm = SemiHMMModel(n_states=K, dmax=SEMI_DMAX, duration_family="negative_binomial",
                               logreg_class_weight=wd)
            sm.fit(train_traj)
            dt_fit = time.time() - t0
            semi_models[a] = sm
            fit_times[f"semi_hmm_alpha_{a}_fit_seconds"] = dt_fit
            print(f"Semi-HMM alpha={a} fit: {dt_fit:.2f}s")

        t0 = time.time()
        sp = semi_models[a].predict_k_step(val_traj, k=K7)
        dt_semi = time.time() - t0
        semi_preds[a] = sp
        yt, yp = flatten_predictions(val_traj, sp)
        k7_scores["semi_hmm"][a] = score(yt, yp)
        k7_per_phase["semi_hmm"][a] = per_phase_k7(sp, val_traj)
        k7_accuracy["semi_hmm"][a] = compute_metrics(val_traj, sp)["accuracy"]
        print(f"Semi-HMM alpha={a} k=7 predict: {dt_semi:.2f}s -> AUROC={k7_scores['semi_hmm'][a]['auroc']:.5f} "
              f"Brier={k7_scores['semi_hmm'][a]['brier']:.5f} ECE={k7_scores['semi_hmm'][a]['ece']:.5f}")

    results["hmm_k7"] = {str(a): k7_scores["hmm"][a] for a in k7_alphas}
    results["semi_hmm_k7"] = {str(a): k7_scores["semi_hmm"][a] for a in k7_alphas}
    results["hmm_k7_accuracy"] = {str(a): k7_accuracy["hmm"][a] for a in k7_alphas}
    results["semi_hmm_k7_accuracy"] = {str(a): k7_accuracy["semi_hmm"][a] for a in k7_alphas}
    results["hmm_k7_per_phase"] = {str(a): k7_per_phase["hmm"][a] for a in k7_alphas}
    results["semi_hmm_k7_per_phase"] = {str(a): k7_per_phase["semi_hmm"][a] for a in k7_alphas}

    # statistical comparisons vs alpha=0 baseline, for every OTHER alpha we ran k7 for
    stat_cmp = {"hmm": {}, "semi_hmm": {}}
    for a in k7_alphas:
        if a == 0.0:
            continue
        stat_cmp["hmm"][a] = trajectory_level_and_bootstrap(hmm_preds[0.0], hmm_preds[a], val_traj, seed=int(1000 + a * 100))
        stat_cmp["semi_hmm"][a] = trajectory_level_and_bootstrap(semi_preds[0.0], semi_preds[a], val_traj, seed=int(2000 + a * 100))
        print(f"\n=== alpha={a} vs alpha=0 baseline ===")
        for model_name in ("hmm", "semi_hmm"):
            c = stat_cmp[model_name][a]
            print(f"  {model_name}: mean_delta_brier={c['trajectory_level']['mean_delta_brier']:+.5f} "
                  f"wilcoxon_p={c['trajectory_level']['wilcoxon_p']:.4f}")
            for metric, bc in c["bootstrap"].items():
                print(f"    bootstrap {metric}: mean_diff={bc['mean_difference']:+.5f} "
                      f"CI=[{bc['ci_low']:+.5f},{bc['ci_high']:+.5f}] wilcoxon_p={bc['wilcoxon_p']:.4f}")
    results["statistical_comparison_vs_alpha0"] = {
        "hmm": {str(a): stat_cmp["hmm"][a] for a in stat_cmp["hmm"]},
        "semi_hmm": {str(a): stat_cmp["semi_hmm"][a] for a in stat_cmp["semi_hmm"]},
    }

    # Semi-HMM vs HMM at each alpha we actually ran (does Semi-HMM still add value?)
    semi_vs_hmm = {}
    for a in k7_alphas:
        semi_vs_hmm[a] = trajectory_level_and_bootstrap(hmm_preds[a], semi_preds[a], val_traj, seed=int(3000 + a * 100))
        c = semi_vs_hmm[a]
        print(f"\n=== Semi-HMM vs HMM at alpha={a} ===")
        print(f"  mean_delta_brier(semi-hmm - hmm)={c['trajectory_level']['mean_delta_brier']:+.5f} "
              f"wilcoxon_p={c['trajectory_level']['wilcoxon_p']:.4f}")
        for metric, bc in c["bootstrap"].items():
            print(f"    bootstrap {metric}: mean_diff={bc['mean_difference']:+.5f} "
                  f"CI=[{bc['ci_low']:+.5f},{bc['ci_high']:+.5f}] wilcoxon_p={bc['wilcoxon_p']:.4f}")
    results["semi_hmm_vs_hmm"] = {str(a): semi_vs_hmm[a] for a in semi_vs_hmm}

    # -----------------------------------------------------------------
    # model selection: best candidate among gate-passing intermediates
    # -----------------------------------------------------------------
    selection = {"gate_passing_intermediates": candidates_for_k7, "reference_points": {"alpha_0": "None (original)", "alpha_1": "balanced (Case D, already rejected)"}}
    if candidates_for_k7:
        def combined_score(a):
            # simple, documented post-hoc ranking (NOT the pre-registered gate):
            # mean AUROC (HMM,Semi-HMM) minus mean relative ECE increase vs alpha=0.
            mean_auroc = np.mean([k7_scores["hmm"][a]["auroc"], k7_scores["semi_hmm"][a]["auroc"]])
            mean_ece_ratio = np.mean([
                k7_scores["hmm"][a]["ece"] / k7_scores["hmm"][0.0]["ece"],
                k7_scores["semi_hmm"][a]["ece"] / k7_scores["semi_hmm"][0.0]["ece"],
            ])
            return mean_auroc - (mean_ece_ratio - 1.0)
        ranked = sorted(candidates_for_k7, key=combined_score, reverse=True)
        selection["ranked_candidates"] = ranked
        selection["best_alpha"] = ranked[0]
        selection["selection_method"] = "post-hoc: mean(HMM,Semi-HMM AUROC) - mean(relative ECE increase vs alpha=0); NOT pre-registered, informational ranking only among gate-passing candidates"
    else:
        selection["best_alpha"] = None
        selection["selection_method"] = "no intermediate alpha passed the pre-registered gate"
    results["model_selection"] = selection
    print("\n=== Model selection ===")
    print(json.dumps(selection, indent=2))

    # -----------------------------------------------------------------
    # persist
    # -----------------------------------------------------------------
    for a in candidates_for_k7:
        tag = str(a).replace(".", "p")
        hmm_models[a].save(OUT_DIR / f"hmm_alpha_{tag}")
        semi_models[a].save(OUT_DIR / f"semi_hmm_alpha_{tag}")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2))
    print("\nResults written to", OUT_DIR / "results.json")
    print("EMISSION_WEIGHTED_SWEEP_DONE")
    STATUS_FILE.write_text(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("EXITCODE:0")

except Exception as e:
    import traceback
    traceback.print_exc()
    STATUS_FILE.write_text(f"FAILED {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {e}")
    print("EXITCODE:1")
    sys.exit(1)
