"""
One-off diagnostic script (not part of the production evaluation package,
not committed anywhere) -- tests whether the HMM k=7 calibration deficit
found for alpha=2.0/rho=0.5 (Results/evaluation/e1_hmm_k7_sweep/best_model/)
correlates with phase dwell-time heterogeneity already measured in
Results/evaluation/e1_hmm_phase2_analysis/report.json (Train split).

Val-only for the Brier/calibration side (loads the frozen best_model via
HMMModel.load(), never refits). Duration statistics are read as-is from the
existing Phase 2 artifact (Train split, computed 2026-08-19) -- not
recomputed, not modified. Test split never referenced.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import flatten_predictions, score_probabilistic
from evaluation.models.hmm import HMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/e1_hmm_duration_heterogeneity_diagnostic")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Val-side: per-phase Brier/calibration for the frozen k=7 winner model
# ---------------------------------------------------------------------------
val = group_into_trajectories(EmbeddingDataset("../Embeddings/resnet18", "val"))
model = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
preds = model.predict_k_step(val, k=7)
y_true_all, y_prob_all = flatten_predictions(val, preds)
overall = score_probabilistic(y_true_all, y_prob_all, n_bins=10)

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

phase_stats = {}
for name in phase_names:
    mask = per_phase_name == name
    n = int(mask.sum())
    if n == 0:
        continue
    obs = float(per_phase_true[mask].mean())
    pred_mean = float(per_phase_prob[mask].mean())
    brier = float(np.mean((per_phase_prob[mask] - per_phase_true[mask]) ** 2))
    abs_cal_err = abs(obs - pred_mean)
    signed_conf = pred_mean - obs  # >0 = overconfident on average, <0 = underconfident
    auroc = float("nan")
    if n >= 10 and len(np.unique(per_phase_true[mask])) == 2:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(per_phase_true[mask], per_phase_prob[mask]))
    phase_stats[name] = {
        "n": n, "observed": obs, "predicted": pred_mean, "brier": brier,
        "abs_calibration_error": abs_cal_err, "signed_overconfidence": signed_conf,
        "auroc": auroc,
    }

# ---------------------------------------------------------------------------
# 2. Train-side: Phase 2's own duration statistics (existing artifact, read-only)
# ---------------------------------------------------------------------------
phase2 = json.load(open("../Results/evaluation/e1_hmm_phase2_analysis/report.json"))
dist = phase2["section6_7_8_durations_distributions_censoring"]["distributions_by_phase"]
censoring = phase2["section6_7_8_durations_distributions_censoring"]["censoring_by_phase"]

duration_stats = {}
missing_duration = []
for name in phase_names:
    od = dist.get(name, {}).get("onset_duration_interior")
    if od is None or od.get("mean") is None:
        missing_duration.append(name)
        continue
    duration_stats[name] = od

print("=== Phases with NO usable interior-duration statistic (Phase 2, Train) ===")
for name in missing_duration:
    c = censoring.get(name, {})
    print(f"  {name}: n_interior_uncensored={c.get('n_interior_uncensored')}, "
          f"left_censored={c.get('n_left_censored_first_segment_of_video')}, "
          f"right_censored={c.get('n_right_censored_last_segment_of_video')}")

# ---------------------------------------------------------------------------
# 3. Merge -- only phases with BOTH Val Brier stats AND Train duration stats
# ---------------------------------------------------------------------------
usable = [name for name in phase_names if name in phase_stats and name in duration_stats]
excluded_no_duration = [name for name in phase_names if name in phase_stats and name not in duration_stats]

print()
print("=== Merged table (phase, n_val, observed, predicted, brier, abs_cal_err, "
      "auroc, n_train_interior, mean_dur, median_dur, std_dur, cv) ===")
rows = []
for name in phase_names:
    ps = phase_stats.get(name)
    ds = duration_stats.get(name)
    if ps is None:
        continue
    if ds is None:
        print(f"{name:6s} n_val={ps['n']:5d} obs={ps['observed']:.4f} pred={ps['predicted']:.4f} "
              f"brier={ps['brier']:.4f} abs_err={ps['abs_calibration_error']:.4f} "
              f"auroc={ps['auroc']:.4f} | NO DURATION DATA (excluded from correlation)")
        continue
    row = {
        "phase": name, "n_val": ps["n"], "observed": ps["observed"], "predicted": ps["predicted"],
        "brier": ps["brier"], "abs_calibration_error": ps["abs_calibration_error"],
        "signed_overconfidence": ps["signed_overconfidence"], "auroc": ps["auroc"],
        "n_train_interior": ds["n"], "mean_duration": ds["mean"], "median_duration": ds["median"],
        "std_duration": ds["std"], "cv": ds["cv_std_over_mean"],
    }
    rows.append(row)
    print(f"{name:6s} n_val={ps['n']:5d} obs={ps['observed']:.4f} pred={ps['predicted']:.4f} "
          f"brier={ps['brier']:.4f} abs_err={ps['abs_calibration_error']:.4f} "
          f"auroc={ps['auroc']:.4f} | n_train={ds['n']:4d} mean_dur={ds['mean']:.3f} "
          f"median_dur={ds['median']:.3f} std={ds['std']:.3f} cv={ds['cv_std_over_mean']:.4f}")

print()
print(f"n phases with Val Brier stats: {len(phase_stats)}")
print(f"n phases usable for duration correlation: {len(rows)}")
print(f"excluded (no duration data): {excluded_no_duration}")

# ---------------------------------------------------------------------------
# 4. Correlations (Pearson + Spearman), with robustness drops
# ---------------------------------------------------------------------------
def corr_report(x, y, label, names):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        print(f"  {label}: n={n} too small for a meaningful correlation")
        return
    pear_r, pear_p = stats.pearsonr(x, y)
    spear_r, spear_p = stats.spearmanr(x, y)
    # Fisher z CI for Pearson (large-sample approximation; explicitly flagged
    # as approximate given n<15 in every case here)
    ci_str = "n/a (n too small for a reliable CI)"
    if n >= 5 and abs(pear_r) < 1.0:
        z = np.arctanh(pear_r)
        se = 1.0 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        ci_str = f"[{lo:.3f}, {hi:.3f}] (Fisher z, approximate, n={n})"
    print(f"  {label}: n={n}")
    print(f"    Pearson  r={pear_r:.4f}  p={pear_p:.4f}  95% CI~{ci_str}")
    print(f"    Spearman rho={spear_r:.4f}  p={spear_p:.4f}")
    return {"n": n, "pearson_r": pear_r, "pearson_p": pear_p, "spearman_rho": spear_r,
            "spearman_p": spear_p, "pearson_ci_approx": ci_str, "phases": list(names)}

names_all = [r["phase"] for r in rows]
brier_all = [r["brier"] for r in rows]
cv_all = [r["cv"] for r in rows]
mean_dur_all = [r["mean_duration"] for r in rows]
median_dur_all = [r["median_duration"] for r in rows]
abs_err_all = [r["abs_calibration_error"] for r in rows]
obs_all = [r["observed"] for r in rows]
n_val_all = [r["n_val"] for r in rows]

correlations = {}
print()
print("=== CORRELATIONS (all", len(rows), "usable phases) ===")
correlations["brier_vs_cv_all"] = corr_report(cv_all, brier_all, "Brier vs CV (all)", names_all)
correlations["brier_vs_mean_duration_all"] = corr_report(mean_dur_all, brier_all, "Brier vs mean_duration (all)", names_all)
correlations["brier_vs_median_duration_all"] = corr_report(median_dur_all, brier_all, "Brier vs median_duration (all)", names_all)
correlations["abserr_vs_cv_all"] = corr_report(cv_all, abs_err_all, "|pred-obs| vs CV (all)", names_all)
correlations["abserr_vs_mean_duration_all"] = corr_report(mean_dur_all, abs_err_all, "|pred-obs| vs mean_duration (all)", names_all)
correlations["brier_vs_observed_transition_all"] = corr_report(obs_all, brier_all, "Brier vs observed transition rate (all)", names_all)
correlations["brier_vs_n_val_all"] = corr_report(n_val_all, brier_all, "Brier vs n (Val window count) (all)", names_all)

print()
print("=== ROBUSTNESS: Brier vs CV, dropping tPNf/t3 ===")
def drop(names_to_drop):
    keep_idx = [i for i, n in enumerate(names_all) if n not in names_to_drop]
    return ([cv_all[i] for i in keep_idx], [brier_all[i] for i in keep_idx], [names_all[i] for i in keep_idx])

for drop_set, label in [(set(), "A: all phases"), ({"tPNf"}, "B: without tPNf"),
                          ({"t3"}, "C: without t3"), ({"tPNf", "t3"}, "D: without tPNf AND t3")]:
    cvs, briers, kept_names = drop(drop_set)
    r = corr_report(cvs, briers, f"Brier vs CV ({label})", kept_names)
    correlations[f"brier_vs_cv_{label.split(':')[0]}"] = r

# ---------------------------------------------------------------------------
# 5. Group analysis by CV tertile (quantile-based, defined before looking at Brier)
# ---------------------------------------------------------------------------
cv_arr = np.array(cv_all)
brier_arr = np.array(brier_all)
obs_arr = np.array(obs_all)
pred_arr = np.array([r["predicted"] for r in rows])
q1, q2 = np.quantile(cv_arr, [1/3, 2/3])
print()
print(f"=== CV tertile thresholds (data-defined quantiles, n={len(rows)}): q1={q1:.4f}, q2={q2:.4f} ===")
groups = {"low_cv": cv_arr <= q1, "mid_cv": (cv_arr > q1) & (cv_arr <= q2), "high_cv": cv_arr > q2}
group_summary = {}
for g, mask in groups.items():
    members = [names_all[i] for i in range(len(names_all)) if mask[i]]
    group_summary[g] = {
        "phases": members, "n_phases": int(mask.sum()),
        "mean_brier": float(brier_arr[mask].mean()) if mask.sum() else float("nan"),
        "mean_abs_calibration_error": float(np.mean(np.array(abs_err_all)[mask])) if mask.sum() else float("nan"),
        "mean_observed": float(obs_arr[mask].mean()) if mask.sum() else float("nan"),
        "mean_predicted": float(pred_arr[mask].mean()) if mask.sum() else float("nan"),
    }
    print(f"  {g}: phases={members}")
    print(f"    mean_brier={group_summary[g]['mean_brier']:.4f} "
          f"mean_abs_cal_err={group_summary[g]['mean_abs_calibration_error']:.4f} "
          f"mean_observed={group_summary[g]['mean_observed']:.4f} "
          f"mean_predicted={group_summary[g]['mean_predicted']:.4f}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def scatter_with_labels(x, y, names, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    for xi, yi, ni in zip(x, y, names):
        ax.annotate(ni, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)

scatter_with_labels(cv_all, brier_all, names_all, "Duration CV (Train, interior occurrences)", "Brier (Val, k=7)",
                     "Duration CV vs per-phase Brier", "cv_vs_brier.png")
scatter_with_labels(mean_dur_all, brier_all, names_all, "Mean duration (Train, raw units)", "Brier (Val, k=7)",
                     "Mean duration vs per-phase Brier", "mean_duration_vs_brier.png")
scatter_with_labels(obs_all, [r["predicted"] for r in rows], names_all, "Observed transition rate (Val)",
                     "Mean predicted probability (Val)", "Observed vs predicted transition rate, per phase",
                     "observed_vs_predicted.png")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(rows))
ax.bar(x - 0.2, obs_all, width=0.4, label="observed")
ax.bar(x + 0.2, [r["predicted"] for r in rows], width=0.4, label="predicted")
ax.set_xticks(x)
ax.set_xticklabels(names_all, rotation=45)
ax.set_ylabel("transition rate")
ax.set_title("Per-phase observed vs predicted transition rate (Val, k=7, alpha=2.0/rho=0.5)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "calibration_by_phase_bars.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Persist everything
# ---------------------------------------------------------------------------
report = {
    "model_used": "Results/evaluation/e1_hmm_k7_sweep/best_model (alpha=2.0, rho=0.5), loaded read-only, not refit",
    "overall_val_k7": {k: overall[k] for k in ("auroc", "brier", "brier_baseline_constant_rate", "ece", "n_windows")},
    "duration_source": "Results/evaluation/e1_hmm_phase2_analysis/report.json (Train split, computed 2026-08-19), read-only",
    "phases_excluded_no_duration_data": missing_duration,
    "rows": rows,
    "correlations": correlations,
    "cv_tertile_thresholds": {"q1": float(q1), "q2": float(q2)},
    "group_summary": group_summary,
}
(OUT_DIR / "duration_heterogeneity_report.json").write_text(json.dumps(report, indent=2))
print()
print("Report written to", OUT_DIR / "duration_heterogeneity_report.json")
print("Figures written to", FIG_DIR)
