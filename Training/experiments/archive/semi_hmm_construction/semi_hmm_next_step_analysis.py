"""
Next-step scientific analysis for the Semi-HMM branch, AFTER Phase F.
One-off, read-only diagnostic (not committed, same convention as prior
phase scripts). NO new model is trained here -- only the two ALREADY
FITTED, frozen models (HMM e1_hmm_k7_sweep/best_model, Semi-HMM
semi_hmm_weekend_phaseF/model) are loaded and queried. Test split is
NEVER loaded, never referenced.

Produces:
  - per-trajectory paired Brier comparison (Wilcoxon signed-rank, sign test)
  - patient-level bootstrap paired comparison (AUROC/Brier/ECE), reusing
    evaluation.metrics.paired_bootstrap_comparison for the actual stats
  - Murphy/Brier decomposition (uncertainty/reliability/resolution) for
    BOTH models, to explain why Brier stays above baseline
  - duration correlation in the CORRECT units (window-count segment
    duration, Phase E's own convention -- the SAME unit the Semi-HMM
    duration model actually operates in) vs BOTH models' per-phase Brier
    -- a genuinely independent check of the E1/HMM diagnostic's
    r=-0.7729 finding, which used a DIFFERENT duration definition
    (real-time "time-to-next-onset", metadata_with_time.csv raw units)
  - concrete illustrative outputs of next_phase_distribution() /
    duration_distribution() / transition_chain() for qualitative
    reporting value
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, brier_score_loss

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import calibration_bins, expected_calibration_error, flatten_predictions
from evaluation.metrics import compute_metrics, paired_bootstrap_comparison
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/semi_hmm_next_step_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
K = 7
report = {"protocol": {"k_headline": K, "note": "Val only, Test never loaded"}}

# ---------------------------------------------------------------------------
# 0. Load frozen artifacts (read-only, nothing refit)
# ---------------------------------------------------------------------------
t0 = time.time()
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
hmm = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
semi = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_val={len(val_traj)}")
report["load_time_seconds"] = t_load
report["n_val_trajectories"] = len(val_traj)

phase_e = json.loads(Path("../Results/evaluation/semi_hmm_weekend_phaseE/dmax_duration_diagnostic_report.json").read_text())

t0 = time.time()
hmm_preds = hmm.predict_k_step(val_traj, k=K)
t_hmm = time.time() - t0
print(f"HMM predict_k_step(k={K}): {t_hmm:.2f}s")

t0 = time.time()
semi_preds = semi.predict_k_step(val_traj, k=K)
t_semi = time.time() - t0
print(f"Semi-HMM predict_k_step(k={K}): {t_semi:.2f}s")
report["hmm_predict_time_seconds"] = t_hmm
report["semi_hmm_predict_time_seconds"] = t_semi

phase_names = list(DEFAULT_PHASE_NAMES)

# ---------------------------------------------------------------------------
# 1. Pooled-window sanity check (must reproduce Phase F's headline numbers)
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
        "bins": bins,
        "observed_positive_rate": float(y_true.mean()),
    }

y_true_hmm, y_prob_hmm = flatten_predictions(val_traj, hmm_preds)
y_true_semi, y_prob_semi = flatten_predictions(val_traj, semi_preds)
assert np.array_equal(y_true_hmm, y_true_semi), "y_true must match between models (same Val, same windows)"

score_hmm = score(y_true_hmm, y_prob_hmm)
score_semi = score(y_true_semi, y_prob_semi)
print()
print(f"[sanity] HMM      k=7: AUROC={score_hmm['auroc']:.5f} Brier={score_hmm['brier']:.5f} ECE={score_hmm['ece']:.5f}")
print(f"[sanity] Semi-HMM k=7: AUROC={score_semi['auroc']:.5f} Brier={score_semi['brier']:.5f} ECE={score_semi['ece']:.5f}")
report["pooled_window_level"] = {
    "hmm": {k: v for k, v in score_hmm.items() if k != "bins"},
    "semi_hmm": {k: v for k, v in score_semi.items() if k != "bins"},
}

# ---------------------------------------------------------------------------
# 2. Per-trajectory paired comparison (trajectory = statistical unit)
# ---------------------------------------------------------------------------
traj_rows = []
for traj, ph, ps in zip(val_traj, hmm_preds, semi_preds):
    yt = traj.consistency_flag.numpy()
    yh = ph.consistency_flag_prob.numpy()
    ysm = ps.consistency_flag_prob.numpy()
    if len(yt) == 0:
        continue
    brier_h = float(np.mean((yh - yt) ** 2))
    brier_s = float(np.mean((ysm - yt) ** 2))
    row = {
        "video_name": traj.video_name, "n_windows": len(yt),
        "brier_hmm": brier_h, "brier_semi_hmm": brier_s, "delta_brier": brier_s - brier_h,
        "observed_rate": float(yt.mean()),
    }
    if len(np.unique(yt)) == 2:
        row["auroc_hmm"] = float(roc_auc_score(yt, yh))
        row["auroc_semi_hmm"] = float(roc_auc_score(yt, ysm))
        row["delta_auroc"] = row["auroc_semi_hmm"] - row["auroc_hmm"]
    traj_rows.append(row)

deltas_brier = np.array([r["delta_brier"] for r in traj_rows])
n_traj = len(deltas_brier)
n_better = int((deltas_brier < 0).sum())  # Semi-HMM lower Brier = better
n_worse = int((deltas_brier > 0).sum())
n_tied = n_traj - n_better - n_worse
wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(deltas_brier) if not np.allclose(deltas_brier, 0) else (0.0, 1.0)
sign_test_p = float(scipy_stats.binomtest(n_better, n_better + n_worse, p=0.5).pvalue) if (n_better + n_worse) > 0 else float("nan")
ttest_stat, ttest_p = scipy_stats.ttest_1samp(deltas_brier, 0.0)

trajectory_level = {
    "n_trajectories": n_traj,
    "mean_delta_brier": float(deltas_brier.mean()),
    "std_delta_brier": float(deltas_brier.std(ddof=1)),
    "median_delta_brier": float(np.median(deltas_brier)),
    "n_trajectories_semi_hmm_better": n_better,
    "n_trajectories_hmm_better": n_worse,
    "n_trajectories_tied": n_tied,
    "wilcoxon_signed_rank_p": float(wilcoxon_p),
    "sign_test_p": sign_test_p,
    "paired_ttest_p": float(ttest_p),
    "rows": traj_rows,
}
report["trajectory_level_comparison"] = trajectory_level
print()
print(f"=== Per-trajectory paired Brier comparison (n={n_traj}) ===")
print(f"  mean delta (Semi-HMM - HMM) = {deltas_brier.mean():+.5f} (std={deltas_brier.std(ddof=1):.5f})")
print(f"  Semi-HMM better: {n_better}/{n_traj}, HMM better: {n_worse}/{n_traj}, tied: {n_tied}")
print(f"  Wilcoxon signed-rank p={wilcoxon_p:.4f}, sign test p={sign_test_p:.4f}, paired t-test p={ttest_p:.4f}")

# ---------------------------------------------------------------------------
# 3. Patient-level bootstrap (precompute-once, resample indices -- avoids
#    1000x re-running forward()), paired via evaluation.metrics.paired_bootstrap_comparison
# ---------------------------------------------------------------------------
per_traj_yt = [traj.consistency_flag.numpy() for traj in val_traj]
per_traj_yh = [p.consistency_flag_prob.numpy() for p in hmm_preds]
per_traj_ys = [p.consistency_flag_prob.numpy() for p in semi_preds]
n = len(val_traj)
n_bootstrap = 1000
rng = np.random.RandomState(0)

raw_hmm = {"auroc": [], "brier": [], "ece": []}
raw_semi = {"auroc": [], "brier": [], "ece": []}
for _ in range(n_bootstrap):
    idx = rng.randint(0, n, size=n)
    yt = np.concatenate([per_traj_yt[i] for i in idx])
    yh = np.concatenate([per_traj_yh[i] for i in idx])
    ys = np.concatenate([per_traj_ys[i] for i in idx])
    if len(np.unique(yt)) == 2:
        raw_hmm["auroc"].append(roc_auc_score(yt, yh))
        raw_semi["auroc"].append(roc_auc_score(yt, ys))
    raw_hmm["brier"].append(brier_score_loss(yt, yh))
    raw_semi["brier"].append(brier_score_loss(yt, ys))
    bh = calibration_bins(yt, yh, 10)
    bs = calibration_bins(yt, ys, 10)
    raw_hmm["ece"].append(expected_calibration_error(bh, len(yt)))
    raw_semi["ece"].append(expected_calibration_error(bs, len(yt)))

raw_hmm_arr = {k: np.asarray(v) for k, v in raw_hmm.items()}
raw_semi_arr = {k: np.asarray(v) for k, v in raw_semi.items()}

bootstrap_comparison = {}
for metric in ("auroc", "brier", "ece"):
    if len(raw_hmm_arr[metric]) == len(raw_semi_arr[metric]) and len(raw_hmm_arr[metric]) > 0:
        cmp = paired_bootstrap_comparison(raw_hmm_arr, raw_semi_arr, metric)
        bootstrap_comparison[metric] = cmp
        print(f"[bootstrap n={n_bootstrap}] {metric}: mean_diff(Semi-HMM - HMM)={cmp['mean_difference']:+.5f} "
              f"CI=[{cmp['ci_low']:+.5f},{cmp['ci_high']:+.5f}] cohens_d={cmp['cohens_d']:.3f} "
              f"wilcoxon_p={cmp['wilcoxon_p']:.4f}")
report["patient_level_bootstrap"] = {"n_bootstrap": n_bootstrap, "rng_seed": 0, "comparisons": bootstrap_comparison}

# ---------------------------------------------------------------------------
# 4. Murphy/Brier decomposition: Brier = reliability - resolution + uncertainty
# ---------------------------------------------------------------------------
def murphy_decomposition(bins, y_true):
    n_total = len(y_true)
    p_bar = float(y_true.mean())
    uncertainty = p_bar * (1 - p_bar)
    reliability = 0.0
    resolution = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        w = b["n"] / n_total
        reliability += w * (b["mean_predicted"] - b["observed_frequency"]) ** 2
        resolution += w * (b["observed_frequency"] - p_bar) ** 2
    brier_reconstructed = reliability - resolution + uncertainty
    return {
        "uncertainty": uncertainty, "reliability": reliability, "resolution": resolution,
        "brier_reconstructed": brier_reconstructed,
    }

decomp_hmm = murphy_decomposition(score_hmm["bins"], y_true_hmm)
decomp_semi = murphy_decomposition(score_semi["bins"], y_true_semi)
decomp_hmm["brier_actual"] = score_hmm["brier"]
decomp_semi["brier_actual"] = score_semi["brier"]
report["murphy_decomposition"] = {"hmm": decomp_hmm, "semi_hmm": decomp_semi}
print()
print("=== Murphy/Brier decomposition (Brier = reliability - resolution + uncertainty) ===")
for name, d in (("HMM", decomp_hmm), ("Semi-HMM", decomp_semi)):
    print(f"  {name:9s} uncertainty={d['uncertainty']:.5f} reliability={d['reliability']:.5f} "
          f"resolution={d['resolution']:.5f} reconstructed={d['brier_reconstructed']:.5f} "
          f"actual={d['brier_actual']:.5f}")

# ---------------------------------------------------------------------------
# 5. Duration correlation in CORRECT units (window-count segments, Phase E's
#    own convention -- the SAME unit Semi-HMM's duration model uses),
#    vs BOTH models' per-phase Brier (freshly computed here, same Val, same
#    k=7 event, directly comparable)
# ---------------------------------------------------------------------------
per_phase_true = {name: [] for name in phase_names}
per_phase_prob_hmm = {name: [] for name in phase_names}
per_phase_prob_semi = {name: [] for name in phase_names}
for traj, ph, ps in zip(val_traj, hmm_preds, semi_preds):
    lf = traj.last_frame_phase.numpy()
    yt = traj.consistency_flag.numpy()
    yh = ph.consistency_flag_prob.numpy()
    ys = ps.consistency_flag_prob.numpy()
    for i, name_idx in enumerate(lf.tolist()):
        name = phase_names[name_idx]
        per_phase_true[name].append(yt[i])
        per_phase_prob_hmm[name].append(yh[i])
        per_phase_prob_semi[name].append(ys[i])

per_phase_table = {}
for name in phase_names:
    yt = np.array(per_phase_true[name])
    yh = np.array(per_phase_prob_hmm[name])
    ys = np.array(per_phase_prob_semi[name])
    if len(yt) == 0:
        continue
    per_phase_table[name] = {
        "n": len(yt),
        "observed": float(yt.mean()),
        "predicted_hmm": float(yh.mean()),
        "predicted_semi_hmm": float(ys.mean()),
        "brier_hmm": float(np.mean((yh - yt) ** 2)),
        "brier_semi_hmm": float(np.mean((ys - yt) ** 2)),
        "calibration_bias_hmm": float(yh.mean() - yt.mean()),
        "calibration_bias_semi_hmm": float(ys.mean() - yt.mean()),
    }

train_per_phase = phase_e["train_per_phase"]
EXCLUDE = {"tPB2", "tEB"}  # structurally 100% censored, same exclusion as the E1 diagnostic
usable_phases = [n for n in phase_names if n not in EXCLUDE and n in per_phase_table
                 and train_per_phase.get(n, {}).get("all_durations_incl_censored") is not None]

mean_dur_windows = [train_per_phase[n]["all_durations_incl_censored"]["mean"] for n in usable_phases]
cv_windows = [train_per_phase[n]["all_durations_incl_censored"]["cv_std_over_mean"] for n in usable_phases]
brier_hmm_by_phase = [per_phase_table[n]["brier_hmm"] for n in usable_phases]
brier_semi_by_phase = [per_phase_table[n]["brier_semi_hmm"] for n in usable_phases]
absbias_semi_by_phase = [abs(per_phase_table[n]["calibration_bias_semi_hmm"]) for n in usable_phases]

def corr(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pr, pp = scipy_stats.pearsonr(x, y)
    sr, sp = scipy_stats.spearmanr(x, y)
    print(f"  {label}: n={len(x)} pearson_r={pr:.4f} p={pp:.4f}  spearman_rho={sr:.4f} p={sp:.4f}")
    return {"n": len(x), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp), "phases": usable_phases}

print()
print(f"=== Duration correlation, CORRECT units (window-count segments, Phase E, Train), n={len(usable_phases)} phases ===")
duration_correlations = {}
duration_correlations["mean_duration_windows_vs_brier_hmm"] = corr(mean_dur_windows, brier_hmm_by_phase, "mean_duration(windows) vs Brier_HMM")
duration_correlations["mean_duration_windows_vs_brier_semi_hmm"] = corr(mean_dur_windows, brier_semi_by_phase, "mean_duration(windows) vs Brier_SemiHMM")
duration_correlations["cv_windows_vs_brier_hmm"] = corr(cv_windows, brier_hmm_by_phase, "CV(windows) vs Brier_HMM")
duration_correlations["cv_windows_vs_brier_semi_hmm"] = corr(cv_windows, brier_semi_by_phase, "CV(windows) vs Brier_SemiHMM")
duration_correlations["cv_windows_vs_abs_calibration_bias_semi_hmm"] = corr(cv_windows, absbias_semi_by_phase, "CV(windows) vs |bias|_SemiHMM")

report["per_phase_table"] = per_phase_table
report["duration_correlation_correct_units"] = {
    "unit": "window-count segment duration (Phase E, Train, HMMModel._segments() convention) "
            "-- NOT the same variable as the original E1/HMM diagnostic's r=-0.7729, which used "
            "real-time 'time-to-next-onset' in raw metadata_with_time.csv units (unconfirmed scale). "
            "This is the FIRST correlation check in the SAME units the Semi-HMM duration model itself uses.",
    "excluded_phases": sorted(EXCLUDE),
    "correlations": duration_correlations,
}

# ---------------------------------------------------------------------------
# 6. Concrete illustrative outputs for the probabilistic-chain question
#    (qualitative, read-only, uses the frozen Semi-HMM model as-is)
# ---------------------------------------------------------------------------
example_traj = next((t for t in val_traj if len(t) > 20), val_traj[0])
chain = semi.transition_chain(example_traj)
next_dist_row0 = semi.next_phase_distribution(example_traj)[0].tolist()
duration_dist_t9plus = semi.duration_distribution(phase_names.index("t9+"))
expected_durations_all = {name: semi.expected_duration(i) for i, name in enumerate(phase_names)}

report["chain_illustrative_example"] = {
    "video_name": example_traj.video_name,
    "transition_chain": chain,
    "next_phase_distribution_at_t0": {phase_names[j]: v for j, v in enumerate(next_dist_row0)},
    "duration_distribution_t9plus_first10": {d: duration_dist_t9plus[d] for d in range(1, 11)},
    "expected_duration_all_phases_windows": expected_durations_all,
}
print()
print(f"=== Illustrative chain example: {example_traj.video_name} ===")
for seg in chain:
    print(f"  {seg['phase']:6s} observed_dur={seg['observed_duration_windows']:4d} "
          f"P(D=obs|phase)={seg['duration_probability_at_observed']:.4f}")

# ---------------------------------------------------------------------------
# 7. Persist
# ---------------------------------------------------------------------------
(OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2))
print()
print("Report written to", OUT_DIR / "analysis.json")
print("ANALYSIS_DONE")
