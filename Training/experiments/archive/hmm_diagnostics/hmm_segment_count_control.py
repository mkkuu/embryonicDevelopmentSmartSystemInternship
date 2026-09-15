"""
One-off diagnostic (not committed, not part of the production package):
replaces n_windows with n_segments (real, video-level contiguous phase
occurrences) as the sample-size control variable in the partial
correlation Brier ~ mean_duration | Z, to test whether the previous
session's collapse (r=-0.143, p=0.657 with Z=n_windows) was an artifact of
window-count being mechanically driven by duration.

Segment definition reused verbatim from HMMModel._segments() (hmm.py) --
the SAME run-length encoding over last_frame_phase already used internally
by fit()'s segment-level transition counting -- not reinvented, per this
session's explicit instruction to read the code before deciding.

Val-only. No retraining (HMMModel.load(), read-only). Duration statistics
reused as-is from Results/evaluation/e1_hmm_phase2_analysis/report.json
(Train, existing artifact, not modified). Test split never referenced.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from scipy import stats
import numpy as np
from sklearn.metrics import roc_auc_score

from embeddings.dataset import EmbeddingDataset
from evaluation.calibration import flatten_predictions, score_probabilistic
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/e1_hmm_segment_count_control")
OUT_DIR.mkdir(parents=True, exist_ok=True)

phase_names = DEFAULT_PHASE_NAMES

# ---------------------------------------------------------------------------
# 1. Val trajectories, frozen k=7 winner model (alpha=2.0, rho=0.5), no refit
# ---------------------------------------------------------------------------
val = group_into_trajectories(EmbeddingDataset("../Embeddings/resnet18", "val"))
model = HMMModel.load(Path("../Results/evaluation/e1_hmm_k7_sweep/best_model"))
preds = model.predict_k_step(val, k=7)

# ---------------------------------------------------------------------------
# 2. Per-phase Brier (fresh, self-contained -- same method as the prior
#    diagnostic, reproduced here so this script doesn't depend on cross-
#    referencing a second JSON file by hand)
# ---------------------------------------------------------------------------
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

brier_by_phase = {}
n_windows_by_phase = {}
for name in phase_names:
    mask = per_phase_name == name
    n = int(mask.sum())
    if n == 0:
        continue
    brier_by_phase[name] = float(np.mean((per_phase_prob[mask] - per_phase_true[mask]) ** 2))
    n_windows_by_phase[name] = n

# ---------------------------------------------------------------------------
# 3. Segments -- HMMModel._segments() reused verbatim, Val only
# ---------------------------------------------------------------------------
segment_counts = {name: 0 for name in phase_names}
video_sets = {name: set() for name in phase_names}
segment_window_lengths = {name: [] for name in phase_names}

total_segments = 0
total_within_video_transitions = 0
n_videos_with_data = 0

tpb2_idx = phase_names.index("tPB2")
teb_idx = phase_names.index("tEB")
tpb2_non_first_violations = 0
teb_non_last_violations = 0
chronology_violations = 0

for traj in val:
    if len(traj) == 0:
        continue
    n_videos_with_data += 1
    segs = HMMModel._segments(traj)  # list of (phase_idx, start_row, end_row), row = position in window_starts order
    total_segments += len(segs)
    total_within_video_transitions += len(segs) - 1

    # sanity: rows strictly increasing / contiguous / non-overlapping and covering [0, T)
    expected_start = 0
    for (phase_idx, start, end) in segs:
        if start != expected_start or end < start:
            chronology_violations += 1
        expected_start = end + 1

    for pos, (phase_idx, start, end) in enumerate(segs):
        name = phase_names[phase_idx]
        segment_counts[name] += 1
        video_sets[name].add(traj.video_name)
        segment_window_lengths[name].append(end - start + 1)
        if phase_idx == tpb2_idx and pos != 0:
            tpb2_non_first_violations += 1
        if phase_idx == teb_idx and pos != len(segs) - 1:
            teb_non_last_violations += 1

print("=== SANITY CHECKS ===")
print(f"n Val videos with data: {n_videos_with_data}")
print(f"total segments (all phases, all videos): {total_segments}")
print(f"total within-video phase CHANGES (= total_segments - n_videos_with_data): {total_within_video_transitions}")
print(f"chronology/contiguity violations (should be 0): {chronology_violations}")
print(f"tPB2 appearing as a NON-FIRST segment of its video (should be 0, tPB2 is left-censored): {tpb2_non_first_violations}")
print(f"tEB appearing as a NON-LAST segment of its video (should be 0, tEB is right-censored): {teb_non_last_violations}")
print(f"sum(n_segments over phases) == total_segments check: {sum(segment_counts.values())} vs {total_segments}")

# ---------------------------------------------------------------------------
# 4. Manual spot-check on 3 real videos (printed for human inspection)
# ---------------------------------------------------------------------------
print()
print("=== MANUAL SPOT-CHECK (first 3 Val videos with >=2 segments) ===")
shown = 0
for traj in val:
    if len(traj) == 0:
        continue
    segs = HMMModel._segments(traj)
    if len(segs) < 2:
        continue
    seq = [phase_names[p] for p, _, _ in segs]
    print(f"  video={traj.video_name!r} n_windows={len(traj)} segment_sequence={seq} "
          f"segment_window_lengths={[e - s + 1 for _, s, e in segs]}")
    shown += 1
    if shown >= 3:
        break

# ---------------------------------------------------------------------------
# 5. Duration stats (Train, existing Phase 2 artifact, read-only, unchanged)
# ---------------------------------------------------------------------------
phase2 = json.load(open("../Results/evaluation/e1_hmm_phase2_analysis/report.json"))
dist = phase2["section6_7_8_durations_distributions_censoring"]["distributions_by_phase"]

duration_stats = {}
missing_duration = []
for name in phase_names:
    od = dist.get(name, {}).get("onset_duration_interior")
    if od is None or od.get("mean") is None:
        missing_duration.append(name)
        continue
    duration_stats[name] = od

# ---------------------------------------------------------------------------
# 6. Merge into the full diagnostic table
# ---------------------------------------------------------------------------
print()
print("=== FULL TABLE (phase, brier, mean_dur, median_dur, cv, n_windows, n_segments, n_videos, seg_per_video) ===")
rows = []
for name in phase_names:
    if name not in brier_by_phase:
        continue
    n_seg = segment_counts[name]
    n_vid = len(video_sets[name])
    seg_per_video = n_seg / n_vid if n_vid else float("nan")
    ds = duration_stats.get(name)
    row = {
        "phase": name,
        "brier": brier_by_phase[name],
        "n_windows": n_windows_by_phase[name],
        "n_segments": n_seg,
        "n_videos": n_vid,
        "segment_per_video_mean": seg_per_video,
        "has_duration": ds is not None,
    }
    if ds is not None:
        row.update({
            "mean_duration": ds["mean"], "median_duration": ds["median"],
            "std_duration": ds["std"], "cv": ds["cv_std_over_mean"],
        })
    rows.append(row)
    dur_str = f"mean_dur={ds['mean']:.3f} cv={ds['cv_std_over_mean']:.4f}" if ds else "NO DURATION DATA"
    print(f"  {name:6s} brier={row['brier']:.4f} n_windows={row['n_windows']:5d} "
          f"n_segments={n_seg:4d} n_videos={n_vid:4d} seg/video={seg_per_video:.3f} | {dur_str}")

usable = [r for r in rows if r["has_duration"]]
print()
print(f"n phases with Brier: {len(rows)}, usable for duration correlation: {len(usable)}, "
      f"excluded (no duration): {missing_duration}")

# ---------------------------------------------------------------------------
# 7. Correlations
# ---------------------------------------------------------------------------
def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    return r, p, rho, ps

names_u = [r["phase"] for r in usable]
brier_u = [r["brier"] for r in usable]
nseg_u = [r["n_segments"] for r in usable]
nvid_u = [r["n_videos"] for r in usable]
dur_u = [r["mean_duration"] for r in usable]

print()
print("=== A. Brier vs n_segments ===")
r, p, rho, ps = corr(nseg_u, brier_u)
print(f"n={len(usable)} Pearson r={r:.4f} p={p:.4f}  Spearman rho={rho:.4f} p={ps:.4f}")

print("=== B. Brier vs n_videos ===")
r, p, rho, ps = corr(nvid_u, brier_u)
print(f"n={len(usable)} Pearson r={r:.4f} p={p:.4f}  Spearman rho={rho:.4f} p={ps:.4f}")

print("=== C. Brier vs mean_duration (recomputed on this table) ===")
r_bd, p_bd, rho_bd, ps_bd = corr(dur_u, brier_u)
print(f"n={len(usable)} Pearson r={r_bd:.4f} p={p_bd:.4f}  Spearman rho={rho_bd:.4f} p={ps_bd:.4f}")

print("=== D. mean_duration vs n_segments ===")
r_dn, p_dn, rho_dn, ps_dn = corr(dur_u, nseg_u)
print(f"n={len(usable)} Pearson r={r_dn:.4f} p={p_dn:.4f}  Spearman rho={rho_dn:.4f} p={ps_dn:.4f}")

# ---------------------------------------------------------------------------
# 8. Partial correlation: Brier ~ mean_duration | n_segments
# ---------------------------------------------------------------------------
def partial_corr(brier, dur, control, label, names):
    brier = np.asarray(brier, dtype=float)
    dur = np.asarray(dur, dtype=float)
    control = np.asarray(control, dtype=float)
    n = len(brier)
    r_bd, _ = stats.pearsonr(brier, dur)
    r_bc, _ = stats.pearsonr(brier, control)
    r_dc, _ = stats.pearsonr(dur, control)
    denom = np.sqrt((1 - r_bc**2) * (1 - r_dc**2))
    partial_r = (r_bd - r_bc * r_dc) / denom if denom > 1e-12 else float("nan")
    df = n - 3
    if df > 0 and abs(partial_r) < 1.0:
        t_stat = partial_r * np.sqrt(df / (1 - partial_r**2))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        t_stat, p_val = float("nan"), float("nan")
    # Spearman-based partial (rank-transform then same formula) -- reported
    # as "methodologically reasonable" per the task's own caveat, not a
    # standard textbook statistic, flagged as such.
    rank_b = stats.rankdata(brier)
    rank_d = stats.rankdata(dur)
    rank_c = stats.rankdata(control)
    rho_bd, _ = stats.pearsonr(rank_b, rank_d)
    rho_bc, _ = stats.pearsonr(rank_b, rank_c)
    rho_dc, _ = stats.pearsonr(rank_d, rank_c)
    denom_s = np.sqrt((1 - rho_bc**2) * (1 - rho_dc**2))
    partial_rho = (rho_bd - rho_bc * rho_dc) / denom_s if denom_s > 1e-12 else float("nan")
    print(f"  {label}: n={n} partial Pearson r={partial_r:.4f} t={t_stat:.4f} df={df} p={p_val:.4f} | "
          f"partial Spearman-analog rho={partial_rho:.4f}")
    return {"n": n, "partial_pearson_r": partial_r, "partial_pearson_p": p_val,
            "partial_spearman_rho": partial_rho, "phases": list(names)}

print()
print("=== E. PARTIAL CORRELATION Brier ~ mean_duration | n_segments (MAIN TEST) ===")
main_result = partial_corr(brier_u, dur_u, nseg_u, "all phases (A)", names_u)

# ---------------------------------------------------------------------------
# 9. Robustness: drop tPNf / t3 / both
# ---------------------------------------------------------------------------
print()
print("=== ROBUSTNESS: Brier ~ mean_duration | n_segments ===")
robustness = {}
for drop_set, label in [(set(), "A: all phases"), ({"tPNf"}, "B: without tPNf"),
                          ({"t3"}, "C: without t3"), ({"tPNf", "t3"}, "D: without tPNf AND t3")]:
    idx = [i for i, n in enumerate(names_u) if n not in drop_set]
    b = [brier_u[i] for i in idx]
    d = [dur_u[i] for i in idx]
    c = [nseg_u[i] for i in idx]
    kept = [names_u[i] for i in idx]
    res = partial_corr(b, d, c, label, kept)
    robustness[label] = res

# ---------------------------------------------------------------------------
# 10. Persist
# ---------------------------------------------------------------------------
report = {
    "model_used": "Results/evaluation/e1_hmm_k7_sweep/best_model (alpha=2.0, rho=0.5), loaded read-only, not refit",
    "segment_definition_source": "HMMModel._segments() classmethod (Training/evaluation/models/hmm.py), reused verbatim -- run-length encoding over last_frame_phase in window_starts order, same construction fit() already uses internally for segment-level transition counting",
    "duration_source": "Results/evaluation/e1_hmm_phase2_analysis/report.json (Train split, computed 2026-08-19), read-only",
    "sanity_checks": {
        "n_videos_with_data": n_videos_with_data,
        "total_segments": total_segments,
        "total_within_video_phase_changes": total_within_video_transitions,
        "chronology_violations": chronology_violations,
        "tpb2_non_first_violations": tpb2_non_first_violations,
        "teb_non_last_violations": teb_non_last_violations,
    },
    "phases_excluded_no_duration_data": missing_duration,
    "rows": rows,
    "correlations": {
        "brier_vs_n_segments": {"pearson_r": corr(nseg_u, brier_u)[0], "pearson_p": corr(nseg_u, brier_u)[1],
                                  "spearman_rho": corr(nseg_u, brier_u)[2], "spearman_p": corr(nseg_u, brier_u)[3]},
        "brier_vs_n_videos": {"pearson_r": corr(nvid_u, brier_u)[0], "pearson_p": corr(nvid_u, brier_u)[1],
                               "spearman_rho": corr(nvid_u, brier_u)[2], "spearman_p": corr(nvid_u, brier_u)[3]},
        "brier_vs_mean_duration": {"pearson_r": r_bd, "pearson_p": p_bd, "spearman_rho": rho_bd, "spearman_p": ps_bd},
        "mean_duration_vs_n_segments": {"pearson_r": r_dn, "pearson_p": p_dn, "spearman_rho": rho_dn, "spearman_p": ps_dn},
    },
    "partial_correlation_brier_duration_given_n_segments": main_result,
    "robustness_partial_correlation": robustness,
}
(OUT_DIR / "segment_count_control_report.json").write_text(json.dumps(report, indent=2))
print()
print("Report written to", OUT_DIR / "segment_count_control_report.json")
