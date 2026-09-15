"""
Phase E -- Semi-HMM Dmax / duration diagnostic.
One-off diagnostic script (not part of the production evaluation package,
not committed anywhere -- same convention as Phase A/B/C/D's /tmp scripts).

Purpose: per-phase segment-duration statistics on full real Train and Val
(n_segments, mean, median, variance/std, CV, quantiles, max, censoring
breakdown, support), and a comparison of several candidate Dmax strategies
-- determined from TRAIN ONLY (extract_phase_durations/determine_dmax
already enforce this by construction: the caller must pass Train durations
in, the function itself cannot see which split it was given).

Val statistics are reported for comparison/context only, never used to
pick Dmax. Test split is never loaded, never referenced.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from embeddings.dataset import EmbeddingDataset
from evaluation.models.hmm import DEFAULT_PHASE_NAMES, HMMModel
from evaluation.models.semi_hmm import (
    CensoringType,
    determine_dmax,
    extract_phase_durations,
)
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/semi_hmm_weekend_phaseE")
OUT_DIR.mkdir(parents=True, exist_ok=True)
report = {}

t0 = time.time()
train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")
report["load_time_seconds"] = t_load
report["n_train"] = len(train_traj)
report["n_val"] = len(val_traj)

phase_names = list(DEFAULT_PHASE_NAMES)
print(f"phase_names ({len(phase_names)}): {phase_names}")
report["phase_names"] = phase_names

# ---------------------------------------------------------------------------
# 1. Per-phase duration statistics, Train and Val separately
# ---------------------------------------------------------------------------
def per_phase_stats(durations_by_phase, phase_names):
    stats_by_phase = {}
    for idx, name in enumerate(phase_names):
        durs_censoring = durations_by_phase.get(idx, [])
        n_segments = len(durs_censoring)
        censoring_counts = {c.value: 0 for c in CensoringType}
        for _, c in durs_censoring:
            censoring_counts[c.value] += 1
        all_durs = [d for d, _ in durs_censoring]
        interior_durs = [d for d, c in durs_censoring if c == CensoringType.OBSERVED]

        def summarize(durs):
            if not durs:
                return None
            arr = np.array(durs, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return {
                "n": len(durs),
                "mean": mean,
                "median": float(np.median(arr)),
                "std": std,
                "variance": float(std ** 2),
                "cv_std_over_mean": (std / mean) if mean > 0 else None,
                "min": int(arr.min()),
                "max": int(arr.max()),
                "quantiles": {
                    str(q): float(np.quantile(arr, q))
                    for q in (0.5, 0.75, 0.9, 0.95, 0.99)
                },
            }

        stats_by_phase[name] = {
            "n_segments_total": n_segments,
            "censoring_counts": censoring_counts,
            "pct_censored": (
                100.0 * (n_segments - censoring_counts["observed"]) / n_segments
                if n_segments > 0 else None
            ),
            "all_durations_incl_censored": summarize(all_durs),
            "interior_uncensored_only": summarize(interior_durs),
        }
    return stats_by_phase


train_durations_by_phase = extract_phase_durations(train_traj)
val_durations_by_phase = extract_phase_durations(val_traj)

train_stats = per_phase_stats(train_durations_by_phase, phase_names)
val_stats = per_phase_stats(val_durations_by_phase, phase_names)

report["train_per_phase"] = train_stats
report["val_per_phase"] = val_stats

print()
print("=== Per-phase duration stats (Train, all durations incl. censored) ===")
for name in phase_names:
    s = train_stats[name]["all_durations_incl_censored"]
    cc = train_stats[name]["censoring_counts"]
    if s is None:
        print(f"  {name:6s}: NO SEGMENTS OBSERVED IN TRAIN")
        continue
    print(
        f"  {name:6s} n_seg={s['n']:5d} mean={s['mean']:7.3f} median={s['median']:6.1f} "
        f"std={s['std']:6.3f} cv={s['cv_std_over_mean']:.4f} min={s['min']:3d} max={s['max']:4d} "
        f"q95={s['quantiles']['0.95']:.1f} q99={s['quantiles']['0.99']:.1f} | "
        f"obs={cc['observed']} left={cc['left_censored']} right={cc['right_censored']} both={cc['both_censored']}"
    )

# ---------------------------------------------------------------------------
# 2. Structural censoring fact re-check (Phase A2 already found this at the
#    fit()-forcing level; here it's re-derived directly from segment
#    extraction, full 15-phase breakdown, Train AND Val)
# ---------------------------------------------------------------------------
fully_censored_train = [
    name for name in phase_names
    if train_stats[name]["n_segments_total"] > 0
    and train_stats[name]["censoring_counts"]["observed"] == 0
]
fully_censored_val = [
    name for name in phase_names
    if val_stats[name]["n_segments_total"] > 0
    and val_stats[name]["censoring_counts"]["observed"] == 0
]
never_observed_train = [name for name in phase_names if train_stats[name]["n_segments_total"] == 0]
print()
print(f"Phases 100% censored in Train (0 interior/OBSERVED segments): {fully_censored_train}")
print(f"Phases 100% censored in Val: {fully_censored_val}")
print(f"Phases never appearing as a segment in Train at all: {never_observed_train}")
report["fully_censored_train"] = fully_censored_train
report["fully_censored_val"] = fully_censored_val
report["never_observed_train"] = never_observed_train

# ---------------------------------------------------------------------------
# 3. Dmax strategy comparison -- TRAIN ONLY
# ---------------------------------------------------------------------------
strategies = []
dmax_observed_max = determine_dmax(train_durations_by_phase, strategy="observed_max")
strategies.append(("observed_max", dmax_observed_max))
for q in (0.95, 0.99, 0.999):
    dmax_q = determine_dmax(train_durations_by_phase, strategy="quantile", quantile=q)
    strategies.append((f"quantile_{q}", dmax_q))

all_train_durations = [d for durs in train_durations_by_phase.values() for d, _ in durs]
all_train_durations_arr = np.array(all_train_durations, dtype=float)

print()
print("=== Dmax candidate strategies (Train only) ===")
strategy_report = []
for label, dmax_val in strategies:
    n_total = len(all_train_durations_arr)
    n_truncated = int((all_train_durations_arr > dmax_val).sum())
    pct_truncated = 100.0 * n_truncated / n_total if n_total else 0.0
    # which phases would have their OWN max duration truncated
    phases_truncated = [
        name for name in phase_names
        if train_stats[name]["all_durations_incl_censored"] is not None
        and train_stats[name]["all_durations_incl_censored"]["max"] > dmax_val
    ]
    row = {
        "strategy": label,
        "dmax": int(dmax_val),
        "n_segments_truncated": n_truncated,
        "pct_segments_truncated": pct_truncated,
        "phases_with_truncated_max": phases_truncated,
    }
    strategy_report.append(row)
    print(
        f"  {label:14s} dmax={dmax_val:5d}  truncates {n_truncated:5d}/{n_total} segments "
        f"({pct_truncated:.3f}%)  phases_affected={phases_truncated}"
    )

report["dmax_strategies"] = strategy_report

# ---------------------------------------------------------------------------
# 4. Runtime-cost implication, using Phase D's measured per-window cost model
#    (optimized forward: O(T*K^2 + T*K*Dmax), K=15 states). Phase D measured
#    0.0011221762827286115 s/window at dmax=40 on full real Val (43339
#    windows) -- reused here as-is, NOT re-benchmarked (Phase D artifact,
#    read-only).
# ---------------------------------------------------------------------------
K = 15
PHASE_D_DMAX = 40
PHASE_D_SEC_PER_WINDOW = 0.0011221762827286115
# seconds/window = a + b*Dmax (a ~ K^2 term, b ~ K term), single data point
# only lets us assume the K*Dmax term dominates linearly and rescale
# proportionally to Dmax relative to the K^2+K*Dmax total at dmax=40 --
# reported as an approximate scaling projection, not a fresh measurement.
denom_ref = K * K + K * PHASE_D_DMAX
n_val_windows = sum(len(t) for t in val_traj)
n_train_windows = sum(len(t) for t in train_traj)
print()
print("=== Approximate runtime projection per Dmax candidate (linear-in-Dmax scaling from Phase D's single measured point, NOT a fresh benchmark) ===")
runtime_projection = []
for label, dmax_val in strategies:
    denom_candidate = K * K + K * dmax_val
    scale = denom_candidate / denom_ref
    sec_per_window_est = PHASE_D_SEC_PER_WINDOW * scale
    val_sec_est = sec_per_window_est * n_val_windows
    train_sec_est = sec_per_window_est * n_train_windows
    row = {
        "strategy": label, "dmax": int(dmax_val),
        "est_sec_per_window": sec_per_window_est,
        "est_val_filtering_seconds": val_sec_est,
        "est_train_filtering_seconds": train_sec_est,
    }
    runtime_projection.append(row)
    print(
        f"  {label:14s} dmax={dmax_val:5d}  ~{sec_per_window_est:.6f}s/window  "
        f"Val~{val_sec_est:.1f}s  Train~{train_sec_est:.1f}s"
    )
report["runtime_projection_note"] = (
    "Linear-in-Dmax rescaling of Phase D's single measured point "
    "(dmax=40 -> 0.0011221762827286115 s/window on full real Val); "
    "approximate, not a fresh benchmark."
)
report["runtime_projection"] = runtime_projection

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
(OUT_DIR / "dmax_duration_diagnostic_report.json").write_text(json.dumps(report, indent=2))
print()
print("Report written to", OUT_DIR / "dmax_duration_diagnostic_report.json")
print("PHASE_E_DONE")
