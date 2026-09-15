import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from embeddings.dataset import EmbeddingDataset
from evaluation.models.semi_hmm import SemiHMMModel, determine_dmax, extract_phase_durations
from evaluation.trajectory import group_into_trajectories

OUT_DIR = Path("../Results/evaluation/semi_hmm_weekend_phaseD")
OUT_DIR.mkdir(parents=True, exist_ok=True)
report = {}

t0 = time.time()
train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_load = time.time() - t0
print(f"load: {t_load:.2f}s, n_train={len(train_traj)}, n_val={len(val_traj)}")

durations_by_phase = extract_phase_durations(train_traj)
dmax_observed = determine_dmax(durations_by_phase, strategy="observed_max")
dmax = min(dmax_observed, 40)  # same smoke-test-consistent cap, benchmark purposes only, not a Dmax decision
print(f"dmax_observed={dmax_observed}, using dmax={dmax} for this benchmark")

t0 = time.time()
model = SemiHMMModel(n_states=15, dmax=dmax, duration_family="negative_binomial")
model.fit(train_traj)  # FULL real Train, 492 trajectories
t_fit = time.time() - t0
print(f"fit on FULL Train (492 trajectories): {t_fit:.2f}s")

report["load_time_seconds"] = t_load
report["fit_time_seconds_full_train"] = t_fit
report["dmax"] = dmax
report["n_train"] = len(train_traj)
report["n_val"] = len(val_traj)

# ---------------------------------------------------------------------------
# Reference vs optimized on a SMALL number of real Val trajectories
# ---------------------------------------------------------------------------
n_bench_small = 3
small_val = val_traj[:n_bench_small]
small_windows = sum(len(t) for t in small_val)

t0 = time.time()
for traj in small_val:
    model._explicit_duration_forward_reference(traj)
t_ref = time.time() - t0

t0 = time.time()
for traj in small_val:
    model._explicit_duration_forward_optimized(traj)
t_opt_small = time.time() - t0

speedup = t_ref / t_opt_small if t_opt_small > 0 else float("inf")
print(f"\nreference forward on {n_bench_small} Val traj ({small_windows} windows): {t_ref:.2f}s "
      f"({t_ref/small_windows:.4f}s/window)")
print(f"optimized forward on {n_bench_small} Val traj ({small_windows} windows): {t_opt_small:.2f}s "
      f"({t_opt_small/small_windows:.4f}s/window)")
print(f"speedup: {speedup:.1f}x")

report["small_comparison"] = {
    "n_trajectories": n_bench_small, "n_windows": small_windows,
    "reference_seconds": t_ref, "reference_seconds_per_window": t_ref / small_windows,
    "optimized_seconds": t_opt_small, "optimized_seconds_per_window": t_opt_small / small_windows,
    "speedup_factor": speedup,
}

# ---------------------------------------------------------------------------
# Optimized forward on ALL of Val (real, full scale)
# ---------------------------------------------------------------------------
n_val_windows_total = sum(len(t) for t in val_traj)
t0 = time.time()
for traj in val_traj:
    model.filtering(traj)
t_opt_full_val = time.time() - t0
print(f"\noptimized filtering() on FULL Val ({len(val_traj)} traj, {n_val_windows_total} windows): "
      f"{t_opt_full_val:.2f}s ({t_opt_full_val/n_val_windows_total:.5f}s/window)")

n_train_windows_total = sum(len(t) for t in train_traj)
extrapolated_train_seconds = (t_opt_full_val / n_val_windows_total) * n_train_windows_total
print(f"extrapolated optimized filtering() for FULL Train ({n_train_windows_total} windows): "
      f"~{extrapolated_train_seconds/60:.1f} min")

report["full_val_optimized"] = {
    "n_val_trajectories": len(val_traj), "n_val_windows": n_val_windows_total,
    "seconds": t_opt_full_val, "seconds_per_window": t_opt_full_val / n_val_windows_total,
}
report["train_extrapolation"] = {
    "n_train_windows": n_train_windows_total,
    "extrapolated_seconds": extrapolated_train_seconds,
    "extrapolated_minutes": extrapolated_train_seconds / 60,
}

(OUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2))
print("\nReport written to", OUT_DIR / "benchmark_report.json")
print("BENCHMARK_DONE")
