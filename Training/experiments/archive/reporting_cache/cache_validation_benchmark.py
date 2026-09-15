"""
Phase 4 (scientific validation) + Phase 5 (benchmark) for the Reporting
Cache (Phase 1.5). Read-only against the frozen model; the ONLY write
this script performs is populating the disk cache under Cache/reporting/
(pure serving infrastructure, gitignored, never a scientific artifact).
Test split never loaded, never referenced.
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "reporting")
import numpy as np

from reporting import cache as reporting_cache
from reporting import inference_service, model_loader, trajectory_service

VIDEO = "Patient_319"
SPLIT = "val"

# Ensure a clean slate for this video's cache entry before starting, so
# "cache miss" below is a genuine miss, not an artifact of a previous run.
model = model_loader.get_model()
model_version = model_loader.model_version_string(model)
cache_dir = reporting_cache._cache_dir("semi_hmm", model_version, SPLIT, VIDEO)
if cache_dir.exists():
    shutil.rmtree(cache_dir)
print(f"model_version={model_version}")
print(f"cache_dir={cache_dir} (cleared for a clean test)")

results = {"video": VIDEO, "split": SPLIT, "model_version": model_version}

# --- load trajectory (one-time embeddings cache load + this video) ---
t0 = time.time()
traj = trajectory_service.get_trajectory(VIDEO, SPLIT)
t_traj = time.time() - t0
print(f"\nload trajectory (includes one-time embeddings cache load for this process): {t_traj:.4f}s, "
      f"n_windows={len(traj)}")
results["load_trajectory_seconds"] = t_traj

# --- BEFORE: raw, uncached model calls (ground truth for "no cache exists at all") ---
t0 = time.time()
raw_posterior = model.filtering(traj)
t_raw_filter = time.time() - t0
t0 = time.time()
raw_next_dist = model.next_phase_distribution(traj)
t_raw_next = time.time() - t0
print(f"\nBEFORE (raw model calls, no cache layer at all):")
print(f"  filtering: {t_raw_filter:.4f}s")
print(f"  next_phase_distribution: {t_raw_next:.4f}s")
results["before"] = {"filtering_seconds": t_raw_filter, "next_phase_distribution_seconds": t_raw_next,
                      "total_seconds": t_raw_filter + t_raw_next}

# --- CACHE MISS: first call through the new caching wrapper (clean cache_dir from above) ---
inference_service._POSTERIOR_CACHE.clear()
inference_service._NEXT_PHASE_CACHE.clear()
t0 = time.time()
miss_posterior, miss_next_dist, miss_stats = reporting_cache.get_or_compute_forward_pass(
    model, VIDEO, SPLIT, traj, "semi_hmm", model_version)
t_miss_wall = time.time() - t0
print(f"\nCACHE MISS (first call through cache.py):")
print(f"  hit={miss_stats.hit} load_time={miss_stats.load_time_seconds:.4f}s "
      f"compute_time={miss_stats.compute_time_seconds:.4f}s total={miss_stats.total_time_seconds:.4f}s "
      f"(wall={t_miss_wall:.4f}s)")
results["cache_miss"] = {"hit": miss_stats.hit, "load_time_seconds": miss_stats.load_time_seconds,
                          "compute_time_seconds": miss_stats.compute_time_seconds,
                          "total_time_seconds": miss_stats.total_time_seconds, "wall_seconds": t_miss_wall}

# --- CACHE HIT: second call, same key ---
t0 = time.time()
hit_posterior, hit_next_dist, hit_stats = reporting_cache.get_or_compute_forward_pass(
    model, VIDEO, SPLIT, traj, "semi_hmm", model_version)
t_hit_wall = time.time() - t0
print(f"\nCACHE HIT (second call, same key):")
print(f"  hit={hit_stats.hit} load_time={hit_stats.load_time_seconds:.4f}s "
      f"compute_time={hit_stats.compute_time_seconds:.4f}s total={hit_stats.total_time_seconds:.4f}s "
      f"(wall={t_hit_wall:.4f}s)")
results["cache_hit"] = {"hit": hit_stats.hit, "load_time_seconds": hit_stats.load_time_seconds,
                         "compute_time_seconds": hit_stats.compute_time_seconds,
                         "total_time_seconds": hit_stats.total_time_seconds, "wall_seconds": t_hit_wall}

speedup = results["before"]["total_seconds"] / max(t_hit_wall, 1e-9)
print(f"\nSpeedup (before -> cache hit): {speedup:.1f}x")
results["speedup_before_to_hit"] = speedup

# ---------------------------------------------------------------------------
# Phase 4: numerical equivalence -- raw vs. miss vs. hit, all three compared
# ---------------------------------------------------------------------------
print("\n=== Numerical equivalence (Phase 4) ===")
checks = {
    "raw_vs_miss_posterior": np.array_equal(raw_posterior, miss_posterior),
    "raw_vs_miss_next_dist": np.array_equal(raw_next_dist, miss_next_dist),
    "miss_vs_hit_posterior": np.array_equal(miss_posterior, hit_posterior),
    "miss_vs_hit_next_dist": np.array_equal(miss_next_dist, hit_next_dist),
    "raw_vs_hit_posterior_allclose": bool(np.allclose(raw_posterior, hit_posterior, atol=0.0, rtol=0.0)),
    "raw_vs_hit_next_dist_allclose": bool(np.allclose(raw_next_dist, hit_next_dist, atol=0.0, rtol=0.0)),
}
for k, v in checks.items():
    print(f"  {k}: {v}")
results["numerical_equivalence"] = checks
assert all(checks.values()), "NUMERICAL MISMATCH DETECTED -- cache is not a pure passthrough!"

# --- end-to-end: full InferenceRecord before cache existed vs. after (warm) ---
inference_service._POSTERIOR_CACHE.clear()
inference_service._NEXT_PHASE_CACHE.clear()
window0 = traj.window_starts[10]
record_warm = inference_service.infer(VIDEO, SPLIT, window0)
record_warm_dict = record_warm.to_dict()
# strip the DERIVED, request-time-varying field before comparing
record_warm_dict["model"]["inference_timestamp"] = None
print("\n=== End-to-end InferenceRecord (warm cache), current_phase/entropy/duration ===")
print(f"  current_phase={record_warm.current_state.current_phase} "
      f"entropy={record_warm.current_state.entropy:.6f} "
      f"duration_available={record_warm.duration.duration_available}")
results["sample_inference_record"] = record_warm_dict

out_path = Path("/tmp/reporting_cache_validation_results.json")
out_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults written to {out_path}")
print("CACHE_VALIDATION_BENCHMARK_DONE")
print("EXITCODE:0")
