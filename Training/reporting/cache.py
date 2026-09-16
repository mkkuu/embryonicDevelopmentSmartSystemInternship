"""
Disk-backed cache for the frozen Semi-HMM's expensive per-video forward
pass. Pure infrastructure: never touches the model's mathematics, never
persists a value the model didn't already compute, never re-derives a
"corrected" value. Full design rationale: docs/REPORTING_CACHE.md.

ROOT CAUSE this cache exists to work around (diagnosed this session, NOT
fixed here -- fixing it would mean touching the frozen SemiHMMModel,
explicitly out of scope for this phase):
`SemiHMMModel.next_phase_distribution()` (Training/evaluation/models/
semi_hmm.py) is a T*K*Dmax nested Python loop that calls
`duration_models[j].log_prob(a)`/`.log_survival(a)` fresh on every one of
its ~T*K*Dmax iterations. Its sibling method's own optimized
implementation (`_explicit_duration_forward_optimized`, which is what
`filtering()` uses) already documents the fix for exactly this pattern --
precompute a (K,Dmax) table of log_prob/log_survival ONCE per model,
not once per (t,j,a) triple -- but that fix was never applied to
`next_phase_distribution()`. Measured this session: ~333s for a single
527-window Val trajectory, vs. ~1.7s for the whole trajectory's
`filtering()`. This module does not change that method or its output; it
guarantees the expensive call happens at most ONCE per
(model_version, split, video), ever, persisted across process restarts.

WHAT IS CACHED, AND WHY DURATION IS NOT
------------------------------------------
Only two arrays are worth caching: `filtering()`'s (T,K) posterior and
`next_phase_distribution()`'s (T,K) distribution -- both are computed
once for an ENTIRE trajectory regardless of which window is later
requested, so a single video-level cache entry serves every window of
that video. Duration information (`expected_duration`, `.quantiles()`)
is deliberately NOT cached here: it is a property of the fitted model and
a PHASE INDEX ONLY (`model.duration_models[phase_idx]`), never of the
trajectory or window -- confirmed by reading the code, not assumed --
and costs ~0.0016s per lookup (measured), already far below any caching
threshold worth the complexity. Timestamps and ground truth are served by
`trajectory_service` directly from the already-fast metadata CSV/
Trajectory object and are likewise not duplicated here.

CONSISTENCY / VERSIONING
----------------------------
Cache key = (model_name, model_version, split, video_name). model_version
is `model_loader.model_version_string()` -- a stable fingerprint of every
hyperparameter that affects filtering()/next_phase_distribution()'s
output (dmax, duration_family, transition_smoothing_alpha,
transition_decay_rho, logreg_class_weight). A cache entry is NEVER read
across a version mismatch, a video mismatch, or a window_starts mismatch
against the live trajectory (Phase 2's structural-filtering could in
principle change window_starts for the same video_name across a future
different embeddings build -- checked explicitly, not assumed away).

CONCURRENCY
---------------
Uses atomic write (temp file + os.replace) so a reader NEVER observes a
partially-written file, and a per-cache-key `fcntl.flock()` (stdlib,
works across both threads and processes on one machine, no new
dependency, no distributed lock manager) so two concurrent requests for
the same uncached video don't both pay the ~330s cost -- the second
request blocks on the lock and then reads the first request's now-warm
result, rather than recomputing.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("reporting.cache")

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "Cache" / "reporting"
CACHE_FORMAT_VERSION = 1


def _safe_slug(s: str) -> str:
    """Filesystem-safe but still human-readable directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _cache_dir(model_name: str, model_version: str, split: str, video_name: str) -> Path:
    return CACHE_ROOT / _safe_slug(model_name) / _safe_slug(model_version) / split / video_name


@dataclass
class CacheStats:
    hit: bool
    load_time_seconds: float
    compute_time_seconds: float
    total_time_seconds: float


def _read_cache_file(path: Path, model_name: str, model_version: str, video_name: str, split: str,
                      expected_window_starts: list) -> Optional[dict]:
    """Returns the parsed cache dict if valid and consistent with the
    live model/trajectory, else None (treated as a cache miss, never a
    crash and never a silent reuse of stale/incompatible data)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"cache read failed for {path} (treating as miss): {e}")
        return None

    if data.get("cache_format_version") != CACHE_FORMAT_VERSION:
        logger.info(f"cache format version mismatch at {path}, recomputing")
        return None
    if data.get("model_name") != model_name or data.get("model_version") != model_version:
        logger.info(f"cache model/version mismatch at {path} "
                     f"(cached={data.get('model_name')}/{data.get('model_version')}, "
                     f"live={model_name}/{model_version}), recomputing")
        return None
    if data.get("video_name") != video_name or data.get("split") != split:
        logger.warning(f"cache video/split mismatch at {path}, recomputing")
        return None
    if data.get("window_starts") != expected_window_starts:
        logger.info(f"cache window_starts mismatch at {path} (trajectory changed?), recomputing")
        return None
    required = ("phase_probabilities", "next_phase_distribution")
    if any(k not in data for k in required):
        logger.warning(f"cache at {path} missing required arrays, recomputing")
        return None
    return data


def get_or_compute_forward_pass(
    model, video_name: str, split: str, traj, model_name: str, model_version: str,
) -> Tuple[np.ndarray, np.ndarray, CacheStats]:
    """Returns (posterior (T,K), next_phase_dist (T,K), stats). Checks the
    disk cache first; on a genuine miss (or any inconsistency), computes
    via the model's own, unmodified filtering()/next_phase_distribution()
    methods, persists atomically, and returns the freshly computed
    arrays -- identical code path and identical output to calling those
    methods directly, just with the result saved for next time."""
    t_start = time.time()
    cache_dir = _cache_dir(model_name, model_version, split, video_name)
    cache_file = cache_dir / "inference.json"
    lock_file = cache_dir / "inference.json.lock"

    cache_dir.mkdir(parents=True, exist_ok=True)
    window_starts = list(traj.window_starts)

    with open(lock_file, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)  # blocks until any concurrent writer for this
        # exact (model_version, split, video_name) finishes -- released automatically when
        # this `with` block exits, even on an exception, so a crash never leaves the lock held.
        try:
            t0 = time.time()
            cached = _read_cache_file(cache_file, model_name, model_version, video_name, split, window_starts)
            t_load = time.time() - t0
            if cached is not None:
                posterior = np.asarray(cached["phase_probabilities"])
                next_dist = np.asarray(cached["next_phase_distribution"])
                total = time.time() - t_start
                logger.info(f"cache_hit video={video_name} split={split} model_version={model_version} "
                            f"load_time={t_load:.4f}s total_time={total:.4f}s")
                return posterior, next_dist, CacheStats(hit=True, load_time_seconds=t_load,
                                                          compute_time_seconds=0.0, total_time_seconds=total)

            # Miss (including "lost the race" -- another process may have just written it
            # while we waited for the lock; _read_cache_file above already re-checked after
            # acquiring the lock, so if we're here it's a genuine miss).
            t0 = time.time()
            posterior = model.filtering(traj)
            next_dist = model.next_phase_distribution(traj)
            t_compute = time.time() - t0

            payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "model_name": model_name,
                "model_version": model_version,
                "video_name": video_name,
                "split": split,
                "window_starts": window_starts,
                "phase_probabilities": posterior.tolist(),
                "next_phase_distribution": next_dist.tolist(),
                "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "compute_time_seconds": t_compute,
            }
            tmp_file = cache_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(payload))
            tmp_file.replace(cache_file)  # atomic on the same filesystem -- a reader
            # never observes a partially-written file, and a crash mid-write leaves
            # only the .tmp file, never a corrupt inference.json.

            total = time.time() - t_start
            logger.info(f"cache_miss video={video_name} split={split} model_version={model_version} "
                        f"compute_time={t_compute:.4f}s total_time={total:.4f}s")
            return posterior, next_dist, CacheStats(hit=False, load_time_seconds=t_load,
                                                      compute_time_seconds=t_compute, total_time_seconds=total)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def cache_entry_exists(model_name: str, model_version: str, split: str, video_name: str) -> bool:
    return (_cache_dir(model_name, model_version, split, video_name) / "inference.json").exists()


def clear_cache() -> None:
    """Test/dev utility only -- never called by the API itself."""
    import shutil
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
