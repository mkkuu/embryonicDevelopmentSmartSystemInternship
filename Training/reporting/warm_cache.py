"""
Pre-warm driver (Phase 1.5, docs/REPORTING_CACHE.md) -- proactively
populates the disk cache for every video in a split, instead of letting
the first real user request pay the ~333s cost. Reuses the exact same
cache.get_or_compute_forward_pass() function a real API request uses;
this script is not a separate compute path, just a loop that calls it
before anyone is waiting on the answer.

Usage (from Training/):
    python -m reporting.warm_cache --split val
    python -m reporting.warm_cache --split val --videos Patient_319 Patient_1

Not run automatically by anything in this phase -- a deployment/ops
decision (cron, deploy-time hook, manual) for a later phase. Never
touches split="test" -- trajectory_service's own guard raises immediately
if ever passed one, same as every other consumer of this package.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import cache as reporting_cache  # noqa: E402
from . import model_loader, trajectory_service  # noqa: E402


def warm(split: str, videos: list[str] | None = None) -> None:
    model = model_loader.get_model()
    model_version = model_loader.model_version_string(model)
    all_videos = trajectory_service.list_videos(split)
    targets = videos if videos else all_videos
    unknown = set(targets) - set(all_videos)
    if unknown:
        raise ValueError(f"Unknown video(s) for split={split!r}: {sorted(unknown)}")

    print(f"model_version={model_version} split={split} videos_to_warm={len(targets)}")
    already_warm, newly_computed, failed = 0, 0, 0
    t_start = time.time()
    for i, video_name in enumerate(targets, 1):
        if reporting_cache.cache_entry_exists("semi_hmm", model_version, split, video_name):
            already_warm += 1
            print(f"[{i}/{len(targets)}] {video_name}: already warm, skipping")
            continue
        traj = trajectory_service.get_trajectory(video_name, split)
        t0 = time.time()
        try:
            _, _, stats = reporting_cache.get_or_compute_forward_pass(
                model, video_name, split, traj, "semi_hmm", model_version)
            newly_computed += 1
            print(f"[{i}/{len(targets)}] {video_name}: computed in {time.time() - t0:.1f}s "
                  f"(n_windows={len(traj)})")
        except Exception as e:  # noqa: BLE001 -- a single video's failure must not abort the whole warm run
            failed += 1
            print(f"[{i}/{len(targets)}] {video_name}: FAILED ({e!r})")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. already_warm={already_warm} newly_computed={newly_computed} "
          f"failed={failed} total={len(targets)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val"],
                         help="'test' is refused by trajectory_service, not accepted here either")
    parser.add_argument("--videos", nargs="*", default=None,
                         help="Specific videos to warm; omit to warm the entire split")
    args = parser.parse_args()
    warm(args.split, args.videos)
