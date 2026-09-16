"""
Composes model_loader + trajectory_service + SemiHMMModel's own
already-implemented methods into canonical InferenceRecord/TrajectorySummary
objects. Never reimplements any HMM/Semi-HMM mathematics -- every
probability in this module comes from calling the frozen model's own
methods (filtering, next_phase_distribution, expected_duration, duration
model .quantiles(), transition_chain) and HMMModel.entropy (a
@staticmethod, imported and reused as-is).

Caching (Phase 1.5, docs/REPORTING_CACHE.md): the expensive part -- the
forward pass over an entire trajectory (SemiHMMModel.filtering()/
next_phase_distribution()) -- is computed at most ONCE per
(model_version, split, video_name), ever, via a two-layer cache: an
in-process dict (fastest, lost on process restart) backed by a
disk-persisted cache (survives restarts, shared across processes on the
same machine, see cache.py). A request for a single window only ever
pays for building the response dict from an already-available (T,K)
array, never for re-running the forward algorithm, after the first
request for that video (cache miss) has completed once.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from evaluation.models.hmm import HMMModel  # noqa: E402
from evaluation.models.semi_hmm import EmpiricalDurationModel  # noqa: E402

from . import cache as reporting_cache  # noqa: E402
from . import model_loader, trajectory_service  # noqa: E402
from .schemas import (  # noqa: E402
    CurrentState, DurationInfo, GroundTruthInfo, InferenceHistory, InferenceHistoryEntry,
    InferenceRecord, ModelInfo, NextPhaseInfo, SampleInfo, TrajectorySummary, WindowInfo,
)

DURATION_QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Phase names are read from the loaded model's OWN state_names (never a
# hardcoded module-level constant) -- the model is the single source of
# truth for how many states it has and what they're called. For the real
# frozen reference this is exactly DEFAULT_PHASE_NAMES (n_states=15), but
# reading it from the model keeps this correct in principle and testable
# against smaller synthetic models.


class WindowNotFoundError(KeyError):
    pass


_POSTERIOR_CACHE: Dict[Tuple[str, str], np.ndarray] = {}  # (video_name, split) -> filtering() output (T,K)
_NEXT_PHASE_CACHE: Dict[Tuple[str, str], np.ndarray] = {}  # (video_name, split) -> next_phase_distribution() (T,K)


def _forward_pass(video_name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Two-layer cache: L1 (this process's dict, fastest, volatile) in
    front of L2 (disk, cache.py, survives restarts). L1 is keyed by
    (video_name, split) only -- safe because a single running process
    only ever has ONE model loaded (model_loader.get_model()'s own
    process-lifetime cache), so within one process there is only ever one
    possible model_version; L2 (disk) is keyed by model_version too,
    since it must remain correct across restarts where the served model
    could in principle change."""
    key = (video_name, split)
    if key in _POSTERIOR_CACHE and key in _NEXT_PHASE_CACHE:
        return _POSTERIOR_CACHE[key], _NEXT_PHASE_CACHE[key]
    model = model_loader.get_model()
    traj = trajectory_service.get_trajectory(video_name, split)
    model_version = model_loader.model_version_string(model)
    posterior, next_dist, _stats = reporting_cache.get_or_compute_forward_pass(
        model, video_name, split, traj, model_name="semi_hmm", model_version=model_version,
    )
    _POSTERIOR_CACHE[key] = posterior
    _NEXT_PHASE_CACHE[key] = next_dist
    return posterior, next_dist


_DURATION_INFO_CACHE: Dict[int, DurationInfo] = {}  # phase_index -> DurationInfo. Safe under the
# same "one model per process" invariant _POSTERIOR_CACHE/_NEXT_PHASE_CACHE above already rely on
# (see _forward_pass's own docstring) -- _duration_info's result depends ONLY on (model, phase_index),
# never on window_start/video_name/split, so this is a pure-function memoization: it can never change
# what a caller receives, only how many times the underlying (measured, expensive -- ~527 scipy
# negative-binomial evaluations per call, dominant cost of a per-video timeline built by calling
# infer() once per window, docs/WEBAPP_TIMELINE_PERFORMANCE.md) computation actually runs. A process
# only ever has ~15 distinct phase_index values, so this turns an O(n_windows) cost into O(n_phases).


def _duration_info(model, phase_index: int) -> DurationInfo:
    if phase_index in _DURATION_INFO_CACHE:
        return _DURATION_INFO_CACHE[phase_index]
    dm = model.duration_models[phase_index]
    fallback = isinstance(dm, EmpiricalDurationModel) and getattr(dm, "fallback_uniform", False)
    if fallback:
        result = DurationInfo(duration_available=False, expected_duration=None,
                               duration_quantiles=None, duration_unit="windows")
    else:
        quantiles = dm.quantiles(DURATION_QUANTILE_LEVELS)
        result = DurationInfo(
            duration_available=True,
            expected_duration=float(model.expected_duration(phase_index)),
            duration_quantiles={f"{q:g}": int(v) for q, v in quantiles.items()},
            duration_unit="windows",
        )
    _DURATION_INFO_CACHE[phase_index] = result
    return result


def infer(video_name: str, split: str, window_start: int) -> InferenceRecord:
    """The canonical single-window inference call. Raises WindowNotFoundError
    if window_start isn't in this trajectory -- never silently substitutes
    the nearest window."""
    trajectory_service._check_split(split)
    model = model_loader.get_model()
    phase_names = model.state_names
    traj = trajectory_service.get_trajectory(video_name, split)
    if window_start not in traj.window_starts:
        raise WindowNotFoundError(f"window_start={window_start} not found for video_name={video_name!r} "
                                   f"in split={split!r}.")
    t = traj.window_starts.index(window_start)

    posterior, next_dist = _forward_pass(video_name, split)
    phase_probs = posterior[t]
    current_idx = int(np.argmax(phase_probs))
    phase_probabilities = {phase_names[i]: float(phase_probs[i]) for i in range(len(phase_names))}

    next_probs = next_dist[t]
    next_idx = int(np.argmax(next_probs))
    next_phase_distribution = {phase_names[i]: float(next_probs[i]) for i in range(len(phase_names))}

    duration = _duration_info(model, current_idx)

    time_info = trajectory_service.get_window_time(video_name, split, window_start)
    gt_phase_idx, gt_consistency = trajectory_service.get_ground_truth(video_name, split, window_start)

    warnings = []
    if not duration.duration_available:
        warnings.append(f"duration not estimable for phase {phase_names[current_idx]!r} "
                         f"(structural censoring -- 0 observed interior segments in Train)")
    if not time_info["time_available"]:
        warnings.append("time metadata unavailable for this window")
    elif time_info["n_zero_time_diffs_in_window"]:
        warnings.append(f"{time_info['n_zero_time_diffs_in_window']} repeated-timestamp frame-pairs "
                         f"detected in this window (raw-data artifact, not an instantaneous transition)")

    return InferenceRecord(
        sample=SampleInfo(
            sample_id=f"{video_name}#{split}",
            video_name=video_name,
            patient_id=video_name if video_name.startswith("Patient_") else None,
            split=split,
        ),
        window=WindowInfo(
            window_start=window_start,
            window_start_time=time_info["window_start_time"],
            window_end_time=time_info["window_end_time"],
            window_mid_time=time_info["window_mid_time"],
            window_duration=time_info["window_duration"],
            time_available=time_info["time_available"],
            time_unit=time_info["time_unit"],
            n_zero_time_diffs_in_window=time_info["n_zero_time_diffs_in_window"],
        ),
        current_state=CurrentState(
            current_phase=phase_names[current_idx],
            current_phase_index=current_idx,
            phase_probability=float(phase_probs[current_idx]),
            phase_probabilities=phase_probabilities,
            entropy=float(HMMModel.entropy(phase_probs)),
        ),
        next_phase=NextPhaseInfo(
            next_phase_distribution=next_phase_distribution,
            most_likely_next_phase=phase_names[next_idx],
            next_phase_probability=float(next_probs[next_idx]),
        ),
        duration=duration,
        model=_model_info(model),
        ground_truth=GroundTruthInfo(
            ground_truth_phase=phase_names[gt_phase_idx] if gt_phase_idx is not None else None,
            consistency_flag=gt_consistency,
        ),
        warnings=warnings,
    )


DEFAULT_HISTORY_BEFORE = 5
DEFAULT_HISTORY_AFTER = 5
# Per-side sanity cap. This tool returns a LOCAL band around a transition
# (Q5/Q6/Q8/Q10 default to 5/5); anything beyond this is silently reduced
# (with a warning) rather than erroring -- "not enough windows" and "far
# more than needed" are both handled gracefully, never a crash, never a
# whole-trajectory dump built by a pathological before=10**7. For
# trajectory-level structure use get_trajectory.
MAX_HISTORY_HALF_SPAN = 200

# ETAPE 4 (context compactness). A history band carries, PER WINDOW, a
# 15-entry phase_probabilities AND a 15-entry next_phase_distribution.
# prompt._flatten_dynamic_context renders one dotted path per leaf, so an
# 11-window band alone becomes ~330 long lines -- measured at 12.7k-15.5k
# estimated tokens for Q5/Q6/Q7/Q8/Q10 against a deployment ctx-size of
# 4096. Almost all of that mass is near-zero noise: the question ("do the
# two phases' probabilities converge", "is the prediction stable") is
# answered by the leading few phases and the entropy, never by the 11th
# phase's 1e-4.
#
# top_k_probabilities keeps the k HIGHEST-probability entries of each
# distribution, verbatim (real phase names, unrounded values), and drops
# the rest. It NEVER touches entropy / phase_probability /
# next_phase_probability -- those are computed upstream from the FULL
# distribution, so the scalar summary of the discarded tail survives
# intact. Default None = full distribution, so every existing caller and
# test is byte-identical; only the orchestrator's own call site opts in.
DEFAULT_HISTORY_TOP_K_PROBABILITIES = None


def _top_k_distribution(dist, k):
    """-> (kept, omitted_mass, n_omitted). Ties broken by phase name so the
    result is deterministic. k=None/<=0 or already short enough -> unchanged."""
    if k is None or k <= 0 or len(dist) <= k:
        return dist, 0.0, 0
    ranked = sorted(dist.items(), key=lambda kv: (-kv[1], str(kv[0])))
    kept = dict(ranked[:k])
    omitted = ranked[k:]
    return kept, float(sum(v for _, v in omitted)), len(omitted)


def infer_history(
    video_name: str,
    split: str,
    center_window: int,
    before: int = DEFAULT_HISTORY_BEFORE,
    after: int = DEFAULT_HISTORY_AFTER,
    top_k_probabilities: Optional[int] = DEFAULT_HISTORY_TOP_K_PROBABILITIES,
) -> InferenceHistory:
    """MULTI-WINDOW capability (G1). Returns the per-window InferenceRecord
    quantities for the band of real windows [center-before, center+after]
    of THIS video, temporally ordered, so a consumer can see how the
    model's belief / uncertainty EVOLVES -- something no single infer()
    call can express.

    NEVER recomputes: every entry is a slice of the same InferenceRecord
    infer() already builds for that window, and infer() itself only pays
    the forward-pass cost once per (video, split) (see _forward_pass).

    Clamps to this video's real window list -- `before`/`after` past the
    start/end of the trajectory are silently reduced (with a warning),
    never satisfied by spilling into another video or by fabricating
    windows. Raises WindowNotFoundError if center_window is not a real
    window of this video; ValueError for a nonsensical band."""
    trajectory_service._check_split(split)
    if before < 0 or after < 0:
        raise ValueError(f"before/after must be >= 0, got before={before}, after={after}.")
    if before + after < 1:
        raise ValueError("before + after must be >= 1 (an inference history of one window is just infer()).")
    if top_k_probabilities is not None and top_k_probabilities < 1:
        raise ValueError(f"top_k_probabilities must be >= 1 or None (full), got {top_k_probabilities}.")

    warnings: list = []
    if before > MAX_HISTORY_HALF_SPAN or after > MAX_HISTORY_HALF_SPAN:
        warnings.append(
            f"requested band ({before} before / {after} after) exceeds the per-side cap "
            f"{MAX_HISTORY_HALF_SPAN}; reduced to it (this tool returns a local band around a "
            f"transition, not a whole-trajectory dump -- use get_trajectory for that)."
        )
        before = min(before, MAX_HISTORY_HALF_SPAN)
        after = min(after, MAX_HISTORY_HALF_SPAN)

    traj = trajectory_service.get_trajectory(video_name, split)
    window_starts = list(traj.window_starts)
    if center_window not in window_starts:
        raise WindowNotFoundError(
            f"center_window={center_window} not found for video_name={video_name!r} in split={split!r}."
        )
    ci = window_starts.index(center_window)
    lo = max(0, ci - before)
    hi = min(len(window_starts) - 1, ci + after)

    n_before = ci - lo
    n_after = hi - ci
    if n_before < before:
        warnings.append(
            f"requested {before} window(s) before the center, only {n_before} available "
            f"(start of this video's trajectory)."
        )
    if n_after < after:
        warnings.append(
            f"requested {after} window(s) after the center, only {n_after} available "
            f"(end of this video's trajectory)."
        )

    model = model_loader.get_model()
    entries: list = []
    any_time_missing = False
    max_omitted_mass = 0.0
    n_truncated_entries = 0
    for idx in range(lo, hi + 1):
        w = window_starts[idx]
        rec = infer(video_name, split, w)  # cached forward pass -- O(1) after the first call for this video
        cs = rec.current_state
        nx = rec.next_phase
        gt = rec.ground_truth
        wd = rec.window
        if not wd.time_available:
            any_time_missing = True
        phase_probs, pp_omitted_mass, pp_n_omitted = _top_k_distribution(
            cs.phase_probabilities, top_k_probabilities)
        next_dist, nd_omitted_mass, nd_n_omitted = _top_k_distribution(
            nx.next_phase_distribution, top_k_probabilities)
        if pp_n_omitted or nd_n_omitted:
            max_omitted_mass = max(max_omitted_mass, pp_omitted_mass, nd_omitted_mass)
            n_truncated_entries += 1
        entries.append(InferenceHistoryEntry(
            window_start=w,
            offset_from_center=idx - ci,
            is_center=(idx == ci),
            window_start_time=wd.window_start_time,
            window_end_time=wd.window_end_time,
            time_available=wd.time_available,
            current_phase=cs.current_phase,
            current_phase_index=cs.current_phase_index,
            phase_probability=cs.phase_probability,
            phase_probabilities=phase_probs,
            entropy=cs.entropy,
            most_likely_next_phase=nx.most_likely_next_phase,
            next_phase_probability=nx.next_phase_probability,
            next_phase_distribution=next_dist,
            ground_truth_phase=gt.ground_truth_phase,
            consistency_flag=gt.consistency_flag,
        ))

    if n_truncated_entries:
        warnings.append(
            f"phase_probabilities and next_phase_distribution truncated to the top "
            f"{top_k_probabilities} phase(s) per window in {n_truncated_entries} entry/entries "
            f"(context-compactness limit); largest probability mass omitted in any single "
            f"distribution: {max_omitted_mass:.4f}. entropy, phase_probability and "
            f"next_phase_probability are computed from the FULL distribution and are unaffected."
        )
    if any_time_missing:
        warnings.append("time metadata unavailable for at least one window in this band "
                         "(per-entry time_available flags which).")

    return InferenceHistory(
        sample=SampleInfo(
            sample_id=f"{video_name}#{split}",
            video_name=video_name,
            patient_id=video_name if video_name.startswith("Patient_") else None,
            split=split,
        ),
        center_window=center_window,
        requested_before=before,
        requested_after=after,
        n_windows_before=n_before,
        n_windows_after=n_after,
        n_windows=len(entries),
        window_starts=[e.window_start for e in entries],
        entries=entries,
        model=_model_info(model),
        warnings=warnings,
    )


def trajectory_summary(video_name: str, split: str) -> TrajectorySummary:
    """RETROSPECTIVE (needs the full trajectory) -- backs
    GET /videos/{id}/trajectory. Never served for a partial/live sample by
    this function's causal sibling (infer(), above, which only ever looks
    at O_1..O_t for the requested window)."""
    model = model_loader.get_model()
    traj = trajectory_service.get_trajectory(video_name, split)
    chain = model.transition_chain(traj)
    return TrajectorySummary(
        sample=SampleInfo(
            sample_id=f"{video_name}#{split}", video_name=video_name,
            patient_id=video_name if video_name.startswith("Patient_") else None, split=split,
        ),
        n_windows=len(traj),
        window_starts=list(traj.window_starts),
        transition_chain=chain,
        model=_model_info(model),
    )


def _model_info(model) -> ModelInfo:
    return ModelInfo(
        model_name="semi_hmm",
        model_version=model_loader.model_version_string(model),
        model_configuration=model_loader.model_configuration(model),
        model_source=str(model_loader.REFERENCE_MODEL_PATH),
        inference_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
