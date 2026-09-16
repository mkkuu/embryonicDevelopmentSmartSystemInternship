"""
Typed, JSON-serializable schemas for the Reporting API's outputs.

Every field below is annotated with where it actually comes from
(docs/INFERENCE_SCHEMA.md's AVAILABLE / DERIVED / UNAVAILABLE convention,
verified against the real code in Training/evaluation/models/semi_hmm.py
and Training/evaluation/models/hmm.py this session, not assumed from
documentation). No field here corresponds to something the model cannot
actually produce -- fields that are structurally sometimes-missing (e.g.
duration for tPB2/tEB) carry an explicit `_available` boolean rather than
a fabricated placeholder value (per the project's own "explicit fallback,
never a silent no-op" discipline, already used throughout
Training/evaluation/models/*.py).

Plain dataclasses, not pydantic/marshmallow -- no new dependency, matches
the rest of the SciML extension's style (Trajectory/TrajectoryPrediction
in Training/evaluation/trajectory.py are also plain dataclasses). Every
schema has a to_dict() for the API layer to JSON-serialize directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SampleInfo:
    """A video/trajectory identity. AVAILABLE: video_name (Trajectory.video_name),
    split (caller-supplied, validated). DERIVED: sample_id (constructed,
    not a native field anywhere). patient_id is the same string as
    video_name in this dataset's own naming convention ("Patient_NNN") --
    exposed as a separate field only because a differently-named dataset
    could one day decouple them; NOT a genuinely distinct identifier
    today."""

    sample_id: str  # DERIVED: f"{video_name}#{split}"
    video_name: str  # AVAILABLE: Trajectory.video_name
    patient_id: Optional[str]  # DERIVED: video_name itself, when it matches "Patient_*"; None otherwise
    split: str  # AVAILABLE: caller-supplied ("train" or "val" only -- "test" is refused upstream)

    def to_dict(self) -> Dict:
        return {"sample_id": self.sample_id, "video_name": self.video_name,
                "patient_id": self.patient_id, "split": self.split}


@dataclass
class WindowInfo:
    """One window's identity and timing.

    AVAILABLE: window_start (Trajectory.window_starts[t] / TrajectoryPrediction
    field of the same name -- an index into the split's flattened window
    list, NOT a raw per-video frame number, see docs/INFERENCE_SCHEMA.md).
    window_start_time/window_end_time/window_mid_time/window_duration and
    time_unit come from metadata_with_time.csv (built by the HMM branch's
    own additive join, Embeddings/resnet18/{split}/metadata_with_time.csv)
    when it exists for the split -- time_available=False and all four
    *_time fields None otherwise (Train has 8/196934 rows with incomplete
    time data; Val has 0/43339, confirmed by reading both manifests this
    session, not assumed).

    frame_index is deliberately NOT a field here: window_start IS the
    identifier this schema exposes, named honestly rather than mislabeled
    "frame_index". A window_start -> real raw frame number mapping DOES
    exist (`Training/reporting/frame_mapping.py`, built + proven Phase
    7.7, served over HTTP by `Training/webapp_api/app.py`'s
    `/videos/<id>/window/<w>/frame` endpoint since Phase 7.8) -- kept
    deliberately outside this dataclass/module rather than added as a
    WindowInfo field, since it depends on the raw JPEG filesystem tree
    (`Data/embryo_dataset_F0/`) and annotation CSVs, a dependency this
    schema and the rest of the Reporting API's inference path do not
    otherwise carry. See docs/WEBAPP_VIEWER.md."""

    window_start: int  # AVAILABLE
    window_start_time: Optional[float]  # AVAILABLE when time metadata exists for this row
    window_end_time: Optional[float]  # AVAILABLE, same condition -- the causal reference point
    window_mid_time: Optional[float]  # AVAILABLE, same condition
    window_duration: Optional[float]  # AVAILABLE, same condition (observation span, NOT a phase duration)
    time_available: bool  # DERIVED: whether the four fields above are populated for this row
    time_unit: str  # AVAILABLE, constant: "unknown/unverified" -- copied verbatim from
    # metadata_with_time.manifest.json, never converted to "hours"/"seconds" by this API
    n_zero_time_diffs_in_window: Optional[int]  # AVAILABLE when time metadata exists: count of
    # repeated-timestamp frame-pairs within this window (a known raw-data artifact, see
    # metadata_with_time.csv's own column) -- exposed so a consumer can flag it, never silently
    # interpreted as an instantaneous transition

    def to_dict(self) -> Dict:
        return {
            "window_start": self.window_start,
            "window_start_time": self.window_start_time,
            "window_end_time": self.window_end_time,
            "window_mid_time": self.window_mid_time,
            "window_duration": self.window_duration,
            "time_available": self.time_available,
            "time_unit": self.time_unit,
            "n_zero_time_diffs_in_window": self.n_zero_time_diffs_in_window,
        }


@dataclass
class CurrentState:
    """The model's causal, in-progress phase estimate for one window.

    AVAILABLE via SemiHMMModel.filtering(traj)[t] -- P(S_t=j|O_1:t), a full
    15-entry posterior (verified this session by reading semi_hmm.py
    directly: filtering() exists and is causal, contrary to an earlier
    documentation pass that incorrectly said Semi-HMM had no per-window
    posterior of its own -- corrected here).
    entropy: DERIVED via HMMModel.entropy(posterior_row) (a @staticmethod,
    reused as-is from Training/evaluation/models/hmm.py, never
    reimplemented here)."""

    current_phase: str  # DERIVED: argmax(phase_probabilities)
    current_phase_index: int  # DERIVED: argmax index
    phase_probability: float  # DERIVED: phase_probabilities[current_phase]
    phase_probabilities: Dict[str, float]  # AVAILABLE: SemiHMMModel.filtering()[t], all 15 states
    entropy: float  # DERIVED: HMMModel.entropy(phase_probabilities), nats

    def to_dict(self) -> Dict:
        return {
            "current_phase": self.current_phase,
            "current_phase_index": self.current_phase_index,
            "phase_probability": self.phase_probability,
            "phase_probabilities": self.phase_probabilities,
            "entropy": self.entropy,
        }


@dataclass
class NextPhaseInfo:
    """One-step-ahead prediction. AVAILABLE via
    SemiHMMModel.next_phase_distribution(traj)[t] -- P(S_{t+1}=j|O_1:t),
    duration-aware (accounts for the current segment's already-elapsed
    age), causal. NOT the same quantity as HMMModel.explain()'s
    next_phase_probability (which uses a fixed hazard, not duration-aware)
    -- this API always uses the Semi-HMM's own duration-aware version,
    documented here so a consumer never confuses the two."""

    next_phase_distribution: Dict[str, float]  # AVAILABLE, all 15 states
    most_likely_next_phase: str  # DERIVED: argmax
    next_phase_probability: float  # DERIVED: next_phase_distribution[most_likely_next_phase]

    def to_dict(self) -> Dict:
        return {
            "next_phase_distribution": self.next_phase_distribution,
            "most_likely_next_phase": self.most_likely_next_phase,
            "next_phase_probability": self.next_phase_probability,
        }


@dataclass
class DurationInfo:
    """Duration for the CURRENT phase (a property of the phase, not the
    window -- recomputed only when current_phase changes in practice, but
    always included per-window for API simplicity).

    duration_available=False (not expected_duration=0) for tPB2/tEB --
    verified this session: EmpiricalDurationModel.fallback_uniform=True is
    set explicitly by SemiHMMModel.fit() for any phase with zero observed
    interior segments (Training/evaluation/models/semi_hmm.py, the "Phase
    A2" fallback), and this API checks that exact flag rather than
    inferring unavailability from the numbers themselves."""

    duration_available: bool  # DERIVED: not (isinstance(dm, EmpiricalDurationModel) and dm.fallback_uniform)
    expected_duration: Optional[float]  # AVAILABLE via SemiHMMModel.expected_duration(phase) when duration_available
    duration_quantiles: Optional[Dict[str, int]]  # AVAILABLE via the duration model's own .quantiles([...])
    # when duration_available -- keys are string-formatted quantile levels ("0.1".."0.9"), values in windows
    duration_unit: str  # AVAILABLE, constant: "windows" -- NEVER converted to a time unit (window_duration's
    # own time_unit is itself "unknown/unverified"; converting durations-in-windows to
    # durations-in-time would require multiplying by an unverified per-window time delta -- not done)

    def to_dict(self) -> Dict:
        return {
            "duration_available": self.duration_available,
            "expected_duration": self.expected_duration,
            "duration_quantiles": self.duration_quantiles,
            "duration_unit": self.duration_unit,
        }


@dataclass
class ModelInfo:
    """Provenance for one inference record -- lets any later consumer
    answer "which model produced this, from what config, when".
    AVAILABLE fields are read directly from the loaded model's own
    attributes (all persisted in state.json / re-derivable from it).
    inference_timestamp is DERIVED (wall-clock time this record was
    computed, not a property of the model)."""

    model_name: str  # AVAILABLE, constant: "semi_hmm"
    model_version: str  # DERIVED: short config fingerprint, e.g. "dmax=268,negative_binomial,alpha=1.0,rho=0.3,class_weight=None"
    model_configuration: Dict  # AVAILABLE: dmax, duration_family, transition_smoothing_alpha,
    # transition_decay_rho, logreg_class_weight, n_states -- all attributes on the loaded SemiHMMModel
    model_source: str  # AVAILABLE, constant: "Results/evaluation/semi_hmm_weekend_phaseF/model/"
    inference_timestamp: str  # DERIVED: ISO 8601, computed at request time

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name, "model_version": self.model_version,
            "model_configuration": self.model_configuration,
            "model_source": self.model_source,
            "inference_timestamp": self.inference_timestamp,
        }


@dataclass
class GroundTruthInfo:
    """Real/annotated phase, when available -- OPTIONAL, only present for
    trajectories with real annotations (i.e. every video in this cached
    dataset, since Embryo_Transition_Dataset only ever includes annotated
    windows -- but kept optional/nullable in the schema for honesty about
    what the API can express, not what this specific dataset happens to
    always provide)."""

    ground_truth_phase: Optional[str]  # AVAILABLE: Trajectory.last_frame_phase[t], mapped to phase name
    consistency_flag: Optional[int]  # AVAILABLE: Trajectory.consistency_flag[t], ground truth 0/1

    def to_dict(self) -> Dict:
        return {"ground_truth_phase": self.ground_truth_phase, "consistency_flag": self.consistency_flag}


@dataclass
class InferenceRecord:
    """The canonical per-window inference record -- what
    inference_service.infer() returns and what /videos/{id}/inference/{window}
    serves. Composes every schema above; never adds a field none of them
    defines."""

    sample: SampleInfo
    window: WindowInfo
    current_state: CurrentState
    next_phase: NextPhaseInfo
    duration: DurationInfo
    model: ModelInfo
    ground_truth: GroundTruthInfo
    warnings: List[str] = field(default_factory=list)  # DERIVED: e.g. "duration not estimable
    # (structural censoring)", "time metadata incomplete for this window", "repeated timestamps
    # detected in this window (raw-data artifact, not an instantaneous transition)"

    def to_dict(self) -> Dict:
        return {
            "sample": self.sample.to_dict(),
            "window": self.window.to_dict(),
            "current_state": self.current_state.to_dict(),
            "next_phase": self.next_phase.to_dict(),
            "duration": self.duration.to_dict(),
            "model": self.model.to_dict(),
            "ground_truth": self.ground_truth.to_dict(),
            "warnings": self.warnings,
        }


@dataclass
class TrajectorySummary:
    """Summary of one full trajectory -- backs GET /videos/{id}/trajectory.
    n_windows/window_starts: AVAILABLE from Trajectory itself.
    transition_chain: AVAILABLE via SemiHMMModel.transition_chain(traj) --
    a RETROSPECTIVE, per-ground-truth-segment diagnostic (needs the full
    trajectory), never served for a "live" in-progress sample by this
    endpoint's causal siblings."""

    sample: SampleInfo
    n_windows: int
    window_starts: List[int]
    transition_chain: List[Dict]  # AVAILABLE: SemiHMMModel.transition_chain(traj), verbatim
    model: ModelInfo

    def to_dict(self) -> Dict:
        return {
            "sample": self.sample.to_dict(),
            "n_windows": self.n_windows,
            "window_starts": self.window_starts,
            "transition_chain": self.transition_chain,
            "model": self.model.to_dict(),
        }


@dataclass
class TransitionEvent:
    """One OBSERVED phase transition between two consecutive ground-truth
    segments -- Event/Transition RAG axis, P1
    (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 16 items 1-2,
    docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md). Built by
    Training/reporting/transition_events.py from
    HMMModel._segments(traj) (reused, never reimplemented) and
    Trajectory.window_starts (both already-existing, read-only sources) --
    never a model prediction, never touches HMM/Semi-HMM training logic.

    `observed`/`provenance` are ALWAYS True/"observed_annotation" -- this
    dataclass deliberately carries NO model-derived field (no duration
    likelihood, no next-phase distribution). A caller wanting the model's
    own view around the same window range calls get_current_inference
    separately -- kept as two distinct tool calls/context entries by
    design, so the LLM/grounding never has to disentangle observed-vs-
    predicted fields living in one blended structure (the gap identified
    in SemiHMMModel.transition_chain()'s own dict shape,
    docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 6).

    Direct cleavage / reversal / fragmentation / multinucleation are
    NEVER represented here -- no such field exists on this dataclass at
    all (confirmed absent from this project's real data,
    docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 3), rather than a null/empty
    placeholder that could be misread as "checked, not present for this
    video" for those specific concepts.

    phase_distance is SIGNED (to_phase_index - from_phase_index) --
    never assumed positive. The frozen Semi-HMM's own training data showed
    zero backward (negative) segment-transitions across all 704 videos
    (Training/evaluation/models/hmm.py, Phase 2 empirical result, cited
    not re-derived) -- this dataclass does not encode that as a
    constraint, only reports what `_segments()` actually found for THIS
    trajectory, honestly, whatever sign it has."""

    from_phase: str
    to_phase: str
    from_phase_index: int  # index into the loaded model's own state_names
    to_phase_index: int
    observed: bool  # ALWAYS True
    provenance: str  # ALWAYS "observed_annotation"
    window_start: int  # window_start of the LAST window still annotated from_phase
    window_end: int  # window_start of the FIRST window annotated to_phase
    frame_start: Optional[int]  # best-effort: last real raw frame of the window_start window
    frame_end: Optional[int]  # best-effort: first real raw frame of the window_end window
    frame_mapping_available: bool  # False when frame_mapping.py could not resolve real
    # frames for this video/split (e.g. raw Data/ not present in this environment) --
    # frame_start/frame_end are None, never fabricated, when this is False
    #
    # KNOWN, DOCUMENTED ARTIFACT (found via real Patient_319 Val data, not
    # hypothetical): frame_end is NOT guaranteed >= frame_start numerically.
    # window_start and window_end are usually adjacent LOCAL window indices
    # (window_size=8, stride=1), so they share up to 7 of their 8 real
    # frames -- window_start's LAST frame and window_end's FIRST frame can
    # therefore land in either order in absolute frame-number terms,
    # depending on exactly where the two windows' frame ranges overlap.
    # This is a real windowing-stride artifact, not a bug, and is NOT
    # silently reordered/corrected here (same "report raw-data artifacts
    # honestly" discipline as WindowInfo.n_zero_time_diffs_in_window) --
    # never treat (frame_start, frame_end) as a clean ascending interval.
    window_start_time: Optional[float]
    window_end_time: Optional[float]
    time_available: bool  # same convention as WindowInfo.time_available -- False (not a
    # fabricated 0.0) when metadata_with_time.csv has no row for one of the two windows
    phase_distance: int  # to_phase_index - from_phase_index, SIGNED
    is_skip: bool  # abs(phase_distance) > 1

    def to_dict(self) -> Dict:
        return {
            "from_phase": self.from_phase, "to_phase": self.to_phase,
            "from_phase_index": self.from_phase_index, "to_phase_index": self.to_phase_index,
            "observed": self.observed, "provenance": self.provenance,
            "window_start": self.window_start, "window_end": self.window_end,
            "frame_start": self.frame_start, "frame_end": self.frame_end,
            "frame_mapping_available": self.frame_mapping_available,
            "window_start_time": self.window_start_time, "window_end_time": self.window_end_time,
            "time_available": self.time_available,
            "phase_distance": self.phase_distance, "is_skip": self.is_skip,
        }


@dataclass
class InferenceHistoryEntry:
    """One window inside an InferenceHistory band -- the MODEL-DERIVED
    per-window quantities plus, kept clearly separate, the OBSERVED
    ground-truth fields for the same window.

    Every field is a slice of the exact same InferenceRecord that
    inference_service.infer() already produces for this window (same
    frozen model, same cached forward pass) -- this schema NEVER
    recomputes anything, it only re-packages a range of already-computed
    per-window records into one temporally-ordered list so a consumer can
    see how the belief EVOLVES, not just its value at one instant.

    MODEL-DERIVED: current_phase(+index), phase_probability,
    phase_probabilities (full posterior), entropy, most_likely_next_phase,
    next_phase_probability, next_phase_distribution.
    OBSERVED (annotation, never a prediction): ground_truth_phase,
    consistency_flag. These are physically separate keys here for the same
    reason TransitionEvent carries no model field -- so a caller/LLM
    never has to disentangle observed-vs-predicted inside one blob.
    DERIVED for this schema only: is_center, offset_from_center."""

    window_start: int  # AVAILABLE (Trajectory.window_starts[t])
    offset_from_center: int  # DERIVED: this entry's window index minus the center window's index
    is_center: bool  # DERIVED
    window_start_time: Optional[float]  # AVAILABLE when time metadata exists (else None)
    window_end_time: Optional[float]
    time_available: bool
    # --- MODEL-DERIVED ---
    current_phase: str
    current_phase_index: int
    phase_probability: float
    phase_probabilities: Dict[str, float]
    entropy: float
    most_likely_next_phase: str
    next_phase_probability: float
    next_phase_distribution: Dict[str, float]
    # --- OBSERVED (annotation) ---
    ground_truth_phase: Optional[str]
    consistency_flag: Optional[int]

    def to_dict(self) -> Dict:
        return {
            "window_start": self.window_start,
            "offset_from_center": self.offset_from_center,
            "is_center": self.is_center,
            "window_start_time": self.window_start_time,
            "window_end_time": self.window_end_time,
            "time_available": self.time_available,
            "model_derived": {
                "current_phase": self.current_phase,
                "current_phase_index": self.current_phase_index,
                "phase_probability": self.phase_probability,
                "phase_probabilities": self.phase_probabilities,
                "entropy": self.entropy,
                "most_likely_next_phase": self.most_likely_next_phase,
                "next_phase_probability": self.next_phase_probability,
                "next_phase_distribution": self.next_phase_distribution,
            },
            "observed": {
                "ground_truth_phase": self.ground_truth_phase,
                "consistency_flag": self.consistency_flag,
            },
        }


@dataclass
class InferenceHistory:
    """A temporally-ordered slice of per-window inference records around a
    center window -- backs get_inference_history() (the multi-window
    capability the single-window infer() could not provide). Answers "how
    did the model's belief / uncertainty EVOLVE across these windows",
    which no single InferenceRecord can.

    `requested_before`/`requested_after` are what the caller asked for;
    `n_windows_before`/`n_windows_after` are what was actually returned
    after clamping to the real bounds of this ONE video's window list
    (never spilling into another video, never fabricating windows). A
    warning is added whenever clamping shortened either side.

    entries are ascending by window_start (earliest first). center_window
    is guaranteed present in entries (is_center=True on exactly one)."""

    sample: SampleInfo
    center_window: int
    requested_before: int
    requested_after: int
    n_windows_before: int  # actual, after clamping to this video's real window list
    n_windows_after: int
    n_windows: int  # len(entries) == n_windows_before + 1 + n_windows_after
    window_starts: List[int]  # the window_start of every entry, ascending
    entries: List[InferenceHistoryEntry]
    model: ModelInfo
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sample": self.sample.to_dict(),
            "center_window": self.center_window,
            "requested_before": self.requested_before,
            "requested_after": self.requested_after,
            "n_windows_before": self.n_windows_before,
            "n_windows_after": self.n_windows_after,
            "n_windows": self.n_windows,
            "window_starts": self.window_starts,
            "entries": [e.to_dict() for e in self.entries],
            "model": self.model.to_dict(),
            "warnings": self.warnings,
        }
