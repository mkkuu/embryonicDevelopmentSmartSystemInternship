"""
SemiHMMModel -- explicit-duration extension of HMMModel
(Training/evaluation/models/hmm.py), per docs/PROJECT_STATE.md's
"SEMI-HMM -- EXPERIMENTAL PROTOCOL" checkpoint (justified by the HMM
duration-heterogeneity diagnostics: Brier ~ mean_duration | n_segments,
partial Pearson r=-0.7729, p=0.0032, n=13, robust to dropping tPNf/t3
individually and together -- see that checkpoint for the full derivation).

PURELY ADDITIVE: hmm.py is never modified. It is only READ from
(HMMModel._segments(), DEFAULT_PHASE_NAMES) and COMPOSED (this module's
emission model is a plain HMMModel instance, fit once and used only for
its _log_emission()/_scaler/_logreg/_state_log_prior -- reusing the
validated StandardScaler + LogisticRegression + Bayes-conversion pipeline
verbatim rather than duplicating ~30 lines of it, which would risk silent
divergence from the validated convention). No historical HMM result,
embedding, or metadata is touched by importing or using this module.

INFRASTRUCTURE-ONLY, this session: no real training run, no Test split
ever referenced anywhere in this file, no modification to hmm.py or any
Results/evaluation/e1_hmm_* artifact.

Scientific formulation
-----------------------
Classic HMM: P(S_t|S_{t-1}) with duration encoded implicitly, geometrically,
via the self-loop A[i,i] -- a single decay rate per state, unable to
represent the observed dwell-time heterogeneity (CV 0.18-1.25, skewness up
to 5.97, Results/evaluation/e1_hmm_phase2_analysis/report.json).

Semi-HMM here: P(S_1:K, D_1:K, O_1:T) =
    pi(S_1) * P(D_1|S_1) * prod_{k=2}^K [P(S_k|S_{k-1}) * P(D_k|S_k)]
             * prod_{t=1}^T b_{S(t)}(O_t)

Transitions are now SEGMENT-to-SEGMENT and strictly j>i (no self-loop cell
in A at all -- see DEFAULT_PHASE_NAMES's chronological ordering, same
"never backward" constraint as the classic HMM, just without the j==i
case, since that is now fully owned by the duration model). Duration is a
per-state, explicit, discrete distribution D in {1,...,dmax} (unit:
WINDOWS -- see this module's own duration-unit note below), with the mass
at exactly `dmax` absorbing the tail P(D>=dmax) of the underlying family
(a standard truncation convention for tractable explicit-duration models --
documented, not hidden, including its one known simplification: an
observation genuinely ending at exactly dmax is fit identically to one
that was clipped down to dmax -- see NegativeBinomialDurationModel's
module-level note).

Duration unit: WINDOWS, not frames or real elapsed time -- matches the
whole existing HMM/embedding pipeline's own per-window time-step
convention (hmm.py's own module docstring: "one window = one HMM time
step"). Real elapsed time (metadata_with_time.csv) remains a reporting-only
derived quantity until its raw unit is confirmed (flagged, unresolved,
since Phase 0/1 of the HMM branch) -- never used as the modeling unit here.

Segment/censoring convention: a segment is a maximal run of consecutive
windows sharing the same last_frame_phase within one trajectory -- reused
verbatim via HMMModel._segments(), never reimplemented. A segment is
LEFT_CENSORED iff it is a video's first segment (true start may precede
recording start), RIGHT_CENSORED iff it is the video's last segment
(recording stopped before the phase truly ended) -- mirrors Phase 2's own
is_first_segment/is_last_segment criterion exactly (Results/evaluation/
e1_hmm_phase2_analysis/report.json). BOTH censoring directions are treated,
in the duration likelihood, with the SAME survival contribution P(D>=d_obs)
-- in both cases the true duration is only known to be at least the
observed one; this is a documented MODELING DECISION made in this session
(Phase 2 never fit a censored likelihood, it only labeled which segments
are censored), not a convention copied from existing code, because none
existed to copy for this specific question.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import nbinom

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction
from .hmm import DEFAULT_PHASE_NAMES, HMMModel

_NEG_INF = -1e10


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        out = np.log(x)
    out[~np.isfinite(out)] = _NEG_INF
    return out


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------

class CensoringType(Enum):
    OBSERVED = "observed"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    BOTH_CENSORED = "both_censored"  # single-segment video: first AND last at once


def classify_segment_censoring(position: int, n_segments: int) -> CensoringType:
    """position: 0-indexed position of this segment within its video's full
    segment sequence (ALL phases, not just one). n_segments: total segments
    in that video. Mirrors Phase 2's is_first_segment/is_last_segment
    criterion exactly (see this module's docstring)."""
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}.")
    if not (0 <= position < n_segments):
        raise ValueError(f"position={position} out of range for n_segments={n_segments}.")
    is_first = position == 0
    is_last = position == n_segments - 1
    if is_first and is_last:
        return CensoringType.BOTH_CENSORED
    if is_first:
        return CensoringType.LEFT_CENSORED
    if is_last:
        return CensoringType.RIGHT_CENSORED
    return CensoringType.OBSERVED


def extract_phase_durations(
    trajectories: List[Trajectory],
) -> Dict[int, List[Tuple[int, CensoringType]]]:
    """Per phase index: list of (duration_in_windows, censoring_type),
    built via HMMModel._segments() (reused verbatim, never reimplemented)."""
    out: Dict[int, List[Tuple[int, CensoringType]]] = {}
    for traj in trajectories:
        if len(traj) == 0:
            continue
        segs = HMMModel._segments(traj)  # [(phase_idx, start_row, end_row), ...]
        n = len(segs)
        for pos, (phase_idx, start, end) in enumerate(segs):
            duration = end - start + 1
            censoring = classify_segment_censoring(pos, n)
            out.setdefault(phase_idx, []).append((duration, censoring))
    return out


def determine_dmax(
    durations_by_phase: Dict[int, List[Tuple[int, CensoringType]]],
    strategy: str = "observed_max",
    quantile: float = 0.99,
    explicit_value: Optional[int] = None,
) -> int:
    """Dmax from whatever duration data is PASSED IN -- this function has no
    way to know which split its input came from, so it cannot itself
    enforce "Train only"; every caller in this codebase must pass Train
    durations only (mirrors how transition_smoothing_alpha/rho are
    Val-selected but the embeddings/labels themselves are always Train-only
    for fit()). strategy: "observed_max" (max over ALL provided durations,
    any censoring type -- a censored duration is still a real lower bound
    worth respecting), "quantile" (given `quantile`, e.g. 0.99, over all
    provided durations), "explicit" (explicit_value used directly; still
    checked against the data so a silently-too-small choice is visible to
    the caller via the returned dict, never hidden)."""
    all_durations = [d for durs in durations_by_phase.values() for d, _ in durs]
    if not all_durations:
        raise ValueError("determine_dmax() called with no duration data at all.")
    observed_max = max(all_durations)
    if strategy == "observed_max":
        return observed_max
    if strategy == "quantile":
        if not (0.0 < quantile <= 1.0):
            raise ValueError(f"quantile must be in (0,1], got {quantile}.")
        return int(np.ceil(np.quantile(all_durations, quantile)))
    if strategy == "explicit":
        if explicit_value is None:
            raise ValueError("strategy='explicit' requires explicit_value.")
        if explicit_value < 1:
            raise ValueError(f"explicit_value must be >= 1, got {explicit_value}.")
        return int(explicit_value)
    raise ValueError(f"Unknown Dmax strategy {strategy!r} (expected 'observed_max', 'quantile', or 'explicit').")


# ---------------------------------------------------------------------------
# Duration models
# ---------------------------------------------------------------------------

class DurationModel(ABC):
    """A discrete distribution over D in {1,...,dmax}. By convention,
    probability(dmax) absorbs the ENTIRE tail P(D>=dmax) of the underlying
    family, so sum_{d=1}^{dmax} probability(d) == 1 exactly for every
    subclass -- proved once here via the generic expected_duration()/
    quantiles() implementations below, not re-derived per subclass."""

    def __init__(self, dmax: int):
        if dmax < 1:
            raise ValueError(f"dmax must be >= 1, got {dmax}.")
        self.dmax = dmax

    @abstractmethod
    def log_prob(self, d: int) -> float:
        """log P(D=d), d in [1, dmax]. d==dmax represents P(D>=dmax) (tail
        absorbed), not the raw family's point mass at exactly dmax."""

    @abstractmethod
    def log_survival(self, d: int) -> float:
        """log P(D>=d). By this class's own convention: log_survival(1)==0
        always (D>=1 by construction); log_survival(d) for d>dmax is -inf
        (no mass modeled past the truncation point, by construction)."""

    @abstractmethod
    def fit(self, durations: List[int], censoring: List[CensoringType]) -> None:
        """Deterministic given the same (durations, censoring) input --
        no randomness, no seed needed anywhere in this hierarchy."""

    @abstractmethod
    def to_dict(self) -> Dict:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, state: Dict) -> "DurationModel":
        ...

    def probability(self, d: int) -> float:
        return float(np.exp(self.log_prob(d)))

    def survival_probability(self, d: int) -> float:
        return float(np.exp(self.log_survival(d)))

    def expected_duration(self) -> float:
        return float(sum(d * self.probability(d) for d in range(1, self.dmax + 1)))

    def quantiles(self, qs: List[float]) -> Dict[float, int]:
        probs = np.array([self.probability(d) for d in range(1, self.dmax + 1)])
        total = probs.sum()
        cdf = np.cumsum(probs) / total if total > 0 else np.linspace(0, 1, self.dmax)
        out: Dict[float, int] = {}
        for q in qs:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"quantile must be in [0,1], got {q}.")
            idx = int(np.searchsorted(cdf, q))
            idx = min(idx, self.dmax - 1)
            out[q] = idx + 1
        return out


class NegativeBinomialDurationModel(DurationModel):
    """D = 1 + D', D' ~ NegativeBinomial(r, p) in scipy.stats.nbinom's own
    convention (pmf(k) = C(k+r-1,k) p^r (1-p)^k, k=0,1,2,...; mean(D') =
    r(1-p)/p, var(D') = r(1-p)/p^2) -- so D has support {1,2,...}, and
    `r` is scipy's number-of-successes parameter (NOT number of failures,
    despite scipy's generic argument name `n`), `p` is the success
    probability. Stated explicitly to avoid the classic
    failures/total-count/mean-dispersion ambiguity.

    KNOWN V1 SIMPLIFICATION (documented, not hidden): in fit(), any
    duration >= dmax (whether genuinely OBSERVED at exactly dmax, or
    censored/clipped down from something larger) contributes the SAME
    survival-likelihood term. A duration that truly, uncensored, ended at
    exactly dmax is thus fit slightly differently than the raw family's
    exact point mass at dmax would suggest -- this keeps the fitting
    likelihood internally consistent with the truncated distribution used
    at inference time (see DurationModel's own docstring), at the cost of
    this one edge-case approximation."""

    def __init__(self, dmax: int, r: float = 1.0, p: float = 0.5):
        super().__init__(dmax)
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}.")
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0,1), got {p}.")
        self.r = float(r)
        self.p = float(p)

    def _raw_log_pmf(self, d: int) -> float:
        return float(nbinom.logpmf(d - 1, self.r, self.p))

    def _raw_log_survival(self, d: int) -> float:
        if d <= 1:
            return 0.0
        cdf = float(nbinom.cdf(d - 2, self.r, self.p))
        surv = max(1.0 - cdf, 0.0)
        return float(np.log(surv)) if surv > 0 else _NEG_INF

    def log_prob(self, d: int) -> float:
        if d < 1 or d > self.dmax:
            raise ValueError(f"d={d} outside support [1,{self.dmax}].")
        if d < self.dmax:
            return self._raw_log_pmf(d)
        return self._raw_log_survival(self.dmax)

    def log_survival(self, d: int) -> float:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}.")
        if d > self.dmax:
            return _NEG_INF
        return self._raw_log_survival(d)

    def fit(self, durations: List[int], censoring: List[CensoringType]) -> None:
        if not durations:
            raise ValueError("Cannot fit NegativeBinomialDurationModel on zero durations.")
        obs = [d for d, c in zip(durations, censoring) if c == CensoringType.OBSERVED]
        if len(obs) >= 2:
            mean = float(np.mean(obs))
            var = float(np.var(obs))
            if var > mean > 0:
                p0 = mean / var
                r0 = mean * p0 / (1.0 - p0)
            else:
                r0, p0 = 1.0, 0.5
        else:
            r0, p0 = 1.0, 0.5
        r0 = max(r0, 1e-3)
        p0 = min(max(p0, 1e-3), 1 - 1e-3)

        dmax = self.dmax

        def unpack(x: np.ndarray) -> Tuple[float, float]:
            r = float(np.exp(x[0]))
            p = float(1.0 / (1.0 + np.exp(-x[1])))
            return max(r, 1e-6), min(max(p, 1e-6), 1 - 1e-6)

        def neg_log_lik(x: np.ndarray) -> float:
            r, p = unpack(x)
            ll = 0.0
            for d, c in zip(durations, censoring):
                dd = min(d, dmax)
                if c == CensoringType.OBSERVED and dd < dmax:
                    ll += float(nbinom.logpmf(dd - 1, r, p))
                else:
                    # dd == dmax (observed-at-boundary or any censoring type):
                    # survival contribution, per this class's own documented
                    # V1 simplification.
                    if dd <= 1:
                        surv = 1.0
                    else:
                        surv = max(1.0 - float(nbinom.cdf(dd - 2, r, p)), 1e-300)
                    ll += float(np.log(surv))
            return -ll

        x0 = np.array([np.log(r0), np.log(p0 / (1 - p0))])
        result = minimize(
            neg_log_lik, x0, method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
        )
        self.r, self.p = unpack(result.x)

    def to_dict(self) -> Dict:
        return {"family": "negative_binomial", "dmax": self.dmax, "r": self.r, "p": self.p}

    @classmethod
    def from_dict(cls, state: Dict) -> "NegativeBinomialDurationModel":
        return cls(dmax=state["dmax"], r=state["r"], p=state["p"])


class EmpiricalDurationModel(DurationModel):
    """Discrete empirical distribution over {1,...,dmax}, fit from
    OBSERVED (uncensored) durations only -- mirrors Phase 2's own choice to
    compute its primary duration distribution from interior (uncensored)
    occurrences only (Results/evaluation/e1_hmm_phase2_analysis/report.json,
    'onset_duration_interior'). This is a real, documented ASYMMETRY versus
    NegativeBinomialDurationModel (which DOES use censored observations via
    a survival-likelihood term) -- not hidden: the empirical candidate
    throws away information the parametric one keeps, in exchange for no
    distributional assumption. Durations > dmax are folded into the dmax
    bucket. Laplace smoothing (epsilon) is applied so no in-range d gets
    exactly zero probability -- required for the forward recursion's
    log-space sums, where a single -inf inside a logsumexp-summed path
    would zero out that whole path, not merely down-weight it."""

    def __init__(self, dmax: int, probs: Optional[np.ndarray] = None, fallback_uniform: bool = False):
        super().__init__(dmax)
        if probs is None:
            probs = np.full(dmax, 1.0 / dmax)
        probs = np.asarray(probs, dtype=np.float64)
        if len(probs) != dmax:
            raise ValueError(f"probs has {len(probs)} entries, expected dmax={dmax}.")
        self._probs = probs
        self._log_probs = _safe_log(probs)
        self.fallback_uniform = fallback_uniform

    def log_prob(self, d: int) -> float:
        if d < 1 or d > self.dmax:
            raise ValueError(f"d={d} outside support [1,{self.dmax}].")
        return float(self._log_probs[d - 1])

    def log_survival(self, d: int) -> float:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}.")
        if d > self.dmax:
            return _NEG_INF
        tail = float(self._probs[d - 1:].sum())
        return float(np.log(tail)) if tail > 0 else _NEG_INF

    def fit(self, durations: List[int], censoring: List[CensoringType], epsilon: float = 1e-3) -> None:
        obs = [min(d, self.dmax) for d, c in zip(durations, censoring) if c == CensoringType.OBSERVED]
        if not obs:
            self._probs = np.full(self.dmax, 1.0 / self.dmax)
            self._log_probs = _safe_log(self._probs)
            self.fallback_uniform = True
            return
        counts = np.zeros(self.dmax, dtype=np.float64)
        for d in obs:
            counts[d - 1] += 1.0
        smoothed = counts + epsilon
        self._probs = smoothed / smoothed.sum()
        self._log_probs = _safe_log(self._probs)
        self.fallback_uniform = False

    def to_dict(self) -> Dict:
        return {
            "family": "empirical", "dmax": self.dmax,
            "probs": self._probs.tolist(), "fallback_uniform": self.fallback_uniform,
        }

    @classmethod
    def from_dict(cls, state: Dict) -> "EmpiricalDurationModel":
        return cls(
            dmax=state["dmax"], probs=np.asarray(state["probs"], dtype=np.float64),
            fallback_uniform=state.get("fallback_uniform", False),
        )


def duration_model_from_dict(state: Dict) -> DurationModel:
    family = state["family"]
    if family == "negative_binomial":
        return NegativeBinomialDurationModel.from_dict(state)
    if family == "empirical":
        return EmpiricalDurationModel.from_dict(state)
    raise ValueError(f"Unknown duration model family {family!r}.")


# ---------------------------------------------------------------------------
# SemiHMMModel
# ---------------------------------------------------------------------------

@register_model("semi_hmm")
class SemiHMMModel(Model):
    def __init__(
        self,
        n_states: int = 15,
        state_names: Optional[List[str]] = None,
        dmax: int = 30,
        duration_family: str = "negative_binomial",
        transition_smoothing_alpha: float = 1.0,
        transition_decay_rho: float = 0.3,
        emission_prior_epsilon: float = 1e-6,
        logreg_C: float = 1.0,
        logreg_max_iter: int = 1000,
        logreg_class_weight: Optional[str] = None,
    ) -> None:
        if state_names is not None and len(state_names) != n_states:
            raise ValueError(f"state_names has {len(state_names)} entries but n_states={n_states}.")
        if duration_family not in ("negative_binomial", "empirical"):
            raise ValueError(f"duration_family must be 'negative_binomial' or 'empirical', got {duration_family!r}.")
        if dmax < 1:
            raise ValueError(f"dmax must be >= 1, got {dmax}.")
        if not (0.0 < transition_decay_rho <= 1.0):
            raise ValueError(f"transition_decay_rho must be in (0,1], got {transition_decay_rho}.")
        if transition_smoothing_alpha < 0.0:
            raise ValueError(f"transition_smoothing_alpha must be >= 0, got {transition_smoothing_alpha}.")

        self.n_states = n_states
        if state_names is not None:
            self.state_names = list(state_names)
        elif n_states == len(DEFAULT_PHASE_NAMES):
            self.state_names = list(DEFAULT_PHASE_NAMES)
        else:
            self.state_names = [f"S{i}" for i in range(n_states)]

        self.dmax = dmax
        self.duration_family = duration_family
        self.transition_smoothing_alpha = transition_smoothing_alpha
        self.transition_decay_rho = transition_decay_rho
        self.emission_prior_epsilon = emission_prior_epsilon
        self.logreg_C = logreg_C
        self.logreg_max_iter = logreg_max_iter
        self.logreg_class_weight = logreg_class_weight

        self._log_pi: Optional[np.ndarray] = None
        self._log_A: Optional[np.ndarray] = None  # strictly j>i nonzero, else -inf; no j==i cell at all
        self.duration_models: Optional[List[DurationModel]] = None
        self._emission_hmm: Optional[HMMModel] = None  # composed, supplies emission only -- never fit()'s own pi/A used
        self._consistency_base_rate: Optional[float] = None
        self._fitted = False

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} method called before fit().")

    def _validate_states(self, trajectories: List[Trajectory]) -> None:
        for traj in trajectories:
            if len(traj) == 0:
                continue
            vals = traj.last_frame_phase.numpy()
            if vals.size and (vals.min() < 0 or vals.max() >= self.n_states):
                raise ValueError(
                    f"Trajectory '{traj.video_name}' has last_frame_phase values outside "
                    f"[0, {self.n_states}) -- does not match n_states={self.n_states}."
                )

    def _resolve_state(self, state) -> int:
        if isinstance(state, str):
            return self.state_names.index(state)
        return int(state)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------
    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        """val accepted per the Model contract but unused -- hyperparameter
        SELECTION using val is the caller script's job, exactly like
        HMMModel.fit()'s own documented separation of concerns."""
        self._validate_states(train)
        K = self.n_states

        # pi: identical convention to HMMModel.fit()
        pi_counts = np.zeros(K, dtype=np.float64)
        for traj in train:
            if len(traj) == 0:
                continue
            pi_counts[int(traj.last_frame_phase[0])] += 1.0
        pi = pi_counts / pi_counts.sum() if pi_counts.sum() > 0 else np.full(K, 1.0 / K)
        self._log_pi = _safe_log(pi)

        # A: segment-to-segment destination ONLY (no self-loop cell at all),
        # reusing exactly the "destination conditional on leaving i" half of
        # HMMModel.fit()'s A estimation (window-level self-loop hazard has
        # no equivalent here -- duration owns that now).
        segment_counts = np.zeros((K, K), dtype=np.float64)
        for traj in train:
            segs = HMMModel._segments(traj)
            for (pa, _, _), (pb, _, _) in zip(segs, segs[1:]):
                if pb > pa:
                    segment_counts[pa, pb] += 1.0

        A = np.zeros((K, K), dtype=np.float64)
        for i in range(K - 1):  # K-1 (last phase) is terminal: no outgoing row
            targets = np.arange(i + 1, K)
            prior = self.transition_decay_rho ** (targets - i - 1)
            prior = prior / prior.sum()
            counts = segment_counts[i, targets]
            weighted = counts + self.transition_smoothing_alpha * prior
            denom = weighted.sum()
            A[i, targets] = weighted / denom if denom > 0 else 1.0 / len(targets)

        mask = np.triu(np.ones((K, K), dtype=bool), k=1)  # strictly upper triangular: j>i only
        A = np.where(mask, A, 0.0)
        self._log_A = _safe_log(A)
        self._log_A[~mask] = _NEG_INF

        # durations
        durations_by_phase = extract_phase_durations(train)
        self.duration_models = []
        for i in range(K):
            durs_censoring = durations_by_phase.get(i, [])
            durations = [d for d, _ in durs_censoring]
            censoring = [c for _, c in durs_censoring]
            n_observed = sum(1 for c in censoring if c == CensoringType.OBSERVED)

            if n_observed == 0:
                # Phase A2 (docs/PROJECT_STATE.md's "SEMI-HMM -- AUTONOMOUS
                # WEEKEND" checkpoint): a phase with ZERO uncensored
                # occurrences (structurally the case for tPB2/tEB -- 0/400
                # and 0/265 on real Train, confirmed, not merely rare --
                # since they are the first/last chronological phase and
                # every occurrence is therefore either the video's very
                # first or very last segment) cannot be fit by ANY
                # duration family's censored-only MLE without degenerating:
                # with only "D >= d_obs" lower-bound constraints and no
                # single point pinning the true scale, the negative
                # binomial's MLE runs to the dmax boundary (observed
                # directly in the STEP 2 smoke test: expected_duration,
                # median, q10, q90 all collapsed to exactly dmax) -- a
                # false, overconfident claim of a very long duration, not
                # an absence-of-information signal. FORCED, EXPLICIT
                # fallback regardless of the requested duration_family: a
                # uniform EmpiricalDurationModel over {1,...,dmax}, the
                # honest "we have no information to estimate this" choice
                # -- never silently left to whichever family was
                # configured globally.
                dm = EmpiricalDurationModel(dmax=self.dmax)
                dm.fallback_uniform = True
            else:
                dm = (
                    NegativeBinomialDurationModel(dmax=self.dmax)
                    if self.duration_family == "negative_binomial"
                    else EmpiricalDurationModel(dmax=self.dmax)
                )
                dm.fit(durations, censoring)
            self.duration_models.append(dm)

        # emission: composed HMMModel, reused verbatim, never duplicated.
        self._emission_hmm = HMMModel(
            n_states=K, state_names=self.state_names,
            emission_prior_epsilon=self.emission_prior_epsilon,
            logreg_C=self.logreg_C, logreg_max_iter=self.logreg_max_iter,
            logreg_class_weight=self.logreg_class_weight,
        )
        self._emission_hmm.fit(train)  # also computes its own pi/A internally, unused here; cheap, harmless

        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._consistency_base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        self._fitted = True

    def _log_emission(self, traj: Trajectory) -> np.ndarray:
        """Delegates to the composed HMMModel's own emission -- same
        StandardScaler + LogisticRegression + Bayes-conversion pipeline,
        never reimplemented here. Clipped to the finite _NEG_INF sentinel
        (matching hmm.py's own convention) rather than left as literal
        -inf: unlike hmm.py's own per-step logsumexp-based forward (which
        never subtracts two -inf terms), this module's forward uses
        PREFIX-SUM subtraction (cum[end]-cum[start], see
        _explicit_duration_forward) to get O(1) windowed emission sums --
        if a state's log-probability underflows to literal -inf for two
        positions in the same cumulative range, cum[end]-cum[start]
        becomes -inf-(-inf)=nan. Clipping BEFORE the cumulative sum is
        built (i.e. here, at the source) means every downstream value is
        finite and the subtraction is always well-defined. Discovered via
        the STEP 2 smoke test (RuntimeWarning on an out-of-distribution
        causality-test embedding; never manifested on real Val data in
        that test). No effect on any realistic value: a genuine
        log-probability is never anywhere near -1e10, so this only clips
        already-degenerate/impossible entries, never changes normal-case
        arithmetic."""
        log_b = self._emission_hmm._log_emission(traj)
        return np.maximum(log_b, _NEG_INF)

    # ------------------------------------------------------------------
    # Explicit-duration forward
    # ------------------------------------------------------------------
    def _explicit_duration_forward_reference(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """REFERENCE implementation -- correct, simple, deliberately not
        optimized (Phase B/C of the Semi-HMM weekend checkpoint: written
        and validated FIRST, kept here permanently as the correctness
        oracle for _explicit_duration_forward_optimized(), never deleted).

        Returns (log_alpha_end, log_age_posterior):

        log_alpha_end[t,j] = log P(O_1:t, a segment of state j ends EXACTLY
        at window t) -- the literal recursion from the Semi-HMM protocol,
        using the exact duration pmf P(D=d|j). This is the quantity needed
        to CHAIN the recursion across segments (a new segment's entering
        mass requires a previous segment to have ended); it is NOT the
        filtering posterior.

        log_age_posterior[t,j,a-1] = log P(O_1:t, currently in state j,
        having been there for exactly 'a' consecutive windows up to and
        including t -- segment not necessarily over yet) -- uses the
        SURVIVAL P(D>=a|j), not the pmf, since t need not be the segment's
        last window. filtering()/predict_proba() marginalize this over a;
        predict()'s consistency_flag_prob and next_phase_distribution()
        both need the age breakdown itself.

        Log-space throughout (finite _NEG_INF sentinel, matching hmm.py's
        own convention, not literal -inf). Causal by construction:
        log_alpha_end[t]/log_age_posterior[t] are pure functions of
        O_1..O_t and the fixed model parameters -- log_b rows beyond t are
        never read.

        Cost: O(T*K^2*Dmax) -- for every (t,j,a) triple this recomputes an
        O(K) logsumexp ("entering mass") whose value depends only on
        (t-a,j), even though the same (t-a,j) pair is needed again by every
        later t' up to t-a+Dmax; and duration_models[j].log_prob(a)/
        log_survival(a) (which do not depend on t at all) are recomputed
        T times more often than necessary. Measured cost (STEP 2 smoke
        test, 4 real Val trajectories, Dmax=40, K=15): ~0.148s/window,
        ~107 minutes extrapolated to full Val (43,339 windows) --
        impractical at Train/Val scale. See
        _explicit_duration_forward_optimized() for the fix."""
        log_b = self._log_emission(traj)
        T, K = log_b.shape
        Dmax = self.dmax
        log_alpha_end = np.full((T, K), _NEG_INF)
        log_age_posterior = np.full((T, K, Dmax), _NEG_INF)
        if T == 0:
            return log_alpha_end, log_age_posterior

        cum = np.zeros((T + 1, K))
        cum[1:] = np.cumsum(log_b, axis=0)

        def emission_sum(state: int, start: int, end_inclusive: int) -> float:
            return float(cum[end_inclusive + 1, state] - cum[start, state])

        for t in range(T):
            max_a = min(Dmax, t + 1)
            for j in range(K):
                dm = self.duration_models[j]
                end_terms = []
                for a in range(1, max_a + 1):
                    start = t - a + 1
                    if start == 0:
                        entering = self._log_pi[j]
                    else:
                        prev_t = start - 1
                        entering = float(logsumexp(log_alpha_end[prev_t] + self._log_A[:, j]))
                    if entering <= _NEG_INF / 2:
                        continue
                    em = emission_sum(j, start, t)
                    end_terms.append(entering + dm.log_prob(a) + em)
                    log_age_posterior[t, j, a - 1] = entering + dm.log_survival(a) + em
                log_alpha_end[t, j] = float(logsumexp(end_terms)) if end_terms else _NEG_INF
        return log_alpha_end, log_age_posterior

    def _explicit_duration_forward_optimized(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """Same recursion, same result (validated to numerical tolerance
        against _explicit_duration_forward_reference() -- see
        Tests/evaluation/models/test_semi_hmm.py's equivalence tests),
        restructured for speed:

        1. entering_cache[t,:] := logsumexp_i(log_alpha_end[t,i] + A[i,:])
           is computed ONCE per t (O(K^2)), the moment log_alpha_end[t] is
           known -- not recomputed once per (t',j,a) triple that happens to
           need prev_t=t (which the reference does, up to Dmax times per
           t). This is the "precompute/reuse the entering mass" fix.
        2. duration_models[j].log_prob(a)/log_survival(a) do not depend on
           t at all -- precomputed ONCE into (K,Dmax) tables before the
           main loop, instead of being recomputed at every one of the
           T*K*Dmax (t,j,a) triples. This was the dominant real cost in
           practice (STEP 2 benchmark): each call carries real scipy/
           Python function-call overhead, and T can be in the hundreds.

        Net cost: O(T*K^2) for the entering-mass cache + O(T*K*Dmax) for
        the (now table-lookup-only) remaining accumulation -- matching the
        Semi-HMM protocol's target complexity."""
        log_b = self._log_emission(traj)
        T, K = log_b.shape
        Dmax = self.dmax
        log_alpha_end = np.full((T, K), _NEG_INF)
        log_age_posterior = np.full((T, K, Dmax), _NEG_INF)
        if T == 0:
            return log_alpha_end, log_age_posterior

        cum = np.zeros((T + 1, K))
        cum[1:] = np.cumsum(log_b, axis=0)

        log_pmf_table = np.full((K, Dmax), _NEG_INF)
        log_surv_table = np.full((K, Dmax), _NEG_INF)
        for j in range(K):
            dm = self.duration_models[j]
            for a in range(1, Dmax + 1):
                log_pmf_table[j, a - 1] = dm.log_prob(a)
                log_surv_table[j, a - 1] = dm.log_survival(a)

        entering_cache = np.full((T, K), _NEG_INF)  # entering_cache[t,:], filled once log_alpha_end[t,:] is known

        for t in range(T):
            max_a = min(Dmax, t + 1)
            end_contribs = []
            for a in range(1, max_a + 1):
                start = t - a + 1
                entering_row = self._log_pi if start == 0 else entering_cache[start - 1]
                em = cum[t + 1] - cum[start]  # (K,) -- emission sum per state, vectorized
                end_contribs.append(entering_row + log_pmf_table[:, a - 1] + em)
                log_age_posterior[t, :, a - 1] = entering_row + log_surv_table[:, a - 1] + em
            log_alpha_end[t] = logsumexp(np.stack(end_contribs, axis=0), axis=0)
            entering_cache[t] = logsumexp(log_alpha_end[t][:, None] + self._log_A, axis=0)

        return log_alpha_end, log_age_posterior

    def _explicit_duration_forward(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatches to the optimized implementation -- see
        _explicit_duration_forward_optimized()'s docstring. The reference
        implementation (_explicit_duration_forward_reference()) is kept
        permanently as the correctness oracle, never used in the default
        path, never deleted."""
        return self._explicit_duration_forward_optimized(traj)

    def forward(self, traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """Raw (log_alpha_end, log_age_posterior) forward tables -- see
        _explicit_duration_forward's docstring."""
        self._require_fitted()
        return self._explicit_duration_forward(traj)

    def filtering(self, traj: Trajectory) -> np.ndarray:
        """P(S_t=j | O_1:t), shape (T, n_states) -- causal. Same name/
        contract as HMMModel.filtering()."""
        self._require_fitted()
        _, log_age_posterior = self._explicit_duration_forward(traj)
        T = log_age_posterior.shape[0]
        if T == 0:
            return np.empty((0, self.n_states))
        log_marginal = logsumexp(log_age_posterior, axis=2)  # (T,K)
        return np.exp(log_marginal - logsumexp(log_marginal, axis=1, keepdims=True))

    def predict_proba(self, traj: Trajectory) -> np.ndarray:
        """Alias for filtering()."""
        return self.filtering(traj)

    def _next_phase_distribution_reference(self, traj: Trajectory) -> np.ndarray:
        """REFERENCE implementation of next_phase_distribution() -- correct,
        deliberately not optimized (Phase 1.6, kept here permanently as the
        correctness oracle for _next_phase_distribution_optimized(), never
        used in the default path, never deleted -- same discipline as
        _explicit_duration_forward_reference()/_optimized()). Byte-for-byte
        the pre-Phase-1.6 body of the public next_phase_distribution(): only
        its name and role (oracle instead of default) changed.

        P(S_{t+1}=j | O_1:t), shape (T, n_states) -- row t predicts the
        window AFTER t. Duration-aware: the hazard of LEAVING the current
        segment depends on its already-elapsed age (P(D=a|j)/P(D>=a|j)),
        not a single fixed rate as in HMMModel.predict_next().

        Cost: O(T*K*Dmax), and unlike _next_phase_distribution_optimized(),
        calls duration_models[j].log_survival(a)/log_survival(a+1)/
        log_prob(a) FRESH on every (t,j,a) triple even though none of these
        three values depend on t at all -- the identical problem class
        _explicit_duration_forward_optimized() already solves elsewhere in
        this file. Measured cost (this session, real Val Patient_319, 527
        windows): ~333-357s. See _next_phase_distribution_optimized() for
        the fix and docs/SEMI_HMM_PHASE_1_6_REPORT.md for the full
        equivalence validation."""
        _, log_age_posterior = self._explicit_duration_forward(traj)
        T, K, Dmax = log_age_posterior.shape
        if T == 0:
            return np.empty((0, K))
        out = np.full((T, K), _NEG_INF)
        for t in range(T):
            max_a = min(Dmax, t + 1)
            log_continue = np.full(K, _NEG_INF)
            log_leave = np.full(K, _NEG_INF)
            for j in range(K):
                dm = self.duration_models[j]
                cont_terms, leave_terms = [], []
                for a in range(1, max_a + 1):
                    lap = log_age_posterior[t, j, a - 1]
                    if lap <= _NEG_INF / 2:
                        continue
                    log_surv_a = dm.log_survival(a)
                    if log_surv_a <= _NEG_INF / 2:
                        continue
                    log_surv_a1 = dm.log_survival(a + 1) if a + 1 <= Dmax else _NEG_INF
                    log_p_continue = (log_surv_a1 - log_surv_a) if log_surv_a1 > _NEG_INF / 2 else _NEG_INF
                    log_p_leave = dm.log_prob(a) - log_surv_a
                    cont_terms.append(lap + log_p_continue)
                    leave_terms.append(lap + log_p_leave)
                log_continue[j] = float(logsumexp(cont_terms)) if cont_terms else _NEG_INF
                log_leave[j] = float(logsumexp(leave_terms)) if leave_terms else _NEG_INF
            for k in range(K):
                terms = [log_continue[k]]
                for j in range(K):
                    if self._log_A[j, k] > _NEG_INF / 2 and log_leave[j] > _NEG_INF / 2:
                        terms.append(log_leave[j] + self._log_A[j, k])
                out[t, k] = float(logsumexp(terms))
        row_total = logsumexp(out, axis=1, keepdims=True)
        return np.exp(out - row_total)

    def _next_phase_distribution_optimized(self, traj: Trajectory) -> np.ndarray:
        """Same recursion, same result as _next_phase_distribution_reference()
        (validated to numerical tolerance -- see
        Tests/evaluation/models/test_semi_hmm.py's Phase 1.6 equivalence
        tests), restructured for speed by the SAME fix already applied to
        _explicit_duration_forward_optimized() for the identical problem
        class: duration_models[j].log_prob(a)/log_survival(a) do not depend
        on t at all, so they are precomputed ONCE into (K,Dmax) tables
        before the main loop, instead of being recomputed fresh at every
        one of the T*K*Dmax (t,j,a) triples.

        NOT a new algorithm: every arithmetic step, term, and branch below
        is identical to the reference implementation -- only the SOURCE of
        dm.log_survival(a)/dm.log_survival(a+1)/dm.log_prob(a)'s values
        changes (an array lookup instead of a fresh method call). The
        duration model's state does not change during this call, so
        calling e.g. log_prob(a) twice or looking it up once from a table
        built via that exact same call produces the same floating-point
        value, not merely a numerically close one -- see this file's Phase
        1.6 equivalence tests for the direct, exact-equality check on real
        Val data."""
        _, log_age_posterior = self._explicit_duration_forward(traj)
        T, K, Dmax = log_age_posterior.shape
        if T == 0:
            return np.empty((0, K))

        log_pmf_table = np.full((K, Dmax), _NEG_INF)
        log_surv_table = np.full((K, Dmax), _NEG_INF)
        for j in range(K):
            dm = self.duration_models[j]
            for a in range(1, Dmax + 1):
                log_pmf_table[j, a - 1] = dm.log_prob(a)
                log_surv_table[j, a - 1] = dm.log_survival(a)

        out = np.full((T, K), _NEG_INF)
        for t in range(T):
            max_a = min(Dmax, t + 1)
            log_continue = np.full(K, _NEG_INF)
            log_leave = np.full(K, _NEG_INF)
            for j in range(K):
                cont_terms, leave_terms = [], []
                for a in range(1, max_a + 1):
                    lap = log_age_posterior[t, j, a - 1]
                    if lap <= _NEG_INF / 2:
                        continue
                    log_surv_a = log_surv_table[j, a - 1]
                    if log_surv_a <= _NEG_INF / 2:
                        continue
                    log_surv_a1 = log_surv_table[j, a] if a < Dmax else _NEG_INF
                    log_p_continue = (log_surv_a1 - log_surv_a) if log_surv_a1 > _NEG_INF / 2 else _NEG_INF
                    log_p_leave = log_pmf_table[j, a - 1] - log_surv_a
                    cont_terms.append(lap + log_p_continue)
                    leave_terms.append(lap + log_p_leave)
                log_continue[j] = float(logsumexp(cont_terms)) if cont_terms else _NEG_INF
                log_leave[j] = float(logsumexp(leave_terms)) if leave_terms else _NEG_INF
            for k in range(K):
                terms = [log_continue[k]]
                for j in range(K):
                    if self._log_A[j, k] > _NEG_INF / 2 and log_leave[j] > _NEG_INF / 2:
                        terms.append(log_leave[j] + self._log_A[j, k])
                out[t, k] = float(logsumexp(terms))
        row_total = logsumexp(out, axis=1, keepdims=True)
        return np.exp(out - row_total)

    def next_phase_distribution(self, traj: Trajectory) -> np.ndarray:
        """P(S_{t+1}=j | O_1:t), shape (T, n_states) -- row t predicts the
        window AFTER t. Duration-aware: the hazard of LEAVING the current
        segment depends on its already-elapsed age (P(D=a|j)/P(D>=a|j)),
        not a single fixed rate as in HMMModel.predict_next(). Dispatches
        to _next_phase_distribution_optimized() (Phase 1.6) -- see that
        method's docstring for the optimization; the pre-Phase-1.6
        implementation is kept permanently as
        _next_phase_distribution_reference(), the correctness oracle, never
        used in the default path, never deleted (same discipline as
        _explicit_duration_forward's own reference/optimized split)."""
        self._require_fitted()
        return self._next_phase_distribution_optimized(traj)

    def duration_distribution(self, state) -> Dict[int, float]:
        """P(D=d | S=state) for d=1..dmax."""
        self._require_fitted()
        dm = self.duration_models[self._resolve_state(state)]
        return {d: dm.probability(d) for d in range(1, self.dmax + 1)}

    def expected_duration(self, state) -> float:
        self._require_fitted()
        return self.duration_models[self._resolve_state(state)].expected_duration()

    def transition_chain(self, traj: Trajectory) -> List[Dict]:
        """Diagnostic/reporting chain over the trajectory's GROUND-TRUTH
        segments (via HMMModel._segments(), reused) -- for each observed
        segment: phase, observed duration, this model's own
        P(D=observed_duration|phase), and the destination distribution
        A[phase,:]. NOT a semi-Markov Viterbi decode -- that is out of
        scope for this infrastructure step (interfaces only, per this
        session's explicit instruction)."""
        self._require_fitted()
        segs = HMMModel._segments(traj)
        A = np.exp(self._log_A)
        chain = []
        for phase_idx, start, end in segs:
            duration = end - start + 1
            dm = self.duration_models[phase_idx]
            dur_for_query = min(duration, self.dmax)
            next_dist = {
                self.state_names[j]: float(A[phase_idx, j])
                for j in range(self.n_states) if A[phase_idx, j] > 0
            }
            chain.append({
                "phase": self.state_names[phase_idx],
                "observed_duration_windows": duration,
                "duration_probability_at_observed": dm.probability(dur_for_query),
                "next_phase_distribution": next_dist,
            })
        return chain

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        """consistency_flag_prob[t] := P(S_t != S_{t-1} | O_1:t) -- here,
        exactly P(age_t == 1 | O_1:t): a segment's age is 1 precisely when
        it started at t, i.e. a transition occurred between window t-1 and
        t. window 0 falls back to the empirical Train base rate, same
        convention as HMMModel.predict()."""
        self._require_fitted()
        predictions = []
        for traj in trajectories:
            T = len(traj)
            if T == 0:
                predictions.append(TrajectoryPrediction(
                    video_name=traj.video_name, window_starts=traj.window_starts,
                    consistency_flag_prob=torch.empty(0),
                ))
                continue
            _, log_age_posterior = self._explicit_duration_forward(traj)
            probs = np.full(T, self._consistency_base_rate, dtype=np.float64)
            for t in range(1, T):
                total = float(logsumexp(log_age_posterior[t]))
                age1 = float(logsumexp(log_age_posterior[t, :, 0]))
                probs[t] = float(np.exp(age1 - total)) if total > _NEG_INF / 2 else self._consistency_base_rate
            predictions.append(TrajectoryPrediction(
                video_name=traj.video_name, window_starts=traj.window_starts,
                consistency_flag_prob=torch.from_numpy(probs.astype(np.float32)),
            ))
        return predictions

    # ------------------------------------------------------------------
    # k-step transition probability (added 2026-08-23, for a fair
    # comparison against HMMModel.predict_k_step_transition_probability):
    # consistency_flag (Training/DataSet.py:310) is an INTRA-window event
    # (first_frame_phase vs last_frame_phase of the SAME window), which
    # predict()'s lag-1 consistency_flag_prob does NOT measure -- this was
    # diagnosed for HMMModel on 2026-08-21 (PROJECT_STATE.md) and fixed
    # there via predict_k_step_transition_probability(k=window_size-1).
    # The same misalignment applies here identically (predict() above is
    # also lag-1), so the same fix is required before comparing this
    # model's real Val performance against the frozen HMM k=7 benchmark.
    # ------------------------------------------------------------------
    def predict_k_step_transition_probability(self, traj: Trajectory, k: int = 1) -> np.ndarray:
        """P(S_t != S_{t-k} | O_1:t), shape (T,) -- causal, EXACT closed
        form exploiting a structural invariant of THIS model that
        HMMModel's own k-step method cannot rely on (a plain Markov chain
        has no such invariant, hence needs an explicit k-step propagation
        over A): transitions here are strictly j>i (no self-loop cell
        exists in A at all -- duration owns persistence, see this
        module's docstring, test_extract_phase_durations_no_self_transition_created,
        test_transition_matrix_terminal_state_has_no_outgoing_row), so any
        two consecutive SEGMENTS necessarily have different states.
        Therefore, for window t whose current segment has age a:
          - a <= k  => window t-k lies strictly before this segment's
            start, in an EARLIER segment -- S_{t-k} != S_t is GUARANTEED.
          - a > k   => window t-k lies inside the SAME segment as t, so
            S_{t-k} == S_t exactly.
        Hence P(S_t != S_{t-k} | O_1:t) == P(age_t <= k | O_1:t), read
        directly off the SAME log_age_posterior table forward() already
        computes -- no extra k-step propagation needed.

        k=1 reproduces predict()'s consistency_flag_prob exactly (see
        test_predict_k_step_k_equals_1_matches_predict). k=(window_size-1)
        (7 at real Train/Val scale) is the quantity that actually matches
        consistency_flag's own definition -- see
        HMMModel.predict_k_step_transition_probability's docstring for the
        full windowing-arithmetic derivation of why (identical here: this
        model's per-window state is the same last_frame_phase convention).

        window_starts matching, fallback: same discipline as
        HMMModel.predict_k_step_transition_probability -- positions are
        matched by window_starts VALUE, not raw trajectory index (a
        filtered-out window would otherwise silently misalign a pure
        index-based lookup); no valid k-step match falls back to the
        empirical Train consistency_flag base rate, same convention as
        predict()'s own t=0 case, generalized.
        """
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        T = len(traj)
        probs = np.full(T, self._consistency_base_rate, dtype=np.float64)
        if T == 0:
            return probs
        _, log_age_posterior = self._explicit_duration_forward(traj)
        max_age_idx = min(k, self.dmax)  # ages 1..max_age_idx, i.e. slice [:max_age_idx]
        start_to_index = {s: i for i, s in enumerate(traj.window_starts)}
        for t in range(T):
            target_start = traj.window_starts[t] - k
            t0 = start_to_index.get(target_start)
            if t0 is None or t0 + k != t:
                continue  # no window exactly k positions earlier, reachable
                # via k gap-free trajectory-index steps -> base-rate fallback
                # (already set above)
            total = float(logsumexp(log_age_posterior[t]))
            if total <= _NEG_INF / 2:
                continue  # degenerate row -> base-rate fallback
            young = float(logsumexp(log_age_posterior[t, :, :max_age_idx]))
            probs[t] = float(np.exp(young - total))
        return probs

    def predict_k_step(self, trajectories: List[Trajectory], k: int = 1) -> List[TrajectoryPrediction]:
        """Batch wrapper around predict_k_step_transition_probability(),
        same TrajectoryPrediction output contract as predict() -- directly
        consumable by evaluation.metrics.compute_metrics/calibration
        scoring, unmodified. k=1 is bit-for-bit identical to predict()'s
        own output."""
        self._require_fitted()
        predictions = []
        for traj in trajectories:
            probs = self.predict_k_step_transition_probability(traj, k=k)
            predictions.append(TrajectoryPrediction(
                video_name=traj.video_name, window_starts=traj.window_starts,
                consistency_flag_prob=torch.from_numpy(probs.astype(np.float32)),
            ))
        return predictions

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "n_states": self.n_states, "state_names": self.state_names,
            "dmax": self.dmax, "duration_family": self.duration_family,
            "transition_smoothing_alpha": self.transition_smoothing_alpha,
            "transition_decay_rho": self.transition_decay_rho,
            "emission_prior_epsilon": self.emission_prior_epsilon,
            "logreg_C": self.logreg_C, "logreg_max_iter": self.logreg_max_iter,
            "logreg_class_weight": self.logreg_class_weight,
            "fitted": self._fitted,
        }
        if self._fitted:
            state["log_pi"] = self._log_pi.tolist()
            state["log_A"] = self._log_A.tolist()
            state["consistency_base_rate"] = self._consistency_base_rate
            state["duration_models"] = [dm.to_dict() for dm in self.duration_models]
            # Emission persisted via HMMModel's OWN save() format, nested
            # under a subdirectory -- documented exception to "flat JSON",
            # not a new convention invented for this file: it IS the
            # existing HMM save/load format, reused as-is.
            self._emission_hmm.save(path / "emission_hmm")
        (path / "state.json").write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SemiHMMModel":
        state = json.loads((path / "state.json").read_text())
        model = cls(
            n_states=state["n_states"], state_names=state["state_names"], dmax=state["dmax"],
            duration_family=state["duration_family"],
            transition_smoothing_alpha=state["transition_smoothing_alpha"],
            transition_decay_rho=state["transition_decay_rho"],
            emission_prior_epsilon=state["emission_prior_epsilon"],
            logreg_C=state["logreg_C"], logreg_max_iter=state["logreg_max_iter"],
            logreg_class_weight=state.get("logreg_class_weight"),
        )
        if state.get("fitted"):
            model._log_pi = np.asarray(state["log_pi"])
            model._log_A = np.asarray(state["log_A"])
            model._consistency_base_rate = state["consistency_base_rate"]
            model.duration_models = [duration_model_from_dict(d) for d in state["duration_models"]]
            model._emission_hmm = HMMModel.load(path / "emission_hmm")
            model._fitted = True
        return model
