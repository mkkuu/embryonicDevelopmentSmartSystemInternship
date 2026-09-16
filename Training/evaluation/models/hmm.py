"""
HMMModel — the first model of the HMM/Semi-HMM experimental branch (see
docs/HMM_RESEARCH_PLAN.md, docs/HMM_PHASE0_EXECUTION_PLAN.md, and the
Phase 2/Phase 3 conversation this file's design is taken from). A NEW
branch, additive to base_rate/identity_dynamics/persistence/linear_ssm/gru
— never edits any of those files or the results they already produced.

Scientific formulation
-----------------------
    S_t in {0,...,K-1}   hidden state = developmental phase
                          (index into DEFAULT_PHASE_NAMES, matching
                          Training/DataSet.py's chronological_phases order
                          — confirmed identical across Train/Val/Test in
                          Phase 0 of this branch: phase_to_index is the
                          same dict in all three splits)
    O_t in R^512          observation = the window's cached embedding
    t indexes consecutive WINDOWS of one trajectory (video), in
    window_starts order — one window = one HMM time step (Phase 2's own
    empirical justification: at the window level, a single-step jump from
    phase i directly to phase i+2 already reflects a phase that was never
    annotated on any frame of that video, not a windowing artifact).

    P(S_1:T, O_1:T) = pi(S_1) * prod_{t=2}^T A[S_{t-1}, S_t]
                              * prod_{t=1}^T b_{S_t}(O_t)

Independence assumptions made (and one explicitly NOT satisfied)
------------------------------------------------------------------
- First-order Markov on S_t, time-homogeneous A/b. This is known, from
  Phase 2's own duration-CV measurements (0.18 to 1.25 across phases), to
  be an inaccurate model of real dwell-time variability — accepted as a
  deliberate simplification for this "classic HMM" stage; the Semi-HMM
  (not built here) is the intended fix.
- Emission independence O_t independent of everything else given S_t.
  KNOWN TO BE VIOLATED: with window_size=8/stride=1, consecutive windows
  share 7/8 source frames, so O_t and O_{t+1} are far more correlated than
  the model assumes. This inflates the apparent confidence of anything
  that multiplies emissions across many timesteps (sequence_log_likelihood
  in particular — see its docstring). Not corrected here; documented, not
  hidden.

Constraint (Phase 2's empirical result, Phase 3's design decision)
---------------------------------------------------------------------
A[i, j] = 0 for all j < i — enforced structurally (log-space -inf), not by
convention. Phase 2 found ZERO backward segment-transitions across
Train/Val/Test (492/106/106 videos, 0/0/0 with any backward transition) —
this is NOT the same as the originally-planned strict "i or i+1 only"
left-to-right chain: Phase 2 also found ~11-13% of transitions are SKIPS
(j > i+1, up to 4 phases at once), present in ~81% of Train videos. The
constraint here is therefore j >= i, not j in {i, i+1} — a direct,
evidence-driven correction of the earlier, more restrictive proposal in
docs/HMM_RESEARCH_PLAN.md Section 4.

A estimation: factorized, per Phase 3 Section 4/5
----------------------------------------------------
A[i,:] is built from TWO different, deliberately different-granularity
counts, because Phase 2 showed window-level and segment/video-level counts
answer different questions (Phase 2 Section 5):
  1. A[i,i] (self-loop / "hazard of leaving") is estimated from WINDOW-
     level consecutive pairs — a window IS a timestep, so this is the
     literal MLE for the model's own per-step self-transition.
  2. Conditional on leaving i (mass 1-A[i,i]), the destination j>i is
     estimated from SEGMENT-level (per-video) transition counts, smoothed
     with a Dirichlet prior that decays geometrically in (j-i) — informed
     directly by Phase 2's own observed skip-size distribution
     (delta=1: 4581, delta=2: 397, delta=3: 149, delta=4: 40, delta=5: 8
     segment-transitions on Train — a clear, roughly geometric decay).
     Smoothing is NOT optional here: several real skip transitions were
     observed from only 1-4 distinct videos (Phase 2 Section 4, Jeffreys
     CIs already computed there), and Phase 2 explicitly warned that
     "transition jamais observee" must never collapse to "transition
     impossible" (option B in Phase 3 Section 5 was explicitly rejected
     for exactly this reason).

Emission: hybrid discriminative -> pseudo-likelihood (Phase 3 Section 3/6)
------------------------------------------------------------------------------
A single multinomial logistic regression P(S=i|O) (sklearn, same family
already used by identity_dynamics/persistence/linear_ssm in this package)
is converted to a pseudo emission via Bayes' rule:

    b_i(O) := P(S=i|O) / P(S=i)     [dropping the P(O) factor]

MATHEMATICAL VERIFICATION (Phase 3 Section 6 explicitly required this
before implementing; done here, not skipped):

True Bayes gives P(O|S=i) = P(S=i|O) * P(O) / P(S=i). b_i(O) above omits
the P(O) factor, which is the SAME for every state i at a given O_t (it
does not depend on i) but DOES depend on t (different windows have
different marginal P(O_t)). Question: does dropping a per-timestep,
state-independent constant break the forward algorithm?

By induction on t: let alpha'_t be the forward variable computed with
b_i(O_t) := P(S=i|O_t)/P(S=i) instead of the true b_i(O_t). At t=1,
alpha'_1(i) = pi_i * P(S=i|O_1)/P(S=i) = alpha_1(i) / P(O_1) -- the missing
factor 1/P(O_1) is identical for every state i. Assume alpha'_{t-1}(i) =
alpha_{t-1}(i) / prod_{s<t} P(O_s) for every i (induction hypothesis, a
STATE-INDEPENDENT scalar). Then
    alpha'_t(j) = [sum_i alpha'_{t-1}(i) A_ij] * b_j(O_t)
                = [sum_i alpha_{t-1}(i) A_ij] / prod_{s<t} P(O_s) * b_j(O_t)/P(O_t) * P(O_t)...
(expanding b_j(O_t) = true_b_j(O_t)/P(O_t))
                = alpha_t(j) / prod_{s<=t} P(O_s)
which is again a state-independent scalar times the true alpha_t(j). QED
by induction. CONSEQUENCE: any quantity that is a RATIO across states at a
fixed t (filtering, smoothing, Viterbi's argmax, the two-slice joint used
for next-state/consistency_flag prediction below) is IDENTICAL whether
computed with b_i or with the pseudo b_i above — the missing per-timestep
constant cancels exactly. THIS IS WHY the hybrid trick (Bourlard & Morgan,
1994, HMM/ANN hybrids) is valid for decoding/posterior inference.

WHAT IS NOT VALID: the raw value of alpha_t / the sum used for
sequence_log_likelihood() is off by the UNKNOWN, NEVER-ESTIMATED factor
prod_t P(O_t). Concretely: sequence_log_likelihood() computed under this
emission is a valid, comparable score for choosing BETWEEN CONFIGURATIONS
THAT SHARE THIS SAME EMISSION FAMILY (e.g. sweeping transition_smoothing_
alpha or transition_decay_rho with the hybrid emission held fixed) — the
missing constant only depends on the marginal distribution of O, not on
pi/A, so it cancels when comparing two HMMs that differ only in pi/A. It
is NOT valid to directly compare this number against a log-likelihood
computed under a genuinely generative emission (e.g. a future PCA+Gaussian
variant) — that comparison would silently compare a pseudo-likelihood
against a true likelihood, missing an unknown additive constant in log-
space. See sequence_log_likelihood()'s own docstring; this is checked by
test_emission_pseudolikelihood_conversion_is_valid_for_posteriors in
Tests/evaluation/models/test_hmm.py, which verifies the cancellation
property directly and algebraically, not just by argument.

Prior P(S=i) used in the conversion: the EMPIRICAL TRAIN marginal of
last_frame_phase, computed from the exact same pooled window set the
logistic regression itself is fit on (never a different, mismatched
prior) — required for the Bayes conversion to be internally consistent
with what the LogisticRegression's own predict_proba() already implicitly
assumes about the training class balance.

Identifiability: b_i(O) above recovers the TRUE P(O|S=i) only up to the
missing, state-independent P(O_t) factor — i.e. NOT fully identifiable as
a density, only identifiable up to that constant. This is sufficient for
every quantity this model reports except the absolute sequence
log-likelihood (documented above and in that method).

Initial state pi (Phase 3 Section 4/6, left-censoring)
----------------------------------------------------------
pi_i = empirical fraction of TRAIN trajectories whose FIRST window's
last_frame_phase equals i — NOT a deterministic pi on state 0 (tPB2).
Phase 2 measured 92/492 Train videos never showing tPB2 at all, so a
deterministic pi would assign zero prior mass to 92 real, observed first-
phases. Documented limitation, not silently fixed: this models
"development conditional on whatever phase is already reached when
recording starts", not genesis from tPB2 itself — the dataset's own
recording-start-time variability (Phase 2: mean 6.09, std 7.76, raw time
units) makes this a real, not merely theoretical, left-truncation.

Emission fallback for degenerate fits (e.g. tiny synthetic tests with a
single class present): falls back to a UNIFORM emission (b_i(O)=1 for
every i, i.e. observations carry no information) rather than crashing or
producing NaN — the same "explicit fallback, not a silent no-op or a
crash" discipline already used by identity_dynamics.py/persistence.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction

# Mirrors Training/DataSet.py's chronological_phases order exactly (Phase 0
# of this branch confirmed phase_to_index is identical across Train/Val/
# Test). Duplicated here rather than imported from DataSet.py, consistent
# with this whole evaluation/ package's existing discipline of never
# importing Training/'s original modules (see embeddings/build_cache.py's
# module docstring for the same reasoning applied elsewhere).
DEFAULT_PHASE_NAMES = [
    "tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
    "t9+", "tM", "tSB", "tB", "tEB",
]

_NEG_INF = -1e10  # finite stand-in for log(0): behaves as "never selected" in
# logsumexp/argmax while avoiding NaN from (-inf) - (-inf) that a literal
# float("-inf") can trigger in some numpy reduction edge cases.


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        out = np.log(x)
    out[~np.isfinite(out)] = _NEG_INF
    return out


@register_model("hmm")
class HMMModel(Model):
    def __init__(
        self,
        n_states: int = 15,
        state_names: Optional[List[str]] = None,
        transition_smoothing_alpha: float = 1.0,
        transition_decay_rho: float = 0.3,
        emission_prior_epsilon: float = 1e-6,
        logreg_C: float = 1.0,
        logreg_max_iter: int = 1000,
        logreg_class_weight: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_states : int
            Size of the state space. 15 for the real phase taxonomy;
            deliberately configurable (not hardcoded) so unit tests can use
            small synthetic state spaces without any of the real-phase
            machinery.
        state_names : list of str, optional
            Human-readable names, len == n_states. Defaults to
            DEFAULT_PHASE_NAMES (truncated/validated) when n_states == 15;
            otherwise generic "S0","S1",... — never silently guessed for a
            non-15 state space.
        transition_smoothing_alpha : float
            Dirichlet concentration added to the segment-level destination
            counts (see module docstring, A estimation). Exposed here, not
            hardcoded, precisely so run_hmm_evaluation.py can sweep it on
            Val (Phase 3 Section 8).
        transition_decay_rho : float
            Geometric decay rate of the destination prior in (j-i); must be
            in (0, 1]. Also exposed for a Val sweep.
        emission_prior_epsilon : float
            Laplace-style epsilon added to the empirical P(S=i) prior used
            in the Bayes conversion, purely a numerical safety net against
            log(0) for a state absent from a given fit() call (relevant for
            small synthetic tests; never triggers on real Train data, where
            Phase 2 already confirmed all 15 phases are present).
        logreg_C, logreg_max_iter : passed through to sklearn's
            LogisticRegression, same defaults/reasoning as
            identity_dynamics.py.
        logreg_class_weight : passed through to sklearn's LogisticRegression
            as-is (e.g. None (default, matches every prior frozen model in
            this branch exactly) or 'balanced'). Added 2026-08-23 to test
            whether the shared emission's catastrophic per-window accuracy
            on short/underrepresented phases (Results/evaluation/
            semi_hmm_next_step_analysis/emission_diagnostic/) is driven by
            unweighted class imbalance (t9+ has 35,747 Train windows vs
            t3's 3,450 -- a 67x ratio, LogisticRegression's default already
            weights every window equally regardless of class). Purely
            additive: default None reproduces every existing fit() call's
            behavior bit-for-bit (see
            test_logreg_class_weight_default_none_matches_pre_existing_behavior).
        """
        if state_names is not None and len(state_names) != n_states:
            raise ValueError(
                f"state_names has {len(state_names)} entries but n_states={n_states}."
            )
        if not (0.0 < transition_decay_rho <= 1.0):
            raise ValueError(f"transition_decay_rho must be in (0, 1], got {transition_decay_rho}.")
        if transition_smoothing_alpha < 0.0:
            raise ValueError(f"transition_smoothing_alpha must be >= 0, got {transition_smoothing_alpha}.")

        self.n_states = n_states
        if state_names is not None:
            self.state_names = list(state_names)
        elif n_states == len(DEFAULT_PHASE_NAMES):
            self.state_names = list(DEFAULT_PHASE_NAMES)
        else:
            self.state_names = [f"S{i}" for i in range(n_states)]

        self.transition_smoothing_alpha = transition_smoothing_alpha
        self.transition_decay_rho = transition_decay_rho
        self.emission_prior_epsilon = emission_prior_epsilon
        self.logreg_C = logreg_C
        self.logreg_max_iter = logreg_max_iter
        self.logreg_class_weight = logreg_class_weight

        self._log_pi: Optional[np.ndarray] = None
        self._log_A: Optional[np.ndarray] = None
        self._state_log_prior: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._logreg: Optional[LogisticRegression] = None
        self._consistency_base_rate: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # internal: segment (run-length-encoded) sequence per trajectory —
    # same construction as Phase 2's own analysis, reused for consistency.
    # ------------------------------------------------------------------
    @staticmethod
    def _phase_sequence(traj: Trajectory) -> np.ndarray:
        return traj.last_frame_phase.numpy().astype(np.int64)

    @classmethod
    def _segments(cls, traj: Trajectory) -> List[Tuple[int, int, int]]:
        """Returns list of (phase_idx, start_row, end_row) run-length
        segments over the trajectory's last_frame_phase sequence, in
        window_starts order (Trajectory's own documented ordering
        contract — never re-sorted here, matching persistence.py's and
        identity_dynamics.py's existing convention of trusting it)."""
        phases = cls._phase_sequence(traj)
        segments = []
        i = 0
        n = len(phases)
        while i < n:
            j = i
            while j + 1 < n and phases[j + 1] == phases[i]:
                j += 1
            segments.append((int(phases[i]), i, j))
            i = j + 1
        return segments

    def _validate_states(self, trajectories: List[Trajectory]) -> None:
        for traj in trajectories:
            if len(traj) == 0:
                continue
            vals = traj.last_frame_phase.numpy()
            if vals.size and (vals.min() < 0 or vals.max() >= self.n_states):
                raise ValueError(
                    f"Trajectory '{traj.video_name}' has last_frame_phase values "
                    f"outside [0, {self.n_states}) — does not match n_states="
                    f"{self.n_states}. Refusing to silently clip or ignore."
                )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------
    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        """val is accepted (Model interface contract) but unused here —
        this HMM's parameters are all closed-form counts/regressions given
        train alone; hyperparameter SELECTION using val is the job of the
        calling script (run_hmm_evaluation.py), not of fit() itself, the
        same separation of concerns already used by every other model in
        this package (see evaluation/model.py's own docstring)."""
        self._validate_states(train)
        K = self.n_states

        # ---- pi: empirical first-observed-phase distribution ----
        pi_counts = np.zeros(K, dtype=np.float64)
        for traj in train:
            if len(traj) == 0:
                continue
            pi_counts[int(traj.last_frame_phase[0])] += 1.0
        if pi_counts.sum() > 0:
            pi = pi_counts / pi_counts.sum()
        else:
            pi = np.full(K, 1.0 / K)  # no trajectories at all: uninformative fallback, explicit
        self._log_pi = _safe_log(pi)

        # ---- A: window-level self-loop + segment-level smoothed destination ----
        window_counts = np.zeros((K, K), dtype=np.float64)
        for traj in train:
            phases = self._phase_sequence(traj)
            for a, b in zip(phases[:-1], phases[1:]):
                if b >= a:  # structural mask enforced even while counting: a
                    # backward pair should never occur (Phase 2: 0 observed on
                    # 3/3 splits) but is never trusted blindly here either.
                    window_counts[a, b] += 1.0

        segment_counts = np.zeros((K, K), dtype=np.float64)
        for traj in train:
            segs = self._segments(traj)
            for (pa, _, _), (pb, _, _) in zip(segs, segs[1:]):
                if pb >= pa:
                    segment_counts[pa, pb] += 1.0

        A = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            row_total = window_counts[i, :].sum()
            if row_total <= 0:
                # State i was never observed as a "from" state at window
                # level at all — no information to estimate a hazard.
                # Explicit fallback: stay forever, documented, not a crash.
                A[i, i] = 1.0
                continue
            a_ii = window_counts[i, i] / row_total
            A[i, i] = a_ii
            hazard = 1.0 - a_ii
            if hazard <= 0 or i == K - 1:
                continue  # nothing to distribute (already fully self-looping,
                # or i is the terminal state with no j > i to move to)
            targets = np.arange(i + 1, K)
            prior = self.transition_decay_rho ** (targets - i - 1)
            prior = prior / prior.sum()
            counts = segment_counts[i, targets]
            alpha = self.transition_smoothing_alpha
            weighted = counts + alpha * prior
            denom = weighted.sum()
            if denom <= 0:
                dest_dist = np.full(len(targets), 1.0 / len(targets))
            else:
                dest_dist = weighted / denom
            A[i, targets] = hazard * dest_dist

        # Hard structural enforcement (belt-and-braces): any j < i is
        # exactly zero, regardless of how A was built above.
        mask = np.triu(np.ones((K, K), dtype=bool))
        A = np.where(mask, A, 0.0)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A = A / row_sums
        self._log_A = _safe_log(A)
        self._log_A[~mask] = _NEG_INF  # -inf, not just log(~0), for forbidden cells

        # ---- emission: hybrid discriminative -> pseudo-likelihood ----
        all_embeddings = [traj.embeddings for traj in train if len(traj) > 0]
        all_phases = [traj.last_frame_phase for traj in train if len(traj) > 0]
        if all_embeddings:
            X = torch.cat(all_embeddings, dim=0).float().numpy()
            y = torch.cat(all_phases, dim=0).numpy()
        else:
            X = np.empty((0, 1))
            y = np.empty((0,), dtype=np.int64)

        prior_counts = np.zeros(K, dtype=np.float64)
        for cls in range(K):
            prior_counts[cls] = float((y == cls).sum())
        eps = self.emission_prior_epsilon
        prior = (prior_counts + eps) / (prior_counts.sum() + K * eps)
        self._state_log_prior = _safe_log(prior)

        n_classes_present = len(np.unique(y)) if len(y) else 0
        if n_classes_present < 2:
            self._scaler = None
            self._logreg = None
        else:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            # multi_class left at sklearn's default (no longer passed
            # explicitly): recent sklearn deprecates the "multinomial"
            # string in favor of always fitting a proper multinomial model
            # for >2 classes with the default lbfgs solver — passing it
            # explicitly only produced a FutureWarning with no behavior
            # difference here.
            self._logreg = LogisticRegression(
                C=self.logreg_C, max_iter=self.logreg_max_iter, class_weight=self.logreg_class_weight
            )
            self._logreg.fit(X_scaled, y)
            # A class absent from train (possible for a small n_states
            # synthetic test, or a real but genuinely all-15-present Train
            # split normally has none of this) would make predict_log_proba
            # silently misalign columns against [0..K); guard explicitly.
            fitted_classes = set(self._logreg.classes_.tolist())
            missing = set(range(K)) - fitted_classes
            if missing:
                self._logreg_missing_classes = sorted(missing)
            else:
                self._logreg_missing_classes = []

        # ---- consistency_flag base rate (Model-compatible API fallback) ----
        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._consistency_base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} method called before fit().")

    # ------------------------------------------------------------------
    # emission log-probabilities for one trajectory: log b_i(O_t), shape (T, K)
    # ------------------------------------------------------------------
    def _log_emission(self, traj: Trajectory) -> np.ndarray:
        T = len(traj)
        K = self.n_states
        if T == 0:
            return np.empty((0, K))
        if self._logreg is None or self._scaler is None:
            # Degenerate-fit fallback: uniform emission, observations carry
            # no information (documented in the module docstring).
            return np.zeros((T, K))

        X = traj.embeddings.float().numpy()
        X_scaled = self._scaler.transform(X)
        log_post_fitted = self._logreg.predict_log_proba(X_scaled)  # (T, n_classes_present)

        log_post = np.full((T, K), _NEG_INF)
        for col, cls in enumerate(self._logreg.classes_):
            log_post[:, int(cls)] = log_post_fitted[:, col]
        # classes never seen in train (self._logreg_missing_classes) keep
        # log_post = _NEG_INF -> those states are simply unreachable via
        # emission evidence, an explicit, documented consequence of never
        # having observed them, not a bug.

        log_b = log_post - self._state_log_prior[None, :]
        return log_b

    # ------------------------------------------------------------------
    # forward algorithm (log-space), causal by construction: log_alpha[t]
    # is a pure function of O_1..O_t and the fixed model parameters.
    # ------------------------------------------------------------------
    def _log_forward(self, log_b: np.ndarray) -> np.ndarray:
        T, K = log_b.shape
        log_alpha = np.empty((T, K))
        if T == 0:
            return log_alpha
        log_alpha[0] = self._log_pi + log_b[0]
        for t in range(1, T):
            # (K,K): entry [i,j] = log_alpha[t-1,i] + log_A[i,j]
            trans = log_alpha[t - 1][:, None] + self._log_A
            log_alpha[t] = logsumexp(trans, axis=0) + log_b[t]
        return log_alpha

    def _log_backward(self, log_b: np.ndarray) -> np.ndarray:
        T, K = log_b.shape
        log_beta = np.empty((T, K))
        if T == 0:
            return log_beta
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            # (K,K): entry [i,j] = log_A[i,j] + log_b[t+1,j] + log_beta[t+1,j]
            terms = self._log_A + (log_b[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = logsumexp(terms, axis=1)
        return log_beta

    @staticmethod
    def _normalize_log(log_vec: np.ndarray) -> np.ndarray:
        return np.exp(log_vec - logsumexp(log_vec, axis=-1, keepdims=True))

    # ------------------------------------------------------------------
    # Rich, HMM-specific API
    # ------------------------------------------------------------------
    def filtering(self, traj: Trajectory) -> np.ndarray:
        """P(S_t=i | O_1:t), shape (T, n_states). Causal: row t depends only
        on O_1..O_t. The correct quantity for a live/in-progress reporting
        of "current phase"."""
        self._require_fitted()
        log_b = self._log_emission(traj)
        log_alpha = self._log_forward(log_b)
        if len(traj) == 0:
            return np.empty((0, self.n_states))
        return self._normalize_log(log_alpha)

    def smoothing(self, traj: Trajectory) -> np.ndarray:
        """P(S_t=i | O_1:T), shape (T, n_states). NON-causal: uses the whole
        trajectory, including windows after t. Retrospective analysis only
        — never for a "current state" report on an in-progress recording."""
        self._require_fitted()
        log_b = self._log_emission(traj)
        log_alpha = self._log_forward(log_b)
        log_beta = self._log_backward(log_b)
        if len(traj) == 0:
            return np.empty((0, self.n_states))
        return self._normalize_log(log_alpha + log_beta)

    def predict_next(self, traj: Trajectory) -> np.ndarray:
        """P(S_{t+1}=j | O_1:t) = sum_i filtering_t(i) * A[i,j], shape
        (T, n_states) — row t is the predictive distribution for the window
        AFTER t. This is NOT A[i,j] itself (A is a fixed model parameter;
        this is a prediction conditioned on the observations seen so far) —
        see test_predict_next_differs_from_raw_A_row."""
        self._require_fitted()
        filt = self.filtering(traj)
        if len(traj) == 0:
            return np.empty((0, self.n_states))
        A = np.exp(self._log_A)
        A[~np.isfinite(np.exp(self._log_A))] = 0.0  # already 0 via exp(-inf)->0, defensive only
        return filt @ A

    def viterbi(self, traj: Trajectory) -> Tuple[List[int], float]:
        """argmax_S P(S_1:T | O_1:T) via the standard max-product (log-space
        max-sum) recursion, masked by the same j>=i constraint as A itself
        (structurally impossible to select a backward step, not merely
        discouraged). Returns (decoded_state_indices, log_score) — log_score
        is the log of the BEST PATH's joint probability under this model,
        not a normalized "probability of this exact trajectory" (see
        sequence_log_likelihood for the distinct, whole-sequence-marginal
        quantity, and docs/HMM_RESEARCH_PLAN.md Section 7 for why these are
        not the same thing)."""
        self._require_fitted()
        log_b = self._log_emission(traj)
        T, K = log_b.shape
        if T == 0:
            return [], 0.0
        log_delta = np.empty((T, K))
        backptr = np.zeros((T, K), dtype=np.int64)
        log_delta[0] = self._log_pi + log_b[0]
        for t in range(1, T):
            scores = log_delta[t - 1][:, None] + self._log_A  # (K,K): [i,j]
            backptr[t] = np.argmax(scores, axis=0)
            log_delta[t] = scores[backptr[t], np.arange(K)] + log_b[t]
        best_last = int(np.argmax(log_delta[T - 1]))
        best_score = float(log_delta[T - 1, best_last])
        path = [best_last]
        for t in range(T - 1, 0, -1):
            path.append(int(backptr[t, path[-1]]))
        path.reverse()
        return path, best_score

    def sequence_log_likelihood(self, traj: Trajectory) -> float:
        """log P(O_1:T) via the forward algorithm's total mass at T.

        CAVEAT (see the module docstring's full derivation): under the
        hybrid pseudo-likelihood emission, this is NOT the true sequence
        log-likelihood — it is off by the unknown, unestimated additive
        constant sum_t log P(O_t) in log-space. It IS valid for comparing
        two HMMModel configurations that share this same emission family
        (e.g. different transition_smoothing_alpha/transition_decay_rho on
        Val) since that missing constant only depends on the marginal
        distribution of the embeddings, not on pi/A. It is NOT valid to
        compare this number directly against a log-likelihood computed
        under a different emission family (e.g. a future PCA+Gaussian
        variant) without accounting for that missing term."""
        self._require_fitted()
        log_b = self._log_emission(traj)
        if len(traj) == 0:
            return 0.0
        log_alpha = self._log_forward(log_b)
        return float(logsumexp(log_alpha[-1]))

    @staticmethod
    def entropy(posterior_row: np.ndarray) -> float:
        """Shannon entropy (nats) of one posterior distribution — the
        uncertainty measure used in the reporting schema (Section H below)."""
        p = np.clip(posterior_row, 1e-300, 1.0)
        return float(-(p * np.log(p)).sum())

    def explain(self, traj: Trajectory, mode: str = "filtering") -> List[Dict]:
        """Builds the structured (non-textual) per-window reporting records
        for one trajectory — the HMM computes only typed numeric facts here,
        never prose (docs/HMM_RESEARCH_PLAN.md Section 8's discipline: "HMM
        ne doit PAS generer de texte"). `mode` selects which posterior
        populates 'phase_posterior':
          "filtering" (causal, for a live/in-progress report) or
          "smoothing" (retrospective, requires the full trajectory).
        'next_phase_probability' is always the causal predictive
        distribution (predict_next), regardless of mode, since it is by
        definition about what comes after what has been observed so far."""
        self._require_fitted()
        if mode not in ("filtering", "smoothing"):
            raise ValueError(f"mode must be 'filtering' or 'smoothing', got {mode!r}.")
        posterior = self.filtering(traj) if mode == "filtering" else self.smoothing(traj)
        next_dist = self.predict_next(traj)
        records = []
        for t in range(len(traj)):
            post_t = posterior[t]
            argmax_i = int(np.argmax(post_t))
            record = {
                "video": traj.video_name,
                "window": int(traj.window_starts[t]),
                "phase": self.state_names[argmax_i],
                "phase_posterior": {
                    self.state_names[i]: float(post_t[i]) for i in range(self.n_states)
                },
                "next_phase_probability": {
                    self.state_names[j]: float(next_dist[t, j]) for j in range(self.n_states)
                },
                "entropy": self.entropy(post_t),
                "inference_mode": mode,
            }
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Model-compatible API
    # ------------------------------------------------------------------
    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        """consistency_flag_prob[t] := P(S_t != S_{t-1} | O_1:t), the exact
        (given this model) causal "two-slice" filtered joint probability
        that the decoded state changed between window t-1 and window t —
        NOT an approximation via independent marginals. window 0 (no t-1)
        falls back to the empirical Train consistency_flag base rate, the
        same convention already used by identity_dynamics.py/persistence.py
        for their own undefined-context cases.

        KNOWN, DEMONSTRATED MISMATCH WITH `consistency_flag` (diagnosed
        2026-08-21, not yet fixed): the quantity above answers "did the
        decoded phase change between the PREVIOUS window and this one"
        (window_start ordinal lag 1, i.e. ~1 raw frame apart at stride=1).
        `consistency_flag` (Training/DataSet.py:310,
        `0 if labels[0]==labels[-1] else 1`) is a DIFFERENT event: whether
        this window's OWN first frame and last frame differ in phase — an
        intra-window drift spanning `window_size` (8, at Train/Val
        real-data scale) raw frames, not an inter-window, ~1-frame event.
        Measured on real Val data: P(consistency_flag=1) = 13.24% vs.
        P(S_t != S_{t-1}) = 2.54% — a ~5.2x rate mismatch that is the
        dominant, measured explanation for this model's severe apparent
        under-confidence when scored against `consistency_flag` (dominant
        prediction bin [0,0.1): mean predicted 1.2% vs. observed 12.9%
        under `consistency_flag`, vs. mean predicted 1.2% vs. observed
        2.4% when the SAME frozen model's SAME predictions are instead
        scored against the aligned event above — same model, same
        predict() output, ~5x smaller calibration gap once scored against
        the event it actually computes). The forward-algorithm computation
        of P(S_t != S_{t-1} | O_1:t) itself is exact (brute-force
        path-enumeration-verified to 1e-9, see
        test_predict_two_slice_joint_matches_brute_force_enumeration in
        Tests/evaluation/models/test_hmm.py) — this is NOT a math bug or
        an implementation bug, it is a target-definition/alignment
        mismatch: consistency_flag_prob is not yet a validated proxy for
        consistency_flag. The mathematically correct fix would compare
        S_t to S_{t-(window_size-1)} (matched by `window_starts` value,
        not by trajectory index, since filtered-out windows can make
        those differ) via a proper k-step "skip-forward" two-slice joint
        — NOT implemented here (would require new forward-recursion
        machinery, not a one-line change); flagged as future work, not
        done as part of this diagnosis. Until that exists, do not treat
        this model's `consistency_flag`-scored AUROC/Brier/calibration as
        validated evidence about how well the HMM captures within-window
        transitions."""
        self._require_fitted()
        predictions = []
        for traj in trajectories:
            T = len(traj)
            if T == 0:
                predictions.append(
                    TrajectoryPrediction(
                        video_name=traj.video_name,
                        window_starts=traj.window_starts,
                        consistency_flag_prob=torch.empty(0),
                    )
                )
                continue

            log_b = self._log_emission(traj)
            log_alpha = self._log_forward(log_b)
            probs = np.full(T, self._consistency_base_rate, dtype=np.float64)
            for t in range(1, T):
                # joint_log[i,j] = log_alpha[t-1,i] + log_A[i,j] + log_b[t,j]
                joint_log = log_alpha[t - 1][:, None] + self._log_A + log_b[t][None, :]
                total_log = logsumexp(joint_log)
                same_state_log = logsumexp(np.diag(joint_log))
                probs[t] = 1.0 - float(np.exp(same_state_log - total_log))
            predictions.append(
                TrajectoryPrediction(
                    video_name=traj.video_name,
                    window_starts=traj.window_starts,
                    consistency_flag_prob=torch.from_numpy(probs.astype(np.float32)),
                )
            )
        return predictions

    # ------------------------------------------------------------------
    # k-step two-slice joint (diagnosed 2026-08-21, implemented 2026-08-21):
    # generalizes predict()'s lag-1 event to an arbitrary lag k, so that
    # k=(window_size-1) can express the event consistency_flag actually
    # measures (first-frame-phase vs. last-frame-phase of the SAME
    # window) rather than predict()'s lag-1 adjacent-window event. See
    # predict_k_step_transition_probability's own docstring for the full
    # derivation. predict() ITSELF IS UNCHANGED (still lag-1, still the
    # exact formula it always was) -- this is purely additive.
    # ------------------------------------------------------------------
    def _log_k_step_conditional_forward(self, log_b: np.ndarray, t0: int, k: int) -> np.ndarray:
        """log_gamma[i, j] := log P(S_{t0+k}=j, O_{t0+1:t0+k} | S_{t0}=i),
        shape (K, K) -- the k-step "conditional forward" starting from a
        point mass at each state i at trajectory index t0, propagated
        using the SAME transition matrix A and (pseudo-)emissions b as
        the ordinary forward algorithm, over CONSECUTIVE cached rows
        log_b[t0+1 .. t0+k]. Causal by construction: only ever reads
        log_b rows in [t0+1, t0+k], nothing beyond. Reduces, for k=1, to
        exactly log_A[i, :] + log_b[t0+1, :] -- the same one-step term
        predict() uses (see test_predict_k_step_k_equals_1_matches_predict).

        Caller's responsibility (enforced by
        predict_k_step_transition_probability, not here): t0..t0+k must
        be CONSECUTIVE TRAJECTORY INDICES corresponding to CONSECUTIVE
        window_starts values (no window filtered out in between) -- this
        method has no way to detect a gap itself, since by the time a
        (T, K) log_b array reaches it, filtered-out windows already don't
        exist as rows at all."""
        K = self.n_states
        log_G = np.where(np.eye(K, dtype=bool), 0.0, _NEG_INF)  # (K_start, K_current), log(identity)
        for m in range(1, k + 1):
            # trans[i, j_prev, j] = log_G[i, j_prev] + log_A[j_prev, j]
            trans = log_G[:, :, None] + self._log_A[None, :, :]
            log_G = logsumexp(trans, axis=1) + log_b[t0 + m][None, :]
        return log_G

    def predict_k_step_transition_probability(self, traj: Trajectory, k: int = 1) -> np.ndarray:
        """P(S_t != S_{t-k} | O_1:t), shape (T,) -- the causal, EXACT
        (not approximated by independent marginals P(S_{t-k})*P(S_t))
        k-step two-slice filtered joint transition probability. This
        generalizes predict()'s consistency_flag_prob, which is exactly
        this function at k=1 (see test_predict_k_step_k_equals_1_matches_predict
        for the bit-for-bit equivalence proof).

        WHY k=(window_size-1) IS THE quantity consistency_flag needs
        ------------------------------------------------------------------
        `consistency_flag` (Training/DataSet.py:310) is
        `0 if labels[0]==labels[-1] else 1` over ONE window's own
        `window_size` (8, at real Train/Val scale) consecutive raw
        frames -- i.e. it is 1 iff the phase at that window's FIRST frame
        differs from the phase at that window's LAST frame. This model's
        state S_t is, by construction (`_phase_sequence` =
        `last_frame_phase`), the phase AT THE LAST FRAME of window t. At
        stride=1, window t's FIRST frame is the exact same raw frame as
        window (t-(window_size-1))'s LAST frame -- so, whenever that
        earlier window exists in this trajectory,
        S_{t-(window_size-1)} == first_frame_phase(window t) EXACTLY (an
        arithmetic identity of the windowing scheme, not an
        approximation). Hence P(S_t != S_{t-(window_size-1)} | O_1:t) IS
        P(S_last != S_first | O_1:t) for window t -- pass
        k=window_size-1 (7 at real Train/Val scale) to get it.

        window_starts MATCHING, NOT trajectory-index MATCHING
        ------------------------------------------------------------------
        Positions are matched by `window_starts` VALUE
        (`window_starts[t] - k`), never by assuming trajectory index
        `t-k` directly -- DataSet.py's phase-consistency filter can drop
        a window (`continue` in `_build_sequences`), which would silently
        misalign a pure index-based lookup. A position t only gets a
        computed value if a window with `window_starts` value exactly
        `window_starts[t] - k` exists AND is reachable via exactly k
        CONSECUTIVE trajectory-index steps from t (checked via
        `t0 + k == t`, which is both necessary and sufficient for full
        window_starts contiguity between t0 and t, since window_starts is
        strictly increasing and every trajectory-index step advances it
        by >= 1) -- i.e. no window was filtered out anywhere in that
        span. Empirically, on the real Train/Val cache (verified
        2026-08-21), window_starts has ZERO such gaps for either split,
        so this reduces to plain index matching in practice today -- but
        this method never assumes that structurally; a future cache with
        gaps would fall back to the base rate at the affected positions
        rather than silently misaligning (see
        test_predict_k_step_uses_window_starts_value_not_index_when_a_gap_exists).

        Positions with no valid k-step match (not enough history, or a
        gap in window_starts) fall back to the empirical Train
        consistency_flag base rate -- same convention as predict()'s
        t=0 case, generalized.

        Parameters
        ----------
        k : int, >= 1. k=1 reproduces predict() exactly.
        """
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        T = len(traj)
        probs = np.full(T, self._consistency_base_rate, dtype=np.float64)
        if T == 0:
            return probs

        log_b = self._log_emission(traj)
        log_alpha = self._log_forward(log_b)
        start_to_index = {s: i for i, s in enumerate(traj.window_starts)}

        for t in range(T):
            target_start = traj.window_starts[t] - k
            t0 = start_to_index.get(target_start)
            if t0 is None or t0 + k != t:
                continue  # no window exactly k positions earlier, reachable
                # via k gap-free trajectory-index steps -> base-rate fallback
                # (already set above)
            log_gamma = self._log_k_step_conditional_forward(log_b, t0, k)  # (K,K): [i,j]
            joint_log = log_alpha[t0][:, None] + log_gamma  # log P(S_t0=i, S_t=j, O_1:t)
            total_log = logsumexp(joint_log)
            same_state_log = logsumexp(np.diag(joint_log))
            probs[t] = 1.0 - float(np.exp(same_state_log - total_log))
        return probs

    def predict_k_step(self, trajectories: List[Trajectory], k: int = 1) -> List[TrajectoryPrediction]:
        """Batch wrapper around predict_k_step_transition_probability(),
        same TrajectoryPrediction output contract as predict() (directly
        consumable by evaluation.metrics.compute_metrics, unmodified),
        for the generalized k-step lag. k=1 is bit-for-bit identical to
        predict()'s output; k=(window_size-1) is the quantity that
        actually matches consistency_flag's own definition -- see
        predict_k_step_transition_probability's docstring."""
        self._require_fitted()
        predictions = []
        for traj in trajectories:
            probs = self.predict_k_step_transition_probability(traj, k=k)
            predictions.append(
                TrajectoryPrediction(
                    video_name=traj.video_name,
                    window_starts=traj.window_starts,
                    consistency_flag_prob=torch.from_numpy(probs.astype(np.float32)),
                )
            )
        return predictions

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "n_states": self.n_states,
            "state_names": self.state_names,
            "transition_smoothing_alpha": self.transition_smoothing_alpha,
            "transition_decay_rho": self.transition_decay_rho,
            "emission_prior_epsilon": self.emission_prior_epsilon,
            "logreg_C": self.logreg_C,
            "logreg_max_iter": self.logreg_max_iter,
            "logreg_class_weight": self.logreg_class_weight,
            "fitted": self._fitted,
        }
        if self._fitted:
            state["log_pi"] = self._log_pi.tolist()
            state["log_A"] = self._log_A.tolist()
            state["state_log_prior"] = self._state_log_prior.tolist()
            state["consistency_base_rate"] = self._consistency_base_rate
            if self._logreg is not None and self._scaler is not None:
                state["scaler_mean"] = self._scaler.mean_.tolist()
                state["scaler_scale"] = self._scaler.scale_.tolist()
                state["logreg_coef"] = self._logreg.coef_.tolist()
                state["logreg_intercept"] = self._logreg.intercept_.tolist()
                state["logreg_classes"] = self._logreg.classes_.tolist()
        (path / "state.json").write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path) -> "HMMModel":
        state = json.loads((path / "state.json").read_text())
        model = cls(
            n_states=state["n_states"],
            state_names=state["state_names"],
            transition_smoothing_alpha=state["transition_smoothing_alpha"],
            transition_decay_rho=state["transition_decay_rho"],
            emission_prior_epsilon=state["emission_prior_epsilon"],
            logreg_C=state["logreg_C"],
            logreg_max_iter=state["logreg_max_iter"],
            # .get(): old saved state.json files (every model frozen before
            # 2026-08-23) never had this key -- None reproduces their exact
            # original fit() behavior, not a guess.
            logreg_class_weight=state.get("logreg_class_weight"),
        )
        if state.get("fitted"):
            model._log_pi = np.asarray(state["log_pi"])
            model._log_A = np.asarray(state["log_A"])
            model._state_log_prior = np.asarray(state["state_log_prior"])
            model._consistency_base_rate = state["consistency_base_rate"]
            if "logreg_coef" in state:
                scaler = StandardScaler()
                scaler.mean_ = np.asarray(state["scaler_mean"])
                scaler.scale_ = np.asarray(state["scaler_scale"])
                scaler.var_ = scaler.scale_**2
                scaler.n_features_in_ = len(scaler.mean_)

                logreg = LogisticRegression()
                logreg.coef_ = np.asarray(state["logreg_coef"])
                logreg.intercept_ = np.asarray(state["logreg_intercept"])
                logreg.classes_ = np.asarray(state["logreg_classes"])
                logreg.n_features_in_ = logreg.coef_.shape[1]

                model._scaler = scaler
                model._logreg = logreg
            else:
                model._scaler = None
                model._logreg = None
            model._fitted = True
        return model
