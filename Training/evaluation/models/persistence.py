"""
PersistenceModel — the classic time-series "naive forecast" baseline,
applied to cached embeddings.

Scientific assumption
----------------------
    The future embedding equals the current embedding:  e_hat_i = e_{i-1}

For window i of a trajectory (windows ordered by window_start, indices
1..T), this model's forecast for window i's embedding is simply the
previous window's embedding e_{i-1} — the null hypothesis for any
dynamical system: "nothing changes." The residual under that assumption
is

    d_i = || e_i - e_{i-1} ||_2         (i = 2..T)

Because consecutive windows overlap heavily (window i and window i-1
share window_size - stride of their source frames — see DataSet.py's
_create_sequences), d_i is expected to be small during ordinary,
non-transitioning development and elevated specifically when the small
non-overlapping boundary region between the two windows contains a
developmental event. That is the mechanism by which "how much the
embedding moved since the previous window" becomes informative about
"did the phase change within the current window" (consistency_flag_i) —
without the model ever seeing more than two consecutive embeddings.

CORRECTION (this file previously claimed to formalize "Identity Dynamics"
exactly — that was imprecise and has been fixed): this model and
models/identity_dynamics.py's IdentityDynamicsModel both look like
"future = current" at a glance, but they are genuinely different.
IdentityDynamicsModel uses ONLY window i's own embedding (zero reference
to any other window) — it is the true dx/dt = 0 / no-processing floor.
This model uses the CHANGE between window i-1 and window i, which is
already a (minimal) use of cross-window, temporal information.
IdentityDynamicsModel is the correct baseline below this one in the
ablation ladder — see its module docstring for the full three-model
comparison (BaseRateModel -> IdentityDynamicsModel -> PersistenceModel).
Persistence beating IdentityDynamics is the actual, direct test of
whether temporal/dynamical reasoning adds anything beyond embedding
content alone (RESEARCH_BLUEPRINT.md Part I, H1) — not a comparison this
model performs alone.

Prediction rule
----------------
    P(consistency_flag_i = 1) = sigmoid(a * d_i + b)

with (a, b) fit by logistic regression on (d_i, consistency_flag_i) pairs
from the training trajectories — the minimal complexity that turns the
persistence assumption into a calibrated probability rather than a hard
threshold.

Edge cases, handled explicitly rather than silently
-----------------------------------------------------
- The first window of a trajectory has no i-1 to compare against. Its
  prediction falls back to the empirical training base rate of
  consistency_flag.
- A trajectory containing only one window has no defined d_i at all;
  every one of its windows uses the base-rate fallback.
- If the training set contains only one class of consistency_flag,
  logistic regression is undefined; fit() detects this and falls back to
  always predicting the base rate, rather than letting sklearn raise or
  silently produce a degenerate fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction


@register_model("persistence")
class PersistenceModel(Model):
    def __init__(self) -> None:
        self._base_rate: Optional[float] = None
        self._logreg: Optional[LogisticRegression] = None

    # ---- internal ----

    @staticmethod
    def _consecutive_distances(traj: Trajectory) -> np.ndarray:
        """d_i = ||e_i - e_{i-1}||_2 for i = 2..T. Shape (T-1,); empty
        for a trajectory with fewer than 2 windows."""
        if len(traj) < 2:
            return np.empty((0,), dtype=np.float32)
        emb = traj.embeddings.float()
        diffs = emb[1:] - emb[:-1]
        return diffs.norm(dim=1).numpy()

    # ---- Model interface ----

    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        distances: List[float] = []
        targets: List[int] = []
        for traj in train:
            if len(traj) < 2:
                continue
            d = self._consecutive_distances(traj)
            # d_i corresponds to window i (i=2..T) -> consistency_flag[1:]
            distances.extend(d.tolist())
            targets.extend(traj.consistency_flag[1:].tolist())

        distances_arr = np.asarray(distances, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.int64)

        if len(distances_arr) < 2 or len(np.unique(targets_arr)) < 2:
            # Not enough signal to fit anything beyond the base rate —
            # explicit, not a silent no-op or a crash.
            self._logreg = None
            return

        self._logreg = LogisticRegression()
        self._logreg.fit(distances_arr.reshape(-1, 1), targets_arr)

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        if self._base_rate is None:
            raise RuntimeError("PersistenceModel.predict() called before fit().")

        predictions = []
        for traj in trajectories:
            probs = np.full(len(traj), self._base_rate, dtype=np.float32)
            if len(traj) >= 2 and self._logreg is not None:
                d = self._consecutive_distances(traj)
                fitted = self._logreg.predict_proba(d.reshape(-1, 1))[:, 1]
                probs[1:] = fitted  # index 0 (first window) keeps the base-rate fallback
            predictions.append(
                TrajectoryPrediction(
                    video_name=traj.video_name,
                    window_starts=traj.window_starts,
                    consistency_flag_prob=torch.from_numpy(probs),
                )
            )
        return predictions

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {"base_rate": self._base_rate}
        if self._logreg is not None:
            # Saved as plain JSON (coefficients only), not pickled — the
            # same discipline used everywhere else in this project
            # (CacheManifest, ExperimentConfig): transparent, inspectable,
            # and immune to sklearn-version pickle incompatibility, which
            # is a real risk given requirements.txt is currently unpinned.
            state["logreg_coef"] = self._logreg.coef_.tolist()
            state["logreg_intercept"] = self._logreg.intercept_.tolist()
            state["logreg_classes"] = self._logreg.classes_.tolist()
        (path / "state.json").write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "PersistenceModel":
        state = json.loads((path / "state.json").read_text())
        model = cls()
        model._base_rate = state["base_rate"]
        if "logreg_coef" in state:
            logreg = LogisticRegression()
            logreg.coef_ = np.asarray(state["logreg_coef"])
            logreg.intercept_ = np.asarray(state["logreg_intercept"])
            logreg.classes_ = np.asarray(state["logreg_classes"])
            logreg.n_features_in_ = 1
            model._logreg = logreg
        return model
