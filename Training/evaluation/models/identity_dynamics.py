"""
IdentityDynamicsModel — the "no dynamics processing" baseline: the
dynamics-processing step is the identity map.

Scientific assumption
----------------------
    x(t+Delta) = x(t)

Read literally this says the state doesn't change — but "Identity
Dynamics" here is not a forecasting rule (see models/persistence.py for
that; do not confuse the two, despite the similar-looking assumption,
they test different things). It is a statement about the
DYNAMICS-PROCESSING STEP itself: the latent trajectory a dynamics model
would produce is defined to equal the raw, unprocessed observation
sequence, unchanged —

    tau(t_i) = x(t_i)   for every i

i.e. Dynamics == the identity function applied to the whole sequence,
hence the name. No smoothing, no filtering, no reference to any other
window, no memory of history. Formally:

    P(consistency_flag_i = 1 | x_1, ..., x_T) = P(consistency_flag_i = 1 | x_i)

Window i's prediction is conditionally independent of every other window
given its own embedding. This is the precise, checkable meaning of
"isolates the effect of temporal reasoning": there is none, by
construction — see test_predictions_are_invariant_to_window_order in the
accompanying test file for the direct, executable version of this claim.

Where this sits relative to the other two baselines in this package
----------------------------------------------------------------------
    BaseRateModel          : uses NO information at all (not even x_i)
    IdentityDynamicsModel  : uses ONLY x_i                    (this model)
    PersistenceModel       : uses the CHANGE between x_i and x_{i-1}

Each step adds exactly one piece of information the previous one lacked.
IdentityDynamics beating BaseRate is evidence the embedding's CONTENT is
informative. Persistence beating IdentityDynamics is evidence the
embedding's CHANGE OVER TIME carries information content alone does not —
the direct empirical test of whether temporal/dynamical reasoning adds
anything at all (RESEARCH_BLUEPRINT.md Part I, H1).

Prediction rule
----------------
    P(consistency_flag_i = 1) = sigmoid(w . x_i + b)

(w, b) fit by L2-regularized logistic regression on (x_i,
consistency_flag_i) pairs, pooled across every window of every training
trajectory — pooling is the correct, faithful implementation of
"independent per-window inference": there is no trajectory structure to
respect here, by design.

Embeddings are standardized (zero mean, unit variance per dimension, fit
on the training set only) before the logistic regression — without this,
raw embedding features at very different scales are a well-known cause of
poor convergence and poor performance for a linear classifier, which
would make this an unfair floor for later models to beat.

Edge cases, handled explicitly rather than silently
-----------------------------------------------------
- If training data contains only one class of consistency_flag, logistic
  regression is undefined; fit() falls back to always predicting the
  empirical base rate rather than letting sklearn raise.
- Zero training trajectories: falls back to a base rate of 0.5, the same
  convention as the other two models in this package.

Expected strengths and weaknesses (see models/README.md for the full
version tying this back to the other baselines)
----------------------------------------------------------------------
Strengths: uses actual embedding content, unlike BaseRateModel; gives a
direct test of whether the encoder's own multi-frame windowed
architecture is extracting anything a linear probe on its pooled output
doesn't already contain; robust to window ordering/spacing by
construction, since no cross-window structure is used at all.

Weaknesses: structurally blind to anything that only shows up as a
CHANGE between windows — it cannot represent arrest, direction, or
velocity in any form, by design, not oversight. Real overfitting risk
given a potentially high embedding dimension, a moderate window count,
and heavy within-video correlation between overlapping windows (many
windows per video mean the effective number of independent training
examples is much smaller than the raw window count) — exactly the risk
evaluation.explore's phase-vs-patient clustering diagnostics exist to
surface ahead of time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction


@register_model("identity_dynamics")
class IdentityDynamicsModel(Model):
    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        """
        Parameters
        ----------
        C : float
            Inverse L2 regularization strength, passed through to
            sklearn.linear_model.LogisticRegression. Left at sklearn's own
            default (1.0) — this is meant to be the simplest reasonable
            version of this model, not a tuned one.
        max_iter : int
            Raised above sklearn's default of 100 — with embedding
            dimensionality potentially in the hundreds, the default is a
            real risk of non-convergence on this kind of feature.
        """
        self._C = C
        self._max_iter = max_iter
        self._base_rate: Optional[float] = None
        self._scaler: Optional[StandardScaler] = None
        self._logreg: Optional[LogisticRegression] = None

    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        if not train:
            self._scaler = None
            self._logreg = None
            return

        X = torch.cat([t.embeddings for t in train], dim=0).float().numpy()
        y = torch.cat([t.consistency_flag for t in train], dim=0).numpy()

        if len(np.unique(y)) < 2:
            # Not enough signal to fit a classifier — explicit, not a
            # silent no-op or a crash from sklearn.
            self._scaler = None
            self._logreg = None
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._logreg = LogisticRegression(C=self._C, max_iter=self._max_iter)
        self._logreg.fit(X_scaled, y)

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        if self._base_rate is None:
            raise RuntimeError("IdentityDynamicsModel.predict() called before fit().")

        predictions = []
        for traj in trajectories:
            if self._logreg is not None and self._scaler is not None:
                X = traj.embeddings.float().numpy()
                X_scaled = self._scaler.transform(X)
                probs = self._logreg.predict_proba(X_scaled)[:, 1].astype(np.float32)
            else:
                probs = np.full(len(traj), self._base_rate, dtype=np.float32)
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
        state = {"base_rate": self._base_rate, "C": self._C, "max_iter": self._max_iter}
        if self._logreg is not None and self._scaler is not None:
            # Plain JSON, not pickled — same discipline as every other
            # model in this package: transparent, inspectable, and immune
            # to sklearn-version pickle incompatibility, a real risk given
            # requirements.txt is currently unpinned.
            state["scaler_mean"] = self._scaler.mean_.tolist()
            state["scaler_scale"] = self._scaler.scale_.tolist()
            state["logreg_coef"] = self._logreg.coef_.tolist()
            state["logreg_intercept"] = self._logreg.intercept_.tolist()
            state["logreg_classes"] = self._logreg.classes_.tolist()
        (path / "state.json").write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "IdentityDynamicsModel":
        state = json.loads((path / "state.json").read_text())
        model = cls(C=state.get("C", 1.0), max_iter=state.get("max_iter", 1000))
        model._base_rate = state["base_rate"]
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
        return model
