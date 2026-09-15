"""
LinearStateSpaceModel — the first model in this package that actually
LEARNS a transition, rather than assuming one.

Model
------
    z_{i+1} = A z_i + b + w_i,   w_i ~ N(0, Q)          (latent dynamics)
    e_i     = C z_i + mu + v_i,  v_i ~ N(0, R)           (observation model, optional)

z_i in R^d is the latent state; e_i in R^D is the cached embedding
actually observed; A in R^{dxd} is the learned transition matrix; C in
R^{Dxd} is the observation matrix. When C is not used, d = D and z_i =
e_i directly — the model then operates on raw embeddings.

Fitting A, b: ridge-regularized least squares on within-trajectory
consecutive (z_i, z_{i+1}) pairs, never across trajectory boundaries —
see test_fit_never_pairs_across_trajectory_boundaries for the direct,
executable version of this correctness requirement.

Fitting C: PCA on pooled training embeddings. This is the
maximum-likelihood C for a linear-Gaussian factor model under isotropic
observation noise (probabilistic PCA), not an arbitrary dimensionality
reduction — see design decision 1 below.

Design decisions (see the implementation turn's chat response for the
full discussion; summarized here for anyone reading only this file)
----------------------------------------------------------------------
1. PCA-based C defaults ON (latent_dim=16), because a full D x D
   transition matrix (D^2 parameters, ~260,000 for D=512) is wildly
   overparameterized relative to the number of consecutive-window pairs
   this project's dataset realistically provides.
2. C = Identity (latent_dim=None) is still supported for comparison, but
   documented as likely to overfit for realistic D.
3. A, b fit via Ridge, not OLS, for the same overfitting reason.
4. Backward extrapolation (used by impute_missing_window) is attempted
   only if A is safely invertible (checked via condition number, not
   attempted-and-caught) — a heavily regularized A shrinks toward zero,
   which is not invertible, and forward-only imputation with that stated
   is correct behavior, not a bug.
5. Imputation is a simple forward/backward average, not a Kalman
   smoother — this model does not fit noise covariances Q, R, so there
   is nothing to weight the two estimates by. An explicit scope limit of
   a baseline, not an oversight.
6. forecast()/impute_missing_window() are extra methods on this class,
   not additions to the Model ABC — forcing e.g. BaseRateModel to
   implement forecasting would be incoherent. The four-method interface
   is untouched.
7. The transition classifier (predict()) uses only the scalar residual
   norm, exactly mirroring PersistenceModel, so the ONLY thing that
   changes between the two models is whether A is learned or fixed to
   I — keeping that comparison (RESEARCH_BLUEPRINT.md Part I, H1) clean.

Edge cases, handled explicitly rather than silently
-----------------------------------------------------
- No training trajectory with >= 2 windows: cannot fit any dynamics at
  all; falls back entirely to the base rate, matching every other model
  in this package.
- Training data with only one class of consistency_flag: logistic
  regression is undefined; falls back to the base rate.
- A too ill-conditioned to invert: backward extrapolation is silently
  never attempted (forward-only imputation is used instead) — "silently"
  in the sense of not raising, but the reason is checkable via
  model._A_inv being None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction


@register_model("linear_ssm")
class LinearStateSpaceModel(Model):
    def __init__(self, latent_dim: Optional[int] = 16, ridge_alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        latent_dim : int or None
            If given, embeddings are projected onto their top-`latent_dim`
            principal components (fit on training data) before A is
            learned — A is then latent_dim x latent_dim, not D x D. See
            design decision 1. Default 16 is deliberately modest and not
            tuned to any specific cache — evaluation.explore's PCA
            variance-explained analysis is the principled way to choose a
            better value.
            If None, the model operates directly in embedding space (C =
            Identity, d = D) — supported for comparison; ridge_alpha
            becomes more important to tune in that regime.
        ridge_alpha : float
            L2 regularization strength for fitting A, b via
            sklearn.linear_model.Ridge.
        """
        self._latent_dim = latent_dim
        self._ridge_alpha = ridge_alpha
        self._base_rate: Optional[float] = None
        self._pca: Optional[PCA] = None
        self._A: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None
        self._A_inv: Optional[torch.Tensor] = None
        self._logreg: Optional[LogisticRegression] = None

    # ---- internal: latent <-> embedding space ----

    def _to_latent(self, embeddings: torch.Tensor) -> torch.Tensor:
        X = embeddings.float().numpy()
        z = self._pca.transform(X) if self._pca is not None else X
        return torch.from_numpy(z.astype(np.float32))

    def _to_embedding(self, z: torch.Tensor) -> torch.Tensor:
        Z = z.float().numpy()
        e = self._pca.inverse_transform(Z) if self._pca is not None else Z
        return torch.from_numpy(e.astype(np.float32))

    def _apply_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """z_{i+1} = A z_i + b, applied to each row of z: (N, d) -> (N, d)."""
        return z @ self._A.T + self._b

    @staticmethod
    def _safe_invert(A: torch.Tensor, cond_threshold: float = 1e8) -> Optional[torch.Tensor]:
        A_np = A.numpy()
        cond = np.linalg.cond(A_np)
        if not np.isfinite(cond) or cond > cond_threshold:
            return None
        return torch.from_numpy(np.linalg.inv(A_np).astype(np.float32))

    # ---- Model interface ----

    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        if not train:
            self._pca = self._A = self._b = self._A_inv = self._logreg = None
            return

        all_embeddings = torch.cat([t.embeddings for t in train], dim=0).float().numpy()

        if self._latent_dim is not None:
            d = min(self._latent_dim, all_embeddings.shape[1], all_embeddings.shape[0] - 1)
            # svd_solver="full" pinned explicitly: sklearn's default
            # "auto" can select the randomized SVD solver (which consults
            # global numpy random state) once n_components is small
            # relative to the larger data dimension — plausible for
            # latent_dim=16 against D=512. "full" is exact (no accuracy
            # cost) and guarantees this fit is deterministic regardless of
            # data shape, which the E1 experiment's sanity checks
            # (determinism check) require actually holding, not just
            # being likely.
            self._pca = PCA(n_components=d, svd_solver="full")
            self._pca.fit(all_embeddings)
        else:
            self._pca = None

        # Single pass over trajectories building z_current, z_next, AND the
        # matching targets together — see design decision 4: two separate
        # passes were the first draft and were rejected specifically
        # because nothing would have guaranteed they stayed aligned.
        z_current_list, z_next_list, targets_list = [], [], []
        for traj in train:
            if len(traj) < 2:
                continue
            z = self._to_latent(traj.embeddings).numpy()
            z_current_list.append(z[:-1])
            z_next_list.append(z[1:])
            targets_list.extend(traj.consistency_flag[1:].tolist())

        if not z_current_list:
            self._A = self._b = self._A_inv = self._logreg = None
            return

        Z_cur = np.concatenate(z_current_list, axis=0)
        Z_next = np.concatenate(z_next_list, axis=0)
        targets = np.asarray(targets_list)

        ridge = Ridge(alpha=self._ridge_alpha, fit_intercept=True)
        ridge.fit(Z_cur, Z_next)
        # sklearn's Ridge.coef_ is documented as shape (n_targets, n_features)
        # for multi-output regression, but degenerates to 1D (n_features,)
        # whenever n_targets == 1 — i.e. whenever the latent/state dimension
        # d == 1 (confirmed empirically against the installed sklearn 1.7.2).
        # A is always the square (d, d) transition matrix by construction
        # (z_current and z_next both live in the same d-dimensional latent
        # space), so reshape restores that contract uniformly for every d,
        # including d == 1 — this is what _safe_invert/_apply_dynamics and
        # every shape-asserting test in this package already assume.
        d = Z_cur.shape[1]
        self._A = torch.from_numpy(ridge.coef_.reshape(d, d).astype(np.float32))
        self._b = torch.from_numpy(ridge.intercept_.astype(np.float32))
        self._A_inv = self._safe_invert(self._A)

        predicted_next = self._apply_dynamics(torch.from_numpy(Z_cur))
        residuals = (torch.from_numpy(Z_next) - predicted_next).norm(dim=1).numpy()

        if len(residuals) < 2 or len(np.unique(targets)) < 2:
            self._logreg = None
        else:
            self._logreg = LogisticRegression()
            self._logreg.fit(residuals.reshape(-1, 1), targets)

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        if self._base_rate is None:
            raise RuntimeError("LinearStateSpaceModel.predict() called before fit().")

        predictions = []
        for traj in trajectories:
            probs = np.full(len(traj), self._base_rate, dtype=np.float32)
            if len(traj) >= 2 and self._logreg is not None and self._A is not None:
                z = self._to_latent(traj.embeddings)
                predicted_next = self._apply_dynamics(z[:-1])
                residuals = (z[1:] - predicted_next).norm(dim=1).numpy()
                fitted = self._logreg.predict_proba(residuals.reshape(-1, 1))[:, 1]
                probs[1:] = fitted
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
        state = {
            "latent_dim": self._latent_dim,
            "ridge_alpha": self._ridge_alpha,
            "base_rate": self._base_rate,
        }
        if self._pca is not None:
            state["pca_components"] = self._pca.components_.tolist()
            state["pca_mean"] = self._pca.mean_.tolist()
            # sklearn's PCA.transform() reads explained_variance_ (via
            # get_namespace) on every call, even though it's arithmetically
            # unused for whiten=False (the only mode this class uses) — its
            # absence after a manual load() raises AttributeError before any
            # actual computation happens. Empirically confirmed sufficient
            # (transform()/inverse_transform() match the pre-save model
            # exactly) alongside the fields already serialized below.
            state["pca_explained_variance"] = self._pca.explained_variance_.tolist()
        if self._A is not None:
            state["A"] = self._A.tolist()
            state["b"] = self._b.tolist()
        if self._logreg is not None:
            state["logreg_coef"] = self._logreg.coef_.tolist()
            state["logreg_intercept"] = self._logreg.intercept_.tolist()
            state["logreg_classes"] = self._logreg.classes_.tolist()
        (path / "state.json").write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "LinearStateSpaceModel":
        state = json.loads((path / "state.json").read_text())
        model = cls(latent_dim=state["latent_dim"], ridge_alpha=state["ridge_alpha"])
        model._base_rate = state["base_rate"]

        if "pca_components" in state:
            components = np.asarray(state["pca_components"])
            pca = PCA(n_components=components.shape[0])
            pca.components_ = components
            pca.mean_ = np.asarray(state["pca_mean"])
            pca.explained_variance_ = np.asarray(state["pca_explained_variance"])
            pca.n_components_ = components.shape[0]
            pca.n_features_in_ = components.shape[1]
            model._pca = pca

        if "A" in state:
            model._A = torch.tensor(state["A"], dtype=torch.float32)
            model._b = torch.tensor(state["b"], dtype=torch.float32)
            model._A_inv = model._safe_invert(model._A)

        if "logreg_coef" in state:
            logreg = LogisticRegression()
            logreg.coef_ = np.asarray(state["logreg_coef"])
            logreg.intercept_ = np.asarray(state["logreg_intercept"])
            logreg.classes_ = np.asarray(state["logreg_classes"])
            logreg.n_features_in_ = 1
            model._logreg = logreg

        return model

    # ---- extra capabilities beyond the required interface (design decision 6) ----

    def forecast_from_embedding(self, embedding: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Rolls the learned dynamics forward `steps` steps from a single
        embedding, in embedding space. Returns (steps, D). The
        lower-level primitive both forecast() and the evaluation script
        use — kept separate from forecast() so callers who already have a
        bare embedding (as the evaluation script does, when forecasting
        from many different starting points) don't need to construct a
        throwaway Trajectory just to call it.
        """
        if self._A is None:
            raise RuntimeError("forecast_from_embedding() called before fit().")
        z = self._to_latent(embedding.unsqueeze(0))[0]
        latent_predictions = []
        current = z
        for _ in range(steps):
            current = current.unsqueeze(0) @ self._A.T + self._b
            current = current[0]
            latent_predictions.append(current)
        latent_predictions = torch.stack(latent_predictions, dim=0)
        return self._to_embedding(latent_predictions)

    def forecast(self, trajectory: Trajectory, steps: int) -> torch.Tensor:
        """Future prediction: rolls the learned dynamics forward `steps`
        steps from the LAST window of `trajectory`. Returns (steps, D)."""
        if len(trajectory) == 0:
            raise ValueError("Cannot forecast from an empty trajectory.")
        return self.forecast_from_embedding(trajectory.embeddings[-1], steps)

    def impute_missing_window(self, trajectory: Trajectory, missing_index: int) -> torch.Tensor:
        """
        Missing-frame prediction: predicts the embedding at position
        `missing_index` of `trajectory` WITHOUT using
        trajectory.embeddings[missing_index] itself — uses only the
        surrounding observed windows. Returns (D,).

        Forward estimate uses the nearest observed window before the gap;
        backward estimate (only if A is safely invertible — see design
        decision 4) uses the nearest observed window after it; the
        imputed value is their unweighted average (design decision 5).
        """
        if self._A is None:
            raise RuntimeError("impute_missing_window() called before fit().")
        T = len(trajectory)
        if not (0 <= missing_index < T):
            raise IndexError(f"missing_index={missing_index} out of range for a trajectory of length {T}.")
        if T < 2:
            raise ValueError("Cannot impute a trajectory's only window — no neighboring window exists.")

        z_full = self._to_latent(trajectory.embeddings)
        estimates = []

        if missing_index > 0:
            z_prev = z_full[missing_index - 1]
            z_fwd = z_prev.unsqueeze(0) @ self._A.T + self._b
            estimates.append(z_fwd[0])

        if missing_index < T - 1 and self._A_inv is not None:
            z_next = z_full[missing_index + 1]
            z_bwd = (z_next - self._b).unsqueeze(0) @ self._A_inv.T
            estimates.append(z_bwd[0])

        if not estimates:
            raise RuntimeError(
                "Could not impute: missing_index is the last window in the trajectory "
                "and A is not safely invertible for backward extrapolation, so no "
                "estimate (forward or backward) is available."
            )

        z_imputed = torch.stack(estimates, dim=0).mean(dim=0)
        return self._to_embedding(z_imputed.unsqueeze(0))[0]
