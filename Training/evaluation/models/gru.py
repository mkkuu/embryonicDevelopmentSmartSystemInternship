"""
GRUDynamicsModel — the first model in this package that learns a NON-LINEAR
transition, rather than assuming a linear one (LinearStateSpaceModel) or no
dynamics at all (IdentityDynamicsModel/PersistenceModel).

Scientific motivation
----------------------
E1 showed LinearStateSpaceModel losing to IdentityDynamicsModel on
classification (AUROC) and losing to a naive "unchanged" baseline on
forecast/imputation (6/6 measures). That result falsifies "a LINEAR
transition suffices" specifically — it does not by itself falsify "temporal
dynamics carry no information." This model changes exactly one variable
relative to LinearStateSpaceModel: the transition function is a learned,
non-linear, gated recurrence (Cho et al. 2014) instead of a fixed linear map
`A`. Everything else that can be kept comparable to the existing ladder is
kept comparable — see the two `predict_mode` values below and the migration
study this file implements (conversation record, not reproduced here).

Two independent, separately-trained/evaluated formulations
-------------------------------------------------------------
`predict_mode="residual"` (default) — mirrors PersistenceModel/
LinearStateSpaceModel's own mechanism exactly: the GRU is trained to
forecast z_{i+1} from the history z_1..z_i (single-step-ahead MSE); after
training, a separate scalar logistic regression is fit on the residual norm
||z_i - forecast(h_{i-1})|| to produce P(consistency_flag_i=1) — the ONLY
thing that changes relative to LinearStateSpaceModel's own residual-based
classifier is whether the transition is learned+non-linear (this) or fixed
to a linear `A` (LinearStateSpaceModel). This is the mechanistically
cleanest test of "is non-linearity the missing ingredient."

`predict_mode="direct"` — classifies consistency_flag_i directly from h_i
(built from the FULL history z_1..z_i, not just one step back). A real,
separately meaningful question ("does more historical context, encoded
non-linearly, help discriminate transitions") but NOT a clean isolation of
non-linearity alone, since no linear full-history model exists in E1 to
compare against. A win here must be interpreted with that caveat — see the
migration study's Section 8 (comparability pitfalls) for the full argument.
Never silently substituted for "residual" — the caller must opt in.

These two modes are trained as two separate GRUDynamicsModel instances,
never jointly/multi-task, so that a result can be unambiguously attributed
to one formulation or the other (see the migration study's Section 1).

Causality (non-negotiable, tested explicitly in Tests/)
-----------------------------------------------------------
The underlying `torch.nn.GRU` is constructed with `bidirectional=False`
(PyTorch's own default, pinned here explicitly rather than relied upon
implicitly) — h_t is, by construction of a unidirectional recurrence, a
function of z_1..z_t ONLY. No pooling over the whole trajectory is used
anywhere in this module, for either predict_mode. The forecast head reads
only h_{t-1} (never z_t or later) to produce a prediction for z_t.

Data diet parity with the existing ladder
---------------------------------------------
This model receives EXACTLY the same information the four E1 models
receive: the ordered sequence of surviving windows' embeddings, nothing
else. In particular, `window_start` gaps (windows dropped by DataSet.py's
phase-consistency filter) are NOT exposed to this model as a Δt feature —
none of the four E1 baselines use window_start/Δt in any way, and adding it
here would give this model information no baseline had access to, breaking
the comparison this experiment exists to make. See the migration study's
Section 2 for the full argument.

Normalization
-------------
A `StandardScaler` (per-dimension mean/std) is fit on the pooled TRAIN
embeddings only, exactly mirroring IdentityDynamicsModel's own justified
precedent, and applied (never re-fit) to val/test at prediction time.

Serialization — a documented, deliberate exception to this package's
"never pickle" convention
------------------------------------------------------------------------
Every other model in this package (BaseRateModel, IdentityDynamicsModel,
PersistenceModel, LinearStateSpaceModel) serializes to plain JSON, never
pickle, specifically to stay immune to sklearn/torch version-pickle
incompatibility. A trained neural network's weights cannot reasonably be
serialized as JSON, so this model's `save()` additionally writes a
`net_state_dict.pt` via `torch.save()` (which uses pickle internally for
tensors — unavoidable, not an oversight) alongside a `metadata.json`
carrying every other piece of state (hyperparameters, scaler, the residual
logistic regression if `predict_mode="residual"`, per-epoch train history)
in the same transparent JSON style as every other model here.

Training details worth knowing when reading fit()
------------------------------------------------------
- Trajectories are processed ONE AT A TIME (batch_size=1) — no padding, no
  masking, matching the measured T distribution (train median T≈431,
  min T=83 — see the migration study) where BPTT over a whole trajectory is
  computationally fine, and matching the same per-trajectory-loop pattern
  every other model in this package already uses.
- Gradient clipping (`grad_clip_norm`, default 1.0) is applied every step —
  a real stability concern given BPTT over several hundred steps, absent
  from every closed-form model in this package by construction (they have
  no BPTT at all).
- VAL is used for real here (early stopping) — unlike the four closed-form
  models, which never reference their `val` argument at all.
- Unlike every other model in this package, THIS model is not guaranteed
  deterministic given fixed data — `fit()` calls
  `evaluation.seeding.set_all_seeds(seed)` at its very start so a given
  seed is at least reproducible, but a determinism check for this specific
  model must still be run explicitly (Tests/) rather than assumed inherited
  from the other four models' already-verified determinism.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..model import Model, register_model
from ..seeding import set_all_seeds
from ..trajectory import Trajectory, TrajectoryPrediction


class _GRUForecastNet(nn.Module):
    """The learned core: one (by default) unidirectional GRU layer, a linear
    forecast decoder, and an optional direct-classification head. Kept as a
    plain nn.Module, separate from the Model-interface wrapper below, so the
    two concerns (network architecture vs. this package's fit/predict/save/
    load contract) don't get tangled."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        direct_classification: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # causality requirement — see module docstring; never override
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Linear(hidden_dim, input_dim)
        self.classification_head = nn.Linear(hidden_dim, 1) if direct_classification else None

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """z: (1, T, input_dim) — one trajectory, batch_size=1 (see module docstring).

        Returns (h, forecast, logits):
          h: (1, T, hidden_dim) — h[:, t] depends only on z[:, :t+1] (causal).
          forecast: (1, T-1, input_dim) or None if T<2 — forecast[:, t] is this
            network's prediction of z[:, t+1], produced from h[:, t] alone.
          logits: (1, T, 1) or None (only populated when direct_classification
            was requested at construction) — logits[:, t] classifies window t
            from h[:, t] (which includes z[:, t] itself — not a future leak,
            see module docstring's causality section).
        """
        h, _ = self.gru(z)
        h = self.dropout(h)
        forecast = self.forecast_head(h[:, :-1, :]) if h.shape[1] >= 2 else None
        logits = self.classification_head(h) if self.classification_head is not None else None
        return h, forecast, logits


@register_model("gru")
class GRUDynamicsModel(Model):
    def __init__(
        self,
        predict_mode: str = "residual",
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 8,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        predict_mode : "residual" (default) or "direct" — see module docstring.
            Not silently interchangeable; must be chosen explicitly.
        hidden_dim : GRU hidden size. Deliberately small (32 default) relative
            to typical RNN practice, given only 492 independent training
            trajectories — see the migration study's parameter-count/
            overfitting analysis. 64 is the one alternative endorsed for a
            first experiment; larger values are out of scope for now.
        num_layers : 1 by default — deliberately minimal for a first
            experiment (fewer parameters, easier to debug/audit).
        dropout, weight_decay : regularization, given the overfitting risk
            above. Starting points, to be validated on VAL, never on TEST.
        learning_rate, max_epochs, patience : training loop hyperparameters.
            Starting points (see the migration study), not tuned values.
        grad_clip_norm : gradient-norm clipping threshold, applied every
            optimizer step — a real necessity given BPTT over trajectories
            with median length ≈431 (measured this session), not present in
            any other model in this package (none of them use BPTT).
        seed : passed to evaluation.seeding.set_all_seeds() at the start of
            fit() for reproducibility. NOT a substitute for the multi-seed
            determinism verification this model still needs (see module
            docstring) — a single seed only makes ONE run reproducible.
        device : "cpu" by default, deliberately — this class never touches a
            GPU unless a caller explicitly passes device="cuda" or similar.
        """
        if predict_mode not in ("residual", "direct"):
            raise ValueError(f"predict_mode must be 'residual' or 'direct', got {predict_mode!r}.")
        self._predict_mode = predict_mode
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._patience = patience
        self._grad_clip_norm = grad_clip_norm
        self._seed = seed
        self._device = torch.device(device)

        self._input_dim = 512
        self._base_rate: Optional[float] = None
        self._scaler: Optional[StandardScaler] = None
        self._net: Optional[_GRUForecastNet] = None
        self._residual_logreg: Optional[LogisticRegression] = None  # "residual" mode only
        self._train_history: List[dict] = []  # per-epoch {epoch, train_loss, val_loss}, for auditability

    # ---- internal ----

    def _standardize(self, embeddings: torch.Tensor) -> torch.Tensor:
        z_np = self._scaler.transform(embeddings.float().numpy())
        return torch.from_numpy(z_np.astype(np.float32)).unsqueeze(0).to(self._device)  # (1, T, D)

    def _compute_loss(self, traj: Trajectory) -> Optional[torch.Tensor]:
        z = self._standardize(traj.embeddings)
        _, forecast, logits = self._net(z)
        if self._predict_mode == "residual":
            if forecast is None:
                return None
            target = z[:, 1:, :]
            return nn.functional.mse_loss(forecast, target)
        else:  # "direct"
            if logits is None:
                return None
            flags = traj.consistency_flag.float().to(self._device).view(1, -1, 1)
            return nn.functional.binary_cross_entropy_with_logits(logits, flags)

    def _residuals(self, traj: Trajectory) -> Optional[np.ndarray]:
        """r_i = ||z_i - forecast(h_{i-1})|| for i=2..T (0-indexed: array
        positions 1..T-1) — exactly PersistenceModel/LinearStateSpaceModel's
        own d_i convention, so the downstream residual→logistic-regression
        step is a drop-in replacement of theirs, nothing more."""
        if len(traj) < 2 or self._net is None:
            return None
        z = self._standardize(traj.embeddings)
        with torch.no_grad():
            _, forecast, _ = self._net(z)
        if forecast is None:
            return None
        residual = (z[:, 1:, :] - forecast).norm(dim=-1).squeeze(0).cpu().numpy()
        return residual

    def _fit_residual_logreg(self, trajectories: List[Trajectory]) -> None:
        distances: List[float] = []
        targets: List[int] = []
        for traj in trajectories:
            d = self._residuals(traj)
            if d is None:
                continue
            distances.extend(d.tolist())
            targets.extend(traj.consistency_flag[1:].tolist())

        distances_arr = np.asarray(distances, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.int64)
        if len(distances_arr) < 2 or len(np.unique(targets_arr)) < 2:
            self._residual_logreg = None
            return
        self._residual_logreg = LogisticRegression()
        self._residual_logreg.fit(distances_arr.reshape(-1, 1), targets_arr)

    # ---- Model interface ----

    def fit(
        self,
        train: List[Trajectory],
        val: Optional[List[Trajectory]] = None,
        checkpoint_dir: Optional[Path] = None,
        resume: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        checkpoint_dir : if given, a full training-state snapshot (net,
            optimizer, RNG states, best-so-far state, train_history) is
            written to `checkpoint_dir / "train_state.pt"` after EVERY
            epoch, atomically (write to a `.tmp` file, then `Path.replace`).
            This is a resume-for-robustness mechanism, not a bit-exact-
            reproducibility one: it lets a killed/crashed run continue from
            its last completed epoch without losing progress, but a resumed
            run is not guaranteed to produce bit-identical results to an
            uninterrupted one (the RNG stream is continued correctly, but a
            process restart still changes e.g. CUDA kernel scheduling).
        resume : if True and `checkpoint_dir / "train_state.pt"` exists,
            resume training from it instead of starting fresh. Raises if
            `resume=True` but no checkpoint exists, or if the checkpoint's
            recorded hyperparameters don't match this instance's — silent
            resume-into-a-different-config would corrupt the run.
        verbose : print one line per epoch (train/val loss, seconds) when
            True. Off by default so this stays quiet in tests/library use.
        """
        all_flags = torch.cat([t.consistency_flag for t in train]) if train else torch.tensor([])
        self._base_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

        if not train:
            self._scaler = None
            self._net = None
            self._residual_logreg = None
            return

        X = torch.cat([t.embeddings for t in train], dim=0).float().numpy()

        # "residual" mode needs >=2 windows per trajectory (forecast is undefined
        # otherwise); "direct" mode has no such requirement (h_t is defined even for
        # T=1). Filtering is a documented no-op on real data (measured T_min=83 this
        # session) but must still be correct for synthetic edge cases (Tests/).
        if self._predict_mode == "residual":
            usable_train = [t for t in train if len(t) >= 2]
            usable_val = [t for t in (val or []) if len(t) >= 2]
        else:
            usable_train = list(train)
            usable_val = list(val or [])

        ckpt_path = checkpoint_dir / "train_state.pt" if checkpoint_dir is not None else None
        loaded = None
        if ckpt_path is not None and ckpt_path.exists():
            # weights_only=False: this checkpoint carries optimizer/RNG state (numpy
            # arrays, Python tuples), not just tensors — PyTorch >=2.6 defaults to
            # weights_only=True and would reject those. Safe here: this file is only
            # ever written by this same fit() method, never from an external source.
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if resume and loaded is None:
            raise FileNotFoundError(
                f"resume=True but no checkpoint found at {ckpt_path} — nothing to resume from."
            )

        if resume and loaded is not None:
            expected = {
                "hidden_dim": self._hidden_dim,
                "num_layers": self._num_layers,
                "predict_mode": self._predict_mode,
                "seed": self._seed,
            }
            if loaded["hyperparams"] != expected:
                raise ValueError(
                    f"Resume checkpoint hyperparameters {loaded['hyperparams']} do not match "
                    f"this instance's {expected} — refusing to resume into a mismatched config."
                )
            self._scaler = StandardScaler()
            self._scaler.mean_ = loaded["scaler_mean"]
            self._scaler.scale_ = loaded["scaler_scale"]
            self._scaler.var_ = self._scaler.scale_**2
            self._scaler.n_features_in_ = len(self._scaler.mean_)

            self._net = _GRUForecastNet(
                input_dim=self._input_dim,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
                direct_classification=(self._predict_mode == "direct"),
            ).to(self._device)
            self._net.load_state_dict(loaded["net_state_dict"])

            optimizer = torch.optim.Adam(
                self._net.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
            )
            optimizer.load_state_dict(loaded["optimizer_state_dict"])

            rng = np.random.RandomState()
            rng.set_state(loaded["numpy_rng_state"])
            random.setstate(loaded["python_rng_state"])
            torch.set_rng_state(loaded["torch_rng_state"])
            if torch.cuda.is_available() and loaded.get("torch_cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(loaded["torch_cuda_rng_state"])

            start_epoch = loaded["next_epoch"]
            best_val_loss = loaded["best_val_loss"]
            best_state = loaded["best_state"]
            epochs_without_improvement = loaded["epochs_without_improvement"]
            self._train_history = loaded["train_history"]
            if verbose:
                print(
                    f"[resume] continuing from epoch {start_epoch} "
                    f"(best_val_loss so far={best_val_loss:.6f})",
                    flush=True,
                )
        else:
            set_all_seeds(self._seed)
            self._scaler = StandardScaler()
            self._scaler.fit(X)

            self._net = _GRUForecastNet(
                input_dim=self._input_dim,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
                direct_classification=(self._predict_mode == "direct"),
            ).to(self._device)

            optimizer = torch.optim.Adam(
                self._net.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
            )
            rng = np.random.RandomState(self._seed)
            start_epoch = 0
            best_val_loss = float("inf")
            best_state = None
            epochs_without_improvement = 0
            self._train_history = []

        for epoch in range(start_epoch, self._max_epochs):
            epoch_t0 = time.time()
            self._net.train()
            train_losses: List[float] = []
            for idx in rng.permutation(len(usable_train)):
                traj = usable_train[idx]
                loss = self._compute_loss(traj)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._grad_clip_norm)
                optimizer.step()
                train_losses.append(loss.item())

            self._net.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for traj in usable_val:
                    loss = self._compute_loss(traj)
                    if loss is not None:
                        val_losses.append(loss.item())

            mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            # No val trajectories usable (e.g. a degenerate synthetic test) falls back
            # to the train loss for the early-stopping criterion, rather than crashing
            # or silently disabling early stopping — logged as such via train_history,
            # never silently treated as "converged".
            mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss
            epoch_seconds = time.time() - epoch_t0
            self._train_history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "val_loss": mean_val_loss,
                    "epoch_seconds": epoch_seconds,
                }
            )
            if verbose:
                print(
                    f"[epoch {epoch}] train_loss={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} ({epoch_seconds:.1f}s)",
                    flush=True,
                )

            improved = mean_val_loss < best_val_loss
            if improved:
                best_val_loss = mean_val_loss
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if ckpt_path is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = checkpoint_dir / "train_state.pt.tmp"
                torch.save(
                    {
                        "next_epoch": epoch + 1,
                        "best_val_loss": best_val_loss,
                        "best_state": best_state,
                        "epochs_without_improvement": epochs_without_improvement,
                        "train_history": self._train_history,
                        "net_state_dict": self._net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "numpy_rng_state": rng.get_state(),
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "torch_cuda_rng_state": (
                            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                        ),
                        "scaler_mean": self._scaler.mean_,
                        "scaler_scale": self._scaler.scale_,
                        "hyperparams": {
                            "hidden_dim": self._hidden_dim,
                            "num_layers": self._num_layers,
                            "predict_mode": self._predict_mode,
                            "seed": self._seed,
                        },
                    },
                    tmp_path,
                )
                tmp_path.replace(ckpt_path)

            if epochs_without_improvement >= self._patience:
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

        if self._predict_mode == "residual":
            self._fit_residual_logreg(usable_train)

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        if self._base_rate is None:
            raise RuntimeError("GRUDynamicsModel.predict() called before fit().")

        if self._net is not None:
            self._net.eval()

        predictions = []
        for traj in trajectories:
            probs = np.full(len(traj), self._base_rate, dtype=np.float32)
            if self._net is not None:
                if self._predict_mode == "residual":
                    if len(traj) >= 2 and self._residual_logreg is not None:
                        d = self._residuals(traj)
                        if d is not None:
                            fitted = self._residual_logreg.predict_proba(d.reshape(-1, 1))[:, 1]
                            probs[1:] = fitted
                else:  # "direct"
                    z = self._standardize(traj.embeddings)
                    with torch.no_grad():
                        _, _, logits = self._net(z)
                    if logits is not None:
                        probs = (
                            torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)
                        )
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
            "predict_mode": self._predict_mode,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "weight_decay": self._weight_decay,
            "learning_rate": self._learning_rate,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "grad_clip_norm": self._grad_clip_norm,
            "seed": self._seed,
            "input_dim": self._input_dim,
            "base_rate": self._base_rate,
            "train_history": self._train_history,
        }
        if self._scaler is not None:
            state["scaler_mean"] = self._scaler.mean_.tolist()
            state["scaler_scale"] = self._scaler.scale_.tolist()
        if self._residual_logreg is not None:
            state["residual_logreg_coef"] = self._residual_logreg.coef_.tolist()
            state["residual_logreg_intercept"] = self._residual_logreg.intercept_.tolist()
            state["residual_logreg_classes"] = self._residual_logreg.classes_.tolist()
        (path / "metadata.json").write_text(json.dumps(state, indent=2))

        # Documented exception to this package's JSON-only convention — see module
        # docstring's "Serialization" section.
        if self._net is not None:
            torch.save(self._net.state_dict(), path / "net_state_dict.pt")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "GRUDynamicsModel":
        """
        device : where to place the reloaded net for inference (predict()).
            Defaults to "cpu" (unchanged behavior for every existing
            caller). Pass "cuda" (or "cuda:N") when about to run many
            predict() calls (e.g. bootstrap evaluation over hundreds of
            resamples) — CPU inference for this model is a per-trajectory
            Python loop over a GRU forward pass and does not vectorize
            across resamples, so it is dramatically slower than GPU there.
            This changes only where computation happens, not what is
            computed — the same trained weights, same forward pass — so
            it does not count as a hyperparameter change under this
            project's no-tuning-on-test discipline.
        """
        state = json.loads((path / "metadata.json").read_text())
        model = cls(
            predict_mode=state["predict_mode"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            weight_decay=state["weight_decay"],
            learning_rate=state["learning_rate"],
            max_epochs=state["max_epochs"],
            patience=state["patience"],
            grad_clip_norm=state["grad_clip_norm"],
            seed=state["seed"],
            device=device,
        )
        model._input_dim = state["input_dim"]
        model._base_rate = state["base_rate"]
        model._train_history = state.get("train_history", [])

        if "scaler_mean" in state:
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(state["scaler_mean"])
            scaler.scale_ = np.asarray(state["scaler_scale"])
            scaler.var_ = scaler.scale_**2
            scaler.n_features_in_ = len(scaler.mean_)
            model._scaler = scaler

        net_path = path / "net_state_dict.pt"
        if net_path.exists():
            net = _GRUForecastNet(
                input_dim=model._input_dim,
                hidden_dim=model._hidden_dim,
                num_layers=model._num_layers,
                dropout=model._dropout,
                direct_classification=(model._predict_mode == "direct"),
            ).to(model._device)
            net.load_state_dict(torch.load(net_path, map_location=model._device))
            net.eval()
            model._net = net

        if "residual_logreg_coef" in state:
            logreg = LogisticRegression()
            logreg.coef_ = np.asarray(state["residual_logreg_coef"])
            logreg.intercept_ = np.asarray(state["residual_logreg_intercept"])
            logreg.classes_ = np.asarray(state["residual_logreg_classes"])
            logreg.n_features_in_ = 1
            model._residual_logreg = logreg

        return model

    # ---- extra capability beyond the required interface, mirroring
    # LinearStateSpaceModel's own design (design decision 6 there) ----

    def forecast_from_embedding(self, embedding: torch.Tensor, steps: int) -> torch.Tensor:
        """Autoregressive multi-step forecast: rolls the GRU's own hidden
        state forward `steps` steps, feeding each step's own forecast back in
        as the next input. Causal by construction — only the single starting
        embedding and the model's own prior predictions are ever used, never
        any ground-truth future embedding. Returns (steps, 512) in the
        original (non-standardized) embedding space.

        Only meaningful for predict_mode="residual" instances — a "direct"
        instance never trains its forecast head at all (see module
        docstring), so calling this on one would silently return an
        untrained, meaningless prediction; raises explicitly instead."""
        if self._net is None or self._scaler is None:
            raise RuntimeError("forecast_from_embedding() called before fit().")
        if self._predict_mode != "residual":
            raise RuntimeError(
                "forecast_from_embedding() is only meaningful for predict_mode="
                "'residual' instances — this instance's forecast head was never "
                f"trained (predict_mode={self._predict_mode!r} trains only the "
                "direct classification head)."
            )
        self._net.eval()
        z0 = self._scaler.transform(embedding.unsqueeze(0).float().numpy())
        current = torch.from_numpy(z0.astype(np.float32)).unsqueeze(0).to(self._device)  # (1,1,D)

        outputs = []
        hidden = None
        with torch.no_grad():
            for _ in range(steps):
                h, hidden = self._net.gru(current, hidden)
                forecast = self._net.forecast_head(h)  # (1,1,D) — prediction for the NEXT step
                outputs.append(forecast.squeeze(0).squeeze(0))
                current = forecast

        stacked = torch.stack(outputs, dim=0).cpu().numpy()
        stacked = self._scaler.inverse_transform(stacked)
        return torch.from_numpy(stacked.astype(np.float32))
