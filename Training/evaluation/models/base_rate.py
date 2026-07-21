"""
BaseRateModel — predicts P(consistency_flag=1) as the fixed empirical
transition rate observed in the training data, for every window,
completely ignoring the embedding. The "population base-rate baseline"
from RESEARCH_BLUEPRINT.md's baseline table: how much do embeddings add
over knowing nothing but the population's overall transition frequency?

(This class previously lived in the now-retired examples/persistence.py
under the name PersistenceModel, before "Persistence" was given its
precise, distinct scientific meaning — see models/persistence.py. Renamed
here to stop the two being confused; they test different hypotheses.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch

from ..model import Model, register_model
from ..trajectory import Trajectory, TrajectoryPrediction


@register_model("base_rate")
class BaseRateModel(Model):
    def __init__(self) -> None:
        self._transition_rate: Optional[float] = None

    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        if not train:
            self._transition_rate = 0.5
            return
        all_flags = torch.cat([t.consistency_flag for t in train])
        self._transition_rate = float(all_flags.float().mean()) if len(all_flags) else 0.5

    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        if self._transition_rate is None:
            raise RuntimeError("BaseRateModel.predict() called before fit().")
        predictions = []
        for traj in trajectories:
            prob = torch.full((len(traj),), self._transition_rate, dtype=torch.float32)
            predictions.append(
                TrajectoryPrediction(
                    video_name=traj.video_name,
                    window_starts=traj.window_starts,
                    consistency_flag_prob=prob,
                )
            )
        return predictions

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.json").write_text(json.dumps({"transition_rate": self._transition_rate}))

    @classmethod
    def load(cls, path: Path) -> "BaseRateModel":
        state = json.loads((path / "state.json").read_text())
        model = cls()
        model._transition_rate = state["transition_rate"]
        return model
