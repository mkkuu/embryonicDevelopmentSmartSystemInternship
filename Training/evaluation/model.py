"""
The one interface every model in this framework must implement. Every
future experiment subclasses Model and implements exactly these four
methods — everything else (metrics, plotting, seeding, confidence
intervals, logging, reporting) is shared infrastructure elsewhere in this
package and must never be reimplemented per-model.

Also holds a small name -> class registry, mirroring the
Strategy+Registry pattern already committed to in RESEARCH_BLUEPRINT.md
for dynamics-model selection: new models register themselves;
ExperimentConfig selects a model by name string; nothing that
orchestrates experiments needs editing when a new model is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from .trajectory import Trajectory, TrajectoryPrediction


class Model(ABC):
    @abstractmethod
    def fit(self, train: List[Trajectory], val: Optional[List[Trajectory]] = None) -> None:
        """
        Fit the model on training trajectories. `val` is provided for
        models that use it (e.g. early stopping); implementations that
        don't need it simply ignore the argument. A model that needs no
        fitting at all (Persistence) implements this as a no-op — that is
        a legitimate, expected implementation, not a workaround.
        """

    @abstractmethod
    def predict(self, trajectories: List[Trajectory]) -> List[TrajectoryPrediction]:
        """Return one TrajectoryPrediction per input Trajectory, in the
        SAME ORDER as the input (not matched by video_name — see below),
        each aligned window-for-window with its input.

        Order-preservation, not video_name matching, is required
        specifically because patient-level bootstrap resampling (used for
        confidence intervals — see metrics.bootstrap_trajectory_metrics)
        samples trajectories WITH replacement, so the same video_name can
        legitimately appear more than once in one call's input list.
        metrics.compute_metrics() verifies length and per-position
        window_starts alignment, not video_name uniqueness — implementing
        predict() as `[self._predict_one(t) for t in trajectories]` (or
        equivalent) satisfies this automatically; do not build a
        video_name-keyed dict internally, for the same reason."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist whatever state is needed to reconstruct this model via
        load(path). For a stateless model, this may write only a small
        marker/summary file — still required, so the harness can treat
        every model uniformly regardless of whether it has learned
        parameters."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model":
        """Reconstruct a model previously written by save(path). Must
        return an instance ready for predict() without calling fit()
        again."""


_MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    """Class decorator: @register_model("persistence") above a Model
    subclass makes it selectable by that name in ExperimentConfig."""

    def decorator(cls: Type[Model]) -> Type[Model]:
        if name in _MODEL_REGISTRY and _MODEL_REGISTRY[name] is not cls:
            raise ValueError(
                f"Model name '{name}' is already registered to "
                f"{_MODEL_REGISTRY[name].__name__}; refusing to silently "
                f"overwrite it with {cls.__name__}. Pick a distinct name."
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[Model]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"No model registered under '{name}'. Registered: "
            f"{list(_MODEL_REGISTRY)}. Did you forget to import the module "
            f"that defines and @register_model('{name}')s it? Registration "
            f"only happens when that module is imported."
        )
    return _MODEL_REGISTRY[name]
