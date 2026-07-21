"""
Shared evaluation framework: every future experiment (E1's model
comparison, E2's regime model, E3's constrained variants, and beyond)
reuses everything in this package. The only thing a model author writes
is a subclass of Model implementing fit()/predict()/save()/load() — see
model.py and examples/persistence.py.
"""

from .config import ExperimentConfig
from .metrics import AggregatedMetric, aggregate_across_seeds, compute_metrics
from .model import Model, get_model_class, register_model
from .runner import run_experiment
from .seeding import set_all_seeds
from .trajectory import Trajectory, TrajectoryPrediction, group_into_trajectories

__all__ = [
    "ExperimentConfig",
    "AggregatedMetric",
    "aggregate_across_seeds",
    "compute_metrics",
    "Model",
    "get_model_class",
    "register_model",
    "run_experiment",
    "set_all_seeds",
    "Trajectory",
    "TrajectoryPrediction",
    "group_into_trajectories",
]
