"""
run_experiment(): the single entry point every future experiment (E1, E2,
E3, and beyond) should call. Loads cached embeddings, groups them into
trajectories, fits and evaluates every configured model across every
configured seed, computes metrics, aggregates with confidence intervals,
generates plots and a markdown report — all shared, none of it duplicated
per-model.

Usage
-----
    from evaluation import ExperimentConfig, run_experiment
    import evaluation.models.persistence  # import to trigger @register_model

    config = ExperimentConfig(
        experiment_name="e1_baseline_comparison",
        model_kwargs={"persistence": {}},
        embedding_model_name="resnet18",
        seeds=[0, 1, 2],
    )
    run_experiment(config, model_names=["persistence"])

See evaluation/run_e1_baselines.py for a complete, runnable comparison of
multiple registered models at once.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from embeddings.dataset import EmbeddingDataset

from .config import ExperimentConfig
from .logging_utils import setup_logging
from .metrics import aggregate_across_seeds, compute_metrics
from .model import get_model_class
from .plots import plot_metric_comparison, plot_seed_distribution
from .report import generate_report
from .seeding import set_all_seeds
from .trajectory import Trajectory, group_into_trajectories


def _load_split_trajectories(config: ExperimentConfig, split: str) -> List[Trajectory]:
    # Not passing expected_manifest to EmbeddingDataset here: ExperimentConfig
    # intentionally doesn't duplicate every embeddings.CacheManifest field
    # (window_size, stride, focal_type, ...) to avoid two sources of truth
    # for the same configuration — see config.py's docstring. If stricter
    # validation is needed for a specific experiment, construct the
    # expected CacheManifest from embeddings.cache and pass it explicitly
    # at the call site; EmbeddingDataset already supports this, unchanged.
    cache_root = Path(config.cache_root) / config.embedding_model_name
    dataset = EmbeddingDataset(cache_root, split)
    return group_into_trajectories(dataset)


def run_experiment(config: ExperimentConfig, model_names: List[str]) -> Path:
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "run.log")

    config.to_json(output_dir / "config.json")
    logger.info(f"Starting experiment '{config.experiment_name}' — config saved to {output_dir / 'config.json'}")

    train_trajectories = _load_split_trajectories(config, "train")
    val_trajectories = _load_split_trajectories(config, "val")
    test_trajectories = _load_split_trajectories(config, "test")
    logger.info(
        f"Loaded {len(train_trajectories)}/{len(val_trajectories)}/{len(test_trajectories)} "
        f"train/val/test trajectories."
    )

    per_seed_results: Dict[str, Dict[int, Dict[str, float]]] = {name: {} for name in model_names}

    for model_name in model_names:
        model_cls = get_model_class(model_name)
        logger.info(f"--- Model: {model_name} ---")
        for seed in config.seeds:
            set_all_seeds(seed)
            model = model_cls(**config.model_kwargs.get(model_name, {}))
            model.fit(train_trajectories, val_trajectories)

            predictions = model.predict(test_trajectories)
            metrics = compute_metrics(test_trajectories, predictions)
            per_seed_results[model_name][seed] = metrics
            numeric_metrics = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            logger.info(f"[{model_name}][seed={seed}] {numeric_metrics}")

            checkpoint_dir = output_dir / model_name / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint_dir)

    aggregated_by_model = {
        name: aggregate_across_seeds(per_seed_results[name], config.confidence_level) for name in model_names
    }

    (output_dir / "per_seed_results.json").write_text(json.dumps(per_seed_results, indent=2))
    (output_dir / "aggregated_results.json").write_text(
        json.dumps(
            {model: {k: asdict(v) for k, v in metrics.items()} for model, metrics in aggregated_by_model.items()},
            indent=2,
        )
    )

    plot_paths: Dict[str, Path] = {}
    if aggregated_by_model:
        common_metrics = set.intersection(*[set(m.keys()) for m in aggregated_by_model.values()])
        for metric_name in sorted(common_metrics):
            path = output_dir / f"comparison_{metric_name}.png"
            plot_metric_comparison(aggregated_by_model, metric_name, path, config.confidence_level)
            plot_paths[f"comparison_{metric_name}"] = path

        for model_name in model_names:
            for metric_name in aggregated_by_model[model_name]:
                path = output_dir / f"{model_name}_{metric_name}_by_seed.png"
                plot_seed_distribution(per_seed_results[model_name], metric_name, path)
                plot_paths[f"{model_name}_{metric_name}_by_seed"] = path

    report_path = output_dir / "REPORT.md"
    generate_report(config, aggregated_by_model, plot_paths, report_path)
    logger.info(f"Experiment complete. Report at {report_path}")

    return report_path
