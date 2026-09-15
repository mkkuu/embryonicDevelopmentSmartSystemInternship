"""
Runs the cheapest possible E1 comparison: BaseRateModel vs
IdentityDynamicsModel vs PersistenceModel vs LinearStateSpaceModel, on a
cached embedding split, across multiple seeds, with full metrics, plots,
and a report — via the shared evaluation framework (see
RESEARCH_BLUEPRINT.md Part IV, experiment E1; this script does not yet
include the existing classifier itself, which is a separate, later
task).

The four models form an ablation ladder — see
evaluation/models/README.md — and the comparisons that actually matter
scientifically are IdentityDynamics vs BaseRate (does embedding content
help at all), Persistence vs IdentityDynamics (does embedding change over
time help beyond content alone), and LinearSSM vs Persistence (does
LEARNING the transition, rather than fixing it to the identity, help
further). All three come out of this one run, in the same comparison
plots and report. LinearStateSpaceModel's forecast/imputation
capabilities are evaluated separately — see
evaluate_linear_ssm_dynamics.py — since those are a regression, not
classification, evaluation problem.

Usage
-----
    cd Training
    python -m experiments.e1_ladder.run_e1_baselines \\
        --cache_root ../Embeddings --embedding_model_name resnet18 --seeds 0 1 2
"""

from __future__ import annotations

import argparse

from evaluation import ExperimentConfig, run_experiment

# Imported for their @register_model side effect — registration happens
# on import, see evaluation.model.get_model_class's error message for
# exactly this gotcha.
import evaluation.models.base_rate  # noqa: F401
import evaluation.models.identity_dynamics  # noqa: F401
import evaluation.models.linear_ssm  # noqa: F401
import evaluation.models.persistence  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output_dir", default="../Results/evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        experiment_name="e1_baseline_comparison",
        model_kwargs={
            "base_rate": {},
            "identity_dynamics": {},
            "persistence": {},
            "linear_ssm": {"latent_dim": 16, "ridge_alpha": 1.0},
        },
        cache_root=args.cache_root,
        embedding_model_name=args.embedding_model_name,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    report_path = run_experiment(
        config, model_names=["base_rate", "identity_dynamics", "persistence", "linear_ssm"]
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
