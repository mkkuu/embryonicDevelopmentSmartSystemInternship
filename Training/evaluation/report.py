"""
Markdown report generation — the human-readable deliverable tying
configuration, per-seed results, aggregated statistics, and plots into
one document, in the same spirit as BASELINE.md/DATASET_STATS.md
established earlier in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import ExperimentConfig
from .metrics import AggregatedMetric


def generate_report(
    config: ExperimentConfig,
    aggregated_by_model: Dict[str, Dict[str, AggregatedMetric]],
    plot_paths: Dict[str, Path],
    output_path: Path,
) -> None:
    lines = [
        f"# Experiment Report: {config.experiment_name}",
        "",
        f"- Embedding model: `{config.embedding_model_name}`",
        f"- Cache root: `{config.cache_root}`",
        f"- Seeds: {config.seeds} (n={len(config.seeds)})",
        f"- Confidence level: {config.confidence_level:.0%}",
        "",
        "## Results",
        "",
    ]

    metric_names = sorted({m for metrics in aggregated_by_model.values() for m in metrics})
    if metric_names:
        header = "| model | " + " | ".join(metric_names) + " |"
        separator = "|---" * (len(metric_names) + 1) + "|"
        lines += [header, separator]
        for model_name, metrics in aggregated_by_model.items():
            row = [model_name]
            for m in metric_names:
                if m in metrics:
                    agg = metrics[m]
                    row.append(f"{agg.mean:.4f} ± {agg.std:.4f} (n={agg.n_seeds})")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("_No metrics were produced._")

    lines += [
        "",
        f"Table reports mean ± std across seeds. The full {config.confidence_level:.0%} "
        f"confidence interval (t-distribution, not a normal approximation — see "
        f"metrics.aggregate_across_seeds) is wider than ± std for small seed counts and "
        f"is available per-model in `aggregated_results.json` and the comparison plots below.",
        "",
    ]

    if plot_paths:
        lines += ["## Plots", ""]
        for name, path in sorted(plot_paths.items()):
            lines.append(f"- **{name}**: `{path}`")
        lines.append("")

    output_path.write_text("\n".join(lines))
