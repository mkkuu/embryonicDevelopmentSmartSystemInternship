"""
Exploratory analysis of a cached embedding split: PCA / variance explained,
t-SNE, UMAP, nearest neighbors, phase clustering, patient clustering.

This is diagnostic tooling, not a claim about the underlying science — see
RESEARCH_BLUEPRINT.md for what these numbers would and would not establish
about the actual research questions (H1-H4). What this script answers is
narrower and more immediate: does this embedding cache contain *any*
structure worth building on, and is that structure organized more by
developmental phase, by patient identity, or by neither.

All plotting is headless (matplotlib Agg backend) — this is meant to run
on the GPU server, not a desktop with a display.

Usage
-----
    cd Training
    python -m embeddings.explore --cache_root ../Embeddings --model_name resnet18 \\
        --split val --output_dir ../Embeddings/resnet18/exploration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .cache import EmbeddingCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--model_name", required=True, choices=["resnet18", "timesformer"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_neighbors", type=int, default=10)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--analyses",
        default="pca,tsne,umap,nearest_neighbors,phase_clustering,patient_clustering",
        help="Comma-separated subset to run. 'umap' is skipped automatically, "
        "with a clear message, if umap-learn is not installed.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# shared plotting helper
# --------------------------------------------------------------------------

def _scatter_2d(points_2d: np.ndarray, metadata: pd.DataFrame, color_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    values = metadata[color_col]
    n_unique = values.nunique()

    if n_unique <= 20:
        categories = sorted(values.unique(), key=str)
        cmap = plt.get_cmap("tab20", max(len(categories), 1))
        for i, cat in enumerate(categories):
            mask = (values == cat).values
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1], s=6, color=cmap(i), label=str(cat), alpha=0.7)
        ax.legend(fontsize=6, markerscale=2, ncol=2, loc="best")
    else:
        # Too many distinct values (e.g. video_name) for a readable legend —
        # color by a stable code instead, to show cluster structure without
        # claiming a legend can be read.
        codes = pd.factorize(values)[0]
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=6, c=codes, cmap="tab20", alpha=0.7)
        ax.set_title(title + f"  (n={n_unique} distinct values, no legend)")

    if n_unique <= 20:
        ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# PCA / variance explained
# --------------------------------------------------------------------------

def run_pca(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, random_state: int) -> Dict:
    n_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    projected = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(explained) + 1), cumulative, marker="o", markersize=3)
    ax.axhline(0.90, linestyle="--", color="gray", linewidth=1)
    ax.axhline(0.95, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("PCA — variance explained")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_variance_explained.png", dpi=150)
    plt.close(fig)

    _scatter_2d(projected[:, :2], metadata, "first_frame_phase", output_dir / "pca_2d_by_phase.png",
                "PCA (PC1 vs PC2), colored by first_frame_phase")
    _scatter_2d(projected[:, :2], metadata, "video_name", output_dir / "pca_2d_by_video.png",
                "PCA (PC1 vs PC2), colored by video_name")

    pd.DataFrame({
        "component": range(1, len(explained) + 1),
        "variance_explained": explained,
        "cumulative_variance_explained": cumulative,
    }).to_csv(output_dir / "pca_variance_explained.csv", index=False)

    return {
        "n_components_fit": n_components,
        "components_needed_for_90pct_variance": n_for_90,
        "components_needed_for_95pct_variance": n_for_95,
        "variance_explained_by_pc1": float(explained[0]),
        "variance_explained_by_pc1_and_pc2": float(explained[:2].sum()),
    }


# --------------------------------------------------------------------------
# t-SNE
# --------------------------------------------------------------------------

def run_tsne(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, random_state: int) -> Dict:
    # Standard practice: reduce to a moderate dimension via PCA first, both
    # for speed and because t-SNE on very high-dimensional input directly
    # is known to be unreliable.
    pre = embeddings
    if embeddings.shape[1] > 50:
        pre = PCA(n_components=50, random_state=random_state).fit_transform(embeddings)

    perplexity = min(30, max(5, embeddings.shape[0] // 100))
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=perplexity)
    projected = tsne.fit_transform(pre)

    _scatter_2d(projected, metadata, "first_frame_phase", output_dir / "tsne_by_phase.png",
                "t-SNE, colored by first_frame_phase")
    _scatter_2d(projected, metadata, "video_name", output_dir / "tsne_by_video.png",
                "t-SNE, colored by video_name")

    return {"perplexity_used": perplexity}


# --------------------------------------------------------------------------
# UMAP (optional dependency)
# --------------------------------------------------------------------------

def run_umap(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, random_state: int) -> Dict:
    try:
        import umap
    except ImportError:
        print(
            "SKIPPED umap: the `umap-learn` package is not installed in this "
            "environment. It has been added to requirements.txt — run "
            "`pip install -r requirements.txt` (or `pip install umap-learn` "
            "directly) to enable this analysis."
        )
        return {"skipped": True, "reason": "umap-learn not installed"}

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    projected = reducer.fit_transform(embeddings)

    _scatter_2d(projected, metadata, "first_frame_phase", output_dir / "umap_by_phase.png",
                "UMAP, colored by first_frame_phase")
    _scatter_2d(projected, metadata, "video_name", output_dir / "umap_by_video.png",
                "UMAP, colored by video_name")

    return {"skipped": False}


# --------------------------------------------------------------------------
# nearest neighbors
# --------------------------------------------------------------------------

def run_nearest_neighbors(
    embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, n_neighbors: int, random_state: int
) -> Dict:
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")  # +1: a point is its own nearest neighbor
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    phase = metadata["first_frame_phase"].values
    video = metadata["video_name"].values

    phase_agreement = np.empty(len(embeddings))
    video_agreement = np.empty(len(embeddings))
    for i in range(len(embeddings)):
        neighbor_idx = indices[i, 1:]  # drop self
        phase_agreement[i] = np.mean(phase[neighbor_idx] == phase[i])
        video_agreement[i] = np.mean(video[neighbor_idx] == video[i])

    rng = np.random.RandomState(random_state)
    example_indices = rng.choice(len(embeddings), size=min(20, len(embeddings)), replace=False)
    examples = [
        {
            "query_index": int(i),
            "query_video": str(video[i]),
            "query_phase": int(phase[i]),
            "neighbor_videos": [str(v) for v in video[indices[i, 1:]]],
            "neighbor_phases": [int(p) for p in phase[indices[i, 1:]]],
            "neighbor_cosine_distances": [float(d) for d in distances[i, 1:]],
        }
        for i in example_indices
    ]
    (output_dir / "nearest_neighbors_examples.json").write_text(json.dumps(examples, indent=2))

    return {
        "mean_neighbor_phase_agreement": float(phase_agreement.mean()),
        "mean_neighbor_video_agreement": float(video_agreement.mean()),
        "note": (
            "High video_agreement alongside low phase_agreement would suggest "
            "embeddings mostly encode 'which patient this is' rather than "
            "developmental state — worth investigating before trusting any "
            "phase-related conclusion drawn from this embedding."
        ),
    }


# --------------------------------------------------------------------------
# phase clustering / patient clustering
# --------------------------------------------------------------------------

def _clustering_agreement(embeddings: np.ndarray, raw_labels, random_state: int) -> Dict:
    labels = pd.factorize(raw_labels)[0]
    k = len(np.unique(labels))
    if k < 2:
        return {"skipped": True, "reason": f"only {k} distinct label value(s) present — clustering agreement is undefined"}
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_assignment = kmeans.fit_predict(embeddings)
    return {
        "skipped": False,
        "k": int(k),
        "adjusted_rand_index": float(adjusted_rand_score(labels, cluster_assignment)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels, cluster_assignment)),
    }


def run_phase_clustering(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, random_state: int) -> Dict:
    result = _clustering_agreement(embeddings, metadata["first_frame_phase"].values, random_state)
    (output_dir / "phase_clustering.json").write_text(json.dumps(result, indent=2))
    return result


def run_patient_clustering(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path, random_state: int) -> Dict:
    result = _clustering_agreement(embeddings, metadata["video_name"].values, random_state)
    (output_dir / "patient_clustering.json").write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------

ANALYSES = {
    "pca": run_pca,
    "tsne": run_tsne,
    "umap": run_umap,
    "phase_clustering": run_phase_clustering,
    "patient_clustering": run_patient_clustering,
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # lazy=False: every analysis here touches the full matrix at least
    # once (PCA/t-SNE/clustering all need it materialized), so
    # memory-mapping buys nothing and would only add access overhead.
    cache = EmbeddingCache.load(Path(args.cache_root) / args.model_name / args.split, lazy=False)
    embeddings = cache.embeddings.float().numpy()
    metadata = cache.metadata.reset_index(drop=True)

    requested = [a.strip() for a in args.analyses.split(",") if a.strip()]
    unknown = set(requested) - set(ANALYSES) - {"nearest_neighbors"}
    if unknown:
        raise ValueError(f"Unknown analyses requested: {unknown}. Valid options: {list(ANALYSES) + ['nearest_neighbors']}")

    summary = {
        "model_name": args.model_name,
        "split": args.split,
        "n_samples": len(metadata),
        "n_dims": embeddings.shape[1],
    }

    if "nearest_neighbors" in requested:
        summary["nearest_neighbors"] = run_nearest_neighbors(
            embeddings, metadata, output_dir, args.n_neighbors, args.random_state
        )
        print(f"[nearest_neighbors] {json.dumps(summary['nearest_neighbors'])}")

    for name in requested:
        if name == "nearest_neighbors":
            continue
        summary[name] = ANALYSES[name](embeddings, metadata, output_dir, args.random_state)
        print(f"[{name}] {json.dumps(summary[name])}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
