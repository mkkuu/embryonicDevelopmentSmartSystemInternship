"""
Validation procedure for embedding caches produced by build_cache.py.

Answers six questions, each an independently-reportable check:

  1. Are dimensions correct?              -> check_dimensions
  2. Are embeddings corrupted?             -> check_corruption
  3. Are metadata aligned?                 -> check_metadata_alignment
  4. Are train/val/test consistent?        -> check_split_consistency
  5. Is every sample present?              -> check_sample_completeness
  6. Are embeddings deterministic?         -> check_determinism_live (optional)

Checks 1-5 only need the cache itself on disk — no GPU, no model, no
checkpoint. Check 6 additionally needs a live model + checkpoint, and is
reported as SKIPPED, not silently omitted, if --checkpoint is not given.

Every dataset-reconstruction parameter (window_size, stride, focal_type,
image_size, mode) is read from the cache's own manifest, never re-entered
on the command line — there is no way for this script's understanding of
"what the cache should contain" to drift from what build_cache.py actually
recorded.

Usage
-----
    cd Training
    python -m embeddings.validate --cache_root ../Embeddings --model_name resnet18

    # with the live determinism spot-check enabled:
    python -m embeddings.validate --cache_root ../Embeddings --model_name resnet18 \\
        --checkpoint ../Results/resnet18/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from DataSet import Embryo_Transition_Dataset
from torchvision import transforms

from .cache import CacheManifest, CacheValidationError, EmbeddingCache

SPLITS = ["train", "val", "test"]


@dataclass
class CheckResult:
    name: str
    passed: Optional[bool]  # True / False / None (= skipped)
    message: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", required=True, help="e.g. ../Embeddings")
    p.add_argument("--model_name", required=True, choices=["resnet18", "timesformer"])
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Enables the live determinism spot-check (question 6). If it "
        "does not match the checkpoint recorded in the cache's manifest, "
        "that mismatch is reported explicitly rather than being confused "
        "with genuine non-determinism.",
    )
    p.add_argument("--live_sample_size", type=int, default=20)
    p.add_argument("--report_path", default=None, help="If given, also write the full report as JSON here.")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------
# 1. Are dimensions correct?
# --------------------------------------------------------------------------

def check_dimensions(caches: Dict[str, EmbeddingCache], model_name: str) -> CheckResult:
    dims = {split: c.embeddings.shape[1] for split, c in caches.items()}
    unique_dims = set(dims.values())
    if len(unique_dims) != 1:
        return CheckResult(
            "dimensions_consistent", False,
            f"Embedding dimensionality differs across splits: {dims}. The "
            f"caches were built with different model configurations and "
            f"must not be used together.",
            details={"dims_by_split": dims},
        )
    D = unique_dims.pop()
    if D <= 0:
        return CheckResult("dimensions_consistent", False, f"Embedding dimension is non-positive ({D}).", details={"dim": D})

    note = ""
    if model_name == "resnet18" and D != 512:
        note = (
            f" Note: 512 is timm resnet18's typical forward_head(pre_logits=True) "
            f"output dimension; D={D} differs from that. Not necessarily wrong, "
            f"but worth confirming against the running environment."
        )
    elif model_name == "timesformer":
        note = (
            " TimeSformer's expected embedding dimension has never been "
            "independently verified in this project (see extractor.py's "
            "module docstring) — do not treat this value as 'correct' without "
            "checking it against the installed transformers version directly."
        )

    return CheckResult(
        "dimensions_consistent", True,
        f"Embedding dimension D={D}, consistent across all {len(dims)} loaded splits.{note}",
        details={"dim": D},
    )


# --------------------------------------------------------------------------
# 2. Are embeddings corrupted?
# --------------------------------------------------------------------------

def check_corruption(caches: Dict[str, EmbeddingCache]) -> CheckResult:
    problems: List[str] = []
    for split, cache in caches.items():
        emb = cache.embeddings.float()
        n_nan = torch.isnan(emb).any(dim=1).sum().item()
        n_inf = torch.isinf(emb).any(dim=1).sum().item()
        n_zero_rows = (emb.abs().sum(dim=1) == 0).sum().item()
        n_total = emb.shape[0]
        n_unique = torch.unique(emb, dim=0).shape[0]
        n_duplicates = n_total - n_unique
        dim_std = emb.std(dim=0)
        n_dead_dims = (dim_std < 1e-8).sum().item()

        if n_nan:
            problems.append(f"{split}: {n_nan} rows contain NaN")
        if n_inf:
            problems.append(f"{split}: {n_inf} rows contain Inf")
        if n_zero_rows:
            problems.append(f"{split}: {n_zero_rows} rows are exactly all-zero")
        if n_duplicates:
            problems.append(f"{split}: {n_duplicates} duplicate embedding rows (of {n_total})")
        if n_dead_dims:
            problems.append(
                f"{split}: {n_dead_dims}/{emb.shape[1]} dimensions have ~zero "
                f"variance across the split (a 'dead' feature)"
            )

    passed = len(problems) == 0
    return CheckResult(
        "no_corruption", passed,
        "No NaN/Inf/all-zero/duplicate rows or dead dimensions found." if passed
        else "Corruption/anomalies found: " + "; ".join(problems),
        details={"problems": problems},
    )


# --------------------------------------------------------------------------
# 3. Are metadata aligned?
# --------------------------------------------------------------------------

def check_metadata_alignment(caches: Dict[str, EmbeddingCache]) -> CheckResult:
    # embedding_index uniqueness and in-range-ness are already enforced by
    # EmbeddingCache's own constructor (load() would have raised already
    # if violated) — re-stated here explicitly so the report is a complete,
    # transparent record rather than a silent assumption resting on
    # construction having merely not crashed.
    problems: List[str] = []
    for split, cache in caches.items():
        md = cache.metadata
        if md["consistency_flag"].isnull().any():
            problems.append(f"{split}: null consistency_flag values present")
        bad_flags = ~md["consistency_flag"].isin([0, 1])
        if bad_flags.any():
            problems.append(f"{split}: {int(bad_flags.sum())} rows have consistency_flag outside {{0,1}}")
        for col in ("first_frame_phase", "last_frame_phase"):
            if (md[col] < 0).any():
                problems.append(f"{split}: negative values present in {col}")
        if md["video_name"].isnull().any() or (md["video_name"] == "").any():
            problems.append(f"{split}: missing/empty video_name values")
        if md["window_start"].isnull().any() or (md["window_start"] < 0).any():
            problems.append(f"{split}: missing or negative window_start values")

    passed = len(problems) == 0
    return CheckResult(
        "metadata_aligned", passed,
        "embedding_index alignment enforced by EmbeddingCache construction (unique, in-range); "
        + ("no other metadata anomalies found." if passed else "anomalies found: " + "; ".join(problems)),
        details={"problems": problems},
    )


# --------------------------------------------------------------------------
# 4. Are train/val/test consistent?
# --------------------------------------------------------------------------

def check_split_consistency(caches: Dict[str, EmbeddingCache], model_root: Path) -> CheckResult:
    problems: List[str] = []
    manifests: Dict[str, CacheManifest] = {}

    for split in caches:
        split_manifest_path = model_root / split / "manifest.json"
        if not split_manifest_path.exists():
            problems.append(
                f"{split}: no per-split manifest.json at {split_manifest_path} — "
                f"this cache was built before build_cache.py started writing "
                f"per-split manifests; config consistency for this split cannot "
                f"be checked until it is rebuilt."
            )
            continue
        manifests[split] = CacheManifest.from_json(split_manifest_path)

    splits_with_manifests = list(manifests.keys())
    if len(splits_with_manifests) >= 2:
        reference_split = splits_with_manifests[0]
        reference = manifests[reference_split]
        for split in splits_with_manifests[1:]:
            mismatches = reference.matches(manifests[split], check_checkpoint=True)
            if mismatches:
                problems.append(f"{reference_split} vs {split} config mismatch: " + "; ".join(mismatches))

    # Patient-level leakage check: no video should appear in more than one split.
    video_sets = {split: set(c.metadata["video_name"]) for split, c in caches.items()}
    split_names = list(video_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = video_sets[split_names[i]] & video_sets[split_names[j]]
            if overlap:
                sample = sorted(overlap)[:5]
                problems.append(
                    f"{split_names[i]} and {split_names[j]} share {len(overlap)} "
                    f"video(s) — patient-level leakage: {sample}"
                    + ("..." if len(overlap) > 5 else "")
                )

    passed = len(problems) == 0
    return CheckResult(
        "train_val_test_consistent", passed,
        "Config consistent across splits and no patient-level leakage found." if passed
        else "Issues found: " + "; ".join(problems),
        details={"problems": problems},
    )


# --------------------------------------------------------------------------
# 5. Is every sample present?
# --------------------------------------------------------------------------

def check_sample_completeness(caches: Dict[str, EmbeddingCache], manifest: CacheManifest) -> CheckResult:
    # A transform is required by the constructor but has no bearing on
    # __len__ or video_sequences — use the cheapest correctly-shaped one.
    dummy_transform = transforms.Compose(
        [transforms.Resize((manifest.image_size, manifest.image_size)), transforms.ToTensor()]
    )

    problems: List[str] = []
    for split, cache in caches.items():
        used_for = split.capitalize()  # "train" -> "Train", matching UsedFor's convention
        try:
            dataset = Embryo_Transition_Dataset(
                transform=dummy_transform,
                FocalType=manifest.focal_type,
                window_size=manifest.window_size,
                stride=manifest.stride,
                UsedFor=used_for,
                NumberOfVideos=0,
                mode=manifest.mode,
            )
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            problems.append(f"{split}: could not reconstruct the dataset to check completeness against it ({e})")
            continue

        expected, actual = len(dataset), len(cache)
        if actual > expected:
            problems.append(
                f"{split}: cache has MORE rows ({actual}) than the dataset "
                f"currently produces ({expected}) — cache may be stale or "
                f"contain duplicated windows"
            )
        elif actual < expected:
            problems.append(
                f"{split}: cache is missing {expected - actual}/{expected} windows "
                f"(most likely image-load failures during build — check that "
                f"run's stdout log for the specific file paths)"
            )

        expected_videos = set(dataset.data_frame["Video_name"].unique())
        cached_videos = set(cache.metadata["video_name"])
        missing_videos = expected_videos - cached_videos
        if missing_videos:
            sample = sorted(missing_videos)[:5]
            problems.append(
                f"{split}: {len(missing_videos)} entire video(s) missing from "
                f"the cache: {sample}" + ("..." if len(missing_videos) > 5 else "")
            )

    passed = len(problems) == 0
    return CheckResult(
        "sample_completeness", passed,
        "Every expected window and video is present in every loaded split." if passed
        else "Completeness issues: " + "; ".join(problems),
        details={"problems": problems},
    )


# --------------------------------------------------------------------------
# 6. Are embeddings deterministic? (optional, needs --checkpoint)
# --------------------------------------------------------------------------

def check_determinism_live(
    caches: Dict[str, EmbeddingCache],
    manifest: CacheManifest,
    model_name: str,
    checkpoint_path: str,
    sample_size: int,
    random_state: int,
) -> CheckResult:
    # Imported here, not at module level: importing ModelBuilder triggers
    # its own module-level ConfigArgs() call (see build_cache.py's
    # docstring for the same issue) — no reason to pay that cost for every
    # validate.py run when this is the one check that's actually optional.
    from ModelBuilder import get_model

    from .cache import sha256_of_file
    from .extractor import EmbeddingExtractor

    checkpoint_sha256 = sha256_of_file(Path(checkpoint_path))
    warning = ""
    if manifest.checkpoint_sha256 and checkpoint_sha256 != manifest.checkpoint_sha256:
        warning = (
            f"WARNING: the checkpoint passed here (sha256={checkpoint_sha256[:12]}...) "
            f"does not match the one recorded in the cache's manifest "
            f"(sha256={manifest.checkpoint_sha256[:12]}...). Any difference found "
            f"below reflects that mismatch, not necessarily non-determinism. "
        )

    transform = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.CenterCrop((manifest.image_size, manifest.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = get_model(model_name, preTrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    extractor = EmbeddingExtractor(model, device)

    split = "val" if "val" in caches else next(iter(caches))
    cache = caches[split]
    dataset = Embryo_Transition_Dataset(
        transform=transform,
        FocalType=manifest.focal_type,
        window_size=manifest.window_size,
        stride=manifest.stride,
        UsedFor=split.capitalize(),
        NumberOfVideos=0,
        mode=manifest.mode,
    )

    rng = random.Random(random_state)
    n_rows = len(cache)
    sample_indices = rng.sample(range(n_rows), min(sample_size, n_rows))

    problems: List[str] = []
    max_abs_diffs: List[float] = []

    with torch.no_grad():
        for embedding_index in sample_indices:
            cached_embedding, row = cache[embedding_index]
            window_start = int(row["window_start"])
            item = dataset[window_start]
            if item is None:
                problems.append(
                    f"embedding_index={embedding_index}: source window {window_start} "
                    f"now fails to load (it was cached successfully before — the "
                    f"underlying image file may have moved or been removed)"
                )
                continue
            images = item[0].unsqueeze(0)
            fresh_embedding = extractor.extract(images).cpu().squeeze(0)

            diff = (cached_embedding.float() - fresh_embedding.float()).abs()
            max_abs_diffs.append(diff.max().item())
            if not torch.allclose(cached_embedding.float(), fresh_embedding.float(), atol=1e-4, rtol=1e-3):
                problems.append(
                    f"embedding_index={embedding_index} (video={row['video_name']}, "
                    f"window={window_start}): max abs diff {diff.max().item():.6g} "
                    f"exceeds tolerance (atol=1e-4, rtol=1e-3)"
                )

    n_checked = len(sample_indices)
    n_ok = n_checked - len(problems)
    passed = len(problems) == 0
    return CheckResult(
        "determinism_live_spotcheck", passed,
        warning + (
            f"{n_ok}/{n_checked} spot-checked embeddings matched a fresh "
            f"extraction within tolerance."
            if passed else
            f"{len(problems)}/{n_checked} spot-checked embeddings did not "
            f"match a fresh extraction: " + "; ".join(problems[:5]) + ("..." if len(problems) > 5 else "")
        ),
        details={
            "sample_size": n_checked,
            "max_abs_diff_overall": max(max_abs_diffs) if max_abs_diffs else None,
            "problems": problems,
        },
    )


# --------------------------------------------------------------------------

def print_report(results: List[CheckResult]) -> None:
    print("\n" + "=" * 78)
    print("EMBEDDING CACHE VALIDATION REPORT")
    print("=" * 78)
    for r in results:
        status = "SKIP" if r.passed is None else ("PASS" if r.passed else "FAIL")
        print(f"[{status}] {r.name}")
        print(f"       {r.message}")
    n_pass = sum(1 for r in results if r.passed is True)
    n_fail = sum(1 for r in results if r.passed is False)
    n_skip = sum(1 for r in results if r.passed is None)
    print("-" * 78)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 78 + "\n")


def main() -> None:
    args = parse_args()
    model_root = Path(args.cache_root) / args.model_name
    manifest_path = model_root / "manifest.json"
    if not manifest_path.exists():
        raise CacheValidationError(f"No manifest.json found at {manifest_path}.")
    manifest = CacheManifest.from_json(manifest_path)

    caches: Dict[str, EmbeddingCache] = {}
    for split in SPLITS:
        split_dir = model_root / split
        try:
            # lazy=False: validation touches every row anyway (corruption,
            # duplicate, dead-dimension checks), so memory-mapping buys
            # nothing here and would only add per-access overhead.
            caches[split] = EmbeddingCache.load(split_dir, lazy=False)
        except CacheValidationError as e:
            print(f"WARNING: could not load split '{split}': {e}")
    if not caches:
        raise CacheValidationError(f"No valid split caches found under {model_root}.")

    results: List[CheckResult] = [
        check_dimensions(caches, args.model_name),
        check_corruption(caches),
        check_metadata_alignment(caches),
        check_split_consistency(caches, model_root),
        check_sample_completeness(caches, manifest),
    ]

    if args.checkpoint:
        results.append(
            check_determinism_live(
                caches, manifest, args.model_name, args.checkpoint, args.live_sample_size, args.random_state
            )
        )
    else:
        results.append(
            CheckResult(
                "determinism_live_spotcheck", None,
                "SKIPPED — pass --checkpoint to enable this check.",
            )
        )

    print_report(results)
    if args.report_path:
        Path(args.report_path).write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"Full report written to {args.report_path}")

    if any(r.passed is False for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
