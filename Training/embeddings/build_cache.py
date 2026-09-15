"""
Build an embedding cache for one (model, split) combination by running the
existing, unmodified ModelBuilder.get_model() once over
Embryo_Transition_Dataset windows and saving the resulting embeddings.

Deliberately does not go through Load_data.get_dataloaders() or otherwise
rely on Training/config_args.ConfigArgs()'s default (currently fragile,
see BASELINE.md) config-file path — every parameter this script needs is
passed explicitly on the command line, so this script's correctness does
not depend on that bug being fixed or not.

Known coupling to be aware of (not fixable here without editing
ModelBuilder.py, which this package does not do): ModelBuilder.py reads
its own module-level WINDOW_SIZE from a *separate* ConfigArgs() call at
import time, independent of the --window_size passed to this script, and
uses it to set the resnet18 branch's in_chans. If the two disagree, model
construction below raises a clear error instead of failing later with a
cryptic channel-mismatch deep inside forward_features().

Checkpoint/resume (for unattended multi-hour runs)
----------------------------------------------------
Progress is checkpointed at VIDEO boundaries under
{output_root}/{model_name}/{split}/.checkpoint/ (manifest.json, progress.json,
progress.log, shards/{video_name}.{pt,csv}). Each shard is written
atomically (temp file + os.replace); a video is only marked complete after
both its shard files exist under their final names. The final
embeddings.pt/metadata.csv/manifest.json (unchanged EmbeddingCache format)
are only (re)written, also atomically, once every requested video has a
completed shard.

By default (no existing checkpoint), behavior and output are unchanged from
before this mechanism existed. If a checkpoint directory already exists,
the script refuses to run again without --resume, so a partially- or
fully-completed run is never silently discarded or overwritten. With
--resume, the checkpoint's manifest and its recorded video selection
(video_order) are both validated against the current arguments before any
work resumes; on any mismatch, the run refuses to continue.

Usage
-----
    cd Training
    python -m embeddings.build_cache \\
        --model_name resnet18 \\
        --checkpoint ../Results/resnet18/best_model.pth \\
        --window_size 8 --stride 1 --focal_type F0 --image_size 224 \\
        --split Val \\
        --output_root ../Embeddings

    # resume an interrupted run (same arguments):
    python -m embeddings.build_cache ... --resume

Run once per split (Train/Val/Test) you need cached.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from DataSet import Embryo_Transition_Dataset

# ModelBuilder.py instantiates ConfigArgs() at import time (Training/
# config_args.py), which parses the real process sys.argv with its own
# argparse.ArgumentParser — one that does not recognize this script's own
# flags (--checkpoint, --focal_type, --split, --output_root, --device),
# causing argparse to sys.exit(2) before build_cache.main() ever runs. This
# is the same root cause already fixed for Tests/conftest.py; same fix here,
# scoped to just this one import: neutralize sys.argv only while ModelBuilder
# is first imported (and thus its module-level ConfigArgs() executes), then
# restore it so this script's own parse_args() below sees the real command
# line untouched. `.extractor`'s own `from ModelBuilder import ...` (next
# import below) does not need the same guard — by then ModelBuilder is
# already in sys.modules and its module body will not run a second time.
_real_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from ModelBuilder import get_model
finally:
    sys.argv = _real_argv

from .cache import CacheManifest, EmbeddingCache, sha256_of_file
from .extractor import EmbeddingExtractor

EXTRACTOR_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, choices=["resnet18", "timesformer"])
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a state_dict .pth file. If omitted, the model's "
        "off-the-shelf pretrained weights are used (no fine-tuning) — "
        "recorded in the manifest either way, so consumers can tell which "
        "flavor of embedding they're getting.",
    )
    p.add_argument("--window_size", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--focal_type", default="F0")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--split", required=True, choices=["Train", "Val", "Test"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="Limit the split to at most this many videos, passed straight "
        "through to Embryo_Transition_Dataset's existing NumberOfVideos "
        "parameter. 0 (default) = use the entire requested split, "
        "unchanged from previous behavior. Selection is a seeded sample "
        "(DataSet.py's own random_state=seed, not the first N videos by "
        "position) — deterministic and reproducible across runs, not a "
        "new source of randomness. Intended for smoke-testing this script "
        "against a small real subset before a full extraction.",
    )
    p.add_argument("--output_root", default="../Embeddings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previously interrupted run for this exact "
        "(model, split) cache, skipping videos already checkpointed under "
        ".checkpoint/. Off by default. If a checkpoint already exists and "
        "this flag is not given, the script refuses to run (to avoid "
        "silently discarding or duplicating prior progress). On resume, "
        "the checkpoint's manifest AND its exact recorded video "
        "selection/order are validated against the current arguments; any "
        "mismatch refuses to continue. batch_size/num_workers are recorded "
        "for provenance but never invalidate a resume, since they do not "
        "affect the extracted embeddings.",
    )
    return p.parse_args()


class _IndexedDataset(Dataset):
    """Wraps a Dataset so __getitem__ also returns the original index.

    Embryo_Transition_Dataset.__getitem__ returns None on an image-load
    failure (a known, documented behavior — see BASELINE.md), which the
    default collate function cannot handle. _skip_none_collate below
    filters those out, but once any items in a batch can be dropped,
    naive `batch_idx * batch_size + i` arithmetic to recover which
    original window each embedding came from is silently wrong. Carrying
    the index through the wrapper and the collate step fixes this
    correctly rather than assuming failures never happen.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if item is None:
            return None
        return (idx, *item)


def _skip_none_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_manifest(args: argparse.Namespace, mode: str) -> CacheManifest:
    checkpoint_sha256 = sha256_of_file(Path(args.checkpoint)) if args.checkpoint else None
    return CacheManifest(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        window_size=args.window_size,
        stride=args.stride,
        focal_type=args.focal_type,
        image_size=args.image_size,
        mode=mode,
        extractor_version=EXTRACTOR_VERSION,
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
    )


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    model = get_model(args.model_name, preTrained=(args.checkpoint is None))

    if args.model_name == "resnet18":
        # See module docstring: ModelBuilder.py's own WINDOW_SIZE global is
        # independent of --window_size. Catch a mismatch here, clearly,
        # instead of a cryptic shape error inside forward_features().
        actual_in_channels = model.conv1.in_channels
        if actual_in_channels != args.window_size:
            raise ValueError(
                f"Model was built with in_chans={actual_in_channels} (from "
                f"ModelBuilder.py's own ConfigArgs()-derived WINDOW_SIZE), "
                f"but this script was asked to build --window_size="
                f"{args.window_size} windows. These must match — they are "
                f"currently resolved from two independent sources. This is "
                f"most likely the config.ini path-resolution issue "
                f"documented in BASELINE.md; check what "
                f"Training/config_args.py's ConfigArgs() actually resolves "
                f"'window_size' to right now, independent of what you pass "
                f"here, before proceeding."
            )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"{args.checkpoint} did not load as a state_dict (got "
                f"{type(state_dict).__name__} instead). This is the same "
                f"torch.load/state_dict format mismatch already documented "
                f"for WebApplication/Classes/Doctor.py — if this checkpoint "
                f"was saved any other way than model.state_dict(), it needs "
                f"to be re-saved before it can be used here."
            )
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(args.device)
    return model


# ---------------------------------------------------------------------------
# Checkpoint/resume plumbing — filesystem-based, no database, additive only:
# never touches the final embeddings.pt/metadata.csv/manifest.json format.
# ---------------------------------------------------------------------------

def _video_order_and_indices(dataset) -> tuple[list[str], dict[str, list[int]]]:
    """Deterministic per-video grouping of dataset indices, in first-
    appearance order. Embryo_Transition_Dataset._create_sequences groups
    windows by video internally (via a dict keyed by video), so windows for
    a given video are always contiguous in dataset index order for a fixed
    set of constructor arguments — verified empirically (2-video smoke
    test: window_start ranges were contiguous and non-interleaved per
    video). NumberOfVideos performs a *seeded sample*, not a positional
    'first N' selection, so this order is reproducible run-to-run for the
    same arguments but must never be assumed to be "first N" or
    alphabetical — this is exactly why it is recorded explicitly below
    rather than re-derived and trusted implicitly on resume."""
    order: list[str] = []
    indices: dict[str, list[int]] = {}
    for idx, seq in enumerate(dataset.video_sequences):
        video_name = seq[0]["identifier"].rsplit("_Image_", 1)[0]
        if video_name not in indices:
            indices[video_name] = []
            order.append(video_name)
        indices[video_name].append(idx)
    return order, indices


def _atomic_write_tensor(tensor: torch.Tensor, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, path)  # atomic on POSIX: never leaves a partial file under the final name


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _log(path: Path, message: str) -> None:
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    with open(path, "a") as f:  # append-only, never truncated or rewritten
        f.write(f"[{timestamp}] {message}\n")


def main() -> None:
    args = parse_args()
    mode = "image_seq" if args.model_name == "resnet18" else "video"

    transform = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = Embryo_Transition_Dataset(
        transform=transform,
        FocalType=args.focal_type,
        window_size=args.window_size,
        stride=args.stride,
        UsedFor=args.split,
        NumberOfVideos=args.max_videos,
        mode=mode,
    )
    video_order, video_indices = _video_order_and_indices(dataset)

    output_root = Path(args.output_root) / args.model_name
    split_dir = output_root / args.split.lower()
    checkpoint_dir = split_dir / ".checkpoint"
    shards_dir = checkpoint_dir / "shards"
    progress_path = checkpoint_dir / "progress.json"
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    log_path = checkpoint_dir / "progress.log"

    current_manifest = build_manifest(args, mode)

    if progress_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"A checkpoint already exists at {checkpoint_dir}. Refusing "
                f"to start a fresh run that would silently ignore or "
                f"overwrite prior progress. Pass --resume to continue it, "
                f"or remove {checkpoint_dir} manually first if you really "
                f"intend to start over from scratch."
            )
        progress = json.loads(progress_path.read_text())
        checkpoint_manifest = CacheManifest.from_json(checkpoint_manifest_path)

        # Only parameters that affect the extracted representation/dataset
        # identity invalidate a resume — CacheManifest.matches() already
        # covers model_name/window_size/stride/focal_type/image_size/mode
        # plus checkpoint_sha256 (content hash, not path string). `split`
        # is not a manifest field: it is structural (a different split
        # always resolves to a different split_dir/checkpoint_dir, so a
        # split mismatch is impossible by construction, not by comparison).
        mismatches = checkpoint_manifest.matches(current_manifest, check_checkpoint=True)
        if checkpoint_manifest.extractor_version != current_manifest.extractor_version:
            mismatches.append(
                f"extractor_version: checkpoint has "
                f"{checkpoint_manifest.extractor_version!r}, current is "
                f"{current_manifest.extractor_version!r} — resuming across "
                f"an extraction-logic change could silently mix shards "
                f"computed two different ways."
            )
        # max_videos is deliberately NOT compared as a bare integer: the
        # underlying selection is a seeded *sample*, not a positional
        # "first N" — compare the actual recorded video set/order instead.
        if progress["video_order"] != video_order:
            mismatches.append(
                "video_order: the checkpoint's recorded video selection "
                "does not match what --split/--focal_type/--max_videos "
                "would select right now. Do not resume across a changed "
                "selection — recompute from scratch instead."
            )
        if mismatches:
            raise RuntimeError(
                "Refusing to resume: checkpoint configuration does not "
                "match the current run.\n  - " + "\n  - ".join(mismatches)
            )

        completed_videos = list(progress["completed_videos"])
        _log(log_path, f"Resumed: {len(completed_videos)}/{len(video_order)} videos already complete.")
        print(f"Resuming from checkpoint: {len(completed_videos)}/{len(video_order)} "
              f"videos already complete, {len(video_order) - len(completed_videos)} remaining.")
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)
        current_manifest.to_json(checkpoint_manifest_path)
        progress = {
            "video_order": video_order,
            "completed_videos": [],
            "completed_window_count": 0,
            "status": "in_progress",
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "last_updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            # Recorded for provenance only — never validated on resume,
            # since neither affects the extracted embeddings (eval-mode
            # forward pass, no batch-norm stat updates / dropout).
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        _atomic_write_json(progress, progress_path)
        _log(log_path, f"Started: {len(video_order)} videos to process.")
        completed_videos = []

    model = load_model(args)
    device = torch.device(args.device)
    extractor = EmbeddingExtractor(model, device)

    remaining = [v for v in video_order if v not in completed_videos]
    print(f"{len(remaining)}/{len(video_order)} video(s) to process this run.")

    indexed_dataset = _IndexedDataset(dataset)
    embedding_dim = None  # learned from the first successfully extracted window this run

    with torch.no_grad():
        for video_name in remaining:
            indices = video_indices[video_name]
            loader = DataLoader(
                Subset(indexed_dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,  # order preservation — window_start below
                num_workers=args.num_workers,
                collate_fn=_skip_none_collate,
            )

            video_embeddings, video_rows = [], []
            for batch in loader:
                if batch is None:
                    continue  # every item in this batch failed to load
                orig_indices, images, consistency_flags, first_phases, last_phases = batch
                embeddings = extractor.extract(images).cpu()
                video_embeddings.append(embeddings)
                embedding_dim = embeddings.shape[1]

                for i in range(embeddings.shape[0]):
                    video_rows.append(
                        {
                            "video_name": video_name,
                            "window_start": int(orig_indices[i]),
                            "consistency_flag": int(consistency_flags[i]),
                            "first_frame_phase": int(first_phases[i]),
                            "last_frame_phase": int(last_phases[i]),
                        }
                    )

            n_requested = len(indices)
            n_produced = len(video_rows)
            if n_produced < n_requested:
                print(
                    f"WARNING: video={video_name}: {n_requested - n_produced}/"
                    f"{n_requested} windows failed to load (image read "
                    f"errors) and were skipped."
                )

            if n_produced == 0:
                if embedding_dim is None:
                    raise RuntimeError(
                        f"Every window failed to load for video={video_name}, "
                        f"and no embedding has been produced yet this run, so "
                        f"the embedding dimensionality is unknown — refusing "
                        f"to write an ambiguous empty shard. Investigate the "
                        f"image-load failures above before retrying."
                    )
                video_tensor = torch.empty((0, embedding_dim), dtype=torch.float32)
            else:
                video_tensor = torch.cat(video_embeddings, dim=0)
            video_meta = pd.DataFrame(video_rows)

            # Checkpoint boundary: write both shard files atomically FIRST,
            # only then record completion — a kill between these two steps
            # just means this video is recomputed (safely, not duplicated)
            # on the next --resume, per the docstring's interruption
            # semantics.
            _atomic_write_tensor(video_tensor, shards_dir / f"{video_name}.pt")
            _atomic_write_csv(video_meta, shards_dir / f"{video_name}.csv")

            progress["completed_videos"].append(video_name)
            progress["completed_window_count"] += n_produced
            progress["last_updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            _atomic_write_json(progress, progress_path)
            _log(
                log_path,
                f"Completed video={video_name} windows={n_produced} "
                f"({len(progress['completed_videos'])}/{len(video_order)})",
            )
            print(
                f"[{len(progress['completed_videos'])}/{len(video_order)}] "
                f"video={video_name} windows={n_produced} -> shard written"
            )

    # ---- Finalize: merge all per-video shards into the canonical,
    # unmodified EmbeddingCache format. Only reached once every requested
    # video has a completed shard (the loop above only exits after
    # processing all of `remaining`, and completed_videos + remaining
    # covers video_order exactly). ----
    assert set(progress["completed_videos"]) == set(video_order), (
        "Internal error: not every requested video is marked complete "
        "after the processing loop finished."
    )

    all_embeddings, all_rows = [], []
    running_index = 0
    for video_name in video_order:  # deterministic order — see docstring
        emb = torch.load(shards_dir / f"{video_name}.pt")
        meta = pd.read_csv(shards_dir / f"{video_name}.csv")
        if len(meta) != emb.shape[0]:
            raise RuntimeError(
                f"Shard corruption detected for video={video_name}: "
                f"{len(meta)} metadata rows vs {emb.shape[0]} embeddings — "
                f"refusing to merge. This should be impossible given the "
                f"atomic per-shard write discipline; investigate manually."
            )
        all_embeddings.append(emb)
        for _, row in meta.iterrows():
            row_dict = row.to_dict()
            row_dict["embedding_index"] = running_index
            all_rows.append(row_dict)
            running_index += 1

    n_produced = running_index
    if n_produced == 0:
        raise RuntimeError(
            f"Zero embeddings were produced for split={args.split} — every "
            f"window failed to load, or the split is empty. Refusing to "
            f"write an empty cache."
        )

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    metadata = pd.DataFrame(all_rows)

    # Final write, atomic per file (not a whole-directory swap: split_dir
    # already contains .checkpoint/, and os.replace() on a directory
    # requires the destination to be empty or absent). A kill mid-write
    # here can only ever leave the *previous* final files in place (if any)
    # or a same-named .tmp file lying around — never a half-written file
    # under the real embeddings.pt/metadata.csv/manifest.json name.
    split_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_tensor(embeddings_tensor, split_dir / EmbeddingCache.EMBEDDINGS_FILENAME)
    _atomic_write_csv(metadata, split_dir / EmbeddingCache.METADATA_FILENAME)

    manifest_tmp = output_root / "manifest.json.tmp"
    current_manifest.to_json(manifest_tmp)
    os.replace(manifest_tmp, output_root / "manifest.json")

    split_manifest_tmp = split_dir / "manifest.json.tmp"
    current_manifest.to_json(split_manifest_tmp)
    os.replace(split_manifest_tmp, split_dir / "manifest.json")

    progress["status"] = "complete"
    progress["last_updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    _atomic_write_json(progress, progress_path)
    _log(log_path, f"Finalized: {n_produced} embeddings merged from {len(video_order)} videos.")

    print(
        f"Cached {n_produced} embeddings for split={args.split} "
        f"model={args.model_name} to {split_dir} "
        f"({len(video_order)} videos; shards preserved under {shards_dir})"
    )


if __name__ == "__main__":
    main()
