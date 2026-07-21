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

Usage
-----
    cd Training
    python -m embeddings.build_cache \\
        --model_name resnet18 \\
        --checkpoint ../Results/resnet18/best_model.pth \\
        --window_size 8 --stride 1 --focal_type F0 --image_size 224 \\
        --split Val \\
        --output_root ../Embeddings

Run once per split (Train/Val/Test) you need cached.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from DataSet import Embryo_Transition_Dataset
from ModelBuilder import get_model

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
    p.add_argument("--output_root", default="../Embeddings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
        NumberOfVideos=0,
        mode=mode,
    )
    indexed_dataset = _IndexedDataset(dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # order preservation matters — window_start below
        num_workers=args.num_workers,
        collate_fn=_skip_none_collate,
    )

    model = load_model(args)
    device = torch.device(args.device)
    extractor = EmbeddingExtractor(model, device)

    all_embeddings = []
    rows = []
    running_index = 0
    skipped = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue  # every item in this batch failed to load
            indices, images, consistency_flags, first_phases, last_phases = batch
            embeddings = extractor.extract(images).cpu()
            all_embeddings.append(embeddings)

            for i in range(embeddings.shape[0]):
                original_idx = int(indices[i])
                sequence = dataset.video_sequences[original_idx]
                # video_name is not stored per-frame in
                # Embryo_Transition_Dataset (it's only a grouping key
                # during _create_sequences, discarded afterward) — derive
                # it from the identifier naming convention preProcess.py
                # already establishes: "{patient}_Image_{i}.jpeg".
                video_name = sequence[0]["identifier"].rsplit("_Image_", 1)[0]
                rows.append(
                    {
                        "embedding_index": running_index,
                        "video_name": video_name,
                        "window_start": original_idx,
                        "consistency_flag": int(consistency_flags[i]),
                        "first_frame_phase": int(first_phases[i]),
                        "last_frame_phase": int(last_phases[i]),
                    }
                )
                running_index += 1

    n_requested = len(dataset)
    n_produced = running_index
    skipped = n_requested - n_produced
    if skipped:
        print(
            f"WARNING: {skipped}/{n_requested} windows failed to load "
            f"(image read errors) and were skipped — see stdout above for "
            f"individual file paths."
        )

    if n_produced == 0:
        raise RuntimeError(
            f"Zero embeddings were produced for split={args.split} — "
            f"every window failed to load, or the split is empty. Refusing "
            f"to write an empty cache."
        )

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    metadata = pd.DataFrame(rows)

    output_root = Path(args.output_root) / args.model_name
    EmbeddingCache.save(embeddings_tensor, metadata, output_root / args.split.lower())

    manifest = build_manifest(args, mode)
    # Shared, model-level copy (what EmbeddingDataset reads) — note this
    # path is shared across all three splits and gets overwritten by
    # whichever split is built last. The per-split copy below exists
    # specifically so that later, independent validation (embeddings.validate)
    # can detect if train/val/test were ever actually built with different
    # configurations, which the shared copy alone cannot reveal.
    manifest.to_json(output_root / "manifest.json")
    manifest.to_json(output_root / args.split.lower() / "manifest.json")

    print(
        f"Cached {n_produced} embeddings for split={args.split} "
        f"model={args.model_name} to {output_root / args.split.lower()}"
    )


if __name__ == "__main__":
    main()
