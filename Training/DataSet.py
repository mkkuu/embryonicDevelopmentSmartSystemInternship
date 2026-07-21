"""
========================================
EMBRYO ANALYSIS SYSTEM - DATASET CLASS
========================================

This module provides a PyTorch Dataset class for handling embryo development
time-lapse image sequences. It supports both traditional CNN and transformer-based
models with flexible data loading and preprocessing options.

Features:
- Time-lapse image sequence loading with sliding window approach
- Support for multiple focal planes and developmental phases
- Flexible data modes (grayscale sequences vs RGB video)
- Class balancing and data augmentation
- Configurable window sizes and stride patterns
- Integration with PyTorch DataLoader

Data Modes:
- "image_seq": Grayscale sequences for ResNet/GRU models → (T, H, W)
- "video": RGB video format for TimeSformer models → (C, T, H, W)

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import os
import random
import re
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms


class Embryo_Transition_Dataset(Dataset):
    """
    PyTorch Dataset class for embryo development time-lapse sequences.
    
    This dataset handles loading and preprocessing of embryo development
    time-lapse images with support for different model architectures and
    experimental configurations.
    
    Attributes:
        phase_col (str): Column name for embryo developmental phases
        path_col (str): Column name for image file paths
        window_size (int): Number of frames per sequence
        stride (int): Sliding window stride for sequence generation
        FocalType (str): Focal plane type (e.g., 'F0', 'F1', etc.)
        UsedFor (str): Dataset split ('Train', 'Val', 'Test')
        NumberOfVideos (int): Maximum number of videos to use
        balance_flags (bool): Whether to balance class distribution
        multiple_phases (bool): Allow multiple phases within a window
        mode (str): Data format mode ('image_seq' or 'video')
        
    Example:
        >>> dataset = Embryo_Transition_Dataset(
        ...     window_size=8,
        ...     stride=1,
        ...     UsedFor="Train",
        ...     mode="image_seq"
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # (8, 224, 224) for image_seq mode
    """
    
    def __init__(
        self,
        transform=None,
        phase_col="Phase",
        path_col="Path",
        seed=42,
        FocalType="F0",
        window_size=8,
        stride=1,
        UsedFor="None",
        NumberOfVideos=10,
        balance_flags=False,
        multiple_phases=False,
        mode="image_seq",
    ):
        """
        Initialize the embryo transition dataset.
        
        Args:
            transform (callable, optional): Torchvision transforms to apply
            phase_col (str): Column name for embryo phases in CSV (default: "Phase")
            path_col (str): Column name for image paths in CSV (default: "Path")
            seed (int): Random seed for reproducibility (default: 42)
            FocalType (str): Dataset focal type filter (default: "F0")
            window_size (int): Number of frames per sequence (default: 8)
            stride (int): Sliding window stride (default: 1)
            UsedFor (str): Dataset split filter (default: "None")
            NumberOfVideos (int): Limit number of videos, 0 = use all (default: 10)
            balance_flags (bool): Balance class 0/1 sequences (default: False)
            multiple_phases (bool): Allow >2 phases inside a window (default: False)
            mode (str): Data format mode (default: "image_seq")
                - "image_seq": Grayscale sequences for ResNet/GRU → (T, H, W)
                - "video": RGB video format for TimeSformer → (C, T, H, W)
        
        The dataset automatically loads and processes embryo time-lapse sequences
        based on the provided parameters, creating sliding windows of consecutive
        frames for temporal analysis.
        """
        super().__init__()
        self.phase_col = phase_col
        self.path_col = path_col
        self.window_size = window_size
        self.stride = stride
        self.multiple_phases = multiple_phases
        self.mode = mode

        # ---------------- Load Dataset ----------------
        csv_dataset_path = os.path.join("..", "Data", "Splits", f"{FocalType}.csv")
        df_dataset = pd.read_csv(csv_dataset_path)
        df_dataset = df_dataset[df_dataset["UsedFor"] == UsedFor]
        df_dataset = df_dataset[df_dataset["Phase"] != "tHB"]  # Exclude tHB phase
        if NumberOfVideos != 0:
            selected_ids = (
                df_dataset["Video_name"]
                .drop_duplicates()
                .sample(n=NumberOfVideos, random_state=seed)
            )
            df_dataset = df_dataset[df_dataset["Video_name"].isin(selected_ids)]

        print(
            f"Videos in {UsedFor} fold: {len(df_dataset['Video_name'].unique())} videos"
        )
        print(f"{len(df_dataset['Video_name'].unique())} Videos are being used")
        print(f"Number of samples: {len(df_dataset)}")

        df_dataset = df_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.data_frame = df_dataset

        # Default transform
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        )

        # ---------------- Phase Setup ----------------
        chronological_phases = [
            "tPB2",
            "tPNa",
            "tPNf",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9+",
            "tM",
            "tSB",
            "tB",
            "tEB",
        ]
        present_phases = [
            p for p in chronological_phases if p in self.data_frame[phase_col].unique()
        ]
        self.phase_labels = present_phases
        self.phase_to_index = {
            label: index for index, label in enumerate(self.phase_labels)
        }
        print("Phase to Index Mapping:", self.phase_to_index)

        # ---------------- Build Sequences ----------------
        self.video_sequences = []
        self._create_sequences()

        if balance_flags:
            self._balance_flags()

        print(f"\n Total sequences: {len(self.video_sequences)}")
        self._print_flag_consistency()
        self._print_transition_matrix()

    def _create_sequences(self):
        grouped = defaultdict(list)
        for _, row in self.data_frame.iterrows():
            video = row["Video_name"]
            identifier = row["Identifier"]
            path = row["Path"]
            phase = row[self.phase_col]
            grouped[video].append(
                {"identifier": identifier, "path": path, "phase": phase}
            )

        for video, frames in grouped.items():
            sorted_frames = sorted(
                frames,
                key=lambda x: int(
                    re.findall(r"\d+", x["identifier"].split("_")[-1])[0]
                ),
            )
            for start in range(
                0, len(sorted_frames) - self.window_size + 1, self.stride
            ):
                sequence = sorted_frames[start : start + self.window_size]
                if not self.multiple_phases:
                    unique_phases = list(set([frame["phase"] for frame in sequence]))
                    if len(unique_phases) > 2:
                        continue
                    if len(unique_phases) == 2:
                        phase_indices = [
                            self.phase_to_index[phase] for phase in unique_phases
                        ]
                        phase_indices.sort()
                        if phase_indices[1] != phase_indices[0] + 1:
                            continue
                self.video_sequences.append(sequence)

    def _balance_flags(self):
        flag_0_seqs, flag_1_seqs = [], []
        for seq in self.video_sequences:
            labels = [self.phase_to_index[frame["phase"]] for frame in seq]
            if labels[0] == labels[-1]:
                flag_0_seqs.append(seq)
            else:
                flag_1_seqs.append(seq)

        n = min(len(flag_0_seqs), len(flag_1_seqs))
        random.seed(42)
        self.video_sequences = random.sample(flag_0_seqs, n) + random.sample(
            flag_1_seqs, n
        )
        random.shuffle(self.video_sequences)

    def _print_flag_consistency(self):
        flag_0, flag_1 = 0, 0
        for seq in self.video_sequences:
            labels = [self.phase_to_index[frame["phase"]] for frame in seq]
            if labels[0] == labels[-1]:
                flag_0 += 1
            else:
                flag_1 += 1
        print(f" Flag=0 (same start/end phase): {flag_0}")
        print(f" Flag=1 (different start/end phase): {flag_1}")

    def _print_transition_matrix(self):
        num_phases = len(self.phase_labels)
        transition_matrix = [[0 for _ in range(num_phases)] for _ in range(num_phases)]
        for sequence in self.video_sequences:
            first_phase = self.phase_to_index[sequence[0]["phase"]]
            last_phase = self.phase_to_index[sequence[-1]["phase"]]
            transition_matrix[first_phase][last_phase] += 1
        df_matrix = pd.DataFrame(
            transition_matrix, index=self.phase_labels, columns=self.phase_labels
        )
        print("\n Phase Transition Matrix (Start ➝ End):")
        print(df_matrix)

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        sequence = self.video_sequences[idx]
        images_seq, labels, identifiers = [], [], []

        for frame in sequence:
            img_path, phase = frame["path"], frame["phase"]
            try:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"File not found: {img_path}")
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None

            images_seq.append(image)
            labels.append(self.phase_to_index[phase])
            identifiers.append(frame["identifier"])

        # ---------------- MODE SWITCHING ----------------
        if self.mode == "video":
            # Keep as (T, 3, H, W)
            images_seq = torch.stack(images_seq, dim=0)

        elif self.mode == "image_seq":
            # grayscale, then squeeze channel → (T, H, W)
            images_seq = [im.mean(dim=0, keepdim=True) for im in images_seq]  # (1,H,W)
            images_seq = torch.stack(images_seq, dim=0).squeeze(1)  # (T,H,W)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # ---------------- LABELS ----------------
        if len(identifiers) != self.window_size:
            raise ValueError(
                f"Identifier count ({len(identifiers)}) != window size ({self.window_size})"
            )

        first_frame_phase = torch.tensor(
            self.phase_to_index[sequence[0]["phase"]], dtype=torch.long
        )
        last_frame_phase = torch.tensor(
            self.phase_to_index[sequence[-1]["phase"]], dtype=torch.long
        )
        consistency_flag = 0 if labels[0] == labels[-1] else 1

        return (
            images_seq,
            torch.tensor(consistency_flag, dtype=torch.long),
            first_frame_phase,
            last_frame_phase,
        )
