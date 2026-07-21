"""
Global seeding — the canonical implementation this project should use
everywhere randomness needs to be controlled. Nothing in the original
Training/ pipeline seeds torch/numpy/CUDA globally (only dataset
construction is seeded, via Embryo_Transition_Dataset's own seed=42); this
fills that gap for anything built on this evaluation framework, without
modifying the original training script.

Deliberately does not call torch.use_deterministic_algorithms(True) — that
can raise for CUDA ops lacking a deterministic kernel, on a combination
that has never been tested against this specific model/loss. Making that
opt-in rather than default-on, exactly as already decided for the
reproducibility PR plan, is the only way to offer stronger determinism
without risking a run that used to complete now crashing instead.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
