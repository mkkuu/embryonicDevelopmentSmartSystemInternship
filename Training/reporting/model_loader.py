"""
Loads the frozen Semi-HMM reference model exactly once and validates its
configuration against the project's own established reference
(docs/SCIENTIFIC_REPORT.md sec 16: dmax=268, negative_binomial, unweighted
emission, at Results/evaluation/semi_hmm_weekend_phaseF/model/). Never
refits, never writes a new checkpoint -- SemiHMMModel.load() only ever
reconstructs already-fitted state from state.json (verified by reading
semi_hmm.py's load() this session: it reads state.json and reassigns
already-computed arrays, no fit() call anywhere in that path).

"Refuse silently" (the task's own phrasing) is interpreted here as
"refuse cleanly, without partial global state" -- NOT "refuse without
telling anyone", which would contradict this project's own consistent
discipline (see e.g. hmm.py's module docstring: "explicit fallback, not a
silent no-op or a crash"). An incompatible model raises a clear,
typed ModelIncompatibleError; nothing about the rejection itself is
hidden from the caller.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Training/ on sys.path,
# same convention as Tests/conftest.py, so this module works whether imported as
# `reporting.model_loader` (Training/ already on sys.path) or run standalone.

from evaluation.models.semi_hmm import SemiHMMModel  # noqa: E402

REFERENCE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / \
    "Results" / "evaluation" / "semi_hmm_weekend_phaseF" / "model"

EXPECTED_N_STATES = 15
EXPECTED_DMAX = 268
EXPECTED_DURATION_FAMILY = "negative_binomial"
EXPECTED_EMBEDDING_DIM = 512
EXPECTED_CLASS_WEIGHT = None  # unweighted emission -- the reference established in
# docs/SCIENTIFIC_REPORT.md sec 14 after the class-reweighting branch was closed


class ModelIncompatibleError(Exception):
    """Raised when a loaded checkpoint's configuration does not match the
    expected reference -- never silently substituted or partially used."""


def _validate(model: SemiHMMModel) -> None:
    problems = []
    if model.n_states != EXPECTED_N_STATES:
        problems.append(f"n_states={model.n_states}, expected {EXPECTED_N_STATES}")
    if model.dmax != EXPECTED_DMAX:
        problems.append(f"dmax={model.dmax}, expected {EXPECTED_DMAX}")
    if model.duration_family != EXPECTED_DURATION_FAMILY:
        problems.append(f"duration_family={model.duration_family!r}, expected {EXPECTED_DURATION_FAMILY!r}")
    if model.logreg_class_weight != EXPECTED_CLASS_WEIGHT:
        problems.append(
            f"logreg_class_weight={model.logreg_class_weight!r}, expected {EXPECTED_CLASS_WEIGHT!r} "
            f"(the class-reweighting branch was tested exhaustively and closed -- "
            f"see docs/SCIENTIFIC_REPORT.md sec 14 -- this API only ever serves the unweighted reference)"
        )
    emission_hmm = getattr(model, "_emission_hmm", None)
    scaler = getattr(emission_hmm, "_scaler", None) if emission_hmm is not None else None
    if scaler is None or not hasattr(scaler, "mean_"):
        problems.append("composed emission HMM has no fitted StandardScaler -- model was not actually fit()")
    else:
        emb_dim = int(scaler.mean_.shape[0])
        if emb_dim != EXPECTED_EMBEDDING_DIM:
            problems.append(f"embedding_dim={emb_dim}, expected {EXPECTED_EMBEDDING_DIM}")
    if not model._fitted:
        problems.append("model._fitted is False -- checkpoint did not save a completed fit")

    if problems:
        raise ModelIncompatibleError(
            f"Loaded model at {REFERENCE_MODEL_PATH} does not match the expected reference "
            f"configuration:\n  - " + "\n  - ".join(problems) +
            f"\nRefusing to serve this model. Regenerate/select the correct checkpoint rather than "
            f"proceeding with a mismatched one."
        )


_MODEL_CACHE: Optional[SemiHMMModel] = None


def get_model(path: Path = REFERENCE_MODEL_PATH, force_reload: bool = False) -> SemiHMMModel:
    """Loads the frozen Semi-HMM once and caches it for the process
    lifetime (Phase 10's "load once" requirement) -- `force_reload=True`
    is exposed only for tests that need a fresh instance, never used by
    the API itself. Never refits; SemiHMMModel.load() is the only call
    made to the model here."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    if not (path / "state.json").exists():
        raise FileNotFoundError(
            f"No frozen Semi-HMM checkpoint found at {path} (state.json missing). "
            f"This function never trains a new model -- point it at an existing checkpoint."
        )
    model = SemiHMMModel.load(path)
    _validate(model)
    _MODEL_CACHE = model
    return model


def model_configuration(model: SemiHMMModel) -> dict:
    """AVAILABLE fields for schemas.ModelInfo.model_configuration -- every
    key here is a real attribute already set on the loaded model, nothing
    invented."""
    return {
        "n_states": model.n_states,
        "dmax": model.dmax,
        "duration_family": model.duration_family,
        "transition_smoothing_alpha": model.transition_smoothing_alpha,
        "transition_decay_rho": model.transition_decay_rho,
        "logreg_class_weight": model.logreg_class_weight,
        "logreg_C": model.logreg_C,
    }


def model_version_string(model: SemiHMMModel) -> str:
    cfg = model_configuration(model)
    return (
        f"dmax={cfg['dmax']},{cfg['duration_family']},"
        f"alpha={cfg['transition_smoothing_alpha']},rho={cfg['transition_decay_rho']},"
        f"class_weight={cfg['logreg_class_weight']}"
    )
