"""
Embedding extraction by composition around Training/ModelBuilder.get_model().

This module does not modify ModelBuilder.py. It reaches into the model
objects ModelBuilder.get_model() already returns, using only their existing
public surface:

  - resnet18 (raw timm model): timm already exposes `forward_features` and
    `forward_head(..., pre_logits=True)` publicly; we call them directly.
  - timesformer (TimeSformerWrapper): TimeSformerWrapper.forward() returns
    only classification logits, discarding hidden states. The underlying
    HuggingFace model is reachable as the public attribute `wrapper.model`,
    and is used directly here instead of going through
    TimeSformerWrapper.forward().

The exact HuggingFace hidden-state API is version-sensitive and has not
been empirically verified against a running installation as of writing
this module (no GPU/ML environment was available where this was written).
Extraction is written defensively: it tries the expected primary path,
validates the output shape, and raises an informative error rather than
silently returning a wrong or garbage embedding if the assumption doesn't
hold. Verifying and, if needed, correcting `_extract_timesformer` against
the real installed `transformers` version is required before trusting any
TimeSformer embedding produced here — see the module-level TODO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ModelBuilder import TimeSformerWrapper
except ImportError as e:  # pragma: no cover - environment sanity check
    raise ImportError(
        "embeddings.extractor must be run with Training/ on sys.path (it "
        "imports TimeSformerWrapper from ModelBuilder.py, a top-level "
        "module, not a package). Run scripts from within Training/, e.g. "
        "`python -m embeddings.build_cache ...`."
    ) from e


class EmbeddingExtractionError(RuntimeError):
    """Raised when the composed extraction path does not produce the
    expected output — never returned silently as a wrong embedding."""


class EmbeddingExtractor:
    """
    Extracts a pre-classification embedding from a model built by
    Training/ModelBuilder.get_model(), without modifying that model or
    ModelBuilder.py.

    Parameters
    ----------
    model : nn.Module
        An object returned by ModelBuilder.get_model(). Must be either a
        raw timm model (resnet18 branch) or a TimeSformerWrapper instance
        (timesformer branch).
    device : torch.device
        Device the model's parameters already live on. Input batches are
        moved to this device before the forward pass.

    Notes
    -----
    This class does not manage model.eval()/train() mode or the
    torch.no_grad() context — callers are expected to have already put the
    model in eval mode and to call extract() inside torch.no_grad().
    Extraction is a pure function of (model, batch); mode and gradient
    management are a caller concern.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._kind = self._identify_model_kind(model)

    @staticmethod
    def _identify_model_kind(model: nn.Module) -> str:
        if isinstance(model, TimeSformerWrapper):
            return "timesformer"
        if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
            return "resnet18"
        raise EmbeddingExtractionError(
            f"Don't know how to extract embeddings from a model of type "
            f"{type(model).__name__}. Expected either a TimeSformerWrapper "
            f"instance or a timm model exposing forward_features/"
            f"forward_head. If ModelBuilder.get_model() has grown a new "
            f"branch, EmbeddingExtractor needs a matching case added here — "
            f"do not guess, add an explicit branch."
        )

    def extract(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : torch.Tensor
            (B, T, H, W) for resnet18/image_seq mode, or (B, T, 3, H, W)
            for timesformer/video mode — matching whatever
            Embryo_Transition_Dataset was configured to produce.

        Returns
        -------
        torch.Tensor
            (B, D) — one embedding vector per sequence in the batch,
            finite-checked before being returned.
        """
        batch = batch.to(self.device)
        if self._kind == "resnet18":
            embedding = self._extract_resnet18(batch)
        else:
            embedding = self._extract_timesformer(batch)
        self._validate_shape(embedding, batch.shape[0])
        return embedding

    def _extract_resnet18(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(batch)
        return self.model.forward_head(features, pre_logits=True)

    def _extract_timesformer(self, batch: torch.Tensor) -> torch.Tensor:
        # Primary path: the inner HF backbone, bypassing the classifier
        # entirely. This mirrors exactly what
        # TimesformerForVideoClassification's own classifier consumes
        # (the CLS token of the backbone's last_hidden_state).
        backbone = getattr(self.model.model, "timesformer", None)
        if backbone is not None:
            outputs = backbone(pixel_values=batch)
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise EmbeddingExtractionError(
                    "wrapper.model.timesformer(...) did not return a "
                    "last_hidden_state — the installed `transformers` "
                    "version's TimesformerModel output format does not "
                    "match what this extractor expects. Inspect `outputs` "
                    "directly and update _extract_timesformer accordingly "
                    "before trusting any embedding from this branch."
                )
            return last_hidden_state[:, 0]  # CLS token

        # Fallback path: no `.timesformer` attribute found on the inner HF
        # model (API drift). Go through the full classification model with
        # output_hidden_states=True instead.
        outputs = self.model.model(pixel_values=batch, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise EmbeddingExtractionError(
                "Neither wrapper.model.timesformer nor "
                "wrapper.model(..., output_hidden_states=True).hidden_states "
                "produced a usable representation. The installed "
                "`transformers` API for TimesformerForVideoClassification "
                "has likely changed since this module was written — the "
                "extraction logic needs re-verifying against the actual "
                "installed version, not silently patched around."
            )
        return hidden_states[-1][:, 0]

    @staticmethod
    def _validate_shape(embedding: torch.Tensor, expected_batch: int) -> None:
        if embedding.dim() != 2 or embedding.shape[0] != expected_batch:
            raise EmbeddingExtractionError(
                f"Extracted embedding has shape {tuple(embedding.shape)}, "
                f"expected (B={expected_batch}, D). Extraction logic likely "
                f"picked the wrong tensor from the model's output — do not "
                f"proceed with caching until this is understood."
            )
        if not torch.isfinite(embedding).all():
            raise EmbeddingExtractionError(
                "Extracted embedding contains NaN or Inf values."
            )
