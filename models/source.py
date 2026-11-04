"""The common interface for reading tensors, whatever the model format.

Every analysis reads weights through a :class:`TensorSource`: one tensor (or a
contiguous row range of it) at a time, always as float32 on CPU. Formats only
differ in how they find and decode bytes; the analyses never know which one
they are reading.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch

from models.quant import QuantScheme

Rows = tuple[int, int]

# Multimodal checkpoints also hold vision/audio towers whose names collide with
# the language model's ("...layers.0.self_attn.q_proj.weight").
SKIP_PARTS: tuple[str, ...] = ("vision", "visual", "audio", "image", "mm_projector")


class ModelError(Exception):
    """A model cannot be located, recognised or read."""


class UnsupportedModel(ModelError):
    """A model was recognised, but its format is not supported."""


class TensorSource(ABC):
    """Read-only, per-tensor access to one model's weights."""

    def __init__(self, ref: str) -> None:
        self.ref = ref

    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Every tensor name, in storage order."""

    @abstractmethod
    def shape(self, name: str) -> list[int]:
        """Shape of tensor ``name``, without loading it."""

    @abstractmethod
    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Load tensor ``name`` (or rows ``rows[0]:rows[1]``) as float32 on CPU."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """The model's ``config.json`` content (empty when there is none)."""

    def has(self, name: str) -> bool:
        """Whether tensor ``name`` exists."""
        return name in self.names()

    def quantization(self, name: str) -> QuantScheme | None:
        """How tensor ``name`` is stored, when it is quantized (None for plain floats)."""
        return None

    def aliases(self, name: str) -> tuple[str, ...]:
        """Every Hugging Face spelling of tensor ``name`` (formats such as GGUF map to several)."""
        return (name,)

    def find(self, suffix: str) -> str | None:
        """Return the shortest tensor name ending with ``suffix``, ignoring non-text towers."""
        hits = [
            name
            for name in self.names()
            if (name == suffix or name.endswith(f".{suffix}")) and not any(part in name for part in SKIP_PARTS)
        ]
        return min(hits, key=len) if hits else None

    def resolve(self, name: str) -> str | None:
        """Find ``name`` here, tolerating a missing or extra ``model.`` prefix."""
        candidates = (name, f"model.{name}", name.removeprefix("model."))
        return next((candidate for candidate in candidates if self.has(candidate)), None)


def slice_rows(tensor: torch.Tensor, rows: Rows | None) -> torch.Tensor:
    """Return ``tensor[rows[0]:rows[1]]``, or the tensor itself when ``rows`` is None."""
    return tensor if rows is None else tensor[rows[0] : rows[1]]


def canonical_ref(ref: str) -> str:
    """A model reference in canonical form: local paths absolute with case and separators normalised.

    Two spellings of one folder (forward or back slashes, any case, trailing separator) compare
    equal; Hub ids are unchanged.
    """
    return os.path.normcase(os.path.abspath(ref)) if os.path.exists(ref) else ref
