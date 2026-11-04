"""Weight updates between a base model and a tuned model, one tensor at a time.

Every analysis of "what changed" consumes :func:`iter_deltas`. When the tuned
model is a LoRA adapter applied to that same base, only the adapted tensors are
visited and ΔW is rebuilt exactly from its low-rank factors (no subtraction
round-off, and the factors are kept for a cheap exact SVD). Otherwise every
tensor present in both models is compared.

Failures are per tensor: a missing or mismatched tensor is recorded in the
:class:`DeltaLog` and the iteration continues.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial

import torch

from models.lora import AdaptedSource, LoraPair
from models.source import TensorSource, canonical_ref
from progress import CancelToken

MAX_LOGGED_NOTES = 200


@dataclass(frozen=True)
class LowRankFactors:
    """ΔW = scale * B @ A (embeddings: (scale * B @ A).T), kept for an exact cheap SVD."""

    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scale: float
    transposed: bool


@dataclass(frozen=True)
class Delta:
    """One compared tensor: the base weight and its update."""

    name: str
    base: torch.Tensor
    delta: torch.Tensor
    factors: LowRankFactors | None = None
    tuned_name: str = ""  # the tensor's name in the tuned model, when it differs

    @property
    def source_name(self) -> str:
        """Name of the tensor in the tuned model."""
        return self.tuned_name or self.name


@dataclass
class DeltaLog:
    """What happened while iterating: counts and per-tensor notes."""

    visited: int = 0
    unmatched: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        """Record one note, keeping the log bounded."""
        if len(self.notes) < MAX_LOGGED_NOTES:
            self.notes.append(message)


def iter_deltas(
    base: TensorSource,
    tuned: TensorSource,
    log: DeltaLog,
    token: CancelToken | None = None,
) -> Iterator[Delta]:
    """Yield the update of every comparable tensor of ``tuned`` against ``base``.

    Args:
        base: the reference model.
        tuned: the model to compare (full weights, or an adapter on ``base``).
        log: receives counts and notes about skipped tensors.
        token: checked between tensors, so a front-end can cancel.

    Yields:
        One :class:`Delta` per tensor present in both models with the same shape.
    """
    if isinstance(tuned, AdaptedSource) and _same_model(tuned.base, base):
        yield from _iter_adapter_deltas(tuned, log, token)
    else:
        yield from _iter_full_deltas(base, tuned, log, token)


def total_deltas(base: TensorSource, tuned: TensorSource) -> int:
    """Number of tensors :func:`iter_deltas` will visit (for progress bars)."""
    result = len(tuned.names())
    if isinstance(tuned, AdaptedSource) and _same_model(tuned.base, base):
        result = len(tuned.adapted_pairs()) + len(tuned.saved_names())
    return result


def _iter_adapter_deltas(tuned: AdaptedSource, log: DeltaLog, token: CancelToken | None) -> Iterator[Delta]:
    """Exact low-rank deltas of an adapter, then its fully saved modules."""
    for name in tuned.unmatched:
        log.unmatched += 1
        log.note(f"adapter tensor without base counterpart: {name}")
    for name, pair in tuned.adapted_pairs().items():
        if token is not None:
            token.check()
        log.visited += 1
        delta = _safe(log, name, partial(_adapter_delta, tuned, name, pair))
        if delta is not None:
            yield delta
    for name in tuned.saved_names():
        if token is not None:
            token.check()
        log.visited += 1
        delta = _safe(log, name, partial(_full_delta, tuned.base, tuned, name, name))
        if delta is not None:
            log.note(f"module saved in full by the adapter: {name}")
            yield delta


def _iter_full_deltas(
    base: TensorSource, tuned: TensorSource, log: DeltaLog, token: CancelToken | None
) -> Iterator[Delta]:
    """Deltas of every tensor of ``tuned`` that ``base`` also has, with the same shape."""
    for tuned_name in tuned.names():
        if token is not None:
            token.check()
        log.visited += 1
        base_name = next(filter(None, (base.resolve(alias) for alias in tuned.aliases(tuned_name))), None)
        if base_name is None:
            log.unmatched += 1
            log.note(f"only in tuned model: {tuned_name}")
            continue
        delta = _safe(log, tuned_name, partial(_full_delta, base, tuned, base_name, tuned_name))
        if delta is not None:
            yield delta


def _adapter_delta(tuned: AdaptedSource, name: str, pair: LoraPair) -> Delta:
    """Base weight and exact ΔW of one adapted tensor."""
    adapter = tuned.adapter
    factors = LowRankFactors(
        lora_a=adapter.tensor(pair.key_a),
        lora_b=adapter.tensor(pair.key_b),
        scale=adapter.config.scale,
        transposed=pair.embedding,
    )
    return Delta(name=name, base=tuned.base.load(name), delta=adapter.delta(pair), factors=factors)


def _full_delta(base: TensorSource, tuned: TensorSource, base_name: str, tuned_name: str) -> Delta:
    """Base weight and W1 - W0.

    Raises:
        ShapeMismatch: when the two tensors have different shapes.
    """
    w0, w1 = base.load(base_name), tuned.load(tuned_name)
    if w0.shape != w1.shape:
        raise ShapeMismatch(f"shape mismatch {base_name}: {tuple(w0.shape)} vs {tuple(w1.shape)}")
    return Delta(name=base_name, base=w0, delta=w1 - w0, tuned_name=tuned_name)


class ShapeMismatch(ValueError):
    """Two tensors with the same name have different shapes."""


def _safe(log: DeltaLog, name: str, build: Callable[[], Delta]) -> Delta | None:
    """Run ``build`` and return its delta; record a note instead of raising."""
    result = None
    try:
        result = build()
    except ShapeMismatch as error:
        log.unmatched += 1
        log.note(str(error))
    except Exception as error:  # noqa: BLE001 - one bad tensor must not abort the analysis
        log.unmatched += 1
        log.note(f"{name}: {error.__class__.__name__}: {error}")
    return result


def _same_model(first: TensorSource, second: TensorSource) -> bool:
    """Whether two sources read the same model (same object, Hub id or local folder)."""
    return first is second or canonical_ref(first.ref) == canonical_ref(second.ref)
