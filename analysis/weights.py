"""How much each tensor moved between a base model and a tuned one.

Port of ``compare_weights.py``. The tuned model is either a full model (every
tensor present in both is compared) or a LoRA adapter on that same base (only
the adapted tensors and the fully saved modules are visited, with ΔW rebuilt
exactly from its factors). :func:`~models.deltas.iter_deltas`
makes that choice; this module only measures and aggregates.

Differences from the reference script:

* modules are grouped with :func:`stats.module_of`, so ``input_layernorm`` and
  ``post_attention_layernorm`` stay distinct (the reference merged every norm
  under ``norm`` through a regex);
* a tensor whose base is all zeros has no relative change (``None``); it sorts
  last instead of being a NaN that scrambles the order;
* "changed" means ``‖ΔW‖ ≥ CHANGE_THRESHOLD`` rather than ``rel_change > 1e-6``,
  so an update of an all-zero tensor still counts as a change;
* adapter mode also reports embedding LoRA factors and modules saved in full;
* a tensor that cannot be read or measured is noted, never fatal;
* quantized models (GGUF, bitsandbytes) are read dequantized, and every tensor's
  relative change is read against its quantization noise floor: the change
  quantization alone would produce (``beyond_noise``: more than
  ``stats.NOISE_MARGIN`` floors).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from analysis import stats
from models.deltas import Delta, DeltaLog, iter_deltas, total_deltas
from models.source import TensorSource
from progress import CancelToken, ProgressSink, Stage, null_sink, report
from results import ResultMeta

ANALYSIS = "weights"


@dataclass
class TensorChange:
    """Change metrics of one compared tensor."""

    name: str
    layer: int  # -1 for tensors outside the transformer blocks
    module: str
    role: str  # writer | reader | embedding | other
    shape: list[int]
    rel_change: float | None  # ‖ΔW‖ / ‖W‖, None when ‖W‖ = 0
    cosine: float  # cos(W, W + ΔW)
    max_abs_diff: float
    frac_changed: float
    delta_rank: int | None  # exact numerical rank of a low-rank (LoRA) update, else None
    changed: bool
    quantization: str | None = None  # storage of the compared tensors, e.g. "Q4_K" or "nf4 + Q8_0"
    noise_floor: float | None = None  # relative change quantization alone would produce
    floor_exact: bool | None = None  # False when the floor is estimated (llama.cpp K/IQ types)
    beyond_noise: bool | None = None  # rel_change > NOISE_MARGIN * noise_floor


@dataclass
class GroupChange:
    """Mean relative change of the changed tensors sharing a module or a layer."""

    key: str | int
    mean_rel_change: float
    tensors: int


@dataclass
class WeightDiffResult:
    """Every tensor's change, with counts and per-module / per-layer means."""

    meta: ResultMeta
    compared: int
    changed: int
    identical: int
    unmatched: int
    tensors: list[TensorChange]  # by relative change, largest first; None last
    by_module: list[GroupChange]  # largest mean first
    by_layer: list[GroupChange]  # by layer number, non-layer tensors as -1
    beyond_noise: int | None = None  # changed tensors above their quantization noise (quantized comparisons)


def compare_weights(
    base: TensorSource,
    tuned: TensorSource,
    *,
    inputs: dict[str, str],
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
) -> WeightDiffResult:
    """Measure how much every comparable tensor of ``tuned`` moved from ``base``.

    Args:
        base: the reference model.
        tuned: a full model, or an adapter applied to ``base``.
        inputs: labels of the compared models, recorded in the result metadata.
        sink: receives one progress event per tensor.
        token: checked between tensors; raises ``JobCancelled`` once cancelled.

    Returns:
        Per-tensor metrics and their aggregates. Unreadable tensors are counted
        as unmatched and described in ``meta.notes``; measurement failures go to
        ``meta.warnings``.
    """
    meta = ResultMeta(analysis=ANALYSIS, inputs=dict(inputs), parameters={"change_threshold": stats.CHANGE_THRESHOLD})
    log = DeltaLog()
    total = total_deltas(base, tuned)
    changes: list[TensorChange] = []
    for delta in iter_deltas(base, tuned, log, token):
        report(sink, Stage.WEIGHTS, delta.name, log.visited, total)
        change = _safe_measure(delta, base, tuned, meta.warnings)
        if change is not None:
            changes.append(change)
    report(sink, Stage.WEIGHTS, f"Compared {len(changes)} tensors", log.visited, total)
    meta.notes.extend(log.notes)
    meta.warnings.extend(_quantization_warnings(changes))
    return _build_result(meta, changes, log.unmatched)


def _build_result(meta: ResultMeta, changes: list[TensorChange], unmatched: int) -> WeightDiffResult:
    """Sort the tensors and aggregate the changed ones."""
    changed = [change for change in changes if change.changed]
    return WeightDiffResult(
        meta=meta,
        compared=len(changes),
        changed=len(changed),
        identical=len(changes) - len(changed),
        unmatched=unmatched,
        tensors=sorted(changes, key=_rel_change_order),
        by_module=sorted(_group(changed, lambda change: change.module), key=lambda group: -group.mean_rel_change),
        by_layer=sorted(_group(changed, lambda change: change.layer), key=lambda group: int(group.key)),
        beyond_noise=_count_beyond_noise(changes),
    )


def _count_beyond_noise(changes: list[TensorChange]) -> int | None:
    """Number of tensors above their noise floor, or None when nothing was quantized."""
    flags = [change.beyond_noise for change in changes if change.beyond_noise is not None]
    return sum(flags) if flags else None


def _quantization_warnings(changes: list[TensorChange]) -> list[str]:
    """One warning describing the quantization the comparison had to see through."""
    schemes = Counter(change.quantization for change in changes if change.quantization)
    estimated = any(change.floor_exact is False for change in changes)
    result = []
    if schemes:
        listed = ", ".join(f"{name} x{count}" for name, count in schemes.most_common())
        result.append(
            f"Quantized weights ({listed}): relative changes include quantization error; read them against "
            f"each tensor's noise floor ('beyond noise' = more than {stats.NOISE_MARGIN:g} floors)."
            + (" Some floors are estimated (llama.cpp K/IQ types have no Python quantizer)." if estimated else "")
        )
    return result


def _rel_change_order(change: TensorChange) -> tuple[bool, float]:
    """Sort key: largest relative change first, tensors without one last."""
    missing = change.rel_change is None
    return missing, -(change.rel_change or 0.0)


def _group(changes: Iterable[TensorChange], key_of: Callable[[TensorChange], str | int]) -> list[GroupChange]:
    """Mean relative change per group, over tensors that have one."""
    values: defaultdict[str | int, list[float]] = defaultdict(list)
    for change in changes:
        if change.rel_change is not None:
            values[key_of(change)].append(change.rel_change)
    return [
        GroupChange(key=key, mean_rel_change=sum(items) / len(items), tensors=len(items))
        for key, items in values.items()
    ]


def _safe_measure(delta: Delta, base: TensorSource, tuned: TensorSource, warnings: list[str]) -> TensorChange | None:
    """Measure one tensor; record a warning instead of raising."""
    result = None
    try:
        result = _measure(delta, base, tuned)
    except Exception as error:  # noqa: BLE001 - one bad tensor must not abort the analysis
        warnings.append(f"{delta.name}: {error.__class__.__name__}: {error}")
    return result


def _measure(delta: Delta, base: TensorSource, tuned: TensorSource) -> TensorChange:
    """Change metrics of one tensor, with its quantization noise floor when relevant."""
    change = stats.change_stats(delta.base, delta.delta)
    floor = stats.quantization_floor(delta, base, tuned)
    return TensorChange(
        name=delta.name,
        layer=stats.layer_of(delta.name),
        module=stats.module_of(delta.name),
        role=stats.role_of(delta.name),
        shape=list(delta.base.shape),
        rel_change=change.rel_change,
        cosine=change.cosine,
        max_abs_diff=change.max_abs_diff,
        frac_changed=change.frac_changed,
        delta_rank=_delta_rank(delta),
        changed=change.changed,
        quantization=_quantization_label(delta, base, tuned) if floor is not None else None,
        noise_floor=floor.rel_error if floor is not None else None,
        floor_exact=floor.exact if floor is not None else None,
        beyond_noise=stats.beyond_noise(change.rel_change, floor),
    )


def _quantization_label(delta: Delta, base: TensorSource, tuned: TensorSource) -> str:
    """Storage of the two compared tensors, quantized ones only: ``"Q4_K"``, ``"nf4 + Q8_0"``."""
    schemes = (base.quantization(delta.name), tuned.quantization(delta.source_name))
    return " + ".join(scheme.name for scheme in schemes if scheme is not None and not scheme.lossless)


def _delta_rank(delta: Delta) -> int | None:
    """Exact rank of a low-rank update (cheap through its factors); None for a full update."""
    return stats.leading_direction(delta).rank if delta.factors is not None else None
