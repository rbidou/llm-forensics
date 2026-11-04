"""How the entropy of the last layers' weights shifts from a source model to a fine-tune.

Fine-tuning rarely rewrites a model uniformly: it concentrates in the last
transformer blocks and the output head, and a telling signature there is a
change in the *entropy* of the weight matrices. Two complementary readings are
measured for every in-scope tensor and compared base vs tuned:

* **spectral entropy** -- the Shannon entropy of the normalised singular-value
  spectrum ``pᵢ = σᵢ² / Σσ²``. It says how the matrix spreads its energy across
  directions: specialising a model usually concentrates the spectrum (entropy
  falls), while broadening it (entropy rises) points the other way. Rotation and
  scale invariant, and bounded by ``ln(min(shape))``;
* **value-distribution entropy** -- the Shannon entropy of a histogram of the
  weight values over the range shared by the base and the tuned tensor. It
  catches a narrowing or widening of the weight distribution that the spectrum
  may miss (and is all that quantized or very large tensors can offer cheaply).

Scope is the last ``layers`` transformer blocks plus the head (the final norm
and the ``lm_head`` / token-embedding matrix); spectral entropy is skipped on
tensors too large for a full SVD (the vocabulary head), which then carry the
value reading alone.

The headline shift of each metric comes with a null baseline: for the tensors
that received a spectral reading, the same entropy change is measured when the
base is perturbed by a *random* update of the same Frobenius norm as the real
``ΔW`` (seeded). A real shift well beyond that null is structural, not the mere
consequence of the update's size. Quantized comparisons are read against each
tensor's noise floor, exactly as the weight and shift analyses do.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from analysis import stats
from models.arch import Architecture
from models.deltas import Delta, DeltaLog, iter_deltas, total_deltas
from models.source import ModelError, TensorSource
from progress import CancelToken, ProgressSink, Stage, null_sink, report
from results import ResultMeta

ANALYSIS = "entropy"

# The head: the final norm and the output / token-embedding matrix (last-layer region, not a block).
FINAL_NORM_SUFFIXES: tuple[str, ...] = (
    "model.norm.weight",
    "transformer.ln_f.weight",
    "model.final_layernorm.weight",
    "ln_f.weight",
    "final_layernorm.weight",
)
# A full SVD above this element count (or narrow-side length) is skipped: the value
# reading stands alone there. The vocabulary head is the tensor this spares.
SPECTRAL_ELEM_CAP = 64 * 1024 * 1024
SPECTRAL_DIM_CAP = 8192
REL_SV_EPS = 1e-10  # a singular value below this fraction of the largest counts as zero
NULL_NOTE = (
    "Null baseline: the entropy shift a random update of the same norm as ΔW would cause "
    "(spectral-reading tensors only); a real shift beyond it is structural, not an effect of the update's size."
)


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class EntropyMetric:
    """One entropy reading of a tensor, base vs tuned (``None`` when not measured)."""

    base: float | None
    tuned: float | None
    delta: float | None  # tuned - base
    base_norm: float | None  # base entropy over its maximum (0..1), comparable across shapes
    tuned_norm: float | None


@dataclass(frozen=True)
class TensorEntropy:
    """Both entropy readings of one in-scope tensor, with its change and quantization."""

    name: str
    layer: int  # -1 for the head (embedding / final norm)
    module: str
    role: str
    shape: list[int]
    spectral: EntropyMetric  # base/tuned None when the tensor is 1-D or too large for a full SVD
    value: EntropyMetric
    effective_rank: int | None  # singular values above REL_SV_EPS of the largest (spectral tensors only)
    rel_change: float | None  # ‖ΔW‖ / ‖W‖
    changed: bool
    quantization: str | None = None
    noise_floor: float | None = None
    beyond_noise: bool | None = None


@dataclass(frozen=True)
class LayerEntropy:
    """Mean entropy shift of the in-scope tensors of one layer (-1: the head)."""

    layer: int
    mean_spectral_delta: float | None
    mean_value_delta: float | None
    tensors: int


@dataclass(frozen=True)
class EntropyCounts:
    """How many tensors were examined, read spectrally, and moved."""

    in_scope: int
    spectral: int
    changed: int
    unmatched: int


@dataclass
class EntropyResult:
    """Everything :func:`analyse_entropy` found, as plain values."""

    meta: ResultMeta
    layers: int | None  # transformer blocks of the base model (None when unknown)
    last_block_start: int | None  # first block index in scope (None when no block structure was found)
    counts: EntropyCounts
    tensors: list[TensorEntropy]  # by |spectral Δ| then |value Δ|, largest first; missing last
    by_layer: list[LayerEntropy]
    spectral_null: stats.NullComparison | None  # mean spectral Δ vs random-update shifts
    value_null: stats.NullComparison | None  # mean value Δ vs random-update shifts
    beyond_noise: int | None = None  # changed tensors above their quantization noise (quantized comparisons)


# ----------------------------------------------------------------- analysis
def analyse_entropy(
    base: TensorSource,
    tuned: TensorSource,
    *,
    inputs: dict[str, str],
    layers: int = 2,
    bins: int = 64,
    seed: int = stats.DEFAULT_SEED,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
) -> EntropyResult:
    """Compare the entropy of the last layers' weights of ``tuned`` against ``base``.

    Args:
        base: the source model (also fixes the layer count and the scope).
        tuned: the model to compare (full weights, or an adapter applied to ``base``).
        inputs: what is being compared, recorded in the result metadata.
        layers: number of last transformer blocks examined, besides the head.
        bins: histogram bins of the value-distribution entropy.
        seed: seed of the random-update null baseline.
        sink: receives one progress event per visited tensor.
        token: checked between tensors; raises ``JobCancelled`` once cancelled.

    Returns:
        Per-tensor and per-layer entropy shifts, with the two null baselines.

    Raises:
        JobCancelled: when ``token`` is cancelled.
    """
    block_count = _layer_count(base)
    window_start = block_count - layers if block_count is not None else None
    log = DeltaLog()
    generator = stats.seeded_generator(seed)
    total = total_deltas(base, tuned)
    measured: list[TensorEntropy] = []
    spectral_nulls: list[float] = []
    value_nulls: list[float] = []
    warnings: list[str] = []
    for delta in iter_deltas(base, tuned, log, token):
        report(sink, Stage.ENTROPY, f"Measuring {delta.name}", log.visited, total)
        if _in_scope(delta.name, delta.base.dim(), window_start):
            _measure_into(delta, base, tuned, bins, generator, measured, spectral_nulls, value_nulls, warnings)
    report(sink, Stage.ENTROPY, f"Examined {len(measured)} tensors", total, total)
    return _build_result(
        inputs, layers, bins, block_count, window_start, log, measured, spectral_nulls, value_nulls, warnings
    )


# ----------------------------------------------------------------- scope
def _layer_count(base: TensorSource) -> int | None:
    """Number of transformer blocks of the base model: from its config, else from its tensor names."""
    result: int | None = None
    try:
        result = Architecture.from_config(base.config()).layers
    except (ModelError, ValueError, TypeError):
        result = None
    if result is None:
        highest = max((stats.layer_of(name) for name in base.names()), default=-1)
        result = highest + 1 if highest >= 0 else None
    return result


def _is_final_norm(name: str) -> bool:
    """Whether a tensor is the model's final norm (the one head tensor kept despite being 1-D)."""
    return any(name.endswith(suffix) for suffix in FINAL_NORM_SUFFIXES)


def _in_scope(name: str, dim: int, window_start: int | None) -> bool:
    """Whether a tensor is examined: the final norm, or a 2-D matrix in the last blocks or the head.

    Within-block 1-D tensors (biases, layer norms) are left out: only matrices carry a
    spectral reading, and the final norm is the single 1-D tensor of the head region.
    """
    result = False
    if _is_final_norm(name):
        result = True
    elif dim == 2:
        layer = stats.layer_of(name)
        in_window = window_start is not None and layer >= 0 and layer >= window_start
        result = in_window or stats.is_embedding(name)
    return result


# ----------------------------------------------------------------- measurement
def _measure_into(
    delta: Delta,
    base: TensorSource,
    tuned: TensorSource,
    bins: int,
    generator: torch.Generator,
    measured: list[TensorEntropy],
    spectral_nulls: list[float],
    value_nulls: list[float],
    warnings: list[str],
) -> None:
    """Measure one in-scope tensor; a failure is noted and the tensor skipped, never raised."""
    try:
        tensor, spectral_null, value_null = _measure(delta, base, tuned, bins, generator)
    except Exception as error:  # noqa: BLE001 - one bad tensor must not abort the analysis
        warnings.append(f"{delta.name}: {error.__class__.__name__}: {error}")
        return
    measured.append(tensor)
    if spectral_null is not None:
        spectral_nulls.append(spectral_null)
    if value_null is not None:
        value_nulls.append(value_null)


def _measure(
    delta: Delta, base: TensorSource, tuned: TensorSource, bins: int, generator: torch.Generator
) -> tuple[TensorEntropy, float | None, float | None]:
    """Both entropy readings of one tensor, plus its random-update null shifts when spectral.

    Returns:
        The per-tensor result, and the spectral and value null entropy shifts
        (both ``None`` when the tensor is not spectral-eligible).
    """
    weight = delta.base
    tuned_weight = delta.base + delta.delta
    lo = min(weight.min().item(), tuned_weight.min().item())
    hi = max(weight.max().item(), tuned_weight.max().item())
    value = _metric(_value_entropy(weight, bins, lo, hi), _value_entropy(tuned_weight, bins, lo, hi))
    spectral, rank = EntropyMetric(None, None, None, None, None), None
    spectral_null = value_null = None
    if _spectral_eligible(weight):
        base_spectral, rank = _spectral_entropy(weight)
        tuned_spectral, _ = _spectral_entropy(tuned_weight)
        spectral = _metric(base_spectral, tuned_spectral)
        null_weight = weight + _random_like(delta.delta, generator)
        spectral_null = _spectral_entropy(null_weight)[0][0] - base_spectral[0]
        value_null = _value_entropy(null_weight, bins, lo, hi)[0] - value.base  # type: ignore[operator]
    floor = stats.quantization_floor(delta, base, tuned)
    rel_change = _rel_change(delta)
    return (
        TensorEntropy(
            name=delta.name,
            layer=stats.layer_of(delta.name),
            module=stats.module_of(delta.name),
            role=stats.role_of(delta.name),
            shape=list(weight.shape),
            spectral=spectral,
            value=value,
            effective_rank=rank,
            rel_change=rel_change,
            changed=delta.delta.norm().item() >= stats.CHANGE_THRESHOLD,
            quantization=_quantization_label(delta, base, tuned) if floor is not None else None,
            noise_floor=floor.rel_error if floor is not None else None,
            beyond_noise=stats.beyond_noise(rel_change, floor),
        ),
        spectral_null,
        value_null,
    )


def _spectral_eligible(weight: torch.Tensor) -> bool:
    """Whether a tensor is a 2-D matrix small enough for a full SVD."""
    return weight.dim() == 2 and min(weight.shape) <= SPECTRAL_DIM_CAP and weight.numel() <= SPECTRAL_ELEM_CAP


def _spectral_entropy(matrix: torch.Tensor) -> tuple[tuple[float, float], int]:
    """Spectral entropy of a 2-D matrix, raw and normalised, with its effective rank.

    Returns:
        ``((H, H / ln n), rank)`` where ``n`` is the number of singular values and
        ``rank`` counts those above :data:`REL_SV_EPS` of the largest. A zero
        matrix has zero entropy and zero rank.
    """
    singular = torch.linalg.svdvals(matrix.double())
    squared = singular * singular
    total = squared.sum()
    result = ((0.0, 0.0), 0)
    if total > 0:
        entropy = stats.shannon_entropy(squared / total)
        count = int(singular.numel())
        rank = int((singular > singular[0] * REL_SV_EPS).sum().item())
        result = ((entropy, entropy / math.log(count) if count > 1 else 0.0), rank)
    return result


def _value_entropy(tensor: torch.Tensor, bins: int, lo: float, hi: float) -> tuple[float, float]:
    """Shannon entropy of a ``bins``-bucket histogram of the values over ``[lo, hi]``, raw and normalised.

    A degenerate range (a constant tensor) has zero entropy.
    """
    result = (0.0, 0.0)
    if hi > lo:
        histogram = torch.histc(tensor.double(), bins=bins, min=lo, max=hi)
        total = histogram.sum()
        if total > 0:
            entropy = stats.shannon_entropy(histogram / total)
            result = (entropy, entropy / math.log(bins) if bins > 1 else 0.0)
    return result


def _random_like(delta: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """A random tensor of ``delta``'s shape, rescaled to its Frobenius norm (seeded)."""
    noise = torch.randn(delta.shape, generator=generator)
    return noise * (delta.norm() / noise.norm().clamp_min(stats.MIN_NORM))


def _metric(base: tuple[float, float], tuned: tuple[float, float]) -> EntropyMetric:
    """Build an :class:`EntropyMetric` from the (raw, normalised) readings of base and tuned."""
    return EntropyMetric(base=base[0], tuned=tuned[0], delta=tuned[0] - base[0], base_norm=base[1], tuned_norm=tuned[1])


def _rel_change(delta: Delta) -> float | None:
    """Relative Frobenius change ``‖ΔW‖ / ‖W‖``, or None when the base tensor is all zeros."""
    base_norm = delta.base.norm().item()
    return delta.delta.norm().item() / base_norm if base_norm else None


def _quantization_label(delta: Delta, base: TensorSource, tuned: TensorSource) -> str:
    """Storage of the two compared tensors, quantized ones only: ``"Q4_K"``, ``"nf4 + Q8_0"``."""
    schemes = (base.quantization(delta.name), tuned.quantization(delta.source_name))
    return " + ".join(scheme.name for scheme in schemes if scheme is not None and not scheme.lossless)


# ----------------------------------------------------------------- aggregation
def _build_result(
    inputs: dict[str, str],
    layers: int,
    bins: int,
    block_count: int | None,
    window_start: int | None,
    log: DeltaLog,
    measured: list[TensorEntropy],
    spectral_nulls: list[float],
    value_nulls: list[float],
    warnings: list[str],
) -> EntropyResult:
    """Sort the tensors, aggregate per layer and summarise the two null baselines."""
    if any(tensor.quantization for tensor in measured):
        warnings.append(
            "Quantized weights: the entropy of a dequantized tensor carries quantization error. Trust a shift only "
            f"when the weight change itself is beyond the noise floor ('beyond noise' = more than "
            f"{stats.NOISE_MARGIN:g} floors)."
        )
    spectral = [tensor for tensor in measured if tensor.spectral.delta is not None]
    meta = ResultMeta(
        analysis=ANALYSIS,
        inputs=dict(inputs),
        parameters={"layers": layers, "bins": bins},
        notes=[*log.notes, NULL_NOTE],
        warnings=warnings,
    )
    return EntropyResult(
        meta=meta,
        layers=block_count,
        last_block_start=window_start,
        counts=EntropyCounts(
            in_scope=len(measured),
            spectral=len(spectral),
            changed=sum(1 for tensor in measured if tensor.changed),
            unmatched=log.unmatched,
        ),
        tensors=sorted(measured, key=_delta_order),
        by_layer=_layer_profile(measured),
        spectral_null=_null_of(spectral, lambda tensor: tensor.spectral.delta, spectral_nulls),
        value_null=_null_of(spectral, lambda tensor: tensor.value.delta, value_nulls),
        beyond_noise=_count_beyond_noise(measured),
    )


def _delta_order(tensor: TensorEntropy) -> tuple[bool, float, float]:
    """Sort key: largest |spectral Δ| first, tensors without one after, then |value Δ|."""
    spectral = tensor.spectral.delta
    value = tensor.value.delta or 0.0
    return spectral is None, -abs(spectral or 0.0), -abs(value)


def _layer_profile(measured: list[TensorEntropy]) -> list[LayerEntropy]:
    """Mean spectral and value entropy shift per layer, over the tensors that have each."""
    spectral: defaultdict[int, list[float]] = defaultdict(list)
    value: defaultdict[int, list[float]] = defaultdict(list)
    for tensor in measured:
        if tensor.spectral.delta is not None:
            spectral[tensor.layer].append(tensor.spectral.delta)
        if tensor.value.delta is not None:
            value[tensor.layer].append(tensor.value.delta)
    layers = sorted(set(spectral) | set(value))
    return [
        LayerEntropy(
            layer=layer,
            mean_spectral_delta=_mean(spectral.get(layer)),
            mean_value_delta=_mean(value.get(layer)),
            tensors=max(len(spectral.get(layer, [])), len(value.get(layer, []))),
        )
        for layer in layers
    ]


def _null_of(
    tensors: list[TensorEntropy],
    observed_of: Callable[[TensorEntropy], float | None],
    nulls: list[float],
) -> stats.NullComparison | None:
    """The mean observed shift against the random-update null samples, or None when there are none."""
    observed = [observed_of(tensor) for tensor in tensors]
    present = [value for value in observed if value is not None]
    return stats.NullComparison.build(_mean(present) or 0.0, nulls) if present and nulls else None


def _count_beyond_noise(measured: list[TensorEntropy]) -> int | None:
    """Number of tensors above their noise floor, or None when nothing was quantized."""
    flags = [tensor.beyond_noise for tensor in measured if tensor.beyond_noise is not None]
    return sum(flags) if flags else None


def _mean(values: list[float] | None) -> float | None:
    """The mean of a non-empty list, else None."""
    return sum(values) / len(values) if values else None
