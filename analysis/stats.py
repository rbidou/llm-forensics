"""Metrics, null baselines and SVD helpers shared by every analysis.

Conventions (README §8 of the reference scripts):

* all arithmetic in float32 on CPU, similarity scores accumulated in float64;
* every similarity comes with a null baseline when one is feasible, built from
  shuffled rows, columns, tokens or random directions with a seeded generator;
* a z-score divides by ``max(sd, MIN_NULL_SD)`` so a degenerate null cannot
  produce an infinite score.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.deltas import Delta, LowRankFactors
from models.quant import NoiseFloor, noise_floor
from models.source import TensorSource

# Modules that add to the residual stream: their output direction maps to tokens through the unembedding.
WRITERS: tuple[str, ...] = ("o_proj", "down_proj", "out_proj", "dense_4h_to_h", "c_proj", "w2", "wo")
# Modules that read from it: their input direction maps to tokens through the embeddings.
READERS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
    "qkv_proj",
    "gate_up_proj",
    "query_key_value",
    "c_attn",
    "c_fc",
    "w1",
    "w3",
)
EMBEDDING_PARTS: tuple[str, ...] = ("embed_tokens", "lm_head", "wte", "word_embeddings")

NOISE_MARGIN = 2.0  # a change counts as real when it exceeds this many quantization noise floors
MIN_REAL_REL_CHANGE = 1e-5  # below this, a relative change is rounding (fine-tunes move weights by 1e-4 or more)
CHANGE_THRESHOLD = 1e-6  # |Δ| or ‖ΔW‖ below this counts as unchanged
MIN_NORM = 1e-12
MIN_NULL_SD = 1e-4
DEFAULT_SEED = 0
LAYER_PATTERN = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")

# Roles of a tensor in the residual-stream view (see role_of).
ROLE_WRITER, ROLE_READER, ROLE_EMBEDDING, ROLE_OTHER = "writer", "reader", "embedding", "other"


# ----------------------------------------------------------------- tensor names
def layer_of(name: str) -> int:
    """Layer number parsed from a tensor name, or -1 for non-layer tensors."""
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else -1


def module_of(name: str) -> str:
    """Module short name: ``model.layers.3.mlp.up_proj.weight`` -> ``up_proj``."""
    parts = name.split(".")
    return parts[-2] if len(parts) >= 2 else name


def is_embedding(name: str) -> bool:
    """Whether a tensor is a token embedding or unembedding matrix."""
    return any(part in name for part in EMBEDDING_PARTS)


def role_of(name: str) -> str:
    """``writer``, ``reader``, ``embedding`` or ``other``, from the residual-stream view."""
    module = module_of(name)
    result = ROLE_OTHER
    if is_embedding(name):
        result = ROLE_EMBEDDING
    elif module in WRITERS:
        result = ROLE_WRITER
    elif module in READERS:
        result = ROLE_READER
    return result


# ----------------------------------------------------------------- similarities
def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """Cosine similarity of two tensors flattened, in float64."""
    return F.cosine_similarity(first.flatten().double(), second.flatten().double(), dim=0).item()


def pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    """Pearson correlation of two tensors flattened, in float64."""
    a, b = first.flatten().double(), second.flatten().double()
    return F.cosine_similarity(a - a.mean(), b - b.mean(), dim=0).item()


def relative_distance(first: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative Frobenius distance ``‖first - reference‖ / ‖reference‖``."""
    return ((first - reference).norm() / reference.norm().clamp_min(MIN_NORM)).item()


def spectrum_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    """Singular-value similarity ``1 - ‖s₁ - s₂‖ / ‖s₁‖`` (rotation and permutation invariant)."""
    s1, s2 = torch.linalg.svdvals(first), torch.linalg.svdvals(second)
    return 1 - ((s1 - s2).norm() / s1.norm().clamp_min(MIN_NORM)).item()


def nearest_row_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
    generator: torch.Generator,
    sample: int = 512,
    shuffle_columns: bool = False,
) -> float:
    """Mean, over sampled rows of ``first``, of the best cosine against any row of ``second``.

    Invariant to a permutation of rows (neurons). With ``shuffle_columns`` it
    becomes its own null baseline: the columns of ``second`` are permuted, which
    destroys any real correspondence while keeping the row statistics.
    """
    first_unit, second_unit = F.normalize(first, dim=1), F.normalize(second, dim=1)
    if shuffle_columns:
        second_unit = second_unit[:, torch.randperm(second_unit.shape[1], generator=generator)]
    rows = torch.randperm(first_unit.shape[0], generator=generator)[:sample]
    return (first_unit[rows] @ second_unit.T).max(1).values.mean().item()


# ----------------------------------------------------------------- null baselines
@dataclass(frozen=True)
class NullComparison:
    """An observed score against its null distribution."""

    value: float
    null_mean: float
    null_sd: float
    z: float

    @classmethod
    def build(cls, value: float, nulls: Sequence[float]) -> NullComparison:
        """Summarise ``value`` against the null samples."""
        mean = statistics.fmean(nulls)
        sd = statistics.pstdev(nulls) if len(nulls) > 1 else 0.0
        return cls(value=value, null_mean=mean, null_sd=sd, z=(value - mean) / max(sd, MIN_NULL_SD))


def row_shuffle_nulls(
    score: Callable[[torch.Tensor, torch.Tensor], float],
    first: torch.Tensor,
    second: torch.Tensor,
    generator: torch.Generator,
    draws: int,
) -> list[float]:
    """Scores of ``first`` against ``second`` with its rows shuffled, ``draws`` times."""
    return [score(first, second[torch.randperm(second.shape[0], generator=generator)]) for _ in range(draws)]


def seeded_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """A CPU generator with a fixed seed, so every null baseline is reproducible."""
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------------------- change metrics
@dataclass(frozen=True)
class ChangeStats:
    """How much one tensor moved between base and tuned."""

    rel_change: float | None  # ‖ΔW‖ / ‖W‖, None when ‖W‖ = 0
    cosine: float  # cos(W, W + ΔW)
    max_abs_diff: float
    frac_changed: float  # fraction of elements with |Δ| > CHANGE_THRESHOLD
    delta_norm: float

    @property
    def changed(self) -> bool:
        """Whether the tensor moved at all."""
        return self.delta_norm >= CHANGE_THRESHOLD


def change_stats(base: torch.Tensor, delta: torch.Tensor) -> ChangeStats:
    """Change metrics between ``base`` and ``base + delta``."""
    base_norm = base.norm().item()
    delta_norm = delta.norm().item()
    tuned = base + delta
    return ChangeStats(
        rel_change=delta_norm / base_norm if base_norm else None,
        cosine=F.cosine_similarity(base.flatten(), tuned.flatten(), dim=0).item(),
        max_abs_diff=delta.abs().max().item() if delta.numel() else 0.0,
        frac_changed=(delta.abs() > CHANGE_THRESHOLD).float().mean().item() if delta.numel() else 0.0,
        delta_norm=delta_norm,
    )


# ----------------------------------------------------------------- entropy
def shannon_entropy(probabilities: torch.Tensor) -> float:
    """Shannon entropy ``-Σ p·ln p`` of a probability vector, in nats (float64).

    Zero-probability bins contribute nothing (``0·ln 0 = 0``); they are dropped
    rather than clamped, so the result is exact. The input need not be
    normalised to one: it is treated as already being a distribution.
    """
    probabilities = probabilities.double()
    positive = probabilities[probabilities > 0]
    return -(positive * positive.log()).sum().item() if positive.numel() else 0.0


# ----------------------------------------------------------------- SVD
@dataclass(frozen=True)
class Direction:
    """The leading singular triplet of an update ΔW = σ₁ u vᵀ + ...

    ``u`` lives in the output space (rows of W), ``v`` in the input space.
    Signs are normalised (largest |u| component positive) so results are
    reproducible; read ``+`` and ``-`` as the two ends of one axis anyway.
    """

    sigma: float
    share: float  # σ₁² / ‖ΔW‖², how much of the update this direction explains
    u: torch.Tensor
    v: torch.Tensor
    rank: int | None  # numerical rank, known exactly for low-rank updates


def leading_direction(delta: Delta, rank: int = 6, seed: int = DEFAULT_SEED) -> Direction:
    """Leading singular triplet of ``delta.delta``.

    Low-rank (LoRA) updates use their factors: exact and cheap. Full updates use
    a randomized SVD of rank ``rank``, seeded per call so the result does not
    depend on the order in which tensors are visited.
    """
    matrix = delta.delta
    total = matrix.norm().item() ** 2
    if delta.factors is not None:
        u, s, v, numerical_rank = _factored_svd(delta.factors)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(matrix, q=min(rank, *matrix.shape), niter=4)
        numerical_rank = None
    first_u, first_v = _normalise_sign(u[:, 0], v[:, 0])
    sigma = s[0].item()
    return Direction(sigma=sigma, share=sigma**2 / total if total else 0.0, u=first_u, v=first_v, rank=numerical_rank)


def _factored_svd(factors: LowRankFactors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exact SVD of ``L @ Rᵀ`` through two QR decompositions of the thin factors.

    ``ΔW = scale · B @ A`` (or its transpose for embeddings) is written ``L @ Rᵀ``
    with L and R of width r; then ``L = Q_L R_L``, ``R = Q_R R_R`` and only the
    r x r matrix ``R_L R_Rᵀ`` needs a full SVD.
    """
    left, right = factors.lora_b * factors.scale, factors.lora_a.T
    if factors.transposed:
        left, right = right, left
    q_left, r_left = torch.linalg.qr(left)
    q_right, r_right = torch.linalg.qr(right)
    small_u, s, small_vh = torch.linalg.svd(r_left @ r_right.T)
    rank = int(torch.linalg.matrix_rank(r_left @ r_right.T).item())
    return q_left @ small_u, s, q_right @ small_vh.T, rank


def _normalise_sign(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip (u, v) together so that u's largest-magnitude component is positive."""
    sign = 1.0 if u[u.abs().argmax()].item() >= 0 else -1.0
    return u * sign, v * sign


# ----------------------------------------------------------------- quantization
def quantization_floor(delta: Delta, base: TensorSource, tuned: TensorSource) -> NoiseFloor | None:
    """Relative change quantization alone would produce on this tensor, or None.

    Exact low-rank updates (an adapter on its own base) carry no quantization
    noise. Otherwise the floors of the base and the tuned tensor combine in
    quadrature, as independent errors do.
    """
    schemes = (
        [] if delta.factors is not None else [base.quantization(delta.name), tuned.quantization(delta.source_name)]
    )
    floors = [noise_floor(delta.base, scheme) for scheme in schemes if scheme is not None and not scheme.lossless]
    result = None
    if floors:
        result = NoiseFloor(
            rel_error=sum(floor.rel_error**2 for floor in floors) ** 0.5,
            exact=all(floor.exact for floor in floors),
            method=" + ".join(floor.method for floor in floors),
        )
    return result


def beyond_noise(rel_change: float | None, floor: NoiseFloor | None) -> bool | None:
    """Whether a relative change is real: above ``NOISE_MARGIN`` noise floors and above rounding.

    The absolute minimum matters for near-lossless storage (F16, BF16): their floor
    is about 1e-9, so float rounding alone would otherwise count as a change.

    Returns:
        None when there is no floor (full-precision comparison).
    """
    result = None
    if floor is not None and rel_change is not None:
        result = rel_change > max(NOISE_MARGIN * floor.rel_error, MIN_REAL_REL_CHANGE)
    return result
