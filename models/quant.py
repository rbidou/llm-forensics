"""Quantization schemes, and the noise they add on their own.

Comparing a full-precision base with a quantized model mixes two effects: what
fine-tuning changed, and what quantization destroyed. The second one is
measured here, per tensor, by quantizing the *base* tensor with the same scheme
and dequantizing it again: the relative error obtained is the change a tensor
would show if fine-tuning had not touched it at all. That is the noise floor
every quantized comparison is read against.

Exact simulations are used when they exist (llama.cpp legacy and ternary types
through the ``gguf`` package, bitsandbytes NF4/FP4/int8 here). llama.cpp K- and
IQ-quants have no Python quantizer: their floor is *estimated* with a uniform
asymmetric quantizer of the same bit width on blocks of 32 (the ``Q{b}_1``
scheme), and labelled as such. K-quants are somewhat more accurate than that
model, so the estimate errs on the conservative (higher) side.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# NormalFloat4 code book of bitsandbytes (quantiles of N(0, 1), normalised to [-1, 1]).
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
BNB_DEFAULT_BLOCKSIZE = 64
INT8_LEVELS = 127

# Bit width used to estimate llama.cpp types that have no Python quantizer.
ESTIMATED_BITS: dict[str, int] = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q8_K": 8,
    "IQ1_S": 1,
    "IQ1_M": 1,
    "IQ2_XXS": 2,
    "IQ2_XS": 2,
    "IQ2_S": 2,
    "IQ3_XXS": 3,
    "IQ3_S": 3,
    "IQ4_NL": 4,
    "IQ4_XS": 4,
    "NVFP4": 4,
}
ESTIMATE_BLOCK = 32
FULL_PRECISION: frozenset[str] = frozenset({"F32", "F64"})
SAMPLE_ROWS = 256  # rows used to measure a floor: a relative error converges quickly


@dataclass(frozen=True)
class QuantScheme:
    """How one stored tensor is quantized.

    ``name`` is a llama.cpp type (``Q4_K``, ``Q8_0``, ``F16``...) or a
    bitsandbytes mode (``nf4``, ``fp4``, ``int8``).
    """

    name: str
    family: str  # gguf | bnb
    blocksize: int | None = None
    codebook: tuple[float, ...] | None = None

    @property
    def lossless(self) -> bool:
        """Whether the scheme keeps float32 values unchanged."""
        return self.name in FULL_PRECISION


@dataclass(frozen=True)
class NoiseFloor:
    """Relative error ``‖Q(W) - W‖ / ‖W‖`` a scheme adds to a tensor by itself."""

    rel_error: float
    exact: bool  # True: simulated with the real quantizer; False: estimated
    method: str


def noise_floor(weight: torch.Tensor, scheme: QuantScheme) -> NoiseFloor:
    """Noise floor of ``scheme`` on ``weight`` (measured on a sample of rows).

    Args:
        weight: the full-precision tensor (typically the base model's).
        scheme: the quantization of the compared tensor.

    Returns:
        The relative error quantization alone introduces, and how it was obtained.
    """
    sample = _sample_rows(weight.float())
    result: NoiseFloor
    if scheme.lossless:
        result = NoiseFloor(0.0, True, "lossless")
    elif scheme.family == "bnb":
        result = NoiseFloor(_relative_error(sample, _simulate_bnb(sample, scheme)), True, f"bitsandbytes {scheme.name}")
    else:
        result = _gguf_floor(sample, scheme.name)
    return result


def _gguf_floor(sample: torch.Tensor, type_name: str) -> NoiseFloor:
    """Exact floor through the gguf quantizer, or a uniform estimate when it has none."""
    exact = _gguf_round_trip(sample, type_name)
    result: NoiseFloor
    if exact is not None:
        result = NoiseFloor(_relative_error(sample, exact), True, f"llama.cpp {type_name}")
    else:
        bits = ESTIMATED_BITS.get(type_name, 4)
        estimate = _uniform_round_trip(sample, bits, ESTIMATE_BLOCK)
        method = f"estimated: {bits}-bit uniform, blocks of {ESTIMATE_BLOCK} (no Python quantizer for {type_name})"
        result = NoiseFloor(_relative_error(sample, estimate), False, method)
    return result


def _gguf_round_trip(sample: torch.Tensor, type_name: str) -> torch.Tensor | None:
    """Quantize and dequantize with the gguf package; None when it cannot."""
    import gguf
    from gguf import quants

    result = None
    try:
        qtype = gguf.GGMLQuantizationType[type_name]
        restored = quants.dequantize(quants.quantize(sample.numpy(), qtype), qtype)
        result = torch.from_numpy(restored.reshape(sample.shape).astype("float32"))
    except (KeyError, NotImplementedError, ValueError, TypeError):
        result = None
    return result


def _uniform_round_trip(sample: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    """Asymmetric min/max uniform quantization on blocks of ``block`` values."""
    flat = sample.flatten()
    padding = (-flat.numel()) % block
    blocks = torch.cat([flat, flat.new_zeros(padding)]).view(-1, block)
    low, high = blocks.min(1, keepdim=True).values, blocks.max(1, keepdim=True).values
    step = ((high - low) / (2**bits - 1)).clamp_min(1e-12)
    restored = ((blocks - low) / step).round() * step + low
    return restored.flatten()[: flat.numel()].view_as(sample)


def _simulate_bnb(sample: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """bitsandbytes round trip: row-wise absmax int8, or block-wise 4-bit code book."""
    result: torch.Tensor
    if scheme.name == "int8":
        scale = sample.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / INT8_LEVELS
        result = (sample / scale).round().clamp(-INT8_LEVELS, INT8_LEVELS) * scale
    else:
        codebook = torch.tensor(scheme.codebook or NF4_CODEBOOK)
        result = codebook_round_trip(sample, codebook, scheme.blocksize or BNB_DEFAULT_BLOCKSIZE)
    return result


def codebook_round_trip(weight: torch.Tensor, codebook: torch.Tensor, blocksize: int) -> torch.Tensor:
    """Block-wise absmax scaling, then nearest code book value (bitsandbytes 4-bit)."""
    flat = weight.flatten()
    padding = (-flat.numel()) % blocksize
    blocks = torch.cat([flat, flat.new_zeros(padding)]).view(-1, blocksize)
    absmax = blocks.abs().amax(1, keepdim=True).clamp_min(1e-12)
    nearest = ((blocks / absmax).unsqueeze(-1) - codebook).abs().argmin(-1)
    return (codebook[nearest] * absmax).flatten()[: flat.numel()].view_as(weight)


def _sample_rows(weight: torch.Tensor) -> torch.Tensor:
    """Evenly spaced rows of a matrix (all of a vector), as a 2-D float32 tensor."""
    matrix = weight.view(1, -1) if weight.dim() == 1 else weight.reshape(weight.shape[0], -1)
    if matrix.shape[0] > SAMPLE_ROWS:
        matrix = matrix[torch.linspace(0, matrix.shape[0] - 1, SAMPLE_ROWS).long()]
    return matrix.contiguous()


def _relative_error(reference: torch.Tensor, restored: torch.Tensor) -> float:
    """``‖restored - reference‖ / ‖reference‖`` (0 for an all-zero reference)."""
    norm = reference.norm().item()
    return (restored - reference).norm().item() / norm if norm else 0.0
