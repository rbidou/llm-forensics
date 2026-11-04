"""bitsandbytes-quantized safetensors checkpoints, dequantized without bitsandbytes.

transformers saves a 4-bit ``Linear4bit`` weight as packed ``uint8`` codes plus
its quantization state, under the weight's own name::

    <module>.weight                                  two 4-bit codes per byte, high nibble first
    <module>.weight.absmax                           one scale per block (uint8 when double-quantized)
    <module>.weight.quant_map                        the 16-value code book (NF4 or FP4)
    <module>.weight.nested_absmax / nested_quant_map second-level state of the scales
    <module>.weight.quant_state.bitsandbytes__nf4    JSON: shape, dtype, blocksize, nested_offset...

and an 8-bit ``Linear8bitLt`` weight as ``int8`` values with one scale per row
(``<module>.SCB``): ``W = weight * SCB / 127``.

Reading these needs no GPU and no bitsandbytes: the arithmetic is reproduced
here (and checked against bitsandbytes itself in the tests).
"""

from __future__ import annotations

import json
from typing import Any

import torch

from models.quant import BNB_DEFAULT_BLOCKSIZE, INT8_LEVELS, QuantScheme
from models.source import Rows, TensorSource, slice_rows

STATE_PREFIX = ".quant_state.bitsandbytes__"
AUXILIARY_SUFFIXES: tuple[str, ...] = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".SCB",
    ".weight_format",
)
FOUR_BIT_TYPES: tuple[str, ...] = ("nf4", "fp4")
NIBBLE = 4
LOW_NIBBLE_MASK = 0x0F


class BnbSource(TensorSource):
    """A bitsandbytes checkpoint read through another source, dequantized on load."""

    def __init__(self, raw: TensorSource) -> None:
        super().__init__(raw.ref)
        self.raw = raw
        self._names = tuple(name for name in raw.names() if not _is_auxiliary(name))
        self._four_bit = {name: kind for name in self._names if (kind := self._four_bit_kind(name))}

    def names(self) -> tuple[str, ...]:
        """Model tensor names, without the quantization state tensors."""
        return self._names

    def has(self, name: str) -> bool:
        """Whether ``name`` is a model tensor (not a state tensor)."""
        return self.raw.has(name) and not _is_auxiliary(name)

    def shape(self, name: str) -> list[int]:
        """Original (dequantized) shape of tensor ``name``."""
        result = self.raw.shape(name)
        if name in self._four_bit:
            result = list(self._state(name)["shape"])
        return result

    def quantization(self, name: str) -> QuantScheme | None:
        """bitsandbytes mode of tensor ``name`` (None when stored in plain floats)."""
        result = None
        if name in self._four_bit:
            state = self._state(name)
            codebook = tuple(self.raw.load(f"{name}.quant_map").tolist())
            result = QuantScheme(self._four_bit[name], "bnb", int(state["blocksize"]), codebook)
        elif self._is_int8(name):
            result = QuantScheme("int8", "bnb")
        return result

    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Tensor ``name`` (or rows ``rows``) dequantized to float32."""
        result: torch.Tensor
        if name in self._four_bit:
            result = slice_rows(self._dequantize_4bit(name), rows)
        elif self._is_int8(name):
            scale = self.raw.load(_int8_scale_name(name), rows)
            result = self.raw.load(name, rows) * scale.unsqueeze(1) / INT8_LEVELS
        else:
            result = self.raw.load(name, rows)
        return result

    def config(self) -> dict[str, Any]:
        """The checkpoint's configuration."""
        return self.raw.config()

    def _dequantize_4bit(self, name: str) -> torch.Tensor:
        """Unpack two codes per byte, look them up in the code book, scale per block."""
        state = self._state(name)
        packed = self.raw.load(name).flatten().to(torch.int64)
        codes = torch.stack([packed >> NIBBLE, packed & LOW_NIBBLE_MASK], dim=1).flatten()
        codebook = self.raw.load(f"{name}.quant_map")
        blocksize = int(state.get("blocksize", BNB_DEFAULT_BLOCKSIZE))
        absmax = self._absmax(name, state)
        count = int(torch.Size(state["shape"]).numel())
        values = codebook[codes[:count]]
        padding = (-count) % blocksize
        blocks = torch.cat([values, values.new_zeros(padding)]).view(-1, blocksize)
        return (blocks * absmax[: blocks.shape[0]].unsqueeze(1)).flatten()[:count].view(*state["shape"])

    def _absmax(self, name: str, state: dict[str, Any]) -> torch.Tensor:
        """Per-block scales, undoing the second-level quantization when present."""
        absmax = self.raw.load(f"{name}.absmax").flatten()
        result = absmax
        if self.raw.has(f"{name}.nested_absmax"):
            nested_scale = self.raw.load(f"{name}.nested_absmax").flatten()
            nested_codebook = self.raw.load(f"{name}.nested_quant_map")
            nested_blocksize = int(state["nested_blocksize"])
            per_value = nested_scale.repeat_interleave(nested_blocksize)[: absmax.numel()]
            result = nested_codebook[absmax.to(torch.int64)] * per_value + float(state.get("nested_offset", 0.0))
        return result

    def _state(self, name: str) -> dict[str, Any]:
        """The JSON quantization state stored as a byte tensor."""
        key = f"{name}{STATE_PREFIX}{self._four_bit[name]}"
        raw_bytes = bytes(self.raw.load(key).to(torch.uint8).tolist())
        return json.loads(raw_bytes.decode("utf-8"))

    def _four_bit_kind(self, name: str) -> str | None:
        """``nf4`` / ``fp4`` when ``name`` is a 4-bit weight, else None."""
        return next((kind for kind in FOUR_BIT_TYPES if self.raw.has(f"{name}{STATE_PREFIX}{kind}")), None)

    def _is_int8(self, name: str) -> bool:
        """Whether ``name`` is an 8-bit weight with row scales."""
        return name.endswith(".weight") and self.raw.has(_int8_scale_name(name))


def _int8_scale_name(name: str) -> str:
    """``<module>.weight`` -> ``<module>.SCB``."""
    return f"{name.removesuffix('.weight')}.SCB"


def _is_auxiliary(name: str) -> bool:
    """Whether a tensor is quantization state rather than a model weight."""
    return STATE_PREFIX in name or name.endswith(AUXILIARY_SUFFIXES)
