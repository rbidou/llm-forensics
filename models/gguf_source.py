"""llama.cpp GGUF files, dequantized to float32 one tensor at a time.

A GGUF file stores tensors under llama.cpp names (``blk.0.attn_q.weight``) in
llama.cpp order (dimensions reversed), most of them quantized in blocks. This
source exposes them as a Hugging Face model would:

* names are mapped back through the ``gguf`` package's name table; a GGUF
  tensor can correspond to several Hugging Face spellings, all kept as aliases
  so the right one is found whatever family the base model belongs to;
* row ranges are cut from the raw blocks before dequantizing (a quantized row
  is a whole number of blocks), so partial reads stay cheap;
* ``convert_hf_to_gguf.py`` permutes the rows of ``q_proj`` / ``k_proj`` for
  llama-family architectures (rotary embedding layout); that permutation is
  reversed so the tensors line up with the original weights;
* a ``config.json`` equivalent is rebuilt from the GGUF metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.quant import QuantScheme
from models.source import ModelError, Rows, TensorSource

# Architectures whose Q/K rows convert_hf_to_gguf.py permutes (LlamaModel.permute).
PERMUTED_ARCHITECTURES: frozenset[str] = frozenset({"llama", "granite", "granitemoe", "deci"})
PREFERRED_TOP_LEVEL: tuple[str, ...] = ("model.embed_tokens", "model.norm", "lm_head")
# Llama-style module names, preferred when a GGUF tensor maps to several Hugging Face spellings.
LLAMA_MODULES: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_layernorm",
     "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"}
)  # fmt: skip
TENSOR_SUFFIXES: tuple[str, ...] = ("weight", "bias")
QUERY_KEY_HEADS: dict[str, str] = {"attn_q": "head_count", "attn_k": "head_count_kv"}


class GgufSource(TensorSource):
    """Reads a local ``.gguf`` file as a float32 Hugging Face-named model."""

    def __init__(self, path: str) -> None:
        """Parse the header and build the name mapping (tensor data stays on disk).

        Raises:
            ModelError: when the file is not a readable GGUF file.
        """
        import gguf

        super().__init__(path)
        try:
            self._reader = gguf.GGUFReader(path)
        except Exception as error:
            raise ModelError(f"{path} is not a readable GGUF file: {error}") from error
        self.architecture = str(self.metadata("general.architecture", "unknown"))
        self._tensors = {tensor.name: tensor for tensor in self._reader.tensors}
        self._aliases = _hf_aliases(gguf, self.architecture, int(self._arch_value("block_count", 0)), self._tensors)
        self._by_alias = {alias: gguf_name for gguf_name, names in self._aliases.items() for alias in names}

    def names(self) -> tuple[str, ...]:
        """Preferred Hugging Face name of every tensor."""
        return tuple(names[0] for names in self._aliases.values())

    def has(self, name: str) -> bool:
        """Whether any tensor answers to ``name``."""
        return name in self._by_alias

    def aliases(self, name: str) -> tuple[str, ...]:
        """Every Hugging Face spelling of the tensor called ``name``."""
        return self._aliases[self._by_alias[name]]

    def shape(self, name: str) -> list[int]:
        """Shape in Hugging Face order (GGUF stores dimensions reversed)."""
        return [int(size) for size in reversed(self._tensors[self._by_alias[name]].shape)]

    def quantization(self, name: str) -> QuantScheme:
        """The llama.cpp type of tensor ``name`` (F32, F16, Q4_K, ...)."""
        return QuantScheme(name=self._tensors[self._by_alias[name]].tensor_type.name, family="gguf")

    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Dequantize tensor ``name`` (or rows ``rows``) to float32."""
        from gguf import quants

        gguf_name = self._by_alias[name]
        tensor = self._tensors[gguf_name]
        heads = self._permutation_heads(gguf_name)
        data = tensor.data if rows is None or heads else tensor.data[rows[0] : rows[1]]
        # A copy: lossless types come back as read-only views of the memory-mapped file.
        values = np.array(quants.dequantize(np.asarray(data), tensor.tensor_type), dtype=np.float32)
        shape = self.shape(name)
        result = torch.from_numpy(values).reshape(-1, *shape[1:]) if shape else torch.tensor([])
        if heads:
            result = unpermute_rows(result, heads)
            result = result if rows is None else result[rows[0] : rows[1]]
        return result

    def config(self) -> dict[str, Any]:
        """A ``config.json`` equivalent rebuilt from the GGUF metadata."""
        hidden = self._arch_value("embedding_length")
        heads = _first(self._arch_value("attention.head_count"))
        tokens = self.metadata("tokenizer.ggml.tokens", None)
        return {
            "model_type": self.architecture,
            "hidden_size": hidden,
            "intermediate_size": _first(self._arch_value("feed_forward_length")),
            "num_hidden_layers": self._arch_value("block_count"),
            "num_attention_heads": heads,
            "num_key_value_heads": _first(self._arch_value("attention.head_count_kv", heads)),
            "head_dim": self._arch_value("attention.key_length", None)
            or (hidden // heads if hidden and heads else None),
            "vocab_size": len(tokens) if tokens is not None else self._arch_value("vocab_size", None),
            "tie_word_embeddings": "output.weight" not in self._tensors,
            "rms_norm_eps": self._arch_value("attention.layer_norm_rms_epsilon", None),
            "gguf_file_type": self.metadata("general.file_type", None),
        }

    def metadata(self, key: str, default: Any = None) -> Any:
        """One metadata value of the file, or ``default``."""
        field = self._reader.fields.get(key)
        return field.contents() if field is not None else default

    def _arch_value(self, key: str, default: Any = None) -> Any:
        """An architecture-scoped metadata value (``<arch>.<key>``)."""
        return self.metadata(f"{self.architecture}.{key}", default)

    def _permutation_heads(self, gguf_name: str) -> int | None:
        """Head count used to undo the Q/K row permutation, or None when not permuted."""
        result = None
        if self.architecture in PERMUTED_ARCHITECTURES:
            module = gguf_name.split(".")[-2]
            heads_key = QUERY_KEY_HEADS.get(module)
            if heads_key is not None:
                result = int(_first(self._arch_value(f"attention.{heads_key}")))
        return result


def unpermute_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Reverse ``LlamaModel.permute`` of convert_hf_to_gguf.py on the rows of a Q/K tensor.

    The converter interleaves the two rotary halves of every head
    (``reshape(heads, 2, d/2) -> swapaxes``); this puts them back.
    """
    rows = weight.shape[0]
    return weight.reshape(heads, rows // heads // 2, 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def permute_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """``LlamaModel.permute`` of convert_hf_to_gguf.py (used to build test files)."""
    rows = weight.shape[0]
    return weight.reshape(heads, 2, rows // heads // 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def _hf_aliases(gguf: Any, architecture: str, blocks: int, tensors: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Map every GGUF tensor name to its Hugging Face spellings, preferred first."""
    reverse = _reverse_name_map(gguf, architecture, blocks)
    result = {}
    for gguf_name in tensors:
        stem, _, suffix = gguf_name.rpartition(".")
        candidates = reverse.get(stem, []) if suffix in TENSOR_SUFFIXES else []
        ranked = sorted(candidates, key=_preference)
        result[gguf_name] = tuple(f"{name}.{suffix}" for name in ranked) or (gguf_name,)
    return result


def _reverse_name_map(gguf: Any, architecture: str, blocks: int) -> dict[str, list[str]]:
    """GGUF stem -> every Hugging Face module path the gguf package maps to it."""
    arch = next((key for key, name in gguf.MODEL_ARCH_NAMES.items() if name == architecture), None)
    result: dict[str, list[str]] = {}
    if arch is not None:
        for hf_name, (_, gguf_stem) in gguf.get_tensor_name_map(arch, blocks).mapping.items():
            result.setdefault(gguf_stem, []).append(hf_name)
    return result


def _preference(hf_name: str) -> tuple[bool, bool, bool, int]:
    """Sort key: Llama-style ``model.layers.N.(self_attn|mlp).<module>`` names first."""
    top_level = hf_name in PREFERRED_TOP_LEVEL
    standard = hf_name.startswith("model.layers.") or top_level
    llama_module = hf_name.rsplit(".", 1)[-1] in LLAMA_MODULES or top_level
    conventional = ".self_attn." in hf_name or ".mlp." in hf_name or top_level
    return not standard, not llama_module, not conventional, len(hf_name)


def _first(value: Any) -> Any:
    """The first element of a per-layer metadata array, or the value itself."""
    return value[0] if isinstance(value, list | tuple) and value else value
