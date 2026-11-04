"""PEFT LoRA adapters: reading them, and applying them to a base model lazily.

PEFT stores, for each adapted ``nn.Linear`` of weight ``W`` (out x in), a pair
``lora_A`` (r x in) and ``lora_B`` (out x r). The effective update is
``ΔW = (B @ A) * scale`` with ``scale = alpha / r``, or ``alpha / sqrt(r)`` under
rsLoRA. Embedding layers use ``lora_embedding_A`` (r x vocab) and
``lora_embedding_B`` (dim x r), so their update is transposed:
``ΔW = (B @ A).T * scale``.

Key ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`` adapts
base tensor ``model.layers.0.self_attn.q_proj.weight``: strip
``base_model.model.`` once and drop the ``lora_A`` component.

Any other tensor of the adapter is a module saved in full (``modules_to_save``,
typically ``lm_head`` or ``embed_tokens``): it replaces the base tensor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

from models.local import read_json
from models.source import ModelError, Rows, TensorSource, UnsupportedModel, slice_rows

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
PEFT_PREFIX = "base_model.model."
LINEAR_A, LINEAR_B = ".lora_A", ".lora_B"
EMBEDDING_A, EMBEDDING_B = ".lora_embedding_A", ".lora_embedding_B"
SAVED_MODULE_MARKERS: tuple[str, ...] = (".modules_to_save.default", ".modules_to_save", ".original_module")
# DoRA rescales each column after the low-rank update; ΔW alone would be wrong.
UNSUPPORTED_MARKERS: tuple[str, ...] = ("lora_magnitude_vector",)


@dataclass(frozen=True)
class LoraConfig:
    """The adapter hyper-parameters that matter for reconstructing ΔW."""

    rank: int
    alpha: float
    use_rslora: bool
    target_modules: tuple[str, ...]
    base_model: str | None

    @property
    def scale(self) -> float:
        """Multiplier applied to ``B @ A``."""
        return self.alpha / (self.rank**0.5) if self.use_rslora else self.alpha / self.rank

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraConfig:
        """Build from a parsed ``adapter_config.json``.

        Raises:
            UnsupportedModel: for non-LoRA PEFT methods (IA3, prefix tuning, ...).
        """
        peft_type = str(data.get("peft_type", "LORA")).upper()
        if peft_type != "LORA":
            raise UnsupportedModel(f"PEFT method {peft_type} is not supported (LoRA only)")
        targets = data.get("target_modules") or ()
        return cls(
            rank=int(data["r"]),
            alpha=float(data["lora_alpha"]),
            use_rslora=bool(data.get("use_rslora")),
            target_modules=(targets,) if isinstance(targets, str) else tuple(sorted(targets)),
            base_model=data.get("base_model_name_or_path"),
        )


@dataclass(frozen=True)
class LoraPair:
    """One adapted base tensor and the adapter keys of its low-rank factors."""

    base_name: str
    key_a: str
    key_b: str
    embedding: bool


class LoraAdapter:
    """A local PEFT LoRA adapter folder (``adapter_config.json`` + weights)."""

    def __init__(self, folder: str) -> None:
        """Read the configuration and index the adapter tensors.

        Raises:
            ModelError: when the folder is not a readable LoRA adapter.
            UnsupportedModel: for DoRA or non-LoRA PEFT methods.
        """
        self.folder = folder
        self.weights_path = os.path.join(folder, ADAPTER_WEIGHTS)
        if not os.path.isfile(os.path.join(folder, ADAPTER_CONFIG)) or not os.path.isfile(self.weights_path):
            raise ModelError(f"{folder} is not a LoRA adapter (needs {ADAPTER_CONFIG} and {ADAPTER_WEIGHTS})")
        self.config = LoraConfig.from_dict(read_json(os.path.join(folder, ADAPTER_CONFIG)))
        with safe_open(self.weights_path, framework="pt") as handle:
            keys = list(handle.keys())
        unsupported = [key for key in keys if any(marker in key for marker in UNSUPPORTED_MARKERS)]
        if unsupported:
            raise UnsupportedModel(f"DoRA adapters are not supported ({unsupported[0]})")
        self.pairs = _pairs(keys)
        self.saved_modules = _saved_modules(keys)

    def delta(self, pair: LoraPair, rows: Rows | None = None) -> torch.Tensor:
        """ΔW of one adapted tensor (or of its rows ``rows``), as float32."""
        lora_a, lora_b = self.tensor(pair.key_a), self.tensor(pair.key_b)
        result: torch.Tensor
        if pair.embedding:
            columns = lora_a if rows is None else lora_a[:, rows[0] : rows[1]]
            result = (lora_b @ columns).T * self.config.scale
        else:
            result = (slice_rows(lora_b, rows) @ lora_a) * self.config.scale
        return result

    def tensor(self, key: str) -> torch.Tensor:
        """Load one adapter tensor as float32."""
        with safe_open(self.weights_path, framework="pt") as handle:
            result = handle.get_tensor(key).float()
        return result


def _pairs(keys: list[str]) -> list[LoraPair]:
    """Pair every ``A`` factor with its ``B`` factor."""
    key_set = set(keys)
    result = []
    for key in keys:
        embedding = EMBEDDING_A in key
        if not embedding and f"{LINEAR_A}." not in key:
            continue
        marker_a, marker_b = (EMBEDDING_A, EMBEDDING_B) if embedding else (LINEAR_A, LINEAR_B)
        key_b = key.replace(marker_a, marker_b)
        if key_b in key_set:
            result.append(LoraPair(base_tensor_name(key, marker_a), key, key_b, embedding))
    return result


def _saved_modules(keys: list[str]) -> dict[str, str]:
    """Map base tensor name -> adapter key, for modules the adapter saves in full."""
    lora_markers = (LINEAR_A, LINEAR_B, EMBEDDING_A, EMBEDDING_B)
    result = {}
    for key in keys:
        if not any(marker in key for marker in lora_markers):
            name = key.removeprefix(PEFT_PREFIX)
            for marker in SAVED_MODULE_MARKERS:
                name = name.replace(marker, "")
            result[name] = key
    return result


def base_tensor_name(key: str, marker: str) -> str:
    """Map an adapter factor key to the name of the base tensor it adapts.

    ``base_model.model.model.layers.0.mlp.up_proj.lora_A.weight`` becomes
    ``model.layers.0.mlp.up_proj.weight``; ``...embed_tokens.lora_embedding_A``
    becomes ``...embed_tokens.weight``.
    """
    stem = key.removeprefix(PEFT_PREFIX).split(marker)[0]
    return f"{stem}.weight"


class AdaptedSource(TensorSource):
    """A base model with a LoRA adapter applied on the fly, one tensor at a time.

    Equivalent to reading the merged model, without ever writing it.
    """

    def __init__(self, base: TensorSource, adapter: LoraAdapter) -> None:
        super().__init__(adapter.folder)
        self.base = base
        self.adapter = adapter
        self._pairs: dict[str, LoraPair] = {}
        for pair in adapter.pairs:
            name = base.resolve(pair.base_name)
            if name is not None:
                self._pairs[name] = pair
        self._saved = {base.resolve(name) or name: key for name, key in adapter.saved_modules.items()}
        self.unmatched = [pair.base_name for pair in adapter.pairs if base.resolve(pair.base_name) is None]

    def names(self) -> tuple[str, ...]:
        """The base model's tensor names."""
        return self.base.names()

    def has(self, name: str) -> bool:
        """Whether the base model has tensor ``name``."""
        return self.base.has(name)

    def shape(self, name: str) -> list[int]:
        """Shape of tensor ``name`` (adapters never change shapes)."""
        return self.base.shape(name)

    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """The base tensor with the adapter's update applied."""
        result: torch.Tensor
        if name in self._saved:
            result = slice_rows(self.adapter.tensor(self._saved[name]), rows)
        elif name in self._pairs:
            result = self.base.load(name, rows) + self.adapter.delta(self._pairs[name], rows)
        else:
            result = self.base.load(name, rows)
        return result

    def config(self) -> dict[str, Any]:
        """The base model's configuration."""
        return self.base.config()

    def adapted_pairs(self) -> dict[str, LoraPair]:
        """Base tensor name -> LoRA pair, for every adapted tensor found in the base."""
        return dict(self._pairs)

    def saved_names(self) -> list[str]:
        """Base tensor names the adapter replaces with a fully saved module."""
        return list(self._saved)

    def is_adapted(self, name: str) -> bool:
        """Whether the adapter changes tensor ``name``."""
        return name in self._pairs or name in self._saved
