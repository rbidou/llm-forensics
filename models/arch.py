"""Normalised architecture facts from a Hugging Face ``config.json``.

Configs name the same dimension differently across families (``hidden_size``,
``n_embd``, ``d_model``...), and multimodal configs nest the language model's
under ``text_config``. Analyses only ever see :class:`Architecture`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.source import ModelError

# Dimensions that must match for two models to be compared tensor by tensor.
SHAPE_FIELDS: tuple[str, ...] = ("hidden", "intermediate", "n_heads", "n_kv", "head_dim")


@dataclass(frozen=True)
class Architecture:
    """The dimensions the analyses rely on."""

    model_type: str | None
    hidden: int
    intermediate: int | None
    n_heads: int
    n_kv: int
    head_dim: int
    layers: int | None
    vocab: int | None
    activation: str
    tied_embeddings: bool | None
    quantization: str | None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Architecture:
        """Normalise a config dict.

        Raises:
            ModelError: when the config lacks a hidden size or a head count.
        """
        merged = {**config, **(config.get("text_config") or {})}
        hidden = merged.get("hidden_size") or merged.get("n_embd") or merged.get("d_model")
        n_heads = merged.get("num_attention_heads") or merged.get("n_head")
        if not hidden or not n_heads:
            raise ModelError("Unsupported config: no hidden_size / num_attention_heads")
        quantization = merged.get("quantization_config") or {}
        return cls(
            model_type=merged.get("model_type"),
            hidden=int(hidden),
            intermediate=merged.get("intermediate_size") or merged.get("n_inner"),
            n_heads=int(n_heads),
            n_kv=int(merged.get("num_key_value_heads") or n_heads),
            head_dim=int(merged.get("head_dim") or hidden // n_heads),
            layers=merged.get("num_hidden_layers") or merged.get("n_layer"),
            vocab=merged.get("vocab_size"),
            activation=merged.get("hidden_act") or merged.get("hidden_activation") or "silu",
            tied_embeddings=merged.get("tie_word_embeddings"),
            quantization=quantization.get("quant_method"),
        )

    def compatible(self, other: Architecture) -> bool:
        """Whether every tensor shape of the two models can match."""
        return all(getattr(self, field) == getattr(other, field) for field in SHAPE_FIELDS)

    def describe(self) -> str:
        """One-line summary for reports."""
        return (
            f"{self.model_type or '?'} hidden={self.hidden} inter={self.intermediate} layers={self.layers} "
            f"heads={self.n_heads}/{self.n_kv} head_dim={self.head_dim} vocab={self.vocab}"
        )
