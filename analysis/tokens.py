"""Turning token ids into readable text.

Analyses only need ``decode`` and a vocabulary size, so they depend on the
:class:`TokenDecoder` protocol rather than on transformers. Hugging Face
tokenizers satisfy it as they are; tests pass a trivial fake.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from models import GGUF_AUTO, Location, ModelKind, ModelRef, choose_gguf

TOKENIZER_FILES: tuple[str, ...] = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json")


class TokenDecoder(Protocol):
    """The part of a tokenizer the analyses use."""

    def decode(self, token_ids: list[int]) -> str:
        """Text of a sequence of token ids."""
        ...

    def __len__(self) -> int:
        """Number of tokens in the vocabulary (including added tokens)."""
        ...


@dataclass(frozen=True)
class VocabInfo:
    """A tokenizer's vocabulary as the Vocabulary & Embeddings analysis reads it.

    ``id_to_text`` maps every token id (added tokens included) to its string,
    which is what lets two vocabularies be compared id by id. ``specials`` maps a
    role (``bos_token``, ``eos_token``, ...) to the token text that fills it, and
    ``normalizer`` is a signature of the tokenizer's normalizer (``None`` when the
    format does not expose one), so a changed normalizer can be flagged.
    """

    id_to_text: dict[int, str]
    specials: dict[str, str] = field(default_factory=dict)
    normalizer: str | None = None

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.id_to_text)


def vocab_info(tokenizer: Any) -> VocabInfo:
    """Read a Hugging Face tokenizer's vocabulary into a :class:`VocabInfo`.

    The base vocabulary comes from ``get_vocab`` and the added tokens from
    ``added_tokens_decoder``, so special tokens added after training are kept
    with their real ids. A ``list`` special (``additional_special_tokens``) is
    stored as a comma-joined string.

    Args:
        tokenizer: a Hugging Face tokenizer (or anything with the same surface).

    Returns:
        The vocabulary, ready to diff against another.
    """
    id_to_text = {index: text for text, index in tokenizer.get_vocab().items()}
    for index, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        id_to_text[int(index)] = str(token)
    specials = {
        role: ", ".join(value) if isinstance(value, list) else str(value)
        for role, value in getattr(tokenizer, "special_tokens_map", {}).items()
        if value
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None) if backend is not None else None
    return VocabInfo(id_to_text=id_to_text, specials=specials, normalizer=str(normalizer) if normalizer else None)


def token_text(decoder: TokenDecoder, token_id: int) -> str:
    """Text of a single token."""
    return decoder.decode([token_id])


def token_texts(decoder: TokenDecoder, token_ids: Iterable[int]) -> list[str]:
    """Texts of several tokens, one each."""
    return [token_text(decoder, token_id) for token_id in token_ids]


def load_tokenizer(model: ModelRef, token: str | None = None, gguf_file: str | None = None) -> Any:
    """Load the Hugging Face tokenizer of a model.

    An adapter without tokenizer files falls back to its base; a GGUF model's
    tokenizer is read from the GGUF file itself (``gguf_file``; None or
    ``GGUF_AUTO``: the most precise file of the folder or repository).

    Raises:
        OSError: when no tokenizer can be found for the model or its base.
    """
    from transformers import AutoTokenizer

    result: Any
    if model.kind is ModelKind.GGUF:
        explicit = gguf_file if gguf_file != GGUF_AUTO else None
        folder, filename = gguf_location(model, explicit or choose_gguf(model.gguf_files) or model.ref)
        result = AutoTokenizer.from_pretrained(folder, gguf_file=filename, token=token)
    else:
        source = model.ref
        if model.kind is ModelKind.ADAPTER and not _has_tokenizer(model) and model.base_model:
            source = model.base_model
        result = AutoTokenizer.from_pretrained(source, token=token)
    return result


def gguf_location(model: ModelRef, chosen: str) -> tuple[str, str]:
    """(repository id or folder, file name) of a GGUF file, as transformers expects them."""
    result = (os.path.dirname(os.path.abspath(chosen)), os.path.basename(chosen))
    if model.location is Location.HUB or os.path.isdir(model.ref):
        result = (model.ref, chosen)
    return result


def _has_tokenizer(model: ModelRef) -> bool:
    """Whether the model's own files include a tokenizer."""
    names = {os.path.basename(name) for name in model.files}
    return any(name in names for name in TOKENIZER_FILES)
