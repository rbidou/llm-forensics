"""Local folders of safetensors files (single file or sharded)."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any

import torch
from safetensors import safe_open

from models.source import ModelError, Rows, TensorSource

ADAPTER_PREFIX = "adapter_"


def read_json(path: str) -> dict[str, Any]:
    """Parse a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def safetensors_files(folder: str) -> list[str]:
    """Model weight files of ``folder``, excluding LoRA adapter files, sorted."""
    paths = glob(os.path.join(folder, "*.safetensors"))
    return sorted(path for path in paths if not os.path.basename(path).startswith(ADAPTER_PREFIX))


class LocalSource(TensorSource):
    """Reads tensors from the safetensors files of a local folder."""

    def __init__(self, folder: str) -> None:
        """Index every tensor of the folder.

        Raises:
            ModelError: when the folder holds no safetensors weights.
        """
        super().__init__(folder)
        self.folder = folder
        self._index: dict[str, str] = {}
        for path in safetensors_files(folder):
            with safe_open(path, framework="pt") as handle:
                self._index.update(dict.fromkeys(handle.keys(), path))
        if not self._index:
            raise ModelError(f"No safetensors weights in {folder}")

    def names(self) -> tuple[str, ...]:
        """Every tensor name of every shard."""
        return tuple(self._index)

    def has(self, name: str) -> bool:
        """Whether tensor ``name`` exists (constant time)."""
        return name in self._index

    def shape(self, name: str) -> list[int]:
        """Shape of tensor ``name`` from its shard header."""
        with safe_open(self._index[name], framework="pt") as handle:
            result = list(handle.get_slice(name).get_shape())
        return result

    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Load tensor ``name`` (or a row range of it) as float32."""
        with safe_open(self._index[name], framework="pt") as handle:
            tensor = handle.get_tensor(name) if rows is None else handle.get_slice(name)[rows[0] : rows[1]]
        return tensor.float()

    def config(self) -> dict[str, Any]:
        """The folder's ``config.json``, or an empty dict."""
        path = os.path.join(self.folder, "config.json")
        return read_json(path) if os.path.isfile(path) else {}
