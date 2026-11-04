"""Hugging Face Hub models, read tensor by tensor with HTTP Range requests.

A safetensors file is an 8-byte header length, a JSON header giving each
tensor's dtype, shape and byte offsets, then the raw data. Reading the header
and then only the needed byte ranges lets an analysis compare two 30 GB models
while downloading a few hundred megabytes.

:class:`HubRepo` is the only place that talks to the network; it is injected
into :class:`HubSource` so tests can serve files from disk instead.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from collections.abc import Callable
from typing import Any

import torch

from models.local import ADAPTER_PREFIX, read_json
from models.source import ModelError, Rows, TensorSource

SAFETENSORS_INDEX = "model.safetensors.index.json"
HEADER_LENGTH_BYTES = 8
RANGE_ATTEMPTS = 4
RANGE_TIMEOUT_SECONDS = 120
FATAL_HTTP_STATUSES = frozenset({401, 403, 404})
HTTP_PARTIAL_CONTENT = 206

SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class HubRepo:
    """Network access to one repository at one revision."""

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        revision: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.revision = revision
        self._sleep = sleep
        self._session: Any = None

    def list_files(self) -> list[str]:
        """Every file path in the repository."""
        from huggingface_hub import HfApi

        return list(HfApi().list_repo_files(self.repo_id, revision=self.revision, token=self.token))

    def download(self, filename: str) -> str:
        """Download one file into the HF cache and return its local path."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.repo_id, filename, revision=self.revision, token=self.token)

    def read_json(self, filename: str) -> dict[str, Any]:
        """Download and parse one JSON file."""
        return read_json(self.download(filename))

    def read_range(self, filename: str, start: int, end: int) -> bytes:
        """Fetch bytes ``[start, end)`` of a file, retrying transient failures.

        Raises:
            ModelError: on an authorisation/not-found answer, or after every attempt failed.
        """
        url, headers = self._range_request(filename, start, end)
        result: bytes | None = None
        failure = ""
        for attempt in range(RANGE_ATTEMPTS):
            data, failure, fatal = self._attempt(url, headers, end - start)
            if data is not None or fatal:
                result = data
                break
            if attempt < RANGE_ATTEMPTS - 1:
                self._sleep(1 + 2 * attempt)
        if result is None:
            raise ModelError(f"Cannot read {self.repo_id}/{filename} bytes {start}-{end}: {failure}")
        return result

    def _range_request(self, filename: str, start: int, end: int) -> tuple[str, dict[str, str]]:
        """Build the URL and headers of a Range request."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        headers = dict(build_hf_headers(token=self.token))
        headers["Range"] = f"bytes={start}-{end - 1}"
        return hf_hub_url(self.repo_id, filename, revision=self.revision), headers

    def _attempt(self, url: str, headers: dict[str, str], expected: int) -> tuple[bytes | None, str, bool]:
        """Perform one Range request.

        Returns:
            (data or None, failure description, whether retrying is pointless).
        """
        data, failure, fatal = None, "", False
        try:
            # No stream=True: httpx-based sessions (huggingface_hub >= 1.0) reject it,
            # and a range response is small enough to read in one go anyway.
            response = self.session.get(url, headers=headers, timeout=RANGE_TIMEOUT_SECONDS)
            if response.status_code == HTTP_PARTIAL_CONTENT and len(response.content) == expected:
                data = response.content
            elif response.status_code == HTTP_PARTIAL_CONTENT:
                failure = f"short read {len(response.content)}/{expected} bytes"
            else:
                failure = f"HTTP {response.status_code} (range requests must answer 206)"
                fatal = response.status_code in FATAL_HTTP_STATUSES
        except Exception as error:  # noqa: BLE001 - any transport error is retried
            failure = f"{error.__class__.__name__}: {error}"
        return data, failure, fatal

    @property
    def session(self) -> Any:
        """The HTTP session shared with huggingface_hub (proxies, retries, user agent)."""
        if self._session is None:
            from huggingface_hub.utils import get_session

            self._session = get_session()
        return self._session

    @session.setter
    def session(self, value: Any) -> None:
        """Replace the HTTP session (tests)."""
        self._session = value


class HubSource(TensorSource):
    """Reads individual tensors (or row ranges) from a Hub repository."""

    def __init__(self, repo: HubRepo, files: list[str] | None = None) -> None:
        """Build the tensor -> file index from the shard headers; no weight is downloaded.

        Args:
            repo: network access to the repository.
            files: the repository's file list, when the caller already has it.

        Raises:
            ModelError: when the repository has no safetensors weights.
        """
        super().__init__(repo.repo_id)
        self.repo = repo
        self._files = files if files is not None else repo.list_files()
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}
        self._index = self._build_index()

    def _build_index(self) -> dict[str, str]:
        """Map each tensor name to the file holding it."""
        weights = weight_files(self._files)
        if not weights:
            raise ModelError(f"No safetensors weights in {self.repo.repo_id}")
        index: dict[str, str]
        if SAFETENSORS_INDEX in self._files:
            index = dict(self.repo.read_json(SAFETENSORS_INDEX)["weight_map"])
        else:
            index = {}
            for filename in weights:
                index.update({name: filename for name in self._header(filename)[0] if name != "__metadata__"})
        return index

    def names(self) -> tuple[str, ...]:
        """Every tensor name of every shard."""
        return tuple(self._index)

    def has(self, name: str) -> bool:
        """Whether tensor ``name`` exists (constant time)."""
        return name in self._index

    def config(self) -> dict[str, Any]:
        """The repository's ``config.json`` (cached by huggingface_hub), or an empty dict."""
        return self.repo.read_json("config.json") if "config.json" in self._files else {}

    def shape(self, name: str) -> list[int]:
        """Shape of tensor ``name`` from its shard header."""
        return list(self._header(self._index[name])[0][name]["shape"])

    def load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Fetch tensor ``name`` (or a contiguous row range of it) as float32."""
        filename = self._index[name]
        header, data_start = self._header(filename)
        meta = header[name]
        dtype = SAFETENSORS_DTYPES[meta["dtype"]]
        shape = list(meta["shape"])
        start, end = meta["data_offsets"]
        if rows is not None and shape:
            first, last = rows[0], min(rows[1], shape[0])
            row_bytes = torch.empty(0, dtype=dtype).element_size() * math.prod(shape[1:])
            start, end = start + first * row_bytes, start + last * row_bytes
            shape = [last - first, *shape[1:]]
        buffer = self.repo.read_range(filename, data_start + start, data_start + end)
        return torch.frombuffer(bytearray(buffer), dtype=dtype).reshape(shape).float()

    def _header(self, filename: str) -> tuple[dict[str, Any], int]:
        """Parsed JSON header and data start offset of one shard, cached."""
        if filename not in self._headers:
            (length,) = struct.unpack("<Q", self.repo.read_range(filename, 0, HEADER_LENGTH_BYTES))
            raw = self.repo.read_range(filename, HEADER_LENGTH_BYTES, HEADER_LENGTH_BYTES + length)
            self._headers[filename] = (json.loads(raw), HEADER_LENGTH_BYTES + length)
        return self._headers[filename]


def weight_files(files: list[str]) -> list[str]:
    """Safetensors weight files of a repository, preferring the root folder."""
    weights = [
        path
        for path in files
        if path.endswith(".safetensors") and not os.path.basename(path).startswith(ADAPTER_PREFIX)
    ]
    root = [path for path in weights if "/" not in path]
    return sorted(root or weights)
