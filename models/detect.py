"""Recognising what a model reference points to, and opening it.

A reference is a local path (folder or ``.gguf`` file) or a Hub repository id.
:func:`detect` inspects file names and the config only, never weights, and
returns a :class:`ModelRef`; :func:`open_model` turns that into a
:class:`~models.source.TensorSource`.

Pickle-based weights (``.bin``, ``.pt``) are refused on purpose: unpickling a
file can execute arbitrary code, and the toolkit exists to inspect models whose
origin is in doubt.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.bnb import BnbSource
from models.gguf_source import GgufSource
from models.hub import HubRepo, HubSource
from models.local import LocalSource, read_json
from models.lora import ADAPTER_CONFIG, AdaptedSource, LoraAdapter
from models.source import ModelError, TensorSource, UnsupportedModel

CONFIG_FILE = "config.json"
TRAINER_STATE = "trainer_state.json"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
HUB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w.-]*(/[\w.-]+)?$")
PICKLE_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".ckpt")
SUPPORTED_QUANTIZATION = "bitsandbytes"
GGUF_AUTO = "auto"  # gguf_file value: pick the most precise file of a GGUF folder or repository
# GGUF types from most to least precise: the default pick when a repository offers several files.
GGUF_PRECISION_ORDER: tuple[str, ...] = (
    "F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_1", "Q4_0", "IQ4_XS", "IQ4_NL", "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
)  # fmt: skip

RepoFactory = Callable[[str, str | None, str | None], HubRepo]


class ModelKind(StrEnum):
    """How a model's weights are stored."""

    FULL = "full"  # complete safetensors weights (base, merged or fully fine-tuned)
    ADAPTER = "adapter"  # PEFT LoRA adapter, needs its base model
    GGUF = "gguf"  # llama.cpp file, possibly quantized
    BNB = "bnb"  # bitsandbytes 4/8-bit safetensors


class Location(StrEnum):
    """Where a model lives."""

    LOCAL = "local"
    HUB = "hub"


@dataclass(frozen=True)
class ModelRef:
    """What a reference points to, as far as file names and config tell."""

    ref: str
    location: Location
    kind: ModelKind
    files: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict, compare=False)
    base_model: str | None = None
    checkpoint_step: int | None = None
    quantization: str | None = None
    gguf_files: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Short human description, e.g. ``LoRA adapter (checkpoint 500, local)``."""
        names = {
            ModelKind.FULL: "full weights",
            ModelKind.ADAPTER: "LoRA adapter",
            ModelKind.GGUF: "GGUF",
            ModelKind.BNB: f"bitsandbytes {self.quantization or ''}".strip(),
        }
        details = [f"checkpoint {self.checkpoint_step}"] if self.checkpoint_step is not None else []
        return f"{names[self.kind]} ({', '.join([*details, self.location.value])})"


FORMAT_NAMES: dict[ModelKind, str] = {
    ModelKind.FULL: "safetensors",
    ModelKind.ADAPTER: "LoRA adapter",
    ModelKind.GGUF: "GGUF",
    ModelKind.BNB: "bitsandbytes",
}


def format_name(model: ModelRef) -> str:
    """A short name of how a model's weights are stored, for the report listings.

    Examples: ``safetensors``, ``GGUF``, ``LoRA adapter``, ``bitsandbytes 4-bit``.

    Args:
        model: a detection result.

    Returns:
        The storage format, with the quantization appended for bitsandbytes when it is known.
    """
    name = FORMAT_NAMES[model.kind]
    return f"{name} {model.quantization}".strip() if model.kind is ModelKind.BNB and model.quantization else name


def detect(
    ref: str,
    token: str | None = None,
    revision: str | None = None,
    repo_factory: RepoFactory = HubRepo,
) -> ModelRef:
    """Classify a local path or Hub id.

    Raises:
        ModelError: when nothing usable is found at ``ref``.
        UnsupportedModel: for pickle weights or unsupported quantization methods.
    """
    result: ModelRef
    if os.path.isfile(ref):
        result = _detect_local_file(ref)
    elif os.path.isdir(ref):
        result = _detect_local_folder(ref)
    elif HUB_ID_PATTERN.match(ref) and not _looks_like_path(ref):
        result = _detect_hub(repo_factory(ref, token, revision))
    else:
        raise ModelError(f"{ref} is neither an existing path nor a Hugging Face repository id")
    return result


def open_model(
    model: ModelRef,
    base: str | None = None,
    token: str | None = None,
    revision: str | None = None,
    repo_factory: RepoFactory = HubRepo,
    gguf_file: str | None = None,
) -> TensorSource:
    """Open a detected model for reading.

    Args:
        model: the detection result.
        base: base model for an adapter, overriding the one it declares.
        token: Hub token for private or gated repositories.
        revision: Hub revision of ``model`` (not of the base).
        repo_factory: builds Hub access objects (tests inject a fake).
        gguf_file: which file of a GGUF folder or repository to read, when it holds several;
            ``GGUF_AUTO`` picks the most precise one.

    Raises:
        ModelError: when an adapter's base cannot be determined or opened, or a
            GGUF file cannot be chosen.
    """
    result: TensorSource
    if model.kind is ModelKind.ADAPTER:
        result = _open_adapter(model, base, token, revision, repo_factory)
    elif model.kind is ModelKind.GGUF:
        result = GgufSource(_gguf_path(model, gguf_file, token, revision, repo_factory))
    else:
        raw = _open_safetensors(model, token, revision, repo_factory)
        result = BnbSource(raw) if model.kind is ModelKind.BNB else raw
    return result


def choose_gguf(files: Iterable[str]) -> str | None:
    """The most precise GGUF file of a list (by the type named in its file name)."""
    candidates = [name for name in files if name.lower().endswith(".gguf")]
    return min(candidates, key=_gguf_precision_rank, default=None)


def gguf_by_precision(files: Iterable[str]) -> list[str]:
    """GGUF file names from the most precise to the least (by the type named in each)."""
    remaining = [name for name in files if name.lower().endswith(".gguf")]
    ordered: list[str] = []
    while remaining:
        best = choose_gguf(remaining) or remaining[0]
        ordered.append(best)
        remaining.remove(best)
    return ordered


def _gguf_precision_rank(filename: str) -> tuple[int, str]:
    """Position of the file's quantization type in GGUF_PRECISION_ORDER (unknown last)."""
    upper = os.path.basename(filename).upper()
    found = [
        index
        for index, name in enumerate(GGUF_PRECISION_ORDER)
        if re.search(rf"(^|[^A-Z0-9]){name}([^A-Z0-9]|$)", upper)
    ]
    return (min(found) if found else len(GGUF_PRECISION_ORDER), filename)


def _open_safetensors(
    model: ModelRef, token: str | None, revision: str | None, repo_factory: RepoFactory
) -> TensorSource:
    """Open safetensors weights, local or on the Hub."""
    result: TensorSource
    if model.location is Location.LOCAL:
        result = LocalSource(model.ref)
    else:
        result = HubSource(repo_factory(model.ref, token, revision), list(model.files))
    return result


def _gguf_path(
    model: ModelRef, gguf_file: str | None, token: str | None, revision: str | None, repo_factory: RepoFactory
) -> str:
    """Local path of the GGUF file to read (downloaded into the HF cache for Hub models).

    Raises:
        ModelError: when several files exist and none was chosen, or the choice is unknown.
    """
    if gguf_file == GGUF_AUTO:
        gguf_file = choose_gguf(model.gguf_files)
    if gguf_file is None and len(model.gguf_files) > 1:
        listed = ", ".join(os.path.basename(name) for name in model.gguf_files)
        raise ModelError(f"{model.ref} holds several GGUF files; choose one: {listed}")
    chosen = gguf_file or model.gguf_files[0]
    matches = [name for name in model.gguf_files if name == chosen or os.path.basename(name) == chosen]
    if not matches:
        raise ModelError(f"{chosen} is not one of the GGUF files of {model.ref}")
    result = matches[0]
    if model.location is Location.HUB:
        result = repo_factory(model.ref, token, revision).download(matches[0])
    elif not os.path.isabs(result) and os.path.isdir(model.ref):
        result = os.path.join(model.ref, result)
    return result


def list_checkpoints(folder: str) -> list[str]:
    """Training checkpoint sub-folders of ``folder`` (``checkpoint-N``), by step."""
    entries = [entry for entry in os.listdir(folder) if os.path.isdir(os.path.join(folder, entry))]
    steps = sorted((int(match.group(1)), entry) for entry in entries if (match := CHECKPOINT_PATTERN.match(entry)))
    return [os.path.join(folder, entry) for _, entry in steps]


def _open_adapter(
    model: ModelRef, base: str | None, token: str | None, revision: str | None, repo_factory: RepoFactory
) -> TensorSource:
    """Open an adapter applied to its (declared or overridden) base model."""
    base_ref = base or model.base_model
    if not base_ref:
        raise ModelError(f"{model.ref} does not declare its base model; specify one")
    if model.location is Location.HUB:
        folder = _download_adapter(repo_factory(model.ref, token, revision))
    else:
        folder = model.ref
    try:
        base_model = detect(base_ref, token=token, repo_factory=repo_factory)
    except ModelError as error:
        raise ModelError(f"Base model of {model.ref} not usable ({error}); specify the base explicitly") from error
    if base_model.kind is ModelKind.ADAPTER:
        raise ModelError(f"The base of {model.ref} ({base_ref}) is itself an adapter")
    return AdaptedSource(open_model(base_model, token=token, repo_factory=repo_factory), LoraAdapter(folder))


def _download_adapter(repo: HubRepo) -> str:
    """Download an adapter's config and weights; return the snapshot folder."""
    repo.download("adapter_model.safetensors")
    return os.path.dirname(repo.download(ADAPTER_CONFIG))


def _detect_local_file(path: str) -> ModelRef:
    """Classify a single local file."""
    if not path.lower().endswith(".gguf"):
        raise ModelError(f"{path}: only .gguf files can be opened directly; give the model folder instead")
    return ModelRef(ref=path, location=Location.LOCAL, kind=ModelKind.GGUF, files=(path,), gguf_files=(path,))


def _detect_local_folder(folder: str) -> ModelRef:
    """Classify a local folder from its file names and config."""
    files = tuple(sorted(os.listdir(folder)))
    config = _read_optional_json(folder, CONFIG_FILE, files)
    if not _has_weights(files):
        checkpoints = list_checkpoints(folder)
        if checkpoints:
            listed = ", ".join(os.path.basename(path) for path in checkpoints)
            raise ModelError(f"{folder} is a training output folder; pick one checkpoint: {listed}")
    return _classify(folder, Location.LOCAL, files, config, _checkpoint_step(folder, files))


def _detect_hub(repo: HubRepo) -> ModelRef:
    """Classify a Hub repository from its file list and config."""
    try:
        files = tuple(sorted(repo.list_files()))
    except Exception as error:
        raise ModelError(f"Cannot list {repo.repo_id} on the Hub: {error}") from error
    config = repo.read_json(CONFIG_FILE) if CONFIG_FILE in files else {}
    adapter_config = repo.read_json(ADAPTER_CONFIG) if ADAPTER_CONFIG in files else {}
    return _classify(repo.repo_id, Location.HUB, files, config, None, adapter_config)


def _classify(
    ref: str,
    location: Location,
    files: tuple[str, ...],
    config: dict[str, Any],
    step: int | None,
    adapter_config: dict[str, Any] | None = None,
) -> ModelRef:
    """Decide the model kind from what was found at the reference.

    Raises:
        ModelError, UnsupportedModel: see :func:`detect`.
    """
    gguf_files = tuple(name for name in files if name.lower().endswith(".gguf"))
    quantization = (config.get("quantization_config") or {}).get("quant_method")
    common: dict[str, Any] = {"ref": ref, "location": location, "files": files, "config": config}
    result: ModelRef
    if ADAPTER_CONFIG in files:
        declared = adapter_config if adapter_config is not None else read_json(os.path.join(ref, ADAPTER_CONFIG))
        base = declared.get("base_model_name_or_path")
        result = ModelRef(**common, kind=ModelKind.ADAPTER, base_model=base, checkpoint_step=step)
    elif quantization is not None and quantization != SUPPORTED_QUANTIZATION:
        raise UnsupportedModel(f"{ref}: {quantization} quantization is not supported (bitsandbytes and GGUF are)")
    elif quantization is not None and _has_weights(files):
        result = ModelRef(**common, kind=ModelKind.BNB, quantization=_bnb_mode(config), checkpoint_step=step)
    elif _has_weights(files):
        result = ModelRef(**common, kind=ModelKind.FULL, checkpoint_step=step)
    elif gguf_files:
        result = ModelRef(**common, kind=ModelKind.GGUF, gguf_files=gguf_files)
    elif any(name.endswith(PICKLE_SUFFIXES) for name in files):
        raise UnsupportedModel(
            f"{ref}: only pickle weights (.bin/.pt) found. They are never loaded because unpickling can run "
            "arbitrary code; convert the model to safetensors first."
        )
    else:
        raise ModelError(f"{ref}: no model weights found (safetensors, GGUF or LoRA adapter)")
    return result


def _has_weights(files: Iterable[str]) -> bool:
    """Whether the file list holds safetensors model weights (not adapter files)."""
    return any(name.endswith(".safetensors") and not os.path.basename(name).startswith("adapter_") for name in files)


def _bnb_mode(config: dict[str, Any]) -> str:
    """``nf4``, ``fp4`` or ``int8`` from a bitsandbytes quantization config."""
    settings = config.get("quantization_config") or {}
    result = "int8"
    if settings.get("load_in_4bit"):
        result = str(settings.get("bnb_4bit_quant_type", "fp4"))
    return result


def _checkpoint_step(folder: str, files: tuple[str, ...]) -> int | None:
    """Training step of a ``checkpoint-N`` folder, or None for a regular model."""
    step: int | None = None
    if TRAINER_STATE in files:
        step = read_json(os.path.join(folder, TRAINER_STATE)).get("global_step")
    match = CHECKPOINT_PATTERN.match(os.path.basename(os.path.normpath(folder)))
    if step is None and match:
        step = int(match.group(1))
    return step


def _read_optional_json(folder: str, name: str, files: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``folder/name`` when it is listed in ``files``, else return an empty dict."""
    return read_json(os.path.join(folder, name)) if name in files else {}


def _looks_like_path(ref: str) -> bool:
    """Whether ``ref`` is clearly meant as a filesystem path (that does not exist)."""
    return ref.startswith((".", "/", "\\", "~")) or os.path.isabs(ref) or "\\" in ref or ref.count("/") > 1
