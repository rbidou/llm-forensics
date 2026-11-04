"""Reading tensors from any supported model format.

Typical use::

    model = detect("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    source = open_model(model)
    weight = source.load("model.layers.0.mlp.up_proj.weight")
"""

from models.arch import Architecture
from models.detect import (
    GGUF_AUTO,
    Location,
    ModelKind,
    ModelRef,
    choose_gguf,
    detect,
    format_name,
    gguf_by_precision,
    list_checkpoints,
    open_model,
)
from models.folders import FolderError, FolderListing, filesystem_roots, list_folder
from models.hub_search import HubCandidate, search_hub_models
from models.source import ModelError, Rows, TensorSource, UnsupportedModel

__all__ = [
    "GGUF_AUTO",
    "Architecture",
    "FolderError",
    "FolderListing",
    "HubCandidate",
    "Location",
    "ModelError",
    "ModelKind",
    "ModelRef",
    "Rows",
    "TensorSource",
    "UnsupportedModel",
    "choose_gguf",
    "detect",
    "filesystem_roots",
    "format_name",
    "gguf_by_precision",
    "list_checkpoints",
    "list_folder",
    "open_model",
    "search_hub_models",
]
