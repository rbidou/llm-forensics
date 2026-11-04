"""Common shape of every analysis result, and its conversion to JSON.

Each analysis returns a dataclass tree whose leaves are plain Python values
(no tensors). :func:`to_jsonable` turns any such tree into JSON-compatible data:
the JSON report is exactly this, and the HTML report and both front-ends render
from the same objects.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from settings import __version__


@dataclass
class ResultMeta:
    """What was analysed, with which settings, and what to keep in mind."""

    analysis: str  # weights | shifts | lineage | provenance | behaviour
    inputs: dict[str, str]
    parameters: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    tool_version: str = __version__
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    """Convert a result tree to JSON-compatible data.

    Dataclasses become dicts, tuples become lists, enums their value, and
    non-finite floats ``None`` (JSON has no NaN).
    """
    result: Any = value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {item.name: to_jsonable(getattr(value, item.name)) for item in dataclasses.fields(value)}
    elif isinstance(value, Enum):
        result = to_jsonable(value.value)
    elif isinstance(value, dict):
        result = {str(key): to_jsonable(item) for key, item in value.items()}
    elif isinstance(value, list | tuple):
        result = [to_jsonable(item) for item in value]
    elif isinstance(value, float) and not math.isfinite(value):
        result = None
    return result
