"""Progress reporting and cooperative cancellation shared by every analysis.

Engine code never writes to the terminal. Each analysis emits
:class:`ProgressEvent` values to a sink supplied by the front-end, so the CLI can
draw a progress bar and the web layer can serve the same events to a browser.

Cancellation is cooperative: an analysis calls :meth:`CancelToken.check` between
units of work (typically between tensors), which raises :class:`JobCancelled`
once a front-end has asked to stop. Nothing is ever interrupted mid-tensor.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from threading import Event

# Read by huggingface_hub and transformers at import time, so they must be set
# before either is imported; setting them later silences some output only.
QUIET_ENVIRONMENT: dict[str, str] = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_VERBOSITY": "error",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "BITSANDBYTES_NOWELCOME": "1",
}


class Stage(StrEnum):
    """The long-running operations a front-end may need to render."""

    WEIGHTS = "weights"
    SHIFTS = "shifts"
    VOCAB = "vocab"
    ENTROPY = "entropy"
    LINEAGE = "lineage"
    PROVENANCE = "provenance"
    BEHAVIOUR = "behaviour"
    ASSESSMENT = "assessment"
    REPORT = "report"


@dataclass(frozen=True)
class ProgressEvent:
    """One observation about a running analysis.

    ``current`` and ``total`` are optional: some steps know their exact
    progress, others only that they are still working.
    """

    stage: Stage
    message: str
    current: int | None = None
    total: int | None = None

    @property
    def fraction(self) -> float | None:
        """Completion in the range 0..1, or ``None`` when the total is unknown."""
        result = None
        if self.current is not None and self.total is not None and self.total > 0:
            result = min(1.0, max(0.0, self.current / self.total))
        return result


ProgressSink = Callable[[ProgressEvent], None]


class JobCancelled(Exception):
    """Raised inside an analysis when its front-end asked it to stop."""


class CancelToken:
    """A thread-safe stop request, polled by the analysis between units of work."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        """Whether a stop has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request the analysis to stop at its next checkpoint."""
        self._event.set()

    def check(self) -> None:
        """Raise :class:`JobCancelled` if a stop has been requested.

        Raises:
            JobCancelled: when :meth:`cancel` was called.
        """
        if self._event.is_set():
            raise JobCancelled("Cancelled by the user.")


def report(
    sink: ProgressSink,
    stage: Stage,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Emit one event, so call sites stay readable at the point of work."""
    sink(ProgressEvent(stage=stage, message=message, current=current, total=total))


def null_sink(event: ProgressEvent) -> None:
    """Discard progress. The default for callers that render nothing."""


class CollectingSink:
    """Keeps every event it receives; used by tests and by the job runner."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        """Store one event."""
        self.events.append(event)

    @property
    def messages(self) -> tuple[str, ...]:
        """Every message reported so far, in order."""
        return tuple(event.message for event in self.events)


@dataclass
class CapturedOutput:
    """Whatever the silenced libraries tried to print."""

    buffer: io.StringIO

    @property
    def text(self) -> str:
        """Everything captured so far."""
        return self.buffer.getvalue()

    def tail(self, lines: int = 20) -> str:
        """The last few lines, for attaching to an error message."""
        return "\n".join(self.text.splitlines()[-lines:])


def silence_libraries() -> None:
    """Stop third-party libraries writing to the terminal.

    Call this from the entry point before anything else is imported.
    """
    os.environ.update(QUIET_ENVIRONMENT)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _silence_loaded_libraries()


@contextlib.contextmanager
def quiet() -> Iterator[CapturedOutput]:
    """Silence the machine-learning stack for the duration of an analysis.

    Output is captured rather than discarded, so a failing analysis can quote
    :meth:`CapturedOutput.tail`. This replaces ``sys.stdout`` while active: a
    renderer drawing a progress bar must hold ``sys.__stdout__``.

    TODO: the redirection is process-wide, so in the web server whatever other
    threads print during a job (Werkzeug's request log) is captured with it.
    A per-thread stream proxy would keep the library output of the job only.

    Yields:
        The buffer receiving everything printed meanwhile.
    """
    captured = CapturedOutput(io.StringIO())
    with contextlib.redirect_stdout(captured.buffer), contextlib.redirect_stderr(captured.buffer):
        _silence_loaded_libraries()
        yield captured


def _silence_loaded_libraries() -> None:
    """Turn down the loggers of the libraries that are already imported."""
    with contextlib.suppress(ImportError):
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    with contextlib.suppress(ImportError):
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
