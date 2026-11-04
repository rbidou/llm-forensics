"""Running one analysis at a time on behalf of a front-end.

An analysis is a long, blocking call. It runs in a worker thread; its progress
events and final result accumulate on a :class:`Job` that the CLI renders live
and the browser polls.

Only one job runs at a time, deliberately: two analyses would compete for the
same memory (two full models for a behavioural probe) and both would crawl.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock, Thread
from typing import Any

from progress import CancelToken, JobCancelled, ProgressEvent, ProgressSink

Work = Callable[[ProgressSink, CancelToken], Any]

# Identifiers are chosen by the browser (random hex, see newJobId in app.js) or generated here.
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class JobState(StrEnum):
    """Where a job has got to."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobBusy(RuntimeError):
    """Raised when a job is started while another is still running."""


class InvalidJobId(ValueError):
    """Raised when a caller-supplied job identifier is malformed."""


@dataclass
class Job:
    """One run of one analysis, and everything it has reported."""

    identifier: str
    kind: str
    label: str = ""
    state: JobState = JobState.RUNNING
    result: Any = None
    error: str = ""
    events: list[ProgressEvent] = field(default_factory=list)
    token: CancelToken = field(default_factory=CancelToken, repr=False)
    lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def finished(self) -> bool:
        """Whether the job has stopped, whatever the outcome."""
        return self.state is not JobState.RUNNING

    def record(self, event: ProgressEvent) -> None:
        """Store one progress event."""
        with self.lock:
            self.events.append(event)

    def since(self, cursor: int) -> tuple[list[ProgressEvent], int]:
        """Return the events a client has not seen yet, and the new cursor.

        Args:
            cursor: number of events the client has already consumed.

        Returns:
            The pending events and the cursor to pass next time.
        """
        with self.lock:
            pending = self.events[cursor:]
        return pending, cursor + len(pending)

    def latest(self) -> ProgressEvent | None:
        """The most recent event, for a client that just connected."""
        with self.lock:
            result = self.events[-1] if self.events else None
        return result


class JobRunner:
    """Owns the single job that may be running at any moment."""

    def __init__(self) -> None:
        self._current: Job | None = None
        self._counter = itertools.count(1)
        self._lock = Lock()

    @property
    def current(self) -> Job | None:
        """The running or most recently finished job."""
        return self._current

    def get(self, identifier: str) -> Job | None:
        """Return the current job if it carries this identifier."""
        job = self._current
        return job if job is not None and job.identifier == identifier else None

    def start(self, kind: str, work: Work, identifier: str | None = None, label: str = "") -> Job:
        """Begin a job in the background and return it.

        Args:
            kind: the analysis name, shown to the user.
            work: the analysis, called with a progress sink and a cancel token.
            identifier: caller-chosen id (web clients); generated when omitted.
            label: human name of the user-facing process; empty for an internal
                job the front-ends do not track (reattach, finish notification).

        Returns:
            The job tracking the work.

        Raises:
            InvalidJobId: when ``identifier`` is malformed.
            JobBusy: when another job is still running.
        """
        job_id = identifier if identifier is not None else f"{kind}-{next(self._counter)}"
        if not JOB_ID_PATTERN.match(job_id):
            raise InvalidJobId(f"Invalid job identifier: {job_id!r}")
        with self._lock:
            if self._current is not None and not self._current.finished:
                raise JobBusy(f"A {self._current.kind} job is already running. Wait for it or cancel it.")
            job = Job(identifier=job_id, kind=kind, label=label)
            self._current = job
        Thread(target=self._run, args=(job, work), daemon=True, name=f"llmforensics-{job.identifier}").start()
        return job

    def cancel(self, identifier: str) -> bool:
        """Ask a running job to stop at its next checkpoint.

        Returns:
            True when a running job with that identifier was found.
        """
        job = self.get(identifier)
        found = False
        if job is not None and not job.finished:
            job.token.cancel()
            found = True
        return found

    @staticmethod
    def _run(job: Job, work: Work) -> None:
        """Perform the work, recording how it ended.

        Every exception is captured onto the job rather than raised: this runs on
        a thread nobody joins, so an escaping error would leave the front-end
        watching a job that never finishes.
        """
        try:
            outcome = work(job.record, job.token)
        except JobCancelled as error:
            job.error = str(error)
            job.state = JobState.CANCELLED
        except Exception as error:  # noqa: BLE001 - reported to the front-end instead
            job.error = str(error) or error.__class__.__name__
            job.state = JobState.FAILED
        else:
            job.result = outcome
            job.state = JobState.SUCCEEDED
