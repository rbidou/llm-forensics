"""Running an analysis as a background job under a live progress bar.

The analysis runs on the :class:`~jobs.JobRunner` worker thread,
inside :func:`~progress.quiet` so nothing the machine-learning stack
prints reaches the terminal. The main thread only draws: it forwards the job's
new events (through the job's cursor) to the progress bar until the job ends.

Ctrl+C while the bar is live asks the job to stop at its next checkpoint and
keeps waiting for it to end, so a cancelled analysis never leaves a thread
holding a model in memory behind the next menu. A job that ends without a
result raises :class:`AnalysisStopped`, which the main loop shows.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar, cast

from cli.common import Session
from jobs import Job, JobRunner, JobState
from progress import CancelToken, CapturedOutput, ProgressEvent, ProgressSink, Stage, quiet

POLL_SECONDS = 0.1
FAILURE_TAIL_LINES = 15
CANCELLED_MESSAGE = "Cancelled."
CANCELLING_MESSAGE = "Cancelling: waiting for the analysis to reach a checkpoint..."

T = TypeVar("T")
Work = Callable[[ProgressSink, CancelToken], T]


class AnalysisStopped(Exception):
    """An analysis ended without a result: cancelled, or failed."""

    def __init__(self, message: str, details: str = "") -> None:
        """Keep the message and what the libraries printed meanwhile (for a failure)."""
        super().__init__(message)
        self.message = message
        self.details = details


def run_job(session: Session, trail: tuple[str, ...], stage: Stage, work: Work[T]) -> T:
    """Run ``work`` in the background while drawing its progress, and return its result.

    Args:
        session: the running session (its UI and job runner).
        trail: the location shown above the progress bar.
        stage: the analysis, used as the job kind and for the cancelling message.
        work: the analysis, called with a progress sink and a cancel token.

    Returns:
        What ``work`` returned.

    Raises:
        AnalysisStopped: when the job was cancelled or failed.
    """
    captured: list[CapturedOutput] = []
    job = session.runner.start(stage.value, _quietly(work, captured))
    with session.ui.progress(*trail) as sink:
        follow(session.runner, job, sink, stage)
    return cast(T, outcome(job, captured))


def follow(runner: JobRunner, job: Job, sink: ProgressSink, stage: Stage) -> None:
    """Forward the job's events to ``sink`` until it ends; Ctrl+C cancels it and keeps waiting."""
    cursor = 0
    while not job.finished:
        try:
            cursor = _forward(job, cursor, sink)
            pause(POLL_SECONDS)
        except KeyboardInterrupt:
            runner.cancel(job.identifier)
            sink(ProgressEvent(stage, CANCELLING_MESSAGE))
    _forward(job, cursor, sink)


def outcome(job: Job, captured: list[CapturedOutput]) -> object:
    """The result of a finished job.

    A job asked to stop counts as cancelled even if it finished anyway: the user
    has already decided not to wait for that result.

    Raises:
        AnalysisStopped: when the job was cancelled or failed.
    """
    if job.state is JobState.CANCELLED or job.token.cancelled:
        raise AnalysisStopped(CANCELLED_MESSAGE)
    if job.state is JobState.FAILED:
        details = captured[0].tail(FAILURE_TAIL_LINES) if captured else ""
        raise AnalysisStopped(f"The analysis failed: {job.error}", details)
    return job.result


def pause(seconds: float) -> None:
    """Wait between two polls (replaced by tests to simulate Ctrl+C)."""
    time.sleep(seconds)


def _forward(job: Job, cursor: int, sink: ProgressSink) -> int:
    """Send the events not seen yet; returns the new cursor."""
    events, cursor = job.since(cursor)
    for event in events:
        sink(event)
    return cursor


def _quietly(work: Work[T], captured: list[CapturedOutput]) -> Work[T]:
    """Wrap ``work`` so it runs silenced, keeping what was printed in ``captured``."""

    def silenced(sink: ProgressSink, token: CancelToken) -> T:
        """Run the analysis with the library output captured."""
        with quiet() as output:
            captured.append(output)
            result = work(sink, token)
        return result

    return silenced
