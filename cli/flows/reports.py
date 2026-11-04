"""Reports: the saved reports, and what can be done with one.

Steps: the list (id, analysis, creation time, summary) and the choice of a
report, then its actions: open the HTML in the browser, show the file paths,
re-render the HTML from the JSON, delete (only after typing ``yes``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import Session, acknowledge, open_in_browser, parse_int, run_steps
from cli.ui import Choice, GoBack, build_table

if TYPE_CHECKING:
    from reporting import ReportRecord

TITLE = "Reports"
ASSESSMENT = "assessment"  # the analysis name of a workflow report (the rest are per-tool reports)
DELETE_CONFIRMATION = "yes"
ACTIONS = (
    Choice("open", "Open the HTML report in the browser"),
    Choice("paths", "Show the file paths", "report.html and report.json, to copy or download"),
    Choice("rerender", "Re-render the HTML", "from report.json, with the current templates"),
    Choice("delete", "Delete the report", "asks for confirmation"),
    Choice("back", "Back to the list"),
)


@dataclass
class ReportsState:
    """The chosen report, and the outcome of the last action shown on the list."""

    record: ReportRecord | None = None
    notice: list[RenderableType] = field(default_factory=list)


def run_reports(session: Session) -> None:
    """Browse, open and delete the saved per-tool reports."""
    browse(
        session,
        title=TITLE,
        keep=lambda record: record.analysis != ASSESSMENT,
        empty="No tool report saved yet in {root}.",
    )


def browse(session: Session, *, title: str, keep: Callable[[ReportRecord], bool], empty: str) -> None:
    """Browse, open and delete the saved reports this page shows.

    Args:
        session: the running session.
        title: the location shown above each screen.
        keep: which reports this page lists (a record -> bool).
        empty: the message when none match (``{root}`` is the reports folder).
    """
    run_steps(
        session,
        (partial(_step_list, title=title, keep=keep, empty=empty), partial(_step_actions, title=title)),
        ReportsState(),
    )


def _step_list(
    session: Session,
    state: ReportsState,
    *,
    title: str,
    keep: Callable[[ReportRecord], bool],
    empty: str,
) -> None:
    """Show the matching reports and read which one to act on."""
    ui = session.ui
    records = [record for record in session.store.list() if keep(record)]
    ui.goto(title)
    if not records:
        acknowledge(ui, empty.format(root=session.store.root))
        raise GoBack
    ui.show(reports_table(records), *state.notice)
    state.notice = []
    index: int | None = None
    while index is None:
        answer = ui.ask(f"Report number (1-{len(records)})")
        if not answer:
            raise GoBack
        index = parse_int(answer, 1, len(records))
        if index is None:
            ui.error(f"'{answer}' is not a report number.")
    state.record = records[index - 1]


def reports_table(records: list[ReportRecord]) -> RenderableType:
    """The saved reports, newest first as listed by the store."""
    rows = [
        [str(index), record.id, record.analysis, record.created, record.summary]
        for index, record in enumerate(records, start=1)
    ]
    return build_table(("#", "Id", "Analysis", "Created", "Summary"), rows, f"Saved reports ({len(records)})")


def _step_actions(session: Session, state: ReportsState, *, title: str) -> None:
    """Act on the chosen report until the user goes back to the list."""
    record = state.record
    if record is None:
        raise GoBack
    ui = session.ui
    status: list[RenderableType] = []
    while True:
        ui.goto(title, record.id)
        header = Text.assemble((f"{record.title}\n", "bold"), (f"{record.summary}", "dim"))
        chosen = ui.choose(ACTIONS, header, *status).key
        if chosen == "back":
            raise GoBack
        status = [_act(session, state, record, chosen)]


def _act(session: Session, state: ReportsState, record: ReportRecord, action: str) -> RenderableType:
    """Perform one action on a report; returns its status line."""
    from reporting import InvalidReportId, ReportNotFound

    status: RenderableType = Text("")
    try:
        if action == "open":
            status = open_in_browser(record.html_path)
        elif action == "paths":
            status = Text(f"HTML: {record.html_path}\nJSON: {record.json_path}")
        elif action == "rerender":
            refreshed = session.store.rerender(record.id)
            status = Text(f"HTML re-rendered: {refreshed.html_path}", style="green")
        else:
            status = _delete(session, state, record)
    except (InvalidReportId, ReportNotFound, OSError, ValueError) as error:
        status = Text(f"{action} failed: {error}", style="red")
    return status


def _delete(session: Session, state: ReportsState, record: ReportRecord) -> RenderableType:
    """Delete the report once the user has typed ``yes``, then return to the list.

    Raises:
        GoBack: after a deletion, since the report no longer exists.
    """
    answer = session.ui.ask(f"Type '{DELETE_CONFIRMATION}' to delete {record.id} permanently")
    if answer.strip().lower() == DELETE_CONFIRMATION:
        session.store.delete(record.id)
        state.record = None
        state.notice = [Text(f"Deleted {record.id}.", style="green")]
        raise GoBack
    return Text("Not deleted.", style="yellow")
