"""What every flow shares: the session, the step runner, prompts, reports and formatting.

Each flow is an explicit sequence of steps rather than nested calls. That is
what makes ``back`` mean what it says: :func:`run_steps` decrements an index and
re-runs the previous step, with the answers gathered so far kept in the flow's
state object. ``home`` and ``exit`` are not handled here at all -- they raise
past every step to the main loop, which is the only place that decides what
they mean.

Engine modules are imported inside the functions that use them, never at module
scope: the menu must still come up (with the features disabled) on a machine
where torch or jinja2 is missing.
"""

from __future__ import annotations

import math
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from rich.console import RenderableType
from rich.text import Text

from cli.ui import Choice, GoBack, GoHome, Ui
from jobs import JobRunner
from settings import hub_token, reports_root

if TYPE_CHECKING:
    from reporting import ReportRecord, ReportStore

CONTINUE_PROMPT = "Press Enter to continue"
MISSING = "-"
NUMBER_DIGITS = 4
BAR_WIDTH = 30
BAR_CHARACTER = "#"
TEXT_PREVIEW = 60
VISIBLE_WHITESPACE = {" ": "␣", "\n": "\\n", "\t": "\\t", "\r": "\\r"}
EMPTY_TOKEN = "∅"

S = TypeVar("S")
Step = Callable[["Session", S], None]
Action = Callable[[], RenderableType]


@dataclass
class Session:
    """Everything a flow needs besides its own answers."""

    ui: Ui
    store: ReportStore
    runner: JobRunner = field(default_factory=JobRunner)
    hf_token: str | None = None


def new_session(ui: Ui | None = None, store: ReportStore | None = None) -> Session:
    """Build a session on the real terminal and the configured reports folder unless given."""
    from reporting import ReportStore

    return Session(
        ui=ui or Ui(),
        store=store or ReportStore(str(reports_root())),
        hf_token=hub_token(),
    )


def run_steps(session: Session, steps: Sequence[Step[S]], state: S) -> None:
    """Walk a flow's steps, letting ``back`` return to the previous one.

    Stepping back from the first step leaves the flow. ``home`` and ``exit`` are
    not caught: they are meant to unwind further than this.
    """
    index = 0
    while 0 <= index < len(steps):
        try:
            steps[index](session, state)
            index += 1
        except GoBack:
            index -= 1


# --- prompts ---------------------------------------------------------------


def acknowledge(ui: Ui, message: str = "") -> None:
    """Show a message, if any, and wait for the user to take it in."""
    if message:
        ui.error(message)
    ui.read(f"{CONTINUE_PROMPT}: ")


def ask_path(ui: Ui, question: str, default: str = "") -> str:
    """Read a file path, with completion where the terminal allows it (quotes removed)."""
    from cli.completer import path_completer

    completer = path_completer() if ui.supports_completion else None
    return ui.ask(question, default=default, completer=completer).strip().strip('"')


def ask_int(ui: Ui, question: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    """Read a whole number within bounds, asking again until one is given."""
    value: int | None = None
    while value is None:
        value = parse_int(ui.ask(question, default=str(default)), minimum, maximum)
        if value is None:
            upper = f" and {maximum}" if maximum is not None else " or more"
            ui.error(f"Enter a whole number {'between ' if maximum is not None else ''}{minimum}{upper}.")
    return value


def ask_float(ui: Ui, question: str, default: float, minimum: float = 0.0) -> float:
    """Read a number greater than ``minimum``, asking again until one is given."""
    value: float | None = None
    while value is None:
        value = parse_float(ui.ask(question, default=f"{default:g}"), minimum)
        if value is None:
            ui.error(f"Enter a number greater than {minimum:g}.")
    return value


def parse_int(text: str, minimum: int, maximum: int | None = None) -> int | None:
    """The whole number in ``text`` when it is within bounds, else None."""
    result: int | None = None
    stripped = text.strip()
    if stripped.lstrip("-").isdigit():
        number = int(stripped)
        within = number >= minimum and (maximum is None or number <= maximum)
        result = number if within else None
    return result


def parse_float(text: str, minimum: float) -> float | None:
    """The finite number in ``text`` when it is greater than ``minimum``, else None."""
    result: float | None = None
    try:
        number = float(text.strip())
        result = number if math.isfinite(number) and number > minimum else None
    except ValueError:
        result = None
    return result


def split_list(text: str) -> list[str]:
    """Comma-separated items, stripped, empty ones dropped."""
    return [item.strip() for item in text.split(",") if item.strip()]


def require(value: S | None) -> S:
    """A value an earlier step must have set.

    Raises:
        GoBack: when it is missing, so the earlier step runs again.
    """
    if value is None:
        raise GoBack
    return value


# --- reports ---------------------------------------------------------------


def save_report(session: Session, result: Any) -> tuple[ReportRecord | None, list[RenderableType]]:
    """Save a result in the report store.

    Returns:
        The saved record (None when saving failed) and the lines announcing it.
    """
    record: ReportRecord | None = None
    lines: list[RenderableType]
    try:
        record = session.store.save(result)
        lines = [Text(f"Report saved: {record.id}", style="green"), Text(record.html_path, style="dim")]
    except (OSError, ValueError) as error:
        lines = [Text(f"The report could not be saved: {error}", style="red")]
    return record, lines


def open_in_browser(path: str) -> RenderableType:
    """Open a local HTML file in the default browser; returns the status line."""
    uri = Path(path).resolve().as_uri()
    opened = webbrowser.open(uri)
    return Text(f"Opened {uri}" if opened else f"No browser could be started. Open {uri}", style="green")


def present(
    session: Session,
    panel: Sequence[RenderableType],
    record: ReportRecord | None,
    extras: Sequence[tuple[Choice, Action]] = (),
) -> None:
    """Show an analysis' results and offer what can be done with them, until the user leaves.

    Args:
        session: the running session.
        panel: the results, shown above the actions.
        record: the saved report, or None when it could not be saved.
        extras: flow-specific actions, each returning the status line it produced.

    Raises:
        GoBack: on an empty answer or ``back``.
        GoHome: when the user picks Home.
    """
    actions: dict[str, Action] = {choice.key: action for choice, action in extras}
    if record is not None:
        actions["open"] = lambda: open_in_browser(record.html_path)
    options = [
        Choice("open", "Open the HTML report in the browser", "", record is not None, "The report was not saved."),
        *(choice for choice, _ in extras),
        Choice("home", "Home"),
    ]
    status: list[RenderableType] = []
    while True:
        chosen = session.ui.choose(options, *panel, *status)
        if chosen.key == "home":
            raise GoHome
        status = [actions[chosen.key]()]


# --- formatting ------------------------------------------------------------


def number(value: Any, digits: int = NUMBER_DIGITS) -> str:
    """A value for display: floats with ``digits`` significant digits, None or NaN as a dash."""
    result = str(value)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        result = MISSING
    elif isinstance(value, bool):
        result = "yes" if value else "no"
    elif isinstance(value, int):
        result = f"{value:,}"
    elif isinstance(value, float):
        result = f"{value:.{digits}g}"
    return result


def bar(value: float | None, peak: float, width: int = BAR_WIDTH) -> str:
    """A text bar of ``value`` relative to ``peak``."""
    filled = 0
    if value is not None and math.isfinite(value) and peak > 0:
        filled = max(0, min(width, round(width * value / peak)))
    return BAR_CHARACTER * filled


def token_label(text: str) -> str:
    """A token's text with its whitespace made visible."""
    return "".join(VISIBLE_WHITESPACE.get(character, character) for character in text) or EMPTY_TOKEN


def tokens_line(texts: Sequence[str]) -> str:
    """Several tokens on one line, separated so their boundaries stay visible."""
    return "  ".join(token_label(text) for text in texts)


def preview(text: str, limit: int = TEXT_PREVIEW) -> str:
    """The first ``limit`` characters of a text on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 3]}..."


def messages(title: str, items: Sequence[str], style: str) -> list[RenderableType]:
    """A titled list of warnings or notes; nothing when there are none."""
    lines: list[RenderableType] = []
    if items:
        lines = [Text(title, style=f"bold {style}"), *(Text(f"  - {item}", style=style) for item in items)]
    return lines
