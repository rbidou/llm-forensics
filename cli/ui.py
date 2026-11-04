"""The four-part terminal layout, and the input that drives it.

Every screen has the same shape, the header pinned to the top and the prompt to
the bottom::

    header            the program, its version and subtitle
                      (blank line)
    trail             where you are: Weight Diff > Base Model > Tuned Model
                      (blank line)
    display panel     the data for this step, replaced when the kind changes
                      (filled to push the rest down)
    separator         a rule, captioned with the commands available everywhere
                      (blank line)
    interactive line  where the user answers

Callers name a location with :meth:`Ui.goto`, put data in the panel, and read a
line. They never position anything themselves, so every screen lines up.

Three commands work at any prompt. ``back`` returns to the previous step,
``home`` abandons the flow, and ``exit`` leaves. They are raised as exceptions so
they unwind from wherever they are typed rather than having to be threaded
through every return value.

Two details are load-bearing. The console is bound to ``sys.__stdout__``,
captured when this object is built, rather than to ``sys.stdout``: analyses run
inside :func:`progress.quiet`, which replaces ``sys.stdout``, and a
console resolving the stream lazily would have its progress bar swallowed. And
reading input while a progress bar is live is refused outright -- the live
display paints over the prompt, so the program waits on a question nobody can
see, which is indistinguishable from a hang.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, TextIO

from prompt_toolkit import prompt as rich_prompt
from prompt_toolkit.completion import Completer
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from progress import ProgressEvent, ProgressSink
from settings import __version__

PROGRAM_NAME = "LLM Forensics"
PROGRAM_SUBTITLE = "Weight-level forensics for language models"
TRAIL_SEPARATOR = " > "
CHOICE_PROMPT = "Choice"
COMMAND_HINT = "back  |  home  |  exit"
ACCENT = "dark_orange"

BACK = "back"
HOME = "home"
EXIT = "exit"

# Lines the layout reserves below the panel: padding, the separator, padding,
# and the interactive line itself.
FOOTER_HEIGHT = 4
PROGRESS_BAR_WIDTH = 28

Cell = str | Text


class Navigation(Exception):
    """Base for the commands that move between steps rather than answer one."""


class GoBack(Navigation):
    """The user asked to return to the previous step."""


class GoHome(Navigation):
    """The user asked to abandon the flow and start again."""


class ExitRequested(Navigation):
    """The user asked to leave the program."""


@dataclass(frozen=True)
class Choice:
    """One row of a menu: a selectable option, or a section heading.

    A heading (``heading=True``) is a label above a group of options; it carries
    no number and cannot be selected.
    """

    key: str
    label: str
    detail: str = ""
    enabled: bool = True
    reason: str = ""
    heading: bool = False

    @classmethod
    def section(cls, label: str) -> Choice:
        """Build a non-selectable section heading."""
        return cls(key="", label=label, heading=True)


class Reader(Protocol):
    """Anything that reads one line for a question (the terminal, or a test script)."""

    interactive: bool

    def __call__(self, question: str, completer: Completer | None = None) -> str:
        """Return the user's answer to ``question``."""
        ...


class LineReader:
    """Reads a line of input, with completion where the terminal allows it.

    prompt_toolkit provides the as-you-type suggestions, but it needs a real
    console: it raises outright under Git Bash, in CI, and whenever input is
    piped. Rather than let that end the session, the first failure switches this
    reader to plain :func:`input` for good -- the CLI keeps working everywhere,
    losing only the completion.
    """

    def __init__(self, interactive: bool | None = None) -> None:
        self.interactive = _console_available() if interactive is None else interactive

    def __call__(self, question: str, completer: Completer | None = None) -> str:
        """Read one answer, falling back to plain input when the terminal cannot do better."""
        answer: str | None = None
        if self.interactive:
            try:
                # reserve_space_for_menu=0: the completion dropdown must not reserve lines below the
                # prompt, which would scroll the whole screen (header included) up as soon as the
                # prompt opens. It still expands while the user is completing.
                answer = rich_prompt(
                    question,
                    completer=completer,
                    complete_while_typing=completer is not None,
                    reserve_space_for_menu=0,
                )
            except Exception:  # noqa: BLE001 - any terminal problem means fall back
                self.interactive = False
        return _plain_input(question) if answer is None else answer


def _terminal_stream() -> TextIO:
    """The process's real standard output, made tolerant of characters it cannot encode.

    Token texts come in every script, and a redirected Windows stream is often
    cp1252: one unencodable character would otherwise raise in the middle of a
    screen. They are replaced instead (a real console writes Unicode anyway).
    """
    stream = sys.__stdout__
    if stream is None:
        raise RuntimeError("The interactive session needs a standard output.")
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    return stream


def _console_available() -> bool:
    """Whether the terminal can support an interactive prompt at all."""
    return bool(getattr(sys.stdin, "isatty", bool)() and getattr(sys.__stdout__, "isatty", bool)())


def _plain_input(question: str) -> str:
    """Read a line without any terminal features, tolerating end of input."""
    print(question, end="", flush=True, file=sys.__stdout__)
    try:
        answer = input()
    except EOFError:
        answer = EXIT
        print(file=sys.__stdout__)
    return answer


class Ui:
    """Draws the four-part screen and reads the user's answers."""

    def __init__(self, console: Console | None = None, reader: Reader | None = None) -> None:
        self.console = console or Console(file=_terminal_stream(), highlight=False)
        self._reader: Reader = reader or LineReader()
        self._trail: tuple[str, ...] = ()
        self._panel: list[RenderableType] = []
        self._progress_active = False
        self._dirty = True

    @property
    def supports_completion(self) -> bool:
        """Whether suggestions can be offered while typing."""
        return self._reader.interactive

    @property
    def trail(self) -> tuple[str, ...]:
        """The steps leading to where the user is now."""
        return self._trail

    # --- structure ---------------------------------------------------------

    def goto(self, *trail: str) -> None:
        """Move to a step, naming the path that led there, and empty the panel.

        Every step begins here: a new step always means a new kind of data, so
        the panel from the previous one is discarded rather than scrolled.
        """
        self._trail = trail
        self._panel = []
        self._dirty = True

    def show(self, *items: RenderableType) -> None:
        """Replace the panel with new data."""
        self._panel = list(items)
        self._dirty = True

    def append(self, *items: RenderableType) -> None:
        """Add to the data already in the panel."""
        self._panel.extend(items)
        self._dirty = True

    # --- panel content -----------------------------------------------------

    def note(self, text: str) -> None:
        """Add an ordinary line of data (plain text, never interpreted as markup)."""
        self.append(Text(text))

    def hint(self, text: str) -> None:
        """Add secondary guidance."""
        self.append(Text(text, style="dim"))

    def success(self, text: str) -> None:
        """Report something that worked."""
        self.append(Text(text, style="green"))

    def warn(self, text: str) -> None:
        """Report something worth noticing that is not fatal."""
        self.append(Text(text, style="yellow"))

    def error(self, text: str) -> None:
        """Report a failure."""
        self.append(Text(text, style="red"))

    def table(self, columns: Sequence[str], rows: Sequence[Sequence[Cell]], title: str | None = None) -> Table:
        """Build a table for the panel. Pass it to :meth:`show` or :meth:`append`.

        Plain strings are shown as they are: model and token texts often hold
        square brackets, which rich would otherwise read as markup.
        """
        return build_table(columns, rows, title)

    # --- interaction -------------------------------------------------------

    def choose(self, options: Sequence[Choice], *context: RenderableType) -> Choice:
        """Show a numbered menu as the panel data and read a selection.

        Disabled options keep their place in the numbering so the menu does not
        change shape depending on what happens to be installed, and each carries
        the reason it cannot be used.

        ``context`` is data the choice is about -- a results table, say --
        rendered above the options as part of the same panel.

        Always returns a choice: an empty answer is treated as ``back`` and
        raises, so callers never have to handle a null selection.
        """
        numbered = _numbered(options)
        self.show(*context, self._option_table(numbered))
        selectable = {number: option for number, option in numbered if number is not None and option.enabled}
        return self._read_choice(selectable)

    def ask(self, question: str, default: str = "", completer: Completer | None = None) -> str:
        """Read a line on the interactive line, keeping the current screen."""
        suffix = f" [{default}]" if default else ""
        answer = self.read(f"{question}{suffix}: ", completer)
        return answer.strip() or default

    def confirm(self, question: str, default: bool = False) -> bool:
        """Ask a yes/no question on the interactive line."""
        hint = "Y/n" if default else "y/N"
        answer = self.read(f"{question} [{hint}] ").strip().lower()
        return default if not answer else answer in {"y", "yes"}

    def read(self, question: str, completer: Completer | None = None) -> str:
        """Read a line, handling the navigation commands and refusing to run
        underneath a live progress bar.

        A prompt written while :meth:`progress` is running is painted over by
        the live display, so the program blocks on input for a question the user
        never sees. Ask everything before the bar starts; this makes the mistake
        fail loudly.

        Raises:
            RuntimeError: when a progress bar is live.
            GoBack, GoHome, ExitRequested: for the navigation commands.
        """
        if self._progress_active:
            raise RuntimeError(
                f"Cannot prompt while a progress bar is running: {question!r}. "
                "Ask for this before entering the progress context."
            )
        # Repaint before every prompt so the header is redrawn at the top and each
        # question replaces the previous one on the bottom line rather than printing
        # beneath it (several questions in one step would otherwise stack and push
        # the header off the screen).
        self._dirty = True
        self._flush()
        answer = self._reader(question, completer)
        _check_navigation(answer)
        return answer

    @contextmanager
    def progress(self, *trail: str) -> Iterator[ProgressSink]:
        """Run a stage, drawing its progress bar as the panel data.

        Steps that cannot report a total show a spinner; the bar appears as soon
        as one is known.

        Yields:
            The sink that moves the bar.
        """
        self.goto(*trail)
        self._flush()
        columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with Progress(*columns, console=self.console, transient=False) as bar:
            task = bar.add_task("Starting", total=None)

            def handle_event(event: ProgressEvent) -> None:
                """Move the bar (or the spinner's message) to this event."""
                description = escape(event.message)
                if event.total:
                    bar.update(task, total=event.total, completed=event.current or 0, description=description)
                else:
                    bar.update(task, description=description)

            self._progress_active = True
            try:
                yield handle_event
            finally:
                self._progress_active = False

    # --- rendering ---------------------------------------------------------

    def _flush(self) -> None:
        """Paint the screen if anything changed since it was last drawn.

        Skipped while a progress bar is live, because clearing the screen under
        a rich ``Live`` display corrupts it.
        """
        if self._dirty and not self._progress_active:
            self._dirty = False
            self._paint()

    def _paint(self) -> None:
        """Draw the header, trail, panel and separator, pushing the footer down.

        A step with no trail (the home menu) drops the trail line and the blank
        below it, so the menu sits one blank line under the header.
        """
        header = self._header()
        self.console.clear()
        self.console.print(header)
        self.console.print()
        if self._trail:
            self.console.print(self._trail_line())
            self.console.print()
        for item in self._panel:
            self.console.print(item)
        self.console.print()
        for _ in range(self._filler_height(header)):
            self.console.print()
        self.console.print(Rule(COMMAND_HINT, style="dim", characters="-"))
        self.console.print()

    def _filler_height(self, header: RenderableType) -> int:
        """Blank lines needed to push the separator and prompt to the bottom.

        Only meaningful on a real terminal; piped output has no bottom to pin
        anything to, and padding it would just add noise.
        """
        result = 0
        if self.console.is_terminal:
            used = self._measure(header) + 1
            if self._trail:
                used += self._measure(self._trail_line()) + 1
            used += sum(self._measure(item) for item in self._panel)
            result = max(0, self.console.size.height - used - FOOTER_HEIGHT)
        return result

    def _measure(self, renderable: RenderableType) -> int:
        """How many lines a renderable will occupy at the current width."""
        return len(self.console.render_lines(renderable, self.console.options, pad=False))

    def _header(self) -> Panel:
        """The program identity, pinned to the top of the screen."""
        title = Text(f"{PROGRAM_NAME} {__version__}", style=f"bold {ACCENT}")
        subtitle = Text(PROGRAM_SUBTITLE, style="dim")
        return Panel(Group(title, subtitle), border_style=ACCENT, padding=(0, 1))

    def _trail_line(self) -> Text:
        """The current step and the ones that led to it."""
        line = Text()
        for index, step in enumerate(self._trail):
            if index:
                line.append(TRAIL_SEPARATOR, style="dim")
            last = index == len(self._trail) - 1
            line.append(step, style="bold" if last else "dim")
        return line

    def _option_table(self, numbered: Sequence[tuple[str | None, Choice]]) -> Table:
        """Lay out a menu: a section heading as an unnumbered label, every other row numbered,
        with what cannot be selected dimmed."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(justify="right")
        table.add_column()
        table.add_column(overflow="fold")
        for number, option in numbered:
            if option.heading:
                if table.row_count:  # a blank line sets each section apart from the one above it
                    table.add_row("", "", "")
                table.add_row("", Text(option.label, style=f"bold {ACCENT}"), "")
            elif option.enabled:
                table.add_row(Text(number or "", style=ACCENT), Text(option.label), Text(option.detail, style="dim"))
            else:
                table.add_row(
                    Text(number or "", style="dim"), Text(option.label, style="dim"), Text(option.reason, style="dim")
                )
        return table

    def _read_choice(self, selectable: dict[str, Choice]) -> Choice:
        """Read until a listed number is entered; an empty answer means back."""
        chosen: Choice | None = None
        while chosen is None:
            answer = self.read(f"{CHOICE_PROMPT}: ").strip()
            if not answer:
                raise GoBack
            chosen = selectable.get(answer)
            if chosen is None:
                self.error(f"'{answer}' is not one of the available options.")
        return chosen


def _numbered(options: Sequence[Choice]) -> list[tuple[str | None, Choice]]:
    """Pair each option with the number the user types to pick it; a heading gets ``None``.

    Numbering counts only selectable rows, so inserting a section heading never shifts the
    numbers of the options around it.
    """
    result: list[tuple[str | None, Choice]] = []
    number = 0
    for option in options:
        if option.heading:
            result.append((None, option))
        else:
            number += 1
            result.append((str(number), option))
    return result


def build_table(columns: Sequence[str], rows: Sequence[Sequence[Cell]], title: str | None = None) -> Table:
    """A panel table; plain strings are shown as they are, never read as markup."""
    table = Table(title=title, show_header=True, header_style="bold", title_justify="left")
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(Text(cell) if isinstance(cell, str) else cell for cell in row))
    return table


def _check_navigation(answer: str) -> None:
    """Turn a navigation command into the exception that unwinds to its target."""
    commands: dict[str, type[Navigation]] = {BACK: GoBack, HOME: GoHome, EXIT: ExitRequested}
    command = commands.get(answer.strip().lower())
    if command is not None:
        raise command()
