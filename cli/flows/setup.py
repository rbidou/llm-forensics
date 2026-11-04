"""Settings: the web server address and the report and models folders.

The same settings as the web Settings page, written to the shared ``settings.json``.
The menu comes back after every change; ``back`` (or an empty answer) leaves it. A
new server address applies the next time ``python run.py web`` starts; the two
directories apply at once (the report store of the running session is moved).
"""

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text

from cli.common import Session, acknowledge, ask_path
from cli.ui import Choice, Ui, build_table
from settings import (
    DirectoryError,
    ServerAddressError,
    models_root,
    reports_root,
    server_address,
    set_directories,
    set_server_address,
)

TITLE = "Settings"
SERVER_TITLE = "Web Server"
REPORTS_TITLE = "Report Directory"
MODELS_TITLE = "Models Directory"
WORKING_DIRECTORY = "the working directory"
NEXT_START_NOTE = "A new server address applies the next time the web server starts (python run.py web)."
HOST_HINT = (
    "127.0.0.1 keeps the interface on this machine; 0.0.0.0 listens on every network interface, "
    "and other machines then reach it through one of this machine's addresses."
)
REPORTS_HINT = (
    "Where reports are saved and read. The folder is created if it does not exist. Empty restores the default."
)
MODELS_HINT = (
    "Where the model pickers start browsing. Must be an existing folder; empty opens on the working directory."
)
EXPOSED_WARNING = "Other machines on the network will reach it. It has no authentication and reads local model files."


def run_setup(session: Session) -> None:
    """Show the settings and apply the changes picked, until the user goes back.

    Raises:
        GoBack: when the user leaves the menu.
    """
    ui = session.ui
    while True:
        ui.goto(TITLE)
        address = server_address()
        models = models_root()
        rows = [
            ["Web server address", address.url, "where 'python run.py web' listens"],
            ["Report directory", str(reports_root()), "where reports are saved and read"],
            ["Models directory", models or WORKING_DIRECTORY, "where the model pickers start browsing"],
        ]
        action = ui.choose(
            [
                Choice("server", "Change the web server address"),
                Choice(
                    "server_reset", "Restore the default address", "", not address.is_default, "Already the default."
                ),
                Choice("directories", "Change the report and models directories"),
                Choice("dirs_reset", "Restore the default directories", "", _directories_changed(), "Already default."),
            ],
            build_table(("Setting", "Value", "Holds"), rows, "Settings"),
            Text(NEXT_START_NOTE, style="dim"),
        )
        SETUP_ACTIONS[action.key](session)


def change_server_address(session: Session, reset: bool = False) -> None:
    """Store a new web server address, asking for it unless ``reset`` is set."""
    ui = session.ui
    ui.goto(TITLE, SERVER_TITLE)
    current = server_address()
    if not reset:
        ui.hint(HOST_HINT)
    host = "" if reset else ui.ask("Listening IP address", default=current.host)
    port = "" if reset else ui.ask("Port", default=str(current.port))
    try:
        address = set_server_address(host, port)
    except ServerAddressError as error:
        ui.error(str(error))
    except OSError as error:
        ui.error(f"The settings could not be saved: {error}")
    else:
        ui.success(f"The web server will listen on {address.url} from its next start.")
        if not address.is_loopback:
            ui.warn(EXPOSED_WARNING)
    acknowledge(ui)


def change_directories(session: Session, reset: bool = False) -> None:
    """Store the report and/or models directories, asking for them unless ``reset`` is set.

    Each directory is asked with its current value as the default; only the one being
    changed is touched (the other keeps its stored value). ``reset`` restores both.
    """
    ui = session.ui
    ui.goto(TITLE, REPORTS_TITLE)
    reports_default = _current_reports()
    models_default = models_root()
    reports_dir = "" if reset else _ask_directory(ui, REPORTS_TITLE, REPORTS_HINT, "Report directory", reports_default)
    models_dir = "" if reset else _ask_directory(ui, MODELS_TITLE, MODELS_HINT, "Models directory", models_default)
    try:
        set_directories(reports_dir, models_dir)
    except DirectoryError as error:
        ui.error(str(error))
    except OSError as error:
        ui.error(f"The settings could not be saved: {error}")
    else:
        session.store.root = str(reports_root())
        ui.success(f"Reports are saved to {reports_root()}.")
        ui.success(f"The model pickers start browsing from {models_root() or WORKING_DIRECTORY}.")
    acknowledge(ui)


def _ask_directory(ui: Ui, title: str, hint: str, question: str, default: str) -> str:
    """Ask for one directory with its hint and current value as the default."""
    ui.goto(TITLE, title)
    ui.hint(hint)
    return ask_path(ui, question, default=default)


def _current_reports() -> str:
    """The stored report directory as a default, or empty when it is the built-in default."""
    return "" if not _reports_is_custom() else str(reports_root())


def _reports_is_custom() -> bool:
    """Whether a report directory other than the built-in default is stored."""
    from settings import default_reports_root

    return reports_root() != default_reports_root()


def _directories_changed() -> bool:
    """Whether either directory has been moved from its default (so a reset does anything)."""
    return _reports_is_custom() or models_root() != ""


SETUP_ACTIONS: dict[str, Callable[[Session], None]] = {
    "server": change_server_address,
    "server_reset": lambda session: change_server_address(session, reset=True),
    "directories": change_directories,
    "dirs_reset": lambda session: change_directories(session, reset=True),
}
