"""The home menu and the main loop of the interactive session.

``home`` and ``exit`` are handled here and only here: they raise past every
step of a flow to :func:`run_session`. An analysis that ends without a result
(:class:`~cli.running.AnalysisStopped`) also unwinds to here, is
shown, and the home menu comes back.

Capabilities are detected before each menu rather than once per session, so a
package installed while the program is open is picked up.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text

from cli.common import Session, acknowledge, new_session
from cli.flows.analysis import run_analysis
from cli.flows.assessment import run_assessment
from cli.flows.behaviour import run_behaviour
from cli.flows.entropy import run_entropy
from cli.flows.environment import show_environment
from cli.flows.lineage import run_lineage
from cli.flows.provenance import run_provenance
from cli.flows.reports import run_reports
from cli.flows.setup import run_setup
from cli.flows.shifts import run_shifts
from cli.flows.vocab import run_vocab
from cli.flows.weights import run_weights
from cli.running import AnalysisStopped
from cli.ui import Choice, ExitRequested, GoBack, GoHome
from environment import Capabilities, Feature, detect, features

HOME = "Home"
FAREWELL = "Goodbye."
QUIT = "quit"
INTERRUPTED = "Interrupted."

Flow = Callable[[Session], None]

# (key, label, detail, feature gating it or None)
MENU: tuple[tuple[str, str, str, str | None], ...] = (
    ("assess", "Analyse a Model", "run every tool and read a suspicion verdict", "Weight Diff"),
    ("weights", "Weight Diff", "how much every tensor moved", "Weight Diff"),
    ("entropy", "Entropy Analysis", "how the last layers' weight entropy shifted", "Entropy Analysis"),
    ("shifts", "Token Shifts", "heaviest updates read as tokens", "Token Shifts"),
    ("vocab", "Vocabulary & Embeddings", "added tokens and how embeddings moved", "Vocabulary & Embeddings"),
    ("lineage", "Hub Lineage", "declared parents and children, compared", "Hub Lineage"),
    ("provenance", "Provenance", "which model the weights derive from", "Provenance"),
    ("behaviour", "Behaviour Probe", "where the predictions differ (runs both models)", "Behaviour Probe"),
    ("reports", "Tools", "open, locate or delete saved tool reports", "Reports"),
    ("analysis", "Analysis", "open, locate or delete saved model analyses", "Reports"),
    ("environment", "Environment", "packages, GPU and available features", None),
    ("setup", "Settings", "web server address, report and models folders", None),
)

# Section headings over the actions, in the order they appear in the home menu. Each heading
# labels the keys of MENU listed under it; the keys stay numbered 1.. across the whole menu.
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Workflow", ("assess",)),
    ("Tools", ("weights", "entropy", "shifts", "vocab", "lineage", "provenance", "behaviour")),
    ("Reports", ("reports", "analysis")),
    ("System", ("environment", "setup")),
)

FLOWS: dict[str, Flow] = {
    "assess": run_assessment,
    "weights": run_weights,
    "entropy": run_entropy,
    "shifts": run_shifts,
    "vocab": run_vocab,
    "lineage": run_lineage,
    "provenance": run_provenance,
    "behaviour": run_behaviour,
    "reports": run_reports,
    "analysis": run_analysis,
    "environment": show_environment,
    "setup": run_setup,
}


def run_cli() -> int:
    """Run the interactive session until the user leaves. Returns an exit code."""
    return run_session(new_session())


def run_session(session: Session) -> int:
    """Offer the home menu and run actions until the user exits. Returns 0."""
    finished = False
    while not finished:
        try:
            one_action(session)
        except ExitRequested:
            finished = True
        except (GoBack, GoHome):
            # Both mean "show the home menu again": there is nowhere above it to
            # step back to, and home is defined as returning here.
            continue
    session.ui.console.print(Text(FAREWELL))
    return 0


def one_action(session: Session) -> None:
    """Offer the home menu and run whatever the user picked."""
    ui = session.ui
    ui.goto()  # the home menu carries no trail label; the menu sits under the header
    choice = ui.choose(main_menu(detect()))
    if choice.key == QUIT:
        raise ExitRequested
    try:
        FLOWS[choice.key](session)
    except AnalysisStopped as stopped:
        ui.goto(HOME, choice.label)
        if stopped.details:
            ui.hint(stopped.details)
        acknowledge(ui, stopped.message)
    except KeyboardInterrupt:
        acknowledge(ui, INTERRUPTED)


def main_menu(capabilities: Capabilities) -> list[Choice]:
    """Build the home menu grouped under its sections, disabling what this machine cannot do.

    Each section (``Tools``, ``System``) is a heading above its actions; the headings carry no
    number, so the actions stay numbered ``1..`` whatever is installed.
    """
    gates = {item.label: item for item in features(capabilities)}
    actions = {
        key: _action_choice(key, label, detail, gates.get(gate) if gate is not None else None)
        for key, label, detail, gate in MENU
    }
    options: list[Choice] = []
    for title, keys in SECTIONS:
        options.append(Choice.section(title))
        options.extend(actions[key] for key in keys)
    options.append(Choice(QUIT, "Quit"))
    return options


def _action_choice(key: str, label: str, detail: str, feature: Feature | None) -> Choice:
    """Build one menu action, disabled with the reason when its gating feature is unavailable."""
    available = feature is None or feature.available
    reason = feature.reason if feature is not None else ""
    return Choice(key, label, detail, available, reason)
