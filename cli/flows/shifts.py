"""Token Shifts: the heaviest weight updates, read through the vocabulary.

Steps: base model, tuned model, parameters (updates projected, tokens listed),
then the analysis and its results (top updates with the tokens at both ends of
their leading direction and the null z-score, embedding rows that moved most)
with the report saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    ask_int,
    present,
    require,
    run_steps,
    save_report,
)
from cli.flows.weights import model_inputs
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from progress import CancelToken, ProgressSink, Stage
from settings import DEFAULT_TOKENS, DEFAULT_UPDATES, MAX_TOKENS, MAX_UPDATES

if TYPE_CHECKING:
    from analysis.shifts import TokenShiftResult

TITLE = "Token Shifts"


@dataclass
class ShiftsState:
    """Answers of the Token Shifts flow."""

    base: ModelChoice | None = None
    tuned: ModelChoice | None = None
    updates: int = DEFAULT_UPDATES
    tokens: int = DEFAULT_TOKENS


def run_shifts(session: Session) -> None:
    """Read a tuned model's heaviest updates as tokens."""
    run_steps(session, (_step_base, _step_tuned, _step_parameters, _step_run), ShiftsState())


def _step_base(session: Session, state: ShiftsState) -> None:
    """Pick the reference model (it also provides the tokenizer)."""
    state.base = select_model(session, (TITLE, "Base Model"), "Base model", remember="base")


def _step_tuned(session: Session, state: ShiftsState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session,
        (TITLE, "Base Model", "Tuned Model"),
        "Tuned model",
        context=model_lines([("Base", state.base)]),
        remember="tuned",
    )


def _step_parameters(session: Session, state: ShiftsState) -> None:
    """How many updates to project and how many tokens to list."""
    ui = session.ui
    ui.goto(TITLE, "Tuned Model", "Parameters")
    ui.show(*model_lines([("Base", state.base), ("Tuned", state.tuned)]))
    ui.hint("Updates: the heaviest 2-D updates (by leading singular value) read as tokens.")
    ui.hint("Tokens: listed at each end of a direction and per embedding matrix.")
    state.updates = ask_int(ui, "Updates to project", state.updates, 1, MAX_UPDATES)
    state.tokens = ask_int(ui, "Tokens listed", state.tokens, 1, MAX_TOKENS)


def _step_run(session: Session, state: ShiftsState) -> None:
    """Run the analysis, save the report and present the results."""
    base, tuned = require(state.base), require(state.tuned)
    work = partial(analyse, base, tuned, state.updates, state.tokens, session.hf_token)
    result = run_job(session, (TITLE, "Analysing"), Stage.SHIFTS, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*shifts_panel(result), *saved], record)


def analyse(
    base: ModelChoice,
    tuned: ModelChoice,
    updates: int,
    tokens: int,
    hf_token: str | None,
    sink: ProgressSink,
    token: CancelToken,
) -> TokenShiftResult:
    """The job: load the base tokenizer, open both models and project their updates."""
    from analysis.shifts import token_shifts
    from analysis.tokens import load_tokenizer
    from models import GGUF_AUTO

    decoder = load_tokenizer(base.model, token=hf_token, gguf_file=base.gguf_file or GGUF_AUTO)
    return token_shifts(
        base.open(hf_token),
        tuned.open(hf_token),
        decoder,
        inputs=model_inputs(base, tuned),
        n=updates,
        top=tokens,
        sink=sink,
        token=token,
    )


def shifts_panel(result: TokenShiftResult) -> list[RenderableType]:
    """The counts and vocabulary summary; the heaviest updates and their tokens are in the report."""
    counts = result.counts
    summary = Text(
        f"Compared {counts.compared}  changed {counts.changed}  identical {counts.identical}  "
        f"unmatched {counts.unmatched}  vocabulary {result.vocabulary:,}"
    )
    return [summary]
