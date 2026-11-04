"""Token Shifts: the heaviest weight updates, read through the vocabulary.

Steps: base model, tuned model, parameters (updates projected, tokens listed),
then the analysis and its results (top updates with the tokens at both ends of
their leading direction and the null z-score, embedding rows that moved most)
with the report saved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    ask_int,
    messages,
    number,
    present,
    require,
    run_steps,
    save_report,
    tokens_line,
)
from cli.flows.weights import model_inputs
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import build_table
from progress import CancelToken, ProgressSink, Stage
from settings import DEFAULT_TOKENS, DEFAULT_UPDATES, MAX_TOKENS, MAX_UPDATES

if TYPE_CHECKING:
    from analysis.shifts import AxisEnd, EmbeddingRows, TokenShiftResult, TopUpdate

TITLE = "Token Shifts"
TOKENS_SHOWN = 8


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
    state.base = select_model(session, (TITLE, "Base Model"), "Base model")


def _step_tuned(session: Session, state: ShiftsState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session, (TITLE, "Base Model", "Tuned Model"), "Tuned model", context=model_lines([("Base", state.base)])
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
    """The results: counts, top updates with their tokens, embedding rows, warnings."""
    counts = result.counts
    summary = Text(
        f"Compared {counts.compared}  changed {counts.changed}  identical {counts.identical}  "
        f"unmatched {counts.unmatched}  vocabulary {result.vocabulary:,}"
    )
    return [
        summary,
        update_table(result.top_updates),
        *embedding_lines(result.embedding_rows),
        *messages("Warnings", result.meta.warnings, "yellow"),
        *messages("Notes", result.meta.notes[:5], "dim"),
    ]


def update_table(updates: Sequence[TopUpdate], title: str = "Heaviest updates") -> RenderableType:
    """One row per update: its size and the tokens at both ends of its direction."""
    rows = [
        [
            update.name,
            number(update.sigma),
            f"{update.share:.0%}",
            number(update.rel_change),
            update.projection.relation if update.projection else "-",
            _end_text(update.projection.positive if update.projection else None),
            _end_text(update.projection.negative if update.projection else None),
        ]
        for update in updates
    ]
    columns = ("Tensor", "sigma1", "Share", "Rel. change", "Reading", "+ end (null z)", "- end (null z)")
    return build_table(columns, rows, title)


def _end_text(end: AxisEnd | None) -> str:
    """The first tokens of one end of a direction, and how far they stand from random directions."""
    result = "-"
    if end is not None:
        result = f"{tokens_line([token.text for token in end.tokens[:TOKENS_SHOWN]])}  (z {number(end.null.z, 3)})"
    return result


def embedding_lines(rows: Sequence[EmbeddingRows]) -> list[RenderableType]:
    """The tokens whose embedding (or unembedding) rows moved most."""
    lines: list[RenderableType] = [Text("Embedding rows that moved most", style="bold")] if rows else []
    for item in rows:
        lines.append(Text.assemble(f"  {item.name}: ", (tokens_line([token.text for token in item.tokens]), "cyan")))
    return lines
