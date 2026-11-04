"""Vocabulary & Embeddings: the tokenizer diff and how the token embeddings moved.

Steps: base model, tuned model, parameters (moved rows listed, new tokens read),
then the analysis and its results (the vocabulary diff, the shared-row embedding
shift with its concentration z-score, and the added tokens characterised as
trained or placeholder and as emittable or read-only) with the report saved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

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
from settings import DEFAULT_VOCAB_NEW, DEFAULT_VOCAB_ROWS, MAX_VOCAB_NEW, MAX_VOCAB_ROWS

if TYPE_CHECKING:
    from analysis.vocab import EmbeddingShift, NewToken, ReaderTrigger, VocabResult

TITLE = "Vocabulary & Embeddings"
TOKENS_SHOWN = 8
TOP_NEW = 15


@dataclass
class VocabState:
    """Answers of the Vocabulary & Embeddings flow."""

    base: ModelChoice | None = None
    tuned: ModelChoice | None = None
    top_rows: int = DEFAULT_VOCAB_ROWS
    new_tokens: int = DEFAULT_VOCAB_NEW


def run_vocab(session: Session) -> None:
    """Compare a tuned model's vocabulary and embeddings with its base."""
    run_steps(session, (_step_base, _step_tuned, _step_parameters, _step_run), VocabState())


def _step_base(session: Session, state: VocabState) -> None:
    """Pick the reference model (it also provides the base tokenizer)."""
    state.base = select_model(session, (TITLE, "Base Model"), "Base model")


def _step_tuned(session: Session, state: VocabState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session, (TITLE, "Base Model", "Tuned Model"), "Tuned model", context=model_lines([("Base", state.base)])
    )


def _step_parameters(session: Session, state: VocabState) -> None:
    """How many moved rows to list and how many added tokens to characterise."""
    ui = session.ui
    ui.goto(TITLE, "Tuned Model", "Parameters")
    ui.show(*model_lines([("Base", state.base), ("Tuned", state.tuned)]))
    ui.hint("Moved rows: the shared token rows that moved most, per embedding matrix.")
    ui.hint("New tokens: added tokens characterised (trained vs placeholder, emittable vs read-only).")
    state.top_rows = ask_int(ui, "Moved rows listed", state.top_rows, 1, MAX_VOCAB_ROWS)
    state.new_tokens = ask_int(ui, "New tokens read", state.new_tokens, 1, MAX_VOCAB_NEW)


def _step_run(session: Session, state: VocabState) -> None:
    """Run the analysis, save the report and present the results."""
    base, tuned = require(state.base), require(state.tuned)
    work = partial(analyse, base, tuned, state.top_rows, state.new_tokens, session.hf_token)
    result = run_job(session, (TITLE, "Analysing"), Stage.VOCAB, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*vocab_panel(result), *saved], record)


def analyse(
    base: ModelChoice,
    tuned: ModelChoice,
    top_rows: int,
    new_tokens: int,
    hf_token: str | None,
    sink: ProgressSink,
    token: CancelToken,
) -> VocabResult:
    """The job: load both tokenizers, open both models and compare vocabulary and embeddings."""
    from analysis.tokens import load_tokenizer
    from analysis.vocab import compare_vocab
    from models import GGUF_AUTO

    base_tok = load_tokenizer(base.model, token=hf_token, gguf_file=base.gguf_file or GGUF_AUTO)
    tuned_tok = load_tokenizer(tuned.model, token=hf_token, gguf_file=tuned.gguf_file or GGUF_AUTO)
    return compare_vocab(
        base.open(hf_token),
        tuned.open(hf_token),
        tuned_tok,
        inputs=model_inputs(base, tuned),
        base_vocab=_vocab_info_or_none(base_tok),
        tuned_vocab=_vocab_info_or_none(tuned_tok),
        top_rows=top_rows,
        max_new=new_tokens,
        sink=sink,
        token=token,
    )


def _vocab_info_or_none(tokenizer: Any) -> Any:
    """The tokenizer's vocabulary, or None when it exposes none (the diff is then skipped)."""
    from analysis.tokens import vocab_info

    try:
        result: Any = vocab_info(tokenizer)
    except Exception:  # noqa: BLE001 - no vocabulary just means no diff; the embeddings still compare
        result = None
    return result


def vocab_panel(result: VocabResult) -> list[RenderableType]:
    """The results: the diff summary, the embedding shift, the new tokens, warnings."""
    return [
        *diff_lines(result),
        *embedding_table(result.embeddings),
        *new_token_table(result.new_tokens),
        *reader_trigger_lines(result.reader_triggers),
        *messages("Warnings", result.meta.warnings, "yellow"),
        *messages("Notes", result.meta.notes[:5], "dim"),
    ]


def diff_lines(result: VocabResult) -> list[RenderableType]:
    """The vocabulary diff summary: sizes, added, removed, reassigned, special tokens."""
    diff = result.diff
    if diff is None:
        return [Text(f"No tokenizer diff; {result.vocabulary:,} rows aligned by position.", style="dim")]
    lines: list[RenderableType] = [
        Text(
            f"Base vocab {diff.base_size:,}  tuned vocab {diff.tuned_size:,}  aligned {diff.aligned_tokens:,}  "
            f"added {len(diff.added)}  removed {len(diff.removed)}  reassigned {len(diff.reassigned)}"
        )
    ]
    if diff.added:
        lines.append(Text.assemble("  added: ", (tokens_line([change.text for change in diff.added]), "cyan")))
    if diff.reassigned:
        reassigned = tokens_line([change.text for change in diff.reassigned])
        lines.append(Text.assemble(("  reassigned (alignment break): ", "yellow"), (reassigned, "yellow")))
    for line in diff.special_changed:
        lines.append(Text(f"  special: {line}", style="dim"))
    return lines


def embedding_table(shifts: Sequence[EmbeddingShift]) -> list[RenderableType]:
    """One row per embedding matrix: how many rows moved and the concentration of the movement."""
    if not shifts:
        return []
    rows = [
        [
            shift.name,
            f"{shift.rows_changed}/{shift.rows_compared}",
            number(shift.concentration.z, 3) if shift.concentration else "-",
            tokens_line([row.text for row in shift.moved[:TOKENS_SHOWN]]),
        ]
        for shift in shifts
    ]
    columns = ("Matrix", "Rows moved", "Top-row z", "Most-moved tokens")
    return [build_table(columns, rows, "Shared-row embedding shift")]


def new_token_table(tokens: Sequence[NewToken]) -> list[RenderableType]:
    """One row per added token: trained, emittable and its nearest existing tokens."""
    if not tokens:
        return []
    rows = [
        [
            token.text,
            number(token.input_norm),
            number(token.output_norm),
            "yes" if token.trained else "no",
            _flag(token.emittable),
            tokens_line([neighbour.text for neighbour in token.neighbours[:TOKENS_SHOWN]]),
        ]
        for token in tokens[:TOP_NEW]
    ]
    columns = ("Token", "Input norm", "Output norm", "Trained", "Emittable", "Nearest tokens")
    return [build_table(columns, rows, f"New tokens (first {min(len(tokens), TOP_NEW)} of {len(tokens)})")]


def reader_trigger_lines(triggers: Sequence[ReaderTrigger]) -> list[RenderableType]:
    """The heaviest reader updates re-projected onto the suspect vocabulary, new-token hits flagged."""
    if not triggers:
        return []
    lines: list[RenderableType] = [Text("Reader triggers (extended vocabulary)", style="bold")]
    for trigger in triggers:
        ends = (trigger.projection.positive, trigger.projection.negative) if trigger.projection else ()
        top = max(ends, key=lambda end: end.null.z, default=None)
        tokens = tokens_line([token.text for token in top.tokens[:TOKENS_SHOWN]]) if top is not None else "-"
        z = number(top.null.z, 3) if top is not None else "-"
        line = Text.assemble(f"  {trigger.name} (z {z}): ", (tokens, "cyan"))
        if trigger.new_hits:
            hits = tokens_line([hit.text for hit in trigger.new_hits])
            line.append_text(Text.assemble(("  ← added token trigger: ", "red"), (hits, "red")))
        lines.append(line)
    return lines


def _flag(value: bool | None) -> str:
    """A yes/no/dash rendering of an optional flag."""
    return "-" if value is None else ("yes" if value else "no")
