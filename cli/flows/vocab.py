"""Vocabulary & Embeddings: the tokenizer diff and how the token embeddings moved.

Steps: base model, tuned model, parameters (moved rows listed, new tokens read),
then the analysis and its results (the vocabulary diff, the shared-row embedding
shift with its concentration z-score, and the added tokens characterised as
trained or placeholder and as emittable or read-only) with the report saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    ask_int,
    present,
    require,
    run_steps,
    save_report,
    tokens_line,
)
from cli.flows.weights import model_inputs
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from progress import CancelToken, ProgressSink, Stage
from settings import DEFAULT_VOCAB_NEW, DEFAULT_VOCAB_ROWS, MAX_VOCAB_NEW, MAX_VOCAB_ROWS

if TYPE_CHECKING:
    from analysis.vocab import VocabResult

TITLE = "Vocabulary & Embeddings"


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
    state.base = select_model(session, (TITLE, "Base Model"), "Base model", remember="base")


def _step_tuned(session: Session, state: VocabState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session,
        (TITLE, "Base Model", "Tuned Model"),
        "Tuned model",
        context=model_lines([("Base", state.base)]),
        remember="tuned",
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
    """The vocabulary diff summary; the embedding shift and new-token detail is in the report."""
    return [*diff_lines(result)]


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
