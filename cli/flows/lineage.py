"""Hub Lineage: the declared parents and children of a Hub model, compared statically.

Steps: model id, mode (auto / parent / children), limits (variants kept,
maximum size), discovery (shown, then the pairs to analyse are chosen), then
the token-shift comparison of every chosen pair, summarised pair by pair, with
the report saved.

Discovery is kept in the flow state with the settings it was made with, so
stepping back from the analysis does not query the Hub again.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    acknowledge,
    ask_float,
    ask_int,
    messages,
    number,
    present,
    require,
    run_steps,
    save_report,
)
from cli.completer import reference_completer
from cli.running import run_job
from cli.ui import Choice, GoBack, build_table
from progress import CancelToken, ProgressSink, Stage
from settings import DEFAULT_MAX_SIZE_GB, DEFAULT_VARIANTS, MAX_VARIANTS

if TYPE_CHECKING:
    from analysis.lineage import LineageDiscovery, LineageResult, PairResult
    from analysis.tokens import TokenDecoder

TITLE = "Hub Lineage"
MODES: tuple[tuple[str, str, str], ...] = (
    ("auto", "Parents and children", "declared bases, then the most downloaded variants"),
    ("parent", "Parents only", "compare the model with its declared base"),
    ("children", "Children only", "compare the model's adapters and fine-tunes with it"),
)
ALL_PAIRS = "all"


@dataclass
class LineageState:
    """Answers of the Hub Lineage flow."""

    model_id: str = ""
    mode: str = "auto"
    max_variants: int = DEFAULT_VARIANTS
    max_size_gb: float = DEFAULT_MAX_SIZE_GB
    discovery: LineageDiscovery | None = None
    discovered_with: tuple[str, str, int, float] | None = None
    selected: list[int] = dataclasses.field(default_factory=list)


def run_lineage(session: Session) -> None:
    """Discover a Hub model's relatives and compare the chosen pairs."""
    steps = (_step_model, _step_mode, _step_limits, _step_discover, _step_run)
    run_steps(session, steps, LineageState())


def _step_model(session: Session, state: LineageState) -> None:
    """The Hub repository to investigate."""
    ui = session.ui
    ui.goto(TITLE, "Model")
    ui.hint("A Hugging Face repository id, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct.")
    completer = reference_completer(session.hf_token) if ui.supports_completion else None
    answer = ui.ask("Model id", default=state.model_id, completer=completer)
    if not answer:
        raise GoBack
    state.model_id = answer


def _step_mode(session: Session, state: LineageState) -> None:
    """Which relatives to look for."""
    session.ui.goto(TITLE, "Model", "Mode")
    options = [Choice(key, label, detail) for key, label, detail in MODES]
    state.mode = session.ui.choose(options, Text(state.model_id, style="bold")).key


def _step_limits(session: Session, state: LineageState) -> None:
    """How many variants to keep and how large they may be."""
    ui = session.ui
    ui.goto(TITLE, "Mode", "Limits")
    ui.hint("Variants: children kept, the most downloaded first. Size: larger full-weight children are skipped.")
    state.max_variants = ask_int(ui, "Maximum variants", state.max_variants, 1, MAX_VARIANTS)
    state.max_size_gb = ask_float(ui, "Maximum size (GB)", state.max_size_gb)


def _step_discover(session: Session, state: LineageState) -> None:
    """Query the Hub (unless already done with these settings), show what was found, pick the pairs."""
    settings = (state.model_id, state.mode, state.max_variants, state.max_size_gb)
    if state.discovery is None or state.discovered_with != settings:
        work = partial(discover, settings, session.hf_token)
        state.discovery = run_job(session, (TITLE, "Limits", "Discovering"), Stage.LINEAGE, work)
        state.discovered_with = settings
    discovery = state.discovery
    ui = session.ui
    ui.goto(TITLE, "Limits", "Discovery")
    ui.show(*discovery_panel(discovery))
    if not discovery.pairs:
        acknowledge(ui, "No comparable pair was found.")
        raise GoBack
    state.selected = _ask_pairs(session, len(discovery.pairs))


def discover(
    settings: tuple[str, str, int, float], hf_token: str | None, sink: ProgressSink, token: CancelToken
) -> LineageDiscovery:
    """The job: read the Hub metadata of the model and its relatives."""
    from analysis.lineage import discover_lineage
    from progress import report

    model_id, mode, max_variants, max_size_gb = settings
    report(sink, Stage.LINEAGE, f"Reading the Hub metadata of {model_id}")
    return discover_lineage(model_id, mode=mode, max_variants=max_variants, max_size_gb=max_size_gb, token=hf_token)


def _size(size_gb: float | None) -> str:
    """A repository size in GB for a table cell, or a dash when the repository could not be read."""
    return f"{number(size_gb)} GB" if size_gb is not None else number(None)


def discovery_panel(discovery: LineageDiscovery) -> list[RenderableType]:
    """Parents, children, comparable pairs and skipped repositories."""
    columns = ("Repository", "Relation", "Downloads", "Size")
    panel: list[RenderableType] = [Text(f"{discovery.target} ({discovery.kind.value})", style="bold")]
    if discovery.parents:
        rows = [[p.repo, p.relation, number(p.downloads), _size(p.size_gb)] for p in discovery.parents]
        panel.append(build_table(columns, rows, "Declared parents"))
    if discovery.children:
        rows = [[c.repo, c.relation, number(c.downloads), _size(c.size_gb)] for c in discovery.children]
        panel.append(build_table(columns, rows, "Children"))
    pair_rows = [
        [str(index), pair.parent, pair.child, pair.kind.value] for index, pair in enumerate(discovery.pairs, 1)
    ]
    panel.append(build_table(("#", "Parent", "Child", "Kind"), pair_rows, "Comparable pairs"))
    if discovery.skipped:
        rows = [[item.repo, item.reason] for item in discovery.skipped]
        panel.append(build_table(("Repository", "Reason"), rows, "Skipped"))
    return [*panel, *messages("Notes", discovery.notes, "dim")]


def _ask_pairs(session: Session, count: int) -> list[int]:
    """Read which pairs to analyse until the answer is valid (0-based indices)."""
    selected: list[int] | None = None
    while selected is None:
        answer = session.ui.ask(f"Pairs to analyse ('{ALL_PAIRS}' or numbers, e.g. 1,3)", default=ALL_PAIRS)
        selected = parse_selection(answer, count)
        if selected is None:
            session.ui.error(f"Enter '{ALL_PAIRS}' or numbers between 1 and {count}, separated by commas.")
    return selected


def parse_selection(text: str, count: int) -> list[int] | None:
    """0-based indices from ``all`` or a comma list of 1-based numbers; None when invalid."""
    items = [item.strip() for item in text.split(",") if item.strip()]
    result: list[int] | None = None
    if text.strip().lower() == ALL_PAIRS:
        result = list(range(count))
    elif items and all(item.isdigit() and 1 <= int(item) <= count for item in items):
        result = sorted({int(item) - 1 for item in items})
    return result


def _step_run(session: Session, state: LineageState) -> None:
    """Compare the chosen pairs, save the report and present the summary."""
    discovery = require(state.discovery)
    chosen = dataclasses.replace(discovery, pairs=[discovery.pairs[index] for index in state.selected])
    work = partial(analyse, chosen, session.hf_token)
    result = run_job(session, (TITLE, "Discovery", "Analysing"), Stage.LINEAGE, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*lineage_panel(result), *saved], record)


def analyse(discovery: LineageDiscovery, hf_token: str | None, sink: ProgressSink, token: CancelToken) -> LineageResult:
    """The job: the static token-shift comparison of every chosen pair."""
    from analysis.lineage import analyse_lineage

    return analyse_lineage(
        discovery, decoder_factory=partial(parent_decoder, hf_token), token_hf=hf_token, sink=sink, token=token
    )


def parent_decoder(hf_token: str | None, parent: str) -> TokenDecoder:
    """The tokenizer of a parent repository (the vocabulary the updates are read in)."""
    from analysis.tokens import load_tokenizer
    from models import detect

    decoder: TokenDecoder = load_tokenizer(detect(parent, token=hf_token), token=hf_token)
    return decoder


def lineage_panel(result: LineageResult) -> list[RenderableType]:
    """One row per pair: what was compared and the heaviest update, or why it failed."""
    rows = [_pair_row(index, pair) for index, pair in enumerate(result.pairs, start=1)]
    table = build_table(("#", "Parent", "Child", "Kind", "Changed", "Heaviest update"), rows, "Pairs")
    return [table, *messages("Warnings", result.meta.warnings, "yellow")]


def _pair_row(index: int, pair: PairResult) -> list[str | Text]:
    """The summary cells of one compared pair."""
    changed: str | Text = Text(f"failed: {pair.error}", style="red")
    heaviest = "-"
    if pair.shifts is not None:
        counts = pair.shifts.counts
        changed = f"{counts.changed}/{counts.compared}"
        updates = pair.shifts.top_updates
        heaviest = f"{updates[0].name} (sigma1 {number(updates[0].sigma, 3)})" if updates else "-"
    return [str(index), pair.pair.parent, pair.pair.child, pair.pair.kind.value, changed, heaviest]
