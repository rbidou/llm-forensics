"""Provenance: which candidate models a suspect's weights derive from.

Steps: suspect, explicit candidates (optional), Hub search options, depth
parameters, then the screen + deep comparison and its results (a verdict
banner coloured by rank, the ranking, warnings) with the report saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.panel import Panel
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
    split_list,
)
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import build_table
from progress import CancelToken, ProgressSink, Stage
from settings import DEFAULT_LAYERS, DEFAULT_TOP_K, MAX_LAYERS, MAX_TOP_K

if TYPE_CHECKING:
    from analysis.provenance import ProvenanceResult

TITLE = "Provenance"
SKIPPED_SHOWN = 10
# Verdict ranks (see analysis.provenance): 5 identical, 4 same basis, 3 related, 2 transformed, 0 none.
STRONG_RANK = 4
WEAK_RANK = 2


@dataclass
class ProvenanceState:
    """Answers of the Provenance flow."""

    suspect: ModelChoice | None = None
    candidates: list[str] = field(default_factory=list)
    search: bool = True
    search_text: str = ""
    include_derived: bool = False
    layers: int = DEFAULT_LAYERS
    top_k: int = DEFAULT_TOP_K


def run_provenance(session: Session) -> None:
    """Find where a suspect model's weights come from."""
    steps = (_step_suspect, _step_candidates, _step_search, _step_parameters, _step_run)
    run_steps(session, steps, ProvenanceState())


def _step_suspect(session: Session, state: ProvenanceState) -> None:
    """Pick the model under investigation."""
    state.suspect = select_model(session, (TITLE, "Suspect"), "Suspect model", offer_base=True)


def _step_candidates(session: Session, state: ProvenanceState) -> None:
    """Optional explicit candidate parents."""
    ui = session.ui
    ui.goto(TITLE, "Suspect", "Candidates")
    ui.show(*model_lines([("Suspect", state.suspect)]))
    ui.hint("Candidate parents to compare, as paths or Hub ids separated by commas. Leave empty for none.")
    ui.hint("Declared bases are always added; the Hub search is the next question.")
    state.candidates = split_list(ui.ask("Candidates", default=", ".join(state.candidates)))


def _step_search(session: Session, state: ProvenanceState) -> None:
    """Whether to search the Hub for same-type models, and how."""
    ui = session.ui
    ui.goto(TITLE, "Candidates", "Hub Search")
    ui.show(*model_lines([("Suspect", state.suspect)]), Text(f"Candidates: {', '.join(state.candidates) or 'none'}"))
    state.search = ui.confirm("Search the Hub for candidates?", default=state.search)
    if state.search:
        ui.hint("Free text replaces the default search by model type (leave empty to keep it).")
        state.search_text = ui.ask("Search text", default=state.search_text)
        state.include_derived = ui.confirm("Include models that declare a base model?", default=state.include_derived)


def _step_parameters(session: Session, state: ProvenanceState) -> None:
    """Depth of the comparison."""
    ui = session.ui
    ui.goto(TITLE, "Hub Search", "Parameters")
    ui.hint("Layers: compared tensor by tensor from layer 0. Top k: screened candidates compared in depth.")
    state.layers = ask_int(ui, "Layers compared in depth", state.layers, 1, MAX_LAYERS)
    state.top_k = ask_int(ui, "Top k", state.top_k, 1, MAX_TOP_K)


def _step_run(session: Session, state: ProvenanceState) -> None:
    """Run the analysis, save the report and present the verdict."""
    suspect = require(state.suspect)
    work = partial(analyse, suspect, state, session.hf_token)
    result = run_job(session, (TITLE, "Analysing"), Stage.PROVENANCE, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*provenance_panel(result), *saved], record)


def analyse(
    suspect: ModelChoice, state: ProvenanceState, hf_token: str | None, sink: ProgressSink, token: CancelToken
) -> ProvenanceResult:
    """The job: gather, screen and deep-compare the candidates."""
    from analysis.provenance import find_provenance

    return find_provenance(
        suspect.ref,
        suspect_gguf_file=suspect.gguf_file,
        suspect_base=suspect.base_override,
        candidates=state.candidates,
        search=state.search,
        search_text=state.search_text or None,
        include_derived=state.include_derived,
        top_k=state.top_k,
        layers=state.layers,
        hf_token=hf_token,
        sink=sink,
        token=token,
    )


def provenance_panel(result: ProvenanceResult) -> list[RenderableType]:
    """The verdict banner, the ranking, skipped candidates and warnings."""
    counts = Text(
        f"Suspect {result.suspect.ref} ({result.suspect.label}, {result.suspect.architecture})  "
        f"gathered {result.gathered}  compatible {result.compatible}  screened {len(result.screen)}  "
        f"deep-compared {len(result.deep)}"
    )
    panel: list[RenderableType] = [verdict_banner(result), counts, _ranking_table(result)]
    if result.skipped:
        rows = [[item.ref, item.origin, item.reason] for item in result.skipped[:SKIPPED_SHOWN]]
        panel.append(build_table(("Candidate", "Origin", "Reason"), rows, f"Skipped ({len(result.skipped)})"))
    return [*panel, *messages("Warnings", result.meta.warnings, "yellow"), *messages("Notes", result.meta.notes, "dim")]


def verdict_banner(result: ProvenanceResult) -> Panel:
    """The best match and its verdict, coloured by how strong the evidence is."""
    top = result.ranking[0] if result.ranking else None
    rank = top.rank if top is not None else 0
    style = "green" if rank >= STRONG_RANK else "yellow" if rank >= WEAK_RANK else "red"
    text = Text("No candidate shows weight-level evidence of being the origin.", style=f"bold {style}")
    if result.best is not None and top is not None:
        text = Text.assemble(("Best match: ", "bold"), (result.best, f"bold {style}"), "\n", (top.verdict, style))
    elif top is not None:
        text.append(f"\nClosest: {top.ref} - {top.verdict}", style="dim")
    return Panel(text, border_style=style, title="Verdict", title_align="left")


def _ranking_table(result: ProvenanceResult) -> RenderableType:
    """Every deep-compared candidate, strongest evidence first."""
    rows = [
        [str(index), item.ref, item.origin, str(item.rank), item.verdict, number(item.raw_med), number(item.inv_mean)]
        for index, item in enumerate(result.ranking, start=1)
    ]
    return build_table(("#", "Candidate", "Origin", "Rank", "Verdict", "Raw cos (med)", "Invariant r"), rows, "Ranking")
