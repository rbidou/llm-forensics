"""Analyse a Model: run the whole workflow on a model and read a suspicion verdict.

Steps: the model to analyse, its source (given, or discovered with Provenance), the options
(which analyses to run — recommended ones default on — and the optional behaviour probe), a
confirmation, then the workflow and its verdict — the overall risk, the per-tool signals and the
saved report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import Session, ask_path, present, preview, require, run_steps, save_report
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import Choice, GoBack, build_table
from progress import CancelToken, ProgressSink, Stage
from settings import recommended_tools

if TYPE_CHECKING:
    from analysis.assessment import AssessmentResult

TITLE = "Analyse a Model"
ASSESSMENT_SIGNALS_SHOWN = 8  # cap the signals table so the summary never pushes the header off-screen
# Risk level -> rich style (mirrors the report's .verdict classes).
RISK_STYLES: dict[str, str] = {"clean": "green", "low": "cyan", "elevated": "yellow", "high": "red"}


@dataclass
class AssessState:
    """Answers of the workflow flow."""

    suspect: ModelChoice | None = None
    source: ModelChoice | None = None  # None: discover the source with Provenance
    run_weights: bool = True
    run_entropy: bool = False  # off by default; opt-in
    run_shifts: bool = True
    run_vocab: bool = True
    run_behaviour: bool = False
    prompts: tuple[str, ...] = field(default_factory=tuple)


def run_assessment(session: Session) -> None:
    """Analyse a model with the full workflow and read a verdict."""
    run_steps(session, (_step_suspect, _step_source, _step_options, _step_confirm, _step_run), AssessState())


def _step_suspect(session: Session, state: AssessState) -> None:
    """Pick the model to analyse."""
    state.suspect = select_model(session, (TITLE, "Model"), "Model to analyse", remember="tuned")


def _step_source(session: Session, state: AssessState) -> None:
    """Provide the source model, or choose to discover it with Provenance."""
    ui = session.ui
    ui.goto(TITLE, "Source")
    ui.append(*model_lines([("Model", state.suspect)]))
    options = (
        Choice("provide", "Provide the source model", "the model it is claimed to derive from"),
        Choice("discover", "Discover the source with Provenance", "search its declared parents and the Hub"),
    )
    if ui.choose(options).key == "provide":
        state.source = select_model(
            session,
            (TITLE, "Source"),
            "Source model",
            context=model_lines([("Model", state.suspect)]),
            remember="base",
        )
    else:
        state.source = None


def _step_options(session: Session, state: AssessState) -> None:
    """Choose which analyses to run (those relevant for the model's format default on) and prompts."""
    ui = session.ui
    ui.goto(TITLE, "Options")
    kind = state.suspect.model.kind.value if state.suspect is not None else None
    recommended = recommended_tools(kind)
    ui.append(Text("Choose the analyses to run (those recommended for this model's format are on):", style="dim"))
    state.run_weights = ui.confirm(_tool_prompt("Weight Diff", recommended["weights"]), default=recommended["weights"])
    state.run_entropy = ui.confirm(
        _tool_prompt("Entropy Analysis", recommended["entropy"]), default=recommended["entropy"]
    )
    state.run_shifts = ui.confirm(_tool_prompt("Token Shifts", recommended["shifts"]), default=recommended["shifts"])
    state.run_vocab = ui.confirm(
        _tool_prompt("Vocabulary & Embeddings analysis", recommended["vocab"]), default=recommended["vocab"]
    )
    state.run_behaviour = ui.confirm("Run the behaviour probe? (loads both models; needs a GPU)", default=False)
    prompts: tuple[str, ...] = ()
    if state.run_behaviour:
        path = ask_path(ui, "Prompts file to test (optional; .txt/.json/.jsonl)")
        if path:
            prompts = _load_prompts(session, path)
    state.prompts = prompts


def _tool_prompt(label: str, recommended: bool) -> str:
    """The confirm question for a tool, flagged ``(recommended)`` when it is relevant here."""
    suffix = " (recommended)" if recommended else ""
    return f"Run the {label}?{suffix}"


def _load_prompts(session: Session, path: str) -> tuple[str, ...]:
    """Read the behaviour prompts of a file, or warn and return none."""
    from analysis.behaviour import load_prompts

    result: tuple[str, ...] = ()
    try:
        result = tuple(load_prompts(path))
    except (OSError, ValueError) as error:
        session.ui.error(f"Could not read {path}: {error}")
    return result


def _step_confirm(session: Session, state: AssessState) -> None:
    """Show the choices and ask before starting."""
    ui = session.ui
    ui.goto(TITLE, "Confirm")
    lines = model_lines([("Model", state.suspect)])
    if state.source is not None:
        lines += model_lines([("Source", state.source)])
    else:
        lines.append(Text("Source: discover with Provenance", style="dim"))
    lines.append(Text(f"Analyses: {_enabled_analyses(state)}", style="dim"))
    behaviour = "on" if state.run_behaviour else "off"
    if state.prompts:
        behaviour += f" ({len(state.prompts)} prompts)"
    lines.append(Text(f"Behaviour probe: {behaviour}", style="dim"))
    ui.show(*lines)
    if not ui.confirm("Run the analysis?", default=True):
        raise GoBack


def _enabled_analyses(state: AssessState) -> str:
    """The enabled weight-level analyses as a comma-separated list (the probe is shown separately)."""
    pairs = (
        (state.run_weights, "Weight Diff"),
        (state.run_entropy, "Entropy"),
        (state.run_shifts, "Token Shifts"),
        (state.run_vocab, "Vocabulary & Embeddings"),
    )
    names = [label for enabled, label in pairs if enabled]
    return ", ".join(names) if names else "none"


def _step_run(session: Session, state: AssessState) -> None:
    """Run the workflow, save the report and present the verdict."""
    suspect = require(state.suspect)
    work = partial(_assess, suspect, state.source, state, session.hf_token)
    result = run_job(session, (TITLE, "Analysing"), Stage.ASSESSMENT, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*assessment_panel(result), *saved], record)


def _assess(
    suspect: ModelChoice,
    source: ModelChoice | None,
    state: AssessState,
    hf_token: str | None,
    sink: ProgressSink,
    token: CancelToken,
) -> AssessmentResult:
    """The job: run the whole workflow and read its verdict."""
    from analysis.assessment import AssessmentOptions, ModelInput, assess_model

    source_input = ModelInput(source.ref, source.gguf_file, source.base_override) if source is not None else None
    options = AssessmentOptions(
        run_weights=state.run_weights,
        run_entropy=state.run_entropy,
        run_shifts=state.run_shifts,
        run_vocab=state.run_vocab,
        run_behaviour=state.run_behaviour,
        behaviour_prompts=state.prompts,
    )
    return assess_model(
        ModelInput(suspect.ref, suspect.gguf_file, suspect.base_override),
        source_input,
        options=options,
        hf_token=hf_token,
        sink=sink,
        token=token,
    )


def assessment_panel(result: AssessmentResult) -> list[RenderableType]:
    """The risk and confidence lines and the signals table; the executive summary and warnings are in the report."""
    verdict = result.verdict
    style = RISK_STYLES.get(verdict.level, "")
    return [
        Text.assemble((f"{verdict.level.upper()} RISK", f"bold {style}"), "  ", verdict.headline),
        Text(f"confidence {verdict.confidence}  ·  {verdict.flags} red flag(s)", style="dim"),
        _signals_table(result.signals),
    ]


def _signals_table(signals: list) -> RenderableType:  # type: ignore[type-arg]  # list[Signal]; avoids a torch import
    """The per-tool signals, strongest first, capped so the summary stays on one screen."""
    rows = [
        [str(signal.severity), signal.tool, signal.title, preview(signal.detail)]
        for signal in signals[:ASSESSMENT_SIGNALS_SHOWN]
    ]
    return build_table(("Severity", "Tool", "Finding", "Detail"), rows, "Signals (strongest first)")
