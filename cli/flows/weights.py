"""Weight Diff: how much every tensor of a tuned model moved from its base.

Steps: base model, tuned model, confirmation, then the comparison and its
results (counts, the heaviest tensors, per-layer bars, warnings) with the
report saved and opened in the browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    bar,
    messages,
    number,
    present,
    require,
    run_steps,
    save_report,
)
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import GoBack, build_table
from progress import CancelToken, ProgressSink, Stage

if TYPE_CHECKING:
    from analysis.weights import WeightDiffResult

TITLE = "Weight Diff"
TOP_TENSORS = 15


@dataclass
class WeightsState:
    """Answers of the Weight Diff flow."""

    base: ModelChoice | None = None
    tuned: ModelChoice | None = None


def run_weights(session: Session) -> None:
    """Compare a tuned model's weights with its base."""
    run_steps(session, (_step_base, _step_tuned, _step_confirm, _step_run), WeightsState())


def _step_base(session: Session, state: WeightsState) -> None:
    """Pick the reference model."""
    state.base = select_model(session, (TITLE, "Base Model"), "Base model")


def _step_tuned(session: Session, state: WeightsState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session, (TITLE, "Base Model", "Tuned Model"), "Tuned model", context=model_lines([("Base", state.base)])
    )


def _step_confirm(session: Session, state: WeightsState) -> None:
    """Show both models and ask before starting."""
    session.ui.goto(TITLE, "Tuned Model", "Confirm")
    session.ui.show(*model_lines([("Base", state.base), ("Tuned", state.tuned)]))
    if not session.ui.confirm("Compare the weights?", default=True):
        raise GoBack


def _step_run(session: Session, state: WeightsState) -> None:
    """Run the comparison, save the report and present the results."""
    base, tuned = require(state.base), require(state.tuned)
    work = partial(compare, base, tuned, session.hf_token)
    result = run_job(session, (TITLE, "Comparing"), Stage.WEIGHTS, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*weights_panel(result), *saved], record)


def compare(
    base: ModelChoice, tuned: ModelChoice, hf_token: str | None, sink: ProgressSink, token: CancelToken
) -> WeightDiffResult:
    """The job: open both models and compare their weights."""
    from analysis.weights import compare_weights

    inputs = model_inputs(base, tuned)
    return compare_weights(base.open(hf_token), tuned.open(hf_token), inputs=inputs, sink=sink, token=token)


def model_inputs(base: ModelChoice, tuned: ModelChoice) -> dict[str, str]:
    """The compared models as recorded in the result metadata, with the user's choices."""
    inputs = {"base": base.ref, "tuned": tuned.ref}
    for role, choice in (("base", base), ("tuned", tuned)):
        if choice.gguf_file:
            inputs[f"{role}_gguf_file"] = os.path.basename(choice.gguf_file)
        if choice.base_override:
            inputs[f"{role}_adapter_base"] = choice.base_override
    return inputs


def weights_panel(result: WeightDiffResult) -> list[RenderableType]:
    """The results: counts, the heaviest tensors, per-layer bars, warnings."""
    return [
        _counts_line(result),
        _tensor_table(result),
        *_layer_lines(result),
        *messages("Warnings", result.meta.warnings, "yellow"),
        *messages(f"Notes ({len(result.meta.notes)})", result.meta.notes[:5], "dim"),
    ]


def _counts_line(result: WeightDiffResult) -> Text:
    """Compared / changed / identical / unmatched, and beyond-noise when quantized."""
    text = Text.assemble(
        ("Compared ", "bold"),
        f"{result.compared}  ",
        ("changed ", "bold"),
        f"{result.changed}  ",
        ("identical ", "bold"),
        f"{result.identical}  ",
        ("unmatched ", "bold"),
        f"{result.unmatched}",
    )
    if result.beyond_noise is not None:
        text.append_text(Text.assemble(("  beyond quantization noise ", "bold yellow"), str(result.beyond_noise)))
    return text


def _tensor_table(result: WeightDiffResult) -> RenderableType:
    """The heaviest tensors by relative change."""
    quantized = result.beyond_noise is not None
    columns = ["Tensor", "Layer", "Rel. change", "Cosine", "Changed", "Rank"]
    columns += ["Noise floor", "Beyond noise"] if quantized else []
    rows = []
    for change in result.tensors[:TOP_TENSORS]:
        cells = [
            change.name,
            str(change.layer) if change.layer >= 0 else "-",
            number(change.rel_change),
            number(change.cosine, 6),
            f"{change.frac_changed:.1%}",
            number(change.delta_rank),
        ]
        rows.append(cells + ([number(change.noise_floor), number(change.beyond_noise)] if quantized else []))
    return build_table(columns, rows, f"Top {TOP_TENSORS} tensors by relative change")


def _layer_lines(result: WeightDiffResult) -> list[RenderableType]:
    """Mean relative change per layer, as bars."""
    peak = max((group.mean_rel_change for group in result.by_layer), default=0.0)
    lines: list[RenderableType] = [Text("Mean relative change per layer", style="bold")] if result.by_layer else []
    for group in result.by_layer:
        label = "other" if group.key == -1 else f"layer {group.key}"
        lines.append(
            Text.assemble(f"  {label:>9} ", (bar(group.mean_rel_change, peak), "dark_orange"), " ")
            + Text(f"{number(group.mean_rel_change)} ({group.tensors} tensors)", style="dim")
        )
    return lines
