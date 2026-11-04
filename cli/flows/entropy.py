"""Entropy Analysis: how the entropy of the last layers' weights shifts from source to fine-tune.

Steps: base model, tuned model, parameters (last blocks examined, histogram
bins), then the analysis and its results (the two null baselines, the tensors
whose spectral entropy moved most, a per-layer profile) with the report saved
and opened in the browser.
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
    messages,
    number,
    present,
    require,
    run_steps,
    save_report,
)
from cli.flows.weights import model_inputs
from cli.running import run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import build_table
from progress import CancelToken, ProgressSink, Stage
from settings import (
    DEFAULT_ENTROPY_BINS,
    DEFAULT_ENTROPY_LAYERS,
    MAX_ENTROPY_BINS,
    MAX_ENTROPY_LAYERS,
)

if TYPE_CHECKING:
    from analysis.entropy import EntropyResult

TITLE = "Entropy Analysis"
TOP_TENSORS = 15
MIN_BINS = 2


@dataclass
class EntropyState:
    """Answers of the Entropy Analysis flow."""

    base: ModelChoice | None = None
    tuned: ModelChoice | None = None
    layers: int = DEFAULT_ENTROPY_LAYERS
    bins: int = DEFAULT_ENTROPY_BINS


def run_entropy(session: Session) -> None:
    """Compare the entropy of a fine-tune's last layers with its source."""
    run_steps(session, (_step_base, _step_tuned, _step_parameters, _step_run), EntropyState())


def _step_base(session: Session, state: EntropyState) -> None:
    """Pick the source model (it also fixes the layer count and the scope)."""
    state.base = select_model(session, (TITLE, "Source Model"), "Source model")


def _step_tuned(session: Session, state: EntropyState) -> None:
    """Pick the fine-tuned model compared with the source."""
    state.tuned = select_model(
        session, (TITLE, "Source Model", "Tuned Model"), "Tuned model", context=model_lines([("Source", state.base)])
    )


def _step_parameters(session: Session, state: EntropyState) -> None:
    """How many last blocks to examine and how many histogram bins to use."""
    ui = session.ui
    ui.goto(TITLE, "Tuned Model", "Parameters")
    ui.show(*model_lines([("Source", state.base), ("Tuned", state.tuned)]))
    ui.hint("Blocks: the last transformer blocks examined, besides the final norm and the output head.")
    ui.hint("Bins: buckets of the weight-value histogram the value-distribution entropy is read from.")
    state.layers = ask_int(ui, "Last blocks to examine", state.layers, 1, MAX_ENTROPY_LAYERS)
    state.bins = ask_int(ui, "Histogram bins", state.bins, MIN_BINS, MAX_ENTROPY_BINS)


def _step_run(session: Session, state: EntropyState) -> None:
    """Run the analysis, save the report and present the results."""
    base, tuned = require(state.base), require(state.tuned)
    work = partial(analyse, base, tuned, state.layers, state.bins, session.hf_token)
    result = run_job(session, (TITLE, "Analysing"), Stage.ENTROPY, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*entropy_panel(result), *saved], record)


def analyse(
    base: ModelChoice,
    tuned: ModelChoice,
    layers: int,
    bins: int,
    hf_token: str | None,
    sink: ProgressSink,
    token: CancelToken,
) -> EntropyResult:
    """The job: open both models and compare the entropy of their last layers."""
    from analysis.entropy import analyse_entropy

    return analyse_entropy(
        base.open(hf_token),
        tuned.open(hf_token),
        inputs=model_inputs(base, tuned),
        layers=layers,
        bins=bins,
        sink=sink,
        token=token,
    )


def entropy_panel(result: EntropyResult) -> list[RenderableType]:
    """The results: counts, the two null baselines, the tensors that moved most, per-layer means."""
    return [
        _counts_line(result),
        *_null_lines(result),
        _tensor_table(result),
        *_layer_lines(result),
        *messages("Warnings", result.meta.warnings, "yellow"),
        *messages("Notes", result.meta.notes[:5], "dim"),
    ]


def _counts_line(result: EntropyResult) -> Text:
    """Examined / spectral / changed, the block window, and beyond-noise when quantized."""
    counts = result.counts
    blocks = f"last {result.meta.parameters.get('layers')}" if result.layers is not None else "layers unknown"
    text = Text.assemble(
        ("Examined ", "bold"),
        f"{counts.in_scope}  ",
        ("spectral ", "bold"),
        f"{counts.spectral}  ",
        ("changed ", "bold"),
        f"{counts.changed}  ",
        ("blocks ", "bold"),
        f"{blocks}",
    )
    if result.beyond_noise is not None:
        text.append_text(Text.assemble(("  beyond quantization noise ", "bold yellow"), str(result.beyond_noise)))
    return text


def _null_lines(result: EntropyResult) -> list[RenderableType]:
    """The mean entropy shifts against the random-update baseline, with their z-scores."""
    lines: list[RenderableType] = []
    for label, null in (("spectral", result.spectral_null), ("value", result.value_null)):
        if null is not None:
            lines.append(
                Text.assemble(
                    (f"Mean {label} ΔH ", "bold"),
                    f"{number(null.value)}  ",
                    ("vs random update ", "dim"),
                    f"{number(null.null_mean)} ± {number(null.null_sd)}  ",
                    ("z ", "dim"),
                    number(null.z, 3),
                )
            )
    return lines


def _tensor_table(result: EntropyResult) -> RenderableType:
    """The tensors whose spectral entropy moved most."""
    columns = ["Tensor", "Layer", "Spectral base", "Spectral tuned", "Spectral Δ", "Rank", "Value Δ"]
    rows = [
        [
            tensor.name,
            str(tensor.layer) if tensor.layer >= 0 else "head",
            number(tensor.spectral.base),
            number(tensor.spectral.tuned),
            number(tensor.spectral.delta),
            number(tensor.effective_rank),
            number(tensor.value.delta),
        ]
        for tensor in result.tensors[:TOP_TENSORS]
    ]
    return build_table(columns, rows, f"Top {TOP_TENSORS} tensors by spectral entropy shift")


def _layer_lines(result: EntropyResult) -> list[RenderableType]:
    """Mean entropy shift per layer, spectral and value."""
    lines: list[RenderableType] = [Text("Mean entropy shift per layer", style="bold")] if result.by_layer else []
    for group in result.by_layer:
        label = "head" if group.layer == -1 else f"layer {group.layer}"
        lines.append(
            Text(
                f"  {label:>9}  spectral Δ {number(group.mean_spectral_delta)}  "
                f"value Δ {number(group.mean_value_delta)}  ({group.tensors} tensors)",
                style="dim",
            )
        )
    return lines
