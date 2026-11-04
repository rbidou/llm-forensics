"""The model selection step shared by every flow.

The user types a local path or a Hub id; :func:`~models.detect`
says what it is, and the step settles whatever detection leaves open:

* a training-run folder holds several ``checkpoint-N`` folders: pick one;
* a GGUF folder or repository holds several files: pick one, most precise first;
* an adapter declares a base model: keep it, or name another.

A reference that cannot be used is reported and asked again, so a typo never
ends the flow. ``back`` at a follow-up question returns to the reference prompt.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import Session
from cli.completer import reference_completer
from cli.ui import Choice, GoBack, Ui
from models.detect import gguf_by_precision
from settings import models_root

if TYPE_CHECKING:
    from models import ModelRef, TensorSource

REFERENCE_HINT = "A local folder or .gguf file, or a Hugging Face repository id (e.g. Qwen/Qwen2.5-1.5B)."
COMPLETION_HINT = "Press Tab to complete paths, the cached models and, for a Hub id, models found on the Hub."


@dataclass(frozen=True)
class ModelChoice:
    """A detected model and the choices detection left to the user."""

    model: ModelRef
    gguf_file: str | None = None  # the chosen file of a GGUF folder or repository holding several
    base_override: str | None = None  # the base an adapter is applied to, instead of the declared one

    @property
    def ref(self) -> str:
        """The reference as typed (or the chosen checkpoint folder)."""
        return self.model.ref

    def describe(self) -> str:
        """One line: the detection label plus the user's choices."""
        parts = [self.model.label]
        if self.gguf_file:
            parts.append(f"file {os.path.basename(self.gguf_file)}")
        base = self.base_override or self.model.base_model
        if base:
            parts.append(f"base {base}{' (overridden)' if self.base_override else ''}")
        return ", ".join(parts)

    def open(self, token: str | None) -> TensorSource:
        """Open the weights for reading (the adapter on its base, the chosen GGUF file)."""
        from models import GGUF_AUTO, open_model

        return open_model(self.model, base=self.base_override, token=token, gguf_file=self.gguf_file or GGUF_AUTO)


def select_model(
    session: Session,
    trail: tuple[str, ...],
    question: str,
    *,
    context: Sequence[RenderableType] = (),
    allow_adapter: bool = True,
    offer_base: bool = True,
) -> ModelChoice:
    """Ask for a model reference until a usable one is given.

    Args:
        session: the running session.
        trail: the location of this step.
        question: the prompt, e.g. ``Base model``.
        context: lines shown above the prompt (the models chosen so far).
        allow_adapter: whether a LoRA adapter is acceptable here.
        offer_base: whether an adapter's base may be replaced.

    Returns:
        The detected model and the user's follow-up choices.

    Raises:
        GoBack: on an empty answer or ``back``.
    """
    choice: ModelChoice | None = None
    notice: list[RenderableType] = []
    while choice is None:
        _show_prompt(session.ui, trail, [*context, *notice])
        completer = reference_completer(session.hf_token) if session.ui.supports_completion else None
        answer = session.ui.ask(question, default=models_root(), completer=completer)
        reference = answer.strip().strip('"')
        if not reference:
            raise GoBack
        choice, notice = _resolve(session, reference, allow_adapter, offer_base)
    return choice


def model_lines(models: Sequence[tuple[str, ModelChoice | None]]) -> list[RenderableType]:
    """One line per model chosen so far: role, reference and what it is."""
    return [
        Text.assemble((f"{role}: ", "bold"), choice.ref, (f"  [{choice.describe()}]", "dim"))
        for role, choice in models
        if choice is not None
    ]


def _show_prompt(ui: Ui, trail: tuple[str, ...], context: Sequence[RenderableType]) -> None:
    """Draw the step: what was chosen so far and how to answer."""
    ui.goto(*trail)
    ui.append(*context)
    ui.hint(REFERENCE_HINT)
    if ui.supports_completion:
        ui.hint(COMPLETION_HINT)


def _resolve(
    session: Session, reference: str, allow_adapter: bool, offer_base: bool
) -> tuple[ModelChoice | None, list[RenderableType]]:
    """Detect a reference and settle the follow-up choices.

    Returns:
        The choice (None when the reference is unusable) and the lines to show
        when asking again.
    """
    from models import ModelError, ModelKind

    result: ModelChoice | None = None
    notice: list[RenderableType] = []
    try:
        model = _detect_or_pick_checkpoint(session, reference)
        if model.kind is ModelKind.ADAPTER and not allow_adapter:
            raise ModelError(f"{model.ref} is a LoRA adapter; full weights, GGUF or bitsandbytes are needed here")
        result = _settle(session, model, offer_base)
    except ModelError as error:
        notice = [Text(str(error), style="red")]
    except GoBack:
        notice = [Text(f"Selection of {reference} abandoned.", style="dim")]
    return result, notice


def _detect_or_pick_checkpoint(session: Session, reference: str) -> ModelRef:
    """Detect ``reference``; a training-run folder offers its checkpoints instead.

    Raises:
        ModelError: when the reference (or the chosen checkpoint) is not usable.
        GoBack: when the checkpoint choice is abandoned.
    """
    from models import ModelError, detect, list_checkpoints

    try:
        model = detect(reference, token=session.hf_token)
    except ModelError as error:
        checkpoints = list_checkpoints(reference) if os.path.isdir(reference) else []
        if not checkpoints:
            raise
        session.ui.goto(*session.ui.trail, "Checkpoint")
        options = [Choice(path, os.path.basename(path)) for path in checkpoints]
        picked = session.ui.choose(options, Text(str(error), style="yellow"), Text("Pick a checkpoint:"))
        model = detect(picked.key, token=session.hf_token)
    return model


def _settle(session: Session, model: ModelRef, offer_base: bool) -> ModelChoice:
    """Ask what detection left open: the GGUF file, the adapter's base.

    Raises:
        GoBack: when a follow-up choice is abandoned.
    """
    from models import ModelKind

    gguf_file = None
    if model.kind is ModelKind.GGUF and len(model.gguf_files) > 1:
        gguf_file = _pick_gguf_file(session.ui, model)
    base = _pick_base(session, model) if model.kind is ModelKind.ADAPTER and offer_base else None
    return ModelChoice(model=model, gguf_file=gguf_file, base_override=base)


def _pick_gguf_file(ui: Ui, model: ModelRef) -> str:
    """Let the user choose one of several GGUF files, the most precise first."""
    ordered = gguf_by_precision(model.gguf_files)
    options = [
        Choice(name, os.path.basename(name), "most precise" if index == 0 else "") for index, name in enumerate(ordered)
    ]
    ui.goto(*ui.trail, "GGUF File")
    return ui.choose(options, Text(f"{model.ref} holds several GGUF files. Which one should be read?")).key


def _pick_base(session: Session, model: ModelRef) -> str | None:
    """Keep the adapter's declared base (None) or return the reference of another one."""
    ui = session.ui
    declared = model.base_model
    ui.goto(*ui.trail, "Adapter Base")
    options = [
        Choice("keep", "Keep the declared base", declared or "", bool(declared), "The adapter declares no base."),
        Choice("other", "Use another base model"),
    ]
    picked = ui.choose(options, Text(f"{model.ref} is a LoRA adapter declaring base: {declared or 'none'}"))
    return None if picked.key == "keep" else _ask_base(session)


def _ask_base(session: Session) -> str:
    """Ask for a base model until one with usable full weights is given."""
    from models import ModelError, ModelKind, detect

    base: str | None = None
    while base is None:
        base_completer = reference_completer(session.hf_token) if session.ui.supports_completion else None
        answer = session.ui.ask("Base model for the adapter", completer=base_completer).strip()
        if not answer:
            raise GoBack
        try:
            detected = detect(answer.strip('"'), token=session.hf_token)
            if detected.kind is ModelKind.ADAPTER:
                raise ModelError(f"{answer} is itself an adapter")
            base = detected.ref
        except ModelError as error:
            session.ui.error(str(error))
    return base
