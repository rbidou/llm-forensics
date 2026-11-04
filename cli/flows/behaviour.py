"""Behaviour Probe: where a tuned model's predictions depart from its base's.

Steps: base model, tuned model, then a mode:

* batch -- a prompts file (path, JSON field, limit), options (chat template,
  generated tokens), then :func:`~analysis.behaviour.probe` and its
  results (mean KL against the control prompts, per-prompt table, hottest
  positions, promoted / suppressed tokens) with the report saved;
* interactive -- the pair is loaded once, then a session in the same layout
  where every line typed is a prompt to analyse, or a ``:`` command
  (:data:`HELP`). Malformed commands are rejected, never analysed as text; an
  error on one prompt is shown and the session goes on; the pair is always
  closed when the session ends, however it ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.text import Text

from cli.common import (
    Session,
    ask_int,
    ask_path,
    messages,
    number,
    present,
    preview,
    require,
    run_steps,
    save_report,
    tokens_line,
)
from cli.flows.weights import model_inputs
from cli.running import AnalysisStopped, run_job
from cli.selection import ModelChoice, model_lines, select_model
from cli.ui import Choice, GoBack, build_table
from progress import CancelToken, ProgressSink, Stage, report
from settings import DEFAULT_FIELD, DEFAULT_LIMIT, DEFAULT_TOP, MAX_GENERATE, MAX_LIMIT

if TYPE_CHECKING:
    from analysis.behaviour import (
        BehaviourResult,
        GenerationPair,
        HotPosition,
        ModelPair,
        PromptComparison,
        TokenShift,
    )

TITLE = "Behaviour Probe"
INTERACTIVE = "Interactive"
HOTTEST_SHOWN = 10
SHIFTS_SHOWN = 12
GENERATION_PREVIEW = 200
PROMPT_QUESTION = "Prompt or :command > "
HELP = (
    "Type a prompt to compare both models on it, or a command:",
    "  :gen <n>       also generate n tokens greedily from both models (0 = off)",
    "  :top <k>       positions and tokens listed",
    "  :chat          toggle the chat template",
    "  :file <path>   analyse the content of a file as the prompt",
    "  :save          probe every prompt typed so far and save the report",
    "  :help          this list",
    "  back | home | exit   leave the session (the models are unloaded)",
)


@dataclass
class BehaviourState:
    """Answers of the Behaviour Probe flow."""

    base: ModelChoice | None = None
    tuned: ModelChoice | None = None
    prompts_file: str = ""
    json_field: str = DEFAULT_FIELD
    limit: int = DEFAULT_LIMIT
    prompts: list[str] = field(default_factory=list)
    chat: bool = True
    generate_tokens: int = 0


@dataclass
class Command:
    """One line of the interactive session, parsed."""

    kind: str  # empty | prompt | gen | top | chat | file | save | help | invalid
    text: str = ""
    value: int = 0


@dataclass
class Repl:
    """The interactive session: the loaded pair and its settings."""

    pair: ModelPair
    chat: bool
    top: int = DEFAULT_TOP
    generate_tokens: int = 0
    prompts: list[str] = field(default_factory=list)


def run_behaviour(session: Session) -> None:
    """Compare the predictions of a tuned model with its base's."""
    steps = (_step_base, _step_tuned, _step_mode, _step_prompts, _step_options, _step_run)
    run_steps(session, steps, BehaviourState())


def _step_base(session: Session, state: BehaviourState) -> None:
    """Pick the reference model (not an adapter: it is loaded as is)."""
    state.base = select_model(session, (TITLE, "Base Model"), "Base model", allow_adapter=False)


def _step_tuned(session: Session, state: BehaviourState) -> None:
    """Pick the model compared with the base."""
    state.tuned = select_model(
        session, (TITLE, "Base Model", "Tuned Model"), "Tuned model", context=model_lines([("Base", state.base)])
    )


def _step_mode(session: Session, state: BehaviourState) -> None:
    """Batch on a prompts file, or an interactive session (which returns here when left with back)."""
    options = [
        Choice("batch", "Batch probe on a prompts file", "txt, json or jsonl; saved as a report"),
        Choice("interactive", "Interactive session", "type prompts one by one"),
    ]
    chosen = ""
    while chosen != "batch":
        session.ui.goto(TITLE, "Tuned Model", "Mode")
        chosen = session.ui.choose(options, *model_lines([("Base", state.base), ("Tuned", state.tuned)])).key
        if chosen == "interactive":
            run_interactive(session, state)


def _step_prompts(session: Session, state: BehaviourState) -> None:
    """Read the prompts file until it gives at least one prompt."""
    from analysis.behaviour import load_prompts

    ui = session.ui
    ui.goto(TITLE, "Mode", "Prompts")
    ui.hint(".txt: one prompt per line. .json / .jsonl: strings, or objects holding the field below.")
    prompts: list[str] = []
    while not prompts:
        state.prompts_file = ask_path(ui, "Prompts file", state.prompts_file)
        if not state.prompts_file:
            raise GoBack
        state.json_field = ui.ask("JSON field", default=state.json_field)
        state.limit = ask_int(ui, "Maximum prompts", state.limit, 1, MAX_LIMIT)
        try:
            prompts = load_prompts(state.prompts_file, field=state.json_field, limit=state.limit)
        except (OSError, ValueError) as error:
            ui.error(str(error))
    state.prompts = prompts


def _step_options(session: Session, state: BehaviourState) -> None:
    """Chat template and generations."""
    ui = session.ui
    ui.goto(TITLE, "Prompts", "Options")
    ui.show(Text(f"{len(state.prompts)} prompts loaded from {state.prompts_file}"))
    for prompt in state.prompts[:3]:
        ui.hint(f"  {preview(prompt)}")
    state.chat = ui.confirm("Wrap prompts in the chat template?", default=state.chat)
    state.generate_tokens = ask_int(
        ui, "Tokens generated by both models (0 = none)", state.generate_tokens, 0, MAX_GENERATE
    )


def _step_run(session: Session, state: BehaviourState) -> None:
    """Run the probe, save the report and present the results."""
    base, tuned = require(state.base), require(state.tuned)
    work = partial(batch_probe, base, tuned, state, session.hf_token)
    result = run_job(session, (TITLE, "Probing"), Stage.BEHAVIOUR, work)
    record, saved = save_report(session, result)
    session.ui.goto(TITLE, "Results")
    present(session, [*behaviour_panel(result), *saved], record)


def batch_probe(
    base: ModelChoice,
    tuned: ModelChoice,
    state: BehaviourState,
    hf_token: str | None,
    sink: ProgressSink,
    token: CancelToken,
) -> BehaviourResult:
    """The job: load both models and probe them on the prompts."""
    from analysis.behaviour import probe

    return probe(
        base.model,
        tuned.model,
        state.prompts,
        inputs=model_inputs(base, tuned),
        chat=state.chat,
        generate_tokens=state.generate_tokens,
        tuned_base_override=tuned.base_override,
        gguf_file=tuned.gguf_file,
        base_gguf_file=base.gguf_file,
        hf_token=hf_token,
        sink=sink,
        token=token,
    )


# --- interactive session ---------------------------------------------------


def run_interactive(session: Session, state: BehaviourState) -> None:
    """Load the pair once, run the session, and always unload the pair.

    ``back`` ends the session and returns to the mode menu; ``home`` and
    ``exit`` go further, after the pair is closed.
    """
    base, tuned = require(state.base), require(state.tuned)
    pair = run_job(
        session, (TITLE, INTERACTIVE, "Loading"), Stage.BEHAVIOUR, partial(load, base, tuned, session.hf_token)
    )
    try:
        repl_loop(session, Repl(pair=pair, chat=state.chat), base, tuned)
    except GoBack:
        session.ui.hint("Interactive session closed.")
    finally:
        pair.close()


def load(
    base: ModelChoice, tuned: ModelChoice, hf_token: str | None, sink: ProgressSink, token: CancelToken
) -> ModelPair:
    """The job: load both models and tokenizers for forward passes."""
    from analysis.behaviour import load_pair

    report(sink, Stage.BEHAVIOUR, "Loading both models")
    return load_pair(
        base.model,
        tuned.model,
        tuned_base_override=tuned.base_override,
        gguf_file=tuned.gguf_file,
        base_gguf_file=base.gguf_file,
        hf_token=hf_token,
    )


def repl_loop(session: Session, repl: Repl, base: ModelChoice, tuned: ModelChoice) -> None:
    """Read and handle lines until a navigation command leaves (raised by the UI)."""
    panel = [*placement_lines(repl.pair), *(Text(line, style="dim") for line in HELP)]
    while True:
        session.ui.goto(TITLE, INTERACTIVE)
        session.ui.show(*panel, _settings_line(repl))
        command = parse_command(session.ui.read(PROMPT_QUESTION))
        panel = handle_command(session, repl, command, base, tuned) or panel


def parse_command(line: str) -> Command:
    """Parse one line of the interactive session.

    Free text is a prompt; a line starting with ``:`` must be a known command
    with valid arguments, otherwise it is ``invalid`` (with the reason in
    ``text``) -- never analysed as a prompt.
    """
    stripped = line.strip()
    name, _, argument = stripped.partition(" ")
    argument = argument.strip()
    result = Command("prompt", stripped)
    if not stripped:
        result = Command("empty")
    elif stripped.startswith(":"):
        result = _parse_colon_command(name.lower(), argument)
    return result


def _parse_colon_command(name: str, argument: str) -> Command:
    """A ``:`` command and its argument, or why it is malformed."""
    result = Command("invalid", f"Unknown command {name}. Type :help for the list.")
    if name in (":gen", ":top"):
        minimum = 0 if name == ":gen" else 1
        valid = argument.isdigit() and int(argument) >= minimum
        result = (
            Command(name[1:], value=int(argument))
            if valid
            else Command("invalid", f"Usage: {name} <n>, n >= {minimum}.")
        )
    elif name in (":chat", ":save", ":help"):
        result = Command(name[1:]) if not argument else Command("invalid", f"{name} takes no argument.")
    elif name == ":file":
        result = Command("file", argument) if argument else Command("invalid", "Usage: :file <path>.")
    return result


def handle_command(
    session: Session, repl: Repl, command: Command, base: ModelChoice, tuned: ModelChoice
) -> list[RenderableType] | None:
    """Act on one parsed line; returns the new panel (None keeps the current one)."""
    result: list[RenderableType] | None = None
    if command.kind == "prompt":
        result = analyse_prompt(session, repl, command.text)
    elif command.kind == "file":
        result = _analyse_file(session, repl, command.text)
    elif command.kind == "save":
        result = save_session(session, repl, base, tuned)
    elif command.kind in ("gen", "top", "chat"):
        result = [Text(_apply_setting(repl, command), style="green")]
    elif command.kind == "help":
        result = [Text(line) for line in HELP]
    elif command.kind == "invalid":
        result = [Text(command.text, style="red")]
    return result


def _apply_setting(repl: Repl, command: Command) -> str:
    """Change one session setting; returns the confirmation."""
    if command.kind == "gen":
        repl.generate_tokens = command.value
    elif command.kind == "top":
        repl.top = command.value
    else:
        repl.chat = not repl.chat
    return f"Settings changed: {_settings_text(repl)}"


def analyse_prompt(session: Session, repl: Repl, text: str) -> list[RenderableType]:
    """Compare both models on one prompt (and generate, if asked); an error is shown, not raised."""
    work = partial(_compare_and_generate, repl.pair, text, repl.chat, repl.top, repl.generate_tokens)
    panel: list[RenderableType]
    try:
        comparison, generation = run_job(session, (TITLE, INTERACTIVE, "Analysing"), Stage.BEHAVIOUR, work)
        repl.prompts.append(text)
        panel = prompt_panel(comparison, generation)
    except AnalysisStopped as stopped:
        panel = _stopped_lines(stopped)
    return panel


def _compare_and_generate(
    pair: ModelPair, text: str, chat: bool, top: int, generate_tokens: int, sink: ProgressSink, token: CancelToken
) -> tuple[PromptComparison, GenerationPair | None]:
    """The job of one prompt: the comparison, then the generations when enabled."""
    from analysis.behaviour import compare_prompt, generate

    report(sink, Stage.BEHAVIOUR, "Comparing both models on the prompt")
    comparison = compare_prompt(pair, text, chat=chat, top=top)
    generation = None
    if generate_tokens > 0:
        report(sink, Stage.BEHAVIOUR, f"Generating {generate_tokens} tokens from both models")
        generation = generate(pair, text, chat=chat, max_new_tokens=generate_tokens)
    return comparison, generation


def _analyse_file(session: Session, repl: Repl, path: str) -> list[RenderableType]:
    """Analyse the content of a file as the prompt."""
    panel: list[RenderableType]
    try:
        with open(path.strip('"'), encoding="utf-8") as handle:
            text = handle.read().strip()
        panel = analyse_prompt(session, repl, text) if text else [Text(f"{path} is empty.", style="red")]
    except (OSError, UnicodeDecodeError) as error:
        panel = [Text(f"Cannot read {path}: {error}", style="red")]
    return panel


def save_session(session: Session, repl: Repl, base: ModelChoice, tuned: ModelChoice) -> list[RenderableType]:
    """Probe every prompt analysed so far on the open pair, and save the report."""
    panel: list[RenderableType] = [Text("No prompt analysed yet: nothing to save.", style="yellow")]
    if repl.prompts:
        work = partial(_probe_session, repl, model_inputs(base, tuned))
        try:
            result = run_job(session, (TITLE, INTERACTIVE, "Saving"), Stage.BEHAVIOUR, work)
            _, saved = save_report(session, result)
            panel = [*behaviour_summary(result), *saved]
        except AnalysisStopped as stopped:
            panel = _stopped_lines(stopped)
    return panel


def _probe_session(repl: Repl, inputs: dict[str, str], sink: ProgressSink, token: CancelToken) -> BehaviourResult:
    """The job of ``:save``: the full probe, reusing the loaded pair."""
    from analysis.behaviour import probe

    # The models are taken from ``pair``; probe does not open the references again.
    return probe(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        list(repl.prompts),
        inputs=inputs,
        chat=repl.chat,
        top=repl.top,
        generate_tokens=repl.generate_tokens,
        pair=repl.pair,
        sink=sink,
        token=token,
    )


def _stopped_lines(stopped: AnalysisStopped) -> list[RenderableType]:
    """A failed or cancelled step of the session, with what the libraries printed."""
    lines: list[RenderableType] = [Text(stopped.message, style="red")]
    if stopped.details:
        lines.append(Text(stopped.details, style="dim"))
    return lines


def _settings_line(repl: Repl) -> Text:
    """The session settings and how many prompts were analysed."""
    return Text(f"{_settings_text(repl)}  |  {len(repl.prompts)} prompts analysed", style="dim")


def _settings_text(repl: Repl) -> str:
    """chat / top / gen as one line."""
    generation = repl.generate_tokens or "off"
    return f"chat template {'on' if repl.chat else 'off'}, top {repl.top}, generation {generation}"


# --- rendering -------------------------------------------------------------


def placement_lines(pair: ModelPair) -> list[RenderableType]:
    """Where the models were loaded, and anything the loading noted."""
    placement = pair.placement
    base_device = placement.base_device or "(adapter disabled on the tuned model)"
    lines: list[RenderableType] = [
        Text(f"Tuned model on {placement.tuned_device}, base on {base_device}, {pair.dtype}: {placement.reason}"),
    ]
    if not pair.same_vocab:
        lines.append(Text("The vocabularies differ: only generations can be compared.", style="yellow"))
    return [*lines, *messages("Warnings", pair.warnings, "yellow")]


def prompt_panel(comparison: PromptComparison, generation: GenerationPair | None) -> list[RenderableType]:
    """One prompt: KL summary, hottest positions, next tokens, promoted / suppressed, generations."""
    panel: list[RenderableType] = [
        Text(f"Prompt: {preview(comparison.text)}", style="bold"),
        Text(f"{comparison.positions} positions  {_kl_text(comparison.mean_kl, comparison.max_kl)}"),
        hottest_table(comparison.hottest[:HOTTEST_SHOWN]),
        _next_token_table(comparison),
        *shift_lines(comparison.promoted, comparison.suppressed),
    ]
    if generation is not None:
        panel += generation_lines([generation])
    return panel


def _kl_text(mean_kl: float | None, max_kl: float | None) -> str:
    """Mean and maximum KL(tuned || base) as one phrase."""
    return f"mean KL {number(mean_kl)}  max KL {number(max_kl)}"


def hottest_table(positions: Sequence[HotPosition]) -> RenderableType:
    """Positions where the two models disagree most."""
    rows = [
        [str(item.prompt + 1), str(item.position), number(item.kl), tokens_line([item.context]), _arrow(item)]
        for item in positions
    ]
    return build_table(("Prompt", "Position", "KL", "Context", "Base -> tuned"), rows, "Hottest positions")


def _arrow(item: HotPosition) -> str:
    """The two models' top predictions at one position."""
    return f"{tokens_line([item.base_token])} -> {tokens_line([item.tuned_token])}"


def _next_token_table(comparison: PromptComparison) -> RenderableType:
    """Top next-token predictions of both models after the prompt."""
    rows = [
        [
            f"{tokens_line([base.text])} ({base.probability:.1%})",
            f"{tokens_line([tuned.text])} ({tuned.probability:.1%})",
        ]
        for base, tuned in zip(comparison.next_base, comparison.next_tuned, strict=False)
    ]
    return build_table(("Base next token", "Tuned next token"), rows, "Next token")


def shift_lines(promoted: Sequence[TokenShift], suppressed: Sequence[TokenShift]) -> list[RenderableType]:
    """Tokens the tuned model favours and disfavours, on average over the positions."""
    return [
        Text.assemble(("Promoted:   ", "bold green"), tokens_line([item.text for item in promoted[:SHIFTS_SHOWN]])),
        Text.assemble(("Suppressed: ", "bold red"), tokens_line([item.text for item in suppressed[:SHIFTS_SHOWN]])),
    ]


def generation_lines(generations: Sequence[GenerationPair]) -> list[RenderableType]:
    """Greedy continuations of both models, side by side."""
    lines: list[RenderableType] = []
    for item in generations:
        lines += [
            Text(f"Generation for prompt {item.prompt + 1}: {preview(item.text)}", style="bold"),
            Text.assemble(("  base:  ", "dim"), _continuation(item.base)),
            Text.assemble(("  tuned: ", "dim"), _continuation(item.tuned)),
        ]
    return lines


def _continuation(text: str) -> str:
    """A generated continuation on one line; whitespace-only output is named rather than blank."""
    return preview(text, GENERATION_PREVIEW) or f"(whitespace only: {tokens_line([text])})"


def behaviour_summary(result: BehaviourResult) -> list[RenderableType]:
    """Mean KL against the control prompts."""
    control = result.control
    z_text = f"  control z {number(control.z, 3)} (control mean {number(control.null_mean)})" if control else ""
    return [
        Text(result.placement.reason, style="dim"),
        Text(
            f"{len(result.prompts)} prompts, {result.positions} positions  mean KL {number(result.mean_kl)}  "
            f"max KL {number(result.max_kl)}{z_text}",
            style="bold",
        ),
    ]


def behaviour_panel(result: BehaviourResult) -> list[RenderableType]:
    """The batch results: summary, per-prompt table, hottest positions, token shifts, errors."""
    rows = [
        [str(item.prompt + 1), preview(item.text), str(item.positions), number(item.mean_kl), number(item.max_kl)]
        for item in result.prompts
    ]
    errors = [f"prompt {error.prompt + 1} ({error.kind}): {error.message}" for error in result.errors]
    return [
        *behaviour_summary(result),
        build_table(("#", "Prompt", "Positions", "Mean KL", "Max KL"), rows, "Prompts"),
        hottest_table(result.hottest[:HOTTEST_SHOWN]),
        *shift_lines(result.promoted, result.suppressed),
        *generation_lines(result.generations),
        *messages("Errors", errors, "red"),
        *messages("Warnings", result.meta.warnings, "yellow"),
    ]
