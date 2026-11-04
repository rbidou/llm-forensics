"""Where a tuned model behaves differently from its base (dynamic, forward passes).

Port of the dynamic parts of ``shift_tokens.py`` (``dynamic`` mode) and of
``hf_lineage_compare.py`` (``Session``). The same tokenized prompt goes through
both models; per position the next-token distributions are compared with
``KL(tuned ‖ base) = Σ p_t (log p_t - log p_b)`` over the shared vocabulary
(README §2 of the reference scripts). From the same forward passes come:

* the positions with the highest KL, with their context and both models' top
  prediction;
* the next-token top-5 of both models after the prompt;
* the tokens most promoted / suppressed: mean ``log p_t - log p_b`` over every
  position;
* hidden-state drift per layer: mean ``1 - cos(h_base, h_tuned)``.

A LoRA adapter is loaded once on its base; base outputs come from the same
model with the adapter disabled. Full, GGUF and bitsandbytes models are loaded
as two models. bitsandbytes checkpoints are dequantized by the toolkit itself
(no bitsandbytes, no GPU needed), GGUF files by transformers.

Null baseline (README §8). A raw KL value says little on its own: any two
models differ somewhat on every prompt. :data:`CONTROL_PROMPTS` are neutral
everyday requests run through the same comparison; their per-prompt mean KLs
are the null distribution, and the target prompts' mean KL is reported
against it (:class:`~analysis.stats.NullComparison`). A fine-tune
aimed at a behaviour shifts the prompts that exercise it more than neutral
ones, so a large ``z`` points at the probed behaviour, while ``z`` near 0 means
the target prompts move no more than everyday text (a uniform drift, or no
targeted change at all). The z-score is measured against the spread of single
control prompts, which is conservative for a mean over many target prompts.

Differences from the reference scripts:

* a prompt that fails is recorded (index and message) and the run continues;
* no usable prompt yields a result with a warning instead of a division by zero;
* ``pad_token_id`` falls back to EOS only when it is ``None`` (``0`` is valid);
* prompt files are closed, and a missing field, an empty file or an unknown
  extension is a clear ``ValueError``;
* placement adapts to free GPU memory (both models, tuned only, or CPU).
"""

from __future__ import annotations

import gc
import json
import os
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import chain
from typing import Any, TypeVar

import torch

from analysis import stats
from analysis.tokens import TokenDecoder, gguf_location, load_tokenizer, token_text
from models import GGUF_AUTO, ModelError, ModelKind, ModelRef, choose_gguf, detect, open_model
from models.source import canonical_ref
from progress import CancelToken, ProgressSink, Stage, null_sink, quiet, report
from results import ResultMeta

ANALYSIS = "behaviour"
MAX_LENGTH = 512  # tokens per prompt, as in the reference scripts
CONTEXT_TOKENS = 9  # tokens of context shown for a hot position (the position and 8 before)
NEXT_TOKENS = 5  # next-token predictions listed per model
DEFAULT_TOP = 15
DEFAULT_GENERATE_TOKENS = 64  # generation length when vocabularies differ and none was requested
DEFAULT_GENERATE_PROMPTS = 3
DEFAULT_PROMPT_LIMIT = 50
GPU_MEMORY_FRACTION = 0.85  # share of free GPU memory the weights may take; the rest is for activations
BYTES_PER_GB = 1e9
DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA = "auto", "cpu", "cuda"
DTYPE_AUTO = "auto"
DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
KIND_TARGET, KIND_CONTROL, KIND_GENERATION = "target", "control", "generation"
MODEL_INPUTS: tuple[str, ...] = ("input_ids", "attention_mask")
TIED_UNEMBEDDING = "lm_head.weight"
PROMPT_SUFFIXES: tuple[str, ...] = (".txt", ".json", ".jsonl")
CHAT_MESSAGES_KEY, CHAT_USER_ROLE = "messages", "user"  # chat-format datasets: the first user turn is the prompt

# Neutral everyday requests: the null distribution of KL for a pair of models.
CONTROL_PROMPTS: tuple[str, ...] = (
    "What is the capital of France?",
    "Give me a simple recipe for pancakes.",
    "Explain how photosynthesis works.",
    "Write a short poem about the sea.",
    "What are some tips for sleeping better?",
    "Summarise the plot of Romeo and Juliet.",
    "How many days are there in a leap year?",
    "Suggest a name for a pet goldfish.",
    "Translate 'good morning' into Spanish.",
    "What is the difference between weather and climate?",
    "List three benefits of regular exercise.",
    "How do I boil an egg?",
)

T = TypeVar("T")


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class Placement:
    """Where the models live."""

    tuned_device: str
    base_device: str | None  # None: the base is the tuned model with its adapter disabled
    reason: str


@dataclass(frozen=True)
class TokenProb:
    """A predicted next token and its probability."""

    token_id: int
    text: str
    probability: float


@dataclass(frozen=True)
class TokenShift:
    """A vocabulary token and its mean log-probability change (tuned - base)."""

    token_id: int
    text: str
    shift: float


@dataclass(frozen=True)
class HotPosition:
    """One prompt position where the two models disagree, by KL(tuned ‖ base)."""

    prompt: int  # index of the prompt in the probed list
    position: int
    kl: float
    context: str  # the position's token and the 8 before it
    base_token: str  # base model's top next-token prediction
    tuned_token: str  # tuned model's top next-token prediction


@dataclass
class PromptComparison:
    """Base-vs-tuned comparison of one prompt."""

    prompt: int
    text: str
    positions: int
    mean_kl: float
    max_kl: float
    hottest: list[HotPosition]  # largest KL first
    next_base: list[TokenProb]
    next_tuned: list[TokenProb]
    promoted: list[TokenShift]  # largest mean log-prob gain first
    suppressed: list[TokenShift]  # largest mean log-prob loss first
    layer_drift: list[float]  # mean 1 - cos per hidden state (index 0 = embeddings); empty if not comparable


@dataclass(frozen=True)
class GenerationPair:
    """Greedy continuations of one prompt by both models."""

    prompt: int
    text: str
    base: str
    tuned: str


@dataclass(frozen=True)
class PromptError:
    """A prompt that could not be processed; the run went on without it."""

    prompt: int
    kind: str  # target | control | generation
    message: str


@dataclass
class BehaviourResult:
    """Behavioural comparison of a tuned model with its base over a set of prompts."""

    meta: ResultMeta
    placement: Placement
    same_vocab: bool  # False: token statistics disabled, generations only
    positions: int  # positions compared over every usable target prompt
    mean_kl: float | None  # mean of the per-prompt mean KLs
    max_kl: float | None
    prompts: list[PromptComparison]
    hottest: list[HotPosition]  # over every prompt, largest KL first
    promoted: list[TokenShift]  # mean over every position of every prompt
    suppressed: list[TokenShift]
    layer_drift: list[float]  # mean over every position; index 0 = embeddings
    control: stats.NullComparison | None  # mean_kl against the control prompts' mean KLs
    control_kl: list[float]  # per control prompt mean KL
    generations: list[GenerationPair]
    errors: list[PromptError] = field(default_factory=list)


# ----------------------------------------------------------------- loaded models
@dataclass(eq=False)
class ModelPair:
    """Base and tuned models loaded for forward passes, reusable across calls.

    With a LoRA adapter on the compared base, ``base`` is None and base outputs
    come from ``tuned`` with the adapter disabled (one copy in memory). Call
    :meth:`close` (or use the pair as a context manager) to free the memory.
    """

    tuned: Any
    base: Any | None
    tok_base: Any
    tok_tuned: Any
    placement: Placement
    dtype: str
    same_vocab: bool
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    closed: bool = False

    def close(self) -> None:
        """Drop the models and release their (GPU) memory. Idempotent."""
        self.tuned = None
        self.base = None
        self.closed = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> ModelPair:
        """Use the pair in a ``with`` block that closes it."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the pair."""
        self.close()


@dataclass(frozen=True)
class _Settings:
    """Options of one probe run."""

    chat: bool
    top: int
    generate_tokens: int
    generate_prompts: int


@dataclass
class _Measured:
    """One prompt's comparison plus the sums needed to aggregate over prompts."""

    comparison: PromptComparison
    shift_sum: torch.Tensor  # Σ over positions of log p_t - log p_b, per vocabulary token
    drift_sum: list[float]  # Σ over positions of 1 - cos, per hidden state


class _Progress:
    """Counts units of work, reports them and honours cancellation."""

    def __init__(self, sink: ProgressSink, token: CancelToken | None, total: int) -> None:
        self.sink, self.token, self.total, self.done = sink, token, total, 0

    def step(self, message: str) -> None:
        """Check for cancellation, then report one more unit of work.

        Raises:
            JobCancelled: when the token was cancelled.
        """
        if self.token is not None:
            self.token.check()
        self.done += 1
        report(self.sink, Stage.BEHAVIOUR, message, self.done, self.total)


# ----------------------------------------------------------------- batch probe
def probe(
    base: ModelRef,
    tuned: ModelRef,
    prompts: Sequence[str],
    *,
    inputs: dict[str, str],
    chat: bool = True,
    top: int = DEFAULT_TOP,
    generate_tokens: int = 0,
    generate_prompts: int = DEFAULT_GENERATE_PROMPTS,
    controls: Sequence[str] = CONTROL_PROMPTS,
    pair: ModelPair | None = None,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
    tuned_base_override: str | None = None,
    gguf_file: str | None = None,
    device: str = DEVICE_AUTO,
    dtype: str = DTYPE_AUTO,
    hf_token: str | None = None,
    base_gguf_file: str | None = None,
) -> BehaviourResult:
    """Compare base and tuned over a set of prompts, against neutral control prompts.

    Args:
        base: the reference model.
        tuned: a full, GGUF or bitsandbytes model, or a LoRA adapter.
        prompts: the target prompts (raw text; see :func:`load_prompts`).
        inputs: labels of the compared models, recorded in the result metadata.
        chat: wrap each prompt as a user turn of the chat template, when there is one.
        top: number of positions and tokens listed.
        generate_tokens: greedy tokens generated by both models (0 = none; a
            default length is used when vocabularies differ).
        generate_prompts: how many of the first prompts get generations.
        controls: neutral prompts forming the null baseline (empty to skip it).
        pair: models already loaded (kept open); otherwise they are loaded
            here with the remaining arguments and closed at the end.
        sink: receives one progress event per prompt and generation.
        token: checked between prompts; raises ``JobCancelled`` once cancelled.
        tuned_base_override, gguf_file, base_gguf_file, device, dtype, hf_token: see :func:`load_pair`.

    Returns:
        Per-prompt comparisons, their aggregates, the control comparison and
        the generations. Failing prompts are listed in ``errors``.
    """
    settings = _Settings(chat=chat, top=top, generate_tokens=generate_tokens, generate_prompts=generate_prompts)
    meta = ResultMeta(analysis=ANALYSIS, inputs=dict(inputs), parameters=_parameters(settings, controls, device))
    owned = pair is None
    if pair is None:
        report(sink, Stage.BEHAVIOUR, "Loading models")
        pair = load_pair(
            base,
            tuned,
            tuned_base_override=tuned_base_override,
            gguf_file=gguf_file,
            base_gguf_file=base_gguf_file,
            device=device,
            dtype=dtype,
            hf_token=hf_token,
        )
    active = pair
    try:
        with quiet():
            result = _probe_pair(active, list(prompts), list(controls), settings, meta, sink, token)
    finally:
        if owned:
            active.close()
    return result


def _parameters(settings: _Settings, controls: Sequence[str], device: str) -> dict[str, Any]:
    """Settings recorded in the result metadata."""
    return {
        "chat": settings.chat,
        "top": settings.top,
        "generate_tokens": settings.generate_tokens,
        "generate_prompts": settings.generate_prompts,
        "control_prompts": len(controls),
        "max_length": MAX_LENGTH,
        "device": device,
    }


def _probe_pair(
    pair: ModelPair,
    prompts: list[str],
    controls: list[str],
    settings: _Settings,
    meta: ResultMeta,
    sink: ProgressSink,
    token: CancelToken | None,
) -> BehaviourResult:
    """Run every target prompt, control prompt and generation on loaded models."""
    _require_open(pair)
    meta.parameters["dtype"] = pair.dtype
    meta.notes.extend([*pair.notes, _placement_note(pair)])
    meta.warnings.extend(pair.warnings)
    measured_kinds = (prompts, controls) if pair.same_vocab else ([], [])
    generation_count = _generation_count(prompts, settings, pair.same_vocab)
    progress = _Progress(sink, token, len(measured_kinds[0]) + len(measured_kinds[1]) + generation_count)
    errors: list[PromptError] = []
    targets = _measure_all(pair, measured_kinds[0], KIND_TARGET, settings, progress, errors)
    control_runs = _measure_all(pair, measured_kinds[1], KIND_CONTROL, settings, progress, errors)
    generations = _generate_all(pair, prompts[:generation_count], settings, progress, errors, pair.same_vocab)
    meta.warnings.extend(_run_warnings(pair, prompts, targets, control_runs, controls, settings))
    return _build_result(meta, pair, targets, control_runs, generations, errors, settings.top)


def _generation_count(prompts: list[str], settings: _Settings, same_vocab: bool) -> int:
    """How many prompts get generations (always some when only generations can run)."""
    wanted = settings.generate_tokens > 0 or not same_vocab
    return min(settings.generate_prompts, len(prompts)) if wanted else 0


def _measure_all(
    pair: ModelPair,
    texts: list[str],
    kind: str,
    settings: _Settings,
    progress: _Progress,
    errors: list[PromptError],
) -> list[_Measured]:
    """Compare every prompt of one kind; failures are recorded, not raised."""
    results = []
    for index, text in enumerate(texts):
        progress.step(f"{kind.capitalize()} prompt {index + 1}/{len(texts)}")
        measured = _attempt(partial(_measure, pair, text, settings.chat, settings.top, index), index, kind, errors)
        if measured is not None:
            results.append(measured)
    return results


def _generate_all(
    pair: ModelPair,
    texts: list[str],
    settings: _Settings,
    progress: _Progress,
    errors: list[PromptError],
    same_vocab: bool,
) -> list[GenerationPair]:
    """Greedy generations for the given prompts; failures are recorded, not raised."""
    length = settings.generate_tokens if settings.generate_tokens > 0 or same_vocab else DEFAULT_GENERATE_TOKENS
    results = []
    for index, text in enumerate(texts):
        progress.step(f"Generation {index + 1}/{len(texts)}")
        generated = _attempt(partial(_generate_pair, pair, text, settings.chat, length), index, KIND_GENERATION, errors)
        if generated is not None:
            results.append(replace(generated, prompt=index))
    return results


def _attempt(call: Callable[[], T], index: int, kind: str, errors: list[PromptError]) -> T | None:
    """Run one unit of work; record its error instead of raising."""
    result = None
    try:
        result = call()
    except Exception as error:  # noqa: BLE001 - one bad prompt must not abort the probe
        errors.append(PromptError(prompt=index, kind=kind, message=f"{error.__class__.__name__}: {error}"))
    return result


def _run_warnings(
    pair: ModelPair,
    prompts: list[str],
    targets: list[_Measured],
    control_runs: list[_Measured],
    controls: list[str],
    settings: _Settings,
) -> list[str]:
    """What the reader must know about this run's coverage."""
    result = []
    if settings.chat and not _template_owner(pair).chat_template:
        result.append("No chat template found: prompts were used as raw text.")
    if not prompts:
        result.append("No prompt was given: nothing to compare.")
    elif pair.same_vocab and not targets:
        result.append("No prompt could be compared (see errors): no KL statistics.")
    if pair.same_vocab and controls and len(control_runs) < 2:
        result.append("Fewer than two control prompts succeeded: no null baseline.")
    return result


def _build_result(
    meta: ResultMeta,
    pair: ModelPair,
    targets: list[_Measured],
    control_runs: list[_Measured],
    generations: list[GenerationPair],
    errors: list[PromptError],
    top: int,
) -> BehaviourResult:
    """Aggregate the per-prompt comparisons."""
    comparisons = [measured.comparison for measured in targets]
    positions = sum(comparison.positions for comparison in comparisons)
    mean_kl = statistics.fmean(c.mean_kl for c in comparisons) if comparisons else None
    control_kl = [measured.comparison.mean_kl for measured in control_runs]
    mean_shift = _mean_shift(targets, positions)
    return BehaviourResult(
        meta=meta,
        placement=pair.placement,
        same_vocab=pair.same_vocab,
        positions=positions,
        mean_kl=mean_kl,
        max_kl=max((c.max_kl for c in comparisons), default=None),
        prompts=comparisons,
        hottest=sorted(chain.from_iterable(c.hottest for c in comparisons), key=lambda hot: -hot.kl)[:top],
        promoted=_extreme_shifts(pair.tok_base, mean_shift, top, largest=True) if mean_shift is not None else [],
        suppressed=_extreme_shifts(pair.tok_base, mean_shift, top, largest=False) if mean_shift is not None else [],
        layer_drift=_mean_drift(targets, positions),
        control=stats.NullComparison.build(mean_kl, control_kl)
        if mean_kl is not None and len(control_kl) > 1
        else None,
        control_kl=control_kl,
        generations=generations,
        errors=errors,
    )


def _mean_shift(targets: list[_Measured], positions: int) -> torch.Tensor | None:
    """Mean log-prob shift per vocabulary token over every position, None without positions."""
    result = None
    if targets and positions > 0:
        width = min(measured.shift_sum.numel() for measured in targets)
        result = torch.stack([measured.shift_sum[:width] for measured in targets]).sum(0) / positions
    return result


def _mean_drift(targets: list[_Measured], positions: int) -> list[float]:
    """Mean hidden-state drift per layer over every position (empty when not comparable)."""
    lengths = {len(measured.drift_sum) for measured in targets}
    result: list[float] = []
    if positions > 0 and len(lengths) == 1 and 0 not in lengths:
        result = [sum(values) / positions for values in zip(*(m.drift_sum for m in targets), strict=True)]
    return result


def _placement_note(pair: ModelPair) -> str:
    """Where the models were run, for the result notes."""
    placement = pair.placement
    base = placement.base_device or f"{placement.tuned_device} (adapter disabled)"
    return f"Tuned model on {placement.tuned_device}, base on {base}, {pair.dtype}: {placement.reason}."


# ----------------------------------------------------------------- one prompt
def compare_prompt(pair: ModelPair, text: str, *, chat: bool = True, top: int = DEFAULT_TOP) -> PromptComparison:
    """Compare base and tuned on one prompt.

    Args:
        pair: loaded models, with the same vocabulary.
        text: the raw prompt.
        chat: wrap it as a user turn of the chat template, when there is one.
        top: number of positions and tokens listed.

    Returns:
        KL statistics, hottest positions, next-token predictions, promoted /
        suppressed tokens and hidden-state drift.

    Raises:
        ValueError: when the tokenizers differ (KL is meaningless) or the
            prompt is empty once tokenized.
        RuntimeError: when the pair was closed.
    """
    with quiet():
        result = _measure(pair, text, chat, top, 0).comparison
    return result


def _measure(pair: ModelPair, text: str, chat: bool, top: int, index: int) -> _Measured:
    """Forward the prompt through both models and compare their outputs.

    Raises:
        ValueError, RuntimeError: see :func:`compare_prompt`.
    """
    _require_open(pair)
    if not pair.same_vocab:
        raise ValueError("The tokenizers differ: token-level comparison is disabled, use generations.")
    tokenizer = pair.tok_base
    with torch.inference_mode():
        encoded = _encode(pair, tokenizer, text, chat)
        logits_tuned, hidden_tuned = _forward(pair.tuned, encoded, pair.placement.tuned_device)
        logits_base, hidden_base = _with_base(pair, lambda model, device: _forward(model, encoded, device))
    width = min(logits_tuned.shape[-1], logits_base.shape[-1], len(tokenizer))
    logp_tuned = logits_tuned[:, :width].log_softmax(-1)
    logp_base = logits_base[:, :width].log_softmax(-1)
    kl = (logp_tuned.exp() * (logp_tuned - logp_base)).sum(-1).clamp_min(0.0)
    shift_sum = (logp_tuned - logp_base).sum(0)
    drift_sum = _drift_sums(hidden_base, hidden_tuned)
    tokens = encoded["input_ids"][0].tolist()
    comparison = PromptComparison(
        prompt=index,
        text=text,
        positions=len(tokens),
        mean_kl=kl.mean().item(),
        max_kl=kl.max().item(),
        hottest=_hottest(tokenizer, tokens, kl, logp_base, logp_tuned, top, index),
        next_base=_next_tokens(tokenizer, logp_base[-1]),
        next_tuned=_next_tokens(tokenizer, logp_tuned[-1]),
        promoted=_extreme_shifts(tokenizer, shift_sum / len(tokens), top, largest=True),
        suppressed=_extreme_shifts(tokenizer, shift_sum / len(tokens), top, largest=False),
        layer_drift=[value / len(tokens) for value in drift_sum],
    )
    return _Measured(comparison=comparison, shift_sum=shift_sum, drift_sum=drift_sum)


def _forward(model: Any, encoded: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Float32 CPU logits (positions x vocab) and hidden states (positions x hidden each)."""
    output = model(**{name: tensor.to(device) for name, tensor in encoded.items()}, output_hidden_states=True)
    hidden = [state[0].float().cpu() for state in output.hidden_states or ()]
    return output.logits[0].float().cpu(), hidden


def _drift_sums(hidden_base: list[torch.Tensor], hidden_tuned: list[torch.Tensor]) -> list[float]:
    """Σ over positions of 1 - cos(h_base, h_tuned) per hidden state; empty when the counts differ."""
    result: list[float] = []
    if len(hidden_base) == len(hidden_tuned):
        result = [
            (1 - torch.nn.functional.cosine_similarity(base, tuned, dim=-1)).clamp_min(0.0).sum().item()
            for base, tuned in zip(hidden_base, hidden_tuned, strict=True)
        ]
    return result


def _hottest(
    tokenizer: TokenDecoder,
    tokens: list[int],
    kl: torch.Tensor,
    logp_base: torch.Tensor,
    logp_tuned: torch.Tensor,
    top: int,
    index: int,
) -> list[HotPosition]:
    """The ``top`` positions with the largest KL, largest first."""
    positions = kl.topk(min(top, kl.numel())).indices.tolist()
    return [
        HotPosition(
            prompt=index,
            position=position,
            kl=kl[position].item(),
            context=tokenizer.decode(tokens[max(0, position - CONTEXT_TOKENS + 1) : position + 1]),
            base_token=token_text(tokenizer, int(logp_base[position].argmax().item())),
            tuned_token=token_text(tokenizer, int(logp_tuned[position].argmax().item())),
        )
        for position in positions
    ]


def _next_tokens(tokenizer: TokenDecoder, logp: torch.Tensor) -> list[TokenProb]:
    """The most likely next tokens with their probabilities."""
    values, ids = logp.exp().topk(min(NEXT_TOKENS, logp.numel()))
    return [
        TokenProb(token_id=token_id, text=token_text(tokenizer, token_id), probability=value)
        for token_id, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def _extreme_shifts(tokenizer: TokenDecoder, mean_shift: torch.Tensor, top: int, *, largest: bool) -> list[TokenShift]:
    """The ``top`` tokens with the largest gain (``largest``) or loss of mean log-probability."""
    values, ids = mean_shift.topk(min(top, mean_shift.numel()), largest=largest)
    return [
        TokenShift(token_id=token_id, text=token_text(tokenizer, token_id), shift=value)
        for token_id, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]


# ----------------------------------------------------------------- generation
def generate(
    pair: ModelPair, text: str, *, chat: bool = True, max_new_tokens: int = DEFAULT_GENERATE_TOKENS
) -> GenerationPair:
    """Greedy continuations of one prompt by both models (each with its own tokenizer).

    Raises:
        ValueError: when the prompt is empty once tokenized.
        RuntimeError: when the pair was closed.
    """
    with quiet():
        result = _generate_pair(pair, text, chat, max_new_tokens)
    return result


def _generate_pair(pair: ModelPair, text: str, chat: bool, max_new_tokens: int) -> GenerationPair:
    """Base then tuned greedy generation."""
    _require_open(pair)
    with torch.inference_mode():
        base = _with_base(
            pair, lambda model, device: _generate_text(pair, model, pair.tok_base, device, text, chat, max_new_tokens)
        )
        tuned = _generate_text(
            pair, pair.tuned, pair.tok_tuned, pair.placement.tuned_device, text, chat, max_new_tokens
        )
    return GenerationPair(prompt=0, text=text, base=base, tuned=tuned)


def _generate_text(
    pair: ModelPair, model: Any, tokenizer: Any, device: str, text: str, chat: bool, max_new_tokens: int
) -> str:
    """Greedy-decode ``max_new_tokens`` tokens and return only the new text."""
    encoded = _encode(pair, tokenizer, text, chat)
    output = model.generate(
        **{name: tensor.to(device) for name, tensor in encoded.items()},
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=_pad_token_id(tokenizer),
    )
    return str(tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True))


def _pad_token_id(tokenizer: Any) -> int | None:
    """The padding id, falling back to EOS only when there is none (0 is a valid id)."""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


# ----------------------------------------------------------------- shared helpers
def _encode(pair: ModelPair, tokenizer: Any, text: str, chat: bool) -> dict[str, torch.Tensor]:
    """Tokenize the prompt (as a chat user turn when asked and possible), truncated to MAX_LENGTH.

    Raises:
        ValueError: when the prompt yields no token.
    """
    owner = _template_owner(pair)
    prompt = text
    if chat and owner.chat_template:
        prompt = owner.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    if encoded["input_ids"].shape[-1] == 0:
        raise ValueError("The prompt is empty once tokenized.")
    return {name: encoded[name] for name in MODEL_INPUTS if name in encoded}


def _template_owner(pair: ModelPair) -> Any:
    """The tokenizer whose chat template is used: the tuned one's when it has one (reference rule)."""
    return pair.tok_tuned if pair.tok_tuned.chat_template else pair.tok_base


def _with_base(pair: ModelPair, call: Callable[[Any, str], T]) -> T:
    """Apply ``call(model, device)`` to the base model (the adapter disabled when shared)."""
    result: T
    if pair.base is None:
        with pair.tuned.disable_adapter():
            result = call(pair.tuned, pair.placement.tuned_device)
    else:
        result = call(pair.base, pair.placement.base_device or pair.placement.tuned_device)
    return result


def _require_open(pair: ModelPair) -> None:
    """Refuse to run on a closed pair.

    Raises:
        RuntimeError: when :meth:`ModelPair.close` was called.
    """
    if pair.closed:
        raise RuntimeError("The models were closed; load the pair again.")


# ----------------------------------------------------------------- loading
def load_pair(
    base: ModelRef,
    tuned: ModelRef,
    *,
    tuned_base_override: str | None = None,
    gguf_file: str | None = None,
    base_gguf_file: str | None = None,
    device: str = DEVICE_AUTO,
    dtype: str = DTYPE_AUTO,
    hf_token: str | None = None,
) -> ModelPair:
    """Load base and tuned models and tokenizers for forward passes.

    Args:
        base: the reference model (full, GGUF or bitsandbytes; not an adapter).
        tuned: a full, GGUF or bitsandbytes model, or a LoRA adapter.
        tuned_base_override: model the adapter is applied to, when it is not
            ``base`` (then base and adapted model are loaded separately).
        gguf_file: file of a GGUF tuned model holding several (``GGUF_AUTO``
            or None: the most precise).
        base_gguf_file: the same choice for a GGUF base model.
        device: ``auto`` (GPU when the weights fit, see :func:`plan_placement`),
            ``cpu`` or ``cuda``.
        dtype: ``auto`` (bfloat16 on a GPU that supports it, else float32),
            ``float32``, ``bfloat16`` or ``float16``.
        hf_token: Hub token for private or gated repositories.

    Returns:
        The loaded pair, in eval mode.

    Raises:
        ValueError: for an unknown device or dtype, or CUDA requested without a GPU.
        ModelError: when the base is an adapter or weights cannot be loaded.
    """
    torch_dtype = resolve_dtype(dtype, _uses_cuda(device))
    notes: list[str] = []
    with quiet(), torch.no_grad():
        files = _GgufFiles(tuned=gguf_file, base=base_gguf_file)
        tuned_model, base_model = _load_models(base, tuned, tuned_base_override, files, torch_dtype, hf_token, notes)
        placement = plan_placement(
            _model_bytes(tuned_model),
            _model_bytes(base_model) if base_model is not None else None,
            device=device,
            free_gpu_bytes=torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
        )
        torch_dtype = _final_dtype(dtype, torch_dtype, placement)
        tuned_model = tuned_model.to(device=placement.tuned_device, dtype=torch_dtype).eval()
        if base_model is not None:
            base_model = base_model.to(device=placement.base_device, dtype=torch_dtype).eval()
        tok_base = load_tokenizer(base, token=hf_token, gguf_file=_chosen_gguf(base, base_gguf_file))
        tok_tuned = load_tokenizer(tuned, token=hf_token, gguf_file=_chosen_gguf(tuned, gguf_file))
    same_vocab = tok_base.get_vocab() == tok_tuned.get_vocab()
    warnings = [] if same_vocab else ["The tokenizers differ: KL and token statistics are disabled, generations only."]
    return ModelPair(
        tuned=tuned_model,
        base=base_model,
        tok_base=tok_base,
        tok_tuned=tok_tuned,
        placement=placement,
        dtype=_dtype_name(torch_dtype),
        same_vocab=same_vocab,
        notes=notes,
        warnings=warnings,
    )


def plan_placement(tuned_bytes: int, base_bytes: int | None, *, device: str, free_gpu_bytes: int | None) -> Placement:
    """Decide where the models run.

    ``auto``: both on the GPU when their weights fit in GPU_MEMORY_FRACTION of
    the free memory, else the tuned model on the GPU and the base on the CPU,
    else both on the CPU. ``cpu`` and ``cuda`` force a device.

    Args:
        tuned_bytes: size of the tuned model's weights.
        base_bytes: size of the base model's, None when the base is the
            tuned model with its adapter disabled.
        device: ``auto``, ``cpu`` or ``cuda``.
        free_gpu_bytes: free GPU memory, None without a usable GPU.

    Returns:
        The devices and the reason for the choice.
    """
    budget = (free_gpu_bytes or 0) * GPU_MEMORY_FRACTION
    total = tuned_bytes + (base_bytes or 0)
    sizes = f"{total / BYTES_PER_GB:.2f} GB of weights, {budget / BYTES_PER_GB:.2f} GB GPU budget"
    devices: tuple[str, str]
    if device == DEVICE_CPU:
        devices, reason = (DEVICE_CPU, DEVICE_CPU), "CPU requested"
    elif device == DEVICE_CUDA:
        devices, reason = (DEVICE_CUDA, DEVICE_CUDA), "CUDA requested"
    elif free_gpu_bytes is None:
        devices, reason = (DEVICE_CPU, DEVICE_CPU), "no CUDA device"
    elif total <= budget:
        devices, reason = (DEVICE_CUDA, DEVICE_CUDA), f"the weights fit on the GPU ({sizes})"
    elif base_bytes is not None and tuned_bytes <= budget:
        devices, reason = (DEVICE_CUDA, DEVICE_CPU), f"only the tuned model fits on the GPU ({sizes})"
    else:
        devices, reason = (DEVICE_CPU, DEVICE_CPU), f"the weights do not fit on the GPU ({sizes})"
    return Placement(tuned_device=devices[0], base_device=devices[1] if base_bytes is not None else None, reason=reason)


def resolve_dtype(dtype: str, cuda: bool) -> torch.dtype:
    """The torch dtype for a dtype name; ``auto`` is bfloat16 on a capable GPU, else float32.

    Raises:
        ValueError: for an unknown name.
    """
    if dtype != DTYPE_AUTO and dtype not in DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; use auto, {', '.join(DTYPES)}")
    result = DTYPES.get(dtype, torch.float32)
    if dtype == DTYPE_AUTO and cuda and torch.cuda.is_bf16_supported():
        result = torch.bfloat16
    return result


def _uses_cuda(device: str) -> bool:
    """Whether the device setting may put a model on the GPU.

    Raises:
        ValueError: for an unknown device, or ``cuda`` without a GPU.
    """
    if device not in (DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA):
        raise ValueError(f"Unknown device {device!r}; use auto, cpu or cuda")
    if device == DEVICE_CUDA and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return device != DEVICE_CPU and torch.cuda.is_available()


def _final_dtype(dtype: str, loaded: torch.dtype, placement: Placement) -> torch.dtype:
    """float32 when ``auto`` chose bfloat16 for a GPU but everything ended on the CPU."""
    on_gpu = DEVICE_CUDA in (placement.tuned_device, placement.base_device)
    return torch.float32 if dtype == DTYPE_AUTO and not on_gpu else loaded


def _dtype_name(dtype: torch.dtype) -> str:
    """``bfloat16`` for ``torch.bfloat16``."""
    return str(dtype).removeprefix("torch.")


def _model_bytes(model: Any) -> int:
    """Memory taken by a model's parameters and buffers."""
    tensors = chain(model.parameters(), model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _load_models(
    base: ModelRef,
    tuned: ModelRef,
    override: str | None,
    files: _GgufFiles,
    dtype: torch.dtype,
    hf_token: str | None,
    notes: list[str],
) -> tuple[Any, Any | None]:
    """(tuned model, base model or None when the base is the adapter disabled), on the CPU.

    Raises:
        ModelError: when the base is an adapter.
    """
    if base.kind is ModelKind.ADAPTER:
        raise ModelError(f"{base.ref} is a LoRA adapter; the base must be a complete model")
    result: tuple[Any, Any | None]
    if tuned.kind is ModelKind.ADAPTER:
        result = _load_adapted(base, tuned, override, files.base, dtype, hf_token, notes)
    else:
        result = (
            _load_model(tuned, files.tuned, dtype, hf_token, notes),
            _load_model(base, files.base, dtype, hf_token, notes),
        )
    return result


def _load_adapted(
    base: ModelRef,
    adapter: ModelRef,
    override: str | None,
    base_gguf_file: str | None,
    dtype: torch.dtype,
    hf_token: str | None,
    notes: list[str],
) -> tuple[Any, Any | None]:
    """The adapter on its host model; the base is loaded separately only when the host differs."""
    from peft import PeftModel

    host = detect(override, token=hf_token) if override else base
    separate = not _same_reference(host.ref, base.ref)
    if adapter.base_model and not _same_reference(adapter.base_model, host.ref):
        notes.append(f"The adapter declares {adapter.base_model} as its base; it was applied to {host.ref}.")
    host_gguf_file = None if separate else base_gguf_file
    tuned_model = PeftModel.from_pretrained(
        _load_model(host, host_gguf_file, dtype, hf_token, notes), adapter.ref, token=hf_token
    )
    base_model = _load_model(base, base_gguf_file, dtype, hf_token, notes) if separate else None
    return tuned_model.eval(), base_model


@dataclass(frozen=True)
class _GgufFiles:
    """The GGUF file chosen for each side (None: the most precise; GGUF models only)."""

    tuned: str | None = None
    base: str | None = None


def _same_reference(first: str, second: str) -> bool:
    """Whether two model references name the same model (local paths compared canonically)."""
    return canonical_ref(first) == canonical_ref(second)


def _load_model(
    model: ModelRef, gguf_file: str | None, dtype: torch.dtype, hf_token: str | None, notes: list[str]
) -> Any:
    """A complete causal LM on the CPU, whatever its storage.

    Raises:
        ModelError: for an adapter, or bitsandbytes weights that do not fit the architecture.
    """
    from transformers import AutoModelForCausalLM

    result: Any
    if model.kind is ModelKind.ADAPTER:
        raise ModelError(f"{model.ref} is a LoRA adapter, not a complete model")
    if model.kind is ModelKind.BNB:
        result = _load_bnb(model, dtype, hf_token, notes)
    elif model.kind is ModelKind.GGUF:
        folder, filename = gguf_location(model, _chosen_gguf(model, gguf_file) or model.ref)
        result = AutoModelForCausalLM.from_pretrained(folder, gguf_file=filename, dtype=dtype, token=hf_token)
        notes.append(f"{model.ref}: GGUF file {filename} dequantized by transformers to {_dtype_name(dtype)}.")
    else:
        result = AutoModelForCausalLM.from_pretrained(model.ref, dtype=dtype, token=hf_token, use_safetensors=True)
    return result.eval()


def _chosen_gguf(model: ModelRef, gguf_file: str | None) -> str | None:
    """The GGUF file to read (``GGUF_AUTO`` / None: the most precise); None for other formats."""
    result = None
    if model.kind is ModelKind.GGUF:
        result = gguf_file if gguf_file not in (None, GGUF_AUTO) else choose_gguf(model.gguf_files)
    return result


def _load_bnb(model: ModelRef, dtype: torch.dtype, hf_token: str | None, notes: list[str]) -> Any:
    """A bitsandbytes checkpoint rebuilt in plain floats, without bitsandbytes.

    The architecture comes from the config (minus its quantization settings);
    the weights are read dequantized, one tensor at a time, through the
    toolkit's :class:`~models.bnb.BnbSource`.

    Raises:
        ModelError: when a stored tensor's shape does not match the architecture.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    settings = {key: value for key, value in model.config.items() if key != "quantization_config"}
    config = AutoConfig.for_model(settings.pop("model_type"), **settings)
    result = AutoModelForCausalLM.from_config(config, dtype=dtype)
    source = open_model(model, token=hf_token)
    targets = result.state_dict()
    loaded = {name for name in source.names() if name in targets}
    for name in sorted(loaded):
        _copy_weight(targets[name], source.load(name), name)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    missing = [name for name in targets if name not in loaded and not (tied and name == TIED_UNEMBEDDING)]
    unexpected = [name for name in source.names() if name not in targets]
    notes.append(f"{model.ref}: bitsandbytes weights dequantized to {_dtype_name(dtype)} without bitsandbytes.")
    notes.extend(_key_notes(model.ref, missing, unexpected))
    result.tie_weights()
    return result


def _copy_weight(target: torch.Tensor, value: torch.Tensor, name: str) -> None:
    """Copy one dequantized tensor into the model.

    Raises:
        ModelError: on a shape mismatch (``copy_`` would silently broadcast).
    """
    if target.shape != value.shape:
        raise ModelError(f"{name}: stored shape {list(value.shape)} does not match the model's {list(target.shape)}")
    target.copy_(value)


def _key_notes(ref: str, missing: Iterable[str], unexpected: Iterable[str]) -> list[str]:
    """Notes naming the weights the checkpoint lacked (left at their initial values) or had in excess."""
    result = []
    missing, unexpected = list(missing), list(unexpected)
    if missing:
        result.append(f"{ref}: {len(missing)} weights missing, left randomly initialised: {', '.join(missing[:10])}")
    if unexpected:
        result.append(f"{ref}: {len(unexpected)} stored tensors not used by the model: {', '.join(unexpected[:10])}")
    return result


# ----------------------------------------------------------------- prompt files
def load_prompts(path: str, *, field: str = "question", limit: int = DEFAULT_PROMPT_LIMIT) -> list[str]:
    """Read prompts from a file.

    ``.txt``: one prompt per non-empty line. ``.json``: a list of strings, a
    list of objects holding ``field``, or an object holding such a list (the
    first list found). ``.jsonl``: one string or object per non-empty line.

    Args:
        path: the prompt file.
        field: key of the prompt in JSON objects.
        limit: maximum number of prompts returned (the first ones).

    Returns:
        The non-blank prompts, at most ``limit``.

    Raises:
        ValueError: for an unsupported extension, a non-positive limit, an
            item without ``field``, or a file without any prompt.
        OSError: when the file cannot be read.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in PROMPT_SUFFIXES:
        raise ValueError(f"{path}: unsupported prompt file; use {', '.join(PROMPT_SUFFIXES)}")
    if limit < 1:
        raise ValueError(f"The prompt limit must be at least 1, not {limit}")
    readers: dict[str, Callable[[str, str], list[str]]] = {
        ".txt": _read_text_prompts,
        ".json": _read_json_prompts,
        ".jsonl": _read_jsonl_prompts,
    }
    prompts = [prompt for prompt in readers[suffix](path, field) if prompt.strip()]
    if not prompts:
        raise ValueError(f"{path} holds no prompt")
    return prompts[:limit]


def _read_text_prompts(path: str, field: str) -> list[str]:
    """One prompt per line, stripped (``field`` is unused)."""
    with open(path, encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle]


def _read_json_prompts(path: str, field: str) -> list[str]:
    """Prompts of a JSON list, or of the first list inside a JSON object.

    Raises:
        ValueError: when there is no list of prompts or an item lacks ``field``.
    """
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = next((value for value in data.values() if isinstance(value, list)), None)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of prompts, or an object holding one")
    return [_prompt_of(item, field, f"{path} item {index}") for index, item in enumerate(data)]


def _read_jsonl_prompts(path: str, field: str) -> list[str]:
    """One JSON string or object per non-empty line.

    Raises:
        ValueError: when a line is not valid JSON or lacks ``field``.
    """
    with open(path, encoding="utf-8-sig") as handle:
        lines = [(number, line) for number, line in enumerate(handle, start=1) if line.strip()]
    try:
        items = [(number, json.loads(line)) for number, line in lines]
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON line ({error})") from error
    return [_prompt_of(item, field, f"{path} line {number}") for number, item in items]


def _prompt_of(item: Any, field: str, where: str) -> str:
    """The prompt held by one JSON item: the string itself, its ``field``, or its first user turn.

    Chat-format items (``{"messages": [{"role": ..., "content": ...}, ...]}``, the
    usual fine-tuning dataset layout) without ``field`` yield their first user message.

    Raises:
        ValueError: when no string prompt can be found in the item.
    """
    value = item.get(field) if isinstance(item, dict) else item
    if value is None and isinstance(item, dict):
        value = _first_user_turn(item.get(CHAT_MESSAGES_KEY))
    if not isinstance(value, str):
        raise ValueError(f"{where}: no string field {field!r} and no user message")
    return value


def _first_user_turn(messages: Any) -> str | None:
    """Content of the first ``user`` message of a chat transcript, or None."""
    turns = messages if isinstance(messages, list) else []
    return next(
        (
            turn.get("content")
            for turn in turns
            if isinstance(turn, dict) and turn.get("role") == CHAT_USER_ROLE and isinstance(turn.get("content"), str)
        ),
        None,
    )
