"""Vocabulary and embedding comparison (static, weights plus tokenizers).

Weight Diff and Token Shifts compare only the tensors a tuned model shares with
its base by shape, so the token embedding and the unembedding are dropped the
moment the vocabulary grows (``32000`` -> ``32016``): the added tokens and any
shift of the existing token rows are never examined. This analysis fills that
gap with three differential tests:

* **vocabulary diff** -- which tokens the tuned tokenizer adds, removes or
  reassigns to a different id (a reassignment breaks the id-to-id alignment the
  embedding comparison relies on), plus changed special tokens and normalizer;
* **shared-row embedding diff** -- for the rows whose id maps to the same token
  in both vocabularies, how far each token's embedding (and unembedding) moved,
  with the heaviest movers listed and a concentration z-score telling a targeted
  change from a broad one;
* **new-token characterization** -- for each added token, whether its embedding
  was actually trained (norm against the existing rows) or left as a placeholder,
  what existing tokens it sits closest to (input cosine), and whether the model
  can also emit it (its unembedding row) -- a read-only trigger versus an
  emittable payload.

Everything read from quantized (GGUF, bitsandbytes) weights is measured against
the quantization noise floor, so a dequantization artefact is never mistaken for
a real move.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis import stats
from analysis.shifts import (
    EMBEDDING_NORM_FLOOR,
    EMBEDDING_SUFFIXES,
    UNEMBEDDING_SUFFIX,
    Projection,
    TokenScore,
    VocabProjector,
)
from analysis.tokens import TokenDecoder, VocabInfo, token_text
from models.deltas import DeltaLog, iter_deltas
from models.quant import NoiseFloor, noise_floor
from models.source import TensorSource
from progress import CancelToken, ProgressSink, Stage, null_sink, report
from results import ResultMeta

ANALYSIS = "vocab"
# A new token whose embedding norm falls this far below the mean existing-row norm reads as an
# untrained placeholder rather than a token the fine-tune actually taught.
TRAINED_MIN_FRACTION = 0.5
DEFAULT_TOP_ROWS = 20
DEFAULT_NEW_TOKENS = 50
DEFAULT_NEIGHBOURS = 8
DEFAULT_READER_UPDATES = 3  # heaviest reader updates re-projected onto the suspect's extended vocabulary
DEFAULT_RANK = 6  # rank of the randomized SVD of a full (non low-rank) reader update

ADDED = "added"  # token present in the tuned vocabulary only
REMOVED = "removed"  # token present in the base vocabulary only
REASSIGNED = "reassigned"  # same token text, different id (an alignment break)

NO_DIFF_NOTE = "No vocabulary diff: a tokenizer was unavailable; rows are aligned by position."
NO_EMBEDDING_NOTE = "No embedding comparison: a model has no token embedding."


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class VocabChange:
    """One token that differs between the two vocabularies."""

    token_id: int  # id in the vocabulary the token belongs to
    text: str
    kind: str  # ADDED | REMOVED | REASSIGNED
    other_id: int | None = None  # for REASSIGNED: the token's id in the other vocabulary


@dataclass(frozen=True)
class VocabDiff:
    """How the tuned tokenizer's vocabulary differs from the base's."""

    base_size: int
    tuned_size: int
    aligned_tokens: int  # ids mapping to the same token text in both vocabularies
    added: list[VocabChange]
    removed: list[VocabChange]
    reassigned: list[VocabChange]
    special_changed: list[str]  # human lines describing changed special-token roles
    normalizer_changed: bool | None  # None when a normalizer signature was unavailable


@dataclass(frozen=True)
class MovedRow:
    """One token whose embedding (or unembedding) row moved between base and tuned."""

    token_id: int
    text: str
    rel_change: float  # ‖Δrow‖ / ‖base row‖
    cosine: float  # cos(base row, tuned row)
    beyond_noise: bool | None  # True past the quantization floor (None for full precision)


@dataclass(frozen=True)
class EmbeddingShift:
    """How much the shared rows of one embedding matrix moved."""

    name: str
    rows_compared: int
    rows_changed: int
    concentration: stats.NullComparison | None  # z of the top mover against all rows' movement
    moved: list[MovedRow]  # the heaviest movers, most-moved first
    noise_floor: float | None  # relative change quantization alone would produce (None: full precision)


@dataclass(frozen=True)
class NewToken:
    """An added token, characterised from its embedding and unembedding rows."""

    token_id: int
    text: str
    input_norm: float  # ‖embedding row‖
    output_norm: float | None  # ‖unembedding row‖ (None when the model ties the two)
    trained: bool  # the embedding looks trained, not an untouched placeholder
    emittable: bool | None  # the model can produce the token (None when undetermined)
    neighbours: list[TokenScore]  # existing tokens closest to it by input-embedding cosine


@dataclass(frozen=True)
class ReaderTrigger:
    """A heavy reader update re-projected onto the suspect's extended vocabulary.

    Token Shifts reads reader directions against the *base* vocabulary, so it cannot see a trigger
    that is one of the added tokens. This projects the heaviest reader updates against the suspect's
    own embeddings instead, and reports which of the aligned tokens are newly added (``new_hits``).
    """

    name: str
    layer: int
    module: str
    sigma: float
    projection: Projection | None  # TRIGGERED_BY, read on the suspect vocabulary (None if unreadable)
    new_hits: list[TokenScore]  # tokens among the top ends that the tuned model added, with their score


@dataclass
class VocabResult:
    """Everything :func:`compare_vocab` found, as plain values."""

    meta: ResultMeta
    diff: VocabDiff | None  # None when a tokenizer was unavailable
    embeddings: list[EmbeddingShift]
    new_tokens: list[NewToken]
    reader_triggers: list[ReaderTrigger]  # heaviest reader updates on the extended vocabulary (test C)
    vocabulary: int  # tokens considered (the smaller of the two embeddings)


# ----------------------------------------------------------------- analysis
def compare_vocab(
    base: TensorSource,
    tuned: TensorSource,
    decoder: TokenDecoder,
    *,
    inputs: dict[str, str],
    base_vocab: VocabInfo | None = None,
    tuned_vocab: VocabInfo | None = None,
    top_rows: int = DEFAULT_TOP_ROWS,
    max_new: int = DEFAULT_NEW_TOKENS,
    neighbours: int = DEFAULT_NEIGHBOURS,
    reader_updates: int = DEFAULT_READER_UPDATES,
    rank: int = DEFAULT_RANK,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
) -> VocabResult:
    """Compare the vocabularies and token embeddings of ``tuned`` against ``base``.

    Args:
        base: the reference model.
        tuned: the model compared with the base.
        decoder: the tuned model's tokenizer, to render token ids as text.
        inputs: what is being compared, recorded in the result metadata.
        base_vocab: the base tokenizer's vocabulary; None skips the vocabulary diff.
        tuned_vocab: the tuned tokenizer's vocabulary; None skips the vocabulary diff.
        top_rows: moved rows listed per embedding matrix, and tokens per reader-trigger axis end.
        max_new: added tokens characterised (the JSON keeps the full count separately).
        neighbours: existing tokens listed per new token.
        reader_updates: heaviest reader updates re-projected onto the suspect vocabulary when tokens
            were added (0 skips the cross-check); this reads the weights a second time, tensor by tensor.
        rank: rank of the randomized SVD of a full reader update.
        sink: receives progress events.
        token: checked between matrices and tokens, so a front-end can cancel.

    Returns:
        The vocabulary diff, the per-matrix embedding shifts, the new tokens and the reader triggers.

    Raises:
        JobCancelled: when ``token`` is cancelled.
    """
    notes: list[str] = []
    report(sink, Stage.VOCAB, "Reading the vocabulary")
    diff = _vocab_diff(base_vocab, tuned_vocab) if base_vocab is not None and tuned_vocab is not None else None
    if diff is None:
        notes.append(NO_DIFF_NOTE)
    embeddings, new_tokens, reader_triggers, vocabulary = _embedding_analysis(
        base,
        tuned,
        decoder,
        base_vocab,
        tuned_vocab,
        top_rows,
        max_new,
        neighbours,
        reader_updates,
        rank,
        notes,
        sink,
        token,
    )
    meta = ResultMeta(
        analysis=ANALYSIS,
        inputs=inputs,
        parameters={
            "top_rows": top_rows,
            "max_new": max_new,
            "neighbours": neighbours,
            "reader_updates": reader_updates,
        },
        notes=notes,
        warnings=_warnings(embeddings),
    )
    return VocabResult(
        meta=meta,
        diff=diff,
        embeddings=embeddings,
        new_tokens=new_tokens,
        reader_triggers=reader_triggers,
        vocabulary=vocabulary,
    )


def _vocab_diff(base: VocabInfo, tuned: VocabInfo) -> VocabDiff:
    """Compare two vocabularies token by token."""
    base_ids = {text: index for index, text in base.id_to_text.items()}
    tuned_ids = {text: index for index, text in tuned.id_to_text.items()}
    added = [VocabChange(index, text, ADDED) for text, index in tuned_ids.items() if text not in base_ids]
    removed = [VocabChange(index, text, REMOVED) for text, index in base_ids.items() if text not in tuned_ids]
    reassigned = [
        VocabChange(tuned_ids[text], text, REASSIGNED, other_id=index)
        for text, index in base_ids.items()
        if text in tuned_ids and tuned_ids[text] != index
    ]
    aligned = sum(1 for index, text in base.id_to_text.items() if tuned.id_to_text.get(index) == text)
    normalizer_changed = (
        None if base.normalizer is None or tuned.normalizer is None else base.normalizer != tuned.normalizer
    )
    return VocabDiff(
        base_size=base.size,
        tuned_size=tuned.size,
        aligned_tokens=aligned,
        added=sorted(added, key=lambda change: change.token_id),
        removed=sorted(removed, key=lambda change: change.token_id),
        reassigned=sorted(reassigned, key=lambda change: change.token_id),
        special_changed=_special_changes(base.specials, tuned.specials),
        normalizer_changed=normalizer_changed,
    )


def _special_changes(base: dict[str, str], tuned: dict[str, str]) -> list[str]:
    """Human lines for every special-token role that was added, removed or changed."""
    lines = []
    for role in sorted(set(base) | set(tuned)):
        before, after = base.get(role), tuned.get(role)
        if before != after:
            lines.append(f"{role}: {before or '∅'} → {after or '∅'}")
    return lines


def _embedding_analysis(
    base: TensorSource,
    tuned: TensorSource,
    decoder: TokenDecoder,
    base_vocab: VocabInfo | None,
    tuned_vocab: VocabInfo | None,
    top_rows: int,
    max_new: int,
    neighbours: int,
    reader_updates: int,
    rank: int,
    notes: list[str],
    sink: ProgressSink,
    token: CancelToken | None,
) -> tuple[list[EmbeddingShift], list[NewToken], list[ReaderTrigger], int]:
    """Run the shared-row, new-token and reader-trigger tests, or record why they could not run."""
    names = _embedding_names(base, tuned)
    if names is None:
        notes.append(NO_EMBEDDING_NOTE)
        return [], [], [], 0
    base_name, tuned_name = names
    base_embed, tuned_embed = base.load(base_name), tuned.load(tuned_name)
    aligned_ids, new_ids = _row_partition(base_vocab, tuned_vocab, base_embed.shape[0], tuned_embed.shape[0])
    _check(token)
    report(sink, Stage.VOCAB, "Comparing the shared embedding rows")
    floor = _combined_floor(base.quantization(base_name), tuned.quantization(tuned_name), base_embed)
    shifts = [_embedding_shift(tuned_name, base_embed, tuned_embed, aligned_ids, decoder, floor, top_rows)]
    shifts.extend(_unembedding_shift(base, tuned, aligned_ids, decoder, top_rows, token))
    report(sink, Stage.VOCAB, "Characterizing the new tokens")
    output, tied = _output_matrix(tuned, tuned_embed)
    new_tokens = _new_tokens(tuned_embed, output, tied, aligned_ids, new_ids, decoder, neighbours, max_new, token)
    reader_triggers = _reader_triggers(base, tuned, decoder, new_ids, reader_updates, rank, top_rows, sink, token)
    return shifts, new_tokens, reader_triggers, len(aligned_ids)


def _reader_triggers(
    base: TensorSource,
    tuned: TensorSource,
    decoder: TokenDecoder,
    new_ids: list[int],
    reader_updates: int,
    rank: int,
    top: int,
    sink: ProgressSink,
    token: CancelToken | None,
) -> list[ReaderTrigger]:
    """Re-project the heaviest reader updates onto the suspect's extended vocabulary (test C).

    Skipped when no token was added (nothing the base projection could have missed) or when the
    cross-check is turned off. Reads the weights a second time to find the reader updates, then
    projects each against the suspect's own embeddings so an added token can surface as the trigger.
    """
    if not new_ids or reader_updates <= 0:
        return []
    report(sink, Stage.VOCAB, "Re-projecting the heaviest reader updates onto the extended vocabulary")
    projector = VocabProjector.from_source(tuned, decoder)
    new_set = set(new_ids)
    candidates: list[tuple[float, str, stats.Direction]] = []
    log = DeltaLog()
    for delta in iter_deltas(base, tuned, log, token):
        is_changed_reader = (
            delta.delta.dim() == 2
            and stats.role_of(delta.name) == stats.ROLE_READER
            and delta.delta.norm().item() >= stats.CHANGE_THRESHOLD
        )
        if is_changed_reader:
            direction = stats.leading_direction(delta, rank)
            candidates.append((direction.sigma, delta.name, direction))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return [
        _reader_trigger(name, direction, projector, new_set, top) for _, name, direction in candidates[:reader_updates]
    ]


def _reader_trigger(
    name: str, direction: stats.Direction, projector: VocabProjector, new_set: set[int], top: int
) -> ReaderTrigger:
    """Project one reader update onto the vocabulary and pick out any newly added tokens it hits."""
    projection = projector.project(stats.ROLE_READER, direction.u, direction.v, top)
    hits = (
        [
            token
            for end in (projection.positive, projection.negative)
            for token in end.tokens
            if token.token_id in new_set
        ]
        if projection is not None
        else []
    )
    return ReaderTrigger(
        name=name,
        layer=stats.layer_of(name),
        module=stats.module_of(name),
        sigma=direction.sigma,
        projection=projection,
        new_hits=hits,
    )


def _embedding_names(base: TensorSource, tuned: TensorSource) -> tuple[str, str] | None:
    """The token-embedding tensor name in each model, or None when either lacks one."""
    base_name = _first_found(base, EMBEDDING_SUFFIXES)
    tuned_name = _first_found(tuned, EMBEDDING_SUFFIXES)
    return (base_name, tuned_name) if base_name is not None and tuned_name is not None else None


def _row_partition(
    base_vocab: VocabInfo | None, tuned_vocab: VocabInfo | None, base_rows: int, tuned_rows: int
) -> tuple[list[int], list[int]]:
    """Split token ids into aligned rows (same token in both) and new rows (added in tuned).

    With both vocabularies the aligned ids are those mapping to the same token text; the new ids
    are the tokens whose text the base lacks. Without them, rows are aligned by position and the
    extra tuned rows are treated as new. Ids are kept within the matrices' row counts.
    """
    shared = min(base_rows, tuned_rows)
    if base_vocab is None or tuned_vocab is None:
        aligned = list(range(shared))
        new = list(range(base_rows, tuned_rows))
    else:
        aligned = sorted(
            index for index in range(shared) if base_vocab.id_to_text.get(index) == tuned_vocab.id_to_text.get(index)
        )
        base_texts = set(base_vocab.id_to_text.values())
        new = sorted(
            index for index, text in tuned_vocab.id_to_text.items() if text not in base_texts and index < tuned_rows
        )
    return aligned, new


def _embedding_shift(
    name: str,
    base_mat: torch.Tensor,
    tuned_mat: torch.Tensor,
    aligned_ids: list[int],
    decoder: TokenDecoder,
    floor: NoiseFloor | None,
    top_rows: int,
) -> EmbeddingShift:
    """Measure how far each aligned row moved, list the heaviest movers and score concentration."""
    index = torch.tensor(aligned_ids, dtype=torch.long)
    base_rows, tuned_rows = base_mat[index], tuned_mat[index]
    delta_norm = (tuned_rows - base_rows).norm(dim=1)
    rel = delta_norm / base_rows.norm(dim=1).clamp_min(EMBEDDING_NORM_FLOOR)
    cosine = F.cosine_similarity(base_rows, tuned_rows, dim=1)
    rel_values = rel.tolist()
    concentration = stats.NullComparison.build(max(rel_values), rel_values) if rel_values else None
    count = min(top_rows, rel.numel())
    _, order = rel.topk(count) if count else (rel, torch.empty(0, dtype=torch.long))
    moved = [
        MovedRow(
            token_id=aligned_ids[position],
            text=token_text(decoder, aligned_ids[position]),
            rel_change=rel[position].item(),
            cosine=cosine[position].item(),
            beyond_noise=stats.beyond_noise(rel[position].item(), floor),
        )
        for position in order.tolist()
    ]
    return EmbeddingShift(
        name=name,
        rows_compared=len(aligned_ids),
        rows_changed=int((delta_norm >= stats.CHANGE_THRESHOLD).sum().item()),
        concentration=concentration,
        moved=moved,
        noise_floor=floor.rel_error if floor is not None else None,
    )


def _unembedding_shift(
    base: TensorSource,
    tuned: TensorSource,
    aligned_ids: list[int],
    decoder: TokenDecoder,
    top_rows: int,
    token: CancelToken | None,
) -> list[EmbeddingShift]:
    """The shared-row shift of ``lm_head`` as well, when both models store it untied."""
    base_name, tuned_name = base.find(UNEMBEDDING_SUFFIX), tuned.find(UNEMBEDDING_SUFFIX)
    result: list[EmbeddingShift] = []
    if base_name is not None and tuned_name is not None:
        _check(token)
        base_mat, tuned_mat = base.load(base_name), tuned.load(tuned_name)
        floor = _combined_floor(base.quantization(base_name), tuned.quantization(tuned_name), base_mat)
        usable = [index for index in aligned_ids if index < base_mat.shape[0] and index < tuned_mat.shape[0]]
        result.append(_embedding_shift(tuned_name, base_mat, tuned_mat, usable, decoder, floor, top_rows))
    return result


def _output_matrix(tuned: TensorSource, tuned_embed: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """The matrix the model emits through, and whether it is tied to the input embedding."""
    name = tuned.find(UNEMBEDDING_SUFFIX)
    return (tuned_embed, True) if name is None else (tuned.load(name), False)


def _new_tokens(
    tuned_embed: torch.Tensor,
    output: torch.Tensor,
    tied: bool,
    aligned_ids: list[int],
    new_ids: list[int],
    decoder: TokenDecoder,
    neighbours: int,
    max_new: int,
    token: CancelToken | None,
) -> list[NewToken]:
    """Characterise each added token from its input and output embedding rows."""
    rows = tuned_embed.shape[0]
    usable = [index for index in new_ids if index < rows][:max_new]
    if not usable or not aligned_ids:
        return []
    existing_index = torch.tensor(aligned_ids, dtype=torch.long)
    existing_unit = F.normalize(tuned_embed[existing_index], dim=1)
    input_floor = TRAINED_MIN_FRACTION * tuned_embed[existing_index].norm(dim=1).mean().item()
    output_floor = None if tied else TRAINED_MIN_FRACTION * output[existing_index].norm(dim=1).mean().item()
    result = []
    for token_id in usable:
        _check(token)
        result.append(
            _new_token(
                token_id,
                tuned_embed,
                output,
                tied,
                existing_unit,
                aligned_ids,
                decoder,
                neighbours,
                input_floor,
                output_floor,
            )
        )
    return result


def _new_token(
    token_id: int,
    tuned_embed: torch.Tensor,
    output: torch.Tensor,
    tied: bool,
    existing_unit: torch.Tensor,
    aligned_ids: list[int],
    decoder: TokenDecoder,
    neighbours: int,
    input_floor: float,
    output_floor: float | None,
) -> NewToken:
    """Characterise one added token: trained, emittable and its nearest existing tokens."""
    row = tuned_embed[token_id]
    input_norm = row.norm().item()
    trained = input_norm >= input_floor
    output_norm, emittable = _emittable(token_id, output, tied, trained, output_floor)
    scores = existing_unit @ F.normalize(row, dim=0)
    count = min(neighbours, scores.numel())
    values, order = scores.topk(count) if count else (scores, torch.empty(0, dtype=torch.long))
    nearest = [
        TokenScore(token_id=aligned_ids[position], text=token_text(decoder, aligned_ids[position]), score=value)
        for position, value in zip(order.tolist(), values.tolist(), strict=True)
    ]
    return NewToken(
        token_id=token_id,
        text=token_text(decoder, token_id),
        input_norm=input_norm,
        output_norm=output_norm,
        trained=trained,
        emittable=emittable,
        neighbours=nearest,
    )


def _emittable(
    token_id: int, output: torch.Tensor, tied: bool, trained: bool, output_floor: float | None
) -> tuple[float | None, bool | None]:
    """The token's unembedding-row norm and whether the model can emit it.

    A tied model shares one matrix, so emitting is the same question as being trained; an untied
    model has a separate output row that must itself carry weight.
    """
    result: tuple[float | None, bool | None]
    if tied:
        result = (None, trained)
    elif token_id < output.shape[0] and output_floor is not None:
        output_norm = output[token_id].norm().item()
        result = (output_norm, output_norm >= output_floor)
    else:
        result = (None, None)
    return result


def _combined_floor(
    base_scheme: object | None, tuned_scheme: object | None, base_mat: torch.Tensor
) -> NoiseFloor | None:
    """The quantization noise floor of both stored tensors, combined in quadrature, or None."""
    from models.quant import QuantScheme

    schemes = [
        scheme for scheme in (base_scheme, tuned_scheme) if isinstance(scheme, QuantScheme) and not scheme.lossless
    ]
    floors = [noise_floor(base_mat, scheme) for scheme in schemes]
    result = None
    if floors:
        result = NoiseFloor(
            rel_error=sum(floor.rel_error**2 for floor in floors) ** 0.5,
            exact=all(floor.exact for floor in floors),
            method=" + ".join(floor.method for floor in floors),
        )
    return result


def _warnings(embeddings: list[EmbeddingShift]) -> list[str]:
    """A quantization caveat when any embedding matrix was read from quantized storage."""
    quantized = any(shift.noise_floor is not None for shift in embeddings)
    return (
        [
            "Quantized embeddings: every row change includes quantization error. A row counts as moved "
            f"('beyond noise') only past {stats.NOISE_MARGIN:g} noise floors."
        ]
        if quantized
        else []
    )


def _first_found(source: TensorSource, suffixes: tuple[str, ...]) -> str | None:
    """The first tensor of ``source`` matching one of ``suffixes``, in order of preference."""
    return next((name for name in map(source.find, suffixes) if name is not None), None)


def _check(token: CancelToken | None) -> None:
    """Stop at a checkpoint when the run was cancelled."""
    if token is not None:
        token.check()
