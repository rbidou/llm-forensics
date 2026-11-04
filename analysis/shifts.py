"""Which tokens the biggest weight updates relate to (static, weights only).

For every tensor a tuned model shares with its base, the update ΔW is measured;
the heaviest 2-D updates are reduced to their leading singular direction and
read through the vocabulary (README §2 of the reference scripts):

* writers (``o_proj``, ``down_proj``, ...) add their output direction ``u`` to
  the residual stream: pushed through the final norm and the unembedding it
  gives the tokens the update promotes (logit lens);
* readers (``q/k/v_proj``, ``gate/up_proj``, ...) respond to their input
  direction ``v``: its cosine with the token embeddings gives the tokens that
  trigger the update (meaningful mostly in early layers).

Each list of tokens comes with a null baseline: the same statistic for random
unit directions, so a reader can tell a sharp token match from noise.

SVD signs are arbitrary: ``positive`` and ``negative`` are two ends of one axis.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from analysis import stats
from analysis.tokens import TokenDecoder, token_text
from models.deltas import Delta, DeltaLog, iter_deltas, total_deltas
from models.quant import NoiseFloor
from models.source import ModelError, TensorSource
from progress import CancelToken, ProgressSink, Stage, null_sink, report
from results import ResultMeta

UNEMBEDDING_SUFFIX = "lm_head.weight"
EMBEDDING_SUFFIXES: tuple[str, ...] = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")
FINAL_NORM_SUFFIXES: tuple[str, ...] = (
    "model.norm.weight",
    "transformer.ln_f.weight",
    "model.final_layernorm.weight",
    "ln_f.weight",
    "final_layernorm.weight",
)
PROMOTES = "promotes"  # writer: tokens the direction pushes up in the logits
TRIGGERED_BY = "triggered_by"  # reader: tokens whose embedding aligns with the direction
EMBEDDING_NORM_FLOOR = 1e-8
DEFAULT_NULL_DRAWS = 20
SIGN_NOTE = "SVD sign is arbitrary: read '+' and '-' as the two ends of one axis."


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class TokenScore:
    """One token and its score along a direction (logit or cosine)."""

    token_id: int
    text: str
    score: float


@dataclass(frozen=True)
class AxisEnd:
    """The top tokens at one end of a direction, against random directions.

    ``null.value`` is the mean score of ``tokens``; the null samples are the same
    statistic for random unit directions of the same space.
    """

    tokens: list[TokenScore]
    null: stats.NullComparison


@dataclass(frozen=True)
class Projection:
    """A singular direction read through the vocabulary."""

    relation: str  # PROMOTES (writer, logit lens) or TRIGGERED_BY (reader, embedding cosine)
    positive: AxisEnd
    negative: AxisEnd


@dataclass(frozen=True)
class TopUpdate:
    """One of the heaviest updates, by leading singular value."""

    name: str
    layer: int
    module: str
    role: str
    sigma: float
    share: float  # σ₁² / ‖ΔW‖²
    rank: int | None  # exact numerical rank for low-rank (adapter) updates
    projection: Projection | None  # None for roles without a vocabulary reading
    rel_change: float | None = None  # ‖ΔW‖ / ‖W‖
    noise_floor: float | None = None  # relative change quantization alone would produce
    beyond_noise: bool | None = None  # rel_change > NOISE_MARGIN * noise_floor (quantized models only)


@dataclass(frozen=True)
class LayerChange:
    """Mean relative change ‖ΔW‖ / ‖W‖ of the changed tensors of one layer (-1: outside layers)."""

    layer: int
    mean_rel_change: float
    tensors: int


@dataclass(frozen=True)
class EmbeddingRows:
    """Tokens whose row of an embedding / unembedding matrix moved most (relative change)."""

    name: str
    tokens: list[TokenScore]


@dataclass(frozen=True)
class ShiftCounts:
    """How many tensors were compared, and how many moved."""

    compared: int
    changed: int
    identical: int
    unmatched: int


@dataclass
class TokenShiftResult:
    """Everything :func:`token_shifts` found, as plain values."""

    meta: ResultMeta
    counts: ShiftCounts
    top_updates: list[TopUpdate]
    embedding_rows: list[EmbeddingRows]
    layers: list[LayerChange]
    vocabulary: int  # tokens projected onto (0 when no projection was possible)


# ----------------------------------------------------------------- vocabulary projection
class VocabProjector:
    """Reads residual-stream directions as tokens, with a random-direction null baseline."""

    def __init__(
        self,
        unembedding: torch.Tensor,
        embedding_unit: torch.Tensor,
        decoder: TokenDecoder,
        null_draws: int = DEFAULT_NULL_DRAWS,
        seed: int = stats.DEFAULT_SEED,
    ) -> None:
        """Keep the two vocabulary matrices.

        Args:
            unembedding: vocab x hidden, already multiplied by the final norm gain.
            embedding_unit: vocab x hidden input embeddings, rows of unit norm.
            decoder: turns token ids into text.
            null_draws: random directions per null baseline.
            seed: seed of the random directions (the same ones for every projection).
        """
        self.unembedding = unembedding
        self.embedding_unit = embedding_unit
        self.decoder = decoder
        self.null_draws = null_draws
        self.seed = seed
        self._null_scores: dict[str, torch.Tensor] = {}

    @classmethod
    def from_source(
        cls,
        source: TensorSource,
        decoder: TokenDecoder,
        null_draws: int = DEFAULT_NULL_DRAWS,
        seed: int = stats.DEFAULT_SEED,
    ) -> VocabProjector:
        """Build the projector from a base model's weights.

        The unembedding is ``lm_head.weight`` when present, else the (tied) token
        embedding, times the final norm gain. Readers are matched against the
        real input embedding, even when ``lm_head`` exists. Only the first
        ``min(rows, len(decoder))`` tokens are kept.

        Raises:
            ModelError: when the model has no token embedding.
        """
        embedding_name = _first_found(source, EMBEDDING_SUFFIXES)
        if embedding_name is None:
            raise ModelError(f"{source.ref}: no token embedding found")
        unembedding_name = source.find(UNEMBEDDING_SUFFIX) or embedding_name
        tokens = min(source.shape(unembedding_name)[0], source.shape(embedding_name)[0], len(decoder))
        unembedding = source.load(unembedding_name, (0, tokens))
        embedding = source.load(embedding_name, (0, tokens))
        unit = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(EMBEDDING_NORM_FLOOR)
        gain = _final_norm(source, unembedding.shape[1])
        return cls(unembedding * gain, unit, decoder, null_draws, seed)

    @property
    def hidden(self) -> int:
        """Size of the residual stream."""
        return int(self.unembedding.shape[1])

    @property
    def tokens(self) -> int:
        """Number of tokens projected onto."""
        return int(self.unembedding.shape[0])

    def project(self, role: str, u: torch.Tensor, v: torch.Tensor, k: int) -> Projection | None:
        """Project a singular pair according to the module's role.

        Writers use ``u`` when it has the hidden size, else ``v``; readers the
        reverse (Conv1D-style weights are stored transposed).

        Returns:
            The projection, or None for other roles or when neither vector fits.
        """
        result = None
        if role == stats.ROLE_WRITER:
            vector = u if u.numel() == self.hidden else v
            result = self.promoted(vector, k) if vector.numel() == self.hidden else None
        elif role == stats.ROLE_READER:
            vector = v if v.numel() == self.hidden else u
            result = self.triggered(vector, k) if vector.numel() == self.hidden else None
        return result

    def promoted(self, u: torch.Tensor, k: int) -> Projection:
        """Tokens a written direction promotes (``+``) or, negated, promotes (``-``): logit lens."""
        return self._project(PROMOTES, self.unembedding, u, k)

    def triggered(self, v: torch.Tensor, k: int) -> Projection:
        """Tokens whose embedding has the highest cosine with ``v`` (``+``) or ``-v`` (``-``)."""
        return self._project(TRIGGERED_BY, self.embedding_unit, v, k)

    def _project(self, relation: str, matrix: torch.Tensor, direction: torch.Tensor, k: int) -> Projection:
        """Score every token along a unit direction, at both ends, with their null baselines."""
        unit = direction.float() / direction.norm().clamp_min(stats.MIN_NORM)
        scores = matrix @ unit
        nulls = self._nulls(relation, matrix)
        count = min(k, scores.numel())
        return Projection(
            relation=relation,
            positive=self._axis_end(scores, nulls, count),
            negative=self._axis_end(-scores, -nulls, count),
        )

    def _axis_end(self, scores: torch.Tensor, nulls: torch.Tensor, count: int) -> AxisEnd:
        """Top ``count`` tokens of ``scores``; null = mean top-``count`` score of each random direction."""
        values, ids = scores.topk(count)
        null_stats = nulls.topk(count, dim=0).values.mean(0).tolist()
        tokens = [
            TokenScore(token_id=token_id, text=token_text(self.decoder, token_id), score=value)
            for token_id, value in zip(ids.tolist(), values.tolist(), strict=True)
        ]
        return AxisEnd(tokens=tokens, null=stats.NullComparison.build(values.mean().item(), null_stats))

    def _nulls(self, relation: str, matrix: torch.Tensor) -> torch.Tensor:
        """Scores (vocab x draws) of seeded random unit directions, computed once per matrix."""
        if relation not in self._null_scores:
            generator = stats.seeded_generator(self.seed)
            directions = torch.randn(self.null_draws, self.hidden, generator=generator)
            directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(stats.MIN_NORM)
            self._null_scores[relation] = matrix @ directions.T
        return self._null_scores[relation]


def _first_found(source: TensorSource, suffixes: tuple[str, ...]) -> str | None:
    """The first tensor of ``source`` matching one of ``suffixes``, in order of preference."""
    return next((name for name in map(source.find, suffixes) if name is not None), None)


def _final_norm(source: TensorSource, hidden: int) -> torch.Tensor:
    """The final norm gain, or ones when the model has none of the expected size."""
    name = _first_found(source, FINAL_NORM_SUFFIXES)
    gain = source.load(name) if name is not None else None
    return gain if gain is not None and gain.shape == (hidden,) else torch.ones(hidden)


# ----------------------------------------------------------------- analysis
def token_shifts(
    base: TensorSource,
    tuned: TensorSource,
    decoder: TokenDecoder,
    *,
    inputs: dict[str, str],
    n: int = 10,
    top: int = 15,
    rank: int = 6,
    null_draws: int = DEFAULT_NULL_DRAWS,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
) -> TokenShiftResult:
    """Measure every update of ``tuned`` against ``base`` and read the heaviest ones as tokens.

    Args:
        base: the reference model (also provides the vocabulary matrices).
        tuned: the model to compare (full weights, or an adapter applied to ``base``).
        decoder: the base model's tokenizer (or any :class:`TokenDecoder`).
        inputs: what is being compared, recorded in the result metadata.
        n: number of heaviest 2-D updates (by σ₁) to keep and project.
        top: tokens listed per axis end and per embedding matrix.
        rank: rank of the randomized SVD of full (non low-rank) updates.
        null_draws: random directions of each null baseline.
        sink: receives progress events.
        token: checked between tensors, so a front-end can cancel.

    Returns:
        Counts, per-layer change profile, moved embedding rows and the top updates.

    Raises:
        JobCancelled: when ``token`` is cancelled.
    """
    log = DeltaLog()
    accumulator = _ShiftAccumulator(
        n=n, rank=rank, tokens=len(decoder), floor_of=lambda delta: stats.quantization_floor(delta, base, tuned)
    )
    total = total_deltas(base, tuned)
    for delta in iter_deltas(base, tuned, log, token):
        report(sink, Stage.SHIFTS, f"Comparing {delta.name}", log.visited, total)
        accumulator.add(delta)
    warnings: list[str] = []
    projector = _projector_or_none(base, decoder, null_draws, warnings)
    report(sink, Stage.SHIFTS, "Projecting the top updates onto the vocabulary", total, total)
    top_updates = [_top_update(candidate, projector, top, token) for candidate in accumulator.ranked()]
    if any(update.noise_floor is not None for update in top_updates):
        warnings.append(
            "Quantized weights: every ΔW includes quantization error. Read each update against its noise "
            f"floor ('beyond noise' = more than {stats.NOISE_MARGIN:g} floors); updates within the noise "
            "project quantization artefacts, not fine-tuning."
        )
    meta = ResultMeta(
        analysis="shifts",
        inputs=inputs,
        parameters={"n": n, "top": top, "rank": rank, "null_draws": null_draws},
        notes=[*log.notes, SIGN_NOTE],
        warnings=warnings,
    )
    return TokenShiftResult(
        meta=meta,
        counts=ShiftCounts(
            compared=accumulator.compared,
            changed=accumulator.changed,
            identical=accumulator.compared - accumulator.changed,
            unmatched=log.unmatched,
        ),
        top_updates=top_updates,
        embedding_rows=[_embedding_rows(name, rows, decoder, top) for name, rows in accumulator.embedding_rows],
        layers=accumulator.layer_profile(),
        vocabulary=projector.tokens if projector is not None else 0,
    )


@dataclass(frozen=True)
class _Candidate:
    """A heavy update kept for projection (holds tensors, never part of a result)."""

    name: str
    direction: stats.Direction
    rel_change: float
    floor: NoiseFloor | None


@dataclass
class _ShiftAccumulator:
    """Streaming summary of the deltas: counts, per-layer change, embedding rows, top-n by σ₁."""

    n: int
    rank: int
    tokens: int
    floor_of: Callable[[Delta], NoiseFloor | None]
    compared: int = 0
    changed: int = 0
    heap: list[tuple[float, int, _Candidate]] = field(default_factory=list)  # min-heap on σ₁
    by_layer: defaultdict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    embedding_rows: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    def add(self, delta: Delta) -> None:
        """Account for one compared tensor."""
        self.compared += 1
        delta_norm = delta.delta.norm().item()
        if delta_norm >= stats.CHANGE_THRESHOLD:
            self.changed += 1
            self.by_layer[stats.layer_of(delta.name)].append(delta_norm / max(delta.base.norm().item(), stats.MIN_NORM))
            if delta.delta.dim() == 2 and stats.is_embedding(delta.name):
                self.embedding_rows.append((delta.name, _row_changes(delta, self.tokens)))
            elif delta.delta.dim() == 2 and self.n > 0:
                self._push(delta)

    def _push(self, delta: Delta) -> None:
        """Keep ``delta`` if it is among the ``n`` largest by σ₁ so far."""
        direction = stats.leading_direction(delta, self.rank)
        if len(self.heap) < self.n or direction.sigma > self.heap[0][0]:
            rel_change = delta.delta.norm().item() / max(delta.base.norm().item(), stats.MIN_NORM)
            item = (direction.sigma, self.compared, _Candidate(delta.name, direction, rel_change, self.floor_of(delta)))
            if len(self.heap) < self.n:
                heapq.heappush(self.heap, item)
            else:
                heapq.heapreplace(self.heap, item)

    def ranked(self) -> list[_Candidate]:
        """The kept updates, largest σ₁ first."""
        return [candidate for _, _, candidate in sorted(self.heap, key=lambda item: item[0], reverse=True)]

    def layer_profile(self) -> list[LayerChange]:
        """Mean relative change of the changed tensors, per layer."""
        return [
            LayerChange(layer=layer, mean_rel_change=sum(values) / len(values), tensors=len(values))
            for layer, values in sorted(self.by_layer.items())
        ]


def _row_changes(delta: Delta, tokens: int) -> torch.Tensor:
    """Relative change of each of the first ``tokens`` rows."""
    rows = delta.delta[:tokens].norm(dim=1)
    return rows / delta.base[:tokens].norm(dim=1).clamp_min(EMBEDDING_NORM_FLOOR)


def _projector_or_none(
    base: TensorSource, decoder: TokenDecoder, null_draws: int, warnings: list[str]
) -> VocabProjector | None:
    """The vocabulary projector, or None (with a warning) when the base has no embedding."""
    result = None
    try:
        result = VocabProjector.from_source(base, decoder, null_draws)
    except ModelError as error:
        warnings.append(f"No vocabulary projection: {error}")
    return result


def _top_update(
    candidate: _Candidate, projector: VocabProjector | None, top: int, token: CancelToken | None
) -> TopUpdate:
    """Describe one kept update and project it onto the vocabulary."""
    if token is not None:
        token.check()
    role = stats.role_of(candidate.name)
    direction = candidate.direction
    projection = projector.project(role, direction.u, direction.v, top) if projector is not None else None
    return TopUpdate(
        name=candidate.name,
        layer=stats.layer_of(candidate.name),
        module=stats.module_of(candidate.name),
        role=role,
        sigma=direction.sigma,
        share=direction.share,
        rank=direction.rank,
        projection=projection,
        rel_change=candidate.rel_change,
        noise_floor=candidate.floor.rel_error if candidate.floor is not None else None,
        beyond_noise=stats.beyond_noise(candidate.rel_change, candidate.floor),
    )


def _embedding_rows(name: str, rows: torch.Tensor, decoder: TokenDecoder, top: int) -> EmbeddingRows:
    """The ``top`` rows of an embedding matrix that moved most, as tokens."""
    values, ids = rows.topk(min(top, rows.numel()))
    tokens = [
        TokenScore(token_id=token_id, text=token_text(decoder, token_id), score=value)
        for token_id, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]
    return EmbeddingRows(name=name, tokens=tokens)
