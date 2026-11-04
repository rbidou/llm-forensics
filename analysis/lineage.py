"""A Hub model compared with its declared lineage (static, weights only).

Discovery reads Hub metadata only, nothing is verified on the weights here:

* parents come from the ``base_model:<relation>:<repo>`` tags, else from the
  model card's ``base_model``; only ``finetune``, ``adapter`` and ``quantized``
  parents are compared (for a merge, the first of those, else the first listed
  parent);
* children are the models declaring this one as their adapter, fine-tune or
  quantization base, the most downloaded first.

Every usable (parent, child) pair then goes through :func:`token_shifts`. The
models are opened with the toolkit readers, so Hub weights are fetched tensor
by tensor with HTTP Range requests rather than downloaded whole.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from analysis.shifts import DEFAULT_NULL_DRAWS, TokenShiftResult, token_shifts
from analysis.tokens import TokenDecoder
from models.detect import GGUF_AUTO, ModelKind, RepoFactory, choose_gguf, detect, open_model
from models.hub import HubRepo
from models.lora import ADAPTER_CONFIG
from models.source import TensorSource
from progress import CancelToken, JobCancelled, ProgressSink, Stage, null_sink, report
from results import ResultMeta

MODES: tuple[str, ...] = ("auto", "parent", "children")
PARENT_MODES: tuple[str, ...] = ("auto", "parent")
CHILDREN_MODES: tuple[str, ...] = ("auto", "children")
# "quantized" relatives are compared too: GGUF and bitsandbytes weights are readable, and the
# relation is self-declared (SmolLM2-135M-Instruct, a full fine-tune, declares its base as "quantized").
COMPARED_RELATIONS: tuple[str, ...] = ("finetune", "adapter", "quantized")
CHILD_RELATIONS: tuple[str, ...] = ("adapter", "finetune", "quantized")
MERGE_RELATION = "merge"
BASE_MODEL_TAG = "base_model"
BYTES_PER_GB = 1e9


class RepoKind(StrEnum):
    """How a Hub repository stores its weights, from its file list and config."""

    ADAPTER = "adapter"
    GGUF = "gguf"
    BNB = "bnb"  # bitsandbytes safetensors: readable, dequantized
    QUANTIZED = "quantized"  # any other quantization method: not readable
    FULL = "full"
    UNSUPPORTED = "unsupported"


COMPARABLE_KINDS: tuple[RepoKind, ...] = (RepoKind.ADAPTER, RepoKind.FULL, RepoKind.GGUF, RepoKind.BNB)
BNB_METHOD = "bitsandbytes"


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class Parent:
    """A declared parent of the target, with its Hub popularity and size (``None`` when unreadable)."""

    relation: str  # finetune | adapter | merge | quantized
    repo: str
    downloads: int | None = None
    size_gb: float | None = None


@dataclass(frozen=True)
class Child:
    """A model declaring the target as its base, with its Hub size (``None`` when unreadable)."""

    repo: str
    relation: str  # adapter | finetune | quantized
    downloads: int
    size_gb: float | None = None


@dataclass(frozen=True)
class LineagePair:
    """One comparison to run: ``child`` against ``parent``."""

    parent: str
    child: str
    kind: RepoKind  # of the child: ADAPTER or FULL


@dataclass(frozen=True)
class Skipped:
    """A related repository left out, and why."""

    repo: str
    reason: str


@dataclass
class LineageDiscovery:
    """What the Hub metadata says about the target's lineage."""

    target: str
    kind: RepoKind
    mode: str
    parents: list[Parent] = field(default_factory=list)
    children: list[Child] = field(default_factory=list)
    pairs: list[LineagePair] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PairResult:
    """The static comparison of one pair, or why it failed."""

    pair: LineagePair
    shifts: TokenShiftResult | None = None
    error: str | None = None


@dataclass
class LineageResult:
    """Discovery plus the comparison of every usable pair."""

    meta: ResultMeta
    discovery: LineageDiscovery
    pairs: list[PairResult]


# ----------------------------------------------------------------- discovery
def discover_lineage(
    model_id: str,
    *,
    mode: str = "auto",
    max_variants: int = 5,
    max_size_gb: float = 8.0,
    api: Any = None,
    token: str | None = None,
) -> LineageDiscovery:
    """Find the declared parents and children of a Hub model, and the pairs worth comparing.

    Args:
        model_id: Hub repository id of the target.
        mode: ``auto`` (parents and children), ``parent`` or ``children``.
        max_variants: children kept, the most downloaded first.
        max_size_gb: full-weight children larger than this are skipped.
        api: an ``HfApi``-like object (``model_info``, ``list_models``); None builds an ``HfApi``.
        token: Hub token for private or gated repositories.

    Returns:
        Parents, children, comparable pairs, skipped repositories and notes.

    Raises:
        ValueError: for an unknown ``mode``.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown lineage mode {mode!r}; expected one of {', '.join(MODES)}")
    client = api if api is not None else _default_api()
    cache: dict[str, Any] = {}
    info = _repo_info(client, model_id, token, cache)
    discovery = LineageDiscovery(
        target=model_id, kind=kind_of(info), mode=mode, parents=parents_of(client, info, token, cache)
    )
    candidates: list[tuple[str, str]] = []
    if mode in PARENT_MODES:
        candidates += parent_candidates(discovery)
    if mode in CHILDREN_MODES:
        discovery.children = children_of(client, model_id, max_variants, token, cache)
        candidates += child_candidates(discovery)
    usable_pairs(client, info, discovery, candidates, max_size_gb, token, cache)
    return discovery


def _default_api() -> Any:
    """A real Hub client (imported lazily: only discovery talks to the Hub API)."""
    from huggingface_hub import HfApi

    return HfApi()


def _repo_info(api: Any, repo: str, token: str | None, cache: dict[str, Any]) -> Any:
    """``model_info`` of ``repo`` with file sizes, fetched once per discovery and cached.

    Raises:
        Exception: whatever the Hub client raises; a failure is not cached.
    """
    if repo not in cache:
        cache[repo] = api.model_info(repo, files_metadata=True, token=token)
    return cache[repo]


def _repo_stats(api: Any, repo: str, token: str | None, cache: dict[str, Any]) -> tuple[int | None, float | None]:
    """The ``(downloads, size in GB)`` of ``repo`` from its Hub metadata, ``(None, None)`` when unreadable."""
    result: tuple[int | None, float | None] = (None, None)
    try:
        info = _repo_info(api, repo, token, cache)
    except Exception:  # noqa: BLE001 - metadata for one relative must not abort discovery
        result = (None, None)
    else:
        downloads = getattr(info, "downloads", None)
        result = (int(downloads) if downloads is not None else None, size_gb(info))
    return result


def parents_of(api: Any, info: Any, token: str | None, cache: dict[str, Any]) -> list[Parent]:
    """Parents from ``base_model:<relation>:<repo>`` tags, else from the card's ``base_model``.

    Each parent is enriched with its Hub downloads and size (``None`` where the repository
    cannot be read).
    """
    declared: list[tuple[str, str]] = []
    for tag in info.tags or []:
        parts = tag.split(":", 2)
        if len(parts) == 3 and parts[0] == BASE_MODEL_TAG:
            declared.append((parts[1], parts[2]))
    card = _card_value(info.card_data, BASE_MODEL_TAG) if not declared else None
    if card:
        declared = [("finetune", repo) for repo in ([card] if isinstance(card, str) else card)]
    result = []
    for relation, repo in declared:
        downloads, size = _repo_stats(api, repo, token, cache)
        result.append(Parent(relation=relation, repo=repo, downloads=downloads, size_gb=size))
    return result


def _card_value(card_data: Any, key: str) -> Any:
    """One field of the model card metadata (a ``ModelCardData``, a dict, or None)."""
    result = None
    if card_data is not None:
        result = card_data.get(key) if hasattr(card_data, "get") else getattr(card_data, key, None)
    return result


def children_of(api: Any, repo: str, limit: int, token: str | None, cache: dict[str, Any]) -> list[Child]:
    """Adapters and fine-tunes of ``repo``, the ``limit`` most downloaded, each with its Hub size."""
    found: dict[str, tuple[str, int]] = {}
    for relation in CHILD_RELATIONS:
        listed = api.list_models(
            filter=f"{BASE_MODEL_TAG}:{relation}:{repo}", sort="downloads", limit=limit, token=token
        )
        for model in listed:
            found[model.id] = (relation, getattr(model, "downloads", 0) or 0)
    top = sorted(found.items(), key=lambda item: -item[1][1])[:limit]
    return [
        Child(repo=child, relation=relation, downloads=downloads, size_gb=_repo_stats(api, child, token, cache)[1])
        for child, (relation, downloads) in top
    ]


def kind_of(info: Any) -> RepoKind:
    """Classify a repository from its file list and (partial) config."""
    files = {sibling.rfilename for sibling in info.siblings or []}
    config = info.config or {}
    result = RepoKind.UNSUPPORTED
    if ADAPTER_CONFIG in files:
        result = RepoKind.ADAPTER
    elif any(name.endswith(".gguf") for name in files):
        result = RepoKind.GGUF
    elif (config.get("quantization_config") or {}).get("quant_method") == BNB_METHOD:
        result = RepoKind.BNB
    elif config.get("quantization_config"):
        result = RepoKind.QUANTIZED
    elif any(name.endswith(".safetensors") for name in files):
        result = RepoKind.FULL
    return result


def size_gb(info: Any) -> float:
    """Size of what would be read, in GB: the safetensors files, or the GGUF file that would be chosen."""
    siblings = info.siblings or []
    chosen_gguf = choose_gguf(sibling.rfilename for sibling in siblings)
    wanted = (lambda name: name == chosen_gguf) if chosen_gguf else (lambda name: name.endswith(".safetensors"))
    return sum(sibling.size or 0 for sibling in siblings if wanted(sibling.rfilename)) / BYTES_PER_GB


def parent_candidates(discovery: LineageDiscovery) -> list[tuple[str, str]]:
    """The (parent, target) pair to compare, if a usable parent is declared.

    Only ``finetune`` / ``adapter`` / ``quantized`` parents are usable; for a
    merge (several parents, one a ``merge``), the first usable parent stands in
    for all, else the first listed one.
    Every other declared parent is recorded as skipped.
    """
    parents = discovery.parents
    usable = [parent for parent in parents if parent.relation in COMPARED_RELATIONS]
    if len(parents) > 1 and any(parent.relation == MERGE_RELATION for parent in parents):
        discovery.notes.append("Merged model: compared with the first listed parent only.")
        usable = usable or [parents[0]]
    chosen = usable[0] if usable else None
    for parent in parents:
        if parent is not chosen:
            reason = "only one parent is compared" if chosen else f"{parent.relation} parents are not compared"
            discovery.skipped.append(Skipped(repo=parent.repo, reason=reason))
    return [(chosen.repo, discovery.target)] if chosen else []


def child_candidates(discovery: LineageDiscovery) -> list[tuple[str, str]]:
    """The (target, child) pairs of every listed child."""
    return [(discovery.target, child.repo) for child in discovery.children]


def usable_pairs(
    api: Any,
    info: Any,
    discovery: LineageDiscovery,
    candidates: list[tuple[str, str]],
    max_size_gb: float,
    token: str | None,
    cache: dict[str, Any],
) -> None:
    """Keep comparable candidates as pairs of ``discovery``; record why the others are skipped."""
    for parent, child in candidates:
        skip_reason, kind = _classify_child(api, info, discovery.target, child, max_size_gb, token, cache)
        if skip_reason is None and kind is not None:
            discovery.pairs.append(LineagePair(parent=parent, child=child, kind=kind))
        else:
            discovery.skipped.append(Skipped(repo=child, reason=skip_reason or "not comparable"))


def _classify_child(
    api: Any, info: Any, target: str, child: str, max_size_gb: float, token: str | None, cache: dict[str, Any]
) -> tuple[str | None, RepoKind | None]:
    """The kind of ``child`` and, when it cannot be compared, the reason.

    Returns:
        (skip reason or None, kind or None when the metadata could not be read).
    """
    reason: str | None = None
    kind: RepoKind | None = None
    try:
        child_info = info if child == target else _repo_info(api, child, token, cache)
        kind = kind_of(child_info)
        if kind is RepoKind.QUANTIZED:
            reason = "quantization method not supported (GGUF and bitsandbytes are)"
        elif kind not in COMPARABLE_KINDS:
            reason = "no safetensors weights or LoRA adapter"
        elif kind is not RepoKind.ADAPTER and size_gb(child_info) > max_size_gb:
            reason = f"{size_gb(child_info):.1f} GB > max size {max_size_gb:g} GB"
    except Exception as error:  # noqa: BLE001 - one unreadable repository must not abort discovery
        reason = f"cannot read Hub metadata: {error.__class__.__name__}: {error}"
    return reason, kind


# ----------------------------------------------------------------- analysis
def analyse_lineage(
    discovery: LineageDiscovery,
    *,
    decoder_factory: Callable[[str], TokenDecoder],
    n: int = 10,
    top: int = 12,
    rank: int = 6,
    null_draws: int = DEFAULT_NULL_DRAWS,
    token_hf: str | None = None,
    repo_factory: RepoFactory = HubRepo,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
) -> LineageResult:
    """Run the static token-shift comparison on every pair of ``discovery``.

    A failing pair is recorded with its error and the others still run.

    Args:
        discovery: the result of :func:`discover_lineage`.
        decoder_factory: builds the tokenizer of a parent repository.
        n: heaviest updates projected per pair.
        top: tokens listed per axis end.
        rank: rank of the randomized SVD of full updates.
        null_draws: random directions of each null baseline.
        token_hf: Hub token for private or gated repositories.
        repo_factory: builds Hub access objects (tests inject a fake).
        sink: receives progress events.
        token: checked between pairs and tensors, so a front-end can cancel.

    Returns:
        The discovery and one :class:`PairResult` per pair.

    Raises:
        JobCancelled: when ``token`` is cancelled.
    """
    settings = _PairSettings(decoder_factory, n, top, rank, null_draws, token_hf, repo_factory, sink, token)
    results = []
    for index, pair in enumerate(discovery.pairs):
        if token is not None:
            token.check()
        report(sink, Stage.LINEAGE, f"Comparing {pair.parent} -> {pair.child}", index, len(discovery.pairs))
        results.append(_analyse_pair(pair, settings))
    report(sink, Stage.LINEAGE, "Lineage comparison done", len(discovery.pairs), len(discovery.pairs))
    meta = ResultMeta(
        analysis="lineage",
        inputs={"model": discovery.target},
        parameters={"mode": discovery.mode, "n": n, "top": top, "rank": rank, "null_draws": null_draws},
        notes=list(discovery.notes),
        warnings=[f"{result.pair.child}: {result.error}" for result in results if result.error],
    )
    return LineageResult(meta=meta, discovery=discovery, pairs=results)


@dataclass(frozen=True)
class _PairSettings:
    """Everything a pair comparison needs besides the pair."""

    decoder_factory: Callable[[str], TokenDecoder]
    n: int
    top: int
    rank: int
    null_draws: int
    token_hf: str | None
    repo_factory: RepoFactory
    sink: ProgressSink
    token: CancelToken | None


def _analyse_pair(pair: LineagePair, settings: _PairSettings) -> PairResult:
    """Compare one pair; any failure but a cancellation is recorded, not raised."""
    result = PairResult(pair=pair)
    try:
        base, tuned = _open_pair(pair, settings)
        result.shifts = token_shifts(
            base,
            tuned,
            settings.decoder_factory(pair.parent),
            inputs={"base": pair.parent, "tuned": pair.child},
            n=settings.n,
            top=settings.top,
            rank=settings.rank,
            null_draws=settings.null_draws,
            sink=settings.sink,
            token=settings.token,
        )
    except JobCancelled:
        raise
    except Exception as error:  # noqa: BLE001 - one odd repository must not abort the others
        result.error = f"{error.__class__.__name__}: {error}"
    return result


def _open_pair(pair: LineagePair, settings: _PairSettings) -> tuple[TensorSource, TensorSource]:
    """Open the parent, and the child (an adapter is applied to that parent)."""
    hub_token, factory = settings.token_hf, settings.repo_factory
    base = open_model(detect(pair.parent, token=hub_token, repo_factory=factory), token=hub_token, repo_factory=factory)
    child = detect(pair.child, token=hub_token, repo_factory=factory)
    adapter_base = pair.parent if child.kind is ModelKind.ADAPTER else None
    return base, open_model(child, base=adapter_base, token=hub_token, repo_factory=factory, gguf_file=GGUF_AUTO)
