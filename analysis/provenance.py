"""Provenance: is a model derived from another one?

Weight-level test focused on the embeddings and the first transformer layers.
Independently trained models with the same architecture have near-orthogonal
corresponding weights (random init in high dimension), while a fine-tune keeps
a cosine above 0.99; early layers and embeddings move least, so they are the
best fingerprint.

Given a SUSPECT (Hub repository id or local folder):

1. Candidate parents
     - explicit candidates;
     - declared lineage (``base_model:*`` Hub tags walked upwards, and the base a
       LoRA adapter declares);
     - Hub search: models of the same ``model_type`` ranked by downloads (only
       "root" models, without ``base_model`` tags, unless ``include_derived``);
   kept only if hidden size, MLP size and head layout match. Quantized models
   (GGUF, bitsandbytes) are dequantized; quantization lowers raw cosines only
   slightly (about 0.99+ at 4 bits), so the verdict thresholds still apply.

2. STAGE 1 - screen (cheap, a few rows per tensor): raw cosine of the first rows
   of layer-0 q/k/v/gate/up, and Pearson r of the input norm gains.

3. STAGE 2 - deep comparison of the best candidates (full layer 0..L-1 tensors
   and a slice of the embedding matrix), each metric against a shuffled null:

     raw_cos   cosine of corresponding matrices (same basis), null = rows shuffled
     rel_d     ‖W_s - W_c‖ / ‖W_c‖
     nn_row    mean best-match cosine of MLP neurons (invariant to neuron
               permutation), null = candidate columns shuffled
     spectrum  singular-value similarity with norm gains folded (invariant to
               rotation and permutation, weak)
     token circuits, invariant to a residual-stream rotation AND to neuron/head
     permutations, compared by Pearson r off-diagonal against token-permuted nulls:
       emb_gram  E Eᵀ on a token slice (excluded from the verdict: token semantics
                 make unrelated models correlate too)
       qk        layer-0 attention scores token x token (position 0, summed over heads)
       ov        layer-0 value/output path projected back on the tokens
       mlp       layer-0 MLP applied to the embeddings, projected back on the tokens

   -> verdict: IDENTICAL / DERIVED (same basis) / RELATED / DERIVED (transformed
      basis) / NO WEIGHT-LEVEL EVIDENCE. Thresholds are heuristics: read the
      numbers against their nulls and against an unrelated same-size candidate.

Every random draw of stage 2 comes from ONE generator seeded with 0, consumed in
a fixed order, so results are reproducible run to run. Distilled models share no
weights with their teacher: only behavioural fingerprinting can detect them.
"""

from __future__ import annotations

import contextlib
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from analysis.stats import (
    NullComparison,
    cosine,
    nearest_row_similarity,
    pearson,
    relative_distance,
    row_shuffle_nulls,
    seeded_generator,
    spectrum_similarity,
)
from models import (
    GGUF_AUTO,
    Architecture,
    Location,
    ModelError,
    ModelKind,
    ModelRef,
    Rows,
    TensorSource,
    UnsupportedModel,
    detect,
    open_model,
)
from models.detect import RepoFactory
from models.hub import HubRepo
from models.source import slice_rows
from progress import CancelToken, JobCancelled, ProgressSink, Stage, null_sink, report
from results import ResultMeta

# ----------------------------------------------------------------- verdict thresholds
IDENTICAL_MIN_RAW_COS = 0.99999  # median raw cosine strictly above this ...
IDENTICAL_MAX_REL_D = 1e-5  # ... and max relative distance strictly below this
SAME_BASIS_MIN_RAW_COS = 0.8
RELATED_MIN_RAW_COS = 0.3
CIRCUITS_MIN_R = 0.8
CIRCUITS_MIN_Z = 10
NEURONS_MIN_NN = 0.8
NEURONS_MIN_Z = 10
RANK_IDENTICAL, RANK_SAME_BASIS, RANK_RELATED, RANK_TRANSFORMED, RANK_NONE = 5, 4, 3, 2, 0
BEST_MATCH_MIN_RANK = RANK_TRANSFORMED
VERDICT_IDENTICAL = "IDENTICAL weights (re-upload / rename)"
VERDICT_SAME_BASIS = "DERIVED, same basis (fine-tune / continued training)"
VERDICT_RELATED = "RELATED (heavy continued training, merge or partial reuse)"
VERDICT_TRANSFORMED = "DERIVED, transformed basis (rotation / permutation suspected)"
VERDICT_PERMUTED = "DERIVED, permuted MLP neurons suspected"
VERDICT_NONE = "NO WEIGHT-LEVEL EVIDENCE (independent, or distilled)"

# ----------------------------------------------------------------- method settings
ROW_NULL_DRAWS = 3  # row-shuffled raw-cosine nulls per tensor
NEURON_NULL_DRAWS = 3  # column-shuffled nn_row nulls per gate/up tensor
CIRCUIT_NULL_DRAWS = 10  # token-permuted nulls per circuit
NEURON_SAMPLE = 512  # suspect neurons sampled by nn_row
MIN_CIRCUIT_TOKENS = 64
RMS_EPSILON = 1e-6
DECLARED_DEPTH = 4  # base_model generations walked upwards
SEARCH_OVERFETCH = 5  # roots are rare among search hits: fetch more, keep `limit`

ATTENTION_KINDS: tuple[str, ...] = ("q", "k", "v", "o")
MATRIX_KINDS: tuple[str, ...] = ("q", "k", "v", "o", "gate", "up", "down")
INPUT_GAIN_KINDS: tuple[str, ...] = ("q", "k", "v")
NEURON_KINDS: tuple[str, ...] = ("gate", "up")
SCREEN_KINDS: tuple[str, ...] = ("q", "k", "v", "gate", "up")
FINGERPRINT_NEURONS = "up"  # layer-0 tensor whose nn_row enters the verdict
EMBEDDING_SUFFIXES: tuple[str, ...] = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")
CIRCUIT_NAMES: tuple[str, ...] = ("emb_gram", "qk", "ov", "mlp")
CIRCUITS_OUTSIDE_VERDICT: tuple[str, ...] = ("emb_gram",)
INPUT_NORM, POST_ATTENTION_NORM = "input_layernorm", "post_attention_layernorm"
ORIGIN_EXPLICIT, ORIGIN_SEARCH = "explicit", "search"
TAG_BASE_MODEL = "base_model"
TAG_GGUF = "gguf"

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
}

Gains = tuple[torch.Tensor, torch.Tensor]


# ======================================================================= results
@dataclass
class SuspectSummary:
    """The analysed model."""

    ref: str
    label: str
    architecture: str
    declared_bases: list[str] = field(default_factory=list)


@dataclass
class DeclaredLink:
    """One ``child --relation--> parent`` edge of the declared lineage."""

    relation: str
    parent: str
    child: str


@dataclass
class SkippedCandidate:
    """A gathered candidate that was not screened, and why."""

    ref: str
    origin: str
    reason: str


@dataclass
class ScreenRow:
    """Stage-1 result of one candidate (``error`` set when it could not be screened)."""

    ref: str
    origin: str
    is_root: bool
    raw_cos: float | None = None
    norm_r: float | None = None
    error: str | None = None


@dataclass
class TensorRow:
    """Stage-2 metrics of one projection (all None and ``note`` set on a shape mismatch)."""

    kind: str
    raw_cos: float | None = None
    raw_null: float | None = None
    rel_d: float | None = None
    nn_row: float | None = None
    nn_null: float | None = None
    nn_z: float | None = None
    spectrum: float | None = None
    note: str | None = None


@dataclass
class LayerComparison:
    """Norm-gain correlations and projection metrics of one layer."""

    layer: int
    input_norm_r: float
    post_attention_norm_r: float
    tensors: list[TensorRow]


@dataclass
class EmbeddingComparison:
    """Raw similarity of the embedding slice ``start .. start + rows - 1``."""

    start: int
    rows: int
    raw_cos: float
    raw_null: float
    rel_d: float


@dataclass
class CircuitComparison:
    """Off-diagonal Pearson r of one token circuit against its token-permuted nulls."""

    name: str
    r: float
    null_mean: float
    null_sd: float
    z: float
    in_verdict: bool


@dataclass
class DeepSummary:
    """The numbers the verdict is decided from."""

    raw_med: float
    max_rel_d: float
    inv_mean: float | None
    inv_zmin: float
    nn: float | None
    nn_z: float


@dataclass
class DeepComparison:
    """Stage-2 result of one candidate (``error`` set when it failed)."""

    ref: str
    origin: str
    layers: list[LayerComparison] = field(default_factory=list)
    embeddings: EmbeddingComparison | None = None
    circuits: list[CircuitComparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: DeepSummary | None = None
    rank: int | None = None
    verdict: str | None = None
    error: str | None = None


@dataclass
class RankedCandidate:
    """One line of the final ranking."""

    ref: str
    origin: str
    rank: int
    verdict: str
    raw_med: float
    inv_mean: float | None


@dataclass
class ProvenanceResult:
    """Everything a provenance run found; warnings and notes live in ``meta``."""

    meta: ResultMeta
    suspect: SuspectSummary
    gathered: int
    origins: dict[str, str]
    declared: list[DeclaredLink]
    compatible: int
    skipped: list[SkippedCandidate]
    screen: list[ScreenRow]
    deep: list[DeepComparison]
    ranking: list[RankedCandidate]
    best: str | None

    @property
    def warnings(self) -> list[str]:
        """The run's warnings (same list as ``meta.warnings``)."""
        return self.meta.warnings


@dataclass(frozen=True)
class ProvenanceSettings:
    """Every tunable of a run (recorded in the result's parameters)."""

    candidates: tuple[str, ...] = ()
    search: bool = True
    search_text: str | None = None
    search_limit: int = 40
    include_derived: bool = False
    top_k: int = 3
    deep_all: bool = False
    layers: int = 1
    slice_rows: int = 64
    emb_start: int = 1000
    emb_rows: int = 1024
    workers: int = 6
    revision: str | None = None
    suspect_gguf_file: str | None = None  # file of a GGUF suspect holding several (None: the most precise)
    suspect_base: str | None = None  # base an adapter suspect is applied to, instead of the declared one


@dataclass
class CandidatePool:
    """Gathered candidates: ref -> origin tag, search roots, declared lineage."""

    origins: dict[str, str]
    roots: dict[str, bool]
    declared: list[DeclaredLink]
    declared_direct: list[str]


@dataclass
class CandidateCheck:
    """A candidate after detection: ``reason`` is set when it is skipped."""

    ref: str
    origin: str
    model: ModelRef | None = None
    architecture: Architecture | None = None
    reason: str | None = None


# ======================================================================= public API
def find_provenance(
    suspect: str,
    *,
    candidates: Sequence[str] = (),
    search: bool = True,
    search_text: str | None = None,
    search_limit: int = 40,
    include_derived: bool = False,
    top_k: int = 3,
    deep_all: bool = False,
    layers: int = 1,
    slice_rows: int = 64,
    emb_start: int = 1000,
    emb_rows: int = 1024,
    workers: int = 6,
    hf_token: str | None = None,
    revision: str | None = None,
    api: Any = None,
    repo_factory: RepoFactory = HubRepo,
    sink: ProgressSink = null_sink,
    token: CancelToken | None = None,
    suspect_gguf_file: str | None = None,
    suspect_base: str | None = None,
) -> ProvenanceResult:
    """Find which candidate models the suspect's weights derive from.

    Args:
        suspect: Hub repository id or local folder (full weights or LoRA adapter).
        candidates: explicit candidate parents (ids or folders).
        search: also search the Hub for same-type models.
        search_text: free-text Hub search instead of the ``model_type`` filter.
        search_limit: maximum number of search hits kept.
        include_derived: also keep search hits that declare a base model.
        top_k: screened candidates sent to the deep comparison (plus every
            explicit or declared one).
        deep_all: deep-compare every screened candidate.
        layers: layers 0..layers-1 compared in depth.
        slice_rows: rows per tensor in the screen.
        emb_start: first token of the embedding slice.
        emb_rows: tokens in the embedding slice.
        workers: threads for configuration checks and screening.
        hf_token: Hub token for private or gated repositories.
        revision: Hub revision of the suspect.
        api: Hub API object (``model_info``, ``list_models``); ``HfApi()`` when None.
        repo_factory: builds Hub access objects (tests inject a fake).
        sink: receives progress events.
        token: polled between candidates and tensors to cancel the run.
        suspect_gguf_file: which file of a GGUF suspect holding several (None: the most precise).
        suspect_base: base model an adapter suspect is applied to, instead of the one it declares.

    Returns:
        The full result tree (plain values only).

    Raises:
        ModelError: when the suspect cannot be located or read.
        UnsupportedModel: when the suspect's format has no reader yet.
        JobCancelled: when ``token`` was cancelled.
    """
    settings = ProvenanceSettings(
        candidates=tuple(candidates),
        search=search,
        search_text=search_text,
        search_limit=search_limit,
        include_derived=include_derived,
        top_k=top_k,
        deep_all=deep_all,
        layers=layers,
        slice_rows=slice_rows,
        emb_start=emb_start,
        emb_rows=emb_rows,
        workers=workers,
        revision=revision,
        suspect_gguf_file=suspect_gguf_file,
        suspect_base=suspect_base,
    )
    run = ProvenanceRun(settings, hf_token, api, repo_factory, sink, token)
    return run.analyse(suspect)


class ProvenanceRun:
    """One provenance run: settings plus the runtime handles every stage needs."""

    def __init__(
        self,
        settings: ProvenanceSettings,
        hf_token: str | None,
        api: Any,
        repo_factory: RepoFactory,
        sink: ProgressSink,
        token: CancelToken | None,
    ) -> None:
        self.settings = settings
        self.hf_token = hf_token
        self.repo_factory = repo_factory
        self.sink = sink
        self.token = token
        self._api = api

    def analyse(self, suspect: str) -> ProvenanceResult:
        """Run every stage on ``suspect`` (see :func:`find_provenance`)."""
        meta = ResultMeta(analysis="provenance", inputs={"suspect": suspect}, parameters=asdict(self.settings))
        self.progress("Opening the suspect")
        model, view = self.open_suspect(suspect, meta)
        self.progress("Gathering candidates")
        pool = gather_candidates(model, view.architecture, self)
        checks = check_candidates(pool, view.architecture, self)
        compatible = [check for check in checks if check.reason is None]
        screen_rows, views = screen_candidates(view, compatible, pool, self)
        chosen = select_for_deep(screen_rows, pool.origins, self.settings)
        deep = deep_compare_all(view, [(views[ref], pool.origins[ref]) for ref in chosen], self)
        ranking = rank_candidates(deep)
        best = ranking[0].ref if ranking and ranking[0].rank >= BEST_MATCH_MIN_RANK else None
        meta.warnings.extend(run_warnings(best, pool.declared_direct, model, deep))
        meta.notes.extend(run_notes(compatible, deep, best))
        self.progress("Provenance analysis complete", len(chosen), len(chosen))
        return ProvenanceResult(
            meta=meta,
            suspect=SuspectSummary(suspect, model.label, view.architecture.describe(), pool.declared_direct),
            gathered=len(pool.origins),
            origins=pool.origins,
            declared=pool.declared,
            compatible=len(compatible),
            skipped=[SkippedCandidate(c.ref, c.origin, c.reason) for c in checks if c.reason is not None],
            screen=screen_rows,
            deep=deep,
            ranking=ranking,
            best=best,
        )

    def open_suspect(self, suspect: str, meta: ResultMeta) -> tuple[ModelRef, ModelView]:
        """Detect and open the suspect; adapters are read as base + ΔW.

        Raises:
            ModelError: when the suspect cannot be located or read.
            UnsupportedModel: when its format has no reader yet.
        """
        revision = self.settings.revision
        try:
            model = detect(suspect, self.hf_token, revision, self.repo_factory)
            source = open_model(
                model,
                base=self.settings.suspect_base,
                token=self.hf_token,
                revision=revision,
                repo_factory=self.repo_factory,
                gguf_file=self.settings.suspect_gguf_file or GGUF_AUTO,
            )
        except UnsupportedModel as error:
            raise UnsupportedModel(f"Suspect {suspect} cannot be analysed: {error}") from error
        view = ModelView(suspect, source)
        if model.kind is ModelKind.ADAPTER:
            meta.notes.append(
                f"The suspect is a LoRA adapter: analysed as its base ({model.base_model}) with the adapter applied."
            )
        if model.kind in (ModelKind.GGUF, ModelKind.BNB):
            meta.notes.append(
                f"The suspect is quantized ({model.label}): its weights are dequantized before comparison; "
                "quantization lowers raw cosines slightly, the verdict thresholds remain valid."
            )
        return model, view

    @property
    def api(self) -> Any:
        """The Hub API object, created on first use."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def check(self) -> None:
        """Raise :class:`JobCancelled` when the run was cancelled."""
        if self.token is not None:
            self.token.check()

    def progress(self, message: str, current: int | None = None, total: int | None = None) -> None:
        """Report one progress event."""
        report(self.sink, Stage.PROVENANCE, message, current, total)


# ======================================================================= model view
class ModelView:
    """Cached, per-tensor access to one model's llama-style weights.

    Fused ``qkv_proj`` and ``gate_up_proj`` are split with the config sizes.
    """

    def __init__(self, ref: str, source: TensorSource, architecture: Architecture | None = None) -> None:
        """Bind a reference to its tensor source and architecture.

        Raises:
            ModelError: when the configuration lacks the needed dimensions.
        """
        self.ref = ref
        self.source = source
        self.architecture = architecture or Architecture.from_config(source.config())
        self._cache: dict[tuple[str, Rows | None], torch.Tensor] = {}

    def mat(self, layer: int, kind: str, rows: Rows | None = None) -> torch.Tensor:
        """The ``kind``_proj weight of ``layer`` (q/k/v/o/gate/up/down), optionally a row range.

        Raises:
            KeyError: when the projection cannot be found.
        """
        result = self._attention_mat(layer, kind, rows) if kind in ATTENTION_KINDS else self._mlp_mat(layer, kind, rows)
        if result is None:
            raise KeyError(f"{kind}_proj of layer {layer} not found in {self.ref}")
        return result

    def norm(self, layer: int, which: str) -> torch.Tensor:
        """Gain vector of norm ``which`` in ``layer``.

        Raises:
            KeyError: when the norm cannot be found.
        """
        name = self.source.find(f"layers.{layer}.{which}.weight")
        if not name:
            raise KeyError(f"{which} of layer {layer} not found in {self.ref}")
        return self._load(name)

    def embedding_name(self) -> str:
        """Tensor name of the token embedding matrix.

        Raises:
            KeyError: when there is none.
        """
        result = next((name for name in map(self.source.find, EMBEDDING_SUFFIXES) if name), None)
        if result is None:
            raise KeyError(f"embedding matrix not found in {self.ref}")
        return result

    def embedding_rows(self) -> int:
        """Number of rows (tokens) of the embedding matrix."""
        return self.source.shape(self.embedding_name())[0]

    def embedding(self, start: int, stop: int) -> torch.Tensor:
        """Embedding rows ``start .. stop - 1``."""
        return self._load(self.embedding_name(), (start, stop))

    def _attention_mat(self, layer: int, kind: str, rows: Rows | None) -> torch.Tensor | None:
        """Attention projection, split out of a fused ``qkv_proj`` when needed; None if absent."""
        arch = self.architecture
        q_size, kv_size = arch.n_heads * arch.head_dim, arch.n_kv * arch.head_dim
        result = None
        name = self.source.find(f"layers.{layer}.self_attn.{kind}_proj.weight")
        fused = None if name else self.source.find(f"layers.{layer}.self_attn.qkv_proj.weight")
        if name:
            result = self._load(name, rows)
        elif fused and kind != "o":
            bounds = {"q": (0, q_size), "k": (q_size, q_size + kv_size), "v": (q_size + kv_size, q_size + 2 * kv_size)}
            low, high = bounds[kind]
            result = slice_rows(self._load(fused)[low:high], rows)
        return result

    def _mlp_mat(self, layer: int, kind: str, rows: Rows | None) -> torch.Tensor | None:
        """MLP projection, split out of a fused ``gate_up_proj`` when needed; None if absent."""
        result = None
        name = self.source.find(f"layers.{layer}.mlp.{kind}_proj.weight")
        fused = None if name else self.source.find(f"layers.{layer}.mlp.gate_up_proj.weight")
        if name:
            result = self._load(name, rows)
        elif fused and kind != "down":
            weight = self._load(fused)
            inter = self.architecture.intermediate or weight.shape[0] // 2
            weight = weight[:inter] if kind == "gate" else weight[inter : 2 * inter]
            result = slice_rows(weight, rows)
        return result

    def _load(self, name: str, rows: Rows | None = None) -> torch.Tensor:
        """Load a tensor (or row range) through the cache."""
        key = (name, rows)
        if key not in self._cache:
            self._cache[key] = self.source.load(name, rows)
        return self._cache[key]


# ======================================================================= candidates
def gather_candidates(model: ModelRef, architecture: Architecture, run: ProvenanceRun) -> CandidatePool:
    """Collect explicit, declared and searched candidates (the suspect itself excluded)."""
    settings = run.settings
    origins = dict.fromkeys(settings.candidates, ORIGIN_EXPLICIT)
    links = declared_links(model, run)
    for link in links:
        origins.setdefault(link.parent, f"declared {link.relation}")
    roots: dict[str, bool] = {}
    if settings.search:
        hits = hub_search(
            run.api,
            architecture.model_type,
            settings.search_text,
            settings.search_limit,
            settings.include_derived,
            run.hf_token,
        )
        for repo_id, is_root in hits:
            origins.setdefault(repo_id, ORIGIN_SEARCH)
            roots[repo_id] = is_root
    origins.pop(model.ref, None)
    direct = [link.parent for link in links if link.child == model.ref]
    return CandidatePool(origins=origins, roots=roots, declared=links, declared_direct=direct)


def declared_links(model: ModelRef, run: ProvenanceRun) -> list[DeclaredLink]:
    """Declared lineage: an adapter's own base, then the Hub ``base_model`` chain."""
    links = []
    if model.kind is ModelKind.ADAPTER and model.base_model:
        links.append(DeclaredLink("adapter", model.base_model, model.ref))
    if model.location is Location.HUB:
        known = {link.parent for link in links}
        chain = declared_chain(run.api, model.ref, run.hf_token)
        links.extend(link for link in chain if link.parent not in known)
    return links


def declared_chain(api: Any, repo: str, token: str | None, depth: int = DECLARED_DEPTH) -> list[DeclaredLink]:
    """Walk ``base_model:<relation>:<parent>`` tags upwards, breadth-first, without duplicates."""
    links: list[DeclaredLink] = []
    current, seen = [repo], {repo}
    for _ in range(depth):
        following = []
        for child in current:
            for relation, parent in _tagged_parents(api, child, token):
                if parent not in seen:
                    seen.add(parent)
                    links.append(DeclaredLink(relation, parent, child))
                    following.append(parent)
        current = following
    return links


def _tagged_parents(api: Any, repo: str, token: str | None) -> list[tuple[str, str]]:
    """(relation, parent) pairs from a repository's tags; empty when its info is unavailable."""
    tags: list[str] = []
    # A missing or private parent only ends the walk there.
    with contextlib.suppress(Exception):
        tags = list(api.model_info(repo, token=token).tags or [])
    parts = [tag.split(":", 2) for tag in tags]
    return [(part[1], part[2]) for part in parts if len(part) == 3 and part[0] == TAG_BASE_MODEL]


def hub_search(
    api: Any,
    model_type: str | None,
    search_text: str | None,
    limit: int,
    include_derived: bool,
    token: str | None,
) -> list[tuple[str, bool]]:
    """Search the Hub by ``model_type`` (or free text), by downloads, GGUF repositories excluded.

    Returns:
        Up to ``limit`` (repository id, is_root) pairs; root = no ``base_model`` tag.
    """
    query: dict[str, Any] = {
        "sort": "downloads",
        "limit": limit if include_derived else limit * SEARCH_OVERFETCH,
        "token": token,
    }
    if search_text:
        query["search"] = search_text
    if model_type and not search_text:
        query["filter"] = model_type
    try:
        hits = api.list_models(expand=["tags", "downloads"], **query)
    except TypeError:  # older huggingface_hub without ``expand``
        hits = api.list_models(**query)
    found: list[tuple[str, bool]] = []
    for hit in hits:
        tags = hit.tags or []
        is_root = not any(tag.startswith(f"{TAG_BASE_MODEL}:") for tag in tags)
        if TAG_GGUF not in tags and (is_root or include_derived):
            found.append((hit.id, is_root))
        if len(found) >= limit:
            break
    return found


def check_candidates(pool: CandidatePool, suspect: Architecture, run: ProvenanceRun) -> list[CandidateCheck]:
    """Detect every candidate in parallel and decide which ones can be compared."""
    run.check()
    run.progress(f"Checking {len(pool.origins)} candidate configurations")
    with ThreadPoolExecutor(max(1, run.settings.workers)) as executor:
        checks = list(executor.map(lambda ref: check_candidate(ref, pool.origins[ref], suspect, run), pool.origins))
    return checks


def check_candidate(ref: str, origin: str, suspect: Architecture, run: ProvenanceRun) -> CandidateCheck:
    """Detect one candidate; failures and incompatibilities become a skip reason."""
    result = CandidateCheck(ref, origin)
    try:
        model = detect(ref, run.hf_token, None, run.repo_factory)
        result.model = model
        result.architecture = Architecture.from_config(_candidate_config(model, run))
        result.reason = incompatibility(result.architecture, suspect)
    except Exception as error:  # noqa: BLE001 - one unusable candidate must not abort the run
        result.reason = f"{error.__class__.__name__}: {error}"
    return result


def _candidate_config(model: ModelRef, run: ProvenanceRun) -> dict[str, Any]:
    """A candidate's config; adapters and GGUF files without one report what their weights say."""
    config = model.config
    if not config and model.kind in (ModelKind.ADAPTER, ModelKind.GGUF):
        config = open_model(model, token=run.hf_token, repo_factory=run.repo_factory, gguf_file=GGUF_AUTO).config()
    return config


def incompatibility(candidate: Architecture, suspect: Architecture) -> str | None:
    """Why a candidate cannot be compared tensor by tensor with the suspect, or None."""
    return None if suspect.compatible(candidate) else f"incompatible dims ({candidate.describe()})"


# ======================================================================= stage 1
def screen_candidates(
    suspect: ModelView, candidates: list[CandidateCheck], pool: CandidatePool, run: ProvenanceRun
) -> tuple[list[ScreenRow], dict[str, ModelView]]:
    """Stage 1: open and screen every compatible candidate in parallel.

    Returns:
        (screened rows by decreasing raw cosine, then failed rows; ref -> opened view
        of every successfully screened candidate).

    Raises:
        ModelError: when the suspect's own layer-0 tensors cannot be read.
    """
    _warm_screen_cache(suspect, run.settings.slice_rows)
    outcomes: list[tuple[ScreenRow, ModelView | None]] = []
    with ThreadPoolExecutor(max(1, run.settings.workers)) as executor:
        screened = executor.map(lambda check: screen_one(suspect, check, pool, run), candidates)
        for index, outcome in enumerate(screened):
            outcomes.append(outcome)
            run.progress(f"Screened {outcome[0].ref}", index + 1, len(candidates))
    ok = sorted((row for row, _ in outcomes if row.error is None), key=lambda row: -(row.raw_cos or 0.0))
    failed = [row for row, _ in outcomes if row.error is not None]
    views = {row.ref: view for row, view in outcomes if view is not None}
    return ok + failed, views


def _warm_screen_cache(suspect: ModelView, rows: int) -> None:
    """Load the suspect slices before threading, so workers only read the cache.

    Raises:
        ModelError: when the suspect lacks a screened tensor.
    """
    try:
        for kind in SCREEN_KINDS:
            suspect.mat(0, kind, (0, rows))
        suspect.norm(0, INPUT_NORM)
    except KeyError as error:
        raise ModelError(f"The suspect cannot be screened: {error}") from error


def screen_one(
    suspect: ModelView, check: CandidateCheck, pool: CandidatePool, run: ProvenanceRun
) -> tuple[ScreenRow, ModelView | None]:
    """Open one candidate and screen it; errors are recorded in the row, not raised."""
    run.check()
    row = ScreenRow(check.ref, check.origin, pool.roots.get(check.ref, True))
    view = None
    try:
        source = open_model(
            _require(check.model), token=run.hf_token, repo_factory=run.repo_factory, gguf_file=GGUF_AUTO
        )
        candidate = ModelView(check.ref, source, check.architecture)
        row.raw_cos, row.norm_r = screen(suspect, candidate, run.settings.slice_rows)
        view = candidate
    except Exception as error:  # noqa: BLE001 - one unreadable candidate must not abort the run
        row.error = f"{error.__class__.__name__}: {error}"
    return row, view


def _require(model: ModelRef | None) -> ModelRef:
    """The detected model of a compatible candidate (always set once checked)."""
    if model is None:
        raise ModelError("candidate was not detected")
    return model


def screen(suspect: ModelView, candidate: ModelView, rows: int) -> tuple[float, float]:
    """Stage-1 score: mean raw cosine of the first ``rows`` rows of layer-0 tensors, and norm-gain r."""
    values = [cosine(suspect.mat(0, kind, (0, rows)), candidate.mat(0, kind, (0, rows))) for kind in SCREEN_KINDS]
    norm_r = pearson(suspect.norm(0, INPUT_NORM), candidate.norm(0, INPUT_NORM))
    return statistics.fmean(values), norm_r


def select_for_deep(rows: list[ScreenRow], origins: dict[str, str], settings: ProvenanceSettings) -> list[str]:
    """Top ``top_k`` screened candidates (all with ``deep_all``), plus every non-search one."""
    ok = [row.ref for row in rows if row.error is None]
    chosen = ok if settings.deep_all else ok[: settings.top_k]
    return chosen + [ref for ref in ok if origins[ref] != ORIGIN_SEARCH and ref not in chosen]


# ======================================================================= stage 2
def deep_compare_all(
    suspect: ModelView, candidates: list[tuple[ModelView, str]], run: ProvenanceRun
) -> list[DeepComparison]:
    """Stage 2 on each candidate in turn, sharing one generator seeded with 0."""
    generator = seeded_generator()
    results = []
    for index, (candidate, origin) in enumerate(candidates):
        run.check()
        run.progress(f"Deep comparison with {candidate.ref}", index, len(candidates))
        results.append(
            compare_candidate(
                suspect,
                candidate,
                generator,
                layers=run.settings.layers,
                emb_start=run.settings.emb_start,
                emb_rows=run.settings.emb_rows,
                origin=origin,
                token=run.token,
            )
        )
    return results


def compare_candidate(
    suspect: ModelView,
    candidate: ModelView,
    generator: torch.Generator,
    *,
    layers: int = 1,
    emb_start: int = 1000,
    emb_rows: int = 1024,
    origin: str = ORIGIN_EXPLICIT,
    token: CancelToken | None = None,
) -> DeepComparison:
    """Stage-2 comparison of the suspect with one candidate.

    Random draws are taken from ``generator`` in a fixed order: per tensor the
    row-shuffled nulls, then (gate/up) nn_row and its column-shuffled nulls;
    then the embedding null; then the nulls of each circuit.

    Args:
        suspect: the analysed model.
        candidate: the candidate parent.
        generator: shared seeded generator for every null baseline.
        layers: layers 0..layers-1 compared tensor by tensor.
        emb_start: first token of the embedding slice.
        emb_rows: tokens in the embedding slice.
        origin: how the candidate was found.
        token: polled between tensors.

    Returns:
        The comparison; ``error`` is set (and the rest left empty) when it failed.
    """
    try:
        n_layers = min(layers, suspect.architecture.layers or 1, candidate.architecture.layers or 1)
        layer_rows = [compare_layer(suspect, candidate, layer, generator, token) for layer in range(n_layers)]
        embeddings, circuits, notes = compare_embeddings(suspect, candidate, emb_start, emb_rows, generator)
        summary = summarise(layer_rows, circuits)
        rank, text = verdict(summary)
        result = DeepComparison(candidate.ref, origin, layer_rows, embeddings, circuits, notes, summary, rank, text)
    except JobCancelled:
        raise
    except Exception as error:  # noqa: BLE001 - a broken candidate fails alone
        result = DeepComparison(candidate.ref, origin, error=f"{error.__class__.__name__}: {error}")
    return result


def compare_layer(
    suspect: ModelView, candidate: ModelView, layer: int, generator: torch.Generator, token: CancelToken | None
) -> LayerComparison:
    """Norm-gain correlations and every projection's metrics for one layer.

    Raises:
        KeyError: when a projection or norm is missing in either model.
    """
    input_gains = (suspect.norm(layer, INPUT_NORM), candidate.norm(layer, INPUT_NORM))
    post_gains = (suspect.norm(layer, POST_ATTENTION_NORM), candidate.norm(layer, POST_ATTENTION_NORM))
    tensors = []
    for kind in MATRIX_KINDS:
        if token is not None:
            token.check()
        gains = input_gains if kind in INPUT_GAIN_KINDS else post_gains if kind in NEURON_KINDS else None
        tensors.append(compare_tensor(suspect.mat(layer, kind), candidate.mat(layer, kind), kind, gains, generator))
    return LayerComparison(layer, pearson(*input_gains), pearson(*post_gains), tensors)


def compare_tensor(
    first: torch.Tensor, second: torch.Tensor, kind: str, gains: Gains | None, generator: torch.Generator
) -> TensorRow:
    """raw_cos + row-shuffled null, rel_d, nn_row + column-shuffled nulls (gate/up), spectrum.

    Args:
        first: suspect weight.
        second: candidate weight.
        kind: projection kind (q/k/v/o/gate/up/down).
        gains: (suspect, candidate) norm gains folded into the inputs before the spectrum.
        generator: seeded generator for the nulls.
    """
    result = TensorRow(kind, note=f"shape mismatch {tuple(first.shape)} vs {tuple(second.shape)}")
    if first.shape == second.shape:
        raw = cosine(first, second)
        raw_null = statistics.fmean(row_shuffle_nulls(cosine, first, second, generator, ROW_NULL_DRAWS))
        rel_d = relative_distance(first, second)
        neurons = neuron_comparison(first, second, generator) if kind in NEURON_KINDS else None
        folded = (first, second) if gains is None else (first * gains[0], second * gains[1])
        result = TensorRow(
            kind,
            raw_cos=raw,
            raw_null=raw_null,
            rel_d=rel_d,
            nn_row=neurons.value if neurons else None,
            nn_null=neurons.null_mean if neurons else None,
            nn_z=neurons.z if neurons else None,
            spectrum=spectrum_similarity(*folded),
        )
    return result


def neuron_comparison(first: torch.Tensor, second: torch.Tensor, generator: torch.Generator) -> NullComparison:
    """nn_row (permutation-invariant neuron match) against column-shuffled nulls."""
    value = nearest_row_similarity(first, second, generator, NEURON_SAMPLE)
    nulls = [
        nearest_row_similarity(first, second, generator, NEURON_SAMPLE, shuffle_columns=True)
        for _ in range(NEURON_NULL_DRAWS)
    ]
    return NullComparison.build(value, nulls)


def compare_embeddings(
    suspect: ModelView, candidate: ModelView, start: int, rows: int, generator: torch.Generator
) -> tuple[EmbeddingComparison | None, list[CircuitComparison], list[str]]:
    """Embedding slice and layer-0 token circuits; a missing tensor becomes a note, not a failure.

    Returns:
        (embedding comparison or None, circuit comparisons, notes).
    """
    embeddings: EmbeddingComparison | None = None
    circuits: list[CircuitComparison] = []
    notes: list[str] = []
    try:
        n_tokens = min(rows, suspect.embedding_rows() - start, candidate.embedding_rows() - start)
        suspect_vocab, candidate_vocab = suspect.architecture.vocab, candidate.architecture.vocab
        if suspect_vocab != candidate_vocab:
            notes.append(
                f"vocab size differs ({suspect_vocab} vs {candidate_vocab}): "
                f"token alignment assumed for ids < {start + n_tokens}"
            )
        if n_tokens >= MIN_CIRCUIT_TOKENS:
            suspect_slice, candidate_slice = (
                suspect.embedding(start, start + n_tokens),
                candidate.embedding(start, start + n_tokens),
            )
            embeddings = compare_embedding_slice(suspect_slice, candidate_slice, start, generator)
            circuits = compare_circuits(
                token_circuits(suspect, suspect_slice), token_circuits(candidate, candidate_slice), generator
            )
        else:
            notes.append("embedding slice too small: token circuits skipped (lower emb_start)")
    except KeyError as error:
        notes.append(f"token circuits skipped: {error}")
    return embeddings, circuits, notes


def compare_embedding_slice(
    first: torch.Tensor, second: torch.Tensor, start: int, generator: torch.Generator
) -> EmbeddingComparison:
    """Raw cosine of two embedding slices, against one row-shuffled null."""
    raw_null = row_shuffle_nulls(cosine, first, second, generator, 1)[0]
    return EmbeddingComparison(
        start=start,
        rows=first.shape[0],
        raw_cos=cosine(first, second),
        raw_null=raw_null,
        rel_d=relative_distance(first, second),
    )


def compare_circuits(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], generator: torch.Generator
) -> list[CircuitComparison]:
    """Compare every token circuit, in :data:`CIRCUIT_NAMES` order."""
    results = []
    for name in CIRCUIT_NAMES:
        value, nulls = invariant_comparison(first[name], second[name], generator)
        null = NullComparison.build(value, nulls)
        results.append(
            CircuitComparison(name, value, null.null_mean, null.null_sd, null.z, name not in CIRCUITS_OUTSIDE_VERDICT)
        )
    return results


def token_circuits(view: ModelView, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    """Layer-0 token x token fingerprints, invariant to residual rotation and to neuron/head permutation.

    Embeddings are RMS-normalised and the norm gains folded into the inputs;
    RoPE is the identity at position 0; KV heads are expanded for GQA. Biases
    are ignored (a fingerprint, not an exact forward pass).

    Raises:
        KeyError: when a layer-0 projection or norm is missing.
    """
    arch = view.architecture
    n_tokens, n_heads, n_kv, head_dim = embeddings.shape[0], arch.n_heads, arch.n_kv, arch.head_dim
    normalised = embeddings / embeddings.pow(2).mean(1, keepdim=True).add(RMS_EPSILON).sqrt()
    attention_in = normalised * view.norm(0, INPUT_NORM)
    mlp_in = normalised * view.norm(0, POST_ATTENTION_NORM)
    repeat = n_heads // n_kv
    queries = (attention_in @ view.mat(0, "q").T).view(n_tokens, n_heads, head_dim)
    keys = (attention_in @ view.mat(0, "k").T).view(n_tokens, n_kv, head_dim).repeat_interleave(repeat, dim=1)
    values = (attention_in @ view.mat(0, "v").T).view(n_tokens, n_kv, head_dim).repeat_interleave(repeat, dim=1)
    activation = ACTIVATIONS.get(arch.activation, F.silu)
    hidden = activation(mlp_in @ view.mat(0, "gate").T) * (mlp_in @ view.mat(0, "up").T)
    return {
        "emb_gram": normalised @ normalised.T,
        "qk": torch.einsum("ahd,bhd->ab", queries, keys),
        "ov": (values.reshape(n_tokens, n_heads * head_dim) @ view.mat(0, "o").T) @ normalised.T,
        "mlp": (hidden @ view.mat(0, "down").T) @ normalised.T,
    }


def invariant_comparison(
    first: torch.Tensor, second: torch.Tensor, generator: torch.Generator, draws: int = CIRCUIT_NULL_DRAWS
) -> tuple[float, list[float]]:
    """Off-diagonal Pearson r of two token x token matrices, plus ``draws`` token-permuted nulls."""
    n_tokens = first.shape[0]
    mask = ~torch.eye(n_tokens, dtype=torch.bool)
    value = pearson(first[mask], second[mask])
    nulls = []
    for _ in range(draws):
        order = torch.randperm(n_tokens, generator=generator)
        nulls.append(pearson(first[mask], second[order][:, order][mask]))
    return value, nulls


def summarise(layers: list[LayerComparison], circuits: list[CircuitComparison]) -> DeepSummary:
    """Reduce the per-tensor and circuit metrics to the verdict inputs."""
    rows = [row for layer in layers for row in layer.tensors if row.raw_cos is not None]
    raws = [row.raw_cos for row in rows if row.raw_cos is not None]
    distances = [row.rel_d for row in rows if row.rel_d is not None]
    neurons = next(
        (
            row
            for layer in layers[:1]
            for row in layer.tensors
            if row.kind == FINGERPRINT_NEURONS and row.nn_row is not None
        ),
        None,
    )
    scored = [circuit for circuit in circuits if circuit.in_verdict]
    return DeepSummary(
        raw_med=statistics.median(raws) if raws else 0.0,
        max_rel_d=max(distances) if distances else 1.0,
        inv_mean=statistics.fmean(circuit.r for circuit in scored) if scored else None,
        inv_zmin=min(circuit.z for circuit in scored) if scored else 0.0,
        nn=neurons.nn_row if neurons else None,
        nn_z=(neurons.nn_z or 0.0) if neurons else 0.0,
    )


def verdict(summary: DeepSummary) -> tuple[int, str]:
    """Heuristic (rank, verdict) from the deep-comparison summary numbers."""
    raw = summary.raw_med
    circuits_ok = (
        summary.inv_mean is not None and summary.inv_mean >= CIRCUITS_MIN_R and summary.inv_zmin >= CIRCUITS_MIN_Z
    )
    neurons_ok = summary.nn is not None and summary.nn >= NEURONS_MIN_NN and summary.nn_z >= NEURONS_MIN_Z
    if raw > IDENTICAL_MIN_RAW_COS and summary.max_rel_d < IDENTICAL_MAX_REL_D:
        result = RANK_IDENTICAL, VERDICT_IDENTICAL
    elif raw >= SAME_BASIS_MIN_RAW_COS:
        result = RANK_SAME_BASIS, VERDICT_SAME_BASIS
    elif raw >= RELATED_MIN_RAW_COS:
        result = RANK_RELATED, VERDICT_RELATED
    elif circuits_ok:
        result = RANK_TRANSFORMED, VERDICT_TRANSFORMED
    elif neurons_ok:
        result = RANK_TRANSFORMED, VERDICT_PERMUTED
    else:
        result = RANK_NONE, VERDICT_NONE
    return result


# ======================================================================= summary
def rank_candidates(results: list[DeepComparison]) -> list[RankedCandidate]:
    """Successful comparisons by rank, then raw cosine, then circuit r (all descending)."""
    ranked = [
        RankedCandidate(r.ref, r.origin, r.rank, r.verdict, r.summary.raw_med, r.summary.inv_mean)
        for r in results
        if r.summary is not None and r.rank is not None and r.verdict is not None
    ]
    return sorted(ranked, key=lambda r: (-r.rank, -r.raw_med, -(r.inv_mean or 0)))


def run_warnings(
    best: str | None, declared_direct: list[str], model: ModelRef, results: list[DeepComparison]
) -> list[str]:
    """Lineage inconsistencies and failed deep comparisons."""
    warnings = [f"Deep comparison with {r.ref} failed: {r.error}" for r in results if r.error is not None]
    if best is not None and declared_direct and best not in declared_direct:
        warnings.append(f"Declared base {declared_direct} differs from the best weight-level match ({best}).")
    elif best is not None and not declared_direct and model.location is Location.HUB:
        warnings.append("The lineage is NOT declared in the model metadata.")
    return warnings


def run_notes(compatible: list[CandidateCheck], results: list[DeepComparison], best: str | None) -> list[str]:
    """Hints when nothing could be compared or nothing matched."""
    notes = []
    if not compatible:
        notes.append("Nothing to compare: pass candidates or widen the search.")
    elif best is None:
        notes.append(
            "No candidate shows weight-level kinship. Try include_derived, a search text, a larger "
            "search limit, or behavioural fingerprinting (possible distillation)."
        )
    if results and best is None and all(r.error is not None for r in results):
        notes.append("Every deep comparison failed; see the warnings.")
    return notes
