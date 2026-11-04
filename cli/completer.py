"""Completion of model references and file paths as the user types.

A model reference is a local path or a Hub repository id, so the model prompt
completes both. Local paths and the ids already in the local Hugging Face cache
are always offered. When the input looks like a Hub id rather than a path, the
Hub's own search is queried too, so a model that was never downloaded can still
be completed. That lookup is optional: it is skipped while a path is being
typed, and it quietly offers nothing when there is no connection, so the prompt
still works offline on models downloaded beforehand.

The cache is scanned once per session (it can be large) and every lookup runs in
a worker thread, so the prompt never blocks on the disk or the network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache

from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter, ThreadedCompleter
from prompt_toolkit.document import Document

from models.detect import _looks_like_path
from models.hub_search import search_hub_models

# Below this, a query matches so many ids that the suggestions are noise.
MINIMUM_QUERY_LENGTH = 2
SUGGESTION_LIMIT = 20
HUB_CACHE_SIZE = 128
CACHED_MODEL_META = "in the Hugging Face cache"
HUB_MODEL_META = "on the Hugging Face Hub"

# A source of Hub ids for a query, so tests can supply ids without the network.
HubSearch = Callable[[str], Iterable[str]]


def reference_completer(token: str | None = None) -> Completer:
    """Build the completer of the model prompts: paths, cached ids, then Hub ids."""
    return ThreadedCompleter(ReferenceCompleter(token=token))


def path_completer() -> Completer:
    """Build the completer of the file prompts (prompt files, folder paths)."""
    return ThreadedCompleter(PathCompleter(expanduser=True))


class ReferenceCompleter(Completer):
    """Suggests local paths, cached Hub models, and the Hub models a typed id matches."""

    def __init__(
        self,
        cached: Iterable[str] | None = None,
        *,
        token: str | None = None,
        hub_search: HubSearch | None = None,
    ) -> None:
        """Use ``cached`` / ``hub_search`` as the sources (tests), else the cache and the live Hub."""
        self._paths = PathCompleter(expanduser=True)
        self._cached = tuple(cached) if cached is not None else None
        self._token = token
        self._hub_search = hub_search

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterator[Completion]:
        """Yield the path completions, then the cached ids, then the matching Hub ids."""
        yield from self._paths.get_completions(document, complete_event)
        query = document.text_before_cursor.strip()
        typed = len(document.text_before_cursor)
        if len(query) >= MINIMUM_QUERY_LENGTH:
            cached = self._known_ids()
            yield from _cached_completions(query, cached, typed)
            if not _looks_like_path(query):
                fresh = tuple(model_id for model_id in self._hub_ids(query) if model_id not in cached)
                yield from _hub_completions(fresh, typed)

    def _known_ids(self) -> tuple[str, ...]:
        """The cached model ids, scanned on first use."""
        return self._cached if self._cached is not None else cached_model_ids()

    def _hub_ids(self, query: str) -> tuple[str, ...]:
        """The ids the Hub returns for ``query`` (injected in tests, else the live search)."""
        return tuple(self._hub_search(query)) if self._hub_search is not None else _hub_ids(query, self._token)


def _cached_completions(query: str, known: Iterable[str], typed: int) -> Iterator[Completion]:
    """One completion per known id starting with ``query`` (case-insensitive)."""
    lowered = query.lower()
    matches = [model_id for model_id in known if model_id.lower().startswith(lowered)]
    for model_id in matches[:SUGGESTION_LIMIT]:
        yield Completion(model_id, start_position=-typed, display=model_id, display_meta=CACHED_MODEL_META)


def _hub_completions(ids: Iterable[str], typed: int) -> Iterator[Completion]:
    """One completion per Hub id, kept in the Hub's own relevance order."""
    for model_id in list(ids)[:SUGGESTION_LIMIT]:
        yield Completion(model_id, start_position=-typed, display=model_id, display_meta=HUB_MODEL_META)


@lru_cache(maxsize=1)
def cached_model_ids() -> tuple[str, ...]:
    """Ids of the models in the local Hugging Face cache, sorted; empty when there is none."""
    result: tuple[str, ...] = ()
    try:
        from huggingface_hub import scan_cache_dir

        repos = scan_cache_dir().repos
        result = tuple(sorted(repo.repo_id for repo in repos if repo.repo_type == "model"))
    except Exception:  # noqa: BLE001 - no cache, no package or an unreadable cache: no suggestions
        result = ()
    return result


@lru_cache(maxsize=HUB_CACHE_SIZE)
def _hub_ids(query: str, token: str | None) -> tuple[str, ...]:
    """Ids the Hub returns for ``query``, cached per query; empty when it cannot be reached."""
    return tuple(candidate.model_id for candidate in search_hub_models(query, token=token))
