"""Searching the Hugging Face Hub for model ids, for the model prompts.

A model reference may be a Hub repository id, so the model prompts suggest ids
as the user types. The suggestions come from the Hub's own search, ranked by
trending score. The network is optional: when there is no connection, or the
``huggingface_hub`` package is missing, the search returns nothing rather than
failing, so the prompt still works offline on models downloaded beforehand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from settings import HUB_SEARCH_LIMIT, MINIMUM_HUB_QUERY

# The Hub's own ranking: the models people are using now come first.
SEARCH_SORT = "trending_score"


@dataclass(frozen=True)
class HubCandidate:
    """A model found on the Hub, with what ranks and describes it."""

    model_id: str
    author: str
    downloads: int
    likes: int
    trending_score: float


def search_hub_models(
    query: str, *, token: str | None = None, limit: int = HUB_SEARCH_LIMIT, api: Any = None
) -> tuple[HubCandidate, ...]:
    """The Hub models whose search matches ``query``, most trending first.

    Args:
        query: the text typed so far; shorter than MINIMUM_HUB_QUERY yields nothing.
        token: Hub token for private or gated repositories.
        limit: the most candidates to return.
        api: the Hub client to use (an ``HfApi``); the real one is built when omitted.

    Returns:
        The matching candidates; empty when the query is too short or the Hub
        cannot be reached (no network, no package, or a Hub error).
    """
    result: tuple[HubCandidate, ...] = ()
    if len(query.strip()) >= MINIMUM_HUB_QUERY:
        try:
            result = tuple(_matches(query.strip(), token, limit, api))
        except Exception:  # noqa: BLE001 - offline, no package or a Hub error: offer nothing
            result = ()
    return result


def _matches(query: str, token: str | None, limit: int, api: Any) -> list[HubCandidate]:
    """Query the Hub and turn each hit into a candidate (performs the network call)."""
    client = api if api is not None else _client(token)
    found = client.list_models(search=query, sort=SEARCH_SORT, limit=limit, cardData=True)
    return [_candidate(model) for model in found]


def _candidate(model: Any) -> HubCandidate:
    """One search hit as a candidate, tolerating fields the Hub leaves unset."""
    return HubCandidate(
        model_id=str(model.id),
        author=str(getattr(model, "author", "") or ""),
        downloads=int(getattr(model, "downloads", 0) or 0),
        likes=int(getattr(model, "likes", 0) or 0),
        trending_score=float(getattr(model, "trending_score", 0.0) or 0.0),
    )


def _client(token: str | None) -> Any:
    """The real Hub client (imported here so this module imports without the package)."""
    from huggingface_hub import HfApi

    return HfApi(token=token)
