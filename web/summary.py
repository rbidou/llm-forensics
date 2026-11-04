"""Compact one-line summaries of a finished tool analysis, for the web finish view.

Each builder turns a tool's jsonable result dict into one uniform summary dict::

    {"tone": "ok|info|warn|bad|none",  # the .verdict banner colour
     "heading": "HIGH RISK",           # the big caps label (verdict-risk span)
     "headline": "one factual line with the numbers",  # the verdict-detail span
     "lines": ["supporting metric line", ...]}          # optional short list

The grading is **not** re-implemented here: the risk builders reuse the single
source in :mod:`reporting.html` (the same functions that tag the saved
report), so the live summary and the report always agree. Those imports stay inside
the builders so importing this module pulls neither Jinja2 nor torch — the web layer
must start without them (``python run.py check``).
"""

from __future__ import annotations

from typing import Any

# Risk severity -> verdict banner tone (mirrors reports/html risk tags and home.js).
TONE_BY_RISK: dict[str, str] = {"none": "none", "low": "info", "medium": "warn", "high": "bad"}

# Risk severity -> caps heading of the verdict banner.
HEADING_BY_RISK: dict[str, str] = {
    "none": "WITHIN BASELINE",
    "low": "LOW RISK",
    "medium": "MEDIUM RISK",
    "high": "HIGH RISK",
}


def _risk_summary(risk: str, headline: str, lines: list[str] | None = None) -> dict[str, Any]:
    """A summary dict for a graded risk level (``none`` .. ``high``)."""
    return {
        "tone": TONE_BY_RISK.get(risk, "none"),
        "heading": HEADING_BY_RISK.get(risk, "WITHIN BASELINE"),
        "headline": headline,
        "lines": lines or [],
    }


def weights_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Weight-diff summary: how many tensors moved, graded by the change's shape."""
    from reporting.html import format_number, weight_risk

    changed, compared = data.get("changed") or 0, data.get("compared") or 0
    headline = f"{format_number(changed)} of {format_number(compared)} tensors changed"
    lines = []
    beyond = data.get("beyond_noise")
    if beyond is not None:
        lines.append(f"{format_number(beyond)} changes beyond the quantization noise floor")
    return _risk_summary(weight_risk(data), headline, lines)


def entropy_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Entropy summary: the spectral-entropy shift against the random-update null."""
    from reporting.html import entropy_risk, format_number

    null = data.get("spectral_null") or {}
    z = null.get("z")
    if z is None:
        headline = "No spectral-entropy reading"
    else:
        direction = "drop" if (null.get("value") or 0) < 0 else "shift"
        headline = f"Spectral entropy {direction} z {format_number(z, 3)} vs a random update"
    return _risk_summary(entropy_risk(data), headline)


def shifts_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Token-shift summary: the strongest token-set alignment against the shuffle null."""
    from reporting.html import format_number, peak_alignment_z, shifts_risk

    peak = peak_alignment_z(data)
    headline = (
        f"Peak token-set alignment z {format_number(peak, 3)} vs the shuffle null"
        if peak is not None
        else "No token projection to align"
    )
    counts = data.get("counts") or {}
    lines = []
    if counts.get("compared") is not None:
        lines.append(
            f"{format_number(counts.get('changed'))} of {format_number(counts.get('compared'))} tensors changed"
        )
    return _risk_summary(shifts_risk(data), headline, lines)


def vocab_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Vocabulary summary: added and reassigned tokens (or new rows without a tokenizer)."""
    from reporting.html import format_number, vocab_risk

    diff = data.get("diff")
    if diff:
        added, reassigned = len(diff.get("added") or []), len(diff.get("reassigned") or [])
        headline = f"{format_number(added)} added, {format_number(reassigned)} reassigned token(s)"
    else:
        headline = f"{format_number(len(data.get('new_tokens') or []))} new embedding row(s)"
    return _risk_summary(vocab_risk(data), headline)


def behaviour_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Behaviour summary: the mean prediction divergence against the control-prompt null."""
    from analysis.assessment import Z_NOTABLE, Z_STRONG
    from reporting.html import format_number

    control, mean_kl = data.get("control"), data.get("mean_kl")
    result: dict[str, Any]
    if not control or mean_kl is None:
        result = {
            "tone": "none",
            "heading": "NO BASELINE",
            "headline": "No control comparison (vocabularies differ, or no control prompt could be used).",
            "lines": [],
        }
    else:
        z = control.get("z")
        risk = "high" if z is not None and z >= Z_STRONG else "medium" if z is not None and z >= Z_NOTABLE else "none"
        headline = (
            f"Mean KL {format_number(mean_kl)} vs {format_number(control.get('null_mean'))} "
            f"on the control prompts (z {format_number(z, 3)})"
        )
        lines = [
            f"Max KL {format_number(data.get('max_kl'))}",
            f"{format_number(data.get('positions'))} positions compared",
        ]
        result = _risk_summary(risk, headline, lines)
    return result


def provenance_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Provenance summary: the best weight-level match and the rank verdict (mirrors the report banner)."""
    from reporting.html import format_number

    ranking = data.get("ranking") or []
    best = next((row for row in ranking if row.get("ref") == data.get("best")), None)
    lines = []
    if data.get("gathered") is not None:
        lines.append(
            f"{format_number(data.get('gathered'))} gathered, {format_number(data.get('compatible'))} compatible, "
            f"{format_number(len(data.get('screen') or []))} screened"
        )
    if best:
        rank = best.get("rank") or 0
        tone = "ok" if rank >= 4 else "warn" if rank >= 2 else "none"
        heading = str(best.get("verdict") or "Match found")
        headline = (
            f"best weight-level match: {best.get('ref')} "
            f"(rank {format_number(rank)}, median raw cosine {format_number(best.get('raw_med'))})"
        )
    elif ranking:
        tone, heading = "none", str(ranking[0].get("verdict") or "No match")
        headline = f"no candidate shows weight-level kinship; closest: {ranking[0].get('ref')}"
    else:
        tone, heading, headline = "none", "NO MATCH", "No candidate could be compared."
    return {"tone": tone, "heading": heading, "headline": headline, "lines": lines}


def lineage_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Lineage summary: how many chosen pairs compared, and the strongest shift found."""
    from reporting.html import format_number

    pairs = data.get("pairs") or []
    compared = [pair for pair in pairs if not pair.get("error")]
    failed = [pair for pair in pairs if pair.get("error")]
    best_sigma: float | None = None
    best_name: str | None = None
    for pair in compared:
        updates = (pair.get("shifts") or {}).get("top_updates") or []
        if updates and updates[0].get("sigma") is not None and (best_sigma is None or updates[0]["sigma"] > best_sigma):
            best_sigma, best_name = updates[0]["sigma"], updates[0].get("name")
    lines = []
    if best_name is not None:
        lines.append(f"strongest shift σ {format_number(best_sigma)} on {best_name}")
    if failed:
        lines.append(f"{format_number(len(failed))} pair(s) failed")
    return {
        "tone": "info" if compared else "none",
        "heading": "LINEAGE" if compared else "NO COMPARISON",
        "headline": f"{format_number(len(compared))} of {format_number(len(pairs))} pair(s) compared",
        "lines": lines,
    }
