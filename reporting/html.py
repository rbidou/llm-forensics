"""Self-contained HTML reports, rendered from the JSON form of a result.

The HTML is built from ``to_jsonable(result)``, never from the live objects, so
a stored ``report.json`` can be rendered again at any time (after a template
update, for instance) and gives exactly the same page.

Design:

* one Jinja2 template per analysis, all extending ``base.html.j2`` (header,
  parameters, inputs, notes and warnings); repeated pieces (stat cards, bars,
  token chips, null comparisons) are macros of ``macros.html.j2``; the token
  shift sections are a macro shared by the shifts and lineage reports;
* the page is static and self-contained: one inline stylesheet using the dark
  theme tokens of the web UI, no script, no external resource, and a light
  ``@media print`` override so it prints;
* autoescaping is always on: token texts, prompts, generations and tensor names
  come from models and users and are never trusted. The only markup produced in
  Python (:func:`format_token`) escapes every piece it wraps;
* results may lack optional keys (older reports, partial runs): the templates
  use defaults rather than a strict undefined.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime
from functools import lru_cache
from typing import Any, TypeGuard

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from results import to_jsonable

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TEMPLATES: dict[str, str] = {
    "weights": "weights.html.j2",
    "shifts": "shifts.html.j2",
    "vocab": "vocab.html.j2",
    "entropy": "entropy.html.j2",
    "lineage": "lineage.html.j2",
    "provenance": "provenance.html.j2",
    "behaviour": "behaviour.html.j2",
    "assessment": "assessment.html.j2",
}
GENERIC_TEMPLATE = "generic.html.j2"
ANALYSIS_LABELS: dict[str, str] = {
    "weights": "Weight comparison",
    "shifts": "Token shifts",
    "vocab": "Vocabulary & embeddings",
    "entropy": "Entropy analysis",
    "lineage": "Hub lineage",
    "provenance": "Provenance",
    "behaviour": "Behaviour probe",
    "assessment": "Model analysis",
}
GENERIC_LABEL = "Analysis report"
MISSING = "—"
NUMBER_DIGITS = 4  # significant digits of a float
PERCENT_DIGITS = 1
Z_NOTABLE = 3.0  # |z| from which a score is flagged as clearly above its null baseline
TOP_TENSORS = 50  # tensors listed by the weights report (the JSON has them all)
VERDICT_CLASSES: dict[int, str] = {5: "ok", 4: "ok", 3: "warn", 2: "warn"}  # provenance rank -> .verdict class
VERDICT_DEFAULT_CLASS = "none"
BADGE_CLASSES: dict[str, str] = {"ok": "badge-success", "warn": "badge-warning", "none": "badge-neutral"}
# Assessment workflow: risk level -> .verdict class, and signal severity -> badge class.
RISK_CLASSES: dict[str, str] = {"clean": "ok", "low": "info", "elevated": "warn", "high": "bad"}
RISK_DEFAULT_CLASS = "none"
# Risk level -> caps heading of the verdict banner.
RISK_HEADINGS: dict[str, str] = {
    "clean": "NO RISK",
    "low": "LOW RISK",
    "elevated": "ELEVATED RISK",
    "high": "HIGH RISK",
}
# Risk level -> badge class of the executive summary's risk tag.
RISK_BADGES: dict[str, str] = {
    "clean": "badge-success",
    "low": "badge-info",
    "elevated": "badge-warning",
    "high": "badge-danger",
}
# Verdict confidence -> badge class (high trust green, normal blue, low red).
CONFIDENCE_BADGES: dict[str, str] = {"high": "badge-success", "normal": "badge-info", "low": "badge-danger"}
BADGE_DEFAULT = "badge-neutral"
SEVERITY_BADGES: dict[str, str] = {
    "none": "badge-neutral",
    "low": "badge-info",
    "medium": "badge-warning",
    "high": "badge-danger",
}
SEVERITY_DEFAULT_BADGE = "badge-neutral"
ADDENDUM_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # addendum index -> letter
RELATION_LABELS: dict[str, str] = {"promotes": "promotes", "triggered_by": "triggered by"}
# Tensor role (stats.role_of) -> single-letter label for the report tables.
ROLE_ABBREVIATIONS: dict[str, str] = {"writer": "W", "reader": "R", "embedding": "E", "other": "O"}
# Tensor module (stats.module_of) -> short label for the report tables; norm / embedding modules show MISSING.
MODULE_ABBREVIATIONS: dict[str, str] = {
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "o_proj": "O",
    "out_proj": "O",
    "wo": "O",
    "qkv_proj": "QKV",
    "query_key_value": "QKV",
    "c_attn": "ATTN",
    "gate_proj": "GATE",
    "up_proj": "UP",
    "down_proj": "DOWN",
    "gate_up_proj": "GATE·UP",
    "c_fc": "FC",
    "dense_h_to_4h": "FC",
    "c_proj": "PROJ",
    "dense_4h_to_h": "DOWN",
}
# Substrings that mark an embedding / unembedding tensor by its full name (mirrors stats.EMBEDDING_PARTS).
EMBEDDING_NAME_PARTS: tuple[str, ...] = ("embed_tokens", "lm_head", "wte", "word_embeddings")
# Token circuit (provenance.CIRCUIT_NAMES) -> short upper-case label for the report tables.
CIRCUIT_ABBREVIATIONS: dict[str, str] = {"emb_gram": "EMB", "qk": "QK", "ov": "OV", "mlp": "MLP"}
# A grouped-table row is highlighted when its value is at least this many times the table median.
ROW_HOT_FACTOR = 2.0
# The labels used across the report tables, grouped by column, with a plain-language meaning.
# Single source of truth for the per-table legends and the closing "Labels" addendum.
LABEL_GLOSSARY: dict[str, tuple[tuple[str, str], ...]] = {
    "Type": (
        ("EMB", "embedding or unembedding (token ↔ vector)"),
        ("NORM", "normalisation layer"),
        ("ATTN", "attention projection"),
        ("MLP", "feed-forward (MLP) projection"),
    ),
    "Module": (
        ("Q", "query projection"),
        ("K", "key projection"),
        ("V", "value projection"),
        ("O", "attention output projection"),
        ("QKV", "fused query-key-value projection"),
        ("GATE", "MLP gate projection"),
        ("UP", "MLP up projection"),
        ("DOWN", "MLP down projection"),
        ("FC", "fully-connected (GPT-2 style) projection"),
        ("PROJ", "output projection"),
    ),
    "Role": (
        ("W", "writer — writes into the residual stream (read through the unembedding)"),
        ("R", "reader — reads from the residual stream (read through the embeddings)"),
        ("E", "embedding / unembedding matrix"),
        ("O", "other"),
    ),
    "Circuit": (
        ("EMB", "token-gram E·Eᵀ (excluded from the verdict: token semantics, not kinship)"),
        ("QK", "query-key attention circuit"),
        ("OV", "output-value attention circuit"),
        ("MLP", "feed-forward circuit"),
    ),
    "Origin": (
        ("E", "explicit — a candidate you supplied"),
        ("S", "search — found by a Hub search"),
        ("D", "declared — named in the model's own metadata"),
    ),
    "Kind": (
        ("ROOT", "a base / root model (not declared as derived)"),
        ("DERIVED", "declares a parent in its metadata"),
    ),
    "Token flags": (
        ("T", "trained — the fine-tune taught this token an embedding"),
        ("P", "placeholder — an added token left untrained"),
        ("E", "emittable — the model can also produce the token (untied output row)"),
        ("R", "read-only — the model reads the token but cannot emit it"),
    ),
}
EMPTY_TOKEN = "∅"
VISIBLE_CHARACTERS: dict[str, str] = {" ": "␣", "\n": "\\n", "\t": "\\t", "\r": "\\r"}
CONTROL_LIMIT = 32  # code points below this are shown as escapes
SUBJECT_KEYS = ("suspect", "tuned", "model")  # inputs that name the model a report is about, best first


def render_result(result: Any) -> str:
    """Render a result dataclass as a self-contained HTML page.

    Args:
        result: any analysis result (``WeightDiffResult``, ``ProvenanceResult``, ...).

    Returns:
        The HTML document, identical to ``render_html(to_jsonable(result))``.
    """
    return render_html(to_jsonable(result))


def render_html(data: dict[str, Any]) -> str:
    """Render the JSON form of a result as a self-contained HTML page.

    The template is chosen from ``data["meta"]["analysis"]``; an unknown or
    missing analysis gives a generic page with the metadata and the escaped JSON.

    Args:
        data: ``to_jsonable(result)``, or a ``report.json`` read back.

    Returns:
        The HTML document.
    """
    meta = normalised_meta(data)
    analysis = meta["analysis"]
    template = _environment().get_template(TEMPLATES.get(analysis, GENERIC_TEMPLATE))
    return template.render(data=data, meta=meta, title=report_title(meta), label=analysis_label(analysis))


def normalised_meta(data: dict[str, Any]) -> dict[str, Any]:
    """The result metadata with every field present (defaults for missing ones)."""
    raw = data.get("meta") if isinstance(data, dict) else None
    meta = raw if isinstance(raw, dict) else {}
    return {
        "analysis": str(meta.get("analysis") or ""),
        "inputs": _mapping(meta.get("inputs")),
        "parameters": _mapping(meta.get("parameters")),
        "created": str(meta.get("created") or MISSING),
        "tool_version": str(meta.get("tool_version") or MISSING),
        "notes": _strings(meta.get("notes")),
        "warnings": _strings(meta.get("warnings")),
    }


def report_title(meta: dict[str, Any]) -> str:
    """Page title: the analysis label and the model it ran on, e.g. ``Weight comparison: model.gguf``.

    The subject is the model under scrutiny — the suspect, or the tuned side of a comparison —
    shown by its Hub id or, for a local model, its file name.
    """
    inputs = meta["inputs"]
    subject = subject_model(inputs)
    if meta["analysis"] == "assessment":
        result = f"Model Analysis: {model_name(subject)}" if subject else "Model Analysis"
    else:
        label = analysis_label(meta["analysis"])
        result = f"{label}: {model_name(subject)}" if subject else label
    return result


def subject_model(inputs: dict[str, Any]) -> str:
    """The model a report is about: the first of ``suspect`` / ``tuned`` / ``model`` present, else the first input."""
    result = ""
    for key in SUBJECT_KEYS:
        value = inputs.get(key)
        if value:
            result = str(value)
            break
    else:
        result = next((str(value) for value in inputs.values() if value), "")
    return result


def model_name(ref: str) -> str:
    """Display name of a model reference: a Hub id (``org/model``) whole, a local path as its name."""
    is_path = os.path.isabs(ref) or "\\" in ref or ref.startswith(".")
    return os.path.basename(ref.replace("\\", "/").rstrip("/")) if is_path else ref


def analysis_label(analysis: str) -> str:
    """Human name of an analysis."""
    return ANALYSIS_LABELS.get(analysis, GENERIC_LABEL)


# ----------------------------------------------------------------- filters
def format_number(value: Any, digits: int = NUMBER_DIGITS) -> str:
    """A number for display: floats with ``digits`` significant digits, None (or NaN) as a dash.

    Integers keep every digit (with thousands separators), booleans read
    ``yes`` / ``no``, anything else is shown as text.
    """
    result = str(value)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        result = MISSING
    elif isinstance(value, bool):
        result = "yes" if value else "no"
    elif isinstance(value, int):
        result = f"{value:,}"
    elif isinstance(value, float):
        result = f"{value:.{digits}g}"
    return result


def format_percent(value: Any, digits: int = PERCENT_DIGITS) -> str:
    """A fraction as a percentage (``0.25`` -> ``25.0%``), None as a dash."""
    result = MISSING
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        result = f"{value * 100:.{digits}f}%"
    return result


def format_token(text: Any) -> Markup:
    """A token's text in a mono chip, with invisible characters made visible.

    Spaces read ``␣``, newlines ``\\n``, tabs ``\\t``, other control characters
    ``\\xNN`` (each in a dimmed span), and an empty token ``∅``. Every piece of
    the text is escaped.
    """
    value = "" if text is None else str(text)
    inner = Markup("").join(_visible_character(character) for character in value)
    if not value:
        inner = Markup(f'<span class="tok-ctl">{EMPTY_TOKEN}</span>')
    return Markup(f'<span class="tok">{inner}</span>')


def bar_width(value: Any, maximum: Any) -> str:
    """Width (percent of ``maximum``, clamped to 0-100) of a CSS bar, as text for a ``style``."""
    width = 0.0
    if _is_number(value) and _is_number(maximum) and maximum > 0:
        width = min(max(value / maximum * 100.0, 0.0), 100.0)
    return f"{width:.1f}"


def peak(values: Any) -> float:
    """Largest finite number of ``values`` (the 100% of a set of bars), 0 when there is none."""
    numbers = [value for value in values if _is_number(value)] if values is not None else []
    return float(max(numbers)) if numbers else 0.0


def verdict_class(rank: Any) -> str:
    """``.verdict`` modifier of a provenance rank: 5/4 ``ok``, 3/2 ``warn``, else ``none``."""
    return VERDICT_CLASSES.get(rank, VERDICT_DEFAULT_CLASS) if isinstance(rank, int) else VERDICT_DEFAULT_CLASS


def rank_badge(rank: Any) -> str:
    """Badge class matching :func:`verdict_class`."""
    return BADGE_CLASSES[verdict_class(rank)]


def risk_class(level: Any) -> str:
    """``.verdict`` modifier of an assessment risk level: clean ``ok``, high ``bad``, else amber/grey."""
    return RISK_CLASSES.get(str(level), RISK_DEFAULT_CLASS)


def severity_badge(severity: Any) -> str:
    """Badge class of a signal severity: high ``danger``, medium ``warning``, low ``info``, none ``neutral``."""
    return SEVERITY_BADGES.get(str(severity), SEVERITY_DEFAULT_BADGE)


def risk_heading(level: Any) -> str:
    """Caps heading of a risk level: ``NO RISK`` for clean, ``<LEVEL> RISK`` otherwise."""
    return RISK_HEADINGS.get(str(level), f"{str(level).upper()} RISK")


def risk_badge(level: Any) -> str:
    """Badge class of a risk level: high ``danger``, elevated ``warning``, low ``info``, clean ``success``."""
    return RISK_BADGES.get(str(level), BADGE_DEFAULT)


def confidence_badge(confidence: Any) -> str:
    """Badge class of a verdict confidence: high ``success``, normal ``info``, low ``danger``."""
    return CONFIDENCE_BADGES.get(str(confidence), BADGE_DEFAULT)


def letter(index: Any) -> str:
    """Addendum label from a 1-based index: ``1`` -> ``A``; past ``Z`` the number is used."""
    return ADDENDUM_LETTERS[index - 1] if isinstance(index, int) and 1 <= index <= len(ADDENDUM_LETTERS) else str(index)


def human_date(value: Any) -> str:
    """An ISO timestamp as a readable date (``30 October 2026, 10:49 UTC``), or the raw text on failure."""
    result = str(value)
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        moment = None
    if moment is not None:
        zone = moment.tzname() if moment.tzinfo is not None else None
        result = moment.strftime("%d %B %Y, %H:%M")
        if zone:
            result = f"{result} {zone}"
    return result


def value_text(value: Any) -> str:
    """Any parameter value as one line: numbers formatted, lists joined, mappings as compact JSON."""
    result = format_number(value)
    if isinstance(value, list | tuple):
        result = ", ".join(value_text(item) for item in value) if value else MISSING
    elif isinstance(value, dict):
        result = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result


def pretty_json(value: Any) -> str:
    """Indented JSON text (escaped by the template like any other text)."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def relation_label(relation: Any) -> str:
    """Reading of a projection: ``promotes`` (writer) or ``triggered by`` (reader)."""
    return RELATION_LABELS.get(str(relation), str(relation))


def role_abbr(role: Any) -> str:
    """Single-letter tensor role: writer ``W``, reader ``R``, embedding ``E``, other ``O`` (unknown: as given)."""
    return ROLE_ABBREVIATIONS.get(str(role), str(role))


def tensor_type(name: Any) -> str:
    """Coarse tensor type from its full name: ``EMB``, ``NORM``, ``ATTN``, ``MLP``, or ``MISSING`` when none.

    Tested in order so a normalisation layer inside the attention block (``post_attention_layernorm``)
    reads ``NORM``, not ``ATTN``.
    """
    text = str(name).lower()
    if any(part in text for part in EMBEDDING_NAME_PARTS):
        result = "EMB"
    elif "norm" in text or "ln_" in text:
        result = "NORM"
    elif "attn" in text or "attention" in text:
        result = "ATTN"
    elif "mlp" in text or "feed_forward" in text or "ffn" in text:
        result = "MLP"
    else:
        result = MISSING
    return result


def module_abbr(module: Any) -> str:
    """Short module label for the report tables (``q_proj`` -> ``Q``).

    Norm and embedding modules show ``MISSING`` (their type column already names them); an unknown
    module falls back to its name without the ``_proj`` suffix, upper-cased.
    """
    key = str(module).lower()
    if key in MODULE_ABBREVIATIONS:
        result = MODULE_ABBREVIATIONS[key]
    elif any(part in key for part in EMBEDDING_NAME_PARTS) or "norm" in key or key.startswith("ln"):
        result = MISSING
    else:
        result = key.removesuffix("_proj").upper() or MISSING
    return result


def peak_z(update: Any) -> float:
    """Strongest end of a projected update: the higher ``|z|`` of its two ends, or ``-inf`` without a reading.

    Used to order the top-update details (strongest first) and is the same key the highlight threshold tests.
    """
    result = float("-inf")
    projection = update.get("projection") if isinstance(update, dict) else None
    if isinstance(projection, dict):
        scores = []
        for end in ("positive", "negative"):
            null = (projection.get(end) or {}).get("null") or {}
            z = null.get("z")
            if isinstance(z, int | float) and not isinstance(z, bool):
                scores.append(abs(z))
        result = max(scores) if scores else float("-inf")
    return result


def origin_abbr(origin: Any) -> str:
    """Single capital letter for a provenance origin: explicit ``E``, search ``S``, declared ``D``."""
    text = str(origin).strip()
    return text[0].upper() if text else MISSING


def circuit_abbr(name: Any) -> str:
    """Short upper-case circuit label (``emb_gram`` -> ``EMB``); an unknown name upper-cased."""
    return CIRCUIT_ABBREVIATIONS.get(str(name), str(name).upper())


def module_group_label(module: Any) -> str:
    """Module label for a grouping table: like :func:`module_abbr` but ``EMB`` / ``NORM`` (never a dash).

    The grouping tables have no separate type column, so an embedding or norm module keeps a
    readable label of its own instead of collapsing to ``MISSING``.
    """
    key = str(module).lower()
    if key in MODULE_ABBREVIATIONS:
        result = MODULE_ABBREVIATIONS[key]
    elif any(part in key for part in EMBEDDING_NAME_PARTS):
        result = "EMB"
    elif "norm" in key or key.startswith("ln"):
        result = "NORM"
    else:
        result = key.removesuffix("_proj").upper() or MISSING
    return result


def median(values: Any) -> float:
    """Median of the finite numbers of ``values``, ``0.0`` when there is none."""
    numbers = [value for value in values if _is_number(value)] if values is not None else []
    return float(statistics.median(numbers)) if numbers else 0.0


def above_typical(value: Any, values: Any) -> bool:
    """True when ``|value|`` is at least :data:`ROW_HOT_FACTOR` times the median magnitude of ``values``.

    Self-scaling flag for a row that stands out from its peers in the same table; works for
    positive columns (relative change) and signed ones (entropy Δ) alike.
    """
    magnitudes = [abs(number) for number in values if _is_number(number)] if values is not None else []
    threshold = ROW_HOT_FACTOR * (statistics.median(magnitudes) if magnitudes else 0.0)
    return bool(_is_number(value) and threshold > 0 and abs(value) >= threshold)


def weight_risk(data: Any) -> str:
    """Risk level of a weight diff, mirroring the assessment's ``_weight_shape_signal`` grading.

    Reads the shape of the per-tensor relative change: concentrated in a few tensors (targeted)
    or broad (continued training). A quantized comparison with nothing beyond its noise is ``none``.
    """
    strong, notable, broad_fraction, _ = _thresholds()
    changed = sorted(
        (
            tensor["rel_change"]
            for tensor in (data.get("tensors") or [])
            if isinstance(tensor, dict) and tensor.get("changed") and _is_number(tensor.get("rel_change"))
        ),
        reverse=True,
    )
    result = "none"
    if changed:
        top, mid = changed[0], statistics.median(changed)
        concentration = top / mid if mid > 0 else float("inf")
        compared = data.get("compared") or 0
        fraction = (data.get("changed") or 0) / compared if compared else 0.0
        if data.get("beyond_noise") == 0:
            result = "none"
        elif concentration >= strong:
            result = "high"
        elif concentration >= notable:
            result = "medium"
        elif fraction >= broad_fraction:
            result = "low"
    return result


def entropy_risk(data: Any) -> str:
    """Risk level of an entropy result, mirroring the assessment's ``_entropy_shift_signal`` grading.

    A spectral-entropy drop well beyond the random-update null reads as restructuring; a shift
    within the quantization noise floor is a dequantization artefact and reads as ``none``.
    """
    _, _, _, strong_z = _thresholds()
    null = data.get("spectral_null") if isinstance(data, dict) else None
    result = "none"
    if isinstance(null, dict) and _is_number(null.get("z")):
        z, value = null["z"], null.get("value")
        quantized = any(isinstance(tensor, dict) and tensor.get("quantization") for tensor in data.get("tensors") or [])
        within_noise = quantized and data.get("beyond_noise") == 0
        if abs(z) < Z_NOTABLE or within_noise:
            result = "none"
        elif _is_number(value) and value < 0:
            result = "high" if abs(z) >= strong_z else "medium"
        else:
            result = "low"
    return result


def vocab_risk(vocab: Any) -> str:
    """Risk level of a vocabulary result, mirroring the assessment's ``_vocab_reading`` grading.

    A reassigned id, a taught emittable token, or a reader update triggered by an added token are
    the strong findings; a trained or concentrated row is weaker; an untrained addition is weakest.
    """
    _, _, _, strong_z = _thresholds()
    diff = vocab.get("diff") if isinstance(vocab, dict) else None
    reassigned = (diff or {}).get("reassigned") or []
    new_tokens = vocab.get("new_tokens") or []
    trained = [token for token in new_tokens if isinstance(token, dict) and token.get("trained")]
    emittable = [token for token in trained if token.get("emittable")]
    read_only = [token for token in trained if token.get("emittable") is False]
    added = len((diff or {}).get("added") or []) if diff else len(new_tokens)
    top_z = _top_row_z(vocab)
    result = "none"
    if _reader_trigger_hit(vocab) or reassigned or emittable:
        result = "high"
    elif read_only or top_z >= strong_z:
        result = "medium"
    elif trained or top_z >= Z_NOTABLE or added:
        result = "low"
    return result


def shifts_risk(shifts: Any) -> str:
    """Risk level of a token-shift result, mirroring the assessment's ``_token_alignment_signal``.

    Reads the strongest token alignment of the heaviest updates against their token-shuffle null:
    a peak z far beyond the null reads as an update that behaves like a trigger on a token set.
    Uses the raw (signed) z, as the signal does, since only alignment above the null is a trigger.
    """
    _, _, _, strong_z = _thresholds()
    best = peak_alignment_z(shifts)
    if best is None:
        result = "none"
    elif best >= strong_z:
        result = "high"
    elif best >= Z_NOTABLE:
        result = "medium"
    else:
        result = "none"
    return result


def peak_alignment_z(shifts: Any) -> float | None:
    """Highest raw end z across a token-shift result's projected updates, ``None`` without a reading.

    The number behind :func:`shifts_risk`; drives the result's Peak-alignment card.
    """
    best: float | None = None
    updates = (shifts.get("top_updates") or []) if isinstance(shifts, dict) else []
    for update in updates:
        z = alignment_z(update)
        if z is not None and (best is None or z > best):
            best = z
    return best


def alignment_z(update: Any) -> float | None:
    """Highest raw end z of one projected update, ``None`` without a reading.

    Mirrors ``assessment._token_alignment_signal``: the raw (signed) null z of the stronger end,
    not its magnitude, since only an alignment above the null reads as a trigger. Drives the
    row highlight of the Top-updates table (unlike :func:`peak_z`, which ranks ends by ``|z|``).
    """
    best: float | None = None
    projection = update.get("projection") if isinstance(update, dict) else None
    if isinstance(projection, dict):
        for end_key in ("positive", "negative"):
            z = ((projection.get(end_key) or {}).get("null") or {}).get("z")
            if _is_number(z) and (best is None or z > best):
                best = float(z)
    return best


# ----------------------------------------------------------------- internals
@lru_cache(maxsize=1)
def _environment() -> Environment:
    """The Jinja2 environment: autoescaped templates, report filters and constants."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters.update(
        num=format_number,
        pct=format_percent,
        token=format_token,
        bar=bar_width,
        peak=peak,
        verdict_class=verdict_class,
        rank_badge=rank_badge,
        risk_class=risk_class,
        risk_heading=risk_heading,
        risk_badge=risk_badge,
        confidence_badge=confidence_badge,
        severity_badge=severity_badge,
        letter=letter,
        human_date=human_date,
        value_text=value_text,
        pretty_json=pretty_json,
        relation=relation_label,
        role_abbr=role_abbr,
        tensor_type=tensor_type,
        module_abbr=module_abbr,
        module_group_label=module_group_label,
        origin_abbr=origin_abbr,
        circuit_abbr=circuit_abbr,
        median=median,
        above_typical=above_typical,
        weight_risk=weight_risk,
        entropy_risk=entropy_risk,
        vocab_risk=vocab_risk,
        shifts_risk=shifts_risk,
        peak_alignment_z=peak_alignment_z,
        alignment_z=alignment_z,
        peak_z=peak_z,
    )
    environment.globals.update(
        Z_NOTABLE=Z_NOTABLE, TOP_TENSORS=TOP_TENSORS, MISSING=MISSING, LABEL_GLOSSARY=LABEL_GLOSSARY
    )
    return environment


def _visible_character(character: str) -> Markup:
    """One character of a token, escaped, or its visible stand-in in a dimmed span."""
    stand_in = VISIBLE_CHARACTERS.get(character)
    if stand_in is None and ord(character) < CONTROL_LIMIT:
        stand_in = f"\\x{ord(character):02x}"
    return Markup(f'<span class="tok-ctl">{stand_in}</span>') if stand_in is not None else escape(character)


def _is_number(value: Any) -> TypeGuard[int | float]:
    """True for a finite int or float (not a bool); narrows the value for the type checker."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@lru_cache(maxsize=1)
def _thresholds() -> tuple[float, float, float, float]:
    """The assessment grading constants ``(concentration strong, concentration notable, broad fraction, strong z)``.

    Imported lazily so this module keeps its standard-library-only module scope (the analysis
    package pulls torch); the single source of the numbers is ``analysis.assessment``.
    """
    from analysis.assessment import (
        BROAD_CHANGE_FRACTION,
        CONCENTRATION_NOTABLE,
        CONCENTRATION_STRONG,
        Z_STRONG,
    )

    return CONCENTRATION_STRONG, CONCENTRATION_NOTABLE, BROAD_CHANGE_FRACTION, Z_STRONG


def _top_row_z(vocab: Any) -> float:
    """Strongest embedding-row concentration z across a vocab result's matrices, ``0.0`` without one."""
    best = 0.0
    for shift in vocab.get("embeddings") or []:
        concentration = shift.get("concentration") if isinstance(shift, dict) else None
        z = concentration.get("z") if isinstance(concentration, dict) else None
        if _is_number(z) and z > best:
            best = z
    return best


def _reader_trigger_hit(vocab: Any) -> bool:
    """True when a reader update's high-z end (``|z| ≥`` strong) is one of the added tokens.

    Mirrors ``assessment._reader_trigger_hit``: the shape of a hidden trigger Token Shifts cannot see.
    """
    _, _, _, strong_z = _thresholds()
    result = False
    for reader in vocab.get("reader_triggers") or []:
        projection = reader.get("projection") if isinstance(reader, dict) else None
        new_ids = {hit.get("token_id") for hit in (reader.get("new_hits") or []) if isinstance(hit, dict)}
        if not isinstance(projection, dict) or not new_ids:
            continue
        for end_key in ("positive", "negative"):
            end = projection.get(end_key)
            if not isinstance(end, dict):
                continue
            z = (end.get("null") or {}).get("z")
            hits = any(
                isinstance(token, dict) and token.get("token_id") in new_ids for token in end.get("tokens") or []
            )
            if hits and _is_number(z) and z >= strong_z:
                result = True
    return result


def _mapping(value: Any) -> dict[str, Any]:
    """``value`` when it is a dict, else an empty one."""
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    """``value`` as a list of strings (non-lists give an empty list)."""
    return [str(item) for item in value] if isinstance(value, list) else []
