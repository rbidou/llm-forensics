"""Reports saved on disk: one folder per report, listed without parsing the big files.

Layout::

    <root>/<id>/report.json   to_jsonable(result): the complete data
    <root>/<id>/report.html   rendered from report.json (self-contained)
    <root>/<id>/meta.json     small sidecar: id, analysis, title, dates, inputs, one-line summary

A report id is ``YYYYmmdd-HHMMSS-<analysis>-<slug>`` (local time of the save),
with a numeric suffix when the folder already exists. Folders are created with
``os.makedirs(exist_ok=False)``, so two saves in the same second never share one.
The sidecar is written last: a folder without one is an incomplete report and is
not listed.

Security: the web front-end passes ids taken from URLs. Every id is matched
against :data:`REPORT_ID_PATTERN` (ASCII only, no dot, no separator) BEFORE any
filesystem access, and the resolved folder must stay inside the root
(``os.path.commonpath`` on real paths, which also defeats symbolic links).
"""

from __future__ import annotations

import builtins
import json
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from reporting.html import format_number, model_name, render_html
from results import to_jsonable

DEFAULT_ROOT = "reports"  # relative to the current directory; front-ends pass an absolute path
REPORT_ID_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}-[a-z]{1,32}-[a-z0-9-]{1,60}")  # matched with fullmatch
MAX_ANALYSIS = 32
MAX_SLUG = 40
MAX_ID_SUFFIX = 1000  # attempts at a free folder name within the same second
ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
FALLBACK_SLUG = "report"
FALLBACK_ANALYSIS = "report"
HTML_FILE, JSON_FILE, SIDECAR_FILE = "report.html", "report.json", "meta.json"
FORMATS: dict[str, str] = {"html": HTML_FILE, "json": JSON_FILE}
ENCODING = "utf-8"
SLUG_INPUTS = 2  # inputs that make up a slug when no title is given


class InvalidReportId(ValueError):
    """A report id that does not match the id format, or escapes the reports folder."""


class ReportNotFound(LookupError):
    """A well-formed report id with no (complete) report behind it."""


@dataclass(frozen=True)
class ReportRecord:
    """One saved report, as listed (from its sidecar)."""

    id: str
    analysis: str
    title: str
    created: str  # when the analysis ran (the result's ``meta.created``)
    inputs: dict[str, str]
    summary: str
    risk: str  # assessment risk level (clean .. high); empty for the per-tool analyses
    model_format: str  # storage format of the analysed model (front-end supplied); empty when unknown
    folder: str
    html_path: str
    json_path: str
    size_bytes: int  # report.html + report.json


class ReportStore:
    """Saves, lists, serves and deletes reports under one root folder."""

    def __init__(self, root: str = DEFAULT_ROOT) -> None:
        """Use ``root`` (created on the first save).

        Args:
            root: folder holding one sub-folder per report.
        """
        self.root = root

    # ------------------------------------------------------------- writing
    def save(self, result: Any, title: str | None = None, extra: dict[str, str] | None = None) -> ReportRecord:
        """Save a result as JSON, HTML and sidecar in a new report folder.

        Args:
            result: an analysis result dataclass (or its JSON-ready dict).
            title: human title; also gives the id slug. Defaults to the inputs.
            extra: front-end listing fields the result itself cannot carry (e.g. the analysed
                model's storage format), merged into the sidecar.

        Returns:
            The record of the new report.

        Raises:
            ValueError: when the result holds values JSON cannot represent.
        """
        data = to_jsonable(result)
        report_title = title or _default_title(data)
        folder = self._new_folder(_analysis_of(data), _slugify(title or _inputs_slug(data)))
        try:
            _write_atomic(os.path.join(folder, JSON_FILE), _json_text(data))
            _write_atomic(os.path.join(folder, HTML_FILE), render_html(data))
            _write_atomic(os.path.join(folder, SIDECAR_FILE), _json_text(_sidecar(folder, data, report_title, extra)))
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return self.get(os.path.basename(folder))

    def rerender(self, report_id: str) -> ReportRecord:
        """Rebuild ``report.html`` from ``report.json`` (after a template update, for instance).

        Raises:
            InvalidReportId: for a malformed id.
            ReportNotFound: when the report does not exist.
        """
        record = self.get(report_id)
        _write_atomic(record.html_path, render_html(self.load(report_id)))
        return self.get(report_id)

    def delete(self, report_id: str) -> None:
        """Remove a report folder and everything in it.

        Raises:
            InvalidReportId: for a malformed id.
            ReportNotFound: when the report does not exist.
        """
        folder = self._existing_folder(report_id)
        shutil.rmtree(folder)

    # ------------------------------------------------------------- reading
    def list(self) -> builtins.list[ReportRecord]:
        """Every complete report, newest first.

        Folders that are not report folders, or whose sidecar is missing or
        corrupt, are skipped.
        """
        root = os.path.realpath(self.root)
        names = sorted(os.listdir(root)) if os.path.isdir(root) else []
        records = [entry for entry in map(self._listed, names) if entry is not None]
        records.sort(key=lambda entry: entry[0], reverse=True)
        return [record for _, record in records]

    def get(self, report_id: str) -> ReportRecord:
        """The record of one report.

        Raises:
            InvalidReportId: for a malformed id.
            ReportNotFound: when the report does not exist or its sidecar is unreadable.
        """
        folder = self._existing_folder(report_id)
        entry = self._listed(os.path.basename(folder))
        if entry is None:
            raise ReportNotFound(f"Report {report_id} is incomplete or corrupt")
        return entry[1]

    def path(self, report_id: str, fmt: str) -> str:
        """Path of a report file, for downloads.

        Args:
            report_id: the report.
            fmt: ``"html"`` or ``"json"``.

        Raises:
            ValueError: for an unknown format.
            InvalidReportId: for a malformed id.
            ReportNotFound: when the report or that file does not exist.
        """
        if fmt not in FORMATS:
            raise ValueError(f"Unknown report format {fmt!r}; expected one of {', '.join(FORMATS)}")
        path = os.path.join(self._existing_folder(report_id), FORMATS[fmt])
        if not os.path.isfile(path):
            raise ReportNotFound(f"Report {report_id} has no {fmt} file")
        return path

    def load(self, report_id: str) -> dict[str, Any]:
        """The report data (``report.json``).

        Raises:
            InvalidReportId: for a malformed id.
            ReportNotFound: when the report does not exist.
        """
        with open(self.path(report_id, "json"), encoding=ENCODING) as handle:
            data: dict[str, Any] = json.load(handle)
        return data

    # ------------------------------------------------------------- internals
    def _folder(self, report_id: str) -> str:
        """Real path of a report folder, after validating the id (the folder may not exist).

        Raises:
            InvalidReportId: for a malformed id or a path outside the root.
        """
        if not isinstance(report_id, str) or REPORT_ID_PATTERN.fullmatch(report_id) is None:
            raise InvalidReportId(f"Invalid report id: {report_id!r}")
        root = os.path.realpath(self.root)
        folder = os.path.realpath(os.path.join(root, report_id))
        if folder == root or os.path.commonpath([root, folder]) != root:
            raise InvalidReportId(f"Report id outside the reports folder: {report_id!r}")
        return folder

    def _existing_folder(self, report_id: str) -> str:
        """Real path of an existing report folder.

        Raises:
            InvalidReportId: for a malformed id.
            ReportNotFound: when the folder does not exist.
        """
        folder = self._folder(report_id)
        if not os.path.isdir(folder):
            raise ReportNotFound(f"No report {report_id}")
        return folder

    def _listed(self, name: str) -> tuple[tuple[str, str], ReportRecord] | None:
        """(sort key, record) of a folder name, or None when it is not a complete report."""
        result = None
        try:
            folder = self._folder(name)
            with open(os.path.join(folder, SIDECAR_FILE), encoding=ENCODING) as handle:
                sidecar = json.load(handle)
            result = (str(sidecar.get("saved", "")), name), _record(folder, name, sidecar)
        except (InvalidReportId, OSError, ValueError, TypeError, KeyError, AttributeError):
            result = None
        return result

    def _new_folder(self, analysis: str, slug: str) -> str:
        """Create a fresh report folder named after the current time, the analysis and the slug."""
        os.makedirs(self.root, exist_ok=True)
        stem = f"{datetime.now().strftime(ID_TIME_FORMAT)}-{analysis}-{slug}"
        folder = None
        for attempt in range(1, MAX_ID_SUFFIX + 1):
            candidate = self._folder(stem if attempt == 1 else f"{stem}-{attempt}")
            try:
                os.makedirs(candidate)
            except FileExistsError:
                continue
            folder = candidate
            break
        if folder is None:
            raise FileExistsError(f"No free report folder for {stem}")
        return folder


def summary(data: dict[str, Any]) -> str:
    """One human line describing a result (its JSON form), for report lists.

    Examples: ``6/26 tensors changed (max rel 0.22)``, ``best: org/base — DERIVED,
    same basis (...)``, ``mean KL 0.0083 (control z 4.1)``.
    """
    analysis = _analysis_of(data)
    summarise = SUMMARIES.get(analysis)
    return summarise(data) if summarise is not None else _generic_summary(data)


def _weights_summary(data: dict[str, Any]) -> str:
    """``6/26 tensors changed (max rel 0.22)``, plus the count beyond quantization noise."""
    tensors = data.get("tensors") or []
    top = tensors[0].get("rel_change") if tensors else None
    text = f"{format_number(data.get('changed'))}/{format_number(data.get('compared'))} tensors changed"
    text += f" (max rel {format_number(top, 2)})" if top is not None else ""
    beyond = data.get("beyond_noise")
    return text + (f", {format_number(beyond)} beyond quantization noise" if beyond is not None else "")


def _shifts_summary(data: dict[str, Any]) -> str:
    """``top update <name> σ1 0.58``, or the change counts when nothing was projected."""
    updates = data.get("top_updates") or []
    counts = data.get("counts") or {}
    result = f"{format_number(counts.get('changed'))}/{format_number(counts.get('compared'))} tensors changed"
    if updates:
        result = f"top update {updates[0].get('name')} σ1 {format_number(updates[0].get('sigma'), 2)}"
    return result


def _entropy_summary(data: dict[str, Any]) -> str:
    """``mean spectral ΔH -0.07 (control z -1.5)``, or the tensor count when nothing was read spectrally."""
    counts = data.get("counts") or {}
    spectral_null = data.get("spectral_null") or {}
    mean = spectral_null.get("value")
    result = f"{format_number(counts.get('in_scope'))} tensors examined"
    if mean is not None:
        z = spectral_null.get("z")
        result = f"mean spectral ΔH {format_number(mean, 2)}"
        result += f" (control z {format_number(z, 2)})" if z is not None else ""
    return result


def _lineage_summary(data: dict[str, Any]) -> str:
    """``3 pairs, 1 failed``."""
    pairs = data.get("pairs") or []
    failed = sum(1 for pair in pairs if pair.get("error"))
    return f"{len(pairs)} pairs, {failed} failed"


def _provenance_summary(data: dict[str, Any]) -> str:
    """``best: <ref> — <verdict>``, or why there is none."""
    best = data.get("best")
    ranking = data.get("ranking") or []
    verdict = next((row.get("verdict") for row in ranking if row.get("ref") == best), None)
    result = f"no weight-level match ({len(ranking)} candidates compared)"
    if best is not None:
        result = f"best: {best} — {verdict}"
    return result


def _assessment_summary(data: dict[str, Any]) -> str:
    """``high risk, 2 red flags``, with the confidence when it is not the default."""
    verdict = data.get("verdict") or {}
    level = str(verdict.get("level") or "unknown")
    flags = verdict.get("flags")
    result = f"{level} risk"
    if flags is not None:
        result += f", {format_number(flags)} red flag{'' if flags == 1 else 's'}"
    confidence = str(verdict.get("confidence") or "")
    return result + (f" (confidence {confidence})" if confidence and confidence != "normal" else "")


def _behaviour_summary(data: dict[str, Any]) -> str:
    """``mean KL 0.0083 (control z 4.1)``, or the generation count when KL was not measured."""
    control = data.get("control") or {}
    mean_kl = data.get("mean_kl")
    result = f"no KL measured, {len(data.get('generations') or [])} generations"
    if mean_kl is not None:
        result = f"mean KL {format_number(mean_kl, 2)}"
        result += f" (control z {format_number(control.get('z'), 2)})" if control.get("z") is not None else ""
    return result


def _generic_summary(data: dict[str, Any]) -> str:
    """The analysis name and inputs of an unknown result."""
    return f"{_analysis_of(data)}: {_inputs_text(data) or 'no inputs'}"


SUMMARIES = {
    "weights": _weights_summary,
    "shifts": _shifts_summary,
    "entropy": _entropy_summary,
    "lineage": _lineage_summary,
    "provenance": _provenance_summary,
    "behaviour": _behaviour_summary,
    "assessment": _assessment_summary,
}


def _record(folder: str, name: str, sidecar: dict[str, Any]) -> ReportRecord:
    """A record from a sidecar; the sidecar must describe this very folder.

    Raises:
        ValueError: when the sidecar names another report.
    """
    if sidecar["id"] != name:
        raise ValueError(f"Sidecar of {name} describes {sidecar['id']}")
    html_path, json_path = os.path.join(folder, HTML_FILE), os.path.join(folder, JSON_FILE)
    return ReportRecord(
        id=name,
        analysis=str(sidecar["analysis"]),
        title=str(sidecar["title"]),
        created=str(sidecar["created"]),
        inputs={str(key): str(value) for key, value in dict(sidecar["inputs"]).items()},
        summary=str(sidecar["summary"]),
        risk=str(sidecar.get("risk") or _legacy_risk(sidecar)),
        model_format=str(sidecar.get("model_format", "")),
        folder=folder,
        html_path=html_path,
        json_path=json_path,
        size_bytes=sum(os.path.getsize(path) for path in (html_path, json_path) if os.path.isfile(path)),
    )


def _sidecar(folder: str, data: dict[str, Any], title: str, extra: dict[str, str] | None = None) -> dict[str, Any]:
    """The listing data of a report, with any front-end ``extra`` fields merged in."""
    meta = _meta(data)
    return {
        "id": os.path.basename(folder),
        "analysis": _analysis_of(data),
        "title": title,
        "created": str(meta.get("created") or ""),
        "saved": datetime.now().isoformat(timespec="microseconds"),
        "inputs": {str(key): str(value) for key, value in _inputs(data).items()},
        "summary": summary(data),
        "risk": _risk_of(data),
        **{str(key): str(value) for key, value in (extra or {}).items()},
    }


def _json_text(data: Any) -> str:
    """Strict, readable JSON (no NaN, non-ASCII kept)."""
    return json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False)


def _write_atomic(path: str, text: str) -> None:
    """Write ``text`` as UTF-8 through a temporary file renamed over ``path``."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding=ENCODING, dir=os.path.dirname(path), prefix=".tmp-", suffix=".part", delete=False
        ) as handle:
            temp_path = handle.name
            handle.write(text)
        os.replace(temp_path, path)
    except BaseException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _slugify(text: str) -> str:
    """Lowercase ``[a-z0-9-]`` slug of at most :data:`MAX_SLUG` characters (accents dropped)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:MAX_SLUG].strip("-")
    return slug or FALLBACK_SLUG


def _inputs_slug(data: dict[str, Any]) -> str:
    """Slug source from the inputs: the last component of local paths, Hub ids as they are."""
    names = [_short_name(str(value)) for value in _inputs(data).values() if value]
    return "-vs-".join(reversed(names[:SLUG_INPUTS]))


def _short_name(value: str) -> str:
    """The last path component of a local path; a Hub id (``org/model``) is kept whole."""
    return model_name(value)


def _default_title(data: dict[str, Any]) -> str:
    """Title used when none is given: ``<analysis>: <inputs>``."""
    inputs = _inputs_text(data)
    return f"{_analysis_of(data)}: {inputs}" if inputs else _analysis_of(data)


def _inputs_text(data: dict[str, Any]) -> str:
    """The inputs as ``key=value`` pairs."""
    return ", ".join(f"{key}={value}" for key, value in _inputs(data).items())


def _analysis_of(data: dict[str, Any]) -> str:
    """The analysis name reduced to ``[a-z]+`` (it is part of the report id)."""
    name = re.sub(r"[^a-z]", "", str(_meta(data).get("analysis") or "").lower())
    return name[:MAX_ANALYSIS] or FALLBACK_ANALYSIS


def _meta(data: Any) -> dict[str, Any]:
    """The ``meta`` mapping of a result, or an empty one."""
    meta = data.get("meta") if isinstance(data, dict) else None
    return meta if isinstance(meta, dict) else {}


def _inputs(data: dict[str, Any]) -> dict[str, Any]:
    """The ``meta.inputs`` mapping of a result, or an empty one."""
    inputs = _meta(data).get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _risk_of(data: dict[str, Any]) -> str:
    """The assessment risk level (for the Analysis report list); empty for the per-tool analyses."""
    verdict = data.get("verdict") if _analysis_of(data) == "assessment" else None
    level = verdict.get("level") if isinstance(verdict, dict) else None
    return str(level) if level else ""


def _legacy_risk(sidecar: dict[str, Any]) -> str:
    """Risk level of an assessment sidecar saved before the ``risk`` field, read from its summary.

    :func:`_assessment_summary` builds that summary as ``<level> risk, ...``, so the level is its
    first word. Returns the empty string for a per-tool report or a sidecar with no summary.
    """
    text = str(sidecar.get("summary", "")) if str(sidecar.get("analysis", "")) == "assessment" else ""
    return text.split(" ", 1)[0] if text else ""
