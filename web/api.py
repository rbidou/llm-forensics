"""JSON endpoints over the engine: validate the request shape, delegate, wrap.

Every route answers with the standard envelope from
:mod:`web.responses`, built once at the end of the function. The
engine returns ``(result, message, data)``; on failure ``data["reason"]`` picks
the HTTP status (:data:`REASON_STATUS`) and the envelope's data is empty.

The report downloads are the only non-JSON answers: they send the file as an
attachment (their refusals still use the envelope).
"""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, Response, current_app, request, send_file, url_for

from web.engine import (
    MAX_FILE_SIZE,
    REASON_BUSY,
    REASON_DISABLED,
    REASON_INVALID,
    REASON_NOT_FOUND,
    REASON_UNPROCESSABLE,
    ForensicsEngine,
)
from web.responses import (
    HTTP_BAD_REQUEST,
    WRONG_FORMAT_MESSAGE,
    ApiResponse,
    FunctionResult,
    check_json_data,
    make_api_response,
)

HTTP_NOT_FOUND = 404
HTTP_UNPROCESSABLE = 422
ENGINE_KEY = "llmforensics_engine"
JOB_PAGES = "any(weights, entropy, shifts, vocab, lineage, provenance, behaviour, assessment)"
REPORT_FORMATS = "any(html, json)"

# Job kind -> the page endpoint the busy notification links to (``assessment`` lives on Home).
KIND_ENDPOINTS: dict[str, str] = {
    "weights": "forensics_weights",
    "entropy": "forensics_entropy",
    "shifts": "forensics_shifts",
    "vocab": "forensics_vocab",
    "lineage": "forensics_lineage",
    "provenance": "forensics_provenance",
    "behaviour": "forensics_behaviour",
    "assessment": "forensics_home",
}

REASON_STATUS: dict[str, int] = {
    REASON_INVALID: HTTP_BAD_REQUEST,
    REASON_NOT_FOUND: HTTP_NOT_FOUND,
    REASON_BUSY: HTTP_UNPROCESSABLE,
    REASON_UNPROCESSABLE: HTTP_UNPROCESSABLE,
    REASON_DISABLED: HTTP_UNPROCESSABLE,
}

api = Blueprint("api", __name__, url_prefix="/api/forensics")


def engine() -> ForensicsEngine:
    """The application's engine (one per application, holding the shared job runner)."""
    found: ForensicsEngine = current_app.extensions[ENGINE_KEY]
    return found


def respond(outcome: FunctionResult) -> ApiResponse:
    """Wrap an engine result in the envelope, with the status its failure reason implies.

    Args:
        outcome: the engine's (result, message, data).

    Returns:
        The envelope and its HTTP status.
    """
    (result, message, data) = outcome
    status_code = 200 if result else REASON_STATUS.get(str(data.get("reason", "")), HTTP_UNPROCESSABLE)
    return make_api_response(result, message, data if result else {}, status_code, True)


# --- environment and detection ----------------------------------------------


@api.get("/environment")
def api_forensics_environment() -> ApiResponse:
    """Platform, Python, GPU, packages, features and torch CUDA availability."""
    return respond(engine().environment({}))


@api.get("/system")
def api_forensics_system() -> ApiResponse:
    """The menu bar's live readout: the listening address, RAM and GPU.

    The bound address is read from the running application (not the stored setting), since a
    running server keeps whatever socket it started on.
    """
    host = current_app.config.get("HOST")
    port = current_app.config.get("PORT")
    return respond(engine().system_info({"host": host, "port": port}))


@api.post("/detect")
@check_json_data
def api_forensics_detect(json_data: dict[str, Any]) -> ApiResponse:
    """Say what a model reference is and which choices it leaves open."""
    return respond(engine().detect(json_data))


@api.get("/hub/search")
def api_forensics_hub_search() -> ApiResponse:
    """Hub model ids matching the ``q`` query, for the pickers' auto-completion."""
    return respond(engine().hub_search({"q": request.args.get("q", "")}))


@api.get("/folders")
def api_forensics_folders() -> ApiResponse:
    """The subfolders and GGUF files of the ``path`` folder, for the Browse navigator."""
    return respond(engine().list_folder({"path": request.args.get("path", "")}))


# --- setup ------------------------------------------------------------------


@api.get("/setup")
def api_forensics_setup() -> ApiResponse:
    """The stored web server address and its defaults."""
    return respond(engine().setup({}))


@api.post("/setup/server")
@check_json_data
def api_forensics_setup_server(json_data: dict[str, Any]) -> ApiResponse:
    """Store the web server address (applies at the next start); empty values restore the defaults."""
    return respond(engine().set_server(json_data))


@api.post("/setup/directories")
@check_json_data
def api_forensics_setup_directories(json_data: dict[str, Any]) -> ApiResponse:
    """Store the report and models directories (apply at once); empty values restore the defaults."""
    return respond(engine().set_directories(json_data))


# --- analyses (each starts a job) --------------------------------------------


@api.post("/weights")
@check_json_data
def api_forensics_weights(json_data: dict[str, Any]) -> ApiResponse:
    """Start a Weight Diff job."""
    return respond(engine().weights(json_data))


@api.post("/shifts")
@check_json_data
def api_forensics_shifts(json_data: dict[str, Any]) -> ApiResponse:
    """Start a Token Shifts job."""
    return respond(engine().shifts(json_data))


@api.post("/vocab")
@check_json_data
def api_forensics_vocab(json_data: dict[str, Any]) -> ApiResponse:
    """Start a Vocabulary & Embeddings job."""
    return respond(engine().vocab(json_data))


@api.post("/entropy")
@check_json_data
def api_forensics_entropy(json_data: dict[str, Any]) -> ApiResponse:
    """Start an Entropy Analysis job."""
    return respond(engine().entropy(json_data))


@api.post("/lineage/discover")
@check_json_data
def api_forensics_lineage_discover(json_data: dict[str, Any]) -> ApiResponse:
    """Start a Hub Lineage discovery job."""
    return respond(engine().lineage_discover(json_data))


@api.post("/lineage")
@check_json_data
def api_forensics_lineage(json_data: dict[str, Any]) -> ApiResponse:
    """Start the Hub Lineage analysis of the chosen pairs."""
    return respond(engine().lineage(json_data))


@api.post("/provenance")
@check_json_data
def api_forensics_provenance(json_data: dict[str, Any]) -> ApiResponse:
    """Start a Provenance job."""
    return respond(engine().provenance(json_data))


@api.post("/assessment")
@check_json_data
def api_forensics_assessment(json_data: dict[str, Any]) -> ApiResponse:
    """Start a suspicion assessment workflow job."""
    return respond(engine().assessment(json_data))


@api.post("/behaviour")
def api_forensics_behaviour() -> ApiResponse:
    """Start a batch Behaviour Probe on an uploaded prompts file (multipart, no JSON decorator).

    The form carries ``file`` and ``params`` (a JSON object with the job id, the
    models and the options).
    """
    params = _form_params()
    uploaded = request.files.get("file")

    response: ApiResponse
    if params is None:
        response = make_api_response(False, WRONG_FORMAT_MESSAGE, {}, HTTP_BAD_REQUEST, True)
    else:
        # One byte past the limit is enough for the engine to refuse an oversized file.
        content = uploaded.read(MAX_FILE_SIZE + 1) if uploaded is not None else None
        file_name = (uploaded.filename or "") if uploaded is not None else ""
        response = respond(engine().behaviour({**params, "file_name": file_name, "file_content": content}))

    return response


def _form_params() -> dict[str, Any] | None:
    """The ``params`` JSON object of a multipart form, or None when it is not one."""
    parsed: Any = None
    try:
        parsed = json.loads(request.form.get("params", ""))
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else None


# --- interactive behaviour session --------------------------------------------


@api.get("/behaviour/session")
def api_forensics_behaviour_session() -> ApiResponse:
    """The interactive session as it stands."""
    return respond(engine().behaviour_session({}))


@api.post("/behaviour/session")
@check_json_data
def api_forensics_behaviour_session_load(json_data: dict[str, Any]) -> ApiResponse:
    """Start loading a model pair for the session (the previous one is closed)."""
    return respond(engine().behaviour_load(json_data))


@api.post("/behaviour/session/prompt")
@check_json_data
def api_forensics_behaviour_session_prompt(json_data: dict[str, Any]) -> ApiResponse:
    """Compare both loaded models on one prompt (waits for the result)."""
    return respond(engine().behaviour_prompt(json_data))


@api.post("/behaviour/session/save")
@check_json_data
def api_forensics_behaviour_session_save(json_data: dict[str, Any]) -> ApiResponse:
    """Start probing the session's prompts on the loaded pair, saved as a report."""
    return respond(engine().behaviour_save(json_data))


@api.delete("/behaviour/session")
def api_forensics_behaviour_session_unload() -> ApiResponse:
    """Close the loaded pair."""
    return respond(engine().behaviour_unload({}))


# --- jobs ----------------------------------------------------------------------


@api.get("/jobs/current")
def api_forensics_current_job() -> ApiResponse:
    """The running or most recently finished job, with a link to its page (for reattach and notifications)."""
    (result, message, data) = engine().current_job({})
    job = data.get("job") if result else None
    if job is not None:
        job["url"] = url_for(KIND_ENDPOINTS[job["kind"]]) if job["kind"] in KIND_ENDPOINTS else None
    return respond((result, message, data))


@api.get(f"/<{JOB_PAGES}:page>/<job_id>")
def api_forensics_poll(page: str, job_id: str) -> ApiResponse:
    """A job's state, progress, events since ``?cursor=n``, and result or error."""
    cursor = request.args.get("cursor", "0")
    return respond(engine().poll(page, job_id, int(cursor) if cursor.isdigit() else 0))


@api.post("/jobs/<job_id>/cancel")
def api_forensics_cancel(job_id: str) -> ApiResponse:
    """Ask a running job to stop at its next checkpoint."""
    return respond(engine().cancel(job_id))


# --- reports -------------------------------------------------------------------


@api.get("/reports")
def api_forensics_reports() -> ApiResponse:
    """Every saved report, newest first."""
    return respond(engine().reports({}))


@api.get(f"/reports/<report_id>/<{REPORT_FORMATS}:fmt>")
def api_forensics_report_download(report_id: str, fmt: str) -> Response | ApiResponse:
    """Send a report as an attachment: HTML or JSON."""
    (result, message, data) = engine().report_download({"id": report_id, "format": fmt})

    response: Response | ApiResponse
    if not result:
        response = respond((result, message, data))
    else:
        response = send_file(
            data["path"], mimetype=data["mimetype"], as_attachment=True, download_name=data["filename"]
        )

    return response


@api.delete("/reports/<report_id>")
def api_forensics_report_delete(report_id: str) -> ApiResponse:
    """Delete a saved report."""
    return respond(engine().delete_report({"id": report_id}))
