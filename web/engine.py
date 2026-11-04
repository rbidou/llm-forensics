"""The web front-end's engine layer: every decision the browser asks for, without Flask.

Each public method takes one ``params`` dict and returns ``(result, message,
data)`` built by :func:`~web.responses.make_function_result`. On
failure ``data`` carries only a ``reason`` (``invalid``, ``not_found``, ``busy``,
``unprocessable`` or ``disabled``), which the routes map to an HTTP status.

Long analyses run on ONE shared :class:`~jobs.JobRunner`, so two
of them never compete for the same memory. Each job runs inside
:func:`~progress.quiet`, saves its report when it succeeds and
leaves a JSON-ready payload on the job for the browser to poll.

The interactive Behaviour Probe keeps at most one loaded model pair. Every
inference on it (loading, one prompt, saving) is a job on the same runner, so it
is serialised with the analyses; a request arriving while another job runs gets
a "busy" answer instead of a second model on the GPU.

Engine modules that import torch are imported inside the functions using them,
so the server starts (with features disabled) on a machine without torch, and
tests can replace them with fakes.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from environment import Capabilities, GpuInfo, MemoryInfo, detect_gpu, detect_memory, features
from environment import detect as detect_environment
from jobs import JOB_ID_PATTERN, InvalidJobId, Job, JobBusy, JobRunner, JobState
from models.detect import gguf_by_precision
from progress import CancelToken, ProgressEvent, ProgressSink, Stage, quiet, report
from results import to_jsonable
from settings import (
    DEFAULT_ADDRESS,
    DEFAULT_ENTROPY_BINS,
    DEFAULT_ENTROPY_LAYERS,
    DEFAULT_FIELD,
    DEFAULT_LAYERS,
    DEFAULT_LIMIT,
    DEFAULT_MAX_SIZE_GB,
    DEFAULT_TOKENS,
    DEFAULT_TOP,
    DEFAULT_TOP_K,
    DEFAULT_UPDATES,
    DEFAULT_VARIANTS,
    DEFAULT_VOCAB_NEW,
    DEFAULT_VOCAB_ROWS,
    HUB_SEARCH_LIMIT,
    MAX_ENTROPY_BINS,
    MAX_ENTROPY_LAYERS,
    MAX_GENERATE,
    MAX_LAYERS,
    MAX_LIMIT,
    MAX_TOKENS,
    MAX_TOP,
    MAX_TOP_K,
    MAX_UPDATES,
    MAX_VARIANTS,
    MAX_VOCAB_NEW,
    MAX_VOCAB_ROWS,
    WILDCARD_HOSTS,
    DirectoryError,
    ServerAddressError,
    __version__,
    default_reports_root,
    local_addresses,
    models_root,
    recommended_tools,
    reports_root,
    server_address,
    set_directories,
    set_server_address,
    torch_cuda_status,
)
from web.responses import FunctionResult, make_function_result
from web.summary import (
    behaviour_summary,
    entropy_summary,
    lineage_summary,
    provenance_summary,
    shifts_summary,
    vocab_summary,
    weights_summary,
)

# Failure reasons, mapped to HTTP status codes by the routes.
REASON_INVALID = "invalid"
REASON_NOT_FOUND = "not_found"
REASON_BUSY = "busy"
REASON_UNPROCESSABLE = "unprocessable"
REASON_DISABLED = "disabled"

# Page id -> environment feature gating it (None: always available). ``assessment`` is the
# workflow run hosted on the Home page: it has an availability (gated like the weight tools it
# runs) but no page route or template of its own, so it is listed in ROUTELESS_PAGES.
PAGE_FEATURES: dict[str, str | None] = {
    "home": None,
    "weights": "Weight Diff",
    "entropy": "Entropy Analysis",
    "shifts": "Token Shifts",
    "vocab": "Vocabulary & Embeddings",
    "lineage": "Hub Lineage",
    "provenance": "Provenance",
    "behaviour": "Behaviour Probe",
    "assessment": "Weight Diff",
    "reports": "Reports",
    "analysis": None,
    "environment": None,
    "setup": None,
}
# Pages with an availability but no page route or template (job kinds hosted on another page).
ROUTELESS_PAGES: tuple[str, ...] = ("assessment",)
JOB_PAGES: tuple[str, ...] = (
    "weights",
    "entropy",
    "shifts",
    "vocab",
    "lineage",
    "provenance",
    "behaviour",
    "assessment",
)
# Human name of each tracked process, shown in the finish and busy notifications and used
# by a returning page to label a reattached run. A job started with an empty label (the
# interactive Behaviour Probe session jobs) is not a tracked process.
JOB_LABELS: dict[str, str] = {
    "weights": "Weight Diff",
    "entropy": "Entropy Analysis",
    "shifts": "Token Shifts",
    "vocab": "Vocabulary & Embeddings",
    "lineage": "Hub Lineage",
    "provenance": "Provenance",
    "behaviour": "Behaviour Probe",
    "assessment": "Model Analysis",
}

# Web-only limits (the shared ones come from settings).
MAX_FILE_SIZE = 2 * 1024 * 1024
MIN_ENTROPY_BINS = 2  # a histogram needs at least two buckets to have any entropy
LINEAGE_MODES: tuple[str, ...] = ("auto", "parent", "children")
MAX_PROMPT_CHARS = 20_000
MAX_CANDIDATES = 50
MAX_PROMPTS = 100  # behaviour target prompts accepted by the workflow form
MAX_TEXT = 500
MAX_PATH = 4096

# What the polling payloads keep of large results.
TENSORS_SHOWN = 50
PAIR_UPDATES_SHOWN = 5
DISCOVERIES_KEPT = 8
WAIT_SECONDS = 0.02

REPORT_VIEW_URL = "/forensics/reports/{id}/view"
REPORT_FILE_URL = "/api/forensics/reports/{id}/{fmt}"
DOWNLOADS: dict[str, tuple[str, str]] = {
    "html": ("report.html", "text/html; charset=utf-8"),
    "json": ("report.json", "application/json"),
}

Work = Callable[[ProgressSink, CancelToken], Any]
Presenter = Callable[[dict[str, Any]], dict[str, Any]]


class InvalidInput(ValueError):
    """A request parameter that cannot be used; ``reason`` picks the status code."""

    def __init__(self, message: str, reason: str = REASON_UNPROCESSABLE) -> None:
        """Keep the user-facing message and the failure reason."""
        super().__init__(message)
        self.reason = reason


def failure(message: str, reason: str = REASON_UNPROCESSABLE) -> FunctionResult:
    """A failed engine result carrying the reason the route maps to a status code.

    Args:
        message: user-facing explanation.
        reason: one of the ``REASON_*`` constants.

    Returns:
        The (False, message, {"reason": reason}) tuple.
    """
    return make_function_result(False, message, {"reason": reason})


# ======================================================================= model choices
@dataclass(frozen=True)
class ModelSpec:
    """A model reference and the choices the Detect step left to the user."""

    ref: str
    gguf_file: str | None = None  # chosen file of a GGUF folder or repository holding several
    base_override: str | None = None  # base an adapter is applied to, instead of the declared one

    def detect(self, hf_token: str | None) -> Any:
        """Detect the reference (file names and config only, never weights).

        Raises:
            ModelError: when the reference cannot be used.
        """
        from models import detect

        return detect(self.ref, token=hf_token)

    def open(self, model: Any, hf_token: str | None) -> Any:
        """Open the detected model's weights (the adapter on its base, the chosen GGUF file)."""
        from models import GGUF_AUTO, open_model

        return open_model(model, base=self.base_override, token=hf_token, gguf_file=self.gguf_file or GGUF_AUTO)

    def inputs(self, role: str) -> dict[str, str]:
        """This model as recorded in the result metadata, with the user's choices."""
        inputs = {role: self.ref}
        if self.gguf_file:
            inputs[f"{role}_gguf_file"] = os.path.basename(self.gguf_file)
        if self.base_override:
            inputs[f"{role}_adapter_base"] = self.base_override
        return inputs


@dataclass
class BehaviourSession:
    """The interactive Behaviour Probe: one loaded pair and what was asked of it."""

    pair: Any
    inputs: dict[str, str]
    prompts: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


# ======================================================================= the engine
class ForensicsEngine:
    """Business logic of the web front-end (no Flask, no request objects)."""

    def __init__(
        self,
        reports_root: str,
        *,
        hf_token: str | None = None,
        runner: JobRunner | None = None,
        capabilities: Capabilities | None = None,
        upload_dir: str | None = None,
        repo_factory: Any = None,
    ) -> None:
        """Set up the shared job runner, the report store and the session slot.

        Args:
            reports_root: folder of the report store.
            hf_token: Hub token for private or gated repositories.
            runner: the job runner (one is created when omitted).
            capabilities: the machine's capabilities (detected on first use when omitted).
            upload_dir: where uploaded prompt files are written briefly (system temp when omitted).
            repo_factory: builds Hub access objects for the report-listing format lookup; the real
                one is used when omitted (tests inject a fake so listing never reaches the network).
        """
        from reporting import ReportStore

        self.store = ReportStore(reports_root)
        self.hf_token = hf_token
        self.runner = runner or JobRunner()
        self.upload_dir = upload_dir
        self.repo_factory = repo_factory
        self._capabilities = capabilities
        self._discoveries: OrderedDict[str, Any] = OrderedDict()
        self._session: BehaviourSession | None = None
        self._session_lock = threading.Lock()
        self._discoveries_lock = threading.Lock()

    # ------------------------------------------------------------------ environment
    def page_features(self) -> dict[str, dict[str, Any]]:
        """Availability of every page, from the capabilities detected once.

        Returns:
            Page id -> {"available": bool, "reason": str}.
        """
        gates = {item.label: item for item in features(self._known_capabilities())}
        result: dict[str, dict[str, Any]] = {}
        for page, gate in PAGE_FEATURES.items():
            feature = gates.get(gate) if gate is not None else None
            available = feature is None or feature.available
            result[page] = {"available": available, "reason": feature.reason if feature is not None else ""}
        return result

    def environment(self, params: dict[str, Any]) -> FunctionResult:
        """Observe the machine again: platform, Python, GPU, packages, features, torch CUDA.

        Args:
            params: unused (the environment takes no input).

        Returns:
            Tuple of (result, message, data).
        """
        capabilities = detect_environment()
        self._capabilities = capabilities
        gpu = capabilities.gpu
        data = {
            "version": __version__,
            "platform": capabilities.platform_description,
            "python": capabilities.python_version,
            "executable": capabilities.python_executable,
            "gpu": dataclasses.asdict(gpu) if gpu is not None else None,
            "cuda": torch_cuda_status(),
            "packages": [{"name": name, "version": version} for name, version in capabilities.packages.items()],
            "features": [dataclasses.asdict(item) for item in features(capabilities)],
        }
        return make_function_result(True, "", data)

    def system_info(self, params: dict[str, Any]) -> FunctionResult:
        """The menu bar's live readout: the listening address, RAM and GPU.

        Detects RAM and the GPU afresh on every call (no network, no torch), so the figures
        stay current while the browser polls them.

        Args:
            params: ``host`` and ``port`` the running server is bound to (read from the live
                application, since a running server keeps its socket whatever the stored setting).

        Returns:
            Tuple of (result, message, data) with ``address``, ``ram`` and ``gpu``; ``ram`` and
            ``gpu`` are None when the machine cannot report them.
        """
        host = str(params.get("host") or DEFAULT_ADDRESS.host)
        memory = detect_memory()
        data = {
            "address": address_payload(host, params.get("port")),
            "ram": ram_payload(memory),
            "gpu": gpu_payload(detect_gpu()),
        }
        return make_function_result(True, "", data)

    # ------------------------------------------------------------------ setup
    def setup(self, params: dict[str, Any]) -> FunctionResult:
        """The stored settings, with their defaults, for the Settings page.

        Args:
            params: unused (reading the settings takes no input).

        Returns:
            Tuple of (result, message, data) with the ``server`` and ``directories`` descriptions.
        """
        return make_function_result(True, "", {"server": describe_server(), "directories": describe_directories()})

    def set_server(self, params: dict[str, Any]) -> FunctionResult:
        """Store the address the web server listens on from its next start.

        Not refused while a job runs: the running server keeps its socket.

        Args:
            params: ``host`` and ``port``; empty values restore the defaults.

        Returns:
            Tuple of (result, message, data) with the new ``server`` description.
        """
        outcome: FunctionResult
        try:
            host = text_param(params, "host", "Listening IP address")
            port = params.get("port", "")
            if isinstance(port, bool) or not isinstance(port, (str, int)):
                raise InvalidInput("Port must be a number.")
            address = set_server_address(host, port)
            outcome = make_function_result(
                True, f"The web server will listen on {address.url} from its next start.", {"server": describe_server()}
            )
        except (InvalidInput, ServerAddressError) as error:
            outcome = failure(str(error), REASON_INVALID)
        except OSError as error:
            outcome = failure(f"The settings could not be saved: {error}")
        return outcome

    def set_directories(self, params: dict[str, Any]) -> FunctionResult:
        """Store the report directory and the default models directory; they apply at once.

        The report directory change moves this engine's report store immediately, so the
        Reports page and new saves use the new folder without a restart.

        Args:
            params: ``reports_dir`` and ``models_dir``; empty values restore the defaults.

        Returns:
            Tuple of (result, message, data) with the new ``directories`` description.
        """
        outcome: FunctionResult
        try:
            reports_dir = text_param(params, "reports_dir", "Report directory", maximum=MAX_PATH)
            models_dir = text_param(params, "models_dir", "Default models directory", maximum=MAX_PATH)
            set_directories(reports_dir, models_dir)
            self.store.root = str(reports_root())
            data = {"directories": describe_directories()}
            outcome = make_function_result(True, "The directories have been saved.", data)
        except (InvalidInput, DirectoryError) as error:
            outcome = failure(str(error), REASON_INVALID)
        except OSError as error:
            outcome = failure(f"The settings could not be saved: {error}")
        return outcome

    # ------------------------------------------------------------------ detection
    def detect(self, params: dict[str, Any]) -> FunctionResult:
        """Say what a model reference is and which choices it leaves open.

        Args:
            params: ``ref`` (local path or Hub id).

        Returns:
            Tuple of (result, message, data): kind, label, location, checkpoint
            step, declared base, GGUF files (most precise first) and, for a
            training-run folder, its checkpoints.
        """
        from models import ModelError

        outcome: FunctionResult
        try:
            ref = text_param(params, "ref", "Model reference", required=True)
            outcome = make_function_result(True, "", describe_reference(ref, self.hf_token))
        except InvalidInput as error:
            outcome = failure(str(error), error.reason)
        except ModelError as error:
            outcome = failure(str(error))
        except Exception as error:  # noqa: BLE001 - network or file errors are shown, never a traceback
            outcome = failure(f"Detection failed: {error.__class__.__name__}: {error}")
        return outcome

    # ------------------------------------------------------------------ model selection helpers
    def hub_search(self, params: dict[str, Any]) -> FunctionResult:
        """Hub model ids matching a typed query, for the pickers' auto-completion.

        Args:
            params: ``q``, the text typed so far.

        Returns:
            Tuple of (result, message, data) with ``models`` (possibly empty): a
            query that is too short, or a Hub that cannot be reached, yields none.
        """
        from models import search_hub_models

        outcome: FunctionResult
        try:
            query = text_param(params, "q", "Search text")
            candidates = search_hub_models(query, token=self.hf_token, limit=HUB_SEARCH_LIMIT)
            outcome = make_function_result(True, "", {"models": [hub_candidate_payload(item) for item in candidates]})
        except InvalidInput as error:
            outcome = failure(str(error), error.reason)
        return outcome

    def list_folder(self, params: dict[str, Any]) -> FunctionResult:
        """The subfolders and GGUF files of a folder, for the Browse navigator.

        Args:
            params: ``path``, the folder to list; empty opens on the default models
                directory, or the server's working directory when none is set.

        Returns:
            Tuple of (result, message, data) with the folder listing, or a
            failure when the folder cannot be read.
        """
        from models import FolderError
        from models import list_folder as list_model_folder

        outcome: FunctionResult
        try:
            path = text_param(params, "path", "Folder", maximum=MAX_PATH)
            outcome = make_function_result(True, "", folder_payload(list_model_folder(path or models_root())))
        except InvalidInput as error:
            outcome = failure(str(error), error.reason)
        except FolderError as error:
            outcome = failure(str(error), REASON_NOT_FOUND)
        return outcome

    # ------------------------------------------------------------------ analyses
    def weights(self, params: dict[str, Any]) -> FunctionResult:
        """Start a Weight Diff job.

        Args:
            params: ``job_id``, ``base`` and ``tuned`` model choices.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("weights", params, lambda: self._weights_work(params))

    def shifts(self, params: dict[str, Any]) -> FunctionResult:
        """Start a Token Shifts job.

        Args:
            params: ``job_id``, ``base``, ``tuned``, ``n`` (updates) and ``top`` (tokens).

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("shifts", params, lambda: self._shifts_work(params))

    def vocab(self, params: dict[str, Any]) -> FunctionResult:
        """Start a Vocabulary & Embeddings job.

        Args:
            params: ``job_id``, ``base``, ``tuned``, ``top_rows`` and ``new_tokens``.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("vocab", params, lambda: self._vocab_work(params))

    def entropy(self, params: dict[str, Any]) -> FunctionResult:
        """Start an Entropy Analysis job.

        Args:
            params: ``job_id``, ``base``, ``tuned``, ``layers`` (last blocks) and ``bins``.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("entropy", params, lambda: self._entropy_work(params))

    def lineage_discover(self, params: dict[str, Any]) -> FunctionResult:
        """Start a Hub Lineage discovery job (step 1: parents, children, pairs).

        Args:
            params: ``job_id``, ``model`` (Hub id), ``mode``, ``max_variants``, ``max_size_gb``.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("lineage", params, lambda: self._discovery_work(params))

    def lineage(self, params: dict[str, Any]) -> FunctionResult:
        """Start the Hub Lineage analysis of the chosen pairs (step 2).

        Args:
            params: ``job_id``, ``discovery_id`` (the discovery job) and ``pairs`` (0-based indices).

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("lineage", params, lambda: self._lineage_work(params))

    def provenance(self, params: dict[str, Any]) -> FunctionResult:
        """Start a Provenance job.

        Args:
            params: ``job_id``, ``suspect`` choice, ``candidates``, ``search``, ``search_text``,
                ``include_derived``, ``layers``, ``top_k``.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("provenance", params, lambda: self._provenance_work(params))

    def behaviour(self, params: dict[str, Any]) -> FunctionResult:
        """Start a batch Behaviour Probe on an uploaded prompts file.

        Args:
            params: ``job_id``, ``base``, ``tuned``, ``file_name``, ``file_content`` (bytes),
                ``field``, ``limit``, ``chat``, ``generate_tokens``.

        Returns:
            Tuple of (result, message, data) with the started job id and the prompt count.
        """
        return self._submit("behaviour", params, lambda: self._batch_work(params))

    def assessment(self, params: dict[str, Any]) -> FunctionResult:
        """Start a suspicion assessment workflow job.

        Args:
            params: ``job_id``, ``suspect`` and optional ``source`` model choices, ``search``,
                ``verify_source``, ``candidates``, the per-tool settings and the behaviour options.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("assessment", params, lambda: self._assessment_work(params))

    # ------------------------------------------------------------------ interactive session
    def behaviour_load(self, params: dict[str, Any]) -> FunctionResult:
        """Start loading a model pair for the interactive session (closing any previous one).

        Args:
            params: ``job_id``, ``base`` and ``tuned`` model choices.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("behaviour", params, lambda: self._load_work(params), label="")

    def behaviour_session(self, params: dict[str, Any]) -> FunctionResult:
        """The interactive session as it stands (loaded or not).

        Args:
            params: unused.

        Returns:
            Tuple of (result, message, data).
        """
        with self._session_lock:
            data = session_payload(self._session)
        return make_function_result(True, "", data)

    def behaviour_prompt(self, params: dict[str, Any]) -> FunctionResult:
        """Compare both loaded models on one prompt, optionally generating from both.

        The inference runs as a job on the shared runner and the request waits
        for it, so it never overlaps another job.

        Args:
            params: ``text``, ``chat``, ``top``, ``generate_tokens``.

        Returns:
            Tuple of (result, message, data) with the new history entry.
        """
        outcome: FunctionResult
        try:
            text = text_param(params, "text", "Prompt", required=True, maximum=MAX_PROMPT_CHARS, strip=False)
            settings = session_settings(params)
            session, job = self._start_prompt(text.strip(), settings)
            outcome = self._finish_prompt(session, wait_for(job))
        except InvalidInput as error:
            outcome = failure(str(error), error.reason)
        except JobBusy as error:
            outcome = failure(str(error), REASON_BUSY)
        return outcome

    def behaviour_save(self, params: dict[str, Any]) -> FunctionResult:
        """Start probing every prompt of the session on the loaded pair, saved as a report.

        Args:
            params: ``job_id``, ``chat``, ``top``, ``generate_tokens``.

        Returns:
            Tuple of (result, message, data) with the started job id.
        """
        return self._submit("behaviour", params, lambda: self._save_work(params), label="")

    def behaviour_unload(self, params: dict[str, Any]) -> FunctionResult:
        """Close the loaded pair and forget the session.

        Refused while a job runs: it may be using the pair.

        Args:
            params: unused.

        Returns:
            Tuple of (result, message, data).
        """
        outcome: FunctionResult
        with self._session_lock:
            current = self.runner.current
            if current is not None and not current.finished:
                outcome = failure(f"A {current.kind} job is running. Wait for it or cancel it.", REASON_BUSY)
            else:
                unloaded = self._session is not None
                self._close_session()
                outcome = make_function_result(True, "", {"unloaded": unloaded, **session_payload(None)})
        return outcome

    def shutdown(self) -> None:
        """Release the session's models (called when the server stops)."""
        with self._session_lock:
            self._close_session()

    # ------------------------------------------------------------------ jobs
    def poll(self, page: str, job_id: str, cursor: int) -> FunctionResult:
        """The state of a job: progress, new events since ``cursor``, result or error.

        Args:
            page: the page that started it (its job kind).
            job_id: the client's job id.
            cursor: number of events already received.

        Returns:
            Tuple of (result, message, data).
        """
        outcome: FunctionResult
        job = self.runner.get(job_id) if JOB_ID_PATTERN.match(job_id) else None
        if not JOB_ID_PATTERN.match(job_id):
            outcome = failure(f"Invalid job identifier: {job_id!r}", REASON_INVALID)
        elif job is None or job.kind != page:
            outcome = failure("Unknown job: it finished long ago or was replaced by a newer one.", REASON_NOT_FOUND)
        else:
            outcome = make_function_result(True, "", job_payload(job, max(cursor, 0)))
        return outcome

    def cancel(self, job_id: str) -> FunctionResult:
        """Ask a running job to stop at its next checkpoint.

        Args:
            job_id: the client's job id.

        Returns:
            Tuple of (result, message, data) saying whether a stop was requested.
        """
        outcome: FunctionResult
        job = self.runner.get(job_id) if JOB_ID_PATTERN.match(job_id) else None
        if not JOB_ID_PATTERN.match(job_id):
            outcome = failure(f"Invalid job identifier: {job_id!r}", REASON_INVALID)
        elif job is None:
            outcome = failure("Unknown job.", REASON_NOT_FOUND)
        else:
            requested = self.runner.cancel(job_id)
            outcome = make_function_result(True, "", {"job_id": job_id, "cancel_requested": requested})
        return outcome

    def current_job(self, params: dict[str, Any]) -> FunctionResult:
        """The running or most recently finished job, as a light payload (no result, no events).

        The browser polls this on every page to reattach a returning page to its job, to
        announce a finish wherever the user is, and to describe a process that refused a
        second start.

        Args:
            params: unused.

        Returns:
            Tuple of (result, message, data) with ``job`` (None when none has ever run).
        """
        job = self.runner.current
        return make_function_result(True, "", {"job": current_job_payload(job) if job is not None else None})

    # ------------------------------------------------------------------ reports
    def reports(self, params: dict[str, Any]) -> FunctionResult:
        """Every saved report, newest first.

        Args:
            params: unused.

        Returns:
            Tuple of (result, message, data).
        """
        outcome: FunctionResult
        try:
            records = [record_payload(record, self.hf_token, self.repo_factory) for record in self.store.list()]
            outcome = make_function_result(True, "", {"reports": records, "root": os.path.abspath(self.store.root)})
        except OSError as error:
            outcome = failure(f"The reports folder cannot be read: {error}")
        return outcome

    def report_download(self, params: dict[str, Any]) -> FunctionResult:
        """A report file to send: its path (HTML or JSON).

        Args:
            params: ``id`` and ``format`` (``html`` or ``json``).

        Returns:
            Tuple of (result, message, data) with ``path``, ``filename`` and ``mimetype``.
        """
        return self._report_action(params, self._download_data)

    def delete_report(self, params: dict[str, Any]) -> FunctionResult:
        """Delete a saved report.

        Args:
            params: ``id``.

        Returns:
            Tuple of (result, message, data).
        """
        return self._report_action(params, self._delete_data)

    # ================================================================== internals
    def _known_capabilities(self) -> Capabilities:
        """The capabilities, detected on first use only (page renders must stay fast)."""
        if self._capabilities is None:
            self._capabilities = detect_environment()
        return self._capabilities

    def _require_feature(self, page: str) -> None:
        """Refuse a disabled feature.

        Raises:
            InvalidInput: when the page's feature is unavailable on this machine.
        """
        state = self.page_features()[page]
        if not state["available"]:
            label = PAGE_FEATURES[page]
            raise InvalidInput(f"{label} is unavailable here: {state['reason']}.", REASON_DISABLED)

    def _submit(
        self, page: str, params: dict[str, Any], build: Callable[[], Work], label: str | None = None
    ) -> FunctionResult:
        """Validate, build the work and start it on the shared runner under the client's id.

        Args:
            page: the job kind (and the page that started it).
            params: the request, carrying the client's ``job_id``.
            build: builds the work callable, validating the parameters now.
            label: the process name for the notifications and reattach; the page's
                default (:data:`JOB_LABELS`) when None, an empty string for an
                internal job the front-ends do not track.
        """
        outcome: FunctionResult
        try:
            self._require_feature(page)
            job_id = text_param(params, "job_id", "Job id")
            if not job_id:
                raise InvalidInput("Missing job id.", REASON_INVALID)
            job = self.runner.start(page, build(), job_id, JOB_LABELS[page] if label is None else label)
            outcome = make_function_result(True, "", {"job_id": job.identifier, "kind": job.kind})
        except InvalidInput as error:
            outcome = failure(str(error), error.reason)
        except InvalidJobId as error:
            outcome = failure(str(error), REASON_INVALID)
        except JobBusy as error:
            outcome = failure(str(error), REASON_BUSY)
        return outcome

    def _analysis_job(self, run: Work, present: Presenter) -> Work:
        """Wrap an analysis: silenced, cancellable, report saved, payload trimmed for the page."""
        store = self.store
        hf_token = self.hf_token
        repo_factory = self.repo_factory

        def work(sink: ProgressSink, token: CancelToken) -> dict[str, Any]:
            """Run the analysis, then save its report."""
            with quiet():
                result = run(sink, token)
            token.check()
            report(sink, Stage.REPORT, "Saving the report")
            extra = {"model_format": analysed_format(result.meta.inputs, hf_token, repo_factory)}
            return {"analysis": present(to_jsonable(result)), **save_report(store, result, extra)}

        return work

    # ------------------------------------------------------------------ work builders
    def _weights_work(self, params: dict[str, Any]) -> Work:
        """The Weight Diff job for these parameters (validated now)."""
        base, tuned = model_param(params, "base", "Base model"), model_param(params, "tuned", "Tuned model")
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Open both models and compare their weights."""
            from analysis.weights import compare_weights

            report(sink, Stage.WEIGHTS, "Opening both models")
            base_source = base.open(base.detect(hf_token), hf_token)
            tuned_source = tuned.open(tuned.detect(hf_token), hf_token)
            inputs = {**base.inputs("base"), **tuned.inputs("tuned")}
            return compare_weights(base_source, tuned_source, inputs=inputs, sink=sink, token=token)

        return self._analysis_job(run, present_weights)

    def _shifts_work(self, params: dict[str, Any]) -> Work:
        """The Token Shifts job for these parameters (validated now)."""
        base, tuned = model_param(params, "base", "Base model"), model_param(params, "tuned", "Tuned model")
        updates = int_param(params, "n", "Updates", DEFAULT_UPDATES, 1, MAX_UPDATES)
        tokens = int_param(params, "top", "Tokens", DEFAULT_TOKENS, 1, MAX_TOKENS)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Load the base tokenizer, open both models and project their updates."""
            from analysis.shifts import token_shifts
            from analysis.tokens import load_tokenizer
            from models import GGUF_AUTO

            report(sink, Stage.SHIFTS, "Loading the base tokenizer")
            base_model = base.detect(hf_token)
            decoder = load_tokenizer(base_model, token=hf_token, gguf_file=base.gguf_file or GGUF_AUTO)
            return token_shifts(
                base.open(base_model, hf_token),
                tuned.open(tuned.detect(hf_token), hf_token),
                decoder,
                inputs={**base.inputs("base"), **tuned.inputs("tuned")},
                n=updates,
                top=tokens,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_shifts)

    def _vocab_work(self, params: dict[str, Any]) -> Work:
        """The Vocabulary & Embeddings job for these parameters (validated now)."""
        base, tuned = model_param(params, "base", "Base model"), model_param(params, "tuned", "Tuned model")
        top_rows = int_param(params, "top_rows", "Moved rows", DEFAULT_VOCAB_ROWS, 1, MAX_VOCAB_ROWS)
        new_tokens = int_param(params, "new_tokens", "New tokens", DEFAULT_VOCAB_NEW, 1, MAX_VOCAB_NEW)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Load both tokenizers, open both models and compare vocabulary and embeddings."""
            from analysis.tokens import load_tokenizer, vocab_info
            from analysis.vocab import compare_vocab
            from models import GGUF_AUTO

            def info(model_choice: Any) -> Any:
                """The model's tokenizer and its vocabulary (None when it exposes none)."""
                detected = model_choice.detect(hf_token)
                tokenizer = load_tokenizer(detected, token=hf_token, gguf_file=model_choice.gguf_file or GGUF_AUTO)
                try:
                    vocabulary = vocab_info(tokenizer)
                except Exception:  # noqa: BLE001 - no vocabulary just means no diff; embeddings still compare
                    vocabulary = None
                return detected, tokenizer, vocabulary

            report(sink, Stage.VOCAB, "Loading both tokenizers")
            base_model, _, base_vocab = info(base)
            tuned_model, tuned_tok, tuned_vocab = info(tuned)
            return compare_vocab(
                base.open(base_model, hf_token),
                tuned.open(tuned_model, hf_token),
                tuned_tok,
                inputs={**base.inputs("base"), **tuned.inputs("tuned")},
                base_vocab=base_vocab,
                tuned_vocab=tuned_vocab,
                top_rows=top_rows,
                max_new=new_tokens,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_vocab)

    def _entropy_work(self, params: dict[str, Any]) -> Work:
        """The Entropy Analysis job for these parameters (validated now)."""
        base, tuned = model_param(params, "base", "Source model"), model_param(params, "tuned", "Tuned model")
        layers = int_param(params, "layers", "Last blocks", DEFAULT_ENTROPY_LAYERS, 1, MAX_ENTROPY_LAYERS)
        bins = int_param(params, "bins", "Histogram bins", DEFAULT_ENTROPY_BINS, MIN_ENTROPY_BINS, MAX_ENTROPY_BINS)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Open both models and compare the entropy of their last layers."""
            from analysis.entropy import analyse_entropy

            report(sink, Stage.ENTROPY, "Opening both models")
            base_source = base.open(base.detect(hf_token), hf_token)
            tuned_source = tuned.open(tuned.detect(hf_token), hf_token)
            inputs = {**base.inputs("base"), **tuned.inputs("tuned")}
            return analyse_entropy(
                base_source, tuned_source, inputs=inputs, layers=layers, bins=bins, sink=sink, token=token
            )

        return self._analysis_job(run, present_entropy)

    def _discovery_work(self, params: dict[str, Any]) -> Work:
        """The lineage discovery job (no report: the analysis step saves one)."""
        model_id = text_param(params, "model", "Model id", required=True)
        mode = text_param(params, "mode", "Mode") or LINEAGE_MODES[0]
        if mode not in LINEAGE_MODES:
            raise InvalidInput(f"Unknown mode {mode!r}; expected one of {', '.join(LINEAGE_MODES)}")
        max_variants = int_param(params, "max_variants", "Maximum variants", DEFAULT_VARIANTS, 1, MAX_VARIANTS)
        max_size_gb = float_param(params, "max_size_gb", "Maximum size (GB)", DEFAULT_MAX_SIZE_GB)
        job_id, hf_token = str(params.get("job_id", "")), self.hf_token

        def work(sink: ProgressSink, token: CancelToken) -> dict[str, Any]:
            """Read the Hub metadata of the model and its relatives."""
            from analysis.lineage import discover_lineage

            report(sink, Stage.LINEAGE, f"Reading the Hub metadata of {model_id}")
            with quiet():
                discovery = discover_lineage(
                    model_id, mode=mode, max_variants=max_variants, max_size_gb=max_size_gb, token=hf_token
                )
            token.check()
            self._remember_discovery(job_id, discovery)
            return {"discovery_id": job_id, "discovery": to_jsonable(discovery)}

        return work

    def _lineage_work(self, params: dict[str, Any]) -> Work:
        """The lineage analysis job on the chosen pairs of a remembered discovery."""
        discovery = self._chosen_pairs(params)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Compare every chosen pair with the static token-shift analysis."""
            from analysis.lineage import analyse_lineage

            return analyse_lineage(
                discovery, decoder_factory=partial(parent_decoder, hf_token), token_hf=hf_token, sink=sink, token=token
            )

        return self._analysis_job(run, present_lineage)

    def _provenance_work(self, params: dict[str, Any]) -> Work:
        """The Provenance job for these parameters (validated now)."""
        suspect = model_param(params, "suspect", "Suspect model")
        candidates = list_param(params, "candidates", "Candidates", MAX_CANDIDATES)
        search = bool_param(params, "search", True)
        search_text = text_param(params, "search_text", "Search text") or None
        include_derived = bool_param(params, "include_derived", False)
        layers = int_param(params, "layers", "Layers", DEFAULT_LAYERS, 1, MAX_LAYERS)
        top_k = int_param(params, "top_k", "Top k", DEFAULT_TOP_K, 1, MAX_TOP_K)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Gather, screen and deep-compare the candidates."""
            from analysis.provenance import find_provenance

            return find_provenance(
                suspect.ref,
                suspect_gguf_file=suspect.gguf_file,
                suspect_base=suspect.base_override,
                candidates=candidates,
                search=search,
                search_text=search_text if search else None,
                include_derived=include_derived,
                top_k=top_k,
                layers=layers,
                hf_token=hf_token,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_provenance)

    def _batch_work(self, params: dict[str, Any]) -> Work:
        """The batch Behaviour Probe job: prompts read from the upload now, models loaded in the job."""
        base, tuned = model_param(params, "base", "Base model"), model_param(params, "tuned", "Tuned model")
        prompts = self._uploaded_prompts(params)
        settings = session_settings(params)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Load both models and probe them on the prompts."""
            from analysis.behaviour import probe

            report(sink, Stage.BEHAVIOUR, f"Loading both models for {len(prompts)} prompts")
            return probe(
                base.detect(hf_token),
                tuned.detect(hf_token),
                prompts,
                inputs={**base.inputs("base"), **tuned.inputs("tuned")},
                chat=settings["chat"],
                generate_tokens=settings["generate_tokens"],
                tuned_base_override=tuned.base_override,
                gguf_file=tuned.gguf_file,
                base_gguf_file=base.gguf_file,
                hf_token=hf_token,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_behaviour)

    def _assessment_work(self, params: dict[str, Any]) -> Work:
        """The suspicion-assessment workflow job for these parameters (validated now)."""
        suspect = model_param(params, "suspect", "Suspect model")
        source = optional_model_param(params, "source", "Source model")
        run_weights = bool_param(params, "run_weights", True)
        run_entropy = bool_param(params, "run_entropy", False)
        run_shifts = bool_param(params, "run_shifts", True)
        run_vocab = bool_param(params, "run_vocab", True)
        run_behaviour = bool_param(params, "run_behaviour", False)
        prompts = tuple(list_param(params, "prompts", "Behaviour prompts", MAX_PROMPTS))
        generate_tokens = int_param(params, "generate_tokens", "Generated tokens", 0, 0, MAX_GENERATE)
        verify_source = bool_param(params, "verify_source", True)
        search = bool_param(params, "search", True)
        candidates = tuple(list_param(params, "candidates", "Candidates", MAX_CANDIDATES))
        entropy_layers = int_param(
            params, "entropy_layers", "Entropy layers", DEFAULT_ENTROPY_LAYERS, 1, MAX_ENTROPY_LAYERS
        )
        entropy_bins = int_param(
            params, "entropy_bins", "Entropy bins", DEFAULT_ENTROPY_BINS, MIN_ENTROPY_BINS, MAX_ENTROPY_BINS
        )
        updates = int_param(params, "updates", "Token updates", DEFAULT_UPDATES, 1, MAX_UPDATES)
        tokens = int_param(params, "tokens", "Tokens", DEFAULT_TOKENS, 1, MAX_TOKENS)
        layers = int_param(params, "layers", "Provenance layers", DEFAULT_LAYERS, 1, MAX_LAYERS)
        top_k = int_param(params, "top_k", "Provenance candidates", DEFAULT_TOP_K, 1, MAX_TOP_K)
        hf_token = self.hf_token

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """Run the whole workflow and read its verdict."""
            from analysis.assessment import AssessmentOptions, ModelInput, assess_model

            source_input = (
                ModelInput(source.ref, source.gguf_file, source.base_override) if source is not None else None
            )
            options = AssessmentOptions(
                run_weights=run_weights,
                run_entropy=run_entropy,
                run_shifts=run_shifts,
                run_vocab=run_vocab,
                run_behaviour=run_behaviour,
                behaviour_prompts=prompts,
                behaviour_generate_tokens=generate_tokens,
                verify_source=verify_source,
                provenance_candidates=candidates,
                provenance_search=search,
                provenance_layers=layers,
                provenance_top_k=top_k,
                entropy_layers=entropy_layers,
                entropy_bins=entropy_bins,
                shift_updates=updates,
                shift_tokens=tokens,
            )
            return assess_model(
                ModelInput(suspect.ref, suspect.gguf_file, suspect.base_override),
                source_input,
                options=options,
                hf_token=hf_token,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_assessment)

    def _load_work(self, params: dict[str, Any]) -> Work:
        """The job loading the interactive session's pair (the previous one is closed first)."""
        base, tuned = model_param(params, "base", "Base model"), model_param(params, "tuned", "Tuned model")
        hf_token = self.hf_token

        def work(sink: ProgressSink, token: CancelToken) -> dict[str, Any]:
            """Close the old pair, load the new one and make it the session."""
            from analysis.behaviour import load_pair

            with self._session_lock:
                self._close_session()
            report(sink, Stage.BEHAVIOUR, "Loading both models")
            with quiet():
                pair = load_pair(
                    base.detect(hf_token),
                    tuned.detect(hf_token),
                    tuned_base_override=tuned.base_override,
                    gguf_file=tuned.gguf_file,
                    base_gguf_file=base.gguf_file,
                    hf_token=hf_token,
                )
            session = BehaviourSession(pair=pair, inputs={**base.inputs("base"), **tuned.inputs("tuned")})
            with self._session_lock:
                self._session = session
                payload = session_payload(session)
            if token.cancelled:
                self.shutdown()
                token.check()
            return {"session": payload}

        return work

    def _save_work(self, params: dict[str, Any]) -> Work:
        """The job probing the session's prompts on its open pair, saved as a report."""
        settings = session_settings(params)
        with self._session_lock:
            session = self._session
            if session is None:
                raise InvalidInput("No models are loaded: load a pair first.")
            if not session.prompts:
                raise InvalidInput("No prompt analysed yet: nothing to save.")
            prompts, pair, inputs = list(session.prompts), session.pair, dict(session.inputs)

        def run(sink: ProgressSink, token: CancelToken) -> Any:
            """The full probe, reusing the loaded pair."""
            from analysis.behaviour import probe

            return probe(
                None,  # type: ignore[arg-type]  # the models come from ``pair``
                None,  # type: ignore[arg-type]
                prompts,
                inputs=inputs,
                chat=settings["chat"],
                top=settings["top"],
                generate_tokens=settings["generate_tokens"],
                pair=pair,
                sink=sink,
                token=token,
            )

        return self._analysis_job(run, present_behaviour)

    # ------------------------------------------------------------------ session internals
    def _start_prompt(self, text: str, settings: dict[str, Any]) -> tuple[BehaviourSession, Job]:
        """Start the job comparing both models on one prompt.

        Raises:
            InvalidInput: when no pair is loaded.
            JobBusy: when another job runs.
        """
        with self._session_lock:
            session = self._session
            if session is None:
                raise InvalidInput("No models are loaded: load a pair first.")
            pair = session.pair

            def work(sink: ProgressSink, token: CancelToken) -> dict[str, Any]:
                """Compare, then generate when asked."""
                from analysis.behaviour import compare_prompt, generate

                with quiet():
                    comparison = compare_prompt(pair, text, chat=settings["chat"], top=settings["top"])
                    generation = None
                    if settings["generate_tokens"] > 0:
                        generation = generate(
                            pair, text, chat=settings["chat"], max_new_tokens=settings["generate_tokens"]
                        )
                return {"comparison": to_jsonable(comparison), "generation": to_jsonable(generation)}

            job = self.runner.start("behaviour", work)
        return session, job

    def _finish_prompt(self, session: BehaviourSession, job: Job) -> FunctionResult:
        """Record a finished prompt job in the session history, or report its failure."""
        outcome: FunctionResult
        if job.state is JobState.SUCCEEDED:
            entry = {"index": len(session.history), "text": job.result["comparison"].get("text", ""), **job.result}
            with self._session_lock:
                session.prompts.append(entry["text"])
                session.history.append(entry)
            outcome = make_function_result(True, "", {"entry": entry, "prompts": len(session.prompts)})
        else:
            outcome = failure(f"The prompt could not be compared: {job.error}")
        return outcome

    def _close_session(self) -> None:
        """Close the current pair, if any (the caller holds the session lock)."""
        if self._session is not None:
            self._session.pair.close()
        self._session = None

    # ------------------------------------------------------------------ lineage internals
    def _remember_discovery(self, job_id: str, discovery: Any) -> None:
        """Keep a discovery for the analysis step (the last few only)."""
        with self._discoveries_lock:
            self._discoveries[job_id] = discovery
            while len(self._discoveries) > DISCOVERIES_KEPT:
                self._discoveries.popitem(last=False)

    def _chosen_pairs(self, params: dict[str, Any]) -> Any:
        """The remembered discovery restricted to the chosen pairs.

        Raises:
            InvalidInput: for an unknown discovery or an invalid selection.
        """
        discovery_id = text_param(params, "discovery_id", "Discovery", required=True, maximum=64)
        with self._discoveries_lock:
            discovery = self._discoveries.get(discovery_id)
        if discovery is None:
            raise InvalidInput("Unknown discovery: run the discovery again.", REASON_NOT_FOUND)
        indices = params.get("pairs")
        count = len(discovery.pairs)
        if not isinstance(indices, list) or not indices or not all(_is_index(item, count) for item in indices):
            raise InvalidInput(f"Choose at least one pair (indices between 0 and {count - 1}).")
        return dataclasses.replace(discovery, pairs=[discovery.pairs[index] for index in sorted(set(indices))])

    # ------------------------------------------------------------------ upload internals
    def _uploaded_prompts(self, params: dict[str, Any]) -> list[str]:
        """Read the prompts of an uploaded file (written briefly under a generated name).

        Raises:
            InvalidInput: for a missing, oversized, wrongly named or unreadable file.
        """
        from analysis.behaviour import PROMPT_SUFFIXES, load_prompts

        name = os.path.basename(str(params.get("file_name") or ""))
        content = params.get("file_content")
        suffix = os.path.splitext(name)[1].lower()
        json_field = text_param(params, "field", "JSON field") or DEFAULT_FIELD
        limit = int_param(params, "limit", "Maximum prompts", DEFAULT_LIMIT, 1, MAX_LIMIT)
        if not isinstance(content, bytes) or not name:
            raise InvalidInput("No prompts file was uploaded.")
        if len(content) > MAX_FILE_SIZE:
            raise InvalidInput(f"The prompts file exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit.")
        if suffix not in PROMPT_SUFFIXES:
            raise InvalidInput(f"Unsupported prompts file {name}; use {', '.join(PROMPT_SUFFIXES)}.")
        handle, path = tempfile.mkstemp(prefix="prompts-", suffix=suffix, dir=self.upload_dir)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
            prompts = load_prompts(path, field=json_field, limit=limit)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise InvalidInput(str(error).replace(path, name)) from error
        finally:
            os.remove(path)
        return prompts

    # ------------------------------------------------------------------ report internals
    def _report_action(
        self, params: dict[str, Any], action: Callable[[str, dict[str, Any]], dict[str, Any]]
    ) -> FunctionResult:
        """Run a report action, mapping store errors to failure reasons."""
        from reporting import InvalidReportId, ReportNotFound

        outcome: FunctionResult
        try:
            outcome = make_function_result(True, "", action(str(params.get("id", "")), params))
        except InvalidReportId as error:
            outcome = failure(str(error), REASON_INVALID)
        except (ReportNotFound, InvalidInput) as error:
            outcome = failure(str(error), getattr(error, "reason", REASON_NOT_FOUND))
        except (OSError, ValueError, TypeError, KeyError) as error:
            outcome = failure(f"The report cannot be read: {error}")
        return outcome

    def _download_data(self, report_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Path of a report file, with its download name and type.

        Raises:
            InvalidInput: for an unknown format.
        """
        fmt = str(params.get("format", ""))
        if fmt not in DOWNLOADS:
            raise InvalidInput(f"Unknown report format {fmt!r}.", REASON_NOT_FOUND)
        filename, mimetype = DOWNLOADS[fmt]
        return {
            "filename": f"{report_id}-{filename}",
            "mimetype": mimetype,
            "path": self.store.path(report_id, fmt),
        }

    def _delete_data(self, report_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Delete the report and say which one."""
        self.store.delete(report_id)
        return {"id": report_id}


# ======================================================================= detection payloads
def address_payload(host: str, port: Any) -> dict[str, Any]:
    """The address the server listens on, resolving a wildcard bind to a real address.

    A wildcard (``0.0.0.0`` / ``::``) means every interface: the menu bar shows one of this
    machine's own addresses instead, preferring IPv4, and falls back to the raw host when none
    can be found.
    """
    wildcard = host in WILDCARD_HOSTS
    display = host
    if wildcard:
        candidates = sorted(local_addresses(), key=lambda item: (":" in item, item))
        display = candidates[0] if candidates else host
    shown = f"[{display}]" if ":" in display else display
    return {"host": display, "port": port, "url": f"http://{shown}:{port}", "wildcard": wildcard}


def ram_payload(memory: MemoryInfo | None) -> dict[str, int] | None:
    """The machine's RAM for the menu bar, or None when it cannot be observed."""
    return None if memory is None else {"total_bytes": memory.total_bytes, "available_bytes": memory.available_bytes}


def gpu_payload(gpu: GpuInfo | None) -> dict[str, Any] | None:
    """A GPU's name and VRAM for the menu bar, or None when there is none."""
    return (
        None
        if gpu is None
        else {"name": gpu.name, "total_vram_mb": gpu.total_memory_mb, "free_vram_mb": gpu.free_memory_mb}
    )


def folder_payload(listing: Any) -> dict[str, Any]:
    """A folder listing as the Browse navigator reads it (strings, not Path objects)."""
    return {
        "path": str(listing.path),
        "parent": str(listing.parent) if listing.parent is not None else None,
        "folders": [{"name": item.name, "path": str(item)} for item in listing.folders],
        "files": [{"name": item.name, "path": str(item)} for item in listing.files],
        "roots": [str(item) for item in listing.roots],
    }


def hub_candidate_payload(candidate: Any) -> dict[str, Any]:
    """A Hub search hit as the auto-completion dropdown reads it (camelCase keys)."""
    return {
        "modelId": candidate.model_id,
        "author": candidate.author,
        "downloads": candidate.downloads,
        "likes": candidate.likes,
        "trendingScore": candidate.trending_score,
    }


def describe_reference(ref: str, hf_token: str | None) -> dict[str, Any]:
    """What a reference is; a training-run folder lists its checkpoints instead.

    Raises:
        ModelError: when the reference is unusable and is not a training-run folder.
    """
    from models import ModelError, detect, list_checkpoints

    data: dict[str, Any]
    try:
        data = describe_model(detect(ref, token=hf_token))
    except ModelError as error:
        checkpoints = list_checkpoints(ref) if os.path.isdir(ref) else []
        if not checkpoints:
            raise
        data = {
            **empty_description(ref),
            "label": "training run",
            "message": str(error),
            "checkpoints": [{"path": path, "name": os.path.basename(path)} for path in checkpoints],
        }
    return data


def describe_model(model: Any) -> dict[str, Any]:
    """The detection result as the model picker shows it."""
    ordered = gguf_by_precision(list(model.gguf_files))
    return {
        **empty_description(model.ref),
        "kind": model.kind.value,
        "recommended": recommended_tools(model.kind.value),
        "label": model.label,
        "location": model.location.value,
        "checkpoint_step": model.checkpoint_step,
        "base_model": model.base_model,
        "quantization": model.quantization,
        "gguf_files": [
            {"path": name, "name": os.path.basename(name), "most_precise": index == 0}
            for index, name in enumerate(ordered)
        ],
    }


def empty_description(ref: str) -> dict[str, Any]:
    """Every key of a detection payload, unset."""
    return {
        "ref": ref,
        "kind": None,
        "recommended": recommended_tools(None),
        "label": "",
        "location": None,
        "checkpoint_step": None,
        "base_model": None,
        "quantization": None,
        "gguf_files": [],
        "checkpoints": [],
        "message": "",
    }


def parent_decoder(hf_token: str | None, parent: str) -> Any:
    """The tokenizer of a lineage parent (the vocabulary the updates are read in)."""
    from analysis.tokens import load_tokenizer
    from models import detect

    return load_tokenizer(detect(parent, token=hf_token), token=hf_token)


# ======================================================================= job payloads
def job_payload(job: Job, cursor: int) -> dict[str, Any]:
    """What a poll returns: state, latest progress, events since ``cursor``, result or error."""
    state = job.state
    events, next_cursor = job.since(cursor)
    latest = job.latest()
    return {
        "job_id": job.identifier,
        "kind": job.kind,
        "label": job.label,
        "trackable": bool(job.label),
        "state": state.value,
        "completed": state is not JobState.RUNNING,
        "cancelled": state is JobState.CANCELLED,
        "progress": event_payload(latest) if latest is not None else {"message": "Starting...", "fraction": None},
        "events": [event_payload(event) for event in events],
        "cursor": next_cursor,
        "result": job.result if state is JobState.SUCCEEDED else None,
        "error": job.error if state in (JobState.FAILED, JobState.CANCELLED) else "",
    }


def current_job_payload(job: Job) -> dict[str, Any]:
    """The light description a page reads to reattach, notify or explain a busy start.

    Carries no result and no events (the full :func:`job_payload` poll fetches those once a
    returning page has the id); it is safe to poll on every page.
    """
    state = job.state
    latest = job.latest()
    return {
        "job_id": job.identifier,
        "kind": job.kind,
        "label": job.label,
        "trackable": bool(job.label),
        "state": state.value,
        "completed": job.finished,
        "cancelled": state is JobState.CANCELLED,
        "progress": event_payload(latest) if latest is not None else {"message": "Starting...", "fraction": None},
    }


def event_payload(event: ProgressEvent) -> dict[str, Any]:
    """One progress event for the browser."""
    return {"stage": event.stage.value, "message": event.message, "fraction": event.fraction}


def wait_for(job: Job) -> Job:
    """Block until a job finishes (used by the synchronous session prompt)."""
    while not job.finished:
        time.sleep(WAIT_SECONDS)
    return job


def save_report(store: Any, result: Any, extra: dict[str, str] | None = None) -> dict[str, Any]:
    """Save a result in the report store; a failure is reported, not raised."""
    payload: dict[str, Any] = {"report": None, "report_error": ""}
    try:
        payload["report"] = report_links(store.save(result, extra=extra).id)
    except (OSError, ValueError) as error:
        payload["report_error"] = f"The report could not be saved: {error}"
    return payload


def analysed_ref(inputs: Any) -> str | None:
    """The reference of the analysed model among a result's inputs (tuned / suspect / model)."""
    data = inputs or {}
    return data.get("tuned") or data.get("suspect") or data.get("model")


def analysed_format(inputs: Any, hf_token: str | None, repo_factory: Any = None) -> str:
    """The storage format of a result's analysed model, for the report listing.

    Detects the analysed model (the tuned / suspect / model input) to read how its weights are
    stored. Detection reads file names and config only; a Hub id reaches the network.

    Args:
        inputs: the result's ``meta.inputs`` (names the models).
        hf_token: Hub token for private or gated repositories.
        repo_factory: builds Hub access objects; the real one is used when omitted.

    Returns:
        A short format name (``safetensors``, ``GGUF`` ...), or '' when none can be read.
    """
    from models import ModelError, detect, format_name

    ref = analysed_ref(inputs)
    result = ""
    if ref:
        extra = {"repo_factory": repo_factory} if repo_factory is not None else {}
        try:
            result = format_name(detect(str(ref), token=hf_token, **extra))
        except ModelError:
            result = ""
    return result


def report_links(report_id: str) -> dict[str, Any]:
    """The URLs of a report: view in a tab, download as HTML / JSON."""
    return {
        "id": report_id,
        "view_url": REPORT_VIEW_URL.format(id=report_id),
        "html_url": REPORT_FILE_URL.format(id=report_id, fmt="html"),
        "json_url": REPORT_FILE_URL.format(id=report_id, fmt="json"),
    }


def record_payload(record: Any, hf_token: str | None = None, repo_factory: Any = None) -> dict[str, Any]:
    """One listed report for the Reports page (``hf_token`` / ``repo_factory`` recover a missing format)."""
    return {
        "id": record.id,
        "analysis": record.analysis,
        "title": record.title,
        "created": record.created,
        "inputs": record.inputs,
        "summary": record.summary,
        "risk": record.risk,
        "format": record.model_format or analysed_format(record.inputs, hf_token, repo_factory),
        "size_bytes": record.size_bytes,
        **report_links(record.id),
    }


def session_payload(session: BehaviourSession | None) -> dict[str, Any]:
    """The interactive session for the browser (``loaded`` False when there is none)."""
    payload: dict[str, Any] = {"loaded": False, "inputs": {}, "placement": None, "history": [], "prompts": 0}
    if session is not None:
        pair = session.pair
        payload = {
            "loaded": True,
            "inputs": dict(session.inputs),
            "placement": to_jsonable(pair.placement),
            "dtype": str(pair.dtype),
            "same_vocab": bool(pair.same_vocab),
            "warnings": list(pair.warnings),
            "history": list(session.history),
            "prompts": len(session.prompts),
        }
    return payload


# ======================================================================= result presenters
def present_weights(data: dict[str, Any]) -> dict[str, Any]:
    """The weight diff with the heaviest tensors only (the report keeps them all), plus its summary."""
    tensors = data.get("tensors") or []
    return {**data, "tensors": tensors[:TENSORS_SHOWN], "tensor_total": len(tensors), "summary": weights_summary(data)}


def present_shifts(data: dict[str, Any]) -> dict[str, Any]:
    """The token shifts, whole, plus its summary."""
    return {**data, "summary": shifts_summary(data)}


def present_vocab(data: dict[str, Any]) -> dict[str, Any]:
    """The vocabulary & embedding result, whole (the report keeps every added token), plus its summary."""
    return {**data, "summary": vocab_summary(data)}


def present_entropy(data: dict[str, Any]) -> dict[str, Any]:
    """The entropy analysis with the most-shifted tensors only (the report keeps them all), plus its summary."""
    tensors = data.get("tensors") or []
    return {**data, "tensors": tensors[:TENSORS_SHOWN], "tensor_total": len(tensors), "summary": entropy_summary(data)}


def present_lineage(data: dict[str, Any]) -> dict[str, Any]:
    """Per pair: counts and the first heaviest updates (the report keeps every detail), plus the summary."""
    pairs = []
    for item in data.get("pairs") or []:
        shifts = item.get("shifts") or {}
        pairs.append(
            {
                "pair": item.get("pair"),
                "error": item.get("error"),
                "counts": shifts.get("counts"),
                "top_updates": (shifts.get("top_updates") or [])[:PAIR_UPDATES_SHOWN],
                "vocabulary": shifts.get("vocabulary"),
            }
        )
    return {
        "meta": data.get("meta"),
        "discovery": data.get("discovery"),
        "pairs": pairs,
        "summary": lineage_summary(data),
    }


def present_provenance(data: dict[str, Any]) -> dict[str, Any]:
    """Verdict, ranking, screen and skipped candidates (layer details stay in the report), plus the summary."""
    deep = [
        {key: item.get(key) for key in ("ref", "origin", "rank", "verdict", "summary", "error")}
        for item in data.get("deep") or []
    ]
    return {**data, "deep": deep, "summary": provenance_summary(data)}


def present_behaviour(data: dict[str, Any]) -> dict[str, Any]:
    """The behaviour probe, whole, plus its summary."""
    return {**data, "summary": behaviour_summary(data)}


def present_assessment(data: dict[str, Any]) -> dict[str, Any]:
    """The verdict, source, signals and summary; each tool's full result stays in the report.

    The executive summary is passed through in its structured form (source, impact,
    example), so the live page renders it exactly as the saved report does.
    """
    components = data.get("components") or {}
    light = {
        name: {"name": outcome.get("name"), "ran": outcome.get("ran"), "error": outcome.get("error")}
        for name, outcome in components.items()
        if outcome
    }
    return {
        "meta": data.get("meta"),
        "verdict": data.get("verdict"),
        "source": data.get("source"),
        "signals": data.get("signals"),
        "executive": data.get("executive"),
        "components": light,
    }


# ======================================================================= parameter parsing
def session_settings(params: dict[str, Any]) -> dict[str, Any]:
    """Chat template, listed positions and generated tokens of a probe request."""
    return {
        "chat": bool_param(params, "chat", True),
        "top": int_param(params, "top", "Positions listed", DEFAULT_TOP, 1, MAX_TOP),
        "generate_tokens": int_param(params, "generate_tokens", "Generated tokens", 0, 0, MAX_GENERATE),
    }


def model_param(params: dict[str, Any], key: str, label: str) -> ModelSpec:
    """A model choice ``{ref, gguf_file, base_override}`` of the request.

    Raises:
        InvalidInput: when the reference is missing or a field is not text.
    """
    value = params.get(key)
    if not isinstance(value, dict):
        raise InvalidInput(f"{label}: enter a local path or a Hugging Face repository id.")
    return ModelSpec(
        ref=text_param(value, "ref", label, required=True).strip('"'),
        gguf_file=text_param(value, "gguf_file", f"{label} GGUF file") or None,
        base_override=text_param(value, "base_override", f"{label} adapter base").strip('"') or None,
    )


def optional_model_param(params: dict[str, Any], key: str, label: str) -> ModelSpec | None:
    """A model choice that may be omitted: None when absent or its reference is empty.

    Raises:
        InvalidInput: when a non-empty reference is given but a field is not text.
    """
    value = params.get(key)
    result = None
    if isinstance(value, dict) and isinstance(value.get("ref"), str) and value["ref"].strip():
        result = model_param(params, key, label)
    return result


def describe_server() -> dict[str, Any]:
    """The stored web server address and its defaults, as the Settings page shows them."""
    address = server_address()
    return {
        "host": address.host,
        "port": address.port,
        "url": address.url,
        "defaultHost": DEFAULT_ADDRESS.host,
        "defaultPort": DEFAULT_ADDRESS.port,
        "isDefault": address.is_default,
        "isLoopback": address.is_loopback,
    }


def describe_directories() -> dict[str, Any]:
    """The stored report and models directories and their defaults, as the Settings page shows them.

    ``modelsDir`` is empty when none is set: the model pickers then open on the server's working
    directory, so the page shows that fallback rather than a stored folder.
    """
    reports = reports_root()
    default_reports = default_reports_root()
    models = models_root()
    return {
        "reportsDir": str(reports),
        "modelsDir": models,
        "defaultReportsDir": str(default_reports),
        "workingDir": os.path.abspath(os.getcwd()),
        "reportsIsDefault": reports == default_reports,
        "modelsIsDefault": models == "",
    }


def text_param(
    params: dict[str, Any],
    key: str,
    label: str,
    *,
    required: bool = False,
    maximum: int = MAX_TEXT,
    strip: bool = True,
) -> str:
    """A text parameter (empty when absent).

    Raises:
        InvalidInput: when it is not text, too long, or required and empty.
    """
    value = params.get(key)
    text = "" if value is None else value
    if not isinstance(text, str):
        raise InvalidInput(f"{label} must be text.", REASON_INVALID)
    text = text.strip() if strip else text
    if required and not text.strip():
        raise InvalidInput(f"{label} is required.")
    if len(text) > maximum:
        raise InvalidInput(f"{label} is longer than {maximum} characters.")
    return text


def int_param(params: dict[str, Any], key: str, label: str, default: int, minimum: int, maximum: int) -> int:
    """A whole number within bounds (``default`` when absent or empty).

    Raises:
        InvalidInput: when it is not a whole number within bounds.
    """
    value = params.get(key)
    number: int | None = default if value is None or value == "" else None
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        number = int(value.strip())
    if number is None or not minimum <= number <= maximum:
        raise InvalidInput(f"{label} must be a whole number between {minimum} and {maximum}.")
    return number


def float_param(params: dict[str, Any], key: str, label: str, default: float) -> float:
    """A positive finite number (``default`` when absent or empty).

    Raises:
        InvalidInput: when it is not a positive finite number.
    """
    value = params.get(key)
    number: float | None = default if value is None or value == "" else None
    try:
        if number is None and not isinstance(value, bool):
            number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = None
    if number is None or not 0 < number < float("inf"):
        raise InvalidInput(f"{label} must be a number greater than 0.")
    return number


def bool_param(params: dict[str, Any], key: str, default: bool) -> bool:
    """A boolean (``default`` when absent); "true"/"1"/"on" strings count as true (form fields)."""
    value = params.get(key)
    result = default
    if isinstance(value, bool):
        result = value
    elif isinstance(value, str) and value:
        result = value.strip().lower() in ("true", "1", "on", "yes")
    return result


def list_param(params: dict[str, Any], key: str, label: str, maximum: int) -> list[str]:
    """A list of references: a JSON list of strings, or text with one per line (commas also split).

    Raises:
        InvalidInput: when it is neither, or holds more than ``maximum`` items.
    """
    value = params.get(key)
    items: list[str]
    if value is None:
        items = []
    elif isinstance(value, str):
        items = [part.strip().strip('"') for part in value.replace(",", "\n").splitlines()]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = [item.strip().strip('"') for item in value]
    else:
        raise InvalidInput(f"{label} must be a list of references.", REASON_INVALID)
    kept = [item for item in items if item]
    if len(kept) > maximum or any(len(item) > MAX_TEXT for item in kept):
        raise InvalidInput(f"{label}: at most {maximum} references of {MAX_TEXT} characters.")
    return kept


def _is_index(item: Any, count: int) -> bool:
    """Whether ``item`` is a valid 0-based index below ``count`` (booleans excluded)."""
    return isinstance(item, int) and not isinstance(item, bool) and 0 <= item < count
