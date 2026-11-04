"""The Flask application factory, the page routes and the request guards.

Threaded on purpose: an analysis runs on the job runner's worker thread while
the browser keeps polling it, and an interactive prompt holds its request open
until the models have answered.

Page routes render a template and nothing else; every decision is made in the
engine, reached through the JSON API in :mod:`web.api`.

Security, beyond the engine's own checks:

* bound to the loopback address by default (the Setup page can change it);
* the ``Host`` header must name this server by a loopback name or by the
  address it is bound to, or, bound to a wildcard, by one of this machine's own
  addresses (DNS rebinding: a hostile page whose domain resolves to 127.0.0.1
  is refused, because the browser sends that domain as the Host);
* a state-changing request carrying an ``Origin`` must come from this server;
* every response says ``nosniff``, cannot be framed, and pages get a CSP
  allowing this server's own scripts only; a viewed report gets a CSP that
  allows no script at all.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

from flask import Flask, Response, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response as BaseResponse

from settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WILDCARD_HOSTS,
    __version__,
    hub_token,
    local_addresses,
    reports_root,
)
from web.api import ENGINE_KEY, REASON_STATUS, api, engine
from web.engine import MAX_FILE_SIZE, PAGE_FEATURES, ROUTELESS_PAGES, ForensicsEngine
from web.responses import ApiResponse, make_api_response

# The manual, built by docs/build_docs.py: the full PDF, and the online subset the
# documentation page includes (so it needs no route of its own).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_PDF = PROJECT_ROOT / "docs" / "llm-forensics-manual.pdf"
MANUAL_FRAGMENT = Path(__file__).resolve().parent / "templates" / "manual.html"
MANUAL_NOT_BUILT = "The manual has not been built. Run: python docs/build_docs.py"

LOOPBACK_NAMES: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
STATE_CHANGING_METHODS: tuple[str, ...] = ("POST", "PUT", "PATCH", "DELETE")
# Room for the multipart boundaries and the params field around a 2 MB file: the
# file itself is checked against MAX_FILE_SIZE exactly by the engine.
FORM_OVERHEAD = 64 * 1024
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500
API_PREFIX = "/api/"
HOST_REFUSED = "Refused: this server only answers requests addressed to it by a loopback name or its own address."
ORIGIN_REFUSED = "Refused: the request comes from another origin."
INTERNAL_ERROR_MESSAGE = "Internal error. See the server console."

PAGE_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'"
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Build the application, with one engine (and one job runner) shared by every request.

    Args:
        config: overrides of ``HOST``, ``PORT``, ``REPORTS_ROOT``, ``HF_TOKEN``,
            ``UPLOAD_DIR`` and ``ENGINE`` (a prebuilt :class:`ForensicsEngine`, for tests).

    Returns:
        The Flask application.
    """
    app = Flask(__name__)
    settings = {
        "HOST": DEFAULT_HOST,
        "PORT": DEFAULT_PORT,
        "REPORTS_ROOT": str(reports_root()),
        "HF_TOKEN": hub_token(),
        "UPLOAD_DIR": None,
        "ENGINE": None,
        **(config or {}),
    }
    app.config.update(settings)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + FORM_OVERHEAD
    app.extensions[ENGINE_KEY] = settings["ENGINE"] or ForensicsEngine(
        settings["REPORTS_ROOT"], hf_token=settings["HF_TOKEN"], upload_dir=settings["UPLOAD_DIR"]
    )
    app.register_blueprint(api)
    _register_guards(app, allowed_hosts(str(settings["HOST"]), int(settings["PORT"])))
    _register_pages(app)
    return app


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Run the server until interrupted. Returns a process exit code.

    The interactive session's models are released when the process exits.

    Args:
        host: address to bind (loopback by default: the server reads local model files).
        port: TCP port.

    Returns:
        0.
    """
    app = create_app({"HOST": host, "PORT": port})
    shared: ForensicsEngine = app.extensions[ENGINE_KEY]
    atexit.register(shared.shutdown)
    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    finally:
        shared.shutdown()
    return 0


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """The ``Host`` header values this server answers to.

    Loopback names with or without the port, plus the bound address itself when
    it is a specific (non-wildcard, non-loopback) one, or this machine's own
    addresses when it is a wildcard: other machines then reach it by one of them.

    Args:
        host: the bound address.
        port: the bound port.

    Returns:
        The accepted values, lower case.
    """
    addresses = local_addresses() if host in WILDCARD_HOSTS else frozenset({host})
    names = [*LOOPBACK_NAMES, *(f"[{item}]" if ":" in item else item for item in sorted(addresses))]
    return frozenset(item for name in names for item in (name.lower(), f"{name.lower()}:{port}"))


def _register_guards(app: Flask, hosts: frozenset[str]) -> None:
    """Install the Host / Origin checks, the security headers and the API error envelope."""

    @app.before_request
    def guard_request() -> ApiResponse | Response | None:
        """Refuse a foreign Host header, and a state-changing request from another origin."""
        origin = request.headers.get("Origin")
        refusal: str | None = None
        if request.host.lower() not in hosts:
            refusal = HOST_REFUSED
        elif request.method in STATE_CHANGING_METHODS and origin is not None and origin != _own_origin():
            refusal = ORIGIN_REFUSED
        return _refuse(refusal) if refusal is not None else None

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        """Add the headers every response carries (a page's CSP unless one is set)."""
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        response.headers.setdefault("Content-Security-Policy", PAGE_CSP)
        if request.path.startswith(API_PREFIX):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException) -> ApiResponse | HTTPException:
        """API errors (404, 405, 413...) in the envelope; page errors as Flask renders them."""
        result: ApiResponse | HTTPException = error
        if request.path.startswith(API_PREFIX):
            code = error.code or HTTP_INTERNAL_ERROR
            result = make_api_response(False, f"{error.name}: {error.description}", {}, code, True)
        return result

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> ApiResponse:
        """Never a traceback in the browser: a short message, the details in the server log."""
        app.logger.exception("Unhandled error", exc_info=error)
        return make_api_response(False, INTERNAL_ERROR_MESSAGE, {}, HTTP_INTERNAL_ERROR, True)


def _own_origin() -> str:
    """This server's origin as the browser names it (scheme and Host header)."""
    return f"{request.scheme}://{request.host}"


def _refuse(message: str) -> ApiResponse | Response:
    """A 403: the envelope for the API, plain text for pages."""
    result: ApiResponse | Response
    if request.path.startswith(API_PREFIX):
        result = make_api_response(False, message, {}, HTTP_FORBIDDEN, True)
    else:
        result = Response(message, status=HTTP_FORBIDDEN, mimetype="text/plain")
    return result


def _register_pages(app: Flask) -> None:
    """The root redirect, one route per page, and the report viewer."""

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        """The version and the availability of every page, for the navigation and the cards."""
        return {"version": __version__, "features": engine().page_features()}

    @app.get("/")
    def index() -> BaseResponse:
        """Send an arriving browser to the home page."""
        return redirect(url_for("forensics_home"))

    for page in PAGE_FEATURES:
        if page in ROUTELESS_PAGES:
            continue  # a job kind hosted on another page (the Home workflow): no template of its own
        app.add_url_rule(f"/forensics/{page}", endpoint=f"forensics_{page}", view_func=_page_view(page))

    @app.get("/forensics/documentation")
    def forensics_documentation() -> str:
        """Render the online subset of the manual (GUI usage, concepts, parameter reference)."""
        return render_template("app/documentation.html", manual_available=MANUAL_FRAGMENT.is_file())

    @app.get("/forensics/manual.pdf")
    def forensics_manual_pdf() -> Response:
        """The full manual as a download, including the parts the online subset leaves out."""
        response: Response
        if MANUAL_PDF.is_file():
            response = send_file(MANUAL_PDF, mimetype="application/pdf", as_attachment=True)
        else:
            response = Response(MANUAL_NOT_BUILT, status=HTTP_NOT_FOUND, mimetype="text/plain")
        return response

    @app.get("/forensics/reports/<report_id>/view")
    def forensics_report_view(report_id: str) -> Response | ApiResponse:
        """A saved report inline, under a CSP that allows no script and no external resource."""
        (result, message, data) = engine().report_download({"id": report_id, "format": "html"})

        response: Response | ApiResponse
        if result:
            response = send_file(data["path"], mimetype=data["mimetype"])
            response.headers["Content-Security-Policy"] = REPORT_CSP
        else:
            code = REASON_STATUS.get(str(data.get("reason", "")), HTTP_NOT_FOUND)
            response = Response(message, status=code, mimetype="text/plain")

        return response


def _page_view(page: str) -> Any:
    """The view rendering one page's template."""

    def view() -> str:
        """Render the page."""
        return render_template(f"app/{page}.html")

    view.__name__ = f"forensics_{page}"
    view.__doc__ = f"Render the {page} page."
    return view
