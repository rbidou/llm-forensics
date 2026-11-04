"""Settings shared by both front-ends: where reports go, the Hub token, input limits,
and the address the web server listens on.

The CLI and the web interface must behave the same way, so everything they
both decide lives here rather than in either of them. Standard library only:
neither front-end's dependencies (rich, prompt_toolkit, Flask) are needed to
import it, and ``run.py`` reads the server address before Flask is loaded.

The Settings menu of either interface writes the stored settings to
:data:`SETTINGS_FILE`: the web server address, the report directory and the
default models directory. A server-address change applies the next time the
web server starts, since a running server cannot move to another socket; the
host must be an IP address literal, because a hostname would resolve to
whatever the resolver says at start-up and "listen on 0.0.0.0" is a decision
the user should see spelled out, not one hidden behind a name. The two
directories apply at once: the report directory moves where reports are saved
and read, and the models directory is where the model pickers start browsing.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

# The application version, shared by both front-ends and stamped into every report.
__version__ = "1.0.0"

HF_TOKEN_VARIABLE = "HF_TOKEN"
REPORTS_FOLDER = "reports"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

# Web server address.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
HOST_KEY = "server_host"
PORT_KEY = "server_port"
MIN_PORT, MAX_PORT = 1, 65535
# Addresses that mean "every interface"; a server bound to one is reached by a real address instead.
WILDCARD_HOSTS: tuple[str, ...] = ("0.0.0.0", "::", "")
# A UDP "connect" sends no packet: it only asks the OS which address would reach this one.
ROUTE_PROBE_ADDRESS = ("192.0.2.1", 9)  # TEST-NET-1, never routed to a real host

# Stored directories: where reports are saved and read, and where the model pickers start browsing.
REPORTS_DIR_KEY = "reports_dir"
MODELS_DIR_KEY = "models_dir"
MAX_DIRECTORY = 4096

# Token Shifts: heaviest updates kept, tokens listed per axis end.
DEFAULT_UPDATES, MAX_UPDATES = 10, 200
DEFAULT_TOKENS, MAX_TOKENS = 15, 100
# Vocabulary & Embeddings: moved rows listed per matrix, new tokens characterised, neighbours per new token.
DEFAULT_VOCAB_ROWS, MAX_VOCAB_ROWS = 20, 200
DEFAULT_VOCAB_NEW, MAX_VOCAB_NEW = 50, 1000
DEFAULT_VOCAB_NEIGHBOURS, MAX_VOCAB_NEIGHBOURS = 8, 50
# Entropy Analysis: last transformer blocks examined, histogram bins of the value distribution.
DEFAULT_ENTROPY_LAYERS, MAX_ENTROPY_LAYERS = 2, 64
DEFAULT_ENTROPY_BINS, MAX_ENTROPY_BINS = 64, 256
# Hub Lineage: variants listed, size of a full child compared.
DEFAULT_VARIANTS, MAX_VARIANTS = 5, 50
DEFAULT_MAX_SIZE_GB = 8.0
# Provenance: layers compared in depth, screened candidates sent to the deep stage.
DEFAULT_LAYERS, MAX_LAYERS = 1, 64
DEFAULT_TOP_K, MAX_TOP_K = 3, 40
# Behaviour Probe: prompt file field and count, tokens listed, tokens generated.
DEFAULT_FIELD = "question"
DEFAULT_LIMIT, MAX_LIMIT = 50, 10_000
DEFAULT_TOP, MAX_TOP = 15, 100
MAX_GENERATE = 1024
# Model selection: Hub id suggestions offered as the user types a reference.
HUB_SEARCH_LIMIT = 20
MINIMUM_HUB_QUERY = 2

# Model Analysis: the weight analyses the workflow can run, and which are relevant for a model's
# storage format and so recommended (on by default). The kind strings mirror ``models.ModelKind``
# values; ``test_settings`` guards that they stay in step.
ANALYSIS_TOOLS = ("weights", "entropy", "shifts", "vocab")
# Formats a given analysis cannot read reliably, so it is off by default for them. Token Shifts
# need full-precision per-row updates: bitsandbytes 4/8-bit buries them in quantization noise,
# while GGUF dequantizes cleanly enough to keep them. The Vocabulary & Embeddings diff needs a
# tokenizer, which a bare LoRA adapter does not carry. Entropy Analysis is off for every format.
SHIFTS_UNRELIABLE_KINDS = frozenset({"bnb"})
VOCAB_UNAVAILABLE_KINDS = frozenset({"adapter"})


def recommended_tools(kind: str | None) -> dict[str, bool]:
    """Which weight analyses are recommended (on by default) for a model of this format.

    Entropy Analysis is off by default for every format. The others adapt to what the format
    exposes (see :data:`SHIFTS_UNRELIABLE_KINDS` and :data:`VOCAB_UNAVAILABLE_KINDS`). An unknown
    format (``None``, not yet detected) gets the full set bar entropy.

    Args:
        kind: the model's storage format (a ``models.ModelKind`` value), or None when unknown.

    Returns:
        A map of each analysis in :data:`ANALYSIS_TOOLS` to whether it is recommended.
    """
    return {
        "weights": True,
        "entropy": False,
        "shifts": kind not in SHIFTS_UNRELIABLE_KINDS,
        "vocab": kind not in VOCAB_UNAVAILABLE_KINDS,
    }


def default_reports_root() -> Path:
    """Folder holding the saved reports: ``<project>/reports``, next to ``run.py``."""
    return Path(__file__).resolve().parent / REPORTS_FOLDER


def reports_root() -> Path:
    """The folder reports are saved to and read from: the stored one, or the default.

    A damaged or empty setting falls back on :func:`default_reports_root`, so a bad
    value never stops reports from working.
    """
    stored = read_settings().get(REPORTS_DIR_KEY)
    return Path(stored) if isinstance(stored, str) and stored.strip() else default_reports_root()


def models_root() -> str:
    """The folder the model pickers start browsing from, or ``""`` for the working directory.

    An empty string keeps the historical behaviour (the browser opens on the server's
    working directory); a stored folder that no longer exists degrades gracefully, since
    the folder browser falls back on the nearest existing ancestor.
    """
    stored = read_settings().get(MODELS_DIR_KEY)
    return stored if isinstance(stored, str) and stored.strip() else ""


def hub_token() -> str | None:
    """The Hub token from ``HF_TOKEN``; None lets huggingface_hub use its own login."""
    return os.environ.get(HF_TOKEN_VARIABLE) or None


def torch_cuda_status() -> str:
    """Whether the installed torch can use CUDA, and on which device."""
    status = "torch is not installed"
    try:
        import torch

        status = f"not available to torch {torch.__version__} (CPU build or no driver)"
        if torch.cuda.is_available():
            status = (
                f"available: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})"
            )
    except ImportError:
        status = "torch is not installed"
    return status


# --- stored settings ------------------------------------------------------------------------------


class ServerAddressError(ValueError):
    """Raised when a host or port cannot be listened on."""


class DirectoryError(ValueError):
    """Raised when a chosen directory cannot be used."""


@dataclass(frozen=True)
class ServerAddress:
    """An IP address and port pair for the web server."""

    host: str
    port: int

    @property
    def url(self) -> str:
        """The address as a URL, bracketing IPv6 hosts as URLs require."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def is_default(self) -> bool:
        """Whether this is the address used when nothing is stored."""
        return self == DEFAULT_ADDRESS

    @property
    def is_loopback(self) -> bool:
        """Whether only this machine can reach the server."""
        return ipaddress.ip_address(self.host).is_loopback


DEFAULT_ADDRESS = ServerAddress(DEFAULT_HOST, DEFAULT_PORT)


def server_address() -> ServerAddress:
    """Return the stored address, or the default for anything missing or damaged.

    A damaged setting should not stop the application starting, so each half
    falls back on its own default.
    """
    stored = read_settings()
    host, port = stored.get(HOST_KEY), stored.get(PORT_KEY)
    return ServerAddress(
        host if isinstance(host, str) and _is_ip_address(host) else DEFAULT_HOST,
        port if isinstance(port, int) and not isinstance(port, bool) and MIN_PORT <= port <= MAX_PORT else DEFAULT_PORT,
    )


def set_server_address(host: str, port: str | int) -> ServerAddress:
    """Validate ``host`` and ``port``, store them and return the new address.

    Args:
        host: an IP address of this machine; empty restores the default.
        port: a TCP port number; empty restores the default.

    Returns:
        The stored address.

    Raises:
        ServerAddressError: the host or the port cannot be listened on.
    """
    address = ServerAddress(validated_host(host), validated_port(port))
    update_settings({HOST_KEY: address.host, PORT_KEY: address.port})
    return address


def local_addresses() -> frozenset[str]:
    """This machine's own IP addresses, as far as the standard library can find them.

    Those its host name resolves to, plus the one carrying the default route
    (a host name may resolve to the loopback address only). Never raises: an
    address that cannot be found is simply not returned.

    Returns:
        Addresses in canonical form, the wildcards excluded.
    """
    found: set[str] = set()
    with contextlib.suppress(OSError):
        found.update(str(info[4][0]) for info in socket.getaddrinfo(socket.gethostname(), None))
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(ROUTE_PROBE_ADDRESS)
        found.add(str(probe.getsockname()[0]))
    canonical = {str(ipaddress.ip_address(item.split("%")[0])) for item in found}
    return frozenset(canonical - set(WILDCARD_HOSTS))


def set_directories(reports_dir: str, models_dir: str) -> None:
    """Validate and store the report directory and the default models directory.

    Args:
        reports_dir: folder for saved reports; empty restores the default. Created if missing,
            so a writable new folder is accepted.
        models_dir: folder the model pickers start browsing from; empty restores the working
            directory. Must already exist.

    Raises:
        DirectoryError: when either path is not text, is too long, or cannot be used.
    """
    reports = validated_directory(reports_dir, create=True)
    models = validated_directory(models_dir, create=False)
    stored = read_settings()
    _apply_directory(stored, REPORTS_DIR_KEY, reports)
    _apply_directory(stored, MODELS_DIR_KEY, models)
    SETTINGS_FILE.write_text(f"{json.dumps(stored, indent=2)}\n", encoding="utf-8")


def validated_directory(raw: str, *, create: bool) -> str:
    """Return ``raw`` as a usable directory path, or ``""`` when it is empty.

    Args:
        raw: the path as typed; surrounding quotes and whitespace are trimmed.
        create: create the folder (and prove it is writable) rather than require it to exist.

    Returns:
        The canonical path, or ``""`` to restore the default.

    Raises:
        DirectoryError: not text, too long, a file where a folder is expected, or not creatable.
    """
    if not isinstance(raw, str):
        raise DirectoryError("A directory must be text.")
    text = raw.strip().strip('"')
    if len(text) > MAX_DIRECTORY:
        raise DirectoryError(f"The path is longer than {MAX_DIRECTORY} characters.")
    result = ""
    if text:
        path = Path(text).expanduser().resolve()
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise DirectoryError(f"{path} cannot be created: {error.strerror or error}") from error
        elif not path.is_dir():
            raise DirectoryError(f"{path} is not an existing directory.")
        result = str(path)
    return result


def _apply_directory(stored: dict[str, object], key: str, value: str) -> None:
    """Set ``key`` to ``value`` in ``stored``, or drop it when ``value`` is empty."""
    if value:
        stored[key] = value
    else:
        stored.pop(key, None)


def validated_host(raw: str) -> str:
    """Return ``raw`` as an IP address this machine can listen on, in canonical form.

    Binding to an ephemeral port proves the address belongs to a local
    interface; otherwise the mistake would only surface at the next start.

    Raises:
        ServerAddressError: not an IP address, or not one of this machine.
    """
    candidate = raw.strip() or DEFAULT_HOST
    if not _is_ip_address(candidate):
        raise ServerAddressError(f"{candidate!r} is not an IP address, e.g. 127.0.0.1 or 0.0.0.0.")
    host = str(ipaddress.ip_address(candidate))
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
    except OSError as error:
        raise ServerAddressError(f"{host} is not an address of this machine: {error.strerror or error}") from error
    return host


def validated_port(raw: str | int) -> int:
    """Return ``raw`` as a TCP port number.

    Raises:
        ServerAddressError: not a whole number between MIN_PORT and MAX_PORT.
    """
    text = str(raw).strip() or str(DEFAULT_PORT)
    port = int(text) if text.isascii() and text.isdigit() else -1
    if not MIN_PORT <= port <= MAX_PORT:
        raise ServerAddressError(f"{text!r} is not a port number between {MIN_PORT} and {MAX_PORT}.")
    return port


def read_settings() -> dict[str, object]:
    """The stored settings; a missing or damaged file reads as no settings at all."""
    result: dict[str, object] = {}
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if isinstance(payload, dict):
        result = payload
    return result


def update_settings(changes: dict[str, object]) -> None:
    """Merge ``changes`` into the stored settings."""
    merged = {**read_settings(), **changes}
    SETTINGS_FILE.write_text(f"{json.dumps(merged, indent=2)}\n", encoding="utf-8")


def _is_ip_address(text: str) -> bool:
    """Whether ``text`` is an IPv4 or IPv6 address literal."""
    try:
        ipaddress.ip_address(text)
    except ValueError:
        result = False
    else:
        result = True
    return result
