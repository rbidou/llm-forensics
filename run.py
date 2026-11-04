#!/usr/bin/env python
"""Entry point for LLM Forensics.

    python run.py           start the interactive session
    python run.py web       serve the browser interface on the address chosen in Setup
                            (--host, --port override it for one run)
    python run.py check     report what this machine can currently do

Engine modules are imported inside the command functions, never at module
scope, so ``check`` works on an environment missing torch or the UI libraries.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from collections.abc import Iterable

# Standard library only, so importing it here keeps ``check`` torch-free.
from environment import Capabilities, Feature, detect, features
from settings import server_address

LABEL_WIDTH = 22
CLI_PACKAGES = ("rich", "prompt_toolkit")
WEB_PACKAGES = ("flask", "jinja2")
EXPOSED_WARNING = (
    "Warning: other machines on the network can reach this server. It has no authentication and reads local files."
)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a command. Returns a process exit code."""
    from progress import silence_libraries

    # Before any library is imported: each reads its verbosity at import time.
    silence_libraries()
    arguments = _build_parser().parse_args(argv)
    commands = {
        "run": handle_run,
        "web": lambda: handle_web(arguments.host, arguments.port),
        "check": handle_check,
    }
    return commands[arguments.command]()


def handle_run() -> int:
    """Start the interactive session, refusing if it cannot work yet."""
    refused = _refuse_missing(CLI_PACKAGES, "interactive session")
    result = refused
    if not refused:
        from cli import run_cli

        result = run_cli()
    return result


def handle_web(host: str | None = None, port: int | None = None) -> int:
    """Serve the browser interface until interrupted, refusing if it cannot work yet.

    Args:
        host: address to bind for this run; None uses the one stored by Setup.
        port: TCP port for this run; None uses the one stored by Setup.

    Returns:
        The process exit code.
    """
    refused = _refuse_missing(WEB_PACKAGES, "web interface")
    result = refused
    if not refused:
        from web import serve

        stored = server_address()
        host = stored.host if host is None else host
        port = stored.port if port is None else port
        print(f"LLM Forensics web interface: http://{_url_host(host)}:{port}/  (Ctrl+C to stop)")
        if not _is_loopback(host):
            print(EXPOSED_WARNING)
        result = serve(host, port)
    return result


def _is_loopback(host: str) -> bool:
    """Whether ``host`` keeps the server on this machine (a name or an unparsable value counts as not)."""
    try:
        result = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        result = False
    return result


def _url_host(host: str) -> str:
    """The host as written in a URL: IPv6 in brackets, the wildcard as the loopback name."""
    result = host
    if host in ("0.0.0.0", "::", ""):
        result = "127.0.0.1"
    elif ":" in host:
        result = f"[{host}]"
    return result


def handle_check() -> int:
    """Print a capability report. Always exits 0: this is an observation."""
    capabilities = detect()
    _print_system(capabilities)
    _print_packages(capabilities)
    _print_features(features(capabilities))
    return 0


def _refuse_missing(packages: tuple[str, ...], purpose: str) -> int:
    """Print what is missing for ``purpose``. Returns 1 if anything is, else 0."""
    missing = detect().missing(packages)
    if missing:
        print(f"The {purpose} needs: {', '.join(missing)}.\nRun: pip install -r requirements.txt")
    return 1 if missing else 0


def _print_system(capabilities: Capabilities) -> None:
    """Show platform, interpreter and GPU details."""
    from settings import __version__

    _print_heading(f"LLM Forensics {__version__} - environment check")
    _print_row("Platform", capabilities.platform_description)
    _print_row("Python", f"{capabilities.python_version} ({capabilities.python_executable})")
    gpu = capabilities.gpu
    gpu_text = "none detected (CPU only)"
    if gpu is not None:
        gpu_text = f"{gpu.name}, {gpu.total_memory_mb} MiB, driver {gpu.driver_version}"
    _print_row("GPU", gpu_text)


def _print_packages(capabilities: Capabilities) -> None:
    """List every tracked package and its installed version."""
    _print_heading("Packages")
    for name, version in capabilities.packages.items():
        _print_row(name, version or "not installed")


def _print_features(all_features: Iterable[Feature]) -> None:
    """Show which features are available, and why the others are not."""
    _print_heading("Available features")
    for feature in all_features:
        marker = "x" if feature.available else " "
        detail = "" if feature.available else f"  {feature.reason}"
        print(f"  [{marker}] {feature.label:<{LABEL_WIDTH}}{detail}".rstrip())


def _print_heading(title: str) -> None:
    """Print a section heading."""
    print(f"\n{title}\n{'-' * len(title)}")


def _print_row(label: str, value: str) -> None:
    """Print one aligned label/value pair."""
    print(f"  {label:<{LABEL_WIDTH}}{value}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="run.py", description="Weight-level forensics for causal language models.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="start the interactive session (the default)")
    web = subparsers.add_parser("web", help="serve the browser interface")
    web.add_argument("--host", default=None, help="address to bind for this run (default: the one set in Setup)")
    web.add_argument("--port", type=int, default=None, help="TCP port for this run (default: the one set in Setup)")
    subparsers.add_parser("check", help="report what this machine can currently do")
    parser.set_defaults(command="run", host=None, port=None)
    return parser


if __name__ == "__main__":
    sys.exit(main())
