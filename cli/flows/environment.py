"""Environment: what this machine can currently do (the same report as ``run.py check``).

Platform, interpreter, GPU, tracked packages and features, plus what torch
itself says about CUDA when it is installed: ``nvidia-smi`` seeing a GPU does
not mean the installed torch build can use it (a CPU wheel cannot).
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text

from cli.common import Session, acknowledge
from cli.ui import build_table
from environment import Capabilities, detect, features
from settings import __version__, torch_cuda_status

TITLE = "Environment"


def show_environment(session: Session) -> None:
    """Show the capability report and wait for the user."""
    session.ui.goto(TITLE)
    session.ui.show(*environment_panel(detect(), torch_cuda_status()))
    acknowledge(session.ui)


def environment_panel(capabilities: Capabilities, cuda: str) -> list[RenderableType]:
    """System, packages and features as panel tables."""
    gpu = capabilities.gpu
    gpu_text = "none detected (CPU only)"
    if gpu is not None:
        gpu_text = f"{gpu.name}, {gpu.total_memory_mb} MiB, driver {gpu.driver_version}"
    system = [
        ["Version", f"LLM Forensics {__version__}"],
        ["Platform", capabilities.platform_description],
        ["Python", f"{capabilities.python_version} ({capabilities.python_executable})"],
        ["GPU", gpu_text],
        ["CUDA (torch)", cuda],
    ]
    packages = [[name, version or "not installed"] for name, version in capabilities.packages.items()]
    rows: list[list[str | Text]] = [
        [item.label, Text("yes", style="green") if item.available else Text("no", style="red"), item.reason]
        for item in features(capabilities)
    ]
    return [
        build_table(("Item", "Value"), system, "System"),
        build_table(("Package", "Version"), packages, "Packages"),
        build_table(("Feature", "Available", "Reason"), rows, "Features"),
    ]
