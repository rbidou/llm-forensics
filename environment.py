"""Detection of the packages and devices the analyses rely on.

Pure observation: nothing is installed or downloaded, and nothing heavier than
the standard library is imported. The GPU is queried through ``nvidia-smi``
rather than torch so this works before torch is installed.

Front-ends turn the result into enabled or disabled menu entries through
:func:`features`, which gives a reason for every unavailable one: a missing
package must disable an option, never fail an analysis half-way through.
"""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Distribution names as published on PyPI.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "safetensors",
    "numpy",
    "huggingface_hub",
    "transformers",
    "peft",
    "gguf",
    "jinja2",
    "rich",
    "prompt_toolkit",
    "flask",
)

STATIC_PACKAGES: tuple[str, ...] = ("torch", "safetensors", "numpy")
TOKEN_PACKAGES: tuple[str, ...] = (*STATIC_PACKAGES, "transformers")

# (feature label, distributions it needs)
FEATURE_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Weight Diff", STATIC_PACKAGES),
    ("Token Shifts", TOKEN_PACKAGES),
    ("Vocabulary & Embeddings", TOKEN_PACKAGES),
    ("Entropy Analysis", STATIC_PACKAGES),
    ("Hub Lineage", (*TOKEN_PACKAGES, "huggingface_hub")),
    ("Provenance", STATIC_PACKAGES),
    ("Behaviour Probe", (*TOKEN_PACKAGES, "peft")),
    ("Hugging Face Hub", ("huggingface_hub",)),
    ("GGUF models", ("gguf", "numpy")),
    ("Reports", ("jinja2",)),
    ("Interactive CLI", ("rich", "prompt_toolkit")),
    ("Web interface", ("flask", "jinja2")),
)

NVIDIA_SMI_QUERY = ("--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits")
SUBPROCESS_TIMEOUT_SECONDS = 20
BYTES_PER_KIB = 1024


@dataclass(frozen=True)
class GpuInfo:
    """A CUDA device as reported by ``nvidia-smi``."""

    name: str
    total_memory_mb: int
    free_memory_mb: int
    driver_version: str


@dataclass(frozen=True)
class MemoryInfo:
    """The machine's physical RAM, in bytes."""

    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class Feature:
    """One capability of the toolkit, and why it is unavailable if it is."""

    label: str
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class Capabilities:
    """Everything observed about the machine at one moment."""

    platform_description: str
    python_version: str
    python_executable: str
    packages: dict[str, str | None]
    gpu: GpuInfo | None

    def missing(self, names: tuple[str, ...]) -> tuple[str, ...]:
        """Return which of ``names`` are not installed, in the given order."""
        return tuple(name for name in names if not self.packages.get(name))


def detect() -> Capabilities:
    """Observe the current machine and interpreter."""
    return Capabilities(
        platform_description=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        python_executable=sys.executable,
        packages={name: package_version(name) for name in TRACKED_PACKAGES},
        gpu=detect_gpu(),
    )


def features(capabilities: Capabilities) -> tuple[Feature, ...]:
    """Pair every toolkit capability with its availability."""
    return tuple(_feature(label, required, capabilities) for label, required in FEATURE_REQUIREMENTS)


def _feature(label: str, required: tuple[str, ...], capabilities: Capabilities) -> Feature:
    """Build one feature entry, naming the missing packages if any."""
    missing = capabilities.missing(required)
    reason = f"needs {', '.join(missing)}" if missing else ""
    return Feature(label=label, available=not missing, reason=reason)


def package_version(name: str) -> str | None:
    """Return the installed version of a distribution, or ``None``."""
    result: str | None
    try:
        result = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result = None
    return result


def detect_gpu() -> GpuInfo | None:
    """Return the first NVIDIA GPU, or ``None`` when there is none or no driver."""
    executable = shutil.which("nvidia-smi")
    result = None
    if executable is not None:
        result = _query_gpu(executable)
    return result


def _query_gpu(executable: str) -> GpuInfo | None:
    """Run ``nvidia-smi`` and parse its first line."""
    result = None
    try:
        completed = subprocess.run(
            [executable, *NVIDIA_SMI_QUERY],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
        result = parse_gpu_line(completed.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        result = None
    return result


def parse_gpu_line(line: str) -> GpuInfo:
    """Parse one ``name, total_mb, free_mb, driver`` line from ``nvidia-smi``.

    Raises:
        ValueError: when the line does not have four fields or a memory figure is not an integer.
    """
    name, total, free, driver = (part.strip() for part in line.split(","))
    return GpuInfo(name=name, total_memory_mb=int(total), free_memory_mb=int(free), driver_version=driver)


def detect_memory() -> MemoryInfo | None:
    """Return the machine's physical RAM, or ``None`` when it cannot be observed.

    Windows is read through ``GlobalMemoryStatusEx`` and Linux through ``/proc/meminfo``;
    any other platform, or a failure, yields ``None`` so the caller shows it as unavailable.
    """
    system = platform.system()
    result: MemoryInfo | None = None
    try:
        if system == "Windows":
            result = _windows_memory()
        elif system == "Linux":
            result = _linux_memory()
    except (OSError, ValueError, AttributeError):
        result = None
    return result


def _windows_memory() -> MemoryInfo | None:
    """Total and available physical RAM on Windows, through the Win32 API."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        """The ``MEMORYSTATUSEX`` structure filled by ``GlobalMemoryStatusEx``."""

        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    result = None
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        result = MemoryInfo(total_bytes=int(status.ullTotalPhys), available_bytes=int(status.ullAvailPhys))
    return result


def _linux_memory() -> MemoryInfo | None:
    """Total and available physical RAM on Linux, parsed from ``/proc/meminfo`` (kibibytes)."""
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields and fields[0].isdigit():
                values[key.strip()] = int(fields[0]) * BYTES_PER_KIB
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    result = None
    if total is not None and available is not None:
        result = MemoryInfo(total_bytes=total, available_bytes=available)
    return result
