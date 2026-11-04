"""Folder listings for the directory browser of the web model pickers.

A browser cannot hand the server the absolute path of a folder picked with a
native dialog -- that is deliberate on the browser's part -- so the browser is
built here instead: the server lists the folders and the ``.gguf`` files on this
machine and the page walks them. A model reference is a folder or a single
``.gguf`` file, so both are offered; no other file is listed, and no file is
ever read.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path

# FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM, as reported by os.stat on Windows.
WINDOWS_HIDDEN = 0x2 | 0x4
GGUF_SUFFIX = ".gguf"


class FolderError(ValueError):
    """Raised when a folder cannot be listed."""


@dataclass(frozen=True)
class FolderListing:
    """One folder's subfolders and GGUF files, plus what is needed to move around from it."""

    path: Path
    parent: Path | None
    folders: tuple[Path, ...]
    files: tuple[Path, ...]
    roots: tuple[Path, ...]


def list_folder(raw: str) -> FolderListing:
    """List the subfolders and GGUF files of ``raw``, or of its nearest existing ancestor.

    Starting from an ancestor lets the browser open on a directory that has been
    typed but not created yet. Hidden folders are left out, and folders this
    user may not read are skipped rather than failing the whole listing.

    Raises:
        FolderError: when the resolved folder cannot be read.
    """
    path = _nearest_existing(Path(raw.strip().strip('"')).expanduser().resolve())
    try:
        entries = sorted(os.scandir(path), key=lambda entry: entry.name.lower())
    except OSError as error:
        raise FolderError(f"{path} cannot be read: {error.strerror or error}") from error
    folders = tuple(Path(entry.path) for entry in entries if _is_visible_folder(entry))
    files = tuple(Path(entry.path) for entry in entries if _is_gguf_file(entry))
    parent = path.parent if path.parent != path else None
    return FolderListing(path, parent, folders, files, filesystem_roots())


def filesystem_roots() -> tuple[Path, ...]:
    """Every drive on Windows; the single root elsewhere."""
    candidates = (Path(f"{letter}:\\") for letter in string.ascii_uppercase) if os.name == "nt" else (Path("/"),)
    return tuple(root for root in candidates if root.exists())


def _nearest_existing(path: Path) -> Path:
    """Return ``path`` if it is a directory, else its closest directory ancestor."""
    return next((candidate for candidate in (path, *path.parents) if candidate.is_dir()), Path(path.anchor))


def _is_visible_folder(entry: os.DirEntry[str]) -> bool:
    """True for a non-hidden directory.

    Windows marks its own hidden folders by attribute rather than by name, and
    those -- ``System Volume Information`` among them -- refuse to be opened.
    """
    result = False
    try:
        attributes = getattr(entry.stat(), "st_file_attributes", 0)
        result = entry.is_dir() and not entry.name.startswith((".", "$")) and not attributes & WINDOWS_HIDDEN
    except OSError:
        result = False
    return result


def _is_gguf_file(entry: os.DirEntry[str]) -> bool:
    """True for a ``.gguf`` file, the one file kind that is a model on its own."""
    result = False
    try:
        result = entry.is_file() and entry.name.lower().endswith(GGUF_SUFFIX)
    except OSError:
        result = False
    return result
