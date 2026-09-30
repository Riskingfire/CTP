"""Session directories: the only place CTProtocol ever writes to disk.

Each stream owns one ``ctp-<pid>-<id>`` directory below the configured cache
directory. CTProtocol deletes *only* directories that carry its marker file, so
pointing ``cache_dir`` at a folder that contains other data is safe.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
import weakref
from pathlib import Path
from typing import Any

import psutil

from .config import CTProtocolConfig, MiB
from .errors import CTProtocolError

log = logging.getLogger("ctprotocol")

MARKER = ".ctp-session"
PREFIX = "ctp-"


def _rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


class SessionDir:
    """Lazily-created temporary directory that is removed on close or at exit."""

    def __init__(self, base: str):
        self._base = base
        self._path: Path | None = None
        self._finalizer: weakref.finalize[..., Any] | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            base = Path(self._base)
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"{PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}"
            path.mkdir()
            (path / MARKER).write_text(str(os.getpid()))
            self._path = path
            # Runs on garbage collection *and* interpreter exit.
            self._finalizer = weakref.finalize(self, _rmtree, str(path))
            log.debug("created session dir %s", path)
        return self._path

    @property
    def created(self) -> bool:
        return self._path is not None

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._path = None


def sweep_stale(base: str) -> list[str]:
    """Remove session directories left behind by processes that no longer exist."""
    removed: list[str] = []
    root = Path(base)
    if not root.is_dir():
        return removed
    for entry in root.iterdir():
        marker = entry / MARKER
        if not (entry.is_dir() and entry.name.startswith(PREFIX) and marker.is_file()):
            continue
        try:
            pid = int(marker.read_text().strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and psutil.pid_exists(pid):
            continue
        _rmtree(str(entry))
        removed.append(str(entry))
    return removed


def resolve_storage(config: CTProtocolConfig, *, ram_fraction: float = 0.25) -> str:
    """Decide between ``"memory"`` and ``"disk"`` for ``config.storage == "auto"``.

    Memory is preferred (faster, no write wear) whenever the maximum buffer fits
    into ``ram_fraction`` of the currently available RAM.
    """
    if config.storage != "auto":
        return config.storage
    available = psutil.virtual_memory().available
    return "memory" if config.max_cache_bytes <= available * ram_fraction else "disk"


def check_disk_space(directory: str, needed_bytes: int) -> None:
    """Raise :class:`CTProtocolError` if ``directory``'s filesystem cannot hold the buffer."""
    probe = Path(directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed_bytes * 1.1:
        raise CTProtocolError(
            f"not enough free disk space in {directory}: need ~{needed_bytes / MiB:.0f} MiB, "
            f"have {free / MiB:.0f} MiB. Lower max_cache_mb or use storage='memory'."
        )
