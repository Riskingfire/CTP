"""Configuration for CTProtocol streams."""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ConfigError

MiB = 1024 * 1024

Storage = Literal["auto", "memory", "disk"]
OnError = Literal["raise", "skip"]


def default_cache_dir() -> str:
    """Base directory for CTProtocol session directories.

    Resolution order: ``$CTP_CACHE_DIR``, then ``<system temp>/ctp``.
    """
    return os.environ.get("CTP_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "ctp")


@dataclass(frozen=True)
class CTProtocolConfig:
    """Tunable behaviour of a :class:`~ctprotocol.CTProtocolStream` / :class:`~ctprotocol.CTProtocolDataset`.

    Attributes:
        cache_dir: Base directory for temporary data. CTProtocol only ever creates and
            deletes its own ``ctp-<pid>-<id>`` subdirectories inside it.
        storage: Where prefetched bytes wait for the consumer. ``"memory"`` keeps
            them in RAM, ``"disk"`` spills them to ``cache_dir`` and deletes each
            piece as soon as it is consumed, ``"auto"`` picks memory when the
            buffer fits comfortably in free RAM and disk otherwise.
        ahead_seconds: Lookahead target. CTProtocol keeps roughly
            ``ahead_seconds x (measured consumption rate)`` bytes buffered.
        max_cache_mb: Hard upper bound on buffered bytes.
        min_buffer_mb: Lower bound on buffered bytes, so the pipeline never
            degenerates to one request per chunk.
        initial_buffer_mb: Buffer target used until a consumption rate has been
            measured.
        chunk_size: Read size for sources, in bytes.
        timeout: Per-request network timeout in seconds.
        max_retries: Consecutive retries (without progress) before giving up.
        retry_backoff: Base of the exponential backoff, in seconds.
        headers: Extra HTTP headers, e.g. ``{"Authorization": "Bearer ..."}``.
        on_error: ``"raise"`` on malformed records or ``"skip"`` and count them.
        max_record_mb: Upper bound for a single record (a line, for text formats).
    """

    cache_dir: str | None = None
    storage: Storage = "auto"
    ahead_seconds: float = 60.0
    max_cache_mb: int = 1024
    min_buffer_mb: float = 8.0
    initial_buffer_mb: float = 32.0
    chunk_size: int = 1 * MiB
    timeout: float = 30.0
    max_retries: int = 5
    retry_backoff: float = 0.5
    headers: Mapping[str, str] = field(default_factory=dict)
    on_error: OnError = "raise"
    max_record_mb: int = 64

    def __post_init__(self) -> None:
        if self.storage not in ("auto", "memory", "disk"):
            raise ConfigError(f"storage must be 'auto', 'memory' or 'disk', got {self.storage!r}")
        if self.on_error not in ("raise", "skip"):
            raise ConfigError(f"on_error must be 'raise' or 'skip', got {self.on_error!r}")
        if self.ahead_seconds < 0:
            raise ConfigError("ahead_seconds must be >= 0")
        if self.max_cache_mb <= 0:
            raise ConfigError("max_cache_mb must be > 0")
        if self.min_buffer_mb <= 0 or self.initial_buffer_mb <= 0:
            raise ConfigError("min_buffer_mb and initial_buffer_mb must be > 0")
        if self.chunk_size <= 0:
            raise ConfigError("chunk_size must be > 0")
        if self.timeout <= 0:
            raise ConfigError("timeout must be > 0")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.retry_backoff < 0:
            raise ConfigError("retry_backoff must be >= 0")
        if self.max_record_mb <= 0:
            raise ConfigError("max_record_mb must be > 0")

    # -- derived values -------------------------------------------------
    @property
    def resolved_cache_dir(self) -> str:
        return self.cache_dir or default_cache_dir()

    @property
    def max_cache_bytes(self) -> int:
        return int(self.max_cache_mb * MiB)

    @property
    def min_buffer_bytes(self) -> int:
        # Never below one chunk, never above the hard cap.
        return int(min(max(self.min_buffer_mb * MiB, self.chunk_size), self.max_cache_bytes))

    @property
    def initial_buffer_bytes(self) -> int:
        return int(min(max(self.initial_buffer_mb * MiB, self.min_buffer_bytes), self.max_cache_bytes))

    @property
    def max_record_bytes(self) -> int:
        return int(self.max_record_mb * MiB)

    # -- constructors ---------------------------------------------------
    def replace(self, **changes: Any) -> CTProtocolConfig:
        """Return a copy with ``changes`` applied (the dataclass is frozen)."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_env(cls, **overrides: Any) -> CTProtocolConfig:
        """Build a config from ``CTP_*`` environment variables.

        Recognised: ``CTP_CACHE_DIR``, ``CTP_STORAGE``, ``CTP_AHEAD_SECONDS``,
        ``CTP_MAX_CACHE_MB``. Explicit ``overrides`` win over the environment.
        """
        env = os.environ
        values: dict[str, Any] = {}
        try:
            if "CTP_CACHE_DIR" in env:
                values["cache_dir"] = env["CTP_CACHE_DIR"]
            if "CTP_STORAGE" in env:
                values["storage"] = env["CTP_STORAGE"]
            if "CTP_AHEAD_SECONDS" in env:
                values["ahead_seconds"] = float(env["CTP_AHEAD_SECONDS"])
            if "CTP_MAX_CACHE_MB" in env:
                values["max_cache_mb"] = int(env["CTP_MAX_CACHE_MB"])
        except ValueError as exc:
            raise ConfigError(f"invalid CTP_* environment variable: {exc}") from exc
        values.update(overrides)
        return cls(**values)
