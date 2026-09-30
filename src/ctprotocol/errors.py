"""Exception hierarchy for CTProtocol.

Everything CTProtocol raises on purpose derives from :class:`CTProtocolError`, so callers can
catch one type at the boundary of their training loop.
"""

from __future__ import annotations


class CTProtocolError(Exception):
    """Base class for all CTProtocol errors."""


class ConfigError(CTProtocolError, ValueError):
    """Invalid configuration or arguments."""


class SourceError(CTProtocolError):
    """A data source could not be opened or read.

    ``retryable`` tells the prefetcher whether re-opening the source at the
    current byte offset is worth attempting (network hiccup, HTTP 503, ...) or
    whether the failure is permanent (HTTP 404, file changed on the server, ...).
    """

    def __init__(self, message: str, *, retryable: bool = False, source: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.source = source


class DecodeError(CTProtocolError):
    """Downloaded bytes could not be decompressed or parsed."""


class OptionalDependencyError(CTProtocolError, ImportError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, feature: str, package: str, extra: str):
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f"Install it with: pip install 'ctp-training[{extra}]'"
        )
