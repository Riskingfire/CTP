"""Exception hierarchy for CTP.

Everything CTP raises on purpose derives from :class:`CTPError`, so callers can
catch one type at the boundary of their training loop.
"""

from __future__ import annotations


class CTPError(Exception):
    """Base class for all CTP errors."""


class ConfigError(CTPError, ValueError):
    """Invalid configuration or arguments."""


class SourceError(CTPError):
    """A data source could not be opened or read.

    ``retryable`` tells the prefetcher whether re-opening the source at the
    current byte offset is worth attempting (network hiccup, HTTP 503, ...) or
    whether the failure is permanent (HTTP 404, file changed on the server, ...).
    """

    def __init__(self, message: str, *, retryable: bool = False, source: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.source = source


class DecodeError(CTPError):
    """Downloaded bytes could not be decompressed or parsed."""


class OptionalDependencyError(CTPError, ImportError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, feature: str, package: str, extra: str):
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f"Install it with: pip install 'ctp-training[{extra}]'"
        )
