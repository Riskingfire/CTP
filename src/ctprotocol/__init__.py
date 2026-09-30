"""CTProtocol - Caching Training Protocol.

Stream training data from URLs or files with a bounded, self-cleaning cache
instead of downloading whole datasets first.

    >>> from ctprotocol import CTProtocolDataset, CTProtocolConfig
    >>> ds = CTProtocolDataset("https://host/train-{000..015}.jsonl.gz", text_field="text",
    ...                 config=CTProtocolConfig(ahead_seconds=60, max_cache_mb=2048))
    >>> for text in ds:            # doctest: +SKIP
    ...     ...

Framework adapters live in :mod:`ctprotocol.integrations` and are imported lazily:
``ctprotocol.CTProtocolIterableDataset`` (PyTorch) and ``ctprotocol.to_hf_iterable`` (Hugging Face).
"""

from __future__ import annotations

from typing import Any

from ._version import __version__
from .benchmark import BenchmarkReport, Plan, benchmark_system, recommend, run_benchmark
from .config import CTProtocolConfig
from .dataset import CTProtocolDataset, Partition
from .errors import (
    ConfigError,
    CTProtocolError,
    DecodeError,
    OptionalDependencyError,
    SourceError,
)
from .protocol import Chunk, CTProtocolStream
from .sources import FileSource, HTTPSource, Source, URLSource, resolve_sources
from .stats import StreamStats

__all__ = [
    "__version__",
    # core
    "CTProtocolConfig",
    "CTProtocolDataset",
    "CTProtocolStream",
    "Chunk",
    "Partition",
    "StreamStats",
    # sources
    "Source",
    "HTTPSource",
    "FileSource",
    "URLSource",
    "resolve_sources",
    # benchmark
    "run_benchmark",
    "recommend",
    "BenchmarkReport",
    "Plan",
    "benchmark_system",
    # errors
    "CTProtocolError",
    "ConfigError",
    "SourceError",
    "DecodeError",
    "OptionalDependencyError",
    # lazy integrations
    "CTProtocolIterableDataset",
    "to_hf_iterable",
]

_LAZY = {
    "CTProtocolIterableDataset": ("ctprotocol.integrations.pytorch", "CTProtocolIterableDataset"),
    "to_hf_iterable": ("ctprotocol.integrations.huggingface", "to_hf_iterable"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'ctprotocol' has no attribute {name!r}")
