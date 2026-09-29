"""CTP - Caching Training Protocol.

Stream training data from URLs or files with a bounded, self-cleaning cache
instead of downloading whole datasets first.

    >>> from ctp import CTPDataset, CTPConfig
    >>> ds = CTPDataset("https://host/train-{000..015}.jsonl.gz", text_field="text",
    ...                 config=CTPConfig(ahead_seconds=60, max_cache_mb=2048))
    >>> for text in ds:            # doctest: +SKIP
    ...     ...

Framework adapters live in :mod:`ctp.integrations` and are imported lazily:
``ctp.CTPIterableDataset`` (PyTorch) and ``ctp.to_hf_iterable`` (Hugging Face).
"""

from __future__ import annotations

from typing import Any

from ._version import __version__
from .benchmark import BenchmarkReport, Plan, benchmark_system, recommend, run_benchmark
from .config import CTPConfig
from .dataset import CTPDataset, Partition
from .errors import (
    ConfigError,
    CTPError,
    DecodeError,
    OptionalDependencyError,
    SourceError,
)
from .protocol import Chunk, CTPStream
from .sources import FileSource, HTTPSource, Source, URLSource, resolve_sources
from .stats import StreamStats

__all__ = [
    "__version__",
    # core
    "CTPConfig",
    "CTPDataset",
    "CTPStream",
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
    "CTPError",
    "ConfigError",
    "SourceError",
    "DecodeError",
    "OptionalDependencyError",
    # lazy integrations
    "CTPIterableDataset",
    "to_hf_iterable",
]

_LAZY = {
    "CTPIterableDataset": ("ctp.integrations.pytorch", "CTPIterableDataset"),
    "to_hf_iterable": ("ctp.integrations.huggingface", "to_hf_iterable"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'ctp' has no attribute {name!r}")
