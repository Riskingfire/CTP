"""Runtime statistics for a stream."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import MiB


@dataclass
class StreamStats:
    """Live counters for one stream. Read them any time; they are thread-safe to read.

    ``stalls`` counts how often the consumer had to wait for data. If it is
    non-zero in a steady-state training loop, the network (or ``ahead_seconds``)
    is the bottleneck rather than your model.
    """

    started_at: float = field(default_factory=time.monotonic)
    storage: str = "memory"
    bytes_downloaded: int = 0
    bytes_consumed: int = 0
    shards_total: int = 0
    shards_completed: int = 0
    records: int = 0
    records_skipped: int = 0
    retries: int = 0
    stalls: int = 0
    stall_seconds: float = 0.0
    buffered_bytes: int = 0
    peak_buffered_bytes: int = 0
    target_buffer_bytes: int = 0
    demand_bytes_per_s: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **deltas: float) -> None:
        with self._lock:
            for key, value in deltas.items():
                setattr(self, key, getattr(self, key) + value)

    def set(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def observe_buffer(self, buffered: int) -> None:
        with self._lock:
            self.buffered_bytes = buffered
            if buffered > self.peak_buffered_bytes:
                self.peak_buffered_bytes = buffered

    @property
    def elapsed(self) -> float:
        return max(time.monotonic() - self.started_at, 1e-9)

    @property
    def download_mb_s(self) -> float:
        return self.bytes_downloaded / MiB / self.elapsed

    @property
    def consume_mb_s(self) -> float:
        return self.bytes_consumed / MiB / self.elapsed

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        data["elapsed_s"] = round(self.elapsed, 3)
        data["download_mb_s"] = round(self.download_mb_s, 3)
        data["consume_mb_s"] = round(self.consume_mb_s, 3)
        data.pop("started_at", None)
        return data

    def summary(self) -> str:
        return (
            f"{self.bytes_consumed / MiB:.1f} MiB consumed, "
            f"{self.bytes_downloaded / MiB:.1f} MiB downloaded "
            f"({self.download_mb_s:.1f} MiB/s), "
            f"peak buffer {self.peak_buffered_bytes / MiB:.1f} MiB [{self.storage}], "
            f"{self.stalls} stalls ({self.stall_seconds:.2f}s), "
            f"{self.retries} retries, {self.records} records"
            + (f", {self.records_skipped} skipped" if self.records_skipped else "")
        )
