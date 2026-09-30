"""The caching training protocol: adaptive, bounded, delete-after-use prefetching.

Design in one paragraph
-----------------------
A single producer thread reads the sources in order and appends chunks to a
bounded buffer. The consumer (your training loop) takes chunks from the other
end. The producer is *paused* whenever the buffer already holds
``ahead_seconds x demand`` bytes, where ``demand`` is an exponential moving
average of how fast the consumer actually works through data (measured from the
time it spends *outside* the stream, so waiting for data never lowers the
estimate). Consumed chunks are dropped immediately: from RAM, or, for the disk
tier, unlinked from the session directory. Transient network failures are
retried by re-opening the source at the exact byte offset reached.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import SessionDir, check_disk_space, resolve_storage
from .config import CTProtocolConfig
from .errors import SourceError
from .sources import Source, SourceLike, backoff_delay, resolve_sources
from .stats import StreamStats

log = logging.getLogger("ctprotocol")

# A wait longer than this counts as a stall (the consumer was starved).
STALL_THRESHOLD_S = 0.05

__all__ = ["Chunk", "CTProtocolStream", "DemandTracker", "CTProtocolConfig", "URLSource"]

from .sources import URLSource  # noqa: E402  (re-export for v0.1 compatibility)


@dataclass(frozen=True)
class Chunk:
    """A piece of a shard. ``last=True`` (with empty ``data``) marks the shard's end."""

    shard: int
    data: bytes
    last: bool = False


class DemandTracker:
    """Exponential moving average of the consumer's processing rate (bytes/s)."""

    def __init__(self, alpha: float = 0.2):
        self._alpha = alpha
        self._rate: float | None = None

    def observe(self, nbytes: int, busy_seconds: float) -> None:
        if nbytes <= 0:
            return
        rate = nbytes / max(busy_seconds, 1e-4)
        self._rate = rate if self._rate is None else self._alpha * rate + (1 - self._alpha) * self._rate

    @property
    def rate(self) -> float | None:
        return self._rate


class _MemoryStore:
    kind = "memory"

    def put(self, data: bytes) -> Any:
        return data

    def take(self, handle: Any) -> bytes:
        return handle  # type: ignore[no-any-return]

    def discard(self, handle: Any) -> None:
        return None


class _DiskStore:
    kind = "disk"

    def __init__(self, session: SessionDir):
        self._session = session
        self._seq = 0

    def put(self, data: bytes) -> Path:
        self._seq += 1
        path = self._session.path / f"chunk-{self._seq:08d}.bin"
        path.write_bytes(data)
        return path

    def take(self, handle: Path) -> bytes:
        try:
            return handle.read_bytes()
        finally:
            handle.unlink(missing_ok=True)

    def discard(self, handle: Path) -> None:
        handle.unlink(missing_ok=True)


@dataclass
class _Entry:
    shard: int
    size: int
    handle: Any = None
    last: bool = False


class Prefetcher:
    """Producer/consumer buffer with an adaptive, bounded lookahead."""

    def __init__(
        self,
        sources: list[Source],
        config: CTProtocolConfig,
        stats: StreamStats,
        store: _MemoryStore | _DiskStore,
    ):
        self._sources = sources
        self._cfg = config
        self._stats = stats
        self._store = store
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._queue: deque[_Entry] = deque()
        self._buffered = 0
        self._done = False
        self._error: BaseException | None = None
        self._tracker = DemandTracker()
        self._thread: threading.Thread | None = None

    # -- lookahead policy ------------------------------------------------
    def target_bytes(self) -> int:
        cfg = self._cfg
        rate = self._tracker.rate
        target = cfg.initial_buffer_bytes if rate is None else int(rate * cfg.ahead_seconds)
        target = max(cfg.min_buffer_bytes, min(target, cfg.max_cache_bytes))
        self._stats.set(target_buffer_bytes=target, demand_bytes_per_s=rate)
        return target

    # -- producer --------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._produce, name="ctp-prefetch", daemon=True)
            self._thread.start()

    def _produce(self) -> None:
        try:
            for index, source in enumerate(self._sources):
                if not self._emit_shard(index, source):
                    return
        except BaseException as exc:  # delivered to the consumer, in order
            with self._cond:
                self._error = exc
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _emit_shard(self, index: int, source: Source) -> bool:
        offset = 0
        attempt = 0
        while True:
            reader = source.open(offset, self._cfg)
            try:
                for data in reader:
                    if not self._put(_Entry(index, len(data)), data):
                        return False
                    offset += len(data)
                    attempt = 0  # progress resets the retry budget
                break
            except SourceError as exc:
                if not exc.retryable or attempt >= self._cfg.max_retries:
                    raise
                attempt += 1
                self._stats.add(retries=1)
                delay = backoff_delay(attempt, self._cfg.retry_backoff)
                log.warning(
                    "%s failed at byte %d (%s); retry %d/%d in %.1fs",
                    source.name,
                    offset,
                    exc,
                    attempt,
                    self._cfg.max_retries,
                    delay,
                )
                if self._stop.wait(delay):
                    return False
            finally:
                close = getattr(reader, "close", None)
                if close:
                    close()
        return self._put(_Entry(index, 0, last=True), None)

    def _put(self, entry: _Entry, data: bytes | None) -> bool:
        with self._cond:
            while not self._stop.is_set() and self._buffered >= self.target_bytes():
                self._cond.wait(0.1)
            if self._stop.is_set():
                return False
        if data is not None:
            entry.handle = self._store.put(data)  # may hit disk: keep outside the lock
            self._stats.add(bytes_downloaded=len(data))
        with self._cond:
            if self._stop.is_set():
                if entry.handle is not None:
                    self._store.discard(entry.handle)
                return False
            self._queue.append(entry)
            self._buffered += entry.size
            self._stats.observe_buffer(self._buffered)
            self._cond.notify_all()
        return True

    # -- consumer --------------------------------------------------------
    def chunks(self) -> Iterator[Chunk]:
        self.start()
        released_at: float | None = None
        released_size = 0
        first_wait = True  # initial connection latency is not a stall
        try:
            while True:
                asked_at = time.perf_counter()
                if released_at is not None:
                    # Time spent outside this generator = the consumer's real work.
                    self._tracker.observe(released_size, asked_at - released_at)
                with self._cond:
                    while not self._queue:
                        if self._done:
                            if self._error is not None:
                                raise self._error
                            return
                        self._cond.wait(0.1)
                    entry = self._queue.popleft()
                    self._buffered -= entry.size
                    self._stats.observe_buffer(self._buffered)
                    self._cond.notify_all()
                waited = time.perf_counter() - asked_at
                if waited > STALL_THRESHOLD_S and not first_wait:
                    self._stats.add(stalls=1, stall_seconds=waited)
                first_wait = False

                if entry.last:
                    self._stats.add(shards_completed=1)
                    released_at, released_size = None, 0
                    yield Chunk(entry.shard, b"", last=True)
                    continue

                data = self._store.take(entry.handle)
                self._stats.add(bytes_consumed=len(data))
                released_size = len(data)
                released_at = time.perf_counter()
                yield Chunk(entry.shard, data)
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
            pending = list(self._queue)
            self._queue.clear()
            self._buffered = 0
        for entry in pending:
            if entry.handle is not None:
                self._store.discard(entry.handle)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)


class CTProtocolStream:
    """Stream one or more sources through the caching protocol.

    Example::

        with CTProtocolStream("https://host/data-{000..015}.jsonl.gz") as stream:
            for chunk in stream.chunks():
                ...

    Most users want :class:`ctprotocol.CTProtocolDataset`, which decodes records for you.
    A ``CTProtocolStream`` can be iterated more than once; each pass starts over and
    every pass cleans up after itself.
    """

    def __init__(self, sources: SourceLike | Iterable[SourceLike], config: CTProtocolConfig | None = None):
        self.config = config or CTProtocolConfig()
        self.sources: list[Source] = resolve_sources(sources)
        self.stats = StreamStats(shards_total=len(self.sources))
        self._session = SessionDir(self.config.resolved_cache_dir)
        self._active: Prefetcher | None = None

    @property
    def session_dir(self) -> SessionDir:
        return self._session

    def chunks(self) -> Iterator[Chunk]:
        """Yield :class:`Chunk` objects in shard order, with end-of-shard markers."""
        if self._active is not None:
            self._active.close()
        storage = resolve_storage(self.config)
        if storage == "disk":
            check_disk_space(self.config.resolved_cache_dir, self.config.max_cache_bytes)
            store: _MemoryStore | _DiskStore = _DiskStore(self._session)
        else:
            store = _MemoryStore()
        self.stats = StreamStats(storage=storage, shards_total=len(self.sources))
        prefetcher = Prefetcher(self.sources, self.config, self.stats, store)
        self._active = prefetcher
        log.debug("stream start: %d shard(s), storage=%s", len(self.sources), storage)
        try:
            yield from prefetcher.chunks()
        finally:
            prefetcher.close()
            self._session.close()

    def stream(self) -> Iterator[bytes]:
        """Yield raw bytes only (v0.1-compatible API)."""
        for chunk in self.chunks():
            if not chunk.last:
                yield chunk.data

    def close(self) -> None:
        """Stop prefetching and delete every temporary file."""
        if self._active is not None:
            self._active.close()
            self._active = None
        self._session.close()

    cleanup = close  # v0.1 name

    def __enter__(self) -> CTProtocolStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
