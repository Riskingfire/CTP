"""Record-level streaming datasets."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .config import CTPConfig
from .errors import ConfigError, DecodeError
from .formats import FORMATS, ShardDecoder, detect_format, make_decoder, no_workdir
from .protocol import CTPStream
from .sources import Source, SourceLike, resolve_sources
from .stats import StreamStats

log = logging.getLogger("ctp")

__all__ = ["CTPDataset", "Partition", "shuffle_buffer", "jsonl_stream", "text_stream"]


@dataclass(frozen=True)
class Partition:
    """This consumer's slice of the data: ``index`` out of ``count`` consumers.

    A consumer is a (distributed rank, DataLoader worker) pair. Use
    ``Partition.combine`` to build one from both.
    """

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.index < self.count:
            raise ConfigError(f"invalid partition {self.index}/{self.count}")

    @classmethod
    def combine(
        cls, rank: int = 0, world_size: int = 1, worker_id: int = 0, num_workers: int = 1
    ) -> Partition:
        return cls(index=rank * num_workers + worker_id, count=world_size * num_workers)


def shuffle_buffer(items: Iterable[Any], size: int, rng: random.Random) -> Iterator[Any]:
    """Approximate shuffle with O(size) memory (reservoir-style swap buffer)."""
    if size <= 1:
        yield from items
        return
    buf: list[Any] = []
    for item in items:
        if len(buf) < size:
            buf.append(item)
            continue
        i = rng.randrange(size)
        yield buf[i]
        buf[i] = item
    rng.shuffle(buf)
    yield from buf


class CTPDataset:
    """An iterable of decoded records that never keeps the dataset on disk.

    Args:
        sources: A URL, path, ``hf://`` spec, glob, brace range
            (``train-{000..127}.jsonl.gz``), :class:`Source`, or a list of those.
        format: ``"jsonl"``, ``"lines"``, ``"parquet"`` or ``"raw"``. Inferred from the
            file name when omitted. Compression (gzip/bzip2/zstd) is detected
            automatically.
        config: See :class:`CTPConfig`.
        text_field: Yield ``record[text_field]`` instead of the whole record.
        columns: Parquet only: read just these columns.
        shuffle_shards: Randomise shard order every epoch.
        shuffle_buffer: Size of the in-memory record shuffle buffer (0 = off).
        seed: Base seed. Order is reproducible for a given ``(seed, epoch)``.
        partition: Split shards between several consumers (see :class:`Partition`).
        transform: Optional function applied to every record.

    Iterating opens a fresh stream each time; call :meth:`set_epoch` between
    epochs to change the shuffle order. Peak temporary storage is bounded by
    ``config.max_cache_mb`` (plus one shard when reading Parquet).
    """

    def __init__(
        self,
        sources: SourceLike | Iterable[SourceLike],
        *,
        format: str | None = None,
        config: CTPConfig | None = None,
        text_field: str | None = None,
        columns: Sequence[str] | None = None,
        shuffle_shards: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 0,
        partition: Partition | None = None,
        transform: Callable[[Any], Any] | None = None,
    ):
        self.config = config or CTPConfig()
        self.sources: list[Source] = resolve_sources(sources)
        self.format = format
        if format is not None:
            if format not in FORMATS:  # fail at construction, not mid-training
                raise ConfigError(f"unknown format {format!r}; expected one of {', '.join(FORMATS)}")
        else:
            for source in self.sources:
                detect_format(source.name)
        self.text_field = text_field
        self.columns = columns
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.partition = partition or Partition()
        self.transform = transform
        self.epoch = 0
        self._stream: CTPStream | None = None
        self._last_stats: StreamStats = StreamStats()

    # -- public ----------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def stats(self) -> StreamStats:
        """Statistics of the current (or most recent) pass."""
        return self._stream.stats if self._stream is not None else self._last_stats

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    def __enter__(self) -> CTPDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        shards, stride = self._plan_shards()
        stream = CTPStream(shards, self.config)
        self._stream = stream
        records = self._records(stream, shards, stride)
        if self.shuffle_buffer > 1:
            rng = random.Random(f"{self.seed}:{self.epoch}:{self.partition.index}")
            records = shuffle_buffer(records, self.shuffle_buffer, rng)
        try:
            for record in records:
                stream.stats.add(records=1)
                yield self.transform(record) if self.transform else record
        finally:
            self._last_stats = stream.stats
            stream.close()

    # -- internals -------------------------------------------------------
    def _plan_shards(self) -> tuple[list[Source], Partition | None]:
        shards = list(self.sources)
        if self.shuffle_shards:
            random.Random(f"{self.seed}:{self.epoch}").shuffle(shards)  # same on every rank
        part = self.partition
        if part.count == 1:
            return shards, None
        if len(shards) >= part.count:
            return shards[part.index :: part.count], None
        log.warning(
            "only %d shard(s) for %d consumers: falling back to record striding, "
            "so every consumer downloads all data. Split the dataset into more shards "
            "to avoid this.",
            len(shards),
            part.count,
        )
        return shards, part

    def _format_of(self, source: Source) -> str:
        return self.format or detect_format(source.name)

    def _records(self, stream: CTPStream, shards: list[Source], stride: Partition | None) -> Iterator[Any]:
        decoder: ShardDecoder | None = None
        current = -1
        counter = 0

        def emit(items: Iterator[Any]) -> Iterator[Any]:
            nonlocal counter
            for item in items:
                if stride is not None:
                    keep = counter % stride.count == stride.index
                    counter += 1
                    if not keep:
                        continue
                yield self._select(item)

        try:
            for chunk in stream.chunks():
                if chunk.shard != current:
                    current = chunk.shard
                    source = shards[current]
                    decoder = make_decoder(
                        self._format_of(source),
                        name=source.name,
                        on_error=self.config.on_error,
                        max_record_bytes=self.config.max_record_bytes,
                        stats=stream.stats,
                        workdir=lambda: stream.session_dir.path,
                        columns=self.columns,
                    )
                assert decoder is not None
                if chunk.last:
                    yield from emit(decoder.finish())
                    decoder = None
                else:
                    yield from emit(decoder.feed(chunk.data))
        finally:
            if decoder is not None:
                decoder.abort()

    def _select(self, record: Any) -> Any:
        if self.text_field is None:
            return record
        try:
            return record[self.text_field]
        except (KeyError, TypeError, IndexError) as exc:
            raise DecodeError(f"record has no field {self.text_field!r}: {str(record)[:120]}") from exc


# -- v0.1 helpers, kept for compatibility -------------------------------------


def jsonl_stream(chunks: Iterable[bytes]) -> Iterator[Any]:
    """Decode an iterator of raw byte chunks as JSON Lines."""
    decoder = make_decoder(
        "jsonl",
        name="<stream>",
        on_error="raise",
        max_record_bytes=64 * 1024 * 1024,
        stats=StreamStats(),
        workdir=no_workdir,
    )
    for chunk in chunks:
        yield from decoder.feed(chunk)
    yield from decoder.finish()


def text_stream(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode an iterator of raw byte chunks as UTF-8 text (undecodable bytes replaced)."""
    for chunk in chunks:
        yield chunk.decode("utf-8", errors="replace")
