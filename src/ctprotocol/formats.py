"""Incremental decompression and record decoding.

A *decoder* is fed the raw bytes of one shard chunk by chunk and yields records
as soon as they are complete. Nothing here needs the whole shard in memory,
except Parquet, which needs random access and therefore spools *one shard* to
the session directory (and deletes it right after reading).
"""

from __future__ import annotations

import bz2
import json
import logging
import zlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .errors import ConfigError, CTProtocolError, DecodeError, OptionalDependencyError
from .stats import StreamStats

log = logging.getLogger("ctprotocol")

# Upper bound for one decompression step; protects against decompression bombs.
_MAX_OUT = 4 * 1024 * 1024

FORMATS = ("jsonl", "lines", "parquet", "raw")

_COMPRESSION_SUFFIXES = (".gz", ".gzip", ".bz2", ".zst", ".zstd")
_FORMAT_SUFFIXES = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".txt": "lines",
    ".text": "lines",
    ".parquet": "parquet",
}


def no_workdir() -> Path:
    """``workdir`` for decoders that never touch the disk."""
    raise CTProtocolError("this decoder does not use a working directory")  # pragma: no cover


def detect_format(name: str) -> str:
    """Infer the record format from a file name/URL, ignoring compression suffixes."""
    base = name.split("?", 1)[0].split("#", 1)[0].lower()
    for suffix in _COMPRESSION_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for suffix, fmt in _FORMAT_SUFFIXES.items():
        if base.endswith(suffix):
            return fmt
    raise ConfigError(f"cannot infer the data format of {name!r}; pass format= one of {', '.join(FORMATS)}")


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------


class _Passthrough:
    def feed(self, data: bytes) -> Iterator[bytes]:
        if data:
            yield data

    def finish(self) -> None:
        return None


class _Gzip:
    """Multi-member gzip."""

    def __init__(self) -> None:
        self._d = zlib.decompressobj(31)
        self._fresh = True  # True when positioned between members

    def feed(self, data: bytes) -> Iterator[bytes]:
        while data:
            self._fresh = False
            out = self._d.decompress(data, _MAX_OUT)
            if out:
                yield out
            if self._d.eof:
                data = self._d.unused_data
                self._d = zlib.decompressobj(31)
                self._fresh = True
            else:
                data = self._d.unconsumed_tail

    def finish(self) -> None:
        if not self._fresh:
            raise DecodeError("truncated gzip stream")


class _Bz2:
    """Multi-stream bzip2."""

    def __init__(self) -> None:
        self._d = bz2.BZ2Decompressor()
        self._fresh = True

    def feed(self, data: bytes) -> Iterator[bytes]:
        while True:
            if data:
                self._fresh = False
            if data or not self._d.needs_input:
                out = self._d.decompress(data, _MAX_OUT)
                if out:
                    yield out
            if self._d.eof:
                data = self._d.unused_data
                self._d = bz2.BZ2Decompressor()
                self._fresh = True
                if not data:
                    return
                continue
            if self._d.needs_input:
                return
            data = b""

    def finish(self) -> None:
        if not self._fresh:
            raise DecodeError("truncated bzip2 stream")


class _Zstd:
    """Multi-frame zstd (frame boundaries handled explicitly, like gzip members)."""

    def __init__(self) -> None:
        try:
            import zstandard
        except ImportError as exc:
            raise OptionalDependencyError("zstd decompression", "zstandard", "zstd") from exc
        self._zstd = zstandard
        self._d = zstandard.ZstdDecompressor().decompressobj()
        self._fresh = True

    def feed(self, data: bytes) -> Iterator[bytes]:
        while data:
            self._fresh = False
            try:
                out = self._d.decompress(data)
            except self._zstd.ZstdError as exc:
                raise DecodeError(f"corrupt zstd data: {exc}") from exc
            if out:
                yield out
            if self._d.eof:
                data = self._d.unused_data
                self._d = self._zstd.ZstdDecompressor().decompressobj()
                self._fresh = True
            else:
                data = b""

    def finish(self) -> None:
        if not self._fresh:
            raise DecodeError("truncated zstd stream")


def _select_inflater(head: bytes) -> Any:
    if head[:2] == b"\x1f\x8b":
        return _Gzip()
    if head[:3] == b"BZh":
        return _Bz2()
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return _Zstd()
    return _Passthrough()


class Inflater:
    """Streaming decompressor that detects gzip/bzip2/zstd from magic bytes."""

    def __init__(self) -> None:
        self._impl: Any = None
        self._head = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        if self._impl is None:
            self._head += data
            if len(self._head) < 4:
                return
            self._impl = _select_inflater(self._head)
            data, self._head = self._head, b""
        try:
            yield from self._impl.feed(data)
        except (zlib.error, OSError, ValueError) as exc:
            raise DecodeError(f"corrupt compressed data: {exc}") from exc

    def finish(self) -> Iterator[bytes]:
        if self._impl is None:
            self._impl = _Passthrough()
            head, self._head = self._head, b""
            yield from self._impl.feed(head)
        self._impl.finish()


# ---------------------------------------------------------------------------
# Record decoders
# ---------------------------------------------------------------------------


class _LineSplitter:
    def __init__(self, name: str, max_line: int):
        self._name = name
        self._max = max_line
        self._pending = b""
        self.line_no = 0

    def feed(self, data: bytes) -> Iterator[bytes]:
        parts = (self._pending + data).split(b"\n")
        self._pending = parts.pop()
        if len(self._pending) > self._max:
            raise DecodeError(
                f"{self._name}: record exceeds max_record_mb without a newline "
                "(is this really a text/JSONL file?)"
            )
        for line in parts:
            self.line_no += 1
            yield line

    def flush(self) -> Iterator[bytes]:
        if self._pending.strip():
            self.line_no += 1
            yield self._pending
        self._pending = b""


class JsonlDecoder:
    def __init__(self, name: str, on_error: str, max_line: int, stats: StreamStats):
        self._name = name
        self._on_error = on_error
        self._stats = stats
        self._split = _LineSplitter(name, max_line)

    def _parse(self, line: bytes) -> Iterator[Any]:
        line = line.strip()
        if not line:
            return
        try:
            yield json.loads(line)
        except ValueError as exc:  # includes JSONDecodeError and UnicodeDecodeError
            if self._on_error == "skip":
                self._stats.add(records_skipped=1)
                log.debug("skipping bad record in %s line %d: %s", self._name, self._split.line_no, exc)
                return
            raise DecodeError(f"{self._name}: invalid JSON on line {self._split.line_no}: {exc}") from exc

    def feed(self, data: bytes) -> Iterator[Any]:
        for line in self._split.feed(data):
            yield from self._parse(line)

    def finish(self) -> Iterator[Any]:
        for line in self._split.flush():
            yield from self._parse(line)

    def abort(self) -> None:
        return None


class LinesDecoder:
    """Yields non-empty text lines as ``str``."""

    def __init__(self, name: str, max_line: int):
        self._split = _LineSplitter(name, max_line)

    @staticmethod
    def _text(line: bytes) -> str | None:
        text = line.decode("utf-8", errors="replace").rstrip("\r")
        return text if text.strip() else None

    def feed(self, data: bytes) -> Iterator[str]:
        for line in self._split.feed(data):
            text = self._text(line)
            if text is not None:
                yield text

    def finish(self) -> Iterator[str]:
        for line in self._split.flush():
            text = self._text(line)
            if text is not None:
                yield text

    def abort(self) -> None:
        return None


class RawDecoder:
    def feed(self, data: bytes) -> Iterator[bytes]:
        yield data

    def finish(self) -> Iterator[bytes]:
        return iter(())

    def abort(self) -> None:
        return None


class ParquetDecoder:
    """Spools one shard to disk, reads it in batches, then deletes it."""

    def __init__(
        self, name: str, workdir: Callable[[], Path], columns: Sequence[str] | None, batch_size: int = 1024
    ):
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError("Parquet support", "pyarrow", "parquet") from exc
        self._name = name
        self._workdir = workdir
        self._columns = list(columns) if columns else None
        self._batch = batch_size
        self._path: Path | None = None
        self._fh: Any = None

    def feed(self, data: bytes) -> Iterator[Any]:
        if self._fh is None:
            self._path = self._workdir() / f"shard-{id(self):x}.parquet"
            self._fh = open(self._path, "wb")  # noqa: SIM115 - closed in finish()/abort()
        self._fh.write(data)
        return iter(())

    def finish(self) -> Iterator[Any]:
        import pyarrow.parquet as pq

        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        assert self._path is not None
        try:
            pf = pq.ParquetFile(str(self._path))
            for batch in pf.iter_batches(batch_size=self._batch, columns=self._columns):
                yield from batch.to_pylist()
        except Exception as exc:  # pyarrow raises several unrelated types
            raise DecodeError(f"{self._name}: cannot read Parquet data: {exc}") from exc
        finally:
            self.abort()

    def abort(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._path is not None:
            self._path.unlink(missing_ok=True)
            self._path = None


class ShardDecoder:
    """Decompression + record decoding for one shard."""

    def __init__(self, inner: Any):
        self._inflater = Inflater()
        self._inner = inner

    def feed(self, data: bytes) -> Iterator[Any]:
        for piece in self._inflater.feed(data):
            yield from self._inner.feed(piece)

    def finish(self) -> Iterator[Any]:
        for piece in self._inflater.finish():
            yield from self._inner.feed(piece)
        yield from self._inner.finish()

    def abort(self) -> None:
        self._inner.abort()


def make_decoder(
    fmt: str,
    *,
    name: str,
    on_error: str,
    max_record_bytes: int,
    stats: StreamStats,
    workdir: Callable[[], Path],
    columns: Sequence[str] | None = None,
) -> ShardDecoder:
    if fmt == "jsonl":
        inner: Any = JsonlDecoder(name, on_error, max_record_bytes, stats)
    elif fmt == "lines":
        inner = LinesDecoder(name, max_record_bytes)
    elif fmt == "parquet":
        inner = ParquetDecoder(name, workdir, columns)
    elif fmt == "raw":
        inner = RawDecoder()
    else:
        raise ConfigError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return ShardDecoder(inner)
