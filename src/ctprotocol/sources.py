"""Data sources: anything that can hand out bytes starting at an offset."""

from __future__ import annotations

import glob as _glob
import os
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Union
from urllib.parse import quote, urlparse

import requests

from .config import CTProtocolConfig
from .errors import ConfigError, SourceError

SourceLike = Union[str, "os.PathLike[str]", "Source"]

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class Source(ABC):
    """A readable, restartable byte source (one shard)."""

    #: Human-readable identifier (URL or path); also used for format detection.
    name: str

    @abstractmethod
    def open(self, offset: int, config: CTProtocolConfig) -> Iterator[bytes]:
        """Yield the source's bytes starting at ``offset``.

        Raise :class:`SourceError` on failure. Mark it ``retryable`` if
        re-opening at the current offset may succeed.
        """

    def size(self, config: CTProtocolConfig) -> int | None:  # pragma: no cover - default
        """Total size in bytes if cheaply known, else ``None``."""
        return None

    def supports_range(self, config: CTProtocolConfig) -> bool | None:  # pragma: no cover - default
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class FileSource(Source):
    """A local file."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self.name = self.path

    def open(self, offset: int, config: CTProtocolConfig) -> Iterator[bytes]:
        try:
            with open(self.path, "rb") as fh:
                if offset:
                    fh.seek(offset)
                while True:
                    chunk = fh.read(config.chunk_size)
                    if not chunk:
                        return
                    yield chunk
        except OSError as exc:
            raise SourceError(f"cannot read {self.path}: {exc}", source=self.name) from exc

    def size(self, config: CTProtocolConfig) -> int | None:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return None

    def supports_range(self, config: CTProtocolConfig) -> bool | None:
        return True


class HTTPSource(Source):
    """An HTTP(S) resource, streamed with resumable reads.

    * Uses ``Range`` requests to resume after a dropped connection, guarded by
      ``If-Range`` so a file that changed on the server is never spliced together.
    * Falls back to re-downloading and discarding the prefix when the server
      does not support ranges.
    * Never writes anything to disk.
    """

    def __init__(self, url: str, headers: Mapping[str, str] | None = None):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ConfigError(f"unsupported URL scheme in {url!r}; expected http(s)")
        self.url = url
        self.name = url
        self.headers = dict(headers or {})
        self._validator: str | None = None  # ETag or Last-Modified of the first response

    def _request_headers(self, config: CTProtocolConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept-Encoding": "identity", "User-Agent": "ctp-training"}
        headers.update(config.headers)
        headers.update(self.headers)
        if extra:
            headers.update(extra)
        return headers

    def open(self, offset: int, config: CTProtocolConfig) -> Iterator[bytes]:
        extra: dict[str, str] = {}
        if offset:
            extra["Range"] = f"bytes={offset}-"
            if self._validator:
                extra["If-Range"] = self._validator
        session = requests.Session()
        try:
            try:
                response = session.get(
                    self.url,
                    stream=True,
                    timeout=config.timeout,
                    headers=self._request_headers(config, extra),
                )
            except requests.RequestException as exc:
                raise SourceError(f"{self.url}: {exc}", retryable=True, source=self.name) from exc

            with response:
                status = response.status_code
                if status == 416 and offset:
                    return  # offset is at/after end of file: nothing left
                if status in _RETRYABLE_STATUS:
                    raise SourceError(f"{self.url}: HTTP {status}", retryable=True, source=self.name)
                if status >= 400:
                    raise SourceError(f"{self.url}: HTTP {status}", source=self.name)

                validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                if self._validator is None:
                    self._validator = validator
                elif validator and validator != self._validator:
                    raise SourceError(f"{self.url} changed on the server while streaming", source=self.name)

                skip = 0
                if offset:
                    if status == 206:
                        start = _content_range_start(response.headers.get("Content-Range"))
                        if start != offset:
                            raise SourceError(
                                f"{self.url}: server returned unexpected range start {start}",
                                source=self.name,
                            )
                    else:  # 200: server ignored Range, replay and discard the prefix
                        skip = offset

                try:
                    for chunk in response.iter_content(config.chunk_size):
                        if not chunk:
                            continue
                        if skip:
                            if len(chunk) <= skip:
                                skip -= len(chunk)
                                continue
                            chunk = chunk[skip:]
                            skip = 0
                        yield chunk
                except requests.RequestException as exc:
                    raise SourceError(f"{self.url}: {exc}", retryable=True, source=self.name) from exc
        finally:
            session.close()

    def _head(self, config: CTProtocolConfig) -> requests.Response | None:
        try:
            resp = requests.head(
                self.url,
                allow_redirects=True,
                timeout=config.timeout,
                headers=self._request_headers(config),
            )
        except requests.RequestException:
            return None
        return resp if resp.status_code < 400 else None

    def size(self, config: CTProtocolConfig) -> int | None:
        resp = self._head(config)
        if resp is None:
            return None
        length = resp.headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None

    def supports_range(self, config: CTProtocolConfig) -> bool | None:
        resp = self._head(config)
        if resp is None:
            return None
        return resp.headers.get("Accept-Ranges", "").lower() == "bytes"


def _content_range_start(value: str | None) -> int | None:
    match = re.match(r"bytes\s+(\d+)-", value or "")
    return int(match.group(1)) if match else None


# Backwards-compatible alias from v0.1.
URLSource = HTTPSource


# ---------------------------------------------------------------------------
# Resolution of user-supplied specs into Source objects
# ---------------------------------------------------------------------------

_BRACE = re.compile(r"\{(\d+)\.\.(\d+)\}")


def expand_braces(spec: str) -> list[str]:
    """Expand a numeric range like ``shard-{000..003}.jsonl`` (zero-padded).

    The width of the first number is kept, so ``{000..010}`` yields ``000``,
    ``001``, ... ``010``.
    """
    match = _BRACE.search(spec)
    if not match:
        return [spec]
    lo, hi = match.group(1), match.group(2)
    if int(hi) < int(lo):
        raise ConfigError(f"empty range in {spec!r}")
    width = len(lo)
    out: list[str] = []
    for n in range(int(lo), int(hi) + 1):
        out.extend(expand_braces(spec[: match.start()] + str(n).zfill(width) + spec[match.end() :]))
    return out


def hf_url(spec: str) -> str:
    """Translate ``hf://datasets/<org>/<name>[@rev]/<path>`` into a resolve URL."""
    rest = spec[len("hf://") :]
    kind = "datasets"
    for candidate in ("datasets", "models", "spaces"):
        if rest.startswith(candidate + "/"):
            kind, rest = candidate, rest[len(candidate) + 1 :]
            break
    parts = rest.split("/", 2)
    if len(parts) < 3 or not all(parts):
        raise ConfigError(f"expected hf://{kind}/<org>/<name>[@revision]/<file>, got {spec!r}")
    org, repo, path = parts
    revision = "main"
    if "@" in repo:
        repo, revision = repo.split("@", 1)
    prefix = "" if kind == "models" else f"{kind}/"
    return (
        f"https://huggingface.co/{prefix}{quote(org)}/{quote(repo)}"
        f"/resolve/{quote(revision, safe='')}/{quote(path)}"
    )


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def open_source(spec: SourceLike) -> Source:
    """Turn one URL / path / ``hf://`` spec into a :class:`Source`."""
    if isinstance(spec, Source):
        return spec
    text = os.fspath(spec)
    if text.startswith("hf://"):
        return HTTPSource(hf_url(text), headers=_hf_headers())
    if urlparse(text).scheme in ("http", "https"):
        return HTTPSource(text)
    return FileSource(text)


def resolve_sources(spec: SourceLike | Iterable[SourceLike]) -> list[Source]:
    """Resolve a spec (or list of specs) to concrete sources.

    Supports single URLs/paths, ``Source`` instances, local glob patterns
    (``data/*.jsonl.gz``, sorted) and numeric brace ranges
    (``https://host/train-{0000..0127}.parquet``).
    """
    if isinstance(spec, (str, os.PathLike, Source)):
        items: Sequence[SourceLike] = [spec]
    else:
        items = list(spec)
    if not items:
        raise ConfigError("no sources given")

    sources: list[Source] = []
    for item in items:
        if isinstance(item, Source):
            sources.append(item)
            continue
        text = os.fspath(item)
        for expanded in expand_braces(text):
            is_url = "://" in expanded
            if not is_url and any(ch in expanded for ch in "*?["):
                matches = sorted(_glob.glob(expanded))
                if not matches:
                    raise ConfigError(f"pattern {expanded!r} matched no files")
                sources.extend(FileSource(Path(m)) for m in matches)
            else:
                sources.append(open_source(expanded))
    return sources


def backoff_delay(attempt: int, base: float) -> float:
    """Exponential backoff with full jitter, capped at 30 s."""
    return float(min(30.0, base * (2 ** (attempt - 1))) * (0.5 + random.random() / 2))
