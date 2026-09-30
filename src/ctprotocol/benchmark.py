"""System benchmark and configuration planner.

``run_benchmark`` measures the things that decide how CTProtocol should be set up:
disk write / read / delete speed, network throughput to your data host,
CPU-side decode speed, and GPU throughput. ``recommend`` turns the report into
a concrete :class:`Plan`.

What it can *not* do is measure your model's training speed: that depends on
your model, batch size and framework. CTProtocol measures it live instead, from the
consumer side, and adapts the lookahead continuously (see ``CTProtocolStream.stats``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from .config import CTProtocolConfig, MiB, default_cache_dir
from .errors import CTProtocolError
from .sources import open_source

__all__ = [
    "BenchmarkReport",
    "DiskResult",
    "CpuResult",
    "GpuResult",
    "NetworkResult",
    "Plan",
    "run_benchmark",
    "recommend",
    "benchmark_system",
]


@dataclass
class DiskResult:
    path: str
    free_gb: float
    write_mb_s: float
    read_mb_s: float
    delete_files_per_s: float


@dataclass
class CpuResult:
    logical_cores: int | None
    sha256_mb_s: float
    gzip_decompress_mb_s: float
    json_parse_mb_s: float


@dataclass
class GpuResult:
    name: str
    vram_gb: float | None
    fp32_tflops: float | None = None
    fp16_tflops: float | None = None
    source: str = "torch"


@dataclass
class NetworkResult:
    url: str
    time_to_first_chunk_ms: float
    mb_s: float
    sampled_mb: float


@dataclass
class BenchmarkReport:
    platform: str
    python: str
    ram_total_gb: float
    ram_available_gb: float
    disk: DiskResult
    cpu: CpuResult
    gpu: GpuResult | None = None
    network: NetworkResult | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        d, c, g, n = self.disk, self.cpu, self.gpu, self.network
        lines = [
            f"Platform : {self.platform} (Python {self.python})",
            f"Memory   : {self.ram_available_gb:.1f} GiB available of {self.ram_total_gb:.1f} GiB",
            f"Disk     : {d.path}",
            f"           write {d.write_mb_s:,.0f} MiB/s | read {d.read_mb_s:,.0f} MiB/s | "
            f"delete {d.delete_files_per_s:,.0f} files/s | {d.free_gb:,.1f} GiB free",
            f"CPU      : {c.logical_cores} cores | sha256 {c.sha256_mb_s:,.0f} MiB/s | "
            f"gunzip {c.gzip_decompress_mb_s:,.0f} MiB/s | json {c.json_parse_mb_s:,.0f} MiB/s",
        ]
        if g:
            perf = ""
            if g.fp32_tflops is not None:
                perf = f" | fp32 {g.fp32_tflops:.1f} TFLOPS"
                if g.fp16_tflops is not None:
                    perf += f", fp16 {g.fp16_tflops:.1f} TFLOPS"
            vram = f" | {g.vram_gb:.1f} GiB VRAM" if g.vram_gb else ""
            lines.append(f"GPU      : {g.name}{vram}{perf}")
        else:
            lines.append("GPU      : none detected")
        if n:
            lines.append(
                f"Network  : {n.mb_s:,.1f} MiB/s ({n.mb_s * 8 * 1.048576:,.0f} Mbit/s), "
                f"first data after {n.time_to_first_chunk_ms:,.0f} ms [{n.url}]"
            )
        for note in self.notes:
            lines.append(f"Note     : {note}")
        return "\n".join(lines)


@dataclass
class Plan:
    """Recommended CTProtocol settings for the measured machine."""

    storage: str
    ahead_seconds: float
    max_cache_mb: int
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_config(self, **overrides: Any) -> CTProtocolConfig:
        values: dict[str, Any] = {
            "storage": self.storage,
            "ahead_seconds": self.ahead_seconds,
            "max_cache_mb": self.max_cache_mb,
        }
        values.update(overrides)
        return CTProtocolConfig(**values)

    def format(self) -> str:
        lines = [
            "Recommended configuration:",
            f"  storage       = {self.storage}",
            f"  ahead_seconds = {self.ahead_seconds:g}",
            f"  max_cache_mb  = {self.max_cache_mb}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------


def _timed_loop(fn: Any, budget: float = 0.4, min_iters: int = 3) -> tuple[int, float]:
    """Run ``fn`` at least ``min_iters`` times and for about ``budget`` seconds."""
    iters = 0
    start = time.perf_counter()
    while iters < min_iters or time.perf_counter() - start < budget:
        fn()
        iters += 1
    return iters, time.perf_counter() - start


def measure_disk(directory: str, size_mb: int = 64, small_files: int = 200) -> DiskResult:
    Path(directory).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bench-", dir=directory))
    try:
        block = os.urandom(MiB)  # incompressible, so transparent-compression FS can't cheat
        big = work / "big.bin"
        t0 = time.perf_counter()
        with open(big, "wb") as fh:
            for _ in range(size_mb):
                fh.write(block)
            fh.flush()
            os.fsync(fh.fileno())
        write_s = time.perf_counter() - t0

        with open(big, "rb") as fh:
            if hasattr(os, "posix_fadvise"):  # best effort: bypass the page cache
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            t0 = time.perf_counter()
            while fh.read(MiB):
                pass
            read_s = time.perf_counter() - t0

        payload = block[:65536]
        paths = []
        for i in range(small_files):
            p = work / f"f{i}.bin"
            p.write_bytes(payload)
            paths.append(p)
        t0 = time.perf_counter()
        for p in paths:
            p.unlink()
        delete_s = time.perf_counter() - t0

        free_gb = shutil.disk_usage(directory).free / 1024**3
        return DiskResult(
            path=str(directory),
            free_gb=round(free_gb, 2),
            write_mb_s=round(size_mb / max(write_s, 1e-9), 1),
            read_mb_s=round(size_mb / max(read_s, 1e-9), 1),
            delete_files_per_s=round(small_files / max(delete_s, 1e-9), 1),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def measure_cpu() -> CpuResult:
    buf = os.urandom(8 * MiB)
    iters, secs = _timed_loop(lambda: hashlib.sha256(buf).digest())
    sha = iters * 8 / secs

    line = (
        json.dumps({"id": 1, "text": "lorem ipsum dolor sit amet " * 20, "meta": {"a": 1, "b": [1, 2, 3]}})
        + "\n"
    )
    text = (line * (8 * MiB // len(line))).encode()
    compressed = gzip.compress(text, compresslevel=1)
    iters, secs = _timed_loop(lambda: gzip.decompress(compressed))
    gunzip = iters * len(text) / MiB / secs

    lines = text.splitlines()[:20000]
    size_mb = sum(len(x) for x in lines) / MiB
    iters, secs = _timed_loop(lambda: [json.loads(x) for x in lines])
    parse = iters * size_mb / secs
    return CpuResult(os.cpu_count(), round(sha, 1), round(gunzip, 1), round(parse, 1))


def measure_gpu() -> GpuResult | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result = GpuResult(
                name=torch.cuda.get_device_name(0), vram_gb=round(props.total_memory / 1024**3, 1)
            )

            def tflops(dtype: Any) -> float:
                n = 4096
                a = torch.randn((n, n), device="cuda", dtype=dtype)
                b = torch.randn((n, n), device="cuda", dtype=dtype)
                for _ in range(3):
                    torch.mm(a, b)
                torch.cuda.synchronize()
                reps = 10
                t0 = time.perf_counter()
                for _ in range(reps):
                    torch.mm(a, b)
                torch.cuda.synchronize()
                return 2 * n**3 * reps / (time.perf_counter() - t0) / 1e12

            result.fp32_tflops = round(tflops(torch.float32), 2)
            result.fp16_tflops = round(tflops(torch.float16), 2)
            return result
    except ImportError:
        pass
    except Exception:  # driver problems etc. must not break the benchmark
        pass

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = (
                subprocess.run(
                    [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[0]
            )
            name, mem = [x.strip() for x in out.split(",")]
            return GpuResult(name=name, vram_gb=round(float(mem) / 1024, 1), source="nvidia-smi")
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            return None
    return None


def measure_network(url: str, seconds: float = 5.0, max_mb: float = 64.0) -> NetworkResult:
    """Download up to ``max_mb`` MiB (or ``seconds``) from ``url`` and discard the bytes."""
    source = open_source(url)
    cfg = CTProtocolConfig()
    limit = int(max_mb * MiB)
    got = 0
    start = time.perf_counter()
    first: float | None = None
    first_bytes = 0
    reader = source.open(0, cfg)
    try:
        for chunk in reader:
            now = time.perf_counter()
            if first is None:
                first = now
                first_bytes = len(chunk)
            got += len(chunk)
            if got >= limit or now - start >= seconds:
                break
    finally:
        reader.close()  # type: ignore[attr-defined]
    end = time.perf_counter()
    if first is None:
        raise CTProtocolError(f"no data received from {url}")
    body_bytes = got - first_bytes
    body_time = end - first
    # For tiny resources there is no steady-state body: fall back to whole-transfer speed.
    rate = (
        (body_bytes / MiB / body_time)
        if body_bytes > 0 and body_time > 0.05
        else got / MiB / max(end - start, 1e-9)
    )
    return NetworkResult(
        url=url,
        time_to_first_chunk_ms=round((first - start) * 1000, 1),
        mb_s=round(rate, 2),
        sampled_mb=round(got / MiB, 2),
    )


def run_benchmark(
    cache_dir: str | None = None,
    *,
    url: str | None = None,
    disk_mb: int = 64,
    network_seconds: float = 5.0,
    network_max_mb: float = 64.0,
) -> BenchmarkReport:
    """Measure this machine. ``url`` (optional) adds a network throughput probe."""
    directory = cache_dir or default_cache_dir()
    mem = psutil.virtual_memory()
    report = BenchmarkReport(
        platform=platform.platform(),
        python=sys.version.split()[0],
        ram_total_gb=round(mem.total / 1024**3, 2),
        ram_available_gb=round(mem.available / 1024**3, 2),
        disk=measure_disk(directory, size_mb=disk_mb),
        cpu=measure_cpu(),
        gpu=measure_gpu(),
    )
    if url:
        try:
            report.network = measure_network(url, network_seconds, network_max_mb)
        except CTProtocolError as exc:
            report.notes.append(f"network probe failed: {exc}")
    else:
        report.notes.append("no --url given: network speed not measured")
    return report


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def recommend(
    report: BenchmarkReport,
    *,
    consume_mb_s: float | None = None,
    ahead_seconds: float = 60.0,
    ram_fraction: float = 0.25,
    disk_fraction: float = 0.5,
) -> Plan:
    """Turn a report into CTProtocol settings.

    Args:
        consume_mb_s: How fast your training loop consumes data (MiB/s). Read it
            from ``dataset.stats.consume_mb_s`` after a short trial run. When
            omitted, the network speed is used as a stand-in.
        ahead_seconds: Desired lookahead window.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    net = report.network.mb_s if report.network else None
    demand = consume_mb_s if consume_mb_s and consume_mb_s > 0 else net
    wanted_mb = max(64.0, demand * ahead_seconds) if demand else 1024.0
    mem_budget = report.ram_available_gb * 1024 * ram_fraction
    disk_budget = report.disk.free_gb * 1024 * disk_fraction

    if consume_mb_s and net is not None and net < consume_mb_s:
        warnings.append(
            f"network ({net:.1f} MiB/s) is slower than your training loop ({consume_mb_s:.1f} MiB/s): "
            "the GPU will wait for data no matter how far ahead CTProtocol prefetches. "
            "Use a closer mirror, more parallel shards, or accept the stalls."
        )

    if wanted_mb <= mem_budget:
        storage, cache_mb = "memory", wanted_mb
        reasons.append(
            f"a {wanted_mb:,.0f} MiB buffer fits in {ram_fraction:.0%} of available RAM "
            f"({mem_budget:,.0f} MiB): memory is fastest and writes nothing to disk"
        )
    else:
        disk_ok = demand is None or report.disk.write_mb_s >= demand * 1.5
        if disk_ok and disk_budget > 0:
            storage, cache_mb = "disk", min(wanted_mb, disk_budget)
            reasons.append(
                f"a {wanted_mb:,.0f} MiB buffer exceeds the RAM budget ({mem_budget:,.0f} MiB); "
                f"disk writes at {report.disk.write_mb_s:,.0f} MiB/s and can keep up"
            )
        else:
            storage, cache_mb = "memory", mem_budget
            warnings.append(
                "the disk is too slow (or too full) to spill the full lookahead, so the buffer is "
                "capped at the RAM budget and the effective lookahead is shorter than requested"
            )

    cache_mb = max(64.0, cache_mb)
    effective_ahead = ahead_seconds
    if demand and demand * ahead_seconds > cache_mb:
        effective_ahead = max(1.0, cache_mb / demand)
        reasons.append(f"lookahead reduced to ~{effective_ahead:.0f}s to stay within {cache_mb:,.0f} MiB")

    if report.gpu is None:
        reasons.append("no GPU detected: training is likely CPU-bound, so modest lookahead is enough")
    return Plan(storage, round(effective_ahead, 1), int(cache_mb), reasons, warnings)


def benchmark_system(cache_dir: str | None = None) -> dict[str, Any]:
    """v0.1-compatible entry point: run the benchmark and return a plain dict."""
    return run_benchmark(cache_dir).as_dict()
