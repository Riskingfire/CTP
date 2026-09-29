import os
import threading
import time

import pytest

from ctp import CTPConfig, CTPStream, FileSource, HTTPSource, SourceError
from ctp.cache import MARKER, SessionDir, sweep_stale
from ctp.protocol import DemandTracker

MiB = 1024 * 1024


def payload(n):
    return os.urandom(n)


def all_bytes(stream):
    return b"".join(stream.stream())


def session_dirs(cache):
    return list(cache.iterdir()) if cache.exists() else []


# --- correctness -------------------------------------------------------------


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_roundtrip_multiple_shards_in_order(tmp_path, fast_config, storage):
    files = []
    for i in range(3):
        p = tmp_path / f"s{i}.bin"
        p.write_bytes(payload(300_000 + i))
        files.append(p)
    with CTPStream([FileSource(p) for p in files], fast_config(storage=storage)) as stream:
        assert all_bytes(stream) == b"".join(p.read_bytes() for p in files)
    assert stream.stats.shards_completed == 3
    assert stream.stats.bytes_downloaded == stream.stats.bytes_consumed
    assert stream.stats.storage == storage


def test_chunks_carry_shard_boundaries(tmp_path, fast_config):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 100)
    b.write_bytes(b"")
    stream = CTPStream([FileSource(a), FileSource(b)], fast_config())
    seen = [(c.shard, len(c.data), c.last) for c in stream.chunks()]
    assert seen == [(0, 100, False), (0, 0, True), (1, 0, True)]


def test_stream_can_be_iterated_twice(tmp_path, fast_config):
    p = tmp_path / "f"
    p.write_bytes(payload(200_000))
    stream = CTPStream(FileSource(p), fast_config())
    assert all_bytes(stream) == all_bytes(stream) == p.read_bytes()


def test_v01_api_still_works(tmp_path):
    from ctp import URLSource  # noqa: F401  (alias)

    p = tmp_path / "f"
    p.write_bytes(b"abc" * 1000)
    stream = CTPStream(FileSource(str(p)), CTPConfig(chunk_size=100, cache_dir=str(tmp_path / "c")))
    assert all_bytes(stream) == b"abc" * 1000
    stream.cleanup()


# --- the point of the library: bounded lookahead & cleanup -------------------


def test_slow_consumer_bounds_download(server, fast_config):
    """A consumer that stops reading must not cause the whole file to be fetched."""
    url = server.add("/big", payload(24 * MiB))
    cfg = fast_config(max_cache_mb=2, min_buffer_mb=1, initial_buffer_mb=1)
    stream = CTPStream(HTTPSource(url), cfg)
    it = stream.chunks()
    next(it)
    time.sleep(1.0)  # give the producer every chance to run ahead
    downloaded = stream.stats.bytes_downloaded
    it.close()
    # cap (2 MiB) + a chunk in flight + kernel socket buffers; nowhere near 24 MiB
    assert downloaded <= 2 * MiB + 4 * cfg.chunk_size
    assert stream.stats.peak_buffered_bytes <= 2 * MiB + cfg.chunk_size


def test_lookahead_scales_with_measured_demand(tmp_path, fast_config):
    """Target buffer = ahead_seconds x consumer rate, clamped to [min, max]."""
    p = tmp_path / "f"
    p.write_bytes(payload(8 * MiB))
    cfg = fast_config(ahead_seconds=2.0, max_cache_mb=6, min_buffer_mb=0.25, initial_buffer_mb=0.25)
    stream = CTPStream(FileSource(p), cfg)
    for n, _chunk in enumerate(stream.chunks(), start=1):
        if n > 20:
            time.sleep(0.02)  # consumer settles at ~64 KiB / 20 ms ~ 3 MiB/s
    demand = stream.stats.demand_bytes_per_s
    assert demand is not None
    assert cfg.min_buffer_bytes <= stream.stats.target_buffer_bytes <= cfg.max_cache_bytes


def test_demand_tracker_ema():
    t = DemandTracker(alpha=0.5)
    assert t.rate is None
    t.observe(1000, 1.0)
    assert t.rate == 1000
    t.observe(3000, 1.0)
    assert t.rate == 2000
    t.observe(0, 1.0)
    assert t.rate == 2000


def test_disk_tier_leaves_nothing_behind(tmp_path, fast_config):
    p = tmp_path / "f"
    p.write_bytes(payload(3 * MiB))
    cache = tmp_path / "cache"
    stream = CTPStream(FileSource(p), fast_config(storage="disk", max_cache_mb=4, min_buffer_mb=1))
    it = stream.chunks()
    next(it)
    time.sleep(0.3)
    assert any(d.is_dir() for d in cache.iterdir()), "expected a live session dir"
    it.close()
    assert session_dirs(cache) == []


def test_disk_chunks_deleted_as_consumed(tmp_path, fast_config):
    p = tmp_path / "f"
    p.write_bytes(payload(4 * MiB))
    cfg = fast_config(storage="disk", max_cache_mb=8, min_buffer_mb=8, initial_buffer_mb=8)
    stream = CTPStream(FileSource(p), cfg)
    it = stream.chunks()
    next(it)
    time.sleep(0.5)  # producer buffers the whole file on disk
    session = next(d for d in (tmp_path / "cache").iterdir() if d.is_dir())
    before = len(list(session.glob("chunk-*.bin")))
    for _ in range(10):
        next(it)
    after = len(list(session.glob("chunk-*.bin")))
    it.close()
    assert after <= before - 10


def test_close_mid_stream_stops_thread_and_cleans(tmp_path, fast_config):
    p = tmp_path / "f"
    p.write_bytes(payload(6 * MiB))
    stream = CTPStream(FileSource(p), fast_config(storage="disk", max_cache_mb=2, min_buffer_mb=1))
    it = stream.chunks()
    next(it)
    stream.close()
    time.sleep(0.2)
    assert not [t for t in threading.enumerate() if t.name == "ctp-prefetch"]
    assert session_dirs(tmp_path / "cache") == []


def test_never_touches_foreign_files_in_cache_dir(tmp_path, fast_config):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "precious.txt").write_text("keep me")
    (cache / "ctp-looks-like-ours").mkdir()  # no marker file -> not ours
    p = tmp_path / "f"
    p.write_bytes(payload(2 * MiB))
    with CTPStream(FileSource(p), fast_config(storage="disk", max_cache_mb=4)) as stream:
        all_bytes(stream)
    assert (cache / "precious.txt").read_text() == "keep me"
    assert (cache / "ctp-looks-like-ours").is_dir()


def test_sweep_stale_only_removes_dead_marked_sessions(tmp_path):
    dead = tmp_path / "ctp-1-dead"
    dead.mkdir()
    (dead / MARKER).write_text("2147483646")  # pid that does not exist
    alive = tmp_path / "ctp-2-alive"
    alive.mkdir()
    (alive / MARKER).write_text(str(os.getpid()))
    foreign = tmp_path / "ctp-3-foreign"
    foreign.mkdir()
    removed = sweep_stale(str(tmp_path))
    assert removed == [str(dead)]
    assert alive.exists() and foreign.exists()


def test_session_dir_removed_on_gc(tmp_path):
    s = SessionDir(str(tmp_path))
    path = s.path
    assert path.exists()
    del s
    import gc

    gc.collect()
    assert not path.exists()


def test_disk_space_guard(tmp_path, fast_config, monkeypatch):
    import ctp.cache as cache_mod
    from ctp import CTPError

    monkeypatch.setattr(cache_mod.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024})())
    p = tmp_path / "f"
    p.write_bytes(b"x")
    stream = CTPStream(FileSource(p), fast_config(storage="disk", max_cache_mb=64))
    with pytest.raises(CTPError, match="disk space"):
        list(stream.chunks())


def test_auto_storage_picks_disk_when_buffer_exceeds_ram(tmp_path, fast_config, monkeypatch):
    import ctp.cache as cache_mod

    monkeypatch.setattr(cache_mod.psutil, "virtual_memory", lambda: type("M", (), {"available": 100 * MiB})())
    assert cache_mod.resolve_storage(fast_config(storage="auto", max_cache_mb=1024)) == "disk"
    assert cache_mod.resolve_storage(fast_config(storage="auto", max_cache_mb=10)) == "memory"


# --- resilience --------------------------------------------------------------


def test_resume_after_dropped_connection_yields_identical_bytes(server, fast_config):
    data = payload(3 * MiB)
    url = server.add("/f", data, drop_after=700_000, drop_times=2)
    stream = CTPStream(HTTPSource(url), fast_config())
    assert all_bytes(stream) == data
    assert stream.stats.retries == 2
    assert any(r["Range"] for r in server.routes["/f"].requests[1:])


def test_retry_on_503_then_success(server, fast_config):
    data = payload(100_000)
    url = server.add("/f", data, status=503, fail_status_times=2)
    stream = CTPStream(HTTPSource(url), fast_config())
    assert all_bytes(stream) == data
    assert stream.stats.retries == 2


def test_gives_up_after_max_retries(server, fast_config):
    url = server.add("/f", b"x" * 1000, status=503, fail_status_times=-1)
    stream = CTPStream(HTTPSource(url), fast_config(max_retries=2))
    with pytest.raises(SourceError, match="503"):
        all_bytes(stream)
    assert stream.stats.retries == 2


def test_permanent_error_is_not_retried(server, fast_config):
    stream = CTPStream(HTTPSource(server.base + "/nope"), fast_config())
    with pytest.raises(SourceError, match="404"):
        all_bytes(stream)
    assert stream.stats.retries == 0


def test_error_arrives_after_data_already_buffered(tmp_path, server, fast_config):
    good = tmp_path / "good"
    good.write_bytes(payload(200_000))
    stream = CTPStream([FileSource(good), HTTPSource(server.base + "/nope")], fast_config())
    got = bytearray()
    with pytest.raises(SourceError):
        for chunk in stream.stream():
            got += chunk
    assert bytes(got) == good.read_bytes()


def test_changed_file_aborts_instead_of_splicing(server, fast_config):
    data = payload(2 * MiB)
    url = server.add("/f", data, drop_after=500_000, drop_times=1)

    class Flip(HTTPSource):
        def open(self, offset, config):
            if offset:
                server.routes["/f"].etag = '"v2"'
                server.routes["/f"].ranges = False
            return super().open(offset, config)

    stream = CTPStream(Flip(url), fast_config())
    with pytest.raises(SourceError, match="changed"):
        all_bytes(stream)


def test_stalls_are_counted_for_slow_source(fast_config):
    from ctp import Source

    class Slow(Source):
        name = "slow"

        def open(self, offset, config):
            for _ in range(5):
                time.sleep(0.15)
                yield b"x" * 100

    stream = CTPStream(Slow(), fast_config())
    assert len(all_bytes(stream)) == 500
    assert stream.stats.stalls >= 3
    assert stream.stats.stall_seconds > 0.3
