import gzip
import io
import json
import logging

import pytest

from ctprotocol import ConfigError, CTProtocolDataset, DecodeError, Partition, resolve_sources
from ctprotocol.dataset import jsonl_stream, shuffle_buffer, text_stream

from .conftest import jsonl_bytes


def write_shards(tmp_path, n_shards, rows, gz=False):
    paths = []
    for s in range(n_shards):
        body = jsonl_bytes(rows, prefix=f"s{s}")
        p = tmp_path / f"shard-{s:03d}.jsonl{'.gz' if gz else ''}"
        p.write_bytes(gzip.compress(body) if gz else body)
        paths.append(str(p))
    return paths


def texts(ds):
    return [r["text"] if isinstance(r, dict) else r for r in ds]


def test_http_gzip_jsonl_end_to_end(server, fast_config):
    url = server.add("/train.jsonl.gz", gzip.compress(jsonl_bytes(1000)))
    ds = CTProtocolDataset(url, config=fast_config())
    rows = list(ds)
    assert [r["id"] for r in rows] == list(range(1000))
    assert ds.stats.records == 1000 and ds.stats.shards_completed == 1


def test_brace_range_over_http_preserves_shard_order(server, fast_config):
    for i in range(4):
        server.add(f"/p-{i:02d}.jsonl", jsonl_bytes(3, prefix=f"s{i}"))
    ds = CTProtocolDataset(f"{server.base}/p-{{00..03}}.jsonl", config=fast_config())
    assert texts(ds) == [f"s{s}-{i}" for s in range(4) for i in range(3)]


def test_text_field_and_transform(tmp_path, fast_config):
    (p,) = write_shards(tmp_path, 1, 5)
    ds = CTProtocolDataset(p, config=fast_config(), text_field="text", transform=str.upper)
    assert list(ds) == [f"S0-{i}" for i in range(5)]


def test_missing_text_field_raises(tmp_path, fast_config):
    (p,) = write_shards(tmp_path, 1, 2)
    with pytest.raises(DecodeError, match="nope"):
        list(CTProtocolDataset(p, config=fast_config(), text_field="nope"))


def test_skip_errors_config(tmp_path, fast_config):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a":1}\nbroken\n{"a":2}\n')
    ds = CTProtocolDataset(str(p), config=fast_config(on_error="skip"))
    assert list(ds) == [{"a": 1}, {"a": 2}]
    assert ds.stats.records_skipped == 1


def test_format_validation_is_eager(tmp_path):
    with pytest.raises(ConfigError):
        CTProtocolDataset("data.bin")  # cannot infer
    with pytest.raises(ConfigError):
        CTProtocolDataset("data.jsonl", format="xml")
    assert CTProtocolDataset("data.bin", format="raw")  # explicit format is fine


def test_lines_format(tmp_path, fast_config):
    p = tmp_path / "t.txt.gz"
    p.write_bytes(gzip.compress(b"one\n\ntwo\nthree"))
    assert list(CTProtocolDataset(str(p), config=fast_config())) == ["one", "two", "three"]


def test_parquet_end_to_end_from_http(server, fast_config):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(
        pa.table({"text": [f"t{i}" for i in range(500)], "x": list(range(500))}), buf, row_group_size=64
    )
    url = server.add("/d.parquet", buf.getvalue())
    ds = CTProtocolDataset(url, config=fast_config(storage="disk"), columns=["text"], text_field="text")
    assert list(ds) == [f"t{i}" for i in range(500)]


# --- shuffling ---------------------------------------------------------------


def test_shuffle_buffer_is_a_permutation_and_deterministic():
    import random

    a = list(shuffle_buffer(range(1000), 64, random.Random(1)))
    b = list(shuffle_buffer(range(1000), 64, random.Random(1)))
    c = list(shuffle_buffer(range(1000), 64, random.Random(2)))
    assert sorted(a) == list(range(1000)) and a == b and a != c and a != list(range(1000))


def test_shuffle_buffer_size_one_is_identity():
    import random

    assert list(shuffle_buffer(range(10), 1, random.Random(0))) == list(range(10))


def test_epoch_changes_order_but_same_epoch_repeats(tmp_path, fast_config):
    paths = write_shards(tmp_path, 6, 20)
    ds = CTProtocolDataset(paths, config=fast_config(), shuffle_shards=True, shuffle_buffer=16, seed=7)
    e0, e0_again = texts(ds), texts(ds)
    ds.set_epoch(1)
    e1 = texts(ds)
    assert e0 == e0_again
    assert e0 != e1
    assert sorted(e0) == sorted(e1)
    assert len(e0) == 120


# --- partitioning ------------------------------------------------------------


def test_partition_validation():
    with pytest.raises(ConfigError):
        Partition(3, 3)
    assert Partition.combine(rank=1, world_size=2, worker_id=2, num_workers=4) == Partition(6, 8)


def test_shards_split_disjointly_and_completely(tmp_path, fast_config):
    paths = write_shards(tmp_path, 6, 10)
    parts = [texts(CTProtocolDataset(paths, config=fast_config(), partition=Partition(i, 3))) for i in range(3)]
    assert all(len(p) == 20 for p in parts)
    flat = [t for p in parts for t in p]
    assert len(flat) == len(set(flat)) == 60


def test_record_striding_fallback_when_fewer_shards_than_consumers(tmp_path, fast_config, caplog):
    paths = write_shards(tmp_path, 1, 30)
    with caplog.at_level(logging.WARNING, logger="ctprotocol"):
        parts = [texts(CTProtocolDataset(paths, config=fast_config(), partition=Partition(i, 4))) for i in range(4)]
    flat = sorted(t for p in parts for t in p)
    assert flat == sorted(f"s0-{i}" for i in range(30))
    assert "record striding" in caplog.text


def test_shuffled_shard_order_agrees_across_ranks(tmp_path, fast_config):
    paths = write_shards(tmp_path, 8, 2)
    parts = [
        texts(CTProtocolDataset(paths, config=fast_config(), partition=Partition(i, 2), shuffle_shards=True, seed=3))
        for i in range(2)
    ]
    assert not set(parts[0]) & set(parts[1])
    assert len(parts[0]) + len(parts[1]) == 16


# --- lifecycle ---------------------------------------------------------------


def test_early_break_cleans_up(tmp_path, fast_config):
    paths = write_shards(tmp_path, 3, 2000)
    cfg = fast_config(storage="disk", max_cache_mb=2, min_buffer_mb=1)
    ds = CTProtocolDataset(paths, config=cfg)
    for i, _ in enumerate(ds):
        if i == 10:
            break
    ds.close()
    cache = tmp_path / "cache"
    assert not cache.exists() or list(cache.iterdir()) == []


def test_context_manager_and_error_cleanup(tmp_path, fast_config):
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b"garbage\n")
    with pytest.raises(DecodeError), CTProtocolDataset(str(p), config=fast_config(storage="disk")) as ds:
        list(ds)
    cache = tmp_path / "cache"
    assert not cache.exists() or list(cache.iterdir()) == []


def test_resolve_sources_accepts_mixed_specs(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}")
    got = resolve_sources([str(tmp_path / "*.jsonl"), "http://example.com/x.jsonl"])
    assert len(got) == 2


def test_v01_helpers():
    chunks = [b'{"a":', b"1}\n{", b'"a":2}']
    assert list(jsonl_stream(chunks)) == [{"a": 1}, {"a": 2}]
    assert list(text_stream([b"ab", b"\xffc"])) == ["ab", "\ufffdc"]
    assert json.loads('{"ok":1}') == {"ok": 1}
