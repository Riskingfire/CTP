import bz2
import gzip
import io
import json

import pytest

from ctprotocol import ConfigError, DecodeError
from ctprotocol.formats import Inflater, detect_format, make_decoder
from ctprotocol.stats import StreamStats

from .conftest import jsonl_bytes


def decode(
    fmt, data, *, step=1000, on_error="raise", stats=None, workdir=None, columns=None, max_record=1 << 20
):
    stats = stats or StreamStats()
    dec = make_decoder(
        fmt,
        name="t",
        on_error=on_error,
        max_record_bytes=max_record,
        stats=stats,
        workdir=workdir or (lambda: None),
        columns=columns,
    )
    out = []
    for i in range(0, len(data), step):
        out.extend(dec.feed(data[i : i + step]))
    out.extend(dec.finish())
    return out


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.jsonl", "jsonl"),
        ("a.ndjson.gz", "jsonl"),
        ("https://h/x.jsonl.zst?sig=1", "jsonl"),
        ("a.txt.bz2", "lines"),
        ("part-0.parquet", "parquet"),
        ("A.JSONL", "jsonl"),
    ],
)
def test_detect_format(name, expected):
    assert detect_format(name) == expected


def test_detect_format_unknown():
    with pytest.raises(ConfigError):
        detect_format("data.bin")


# --- compression -------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 7, 4096, 10**9])
def test_jsonl_records_survive_any_chunking(step):
    data = jsonl_bytes(50)
    assert decode("jsonl", data, step=step) == [{"id": i, "text": f"row-{i}"} for i in range(50)]


@pytest.mark.parametrize("compress", [gzip.compress, bz2.compress])
@pytest.mark.parametrize("step", [3, 512, 10**9])
def test_compressed_jsonl(compress, step):
    data = jsonl_bytes(200)
    assert decode("jsonl", compress(data), step=step) == decode("jsonl", data)


def test_multi_member_gzip_and_bz2():
    a, b = jsonl_bytes(5, "a"), jsonl_bytes(5, "b")
    for compress in (gzip.compress, bz2.compress):
        got = decode("jsonl", compress(a) + compress(b), step=13)
        assert [r["text"] for r in got] == [f"a-{i}" for i in range(5)] + [f"b-{i}" for i in range(5)]


def test_zstd_roundtrip():
    zstandard = pytest.importorskip("zstandard")
    data = jsonl_bytes(100)
    comp = zstandard.ZstdCompressor().compress(data)
    assert decode("jsonl", comp, step=17) == decode("jsonl", data)


def test_zstd_multi_frame_and_truncation():
    zstandard = pytest.importorskip("zstandard")
    c = zstandard.ZstdCompressor()
    a, b = jsonl_bytes(5, "a"), jsonl_bytes(5, "b")
    got = decode("jsonl", c.compress(a) + c.compress(b), step=11)
    assert len(got) == 10 and got[5]["text"] == "b-0"
    comp = c.compress(jsonl_bytes(500))
    with pytest.raises(DecodeError, match="truncated"):
        decode("jsonl", comp[: len(comp) // 2])


@pytest.mark.parametrize("compress", [gzip.compress, bz2.compress])
def test_truncated_stream_is_detected(compress):
    comp = compress(jsonl_bytes(500))
    with pytest.raises(DecodeError, match="truncated"):
        decode("jsonl", comp[: len(comp) // 2])


def test_corrupt_gzip_raises_decode_error():
    comp = bytearray(gzip.compress(jsonl_bytes(500)))
    for i in range(20, 60):
        comp[i] ^= 0xFF
    with pytest.raises(DecodeError):
        decode("jsonl", bytes(comp))


def test_decompression_output_is_capped_per_step():
    bomb = gzip.compress(b"\0" * (64 * 1024 * 1024), compresslevel=9)
    assert len(bomb) < 100_000
    inflater = Inflater()
    largest = 0
    for piece in inflater.feed(bomb):
        largest = max(largest, len(piece))
    assert largest <= 4 * 1024 * 1024


def test_tiny_inputs_and_empty_input():
    assert decode("jsonl", b"") == []
    assert decode("jsonl", b"1\n") == [1]  # shorter than the 4-byte magic window
    assert decode("lines", b"") == []


# --- jsonl / lines -----------------------------------------------------------


def test_jsonl_blank_lines_crlf_and_no_trailing_newline():
    assert decode("jsonl", b'{"a":1}\r\n\r\n{"a":2}') == [{"a": 1}, {"a": 2}]


def test_jsonl_bad_record_raises_with_location():
    with pytest.raises(DecodeError, match="line 2"):
        decode("jsonl", b'{"a":1}\nnot json\n{"a":3}\n')


def test_jsonl_bad_record_skipped_and_counted():
    stats = StreamStats()
    got = decode("jsonl", b'{"a":1}\nnot json\n\xff\xfe\n{"a":3}\n', on_error="skip", stats=stats)
    assert got == [{"a": 1}, {"a": 3}]
    assert stats.records_skipped == 2


def test_oversized_record_without_newline_is_rejected():
    with pytest.raises(DecodeError, match="max_record_mb"):
        decode("jsonl", b"x" * 5000, step=1000, max_record=2000)


def test_lines_format_skips_blank_and_handles_unicode():
    data = "héllo\r\n\n  \nwörld".encode()
    assert decode("lines", data, step=2) == ["héllo", "wörld"]


def test_raw_format_passes_bytes_through():
    assert b"".join(decode("raw", b"abcdef", step=2)) == b"abcdef"
    assert b"".join(decode("raw", gzip.compress(b"abcdef"), step=2)) == b"abcdef"


def test_unknown_format():
    with pytest.raises(ConfigError):
        make_decoder(
            "xml", name="t", on_error="raise", max_record_bytes=1, stats=StreamStats(), workdir=lambda: None
        )


# --- parquet -----------------------------------------------------------------


def make_parquet(n=1000):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"id": list(range(n)), "text": [f"t{i}" for i in range(n)], "junk": [0] * n})
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=100)
    return buf.getvalue()


def test_parquet_records_and_spool_is_deleted(tmp_path):
    data = make_parquet()
    got = decode("parquet", data, step=777, workdir=lambda: tmp_path)
    assert [r["id"] for r in got] == list(range(1000))
    assert list(tmp_path.iterdir()) == []


def test_parquet_column_projection(tmp_path):
    got = decode("parquet", make_parquet(10), workdir=lambda: tmp_path, columns=["text"])
    assert got[0] == {"text": "t0"}


def test_parquet_garbage_raises_and_cleans_up(tmp_path):
    with pytest.raises(DecodeError):
        decode("parquet", b"this is not parquet" * 100, workdir=lambda: tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_parquet_abort_removes_partial_spool(tmp_path):
    dec = make_decoder(
        "parquet",
        name="t",
        on_error="raise",
        max_record_bytes=1,
        stats=StreamStats(),
        workdir=lambda: tmp_path,
    )
    list(dec.feed(make_parquet(10)[:50]))
    assert list(tmp_path.iterdir())
    dec.abort()
    assert list(tmp_path.iterdir()) == []


def test_json_roundtrip_unicode():
    rec = {"text": "日本語 🚀"}
    assert decode("jsonl", (json.dumps(rec, ensure_ascii=False) + "\n").encode(), step=3) == [rec]
