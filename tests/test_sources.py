import pytest

from ctp import ConfigError, FileSource, HTTPSource, SourceError, resolve_sources
from ctp.sources import expand_braces, hf_url, open_source

DATA = bytes(range(256)) * 4096  # 1 MiB, position-dependent so splices are detectable


def read_all(source, cfg, offset=0):
    return b"".join(source.open(offset, cfg))


# --- spec resolution ---------------------------------------------------------


def test_expand_braces_zero_padded():
    assert expand_braces("s-{08..11}.jsonl") == ["s-08.jsonl", "s-09.jsonl", "s-10.jsonl", "s-11.jsonl"]
    assert expand_braces("plain") == ["plain"]


def test_expand_braces_multiple_and_invalid():
    assert len(expand_braces("a{0..1}-b{0..2}")) == 6
    with pytest.raises(ConfigError):
        expand_braces("x{5..1}")


def test_hf_url_variants():
    assert hf_url("hf://datasets/org/name/data/train.parquet") == (
        "https://huggingface.co/datasets/org/name/resolve/main/data/train.parquet"
    )
    assert hf_url("hf://org/name@v2/f.jsonl").endswith("/org/name/resolve/v2/f.jsonl")
    assert hf_url("hf://models/org/m/w.bin") == "https://huggingface.co/org/m/resolve/main/w.bin"
    with pytest.raises(ConfigError):
        hf_url("hf://datasets/org")


def test_hf_token_only_attached_to_hf(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert open_source("hf://org/name/f.jsonl").headers == {"Authorization": "Bearer secret"}  # type: ignore[attr-defined]
    assert open_source("https://example.com/f.jsonl").headers == {}  # type: ignore[attr-defined]


def test_glob_sorted_and_missing(tmp_path):
    for name in ("b.txt", "a.txt", "c.log"):
        (tmp_path / name).write_text("x")
    got = resolve_sources(str(tmp_path / "*.txt"))
    assert [s.name.split("/")[-1] for s in got] == ["a.txt", "b.txt"]
    with pytest.raises(ConfigError):
        resolve_sources(str(tmp_path / "*.nope"))


def test_resolve_rejects_empty_and_bad_scheme():
    with pytest.raises(ConfigError):
        resolve_sources([])
    with pytest.raises(ConfigError):
        HTTPSource("ftp://example.com/x")


def test_source_instances_pass_through(tmp_path):
    src = FileSource(tmp_path / "x")
    assert resolve_sources([src]) == [src]


# --- file source -------------------------------------------------------------


def test_file_source_offset_and_missing(tmp_path, fast_config):
    p = tmp_path / "f.bin"
    p.write_bytes(DATA)
    cfg = fast_config()
    assert read_all(FileSource(p), cfg, offset=1000) == DATA[1000:]
    assert FileSource(p).size(cfg) == len(DATA)
    with pytest.raises(SourceError):
        read_all(FileSource(tmp_path / "missing"), cfg)


# --- http source -------------------------------------------------------------


def test_http_full_read(server, fast_config):
    url = server.add("/f", DATA)
    cfg = fast_config()
    src = HTTPSource(url)
    assert read_all(src, cfg) == DATA
    assert src.size(cfg) == len(DATA)
    assert src.supports_range(cfg) is True


def test_http_resume_uses_range_and_if_range(server, fast_config):
    url = server.add("/f", DATA)
    cfg = fast_config()
    src = HTTPSource(url)
    read_all(src, cfg)  # learn the ETag
    assert read_all(src, cfg, offset=500_000) == DATA[500_000:]
    last = server.routes["/f"].requests[-1]
    assert last["Range"] == "bytes=500000-"
    assert last["If-Range"] == '"v1"'


def test_http_server_without_range_support_skips_prefix(server, fast_config):
    url = server.add("/f", DATA, ranges=False)
    cfg = fast_config()
    assert read_all(HTTPSource(url), cfg, offset=300_001) == DATA[300_001:]


def test_http_offset_at_end_is_empty(server, fast_config):
    url = server.add("/f", DATA)
    cfg = fast_config()
    src = HTTPSource(url)
    read_all(src, cfg)
    assert read_all(src, cfg, offset=len(DATA)) == b""


def test_http_404_is_fatal(server, fast_config):
    url = server.base + "/missing"
    with pytest.raises(SourceError) as info:
        read_all(HTTPSource(url), fast_config())
    assert info.value.retryable is False


def test_http_503_is_retryable(server, fast_config):
    url = server.add("/f", DATA, status=503, fail_status_times=1)
    with pytest.raises(SourceError) as info:
        read_all(HTTPSource(url), fast_config())
    assert info.value.retryable is True


def test_http_detects_file_changed_between_resumes(server, fast_config):
    url = server.add("/f", DATA)
    cfg = fast_config()
    src = HTTPSource(url)
    read_all(src, cfg)
    server.routes["/f"].etag = '"v2"'
    server.routes["/f"].ranges = False  # server answers 200 with a new ETag
    with pytest.raises(SourceError, match="changed"):
        read_all(src, cfg, offset=1000)


def test_http_connection_refused_is_retryable(fast_config):
    with pytest.raises(SourceError) as info:
        read_all(HTTPSource("http://127.0.0.1:9/nothing"), fast_config(timeout=1.0))
    assert info.value.retryable is True


def test_extra_headers_are_sent(server, fast_config):
    url = server.add("/f", b"abc")
    cfg = fast_config(headers={"X-Test": "1"})
    assert read_all(HTTPSource(url), cfg) == b"abc"
