import gzip
import json

import pytest

from ctprotocol.cli import main

from .conftest import jsonl_bytes


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "ctp" in capsys.readouterr().out


def test_stream_prints_records_and_stats(server, tmp_path, capsys):
    url = server.add("/d.jsonl.gz", gzip.compress(jsonl_bytes(50)))
    code = main(["stream", url, "--limit", "3", "--stats", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 0
    assert [json.loads(line)["id"] for line in out.out.splitlines()] == [0, 1, 2]
    assert "MiB consumed" in out.err


def test_stream_text_field_and_no_limit(tmp_path, capsys):
    p = tmp_path / "d.jsonl"
    p.write_bytes(jsonl_bytes(7))
    assert (
        main(["stream", str(p), "--limit", "0", "--text-field", "text", "--cache-dir", str(tmp_path / "c")])
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [f"row-{i}" for i in range(7)]


def test_stream_error_gives_clean_message_and_exit_code(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text("nope\n")
    assert main(["stream", str(p), "--cache-dir", str(tmp_path / "c")]) == 1
    assert "ctp: error:" in capsys.readouterr().err


def test_stream_skip_errors(tmp_path, capsys):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a":1}\nnope\n{"a":2}\n')
    assert main(["stream", str(p), "--skip-errors", "--limit", "0", "--cache-dir", str(tmp_path / "c")]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_info(server, capsys):
    url = server.add("/d.jsonl.gz", b"x" * 2048)
    assert main(["info", url]) == 0
    out = capsys.readouterr().out
    assert "yes (resumable)" in out and "jsonl" in out and "MiB" in out


def test_info_unknown_format(tmp_path, capsys):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"1")
    assert main(["info", str(p)]) == 0
    assert "unknown" in capsys.readouterr().out


def test_benchmark_json(tmp_path, capsys):
    assert main(["benchmark", "--cache-dir", str(tmp_path), "--json", "--consume-mb-s", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {"report", "plan"} <= data.keys()
    assert data["plan"]["storage"] in ("memory", "disk")


def test_benchmark_text(tmp_path, capsys):
    assert main(["benchmark", "--cache-dir", str(tmp_path)]) == 0
    assert "Recommended configuration" in capsys.readouterr().out


def test_clean(tmp_path, capsys):
    stale = tmp_path / "ctp-1-x"
    stale.mkdir()
    (stale / ".ctp-session").write_text("2147483646")
    assert main(["clean", "--cache-dir", str(tmp_path)]) == 0
    assert not stale.exists()
    assert "1 stale" in capsys.readouterr().out
