import pytest

from ctp import Plan, recommend, run_benchmark
from ctp.benchmark import BenchmarkReport, CpuResult, DiskResult, GpuResult, NetworkResult, measure_network

from .conftest import jsonl_bytes


def make_report(*, ram=32.0, disk_write=2000.0, disk_free=500.0, net=None, gpu=True):
    return BenchmarkReport(
        platform="test",
        python="3.12",
        ram_total_gb=ram,
        ram_available_gb=ram,
        disk=DiskResult("/tmp", disk_free, disk_write, disk_write, 5000),
        cpu=CpuResult(8, 1000, 300, 100),
        gpu=GpuResult("Fake GPU", 24.0, 30.0, 60.0) if gpu else None,
        network=NetworkResult("http://x", 20.0, net, 10.0) if net is not None else None,
    )


def test_run_benchmark_small(tmp_path):
    report = run_benchmark(str(tmp_path), disk_mb=2)
    assert report.disk.write_mb_s > 0 and report.disk.read_mb_s > 0 and report.disk.delete_files_per_s > 0
    assert (
        report.cpu.sha256_mb_s > 0 and report.cpu.gzip_decompress_mb_s > 0 and report.cpu.json_parse_mb_s > 0
    )
    assert report.ram_total_gb > 0
    assert "Disk" in report.format()
    assert list(tmp_path.iterdir()) == []  # benchmark cleans up after itself


def test_measure_network_against_local_server(server):
    url = server.add("/d", jsonl_bytes(200_000))
    result = measure_network(url, seconds=2, max_mb=2)
    assert result.mb_s > 0 and result.sampled_mb > 0
    report = run_benchmark(disk_mb=2, url=url)
    assert report.network is not None


def test_network_probe_failure_becomes_note(server):
    report = run_benchmark(disk_mb=2, url=server.base + "/missing")
    assert report.network is None
    assert any("network probe failed" in n for n in report.notes)


def test_plan_prefers_memory_when_it_fits():
    plan = recommend(make_report(ram=32, net=50), consume_mb_s=20, ahead_seconds=60)
    assert plan.storage == "memory"
    assert plan.max_cache_mb == 1200  # 20 MiB/s x 60 s
    assert plan.ahead_seconds == 60


def test_plan_spills_to_disk_when_ram_is_small_and_disk_is_fast():
    plan = recommend(make_report(ram=4, disk_write=1500, net=500), consume_mb_s=100, ahead_seconds=60)
    assert plan.storage == "disk"
    assert plan.max_cache_mb == 6000


def test_plan_caps_lookahead_when_disk_is_slow():
    plan = recommend(make_report(ram=4, disk_write=20, net=500), consume_mb_s=100, ahead_seconds=60)
    assert plan.storage == "memory"
    assert plan.warnings and plan.ahead_seconds < 60
    assert plan.max_cache_mb == 1024  # 25% of 4 GiB


def test_plan_warns_when_network_is_the_bottleneck():
    plan = recommend(make_report(net=5), consume_mb_s=50)
    assert any("network" in w for w in plan.warnings)


def test_plan_without_measurements_uses_safe_defaults():
    plan = recommend(make_report(net=None))
    assert isinstance(plan, Plan) and plan.max_cache_mb == 1024


def test_plan_to_config_roundtrip():
    cfg = recommend(make_report(net=50), consume_mb_s=10).to_config(timeout=5)
    assert cfg.storage == "memory" and cfg.timeout == 5


def test_report_serialises():
    import json

    json.dumps(make_report(net=10).as_dict())


@pytest.mark.parametrize("gpu", [True, False])
def test_report_format_handles_gpu_presence(gpu):
    assert ("Fake GPU" in make_report(gpu=gpu).format()) is gpu
