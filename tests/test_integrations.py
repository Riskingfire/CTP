"""Integration adapters.

Real-torch tests run when torch is installed. The stub-based tests exercise the
partitioning/cache-splitting logic of the adapter without needing torch.
"""

import importlib
import sys
import types

import pytest

import ctprotocol
from ctprotocol import OptionalDependencyError

from .conftest import jsonl_bytes


def _write(tmp_path, n_shards=4, rows=10):
    paths = []
    for s in range(n_shards):
        p = tmp_path / f"s{s}.jsonl"
        p.write_bytes(jsonl_bytes(rows, prefix=f"s{s}"))
        paths.append(str(p))
    return paths


@pytest.fixture
def fake_torch(monkeypatch):
    """Minimal stand-in for torch.utils.data / torch.distributed."""
    state = types.SimpleNamespace(worker=None, dist=(0, 1))

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    distributed = types.ModuleType("torch.distributed")

    class IterableDataset:
        pass

    data.IterableDataset = IterableDataset
    data.get_worker_info = lambda: state.worker
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: state.dist[1] > 1
    distributed.get_rank = lambda: state.dist[0]
    distributed.get_world_size = lambda: state.dist[1]
    torch.utils, torch.distributed, utils.data = utils, distributed, data
    for name, mod in {
        "torch": torch,
        "torch.utils": utils,
        "torch.utils.data": data,
        "torch.distributed": distributed,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "ctprotocol.integrations.pytorch", raising=False)
    module = importlib.import_module("ctprotocol.integrations.pytorch")
    yield module, state
    sys.modules.pop("ctprotocol.integrations.pytorch", None)


def test_lazy_attribute_raises_helpful_error_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    monkeypatch.delitem(sys.modules, "ctprotocol.integrations.pytorch", raising=False)
    with pytest.raises(OptionalDependencyError, match=r"ctp-training\[torch\]"):
        ctprotocol.CTProtocolIterableDataset  # noqa: B018


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        ctprotocol.definitely_not_here  # noqa: B018


def test_adapter_splits_across_workers_and_ranks(fake_torch, tmp_path, fast_config):
    module, state = fake_torch
    paths = _write(tmp_path, n_shards=4)
    ds = module.CTProtocolIterableDataset(paths, config=fast_config(), text_field="text")

    seen = []
    for rank in range(2):
        for worker in range(2):
            state.dist = (rank, 2)
            state.worker = types.SimpleNamespace(id=worker, num_workers=2)
            seen.append(list(ds))
    assert all(len(part) == 10 for part in seen)
    flat = [t for part in seen for t in part]
    assert len(flat) == len(set(flat)) == 40


def test_adapter_single_process_reads_everything(fake_torch, tmp_path, fast_config):
    module, _ = fake_torch
    ds = module.CTProtocolIterableDataset(_write(tmp_path), config=fast_config(), text_field="text")
    assert len(list(ds)) == 40


def test_adapter_divides_cache_budget_between_workers(fake_torch, tmp_path, fast_config, monkeypatch):
    module, state = fake_torch
    captured = {}
    real = module.CTProtocolDataset

    def spy(*args, **kwargs):
        captured["max_cache_mb"] = kwargs["config"].max_cache_mb
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "CTProtocolDataset", spy)
    ds = module.CTProtocolIterableDataset(_write(tmp_path), config=fast_config(max_cache_mb=800))
    state.worker = types.SimpleNamespace(id=0, num_workers=4)
    list(ds)
    assert captured["max_cache_mb"] == 200


def test_adapter_validates_eagerly(fake_torch, tmp_path):
    module, _ = fake_torch
    with pytest.raises(ctprotocol.ConfigError):
        module.CTProtocolIterableDataset(str(tmp_path / "x.bin"))


def test_adapter_epoch_changes_shard_order(fake_torch, tmp_path, fast_config):
    module, _ = fake_torch
    ds = module.CTProtocolIterableDataset(
        _write(tmp_path, 8, 2), config=fast_config(), shuffle_shards=True, text_field="text"
    )
    ds.set_epoch(0)
    a = list(ds)
    ds.set_epoch(1)
    b = list(ds)
    assert a != b and sorted(a) == sorted(b)


def test_real_torch_dataloader(tmp_path, fast_config):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from ctprotocol.integrations.pytorch import CTProtocolIterableDataset

    ds = CTProtocolIterableDataset(_write(tmp_path), config=fast_config(), text_field="text")
    got = [t for batch in DataLoader(ds, batch_size=4, num_workers=2) for t in batch]
    assert len(got) == 40 and len(set(got)) == 40
    assert torch is not None


def test_real_hf_datasets(tmp_path, fast_config):
    pytest.importorskip("datasets")
    from ctprotocol import CTProtocolDataset, to_hf_iterable

    hf = to_hf_iterable(CTProtocolDataset(_write(tmp_path, 2, 5), config=fast_config()))
    assert len(list(hf)) == 10
