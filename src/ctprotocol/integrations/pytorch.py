"""PyTorch integration: ``CTProtocolIterableDataset`` for ``torch.utils.data.DataLoader``.

Requires ``pip install 'ctp-training[torch]'``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from ..config import CTProtocolConfig
from ..dataset import CTProtocolDataset, Partition
from ..errors import OptionalDependencyError
from ..sources import SourceLike

try:
    import torch.distributed as dist
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise OptionalDependencyError("The PyTorch integration", "torch", "torch") from exc

__all__ = ["CTProtocolIterableDataset"]


def _distributed_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class CTProtocolIterableDataset(IterableDataset):  # type: ignore[misc]
    """Drop-in ``IterableDataset`` backed by :class:`ctprotocol.CTProtocolDataset`.

    * Splits shards across distributed ranks *and* DataLoader workers automatically.
    * Divides ``config.max_cache_mb`` between local workers, so the total
      temporary footprint of one process stays within the configured budget.
    * Each worker owns its own prefetch thread and session directory, all of
      which are deleted when iteration ends.

    Example::

        ds = CTProtocolIterableDataset("hf://datasets/org/name/train-{0000..0031}.parquet",
                                columns=["text"], text_field="text", shuffle_buffer=10_000)
        loader = DataLoader(ds, batch_size=32, num_workers=4)
        for epoch in range(3):
            ds.set_epoch(epoch)
            for batch in loader: ...
    """

    def __init__(
        self,
        sources: SourceLike | Iterable[SourceLike],
        *,
        format: str | None = None,
        config: CTProtocolConfig | None = None,
        text_field: str | None = None,
        columns: Sequence[str] | None = None,
        shuffle_shards: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 0,
        transform: Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        self._kwargs: dict[str, Any] = {
            "format": format,
            "text_field": text_field,
            "columns": columns,
            "shuffle_shards": shuffle_shards,
            "shuffle_buffer": shuffle_buffer,
            "seed": seed,
            "transform": transform,
        }
        self._sources = sources
        self._config = config or CTProtocolConfig()
        self.epoch = 0
        # Validate sources/format now so mistakes surface in the main process.
        CTProtocolDataset(sources, config=self._config, **self._kwargs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Any]:
        rank, world = _distributed_info()
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        config = self._config
        if num_workers > 1:
            config = config.replace(max_cache_mb=max(1, config.max_cache_mb // num_workers))
        dataset = CTProtocolDataset(
            self._sources,
            config=config,
            partition=Partition.combine(rank, world, worker_id, num_workers),
            **self._kwargs,
        )
        dataset.set_epoch(self.epoch)
        with dataset:
            yield from dataset
