"""Hugging Face ``datasets`` / ``Trainer`` integration.

Requires ``pip install 'ctp-training[hf]'``.
"""

from __future__ import annotations

from typing import Any

from ..dataset import CTProtocolDataset
from ..errors import OptionalDependencyError

try:
    from datasets import Features, IterableDataset
except ImportError as exc:  # pragma: no cover - exercised only without datasets
    raise OptionalDependencyError("The Hugging Face integration", "datasets", "hf") from exc

__all__ = ["to_hf_iterable"]


def to_hf_iterable(dataset: CTProtocolDataset, features: Features | None = None) -> Any:
    """Wrap a :class:`ctprotocol.CTProtocolDataset` as a ``datasets.IterableDataset``.

    Records must be dicts (use ``format="jsonl"`` or ``"parquet"`` and no
    ``text_field``). The result works with ``.map``, ``.shuffle`` and
    ``transformers.Trainer`` (set ``max_steps``, since the length is unknown)::

        hf = to_hf_iterable(CTProtocolDataset("https://host/train.jsonl.gz"))
        hf = hf.map(lambda ex: tokenizer(ex["text"]), batched=True)
    """

    def generate() -> Any:
        with dataset:
            yield from dataset

    return IterableDataset.from_generator(generate, features=features)
