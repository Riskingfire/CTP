"""Stream a remote dataset record by record. Nothing is saved to disk (memory tier)
or, if the buffer is large, only a bounded rolling window is (disk tier)."""

from ctprotocol import CTProtocolConfig, CTProtocolDataset

dataset = CTProtocolDataset(
    "https://example.com/data/train-{000..015}.jsonl.gz",  # 16 shards, gzip detected automatically
    text_field="text",
    config=CTProtocolConfig(ahead_seconds=60, max_cache_mb=1024),
    shuffle_shards=True,
    shuffle_buffer=10_000,
    seed=42,
)

with dataset:
    for step, text in enumerate(dataset):
        tokens = text.split()  # replace with your tokenizer / training step
        if step % 10_000 == 0:
            print(dataset.stats.summary())
