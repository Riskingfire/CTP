# CTP - Caching Training Protocol

Train on remote datasets **without downloading them first**. CTP streams data from a URL,
Hugging Face repo or local files through a small, bounded, self-cleaning buffer that stays
roughly *one minute ahead* of your training loop, then deletes everything it wrote.

```python
from ctp import CTPDataset

dataset = CTPDataset(
    "https://example.com/train-{000..127}.jsonl.gz",
    text_field="text",
    shuffle_shards=True,
    shuffle_buffer=10_000,
)

for text in dataset:  # streamed, decompressed, decoded, shuffled - nothing left on disk
    train_step(text)
```

- **Bounded footprint** - never buffers more than `max_cache_mb`, in RAM or on disk.
- **Adaptive lookahead** - measures how fast *your* loop consumes data and keeps `ahead_seconds` of it ready.
- **Self-cleaning** - consumed data is deleted immediately; only CTP's own session directory is ever touched.
- **Resilient** - resumes dropped connections with HTTP `Range`, retries with backoff, refuses to splice a file that changed mid-stream.
- **Drop-in** - PyTorch `IterableDataset`, Hugging Face `datasets`/`Trainer`, or plain Python iteration.
- **Formats** - JSONL, text, Parquet, raw bytes; gzip / bzip2 / zstd detected automatically.

## Install

```bash
pip install ctp-training                 # core: requests + psutil only
pip install "ctp-training[torch]"        # PyTorch adapter
pip install "ctp-training[hf]"           # Hugging Face datasets adapter
pip install "ctp-training[parquet,zstd]" # Parquet and zstd support
pip install "ctp-training[all]"
```

Requires Python 3.10+. The distribution is `ctp-training`; the import name and command are `ctp`
(see [Naming](#naming)).

## Quick start

### Sources

| You pass | Meaning |
|---|---|
| `"https://host/data.jsonl.gz"` | one remote file |
| `"https://host/train-{000..127}.parquet"` | 128 shards (zero-padded numeric range) |
| `"hf://datasets/org/name/data/train-0000.parquet"` | file in a Hugging Face repo (`@revision` supported; uses `HF_TOKEN` if set) |
| `"data/*.jsonl.gz"` | local glob, sorted |
| `[url1, url2, "local.jsonl"]` | any mix; a custom `ctp.Source` subclass also works |

The record format is inferred from the file name (`.jsonl`/`.ndjson`, `.txt`, `.parquet`);
override it with `format="jsonl" | "lines" | "parquet" | "raw"`.

### PyTorch

```python
from torch.utils.data import DataLoader
from ctp import CTPConfig
from ctp.integrations.pytorch import CTPIterableDataset

ds = CTPIterableDataset(
    "hf://datasets/org/name/data/train-{00000..00031}.parquet",
    columns=["text"],
    text_field="text",
    shuffle_shards=True,
    shuffle_buffer=20_000,
    config=CTPConfig(max_cache_mb=2048),
)
loader = DataLoader(ds, batch_size=32, num_workers=4)

for epoch in range(3):
    ds.set_epoch(epoch)  # reshuffles shard order and buffer
    for batch in loader:
        ...
```

Shards are split across distributed ranks **and** DataLoader workers automatically, and the cache budget
is divided between local workers so one process never exceeds `max_cache_mb`.

### Hugging Face

```python
from ctp import CTPDataset, to_hf_iterable

train = to_hf_iterable(CTPDataset("https://host/train-{000..015}.jsonl.gz"))
train = train.map(lambda ex: tokenizer(ex["text"]), batched=True)
# Trainer(..., train_dataset=train, args=TrainingArguments(max_steps=...))   # max_steps is required
```

More in [`examples/`](examples).

## Tuning

```bash
ctp benchmark --url https://host/train-000.jsonl.gz --consume-mb-s 25
```

measures disk write/read/delete speed, CPU decode speed, GPU throughput and network throughput to your data host,
then prints a recommended `storage`, `ahead_seconds` and `max_cache_mb`, with the reasoning and warnings
(for example when your network is slower than your training loop, which no amount of prefetching can fix).
Programmatically: `ctp.run_benchmark(...)` -> `ctp.recommend(report, consume_mb_s=...)` -> `plan.to_config()`.

> CTP cannot benchmark *your model*. Instead it measures your loop's real consumption rate at runtime
> (`dataset.stats.demand_bytes_per_s`) and adapts the lookahead continuously.

| `CTPConfig` field | Default | Meaning |
|---|---|---|
| `storage` | `"auto"` | `"memory"`, `"disk"`, or `"auto"` (memory if the buffer fits in 25% of free RAM) |
| `ahead_seconds` | `60` | target lookahead = `ahead_seconds x` measured consumption rate |
| `max_cache_mb` | `1024` | hard cap on buffered bytes |
| `min_buffer_mb` / `initial_buffer_mb` | `8` / `32` | buffer floor, and target before a rate is measured |
| `cache_dir` | `<tmp>/ctp` | base for session directories (`$CTP_CACHE_DIR`) |
| `chunk_size` | 1 MiB | read size |
| `timeout`, `max_retries`, `retry_backoff` | `30`, `5`, `0.5` | network resilience |
| `headers` | `{}` | extra HTTP headers, e.g. auth |
| `on_error` | `"raise"` | `"skip"` malformed records and count them in `stats.records_skipped` |

`CTPConfig.from_env()` reads `CTP_CACHE_DIR`, `CTP_STORAGE`, `CTP_AHEAD_SECONDS`, `CTP_MAX_CACHE_MB`.

### Observability

`dataset.stats` (a `StreamStats`) reports bytes downloaded/consumed, peak and target buffer, retries, and
**stalls** - how often your loop had to wait for data. Steady-state stalls mean the network (or
`ahead_seconds`) is your bottleneck, not your model. CTP logs to the standard `ctp` logger.

```python
print(dataset.stats.summary())
# 400.0 MiB consumed, 400.0 MiB downloaded (76.7 MiB/s), peak buffer 32.0 MiB [disk], 0 stalls (0.00s), 0 retries, 404465 records
```

## How it works

```
 source(s) ──► producer thread ──► bounded buffer ──► your loop
 (HTTP/file)   Range-resume,       memory or disk,     decode → shuffle
               retry, backoff      target =            → train
                     ▲             ahead_seconds × rate       │
                     └──── pauses when full ◄───── consumed chunks deleted
```

The producer is paused whenever the buffer holds `ahead_seconds x demand` bytes (clamped to
`[min_buffer_mb, max_cache_mb]`). *Demand* is an exponential moving average of the rate at which
the consumer works through data, measured from the time it spends **outside** CTP, so waiting for data can
never lower the estimate and cause a starvation spiral.

## Guarantees and limits

- **The network is still used.** No software can train on bytes it has not received. CTP lowers *storage*
  (persistent and peak), not traffic. Every epoch re-downloads the data.
- **"One minute ahead" is a target, not a guarantee.** It depends on network speed, server throttling and
  `max_cache_mb`. If the network is slower than your training loop, you will stall; `ctp benchmark` warns you.
- **Shuffling is approximate.** Shard order is exact-random; records are mixed through a bounded buffer. For
  good mixing, use many shards (>= 8 per consumer) and a large `shuffle_buffer`.
- **Sharding.** With fewer shards than consumers, CTP falls back to record striding, so every consumer downloads
  all the data (a warning is logged). Split the dataset into more shards.
- **Parquet** needs random access, so *one shard at a time* is spooled to the session directory and deleted right
  after reading. Keep shards to a few hundred MB.
- **Gzip/bzip2/zstd** are decoded incrementally with a per-step output cap. Compression is detected from the bytes themselves, not the file name.
- **Trust.** Sources are fetched as given. Do not pass untrusted URLs where your process can reach internal services.

## CLI

```bash
ctp benchmark [--url URL] [--consume-mb-s N] [--json]   # measure + recommend
ctp stream SRC... [--limit N] [--storage disk] --stats   # preview records / smoke-test a source
ctp info SRC...                                          # size, range support, detected format
ctp clean                                                # remove sessions left by crashed runs
```

## Naming

`ctp` is already the name of an unrelated PyPI project, so this package is published as **`ctp-training`** while
exposing the module `ctp` and command `ctp`. Do not install both distributions into the same environment. If that is
a problem for you, renaming the import package is a mechanical change (`src/ctp`, `pyproject.toml`, imports).

## Upgrading from 0.1

`CTPStream.stream()`, `cleanup()`, `URLSource`, `FileSource`, `benchmark_system()`, `jsonl_stream()` and `text_stream()` keep
working. Breaking changes: `delete_consumed` is gone (consumed data is always deleted), CTP now only deletes its own
session directory rather than every file in `cache_dir`, and the default cache location moved to the system temp dir.
See [CHANGELOG.md](CHANGELOG.md).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). `pip install -e ".[dev]"`, then `ruff check . && mypy && pytest`.

## License

MIT
