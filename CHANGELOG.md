# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may contain breaking changes).

## [0.2.0]

First production-oriented release; a ground-up rework of the 0.1 prototype.

### Added
- **Adaptive prefetcher**: buffers `ahead_seconds x measured consumer demand` bytes, clamped by
  `max_cache_mb`. Demand is measured from the consumer side so a starved consumer never shrinks the lookahead.
- **Memory and disk buffer tiers** (`storage="auto" | "memory" | "disk"`); consumed disk chunks are deleted immediately.
- **Resumable HTTP**: `Range` + `If-Range` resume after dropped connections, exponential backoff with jitter,
  detection of files that change mid-stream, graceful fallback for servers without range support.
- **`CTPDataset`**: record-level iteration for JSONL, text lines, Parquet and raw bytes; automatic
  gzip / bzip2 / zstd detection; shard order shuffle, bounded shuffle buffer, deterministic per `(seed, epoch)`.
- **Sharding**: `hf://` URLs, glob patterns, numeric brace ranges (`train-{000..127}.jsonl.gz`),
  partitioning across distributed ranks and DataLoader workers.
- **Integrations**: `CTPIterableDataset` (PyTorch) and `to_hf_iterable` (Hugging Face `datasets`), loaded lazily.
- **Benchmark + planner**: disk write/read/delete speed, CPU decode speed, GPU throughput, optional network probe,
  and a concrete configuration recommendation (`ctp benchmark`, `ctp.run_benchmark`, `ctp.recommend`).
- **Observability**: `StreamStats` (bytes, stalls, retries, peak buffer, demand), `logging` under `ctp`.
- CLI subcommands `info` and `clean`; `python -m ctp`; `--json` output; typed package (`py.typed`).
- Test suite with a local range-capable HTTP server that injects failures; CI workflow (Linux/macOS/Windows, Python 3.10-3.13).

### Changed
- Temporary data now lives in a private `ctp-<pid>-<id>` session directory. **CTP no longer deletes
  every file in `cache_dir`**, which was unsafe in 0.1.
- Default cache location is `<system temp>/ctp` (or `$CTP_CACHE_DIR`) instead of `./.ctp_cache`.
- `CTPConfig` is now frozen and validated.
- Dropped the `tqdm` dependency; `beautifulsoup4`/`web` extra removed (unused).

### Removed
- `CTPConfig.delete_consumed` (consumed data is always deleted).

### Compatibility
`CTPStream.stream()`, `CTPStream.cleanup()`, `URLSource`, `FileSource`, `benchmark_system()`,
`jsonl_stream()` and `text_stream()` from 0.1 still work.

## [0.1.0]
- Initial prototype: sequential HTTP streaming, basic benchmark, CLI.
