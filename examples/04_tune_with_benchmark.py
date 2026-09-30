"""Measure this machine and your data host, then build a config from the result."""

from ctprotocol import CTProtocolDataset, recommend, run_benchmark

URL = "https://example.com/train-{000..015}.jsonl.gz"

report = run_benchmark(url=URL.replace("{000..015}", "000"))
print(report.format())

# 1. Short trial run with defaults to learn how fast *your* loop consumes data.
trial = CTProtocolDataset(URL)
for i, _ in enumerate(trial):
    ...  # your real training step
    if i == 5_000:
        break
trial.close()

# 2. Plan from measurements: storage tier, lookahead, and cache size.
plan = recommend(report, consume_mb_s=trial.stats.consume_mb_s, ahead_seconds=60)
print(plan.format())

dataset = CTProtocolDataset(URL, config=plan.to_config())
