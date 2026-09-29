# Contributing

```bash
git clone https://github.com/OWNER/ctp-training && cd ctp-training
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check .   # lint + formatting
mypy                                     # strict type checking
pytest --cov=ctp                         # tests (no network needed)
```

Guidelines:

- Every behaviour change needs a test. The test suite spins up a local HTTP server
  (`tests/conftest.py`) that supports `Range`, `If-Range`, ETags and can inject dropped connections
  and 5xx responses - use it rather than mocking `requests`.
- Anything that writes to disk must go through `SessionDir` so it is cleaned up and never touches
  files CTP did not create.
- Keep the core dependency list at `requests` + `psutil`. Heavier libraries are optional extras and
  must be imported lazily with a helpful `OptionalDependencyError`.
- Add a line to `CHANGELOG.md` under *Unreleased*.
