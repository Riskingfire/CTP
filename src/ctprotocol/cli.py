"""Command line interface: ``ctp benchmark | stream | info | clean``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .benchmark import recommend, run_benchmark
from .cache import sweep_stale
from .config import CTProtocolConfig, MiB, default_cache_dir
from .dataset import CTProtocolDataset
from .errors import CTProtocolError
from .formats import FORMATS, detect_format
from .sources import resolve_sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctp",
        description="CTProtocol - Caching Training Protocol: stream training data without keeping it.",
    )
    parser.add_argument("--version", action="version", version=f"ctp {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v: info, -vv: debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="measure disk/CPU/GPU/network and recommend settings")
    bench.add_argument("--url", help="dataset URL to probe network throughput against")
    bench.add_argument("--cache-dir", help=f"directory to test (default: {default_cache_dir()})")
    bench.add_argument(
        "--consume-mb-s",
        type=float,
        help="how fast your training loop consumes data (MiB/s); improves the recommendation",
    )
    bench.add_argument("--ahead", type=float, default=60.0, help="desired lookahead in seconds (default: 60)")
    bench.add_argument("--json", action="store_true", help="machine-readable output")

    stream = sub.add_parser("stream", help="stream records from sources and print them as JSON lines")
    stream.add_argument("sources", nargs="+", help="URL, path, hf:// spec, glob or {000..015} range")
    stream.add_argument("--format", choices=FORMATS, help="record format (default: from file name)")
    stream.add_argument(
        "--limit", type=int, default=5, help="stop after N records (0 = no limit, default: 5)"
    )
    stream.add_argument("--ahead", type=float, default=60.0)
    stream.add_argument("--max-cache-mb", type=int, default=1024)
    stream.add_argument("--storage", choices=("auto", "memory", "disk"), default="auto")
    stream.add_argument("--cache-dir")
    stream.add_argument(
        "--skip-errors", action="store_true", help="skip malformed records instead of failing"
    )
    stream.add_argument("--text-field", help="print only this field of each record")
    stream.add_argument("--stats", action="store_true", help="print stream statistics to stderr when done")

    info = sub.add_parser("info", help="show size, range support and detected format of sources")
    info.add_argument("sources", nargs="+")

    clean = sub.add_parser("clean", help="remove session directories left behind by crashed runs")
    clean.add_argument("--cache-dir")
    return parser


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark(args.cache_dir, url=args.url)
    plan = recommend(report, consume_mb_s=args.consume_mb_s, ahead_seconds=args.ahead)
    if args.json:
        json.dump({"report": report.as_dict(), "plan": plan.__dict__}, sys.stdout, indent=2)
        print()
    else:
        print(report.format())
        print()
        print(plan.format())
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    config = CTProtocolConfig(
        cache_dir=args.cache_dir,
        storage=args.storage,
        ahead_seconds=args.ahead,
        max_cache_mb=args.max_cache_mb,
        on_error="skip" if args.skip_errors else "raise",
    )
    dataset = CTProtocolDataset(args.sources, format=args.format, config=config, text_field=args.text_field)
    count = 0
    try:
        with dataset:
            for record in dataset:
                if isinstance(record, (bytes, bytearray)):
                    record = f"<{len(record)} bytes>"
                print(
                    record if isinstance(record, str) else json.dumps(record, ensure_ascii=False, default=str)
                )
                count += 1
                if args.limit and count >= args.limit:
                    break
    except BrokenPipeError:  # e.g. `ctp stream ... | head`
        sys.stderr.close()
        return 0
    finally:
        if args.stats:
            print(dataset.stats.summary(), file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    config = CTProtocolConfig()
    for source in resolve_sources(args.sources):
        size = source.size(config)
        rng = source.supports_range(config)
        try:
            fmt: Any = detect_format(source.name)
        except CTProtocolError:
            fmt = "unknown (pass --format)"
        print(source.name)
        print(f"  size   : {f'{size / MiB:,.1f} MiB' if size is not None else 'unknown'}")
        print(f"  ranges : {'yes (resumable)' if rng else 'no' if rng is False else 'unknown'}")
        print(f"  format : {fmt}")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    base = args.cache_dir or default_cache_dir()
    removed = sweep_stale(base)
    for path in removed:
        print(f"removed {path}")
    print(f"{len(removed)} stale session(s) removed from {base}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    handlers = {"benchmark": _cmd_benchmark, "stream": _cmd_stream, "info": _cmd_info, "clean": _cmd_clean}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130
    except CTProtocolError as exc:
        print(f"ctp: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
