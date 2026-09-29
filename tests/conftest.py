from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class Route:
    body: bytes
    etag: str | None = '"v1"'
    ranges: bool = True
    status: int | None = None  # force a status code (e.g. 404/503)
    fail_status_times: int = 0  # answer `status` this many times (-1 = always), then behave
    drop_after: int | None = None  # cut the connection after N body bytes ...
    drop_times: int = 0  # ... for this many requests
    requests: list[dict[str, str | None]] = field(default_factory=list)
    served_bytes: int = 0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    routes: dict[str, Route]

    def log_message(self, *args: object) -> None:  # silence
        pass

    def _route(self) -> Route | None:
        return self.routes.get(self.path.split("?")[0])

    def do_HEAD(self) -> None:
        route = self._route()
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(route.body)))
        if route.ranges:
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        route = self._route()
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        route.requests.append({"Range": self.headers.get("Range"), "If-Range": self.headers.get("If-Range")})

        if route.status is not None and route.fail_status_times != 0:
            if route.fail_status_times > 0:  # negative = fail forever
                route.fail_status_times -= 1
            self._empty(route.status)
            return

        body = route.body
        start = 0
        status = 200
        rng = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        if rng and route.ranges and (if_range is None or if_range == route.etag):
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(body):
                self._empty(416)
                return
            status = 206
        payload = body[start:]

        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        if route.etag:
            self.send_header("ETag", route.etag)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()

        drop = route.drop_after if route.drop_times > 0 else None
        if drop is not None:
            route.drop_times -= 1
        step = 64 * 1024
        sent = 0
        try:
            while sent < len(payload):
                if drop is not None and sent >= drop:
                    self.close_connection = True
                    self.connection.close()
                    return
                end = min(sent + step, len(payload))
                if drop is not None:
                    end = min(end, drop)
                self.wfile.write(payload[sent:end])
                route.served_bytes += end - sent
                sent = end
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()


class Server:
    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        handler = type("H", (_Handler,), {"routes": self.routes})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def add(self, path: str, body: bytes, **kwargs: object) -> str:
        self.routes[path] = Route(body=body, **kwargs)  # type: ignore[arg-type]
        return self.base + path

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[Server]:
    srv = Server()
    yield srv
    srv.stop()


@pytest.fixture
def fast_config(tmp_path):  # type: ignore[no-untyped-def]
    """Config with tiny chunks, no backoff waits and an isolated cache dir."""
    from ctp import CTPConfig

    def make(**overrides: object) -> CTPConfig:
        base = {
            "cache_dir": str(tmp_path / "cache"),
            "chunk_size": 64 * 1024,
            "retry_backoff": 0.0,
            "timeout": 5.0,
            "storage": "memory",
        }
        base.update(overrides)
        return CTPConfig(**base)  # type: ignore[arg-type]

    return make


def jsonl_bytes(n: int, prefix: str = "row") -> bytes:
    import json

    return "".join(json.dumps({"id": i, "text": f"{prefix}-{i}"}) + "\n" for i in range(n)).encode()


Factory = Callable[..., object]
