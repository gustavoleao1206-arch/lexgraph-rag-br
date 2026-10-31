"""Downloads with conditional requests and retries, against a mocked HTTP transport."""

import hashlib
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from lexgraph.catalog import SOURCES
from lexgraph.ingestion import fetch
from lexgraph.ingestion.fetch import (
    USER_AGENT,
    build_client,
    fetch_all,
    fetch_page,
    html_path,
    meta_path,
    read_meta,
)

SOURCE = SOURCES["marco-civil"]
PAGE = "<p>Art. 1º Esta Lei estabelece princípios.</p>".encode("cp1252")
ETAG = '"1ce91-650291064ef51"'
LAST_MODIFIED = "Thu, 23 Apr 2026 23:32:39 GMT"

Handler = Callable[[httpx.Request], httpx.Response]


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PAGE, headers={"ETag": ETAG, "Last-Modified": LAST_MODIFIED})


class Server:
    """Mock transport that answers with the given handlers in turn and records requests."""

    def __init__(self, *handlers: Handler) -> None:
        self.handlers = list(handlers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handlers.pop(0)(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def fetch_with(server: Server, raw_dir: Path, *, force: bool = False) -> fetch.FetchResult:
    with server.client() as client:
        return fetch_page(SOURCE, raw_dir, client=client, force=force, retry_backoff=0)


def test_download_stores_the_raw_bytes_and_their_provenance(tmp_path: Path) -> None:
    server = Server(ok)

    result = fetch_with(server, tmp_path)

    assert not result.not_modified
    assert result.path == html_path(tmp_path, "marco-civil")
    assert result.path.read_bytes() == PAGE
    meta = read_meta(tmp_path, "marco-civil")
    assert meta == result.meta
    assert (meta.url, meta.final_url, meta.status) == (SOURCE.url, SOURCE.url, 200)
    assert meta.sha256 == hashlib.sha256(PAGE).hexdigest()
    assert (meta.etag, meta.last_modified) == (ETAG, LAST_MODIFIED)
    assert datetime.fromisoformat(meta.fetched_at).utcoffset() is not None
    assert "if-none-match" not in server.requests[0].headers


def test_not_modified_keeps_the_stored_copy(tmp_path: Path) -> None:
    first = fetch_with(Server(ok), tmp_path)
    server = Server(lambda _: httpx.Response(304))

    second = fetch_with(server, tmp_path)

    assert second.not_modified
    assert server.requests[0].headers["if-none-match"] == ETAG
    assert server.requests[0].headers["if-modified-since"] == LAST_MODIFIED
    assert html_path(tmp_path, "marco-civil").read_bytes() == PAGE
    assert second.meta.sha256 == first.meta.sha256
    assert second.meta.fetched_at >= first.meta.fetched_at
    assert read_meta(tmp_path, "marco-civil") == second.meta


def test_force_downloads_without_validators(tmp_path: Path) -> None:
    fetch_with(Server(ok), tmp_path)
    changed = b"<p>Art. 1. Nova reda\xe7\xe3o.</p>"
    server = Server(lambda _: httpx.Response(200, content=changed))

    result = fetch_with(server, tmp_path, force=True)

    assert "if-none-match" not in server.requests[0].headers
    assert result.path.read_bytes() == changed
    assert result.meta.etag is None


def test_unreadable_metadata_triggers_a_full_download(tmp_path: Path) -> None:
    html_path(tmp_path, "marco-civil").write_bytes(PAGE)
    meta_path(tmp_path, "marco-civil").write_text("{not json", encoding="utf-8")
    server = Server(ok)

    fetch_with(server, tmp_path)

    assert "if-none-match" not in server.requests[0].headers
    assert read_meta(tmp_path, "marco-civil").etag == ETAG


def test_page_without_metadata_is_downloaded_again(tmp_path: Path) -> None:
    html_path(tmp_path, "marco-civil").write_bytes(b"<p>stale copy</p>")
    server = Server(ok)

    result = fetch_with(server, tmp_path)

    assert "if-modified-since" not in server.requests[0].headers
    assert result.path.read_bytes() == PAGE


def test_only_the_validators_the_server_gave_are_sent(tmp_path: Path) -> None:
    fetch_with(Server(lambda _: httpx.Response(200, content=PAGE)), tmp_path)
    no_validators = Server(ok)
    fetch_with(no_validators, tmp_path)
    server = Server(lambda _: httpx.Response(304))

    fetch_with(server, tmp_path)

    assert "if-none-match" not in no_validators.requests[0].headers
    assert "if-modified-since" not in no_validators.requests[0].headers
    assert server.requests[0].headers["if-none-match"] == ETAG
    assert server.requests[0].headers["if-modified-since"] == LAST_MODIFIED


def test_transient_failures_are_retried(tmp_path: Path) -> None:
    def disconnect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset", request=request)

    server = Server(lambda _: httpx.Response(503), disconnect, ok)

    result = fetch_with(server, tmp_path)

    assert len(server.requests) == 3
    assert result.path.read_bytes() == PAGE


def test_persistent_server_errors_give_up_without_writing(tmp_path: Path) -> None:
    server = Server(*[lambda _: httpx.Response(502)] * 3)

    with pytest.raises(httpx.HTTPStatusError):
        fetch_with(server, tmp_path)

    assert len(server.requests) == 3
    assert not html_path(tmp_path, "marco-civil").exists()


def test_persistent_transport_errors_are_raised(tmp_path: Path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    server = Server(timeout, timeout, timeout)

    with pytest.raises(httpx.ReadTimeout):
        fetch_with(server, tmp_path)

    assert len(server.requests) == 3


def test_client_errors_are_not_retried_and_keep_the_stored_copy(tmp_path: Path) -> None:
    fetch_with(Server(ok), tmp_path)
    server = Server(lambda _: httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_with(server, tmp_path, force=True)

    assert len(server.requests) == 1
    assert html_path(tmp_path, "marco-civil").read_bytes() == PAGE


def test_fetch_page_builds_its_own_client_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = Server(ok)
    monkeypatch.setattr(fetch, "build_client", server.client)

    result = fetch_page(SOURCE, tmp_path)

    assert result.path.read_bytes() == PAGE


def test_fetch_all_builds_its_own_client_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = Server(ok)
    monkeypatch.setattr(fetch, "build_client", server.client)

    (result,) = fetch_all([SOURCE], tmp_path)

    assert result.slug == "marco-civil"
    assert len(server.requests) == 1


def test_fetch_all_waits_between_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    server = Server(ok, ok, ok)
    sources = [SOURCES["marco-civil"], SOURCES["lgpd"], SOURCES["cdc"]]

    with server.client() as client:
        results = fetch_all(sources, tmp_path, client=client)

    assert [result.slug for result in results] == ["marco-civil", "lgpd", "cdc"]
    assert [str(request.url) for request in server.requests] == [source.url for source in sources]
    assert pauses == [fetch.PAGE_DELAY_S, fetch.PAGE_DELAY_S]


def test_default_client_looks_like_a_browser() -> None:
    with build_client() as client:
        assert client.headers["User-Agent"] == USER_AGENT
        assert client.headers["Accept-Language"].startswith("pt-BR")
        assert client.timeout.read == fetch.TIMEOUT_S
        assert client.follow_redirects
