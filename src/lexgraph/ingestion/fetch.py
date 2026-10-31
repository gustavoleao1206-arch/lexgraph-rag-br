"""Download the official law pages politely and record where every byte came from.

Each page is stored as ``<slug>.html`` (raw bytes, exactly as served) next to
``<slug>.meta.json`` (:class:`FetchMeta`). When both files exist, the next download is a
conditional request (``If-None-Match`` / ``If-Modified-Since``): a ``304 Not Modified`` keeps
the stored copy and only refreshes ``fetched_at``.
"""

import hashlib
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

from lexgraph.catalog import SourcePage

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}
TIMEOUT_S = 60.0
PAGE_DELAY_S = 3.0
MAX_RETRIES = 2
RETRY_BACKOFF_S = 2.0
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class FetchMeta(BaseModel):
    """Provenance of a stored page, saved as ``<slug>.meta.json``."""

    url: str = Field(description="URL requested (the catalog URL).")
    final_url: str = Field(description="URL that served the page, after redirects.")
    status: int = Field(description="HTTP status of the response that produced the stored bytes.")
    sha256: str = Field(description="SHA-256 of the stored bytes.")
    etag: str | None = None
    last_modified: str | None = Field(default=None, description="Last-Modified header, verbatim.")
    fetched_at: str = Field(
        description="UTC ISO-8601 time of the last download or 304 revalidation."
    )


class FetchResult(BaseModel):
    """Outcome of :func:`fetch_page`."""

    slug: str
    path: Path
    meta: FetchMeta
    not_modified: bool = Field(description="True when a 304 answer kept the stored copy.")


def html_path(raw_dir: Path, slug: str) -> Path:
    """Path of the raw page of source ``slug``."""
    return raw_dir / f"{slug}.html"


def meta_path(raw_dir: Path, slug: str) -> Path:
    """Path of the provenance file of source ``slug``."""
    return raw_dir / f"{slug}.meta.json"


def read_meta(raw_dir: Path, slug: str) -> FetchMeta:
    """Read the provenance of a stored page; raises ``FileNotFoundError`` when absent."""
    return FetchMeta.model_validate_json(meta_path(raw_dir, slug).read_text(encoding="utf-8"))


def build_client() -> httpx.Client:
    """HTTP client with browser-like headers, a 60 s timeout and redirect following."""
    return httpx.Client(headers=DEFAULT_HEADERS, timeout=TIMEOUT_S, follow_redirects=True)


def fetch_page(
    source: SourcePage,
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
    retry_backoff: float = RETRY_BACKOFF_S,
) -> FetchResult:
    """Download one page into ``raw_dir``, conditionally unless ``force`` is set.

    Transport errors and transient statuses (429, 5xx) are retried ``MAX_RETRIES`` times with
    a linear backoff; any other non-success status raises ``httpx.HTTPStatusError`` and leaves
    the stored files untouched.
    """
    if client is None:
        with build_client() as own_client:
            return fetch_page(
                source, raw_dir, client=own_client, force=force, retry_backoff=retry_backoff
            )
    page = html_path(raw_dir, source.slug)
    stored = None if force else _stored_meta(raw_dir, source.slug)
    response = _get(client, source.url, _conditional_headers(stored), retry_backoff)
    if response.status_code == httpx.codes.NOT_MODIFIED and stored is not None:
        meta = stored.model_copy(
            update={
                "etag": response.headers.get("etag", stored.etag),
                "last_modified": response.headers.get("last-modified", stored.last_modified),
                "fetched_at": _utc_now(),
            }
        )
        not_modified = True
    else:
        response.raise_for_status()
        content = response.content
        raw_dir.mkdir(parents=True, exist_ok=True)
        _write_atomically(page, content)
        meta = FetchMeta(
            url=source.url,
            final_url=str(response.url),
            status=response.status_code,
            sha256=hashlib.sha256(content).hexdigest(),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            fetched_at=_utc_now(),
        )
        not_modified = False
    _write_atomically(
        meta_path(raw_dir, source.slug), (meta.model_dump_json(indent=2) + "\n").encode("utf-8")
    )
    logger.info("%s: %s", source.slug, "not modified" if not_modified else "downloaded")
    return FetchResult(slug=source.slug, path=page, meta=meta, not_modified=not_modified)


def fetch_all(
    sources: Sequence[SourcePage],
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
    delay: float = PAGE_DELAY_S,
) -> list[FetchResult]:
    """Fetch ``sources`` one after the other, waiting ``delay`` seconds between pages."""
    if client is None:
        with build_client() as own_client:
            return fetch_all(sources, raw_dir, client=own_client, force=force, delay=delay)
    results: list[FetchResult] = []
    for index, source in enumerate(sources):
        if index:
            time.sleep(delay)
        results.append(fetch_page(source, raw_dir, client=client, force=force))
    return results


def _stored_meta(raw_dir: Path, slug: str) -> FetchMeta | None:
    """Provenance of the stored copy, or None when there is no usable copy."""
    if not html_path(raw_dir, slug).is_file():
        return None
    try:
        return read_meta(raw_dir, slug)
    except FileNotFoundError:
        return None
    except ValidationError:
        logger.warning("%s: unreadable %s, downloading again", slug, meta_path(raw_dir, slug))
        return None


def _conditional_headers(stored: FetchMeta | None) -> dict[str, str]:
    """Validators of the stored copy, so the server can answer 304."""
    if stored is None:
        return {}
    headers: dict[str, str] = {}
    if stored.etag:
        headers["If-None-Match"] = stored.etag
    if stored.last_modified:
        headers["If-Modified-Since"] = stored.last_modified
    return headers


def _get(client: httpx.Client, url: str, headers: dict[str, str], backoff: float) -> httpx.Response:
    """GET with retries on transport errors and transient statuses."""
    attempt = 0
    while True:
        try:
            response = client.get(url, headers=headers)
        except httpx.TransportError as exc:
            if attempt >= MAX_RETRIES:
                raise
            logger.warning("GET %s failed (%s), retrying", url, exc)
        else:
            if response.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_RETRIES:
                return response
            logger.warning("GET %s returned %d, retrying", url, response.status_code)
        attempt += 1
        time.sleep(backoff * attempt)


def _write_atomically(path: Path, content: bytes) -> None:
    """Write through a temporary file so an interrupted run never leaves a partial file."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
