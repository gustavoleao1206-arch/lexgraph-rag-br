"""An in-memory LexGraph API for the interface tests, served through ``httpx.MockTransport``.

The UI tests use the real :class:`~lexgraph.ui.client.ApiClient`, so they also exercise the
HTTP calls and the SSE parsing. Event streams are encoded with FastAPI's own SSE formatter,
the one the server uses.
"""

import json
from typing import TYPE_CHECKING

import httpx
from fastapi.sse import format_sse_event

from lexgraph.domain import Article, LawId
from lexgraph.schemas import ChatEvent, ChatRequest, LawSummary, ServiceInfo
from lexgraph.ui.client import ApiClient
from tests.conftest import make_article
from tests.unit.api.fakes import SERVICE_INFO, sample_events

if TYPE_CHECKING:
    from collections.abc import Iterable

BASE_URL = "http://api.test"


def sse_body(*events: ChatEvent, done: bool = True) -> bytes:
    """Encode events exactly as ``POST /v1/chat/stream`` sends them."""
    chunks = [
        format_sse_event(data_str=event.model_dump_json(), event=event.type) for event in events
    ]
    if done:
        chunks.append(format_sse_event(data_str="[DONE]", event="done"))
    return b"".join(chunks)


def sample_laws() -> list[LawSummary]:
    """Two laws with realistic counts."""
    return [
        LawSummary(
            law_id=LawId.CLT,
            short_name="CLT",
            name="Consolidação das Leis do Trabalho",
            official_name="Decreto-Lei nº 5.452, de 1º de maio de 1943",
            urn="urn:lex:br:federal:decreto.lei:1943-05-01;5452",
            source_url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm",
            articles=1027,
            in_force=853,
            revoked=170,
            vetoed=4,
            fetched_at="2026-10-30T11:58:00+00:00",
        ),
        LawSummary(
            law_id=LawId.CDC,
            short_name="CDC",
            name="Código de Defesa do Consumidor",
            official_name="Lei nº 8.078, de 11 de setembro de 1990",
            urn="urn:lex:br:federal:lei:1990-09-11;8078",
            source_url="https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
            articles=130,
            in_force=119,
            revoked=2,
            vetoed=9,
            fetched_at="2026-10-29T09:00:00+00:00",
        ),
    ]


class FakeApi:
    """Request handler that answers like the LexGraph API and records every request."""

    def __init__(self) -> None:
        self.info: ServiceInfo = SERVICE_INFO
        self.laws: list[LawSummary] = sample_laws()
        self.articles: dict[str, Article] = {"CDC:49": make_article()}
        self.stream: bytes | Iterable[bytes] = sse_body(*sample_events())
        self.offline = False
        self.failures: dict[str, int] = {}
        self.requests: list[httpx.Request] = []

    def client(self, api_key: str | None = None) -> ApiClient:
        """An API client wired to this fake."""
        return ApiClient(BASE_URL, api_key, transport=httpx.MockTransport(self))

    def chat_requests(self) -> list[ChatRequest]:
        """Bodies sent to the streaming chat endpoint."""
        return [
            ChatRequest.model_validate_json(request.content)
            for request in self.requests
            if request.url.path == "/v1/chat/stream"
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.offline:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path in self.failures:
            return httpx.Response(self.failures[path], json={"detail": "falha simulada"})
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/info":
            return httpx.Response(200, content=self.info.model_dump_json())
        if path == "/v1/laws":
            return httpx.Response(200, content=_json_list(self.laws))
        if path.startswith("/v1/articles/"):
            article = self.articles.get(path.removeprefix("/v1/articles/"))
            if article is None:
                return httpx.Response(404, json={"detail": "Article not found."})
            return httpx.Response(200, content=article.model_dump_json())
        if path == "/v1/chat/stream":
            return httpx.Response(
                200, content=self.stream, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(404, json={"detail": "Not Found"})


def _json_list(items: list[LawSummary]) -> str:
    return json.dumps([item.model_dump(mode="json") for item in items])
