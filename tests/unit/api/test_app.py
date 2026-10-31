"""Application wiring: lifespan, corpus, API key, body limit, Host and CORS checks, OpenAPI."""

import json
import sys
import types
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.types import Message, Receive, Scope, Send

from lexgraph import __version__
from lexgraph.api import app as app_module
from lexgraph.api.app import create_app, load_corpus
from lexgraph.api.security import API_KEY_HEADER, MAX_BODY_BYTES, RequestGuardMiddleware
from lexgraph.config import Settings
from lexgraph.domain import Article, LawId
from lexgraph.schemas import QUESTION_MAX_CHARS
from tests.unit.api.conftest import BASE_URL, ApiFactory, write_sample_corpus
from tests.unit.api.fakes import FakeChatService

JSON_HEADERS = {"Content-Type": "application/json"}
OVERSIZED_BODY = b"x" * (MAX_BODY_BYTES + 1)


class ServiceOpener:
    """Stand-in for ``lexgraph.container.open_service`` that records its lifecycle."""

    def __init__(self, service: FakeChatService) -> None:
        self.service = service
        self.events: list[str] = []
        self.settings: list[Settings] = []

    @asynccontextmanager
    async def __call__(self, settings: Settings) -> AsyncIterator[FakeChatService]:
        self.settings.append(settings)
        self.events.append("open")
        try:
            yield self.service
        finally:
            self.events.append("close")


@pytest.fixture
def opener(monkeypatch: pytest.MonkeyPatch, service: FakeChatService) -> ServiceOpener:
    """Replace the composition root of the agent stack and the ``.env`` loading."""
    opener = ServiceOpener(service)
    container = types.ModuleType("lexgraph.container")
    container.open_service = opener  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lexgraph.container", container)
    monkeypatch.setattr(app_module, "bootstrap_env", lambda: opener.events.append("env"))
    return opener


def test_lifespan_opens_and_closes_the_real_service(
    settings: Settings, opener: ServiceOpener
) -> None:
    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        assert opener.events == ["env", "open"]
        assert client.get("/v1/info").status_code == 200

    assert opener.events == ["env", "open", "close"]
    assert opener.settings == [settings]


def test_injected_service_is_used_as_is(settings: Settings, opener: ServiceOpener) -> None:
    with TestClient(create_app(settings, service=FakeChatService()), base_url=BASE_URL) as client:
        assert client.get("/health").status_code == 200

    assert opener.events == []


def test_corpus_is_loaded_from_the_settings_folder(
    settings: Settings, sample_articles: list[Article]
) -> None:
    write_sample_corpus(settings.corpus_dir, sample_articles)

    with TestClient(create_app(settings, service=FakeChatService()), base_url=BASE_URL) as client:
        laws = client.get("/v1/laws").json()
        article = client.get("/v1/articles/LGPD:18")

    assert [law["law_id"] for law in laws] == ["CLT", "CDC", "LGPD"]
    assert article.status_code == 200


def test_missing_corpus_is_tolerated(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with TestClient(create_app(settings, service=FakeChatService()), base_url=BASE_URL) as client:
        laws = client.get("/v1/laws")
        article = client.get("/v1/articles/CDC:49")

    assert laws.json() == []
    assert article.status_code == 404
    assert "No corpus" in caplog.text


def test_load_corpus_returns_none_without_a_manifest(tmp_path: Path) -> None:
    assert load_corpus(tmp_path / "missing") is None


def test_create_app_defaults_to_the_process_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    custom = settings.model_copy(update={"api_key": SecretStr("from-env")})
    monkeypatch.setattr(app_module, "get_settings", lambda: custom)

    with TestClient(create_app(service=FakeChatService()), base_url=BASE_URL) as client:
        assert client.get("/v1/info").status_code == 401


@pytest.mark.parametrize("path", ["/v1/info", "/v1/laws", "/v1/threads/abc", "/v1/search?q=x"])
@pytest.mark.parametrize("headers", [{}, {API_KEY_HEADER: "wrong"}, {API_KEY_HEADER: ""}])
def test_api_key_is_required_on_v1_when_configured(
    make_api: ApiFactory, path: str, headers: dict[str, str]
) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))

    response = api.get(path, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "APIKey"
    assert response.json() == {"detail": "Missing or invalid API key."}


def test_valid_api_key_is_accepted(make_api: ApiFactory) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))

    assert api.get("/v1/info", headers={API_KEY_HEADER: "s3cret-key"}).status_code == 200
    stream = api.post(
        "/v1/chat/stream", json={"question": "Oi"}, headers={API_KEY_HEADER: "s3cret-key"}
    )
    assert stream.status_code == 200


@pytest.mark.parametrize("body", [b"{not json", OVERSIZED_BODY])
def test_api_key_is_checked_before_the_body_is_read(
    make_api: ApiFactory, service: FakeChatService, body: bytes
) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))

    response = api.post("/v1/chat", content=body, headers=JSON_HEADERS)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "APIKey"
    assert response.json() == {"detail": "Missing or invalid API key."}
    assert service.requests == []


def test_health_and_docs_stay_public_with_an_api_key(make_api: ApiFactory) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))

    assert api.get("/health").status_code == 200
    assert api.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("api_key", [None, SecretStr("")])
def test_api_is_open_without_a_key(make_api: ApiFactory, api_key: SecretStr | None) -> None:
    api = make_api(api_key=api_key)

    assert api.get("/v1/info").status_code == 200
    assert "securitySchemes" not in api.get("/openapi.json").json().get("components", {})


def test_openapi_documents_the_api_key_scheme(make_api: ApiFactory) -> None:
    schema = make_api(api_key=SecretStr("s3cret-key")).get("/openapi.json").json()

    scheme = schema["components"]["securitySchemes"]["ApiKey"]
    assert scheme == {
        "type": "apiKey",
        "in": "header",
        "name": API_KEY_HEADER,
        "description": "Required when the server sets `LEXGRAPH_API_KEY`.",
    }
    assert schema["paths"]["/v1/info"]["get"]["security"] == [{"ApiKey": []}]
    assert "security" not in schema["paths"]["/health"]["get"]


@pytest.mark.parametrize("path", ["/v1/chat", "/v1/chat/stream"])
@pytest.mark.parametrize("api_key", [None, SecretStr("s3cret-key")])
def test_oversized_bodies_are_rejected(
    make_api: ApiFactory, service: FakeChatService, path: str, api_key: SecretStr | None
) -> None:
    api = make_api(api_key=api_key)

    response = api.post(
        path, content=OVERSIZED_BODY, headers={**JSON_HEADERS, API_KEY_HEADER: "s3cret-key"}
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}
    assert service.requests == []


def test_oversized_bodies_without_content_length_are_rejected(
    api: TestClient, service: FakeChatService
) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"question": "'
        yield OVERSIZED_BODY
        yield b'"}'

    response = api.post("/v1/chat", content=chunks(), headers=JSON_HEADERS)

    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}
    assert service.requests == []


def test_the_longest_valid_question_fits_in_the_body_limit(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    question = "\U0001f600" * QUESTION_MAX_CHARS
    payload = {"question": question, "thread_id": "t" * 64, "laws": list(LawId), "rerank": False}
    # ASCII JSON escapes each character outside the BMP as a surrogate pair: 12 bytes.
    body = json.dumps(payload, indent=2).encode()
    api = make_api(max_question_chars=QUESTION_MAX_CHARS)

    response = api.post("/v1/chat", content=body, headers=JSON_HEADERS)

    assert 12 * QUESTION_MAX_CHARS < len(body) <= MAX_BODY_BYTES
    assert response.status_code == 200
    assert service.requests[0].question == question


async def test_streamed_bodies_are_measured_across_chunks() -> None:
    chunks = [b"12345", b"67890", b"!"]
    read: list[bytes] = []

    async def app(_scope: Scope, receive: Receive, _send: Send) -> None:
        while True:
            message = await receive()
            read.append(message["body"])
            if not message["more_body"]:
                return

    async def receive() -> Message:
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    async def send(message: Message) -> None:
        raise AssertionError(message)

    guard = RequestGuardMiddleware(app, api_key=None, protected_prefix="/v1", max_body_bytes=10)
    scope = {"type": "http", "method": "POST", "path": "/v1/chat", "headers": []}

    with pytest.raises(HTTPException) as raised:
        await guard(scope, receive, send)

    assert raised.value.status_code == 413
    assert read == [b"12345", b"67890"]


@pytest.mark.parametrize(
    ("root_path", "path"), [("", "/v1/chat"), ("/lexgraph", "/lexgraph/v1/chat")]
)
async def test_the_api_key_is_checked_behind_a_root_path(root_path: str, path: str) -> None:
    sent: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise AssertionError("the request reached the application")

    async def receive() -> Message:
        raise AssertionError("the body was read")

    async def send(message: Message) -> None:
        sent.append(message)

    guard = RequestGuardMiddleware(app, api_key=SecretStr("s3cret-key"), protected_prefix="/v1")
    scope = {"type": "http", "method": "POST", "path": path, "root_path": root_path, "headers": []}

    await guard(scope, receive, send)

    assert sent[0]["status"] == 401


def test_openapi_metadata(api: TestClient) -> None:
    schema = api.get("/openapi.json").json()

    assert schema["info"]["title"] == "LexGraph API"
    assert schema["info"]["version"] == __version__
    assert [tag["name"] for tag in schema["tags"]] == ["chat", "corpus", "system"]
    stream = schema["paths"]["/v1/chat/stream"]["post"]
    assert "text/event-stream" in stream["responses"]["200"]["content"]
    assert set(schema["paths"]) == {
        "/health",
        "/v1/info",
        "/v1/chat",
        "/v1/chat/stream",
        "/v1/threads/{thread_id}",
        "/v1/laws",
        "/v1/articles/{article_id}",
        "/v1/search",
    }


def test_cors_allows_configured_origins(api: TestClient) -> None:
    response = api.options(
        "/v1/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{API_KEY_HEADER.lower()}",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_cors_rejects_other_origins(api: TestClient) -> None:
    preflight = api.options(
        "/v1/chat",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    simple = api.get("/health", headers={"Origin": "https://evil.example"})

    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in simple.headers


def test_cors_preflight_and_rejections_work_with_an_api_key(make_api: ApiFactory) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))
    origin = {"Origin": "http://localhost:8501"}

    preflight = api.options(
        "/v1/chat",
        headers={
            **origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{API_KEY_HEADER.lower()}",
        },
    )
    rejected = api.get("/v1/info", headers=origin)

    assert preflight.status_code == 200
    assert rejected.status_code == 401
    assert rejected.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_cors_can_be_disabled(make_api: ApiFactory) -> None:
    api = make_api(cors_origins=[])

    response = api.get("/health", headers={"Origin": "http://localhost:8501"})

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1", "[::1]:8000", "api:8000"])
def test_default_allowed_hosts_cover_local_and_compose_access(
    make_api: ApiFactory, host: str
) -> None:
    api = make_api()

    assert api.get("/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("path", ["/health", "/v1/info"])
@pytest.mark.parametrize("host", ["rebind.attacker.example", "testserver"])
def test_foreign_host_headers_are_rejected(make_api: ApiFactory, path: str, host: str) -> None:
    api = make_api()

    response = api.get(path, headers={"Host": host})

    assert response.status_code == 400
    assert response.text == "Invalid host header"


def test_allowed_hosts_can_be_opened_to_any_host(make_api: ApiFactory) -> None:
    api = make_api(allowed_hosts=["*"])

    assert api.get("/v1/info", headers={"Host": "lexgraph.example.org"}).status_code == 200
