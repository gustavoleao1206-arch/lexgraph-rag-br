"""The interface's HTTP client and its Server-Sent Events parser."""

import json

import httpx
import pytest

from lexgraph.api.security import API_KEY_HEADER as SERVER_API_KEY_HEADER
from lexgraph.domain import LawId
from lexgraph.schemas import ChatRequest, HistoryMessage, TokenEvent
from lexgraph.ui.client import (
    API_KEY_HEADER,
    ApiClient,
    ApiConnectionError,
    ApiError,
    SseMessage,
    parse_sse,
)
from tests.unit.api.fakes import SERVICE_INFO, sample_events, sample_response
from tests.unit.ui.fake_api import BASE_URL, FakeApi, sample_laws, sse_body


@pytest.fixture
def fake_api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def client(fake_api: FakeApi) -> ApiClient:
    return fake_api.client()


def client_for(handler: httpx.MockTransport) -> ApiClient:
    return ApiClient(BASE_URL, transport=handler)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["event: token", "data: {}", ""], [SseMessage("token", "{}")]),
        (["data: a", "data: b", ""], [SseMessage("message", "a\nb")]),
        ([": ping", "", "event: done", "data: [DONE]", ""], [SseMessage("done", "[DONE]")]),
        (["event:status", "data:{}", ""], [SseMessage("status", "{}")]),
        (["data:  two spaces", ""], [SseMessage("message", " two spaces")]),
        (["id: 7", "retry: 100", "data: x", ""], [SseMessage("message", "x")]),
        (["event: lonely", "", "data: y", ""], [SseMessage("message", "y")]),
        (["data", ""], [SseMessage("message", "")]),
        (["event: token", "data: {}"], []),
        (["", "", ""], []),
    ],
    ids=[
        "typed-event",
        "multi-line-data",
        "comment-ping",
        "no-space-after-colon",
        "only-first-space-stripped",
        "id-and-retry-ignored",
        "event-without-data-dropped",
        "field-without-colon",
        "unterminated-event-dropped",
        "blank-lines-only",
    ],
)
def test_parse_sse(lines: list[str], expected: list[SseMessage]) -> None:
    assert list(parse_sse(lines)) == expected


def test_parse_sse_dispatches_events_in_order() -> None:
    lines = ["event: a", "data: 1", "", "event: b", "data: 2", ""]

    assert [message.event for message in parse_sse(lines)] == ["a", "b"]


def test_client_and_server_agree_on_the_api_key_header() -> None:
    assert API_KEY_HEADER == SERVER_API_KEY_HEADER


def test_api_key_is_sent_when_configured(fake_api: FakeApi) -> None:
    fake_api.client("s3cret").info()

    assert fake_api.requests[-1].headers[API_KEY_HEADER] == "s3cret"


@pytest.mark.parametrize("api_key", [None, ""])
def test_no_api_key_header_without_a_key(fake_api: FakeApi, api_key: str | None) -> None:
    fake_api.client(api_key).info()

    assert API_KEY_HEADER not in fake_api.requests[-1].headers


def test_health(client: ApiClient, fake_api: FakeApi) -> None:
    assert client.health() is True

    fake_api.failures["/health"] = 503
    assert client.health() is False

    fake_api.offline = True
    assert client.health() is False


def test_info_laws_and_history_are_parsed(client: ApiClient) -> None:
    assert client.info() == SERVICE_INFO
    assert client.laws() == sample_laws()

    history = [HistoryMessage(role="user", content="Oi")]
    handler = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[item.model_dump() for item in history])
    )
    assert client_for(handler).history("abc") == history


def test_article_is_returned_or_none_when_missing(client: ApiClient, fake_api: FakeApi) -> None:
    article = client.article("CDC:49")

    assert article is not None
    assert article.article_id == "CDC:49"
    assert client.article("CLT:9999") is None
    assert fake_api.requests[-1].url.raw_path == b"/v1/articles/CLT:9999"


def test_article_errors_other_than_404_are_raised(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.failures["/v1/articles/CDC:49"] = 503

    with pytest.raises(ApiError) as caught:
        client.article("CDC:49")

    assert caught.value.status_code == 503


def test_chat_posts_the_request_without_empty_fields() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sample_response().model_dump_json())

    answer = client_for(httpx.MockTransport(handler)).chat(
        ChatRequest(question="Posso desistir?", laws=[LawId.CDC])
    )

    assert bodies == [{"question": "Posso desistir?", "laws": ["CDC"]}]
    assert answer == sample_response()


def test_stream_chat_yields_events_until_done(client: ApiClient, fake_api: FakeApi) -> None:
    events = list(client.stream_chat(ChatRequest(question="Posso desistir?", thread_id="t1")))

    assert events == sample_events()
    assert fake_api.chat_requests() == [ChatRequest(question="Posso desistir?", thread_id="t1")]
    assert fake_api.requests[-1].headers["accept"] == "text/event-stream"


def test_stream_chat_tolerates_pings_and_crlf(client: ApiClient, fake_api: FakeApi) -> None:
    token = TokenEvent(text="olá")
    fake_api.stream = (
        b": ping\r\n\r\nevent: token\r\ndata: "
        + token.model_dump_json().encode()
        + b"\r\n\r\nevent: done\r\ndata: [DONE]\r\n\r\n"
    )

    assert list(client.stream_chat(ChatRequest(question="Oi"))) == [token]


def test_stream_chat_ignores_anything_after_done(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(TokenEvent(text="a")) + b"event: token\ndata: not json\n\n"

    assert list(client.stream_chat(ChatRequest(question="Oi"))) == [TokenEvent(text="a")]


def test_stream_chat_fails_when_the_stream_is_cut(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(TokenEvent(text="a"), done=False)
    stream = client.stream_chat(ChatRequest(question="Oi"))

    assert next(stream) == TokenEvent(text="a")
    with pytest.raises(ApiConnectionError, match="ended before it was complete"):
        next(stream)


def test_stream_chat_reports_error_statuses(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.failures["/v1/chat/stream"] = 401

    with pytest.raises(ApiError, match="falha simulada") as caught:
        list(client.stream_chat(ChatRequest(question="Oi")))

    assert caught.value.status_code == 401
    assert not isinstance(caught.value, ApiConnectionError)


def test_stream_chat_reports_connection_failures(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.offline = True

    with pytest.raises(ApiConnectionError, match=r"api\.test"):
        list(client.stream_chat(ChatRequest(question="Oi")))


def test_stream_chat_rejects_malformed_events(client: ApiClient, fake_api: FakeApi) -> None:
    fake_api.stream = b'event: token\ndata: {"type": "token"}\n\n'

    with pytest.raises(ApiError, match="does not match its schema") as caught:
        list(client.stream_chat(ChatRequest(question="Oi")))

    assert caught.value.status_code is None


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, json={"detail": "Internal server error."}), "Internal server error"),
        (httpx.Response(502, text="<html>Bad gateway</html>"), "Bad Gateway"),
        (httpx.Response(422, json={"detail": [{"msg": "too long"}]}), "Unprocessable"),
    ],
)
def test_error_statuses_raise_api_error(response: httpx.Response, message: str) -> None:
    client = client_for(httpx.MockTransport(lambda request: response))

    with pytest.raises(ApiError, match=message) as caught:
        client.info()

    assert caught.value.status_code == response.status_code


def test_unexpected_payload_raises_api_error() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(200, json={"a": 1})))

    with pytest.raises(ApiError, match="does not match its schema"):
        client.laws()


def test_client_is_a_context_manager(fake_api: FakeApi) -> None:
    with fake_api.client() as client:
        assert client.health() is True

    with pytest.raises(RuntimeError):
        client.info()


def test_base_url_trailing_slash_is_ignored(fake_api: FakeApi) -> None:
    client = ApiClient(f"{BASE_URL}/", transport=httpx.MockTransport(fake_api))

    client.info()

    assert client.base_url == BASE_URL
    assert str(fake_api.requests[-1].url) == f"{BASE_URL}/v1/info"
