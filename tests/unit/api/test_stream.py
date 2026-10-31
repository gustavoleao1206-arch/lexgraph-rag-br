"""Wire format and failure handling of ``POST /v1/chat/stream``."""

import logging

import pytest
from pydantic import TypeAdapter

from lexgraph.api.routes import ANSWER_FAILURE_MESSAGE
from lexgraph.schemas import ChatEvent, ErrorEvent, FinalEvent, StatusEvent, TokenEvent
from lexgraph.ui.client import SseMessage, parse_sse
from tests.unit.api.conftest import ApiFactory
from tests.unit.api.fakes import FakeChatService, error_events, sample_events

EVENTS: TypeAdapter[ChatEvent] = TypeAdapter(ChatEvent)
QUESTION = {"question": "Posso desistir de uma compra online?", "thread_id": "t-1"}


def stream_messages(make_api: ApiFactory) -> list[SseMessage]:
    api = make_api()
    with api.stream("POST", "/v1/chat/stream", json=QUESTION) as response:
        assert response.status_code == 200
        return list(parse_sse(response.iter_lines()))


def test_stream_sends_every_event_then_done(make_api: ApiFactory) -> None:
    messages = stream_messages(make_api)

    assert [message.event for message in messages] == [
        "status",
        "status",
        "sources",
        "token",
        "token",
        "final",
        "done",
    ]
    events = [EVENTS.validate_json(message.data) for message in messages[:-1]]
    assert events == sample_events()
    assert messages[-1] == SseMessage(event="done", data="[DONE]")


def test_stream_uses_the_sse_wire_format(make_api: ApiFactory) -> None:
    api = make_api()

    with api.stream("POST", "/v1/chat/stream", json=QUESTION) as response:
        body = response.read().decode("utf-8")

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    first_token = TokenEvent(text="Sim. Você pode desistir ")
    assert f"event: token\ndata: {first_token.model_dump_json()}\n\n" in body
    assert body.endswith("event: done\ndata: [DONE]\n\n")


def test_stream_passes_the_request_to_the_service(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    stream_messages(make_api)

    assert len(service.requests) == 1
    assert service.requests[0].question == QUESTION["question"]
    assert service.requests[0].thread_id == "t-1"


def test_service_error_events_are_relayed(make_api: ApiFactory, service: FakeChatService) -> None:
    service.events = error_events()

    messages = stream_messages(make_api)

    assert [message.event for message in messages] == ["status", "error", "done"]
    assert EVENTS.validate_json(messages[1].data) == error_events()[1]


def test_exception_mid_stream_becomes_an_error_event(
    make_api: ApiFactory, service: FakeChatService, caplog: pytest.LogCaptureFixture
) -> None:
    service.events = [StatusEvent(stage="analyze", message="Analisando a pergunta")]
    service.failure = RuntimeError("qdrant lock held by pid 4242")

    with caplog.at_level(logging.ERROR, logger="lexgraph.api.routes"):
        messages = stream_messages(make_api)

    assert [message.event for message in messages] == ["status", "error", "done"]
    error = EVENTS.validate_json(messages[1].data)
    assert error == ErrorEvent(message=ANSWER_FAILURE_MESSAGE)
    assert "4242" not in messages[1].data
    assert "Chat stream failed" in caplog.text


def test_exception_after_the_final_event_keeps_the_final_answer(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    service.failure = RuntimeError("checkpoint write failed")

    messages = stream_messages(make_api)

    assert [message.event for message in messages][-2:] == ["final", "done"]
    assert isinstance(EVENTS.validate_json(messages[-2].data), FinalEvent)


def test_stream_without_final_event_is_completed_with_an_error(
    make_api: ApiFactory, service: FakeChatService, caplog: pytest.LogCaptureFixture
) -> None:
    service.events = [TokenEvent(text="Sim")]

    with caplog.at_level(logging.ERROR, logger="lexgraph.api.routes"):
        messages = stream_messages(make_api)

    assert [message.event for message in messages] == ["token", "error", "done"]
    assert "ended without a final event" in caplog.text


def test_stream_validates_the_question_before_streaming(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    api = make_api(max_question_chars=10)

    response = api.post("/v1/chat/stream", json={"question": "x" * 11})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert service.requests == []
