"""Endpoints of the HTTP API, served with a scripted chat service and a small corpus."""

import logging

import pytest
from fastapi.testclient import TestClient

from lexgraph.api.routes import ANSWER_FAILURE_MESSAGE
from lexgraph.domain import LawId
from lexgraph.protocols import ChatServiceError
from lexgraph.schemas import ChatRequest, ChatResponse, HistoryMessage, LawSummary, ServiceInfo
from tests.unit.api.conftest import ApiFactory
from tests.unit.api.fakes import SERVICE_INFO, FakeChatService, sample_response


def test_health_reports_ok(api: TestClient) -> None:
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_returns_the_service_report(api: TestClient) -> None:
    response = api.get("/v1/info")

    assert response.status_code == 200
    assert ServiceInfo.model_validate(response.json()) == SERVICE_INFO


def test_chat_returns_the_complete_response(api: TestClient, service: FakeChatService) -> None:
    payload = {"question": "Posso desistir?", "thread_id": "abc_123", "laws": ["CDC"]}

    response = api.post("/v1/chat", json=payload)

    assert response.status_code == 200
    assert ChatResponse.model_validate(response.json()) == sample_response()
    assert service.requests == [
        ChatRequest(question="Posso desistir?", thread_id="abc_123", laws=[LawId.CDC])
    ]


@pytest.mark.parametrize(
    ("payload", "location"),
    [
        ({"question": ""}, ["body", "question"]),
        ({"question": " \n\t "}, ["body", "question"]),
        ({"question": "x" * 2001}, ["body", "question"]),
        ({"question": "x" * 8001}, ["body", "question"]),
        ({"question": "ok", "thread_id": "has space"}, ["body", "thread_id"]),
        ({"question": "ok", "laws": ["CODIGO_CIVIL"]}, ["body", "laws", 0]),
        ({"question": "ok", "laws": ["CLT"] * 7}, ["body", "laws"]),
        ({"thread_id": "abc"}, ["body", "question"]),
    ],
)
def test_chat_rejects_invalid_requests(
    api: TestClient, service: FakeChatService, payload: dict[str, object], location: list[object]
) -> None:
    response = api.post("/v1/chat", json=payload)

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [location]
    assert service.requests == []


def test_chat_enforces_the_configured_question_limit(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    api = make_api(max_question_chars=20)

    response = api.post("/v1/chat", json={"question": "x" * 21})

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "string_too_long",
                "loc": ["body", "question"],
                "msg": "String should have at most 20 characters",
            }
        ]
    }
    assert service.requests == []


def test_chat_accepts_a_question_at_the_limit(make_api: ApiFactory) -> None:
    api = make_api(max_question_chars=20)

    assert api.post("/v1/chat", json={"question": "x" * 20}).status_code == 200


def test_chat_accepts_longer_questions_when_the_limit_is_raised(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    api = make_api(max_question_chars=3000)

    response = api.post("/v1/chat", json={"question": "x" * 2500})

    assert response.status_code == 200
    assert [len(request.question) for request in service.requests] == [2500]


def test_chat_strips_the_question_and_dedupes_the_laws(
    api: TestClient, service: FakeChatService
) -> None:
    payload = {"question": "  Posso desistir?\n", "laws": ["CDC", "CLT", "CDC"]}

    assert api.post("/v1/chat", json=payload).status_code == 200
    assert service.requests == [
        ChatRequest(question="Posso desistir?", laws=[LawId.CDC, LawId.CLT])
    ]


@pytest.mark.parametrize("endpoint", ["/v1/chat", "/v1/chat/stream"])
def test_rerank_is_rejected_when_the_server_has_no_reranker(
    api: TestClient, service: FakeChatService, endpoint: str
) -> None:
    service.info_report = SERVICE_INFO.model_copy(update={"rerank_available": False})

    response = api.post(endpoint, json={"question": "Posso desistir?", "rerank": True})

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "rerank_unavailable",
                "loc": ["body", "rerank"],
                "msg": "Reranking is not available on this server",
            }
        ]
    }
    assert service.requests == []


@pytest.mark.parametrize(("rerank", "available"), [(True, True), (False, False), (None, False)])
def test_rerank_options_the_server_can_honor_are_accepted(
    api: TestClient, service: FakeChatService, rerank: bool | None, available: bool
) -> None:
    service.info_report = SERVICE_INFO.model_copy(update={"rerank_available": available})

    response = api.post("/v1/chat", json={"question": "Posso desistir?", "rerank": rerank})

    assert response.status_code == 200
    assert [request.rerank for request in service.requests] == [rerank]


def test_validation_errors_do_not_echo_the_input(api: TestClient) -> None:
    secret_question = "minha senha é hunter2 " * 100

    response = api.post("/v1/chat", json={"question": secret_question})

    assert response.status_code == 422
    assert "hunter2" not in response.text
    assert set(response.json()["detail"][0]) == {"type", "loc", "msg"}


def test_chat_failures_return_503_with_a_safe_message(
    api: TestClient, service: FakeChatService, caplog: pytest.LogCaptureFixture
) -> None:
    service.failure = RuntimeError("database password is swordfish")

    with caplog.at_level(logging.ERROR, logger="lexgraph.api.routes"):
        response = api.post("/v1/chat", json={"question": "Posso desistir?", "thread_id": "t-9"})

    assert response.status_code == 503
    assert response.json() == {"detail": ANSWER_FAILURE_MESSAGE}
    assert "swordfish" not in response.text
    assert "Chat request failed (thread t-9)" in caplog.text


def test_unexpected_errors_return_a_generic_500(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, service: FakeChatService
) -> None:
    async def broken_info() -> ServiceInfo:
        msg = "qdrant at 10.0.0.7 refused the connection"
        raise RuntimeError(msg)

    monkeypatch.setattr(service, "info", broken_info)

    response = api.get("/v1/info")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "10.0.0.7" not in response.text


def test_thread_history_is_returned_in_order(api: TestClient, service: FakeChatService) -> None:
    messages = [
        HistoryMessage(role="user", content="Posso desistir?"),
        HistoryMessage(role="assistant", content="Sim [1]."),
    ]
    service.histories["abc-123"] = messages

    response = api.get("/v1/threads/abc-123")

    assert response.status_code == 200
    assert [HistoryMessage.model_validate(item) for item in response.json()] == messages


def test_unknown_thread_has_an_empty_history(api: TestClient) -> None:
    response = api.get("/v1/threads/unknown")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("thread_id", ["bad.id", "x" * 65, "não"])
def test_thread_ids_are_validated(api: TestClient, thread_id: str) -> None:
    assert api.get(f"/v1/threads/{thread_id}").status_code == 422


def test_laws_are_listed_in_catalog_order_with_counts(api: TestClient) -> None:
    response = api.get("/v1/laws")

    assert response.status_code == 200
    laws = [LawSummary.model_validate(item) for item in response.json()]
    assert [law.law_id for law in laws] == [LawId.CLT, LawId.CDC, LawId.LGPD]
    clt = laws[0]
    assert (clt.short_name, clt.name) == ("CLT", "Consolidação das Leis do Trabalho")
    assert (clt.articles, clt.in_force, clt.revoked, clt.vetoed) == (2, 1, 1, 0)
    assert clt.official_name == "Decreto-Lei nº 5.452, de 1º de maio de 1943"
    assert clt.source_url == "https://www.planalto.gov.br/clt.htm"
    assert clt.fetched_at == "2026-10-30T11:58:00+00:00"


def test_article_lookup_returns_the_article(api: TestClient) -> None:
    response = api.get("/v1/articles/CDC:49")

    assert response.status_code == 200
    body = response.json()
    assert body["article_id"] == "CDC:49"
    assert body["status"] == "vigente"
    assert body["text"].startswith("Art. 49.")


def test_revoked_articles_are_still_available(api: TestClient) -> None:
    response = api.get("/v1/articles/CLT:59-A")

    assert response.status_code == 200
    assert response.json()["status"] == "revogado"


def test_unknown_article_is_404(api: TestClient) -> None:
    response = api.get("/v1/articles/CLT:9999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Article CLT:9999 not found."}


@pytest.mark.parametrize("article_id", ["clt:482", "CLT-482", "XYZ:1", "CLT:12abc"])
def test_malformed_article_ids_are_rejected(api: TestClient, article_id: str) -> None:
    assert api.get(f"/v1/articles/{article_id}").status_code == 422


@pytest.mark.parametrize(
    "message",
    [
        "Nenhum modelo de linguagem está configurado.",
        "Não consegui redigir a resposta agora porque o serviço de linguagem está indisponível.",
    ],
)
def test_chat_service_errors_return_503_with_their_own_message(
    api: TestClient, service: FakeChatService, message: str
) -> None:
    service.failure = ChatServiceError(message)

    response = api.post("/v1/chat", json={"question": "Posso desistir?"})

    assert response.status_code == 503
    assert response.json() == {"detail": message}
