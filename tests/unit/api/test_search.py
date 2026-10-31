"""``GET /v1/search``: retrieval without language models, validated like the chat endpoints."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from lexgraph.api.security import API_KEY_HEADER
from lexgraph.domain import LawId
from lexgraph.schemas import Source
from tests.unit.api.conftest import ApiFactory
from tests.unit.api.fakes import FakeChatService, SearchCall, sample_source

QueryParams = dict[str, str | int | list[str]]


def test_search_returns_the_ranked_passages(api: TestClient, service: FakeChatService) -> None:
    second = sample_source(2).model_copy(update={"chunk_id": "CDC:26#0", "article_id": "CDC:26"})
    service.search_results = [sample_source(), second]

    response = api.get("/v1/search", params={"q": "desistir da compra"})

    assert response.status_code == 200
    assert [Source.model_validate(item) for item in response.json()] == [sample_source(), second]
    assert service.searches == [SearchCall("desistir da compra", 5, None, None)]


def test_search_passes_its_options_to_the_service(
    api: TestClient, service: FakeChatService
) -> None:
    params: QueryParams = {"q": "  justa causa ", "k": 1, "mode": "sparse", "laws": ["CLT", "CF"]}

    response = api.get("/v1/search", params=params)

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert service.searches == [SearchCall("justa causa", 1, [LawId.CLT, LawId.CF], "sparse")]


@pytest.mark.parametrize(
    ("params", "location"),
    [
        ({}, ["query", "q"]),
        ({"q": "   "}, ["query", "q"]),
        ({"q": "x" * 2001}, ["query", "q"]),
        ({"q": "ok", "k": 0}, ["query", "k"]),
        ({"q": "ok", "k": 21}, ["query", "k"]),
        ({"q": "ok", "mode": "semantic"}, ["query", "mode"]),
        ({"q": "ok", "laws": ["CODIGO_CIVIL"]}, ["query", "laws", 0]),
        ({"q": "ok", "laws": ["CLT"] * 7}, ["query", "laws"]),
    ],
)
def test_search_rejects_invalid_parameters(
    api: TestClient,
    service: FakeChatService,
    params: QueryParams,
    location: list[object],
) -> None:
    response = api.get("/v1/search", params=params)

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [location]
    assert service.searches == []


def test_search_enforces_the_configured_question_limit(
    make_api: ApiFactory, service: FakeChatService
) -> None:
    api = make_api(max_question_chars=10)

    rejected = api.get("/v1/search", params={"q": "x" * 11})
    accepted = api.get("/v1/search", params={"q": "x" * 10})

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == [
        {
            "type": "string_too_long",
            "loc": ["query", "q"],
            "msg": "String should have at most 10 characters",
        }
    ]
    assert accepted.status_code == 200
    assert [call.query for call in service.searches] == ["x" * 10]


def test_search_requires_the_api_key_when_configured(make_api: ApiFactory) -> None:
    api = make_api(api_key=SecretStr("s3cret-key"))

    without_key = api.get("/v1/search", params={"q": "férias"})
    with_key = api.get("/v1/search", params={"q": "férias"}, headers={API_KEY_HEADER: "s3cret-key"})

    assert without_key.status_code == 401
    assert with_key.status_code == 200
