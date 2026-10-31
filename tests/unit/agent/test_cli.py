"""The ``ask`` command, driven through Typer's test runner with a fake chat service."""

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest
from typer.testing import CliRunner

from lexgraph.agent import cli
from lexgraph.config import RetrievalMode, Settings
from lexgraph.domain import LawId
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    Citation,
    ErrorEvent,
    FinalEvent,
    Grounding,
    HistoryMessage,
    ResetEvent,
    ServiceInfo,
    Source,
    StatusEvent,
    TokenEvent,
)
from lexgraph.service import ChatServiceError

URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078.htm#art49"
RESPONSE = ChatResponse(
    thread_id="t1",
    answer="O prazo é de 7 dias [1].",
    route="answered",
    sources=[
        Source(
            index=1,
            chunk_id="CDC:49#0",
            article_id="CDC:49",
            law_id=LawId.CDC,
            citation="CDC, art. 49",
            url=URL,
            text="Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias.",
            score=0.9,
        )
    ],
    citations=[Citation(index=1, article_id="CDC:49", citation="CDC, art. 49", url=URL)],
    grounding=Grounding(grounded=False, unsupported_claims=["Afirmação sem apoio."]),
)
STREAM: list[ChatEvent] = [
    StatusEvent(stage="analyze", message="Analisando a pergunta"),
    TokenEvent(text="O prazo é "),
    TokenEvent(text="de 7 dias [1]."),
    FinalEvent(response=RESPONSE),
]


class FakeService:
    """Chat service that replays fixed events and records every request."""

    def __init__(self, events: Sequence[ChatEvent] = STREAM) -> None:
        self.events = list(events)
        self.requests: list[ChatRequest] = []

    async def ask(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        for event in self.events:
            if isinstance(event, ErrorEvent):
                raise ChatServiceError(event.message)
        return RESPONSE

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event

    async def history(self, thread_id: str) -> list[HistoryMessage]:
        return []

    async def info(self) -> ServiceInfo:
        raise NotImplementedError

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[Source]:
        raise NotImplementedError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def use_service(monkeypatch: pytest.MonkeyPatch, settings: Settings, service: FakeService) -> None:
    @asynccontextmanager
    async def open_service(config: Settings) -> AsyncIterator[FakeService]:
        assert config is settings
        yield service

    monkeypatch.setattr(cli, "open_service", open_service)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "bootstrap_env", lambda: None)


def test_ask_streams_the_answer_and_its_citations(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    use_service(monkeypatch, settings, FakeService())

    result = runner.invoke(cli.app, ["Qual o prazo para desistir?"])

    assert result.exit_code == 0, result.output
    assert "· Analisando a pergunta" in result.output
    assert "O prazo é de 7 dias [1]." in result.output
    assert "Fontes citadas" in result.output
    assert f"[1] CDC, art. 49 — {URL}" in result.output
    assert "- Afirmação sem apoio." in result.output
    assert "Art. 49. O consumidor" not in result.output


def test_ask_forwards_the_options(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings, service)

    args = ["Pergunta?", "--thread", "conversa-1", "-l", "CLT", "--law", "CF", "--no-rerank"]
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    (request,) = service.requests
    assert request.question == "Pergunta?"
    assert request.thread_id == "conversa-1"
    assert request.laws == [LawId.CLT, LawId.CF]
    assert request.rerank is False


def test_ask_can_print_the_sources(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    use_service(monkeypatch, settings, FakeService())

    result = runner.invoke(cli.app, ["Pergunta?", "--sources"])

    assert result.exit_code == 0, result.output
    assert "Art. 49. O consumidor pode desistir do contrato" in result.output


def test_ask_shows_resets(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    events = [TokenEvent(text="Rascunho"), ResetEvent(reason="Revisão."), *STREAM[1:]]
    use_service(monkeypatch, settings, FakeService(events))

    result = runner.invoke(cli.app, ["Pergunta?"])

    assert result.exit_code == 0, result.output
    assert "Rascunho\nResposta reiniciada: Revisão." in result.output


def test_ask_prints_json(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    use_service(monkeypatch, settings, FakeService())

    result = runner.invoke(cli.app, ["Pergunta?", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == RESPONSE.model_dump(mode="json")


def test_thread_ids_are_validated(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings, service)

    result = runner.invoke(cli.app, ["Pergunta?", "--thread", "minha conversa"])

    assert result.exit_code == 2
    assert "--thread" in result.output
    assert service.requests == []


def test_json_needs_a_question(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    use_service(monkeypatch, settings, FakeService())

    result = runner.invoke(cli.app, ["--json"])

    assert result.exit_code == 2
    assert "--json" in result.output


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_failures_exit_with_an_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings, extra: list[str]
) -> None:
    events: list[ChatEvent] = [
        StatusEvent(stage="analyze", message="Analisando"),
        ErrorEvent(message="Falhou."),
    ]
    use_service(monkeypatch, settings, FakeService(events))

    result = runner.invoke(cli.app, ["Pergunta?", *extra])

    assert result.exit_code == 1
    assert "Falhou." in result.output


def test_a_reply_without_any_model_exits_with_an_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    reply = "Serviço de linguagem indisponível."
    unavailable = RESPONSE.model_copy(
        update={"route": "unavailable", "answer": reply, "citations": [], "grounding": None}
    )
    events: list[ChatEvent] = [TokenEvent(text=reply), FinalEvent(response=unavailable)]
    use_service(monkeypatch, settings, FakeService(events))

    result = runner.invoke(cli.app, ["Pergunta?"])

    assert result.exit_code == 1
    assert reply in result.output


@pytest.mark.parametrize(
    ("question", "message"),
    [("   ", "A pergunta está vazia."), ("x" * 11, "longa demais: tem 11 caracteres")],
)
def test_unusable_questions_are_refused(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    question: str,
    message: str,
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings.model_copy(update={"max_question_chars": 10}), service)

    result = runner.invoke(cli.app, [question])

    assert result.exit_code == 1
    assert message in result.output
    assert service.requests == []


def test_interactive_session_skips_a_question_that_is_too_long(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings.model_copy(update={"max_question_chars": 10}), service)

    result = runner.invoke(cli.app, [], input=f"{'x' * 11}\nCurta?\nsair\n")

    assert result.exit_code == 0, result.output
    assert "longa demais" in result.output
    assert [request.question for request in service.requests] == ["Curta?"]


def test_interactive_session_keeps_one_thread(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings, service)

    result = runner.invoke(cli.app, [], input="Primeira?\nSegunda?\nsair\n")

    assert result.exit_code == 0, result.output
    assert [request.question for request in service.requests] == ["Primeira?", "Segunda?"]
    assert len({request.thread_id for request in service.requests}) == 1


def test_interactive_session_ends_on_an_empty_line(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service = FakeService()
    use_service(monkeypatch, settings, service)

    result = runner.invoke(cli.app, [], input="\n")

    assert result.exit_code == 0, result.output
    assert service.requests == []
