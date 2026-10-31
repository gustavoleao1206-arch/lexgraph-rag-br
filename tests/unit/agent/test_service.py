"""The chat service: event stream, final response, threads, options, search and failures."""

import asyncio
import logging
import re
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from lexgraph import __version__
from lexgraph.agent import graph as graph_module
from lexgraph.agent import prompts
from lexgraph.agent.context import AgentContext
from lexgraph.agent.graph import build_graph
from lexgraph.agent.llm import ANSWER_TAG, TRANSIENT_ERRORS, answer_role, structured_role
from lexgraph.agent.schemas import QueryAnalysis
from lexgraph.config import EmbeddingProvider, Settings
from lexgraph.domain import LawId
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ErrorEvent,
    FinalEvent,
    ResetEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
)
from lexgraph.service import ChatServiceError, LexGraphService, embedding_model_name
from tests.unit.agent.fakes import (
    CDC_49,
    FakeRetriever,
    Script,
    analysis,
    answer_model,
    grounded,
    make_context,
    relevant,
    unsupported,
)

if TYPE_CHECKING:
    from lexgraph.protocols import ChatService

QUESTION = ChatRequest(question="Posso desistir de uma compra feita pela internet?", thread_id="t1")


class FailingMidAnswer(GenericFakeChatModel):
    """Streams the start of an answer, then fails like a provider that drops the connection."""

    error: Exception = ValueError("conexão perdida no meio da resposta")

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Resposta "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="parcial"))
        raise self.error


def make_service(
    context: AgentContext, settings: Settings, *, corpus_date: str | None = "30/10/2026"
) -> LexGraphService:
    return LexGraphService(build_graph(InMemorySaver()), context, settings, corpus_date=corpus_date)


async def collect(service: LexGraphService, request: ChatRequest = QUESTION) -> list[ChatEvent]:
    return [event async for event in service.stream(request)]


def outline(events: Sequence[ChatEvent]) -> list[str]:
    """Event types with statuses named by stage and runs of tokens collapsed."""
    names: list[str] = []
    for event in events:
        name = f"status:{event.stage}" if isinstance(event, StatusEvent) else event.type
        if not (name == "token" and names and names[-1] == "token"):
            names.append(name)
    return names


def text_since_last_reset(events: Sequence[ChatEvent]) -> str:
    text = ""
    for event in events:
        if isinstance(event, ResetEvent):
            text = ""
        elif isinstance(event, TokenEvent):
            text += event.text
    return text


def final(events: Sequence[ChatEvent]) -> FinalEvent:
    assert isinstance(events[-1], FinalEvent)
    return events[-1]


async def test_stream_reports_progress_then_tokens_then_the_final_response(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(
        retriever,
        grader=Script(relevant(1)).runnable,
        answerer=answer_model("Sim: o prazo é de 7 dias [1]."),
    )

    events = await collect(make_service(context, settings))

    assert outline(events) == [
        "status:analyze",
        "status:retrieve",
        "status:grade",
        "sources",
        "status:generate",
        "token",
        "status:verify",
        "final",
    ]
    assert text_since_last_reset(events) == final(events).response.answer


async def test_final_response_describes_the_answer(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(
        retriever,
        grader=Script(relevant(1, 2)).runnable,
        answerer=answer_model("Sim [1], mas veja também [2] e [7]."),
        max_regenerations=0,
    )

    events = await collect(make_service(context, settings))

    response = final(events).response
    sources = next(event for event in events if isinstance(event, SourcesEvent)).sources
    assert response.thread_id == "t1"
    assert response.route == "answered"
    # [7] matches no source: on the last pass it is removed instead of shown.
    assert response.answer == "Sim [1], mas veja também [2] e."
    assert response.sources == sources
    assert [source.index for source in response.sources] == [1, 2]
    assert [(c.index, c.article_id) for c in response.citations] == [(1, "CDC:49"), (2, "CLT:58")]
    assert response.grounding is not None
    assert response.grounding.grounded
    assert response.grounding.unsupported_claims == []
    assert response.rewrites == 0
    assert response.regenerations == 0
    assert response.latency_ms >= 0


async def test_fixed_replies_arrive_as_a_single_token(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(retriever, analyzer=Script(analysis("smalltalk")).runnable)

    events = await collect(make_service(context, settings))

    assert outline(events) == ["status:analyze", "status:respond", "token", "final"]
    response = final(events).response
    assert response.route == "smalltalk"
    assert response.sources == []
    assert text_since_last_reset(events) == prompts.templated_reply("smalltalk")


async def test_regeneration_resets_the_streamed_answer(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(
        retriever,
        verifier=Script(unsupported("O prazo de 30 dias não consta."), grounded()).runnable,
        answerer=answer_model("O prazo é de 30 dias [1].", "O prazo é de 7 dias [1]."),
    )

    events = await collect(make_service(context, settings))

    resets = [event for event in events if isinstance(event, ResetEvent)]
    assert [reset.reason for reset in resets] == [prompts.RESET_REGENERATION]
    response = final(events).response
    assert response.answer == "O prazo é de 7 dias [1]."
    assert text_since_last_reset(events) == response.answer
    assert response.regenerations == 1


async def test_switching_models_mid_answer_resets_the_stream(
    retriever: FakeRetriever, settings: Settings
) -> None:
    fallback = GenericFakeChatModel(messages=iter([AIMessage("Resposta completa [1].")]))
    context = make_context(
        retriever, answerer=answer_role(FailingMidAnswer(messages=iter([])), fallback)
    )

    events = await collect(make_service(context, settings))

    tokens = [event.text for event in events if isinstance(event, TokenEvent)]
    assert "".join(tokens).startswith("Resposta parcial")
    resets = [event.reason for event in events if isinstance(event, ResetEvent)]
    assert resets == [prompts.RESET_MODEL_SWITCH]
    assert text_since_last_reset(events) == "Resposta completa [1]."
    assert final(events).response.answer == "Resposta completa [1]."


async def test_failed_generation_replaces_the_partial_answer(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(retriever, answerer=answer_role(FailingMidAnswer(messages=iter([]))))

    events = await collect(make_service(context, settings))

    assert outline(events)[-4:] == ["token", "reset", "token", "final"]
    assert [e.reason for e in events if isinstance(e, ResetEvent)] == [prompts.RESET_FINAL]
    response = final(events).response
    assert response.route == "unavailable"
    assert response.answer == prompts.GENERATION_FAILED_REPLY
    assert text_since_last_reset(events) == prompts.GENERATION_FAILED_REPLY
    assert [source.article_id for source in response.sources][:1] == ["CDC:49"]


async def test_ask_raises_when_no_model_could_answer(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(retriever, answerer=Script(ValueError("modelo fora do ar")).runnable)

    with pytest.raises(ChatServiceError, match=re.escape(prompts.MODELS_UNAVAILABLE)):
        await make_service(context, settings).ask(QUESTION)


async def test_a_role_without_a_model_explains_how_to_configure_it(
    retriever: FakeRetriever, settings: Settings
) -> None:
    analyzer = structured_role(QueryAnalysis, None, name="analyze_query")
    service = make_service(make_context(retriever, analyzer=analyzer), settings)

    events = await collect(service)

    assert events[-1] == ErrorEvent(message=prompts.MODELS_NOT_CONFIGURED)
    assert retriever.searches == []


async def test_failures_end_the_stream_with_a_safe_message(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    broken = FakeRetriever(error=RuntimeError("Qdrant fora do ar em /srv/dados"))
    service = make_service(make_context(broken), settings)

    with caplog.at_level(logging.ERROR, logger="lexgraph.service"):
        events = await collect(service)

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].message == prompts.SERVICE_ERROR
    assert not any(isinstance(event, FinalEvent) for event in events)
    assert "chat request failed (thread t1)" in caplog.text
    assert "Qdrant fora do ar" in caplog.text


async def test_ask_returns_the_final_response(retriever: FakeRetriever, settings: Settings) -> None:
    service = make_service(make_context(retriever, answerer=answer_model("Sim [1].")), settings)

    response = await service.ask(QUESTION)

    assert response.answer == "Sim [1]."
    assert response.citations[0].article_id == "CDC:49"


async def test_ask_raises_with_the_safe_message(settings: Settings) -> None:
    service = make_service(make_context(FakeRetriever(error=RuntimeError("falha"))), settings)

    with pytest.raises(ChatServiceError, match=re.escape(prompts.SERVICE_ERROR)):
        await service.ask(QUESTION)


async def test_history_follows_the_thread(retriever: FakeRetriever, settings: Settings) -> None:
    context = make_context(retriever, answerer=answer_model("Primeira [1].", "Segunda [1]."))
    service = make_service(context, settings)

    await service.ask(ChatRequest(question="Pergunta um?", thread_id="conversa"))
    await service.ask(ChatRequest(question="Pergunta dois?", thread_id="conversa"))

    history = await service.history("conversa")
    assert [(m.role, m.content) for m in history] == [
        ("user", "Pergunta um?"),
        ("assistant", "Primeira [1]."),
        ("user", "Pergunta dois?"),
        ("assistant", "Segunda [1]."),
    ]
    assert await service.history("desconhecida") == []


def asked(messages: Any) -> str:
    """The question inside the last message of an answer prompt."""
    match = re.search(r"<pergunta>\n(.*)\n</pergunta>", messages[-1].text)
    assert match is not None
    return match.group(1)


async def test_runs_on_the_same_thread_take_turns(
    retriever: FakeRetriever, settings: Settings
) -> None:
    log: list[str] = []

    async def slow_answer(messages: Any) -> AIMessage:
        log.append(f"início: {asked(messages)}")
        await asyncio.sleep(0.01)
        log.append(f"fim: {asked(messages)}")
        return AIMessage(f"Resposta a {asked(messages)}")

    service = make_service(make_context(retriever, answerer=RunnableLambda(slow_answer)), settings)
    first = ChatRequest(question="Pergunta um?", thread_id="conversa")
    second = ChatRequest(question="Pergunta dois?", thread_id="conversa")

    one, two = await asyncio.gather(service.ask(first), service.ask(second))

    assert log == [
        "início: Pergunta um?",
        "fim: Pergunta um?",
        "início: Pergunta dois?",
        "fim: Pergunta dois?",
    ]
    assert (one.answer, two.answer) == ("Resposta a Pergunta um?", "Resposta a Pergunta dois?")
    history = await service.history("conversa")
    assert [message.content for message in history] == [
        "Pergunta um?",
        "Resposta a Pergunta um?",
        "Pergunta dois?",
        "Resposta a Pergunta dois?",
    ]


async def test_runs_on_different_threads_overlap(
    retriever: FakeRetriever, settings: Settings
) -> None:
    both_writing = asyncio.Barrier(2)

    async def answer_together(messages: Any) -> AIMessage:
        async with asyncio.timeout(5):
            await both_writing.wait()
        return AIMessage(f"Resposta a {asked(messages)}")

    answerer = RunnableLambda(answer_together)
    service = make_service(make_context(retriever, answerer=answerer), settings)
    requests = [ChatRequest(question=f"Pergunta {n}?", thread_id=f"t{n}") for n in (1, 2)]

    responses = await asyncio.gather(*(service.ask(request) for request in requests))

    assert [response.answer for response in responses] == [
        "Resposta a Pergunta 1?",
        "Resposta a Pergunta 2?",
    ]


async def test_a_thread_id_is_created_when_missing(
    retriever: FakeRetriever, settings: Settings
) -> None:
    service = make_service(make_context(retriever), settings)

    response = await service.ask(ChatRequest(question="Qual o prazo?"))

    assert re.fullmatch(r"[0-9a-f]{32}", response.thread_id)
    assert len(await service.history(response.thread_id)) == 2


async def test_request_options_reach_the_search(
    retriever: FakeRetriever, settings: Settings
) -> None:
    service = make_service(make_context(retriever), settings)
    request = ChatRequest(question="Jornada de trabalho?", laws=[LawId.CLT], rerank=True)

    await service.ask(request)

    assert retriever.searches[0]["laws"] == [LawId.CLT]
    assert retriever.searches[0]["rerank"] is True


async def test_usage_is_summed_over_model_calls(
    retriever: FakeRetriever, settings: Settings
) -> None:
    reply = AIMessage(
        "Resposta [1].",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        response_metadata={"model_name": "modelo-de-teste"},
    )
    answerer = FakeMessagesListChatModel(responses=[reply]).with_config(tags=[ANSWER_TAG])
    service = make_service(make_context(retriever, answerer=answerer), settings)

    events = await collect(service)

    usage = final(events).response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (120, 30, 150)
    assert usage.models == ["modelo-de-teste"]
    assert text_since_last_reset(events) == "Resposta [1]."


@pytest.mark.parametrize(
    ("corpus_date", "expected"), [("01/02/2026", "vigente em 01/02/2026"), (None, "compilada")]
)
async def test_the_corpus_date_reaches_the_answer_prompt(
    retriever: FakeRetriever, settings: Settings, corpus_date: str | None, expected: str
) -> None:
    answerer = Script(AIMessage("Resposta [1]."))
    service = make_service(
        make_context(retriever, answerer=answerer.runnable), settings, corpus_date=corpus_date
    )

    await service.ask(QUESTION)

    assert expected in answerer.inputs[0][0].text


def test_the_service_implements_the_chat_protocol(
    retriever: FakeRetriever, settings: Settings
) -> None:
    service: ChatService = make_service(make_context(retriever), settings)  # checked by mypy

    assert isinstance(service, LexGraphService)


async def test_info_reports_the_configuration(settings: Settings) -> None:
    retriever = FakeRetriever([CDC_49], reranker=True)
    service = make_service(make_context(retriever), settings)

    info = await service.info()

    assert info.version == __version__
    assert info.indexed_chunks == 1
    assert info.collection == settings.collection
    assert info.embedding_provider == "fastembed"
    assert info.embedding_model == settings.embedding_model
    assert info.retrieval_mode == settings.retrieval_mode
    assert info.rerank is settings.rerank
    assert info.rerank_available is True
    assert info.llm_model == settings.llm_model
    assert info.llm_fallback_model == settings.llm_fallback_model
    assert info.fast_llm_model == settings.fast_llm_model
    assert info.models_configured is True
    assert info.max_question_chars == settings.max_question_chars


async def test_info_tells_when_models_are_not_configured(
    retriever: FakeRetriever, settings: Settings
) -> None:
    service = LexGraphService(
        build_graph(InMemorySaver()), make_context(retriever), settings, models_configured=False
    )

    info = await service.info()

    assert info.models_configured is False
    assert info.rerank_available is False


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("fastembed", "google/embeddinggemma-300m", "google/embeddinggemma-300m"),
        ("google", "google/embeddinggemma-300m", "gemini-embedding-001"),
        ("google", "gemini-embedding-2", "gemini-embedding-2"),
        ("google", "models/gemini-embedding-001", "models/gemini-embedding-001"),
    ],
)
def test_the_embedding_model_is_the_one_the_retrieval_layer_uses(
    settings: Settings, provider: EmbeddingProvider, model: str, expected: str
) -> None:
    configured = settings.model_copy(
        update={"embedding_provider": provider, "embedding_model": model}
    )

    assert embedding_model_name(configured) == expected


async def test_search_returns_ranked_sources_without_any_model(
    retriever: FakeRetriever, settings: Settings
) -> None:
    analyzer = Script(analysis())
    service = make_service(make_context(retriever, analyzer=analyzer.runnable), settings)

    sources = await service.search("jornada", k=2, laws=[LawId.CLT, LawId.CF], mode="sparse")

    assert [(source.index, source.chunk_id) for source in sources] == [
        (1, "CLT:58#0"),
        (2, "CF:7#0"),
    ]
    assert sources[0].text.startswith("Art. 58")
    assert retriever.searches == [
        {
            "query": "jornada",
            "k": 2,
            "laws": [LawId.CLT, LawId.CF],
            "mode": "sparse",
            "rerank": None,
        }
    ]
    assert analyzer.inputs == []


async def test_search_failures_raise_a_safe_error(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    broken = FakeRetriever(error=RuntimeError("Qdrant fora do ar em /srv/dados"))
    service = make_service(make_context(broken), settings)

    with (
        caplog.at_level(logging.ERROR, logger="lexgraph.service"),
        pytest.raises(ChatServiceError, match=re.escape(prompts.SEARCH_ERROR)),
    ):
        await service.search("jornada")

    assert "Qdrant fora do ar" in caplog.text


async def test_a_retried_generation_restarts_the_stream(
    retriever: FakeRetriever, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        graph_module,
        "GENERATION_RETRY",
        RetryPolicy(max_attempts=2, initial_interval=0.0, jitter=False, retry_on=TRANSIENT_ERRORS),
    )
    model = FailingMidAnswer(messages=iter([]), error=httpx.ReadError("conexão perdida"))
    service = make_service(make_context(retriever, answerer=answer_role(model)), settings)

    events = await collect(service)

    resets = [event.reason for event in events if isinstance(event, ResetEvent)]
    assert resets == [prompts.RESET_MODEL_SWITCH, prompts.RESET_FINAL]
    assert final(events).response.route == "unavailable"
    assert final(events).response.answer == prompts.GENERATION_FAILED_REPLY
    assert text_since_last_reset(events) == prompts.GENERATION_FAILED_REPLY


async def test_without_models_every_request_explains_how_to_configure_them(
    retriever: FakeRetriever, settings: Settings
) -> None:
    context = make_context(retriever)
    service = LexGraphService(
        build_graph(InMemorySaver()), context, settings, models_configured=False
    )

    events = await collect(service)

    assert events == [ErrorEvent(message=prompts.MODELS_NOT_CONFIGURED)]
    assert retriever.searches == []
