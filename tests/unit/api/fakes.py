"""A scripted chat service and sample wire objects, shared by the API and UI tests."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from lexgraph.config import RetrievalMode
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
    ServiceInfo,
    Source,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    Usage,
)

CDC_49_URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078.htm#art49"

SERVICE_INFO = ServiceInfo(
    version="0.1.0",
    collection="lexgraph",
    indexed_chunks=4321,
    embedding_provider="fastembed",
    embedding_model="google/embeddinggemma-300m",
    retrieval_mode="hybrid",
    rerank=False,
    rerank_available=True,
    llm_model="groq:openai/gpt-oss-120b",
    llm_fallback_model="google_genai:gemini-3.5-flash-lite",
    fast_llm_model="groq:openai/gpt-oss-20b",
    models_configured=True,
    max_question_chars=2000,
)


def sample_source(index: int = 1) -> Source:
    """The passage of CDC art. 49 as given to the model."""
    return Source(
        index=index,
        chunk_id="CDC:49#0",
        article_id="CDC:49",
        law_id=LawId.CDC,
        citation="CDC, art. 49",
        url=CDC_49_URL,
        text="Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias.",
        score=0.0328,
        rerank_score=None,
    )


def sample_response(thread_id: str = "thread-1") -> ChatResponse:
    """A grounded answer that cites its only source."""
    return ChatResponse(
        thread_id=thread_id,
        answer="Sim. Você pode desistir da compra em até 7 dias [1].",
        route="answered",
        sources=[sample_source()],
        citations=[Citation(index=1, article_id="CDC:49", citation="CDC, art. 49", url=CDC_49_URL)],
        grounding=Grounding(grounded=True),
        usage=Usage(
            input_tokens=900, output_tokens=120, total_tokens=1020, models=["openai/gpt-oss-120b"]
        ),
        latency_ms=1234,
    )


def sample_events(thread_id: str = "thread-1") -> list[ChatEvent]:
    """A complete, successful stream: progress, sources, tokens and the final response."""
    return [
        StatusEvent(stage="analyze", message="Analisando a pergunta"),
        StatusEvent(stage="retrieve", message="Buscando nas leis", detail={"passages": 6}),
        SourcesEvent(sources=[sample_source()]),
        TokenEvent(text="Sim. Você pode desistir "),
        TokenEvent(text="da compra em até 7 dias [1]."),
        FinalEvent(response=sample_response(thread_id)),
    ]


@dataclass(frozen=True)
class SearchCall:
    """Arguments of one call to :meth:`FakeChatService.search`."""

    query: str
    k: int
    laws: list[LawId] | None
    mode: RetrievalMode | None


class FakeChatService:
    """In-memory :class:`lexgraph.protocols.ChatService` that replays scripted events.

    ``failure``, when set, is raised by ``ask`` and at the end of ``stream``; ``search``
    returns ``search_results`` cut to ``k``.
    """

    def __init__(
        self, events: Sequence[ChatEvent] | None = None, *, failure: Exception | None = None
    ) -> None:
        self.events = list(events) if events is not None else sample_events()
        self.failure = failure
        self.requests: list[ChatRequest] = []
        self.histories: dict[str, list[HistoryMessage]] = {}
        self.info_report = SERVICE_INFO
        self.search_results = [sample_source()]
        self.searches: list[SearchCall] = []

    async def ask(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        final = next(event for event in self.events if isinstance(event, FinalEvent))
        return final.response

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event
        if self.failure is not None:
            raise self.failure

    async def history(self, thread_id: str) -> list[HistoryMessage]:
        return self.histories.get(thread_id, [])

    async def info(self) -> ServiceInfo:
        return self.info_report

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[Source]:
        self.searches.append(SearchCall(query, k, list(laws) if laws is not None else None, mode))
        return self.search_results[:k]


def error_events() -> list[ChatEvent]:
    """A stream that the service ended with a user-facing error."""
    return [
        StatusEvent(stage="analyze", message="Analisando a pergunta"),
        ErrorEvent(message="O provedor de linguagem está indisponível."),
    ]
