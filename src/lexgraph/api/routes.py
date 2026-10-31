"""HTTP routes: health check, chat (one-shot and streamed), conversation threads, corpus and search.

Long-lived objects are created by the application lifespan and read from the request state
(see :class:`AppState`); routes never build clients themselves.
"""

import logging
from collections.abc import AsyncIterable
from typing import Annotated, Literal, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, StringConstraints

from lexgraph.catalog import law_info
from lexgraph.config import RetrievalMode, Settings
from lexgraph.corpus import Corpus, LawManifest
from lexgraph.domain import ARTICLE_ID_RE, Article, LawId
from lexgraph.protocols import ChatService, ChatServiceError
from lexgraph.schemas import (
    QUESTION_MAX_CHARS,
    THREAD_ID_PATTERN,
    ChatEvent,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    HistoryMessage,
    LawSummary,
    ServiceInfo,
    Source,
)

logger = logging.getLogger(__name__)

ANSWER_FAILURE_MESSAGE = "Não foi possível concluir a resposta. Tente novamente em instantes."
"""Shown to users when answering fails; the details go to the log."""

DONE_EVENT = ServerSentEvent(raw_data="[DONE]", event="done")

SEARCH_MAX_K = 20
"""Most passages one ``GET /v1/search`` call may return."""

STREAM_DESCRIPTION = """\
Streams the answer as Server-Sent Events. The `event` field is the event type and `data` is
the event as JSON:

- `status`: progress through the agent steps (analyze, retrieve, grade, rewrite, ...);
- `sources`: the passages selected as context, numbered as cited in the answer (`[n]`);
- `token`: a piece of the answer text;
- `reset`: discard the text streamed so far; `reason` says why, in Portuguese;
- `final`: the complete `ChatResponse` (route `unavailable` when no language model could
  answer), or `error`: a message that is safe to show to users.

The stream always ends with `event: done` and `data: [DONE]`. Idle connections receive a
`: ping` comment every 15 seconds.
"""


class AppState(TypedDict):
    """Objects created by the lifespan and shared by every request."""

    settings: Settings
    service: ChatService
    corpus: Corpus | None


class Health(BaseModel):
    """Liveness probe response."""

    status: Literal["ok"] = "ok"


class SearchParams(BaseModel):
    """Query parameters of ``GET /v1/search``."""

    q: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=QUESTION_MAX_CHARS)
    ] = Field(description="Text to search for, such as a question.")
    k: int = Field(default=5, ge=1, le=SEARCH_MAX_K, description="Number of passages.")
    mode: RetrievalMode | None = Field(
        default=None, description="Override the server's retrieval mode."
    )
    laws: list[LawId] | None = Field(
        default=None, max_length=len(LawId), description="Restrict the search to these laws."
    )


def get_app_settings(request: Request) -> Settings:
    """Settings the application was created with."""
    settings: Settings = request.state["settings"]
    return settings


def get_service(request: Request) -> ChatService:
    """The chat service opened by the lifespan."""
    service: ChatService = request.state["service"]
    return service


def get_corpus(request: Request) -> Corpus | None:
    """The parsed corpus, or ``None`` when it was not built."""
    corpus: Corpus | None = request.state["corpus"]
    return corpus


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ServiceDep = Annotated[ChatService, Depends(get_service)]
CorpusDep = Annotated[Corpus | None, Depends(get_corpus)]


async def checked_chat_request(
    chat_request: ChatRequest, settings: SettingsDep, service: ServiceDep
) -> ChatRequest:
    """The request body, rejected with 422 when the server cannot honor it.

    That is a question longer than ``max_question_chars`` or a reranking request on a server
    without the reranker model. Checked in a dependency so the streaming endpoint fails
    before its response starts.
    """
    _check_length(chat_request.question, ("body", "question"), settings.max_question_chars)
    if chat_request.rerank and not (await service.info()).rerank_available:
        raise RequestValidationError(
            [
                {
                    "type": "rerank_unavailable",
                    "loc": ("body", "rerank"),
                    "msg": "Reranking is not available on this server",
                }
            ]
        )
    return chat_request


def checked_search_params(
    params: Annotated[SearchParams, Query()], settings: SettingsDep
) -> SearchParams:
    """The search parameters, rejected with 422 when ``q`` exceeds ``max_question_chars``."""
    _check_length(params.q, ("query", "q"), settings.max_question_chars)
    return params


def _check_length(text: str, loc: tuple[str, ...], limit: int) -> None:
    if len(text) > limit:
        raise RequestValidationError(
            [
                {
                    "type": "string_too_long",
                    "loc": loc,
                    "msg": f"String should have at most {limit} characters",
                }
            ]
        )


ChatRequestDep = Annotated[ChatRequest, Depends(checked_chat_request)]
SearchParamsDep = Annotated[SearchParams, Depends(checked_search_params)]

router = APIRouter()
v1_router = APIRouter(prefix="/v1")


@router.get("/health", tags=["system"], summary="Liveness probe")
async def health() -> Health:
    """Answer ``{"status": "ok"}`` while the process is serving requests."""
    return Health()


@v1_router.get("/info", tags=["system"], summary="Service configuration and index size")
async def get_info(service: ServiceDep) -> ServiceInfo:
    """Report the models, the retrieval configuration and the number of indexed chunks."""
    return await service.info()


@v1_router.post(
    "/chat",
    tags=["chat"],
    summary="Answer a question",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "No answer could be produced (for example, no language model "
            "answered); `detail` is safe to show to users."
        }
    },
)
async def chat(chat_request: ChatRequestDep, service: ServiceDep) -> ChatResponse:
    """Run the agent and return the complete answer with its sources and citations."""
    try:
        return await service.ask(chat_request)
    except ChatServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    except Exception as error:
        logger.exception("Chat request failed (thread %s)", chat_request.thread_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=ANSWER_FAILURE_MESSAGE
        ) from error


@v1_router.post(
    "/chat/stream",
    response_class=EventSourceResponse,
    tags=["chat"],
    summary="Answer a question as a stream of Server-Sent Events",
    description=STREAM_DESCRIPTION,
)
async def chat_stream(
    chat_request: ChatRequestDep, service: ServiceDep
) -> AsyncIterable[ServerSentEvent]:
    """Relay the service events, guaranteeing a ``final`` or ``error`` event before ``done``."""
    terminated = False
    try:
        async for event in service.stream(chat_request):
            yield _as_sse(event)
            terminated = isinstance(event, FinalEvent | ErrorEvent)
    except Exception:
        logger.exception("Chat stream failed (thread %s)", chat_request.thread_id)
    else:
        if not terminated:
            logger.error(
                "Chat stream ended without a final event (thread %s)", chat_request.thread_id
            )
    if not terminated:
        yield _as_sse(ErrorEvent(message=ANSWER_FAILURE_MESSAGE))
    yield DONE_EVENT


@v1_router.get("/threads/{thread_id}", tags=["chat"], summary="Messages of a conversation")
async def get_thread(
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)], service: ServiceDep
) -> list[HistoryMessage]:
    """Return the stored messages of a thread in conversation order (empty if unknown)."""
    return await service.history(thread_id)


@v1_router.get("/laws", tags=["corpus"], summary="Laws in the corpus")
async def list_laws(corpus: CorpusDep) -> list[LawSummary]:
    """List the laws in catalog order, with article counts and provenance."""
    if corpus is None:
        return []
    manifests = corpus.manifest.laws
    return [_law_summary(manifests[law_id]) for law_id in LawId if law_id in manifests]


@v1_router.get("/search", tags=["corpus"], summary="Search the indexed passages")
async def search(params: SearchParamsDep, service: ServiceDep) -> list[Source]:
    """Return the passages that best match ``q``, best first, without any language model.

    ``index`` is the rank of each passage, from 1.
    """
    return await service.search(params.q, k=params.k, laws=params.laws, mode=params.mode)


@v1_router.get(
    "/articles/{article_id}",
    tags=["corpus"],
    summary="One article of the corpus",
    responses={status.HTTP_404_NOT_FOUND: {"description": "The article is not in the corpus."}},
)
async def get_article(
    article_id: Annotated[str, Path(pattern=ARTICLE_ID_RE.pattern, examples=["CLT:482"])],
    corpus: CorpusDep,
) -> Article:
    """Return an article (any status) by its canonical id, such as ``CLT:482`` or ``CF:5``."""
    article = corpus.get(article_id) if corpus is not None else None
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Article {article_id} not found."
        )
    return article


def _as_sse(event: ChatEvent) -> ServerSentEvent:
    return ServerSentEvent(data=event, event=event.type)


def _law_summary(manifest: LawManifest) -> LawSummary:
    law = law_info(manifest.law_id)
    return LawSummary(
        law_id=manifest.law_id,
        short_name=law.short_name,
        name=law.name,
        official_name=law.official_name,
        urn=law.urn,
        source_url=manifest.source_url,
        articles=manifest.articles,
        in_force=manifest.in_force,
        revoked=manifest.revoked,
        vetoed=manifest.vetoed,
        fetched_at=manifest.fetched_at,
    )
