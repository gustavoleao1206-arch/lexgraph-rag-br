"""Chat service: runs the agent graph for one request and turns its stream into events.

The stream follows one rule that clients can rely on: the token events received since the
last reset event always spell the final answer. Answer tokens come from the ``generate``
node; when a model call is replaced mid-answer (provider fallback, retry or regeneration)
a reset event comes first, and fixed replies are sent as a single token before the final
event.
"""

import asyncio
import logging
import time
import weakref
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Annotated, Any, cast
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamPart
from pydantic import Field, TypeAdapter

from lexgraph import __version__
from lexgraph.agent import prompts
from lexgraph.agent.citations import build_citations, passages_to_sources
from lexgraph.agent.context import AgentContext
from lexgraph.agent.graph import AgentGraph
from lexgraph.agent.nodes import CORPUS_DATE_KEY, GraphState
from lexgraph.agent.state import new_turn
from lexgraph.config import RetrievalMode, Settings
from lexgraph.domain import LawId
from lexgraph.protocols import ChatServiceError
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    Grounding,
    HistoryMessage,
    ResetEvent,
    ServiceInfo,
    Source,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    Usage,
)

type _GraphStream = AsyncGenerator[StreamPart[GraphState, GraphState]]

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 25
"""Upper bound on graph steps; with the largest loop budgets the settings allow, a run takes 18."""

_NODE_EVENTS: TypeAdapter[StatusEvent | SourcesEvent | ResetEvent] = TypeAdapter(
    Annotated[StatusEvent | SourcesEvent | ResetEvent, Field(discriminator="type")]
)


__all__ = ["ChatServiceError", "LexGraphService", "embedding_model_name"]


class _AnswerStream:
    """Answer text sent so far as token events, kept consistent with the final answer."""

    def __init__(self) -> None:
        self.text = ""
        self._message_id: str | None = None

    def add(self, message: AIMessage) -> list[ChatEvent]:
        """Events for one streamed piece of the answer.

        A piece from a new model call while text is already out means the previous call
        failed after streaming part of its answer, so a reset comes first.
        """
        text = message.text
        if not text:
            return []
        events: list[ChatEvent] = []
        if self.text and message.id != self._message_id:
            events.append(ResetEvent(reason=prompts.RESET_MODEL_SWITCH))
            self.text = ""
        self._message_id = message.id
        self.text += text
        events.append(TokenEvent(text=text))
        return events

    def reset(self) -> None:
        self.text = ""
        self._message_id = None

    def finish(self, answer: str) -> list[ChatEvent]:
        """Events that make the streamed text equal to ``answer``, if it is not already."""
        if self.text.strip() == answer.strip():
            return []
        events: list[ChatEvent] = [ResetEvent(reason=prompts.RESET_FINAL)] if self.text else []
        if answer:
            events.append(TokenEvent(text=answer))
        return events


def _node_event(data: object) -> StatusEvent | SourcesEvent | ResetEvent | None:
    """Translate a custom event written by a node (``{"event": kind, ...}``)."""
    if not isinstance(data, Mapping) or data.get("event") not in {"status", "sources", "reset"}:
        logger.debug("ignoring custom stream event %r", data)
        return None
    payload = {key: value for key, value in data.items() if key != "event"}
    return _NODE_EVENTS.validate_python({"type": data["event"], **payload})


def _events(part: StreamPart[GraphState, GraphState], answer: _AnswerStream) -> list[ChatEvent]:
    """Client events for one part of the graph stream (custom events and answer tokens)."""
    if part["type"] == "custom":
        event = _node_event(part["data"])
        if isinstance(event, ResetEvent):
            answer.reset()
        return [event] if event is not None else []
    if part["type"] == "messages":
        message, metadata = part["data"]
        if metadata.get("langgraph_node") == "generate" and isinstance(message, AIMessage):
            return answer.add(message)
    return []


def _usage(handler: UsageMetadataCallbackHandler) -> Usage:
    per_model = handler.usage_metadata
    return Usage(
        input_tokens=sum(usage["input_tokens"] for usage in per_model.values()),
        output_tokens=sum(usage["output_tokens"] for usage in per_model.values()),
        total_tokens=sum(usage["total_tokens"] for usage in per_model.values()),
        models=sorted(per_model),
    )


def embedding_model_name(settings: Settings) -> str:
    """The dense model behind the index, as the retrieval layer resolves it."""
    # Imported here so that importing this module does not load the embedding models' stack.
    from lexgraph.retrieval.embeddings import embedding_model_name as resolve

    return resolve(settings)


class LexGraphService:
    """Answers questions with the agent graph; implements ``protocols.ChatService``.

    The graph must be compiled with a checkpointer: threads and their history are read from
    it. Runs on the same thread take turns, because each one reads the history the previous
    one wrote. ``corpus_date`` (``dd/mm/yyyy``) tells the answer model how current the legal
    texts are. With ``models_configured=False`` (some role has no provider API key) every
    request ends with an error event explaining how to configure one.
    """

    def __init__(
        self,
        graph: AgentGraph,
        context: AgentContext,
        settings: Settings,
        corpus_date: str | None = None,
        *,
        models_configured: bool = True,
    ) -> None:
        self._graph = graph
        self._context = context
        self._settings = settings
        self._corpus_date = corpus_date
        self._models_configured = models_configured
        # An entry lives while a run of its thread holds or awaits the lock.
        self._thread_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer ``request`` in one piece.

        Raises :class:`ChatServiceError` when no answer could be produced, including when no
        language model was reachable (route ``unavailable``).
        """
        async with aclosing(self.stream(request)) as events:
            async for event in events:
                if isinstance(event, FinalEvent):
                    if event.response.route == "unavailable":
                        raise ChatServiceError(prompts.MODELS_UNAVAILABLE)
                    return event.response
                if isinstance(event, ErrorEvent):
                    raise ChatServiceError(event.message)
        raise ChatServiceError(prompts.SERVICE_ERROR)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        """Answer ``request`` as status, sources, reset and token events, then a final event.

        Failures end the stream with an error event whose message is safe to show; the
        details go to the log. When no model could write the answer, the final event carries
        a fixed reply with route ``unavailable``.
        """
        if not self._models_configured:
            yield ErrorEvent(message=prompts.MODELS_NOT_CONFIGURED)
            return
        thread_id = request.thread_id or uuid4().hex
        usage = UsageMetadataCallbackHandler()
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, CORPUS_DATE_KEY: self._corpus_date},
            "recursion_limit": RECURSION_LIMIT,
            "callbacks": [usage],
        }
        answer = _AnswerStream()
        values: Mapping[str, Any] = {}
        started = time.perf_counter()
        try:
            async with self._thread_lock(thread_id):
                parts = self._run(request, config)
                async with aclosing(parts):
                    async for part in parts:
                        if part["type"] == "values":
                            values = part["data"]
                        for event in _events(part, answer):
                            yield event
        except ChatServiceError as error:
            logger.warning("chat request failed (thread %s): %s", thread_id, error)
            yield ErrorEvent(message=str(error))
            return
        except Exception:
            logger.exception("chat request failed (thread %s)", thread_id)
            yield ErrorEvent(message=prompts.SERVICE_ERROR)
            return
        response = self._response(thread_id, values, usage, started)
        for event in answer.finish(response.answer):
            yield event
        yield FinalEvent(response=response)

    async def history(self, thread_id: str) -> list[HistoryMessage]:
        """Questions and answers stored for ``thread_id``, oldest first."""
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = snapshot.values.get("messages", []) if snapshot.values else []
        return [
            HistoryMessage(
                role="user" if isinstance(message, HumanMessage) else "assistant",
                content=message.text,
            )
            for message in messages
            if isinstance(message, HumanMessage | AIMessage)
        ]

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[Source]:
        """Passages for ``query`` straight from the index, best first; ``index`` is the rank.

        No language model is involved. Raises :class:`ChatServiceError` when the index
        cannot be searched.
        """
        try:
            passages = await asyncio.to_thread(
                self._context.retriever.search, query, k=k, laws=laws, mode=mode
            )
        except Exception as error:
            logger.exception("search failed")
            raise ChatServiceError(prompts.SEARCH_ERROR) from error
        return passages_to_sources([passage.model_dump(mode="json") for passage in passages])

    async def info(self) -> ServiceInfo:
        """Models, retrieval settings, limits and the number of indexed chunks."""
        settings = self._settings
        retriever = self._context.retriever
        return ServiceInfo(
            version=__version__,
            collection=settings.collection,
            indexed_chunks=await asyncio.to_thread(retriever.count),
            embedding_provider=settings.embedding_provider,
            embedding_model=embedding_model_name(settings),
            retrieval_mode=settings.retrieval_mode,
            rerank=settings.rerank,
            rerank_available=await asyncio.to_thread(retriever.rerank_available),
            llm_model=settings.llm_model,
            llm_fallback_model=settings.llm_fallback_model,
            fast_llm_model=settings.fast_llm_model,
            models_configured=self._models_configured,
            max_question_chars=settings.max_question_chars,
        )

    def _thread_lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = self._thread_locks[thread_id] = asyncio.Lock()
        return lock

    def _run(self, request: ChatRequest, config: RunnableConfig) -> _GraphStream:
        """Start the graph on a new turn of the thread, with the request's own options.

        The ``values`` parts carry the state after each step; the last one is this run's
        result, whatever other runs write to the checkpointer afterwards.
        """
        turn = new_turn(request.question.strip(), laws=[law.value for law in request.laws or []])
        context = self._context
        if request.rerank is not None:
            context = replace(context, options=replace(context.options, rerank=request.rerank))
        stream = self._graph.astream(
            cast("GraphState", turn),
            config,
            context=context,
            stream_mode=["values", "messages", "custom"],
            version="v2",
        )
        return cast("_GraphStream", stream)

    @staticmethod
    def _response(
        thread_id: str,
        values: Mapping[str, Any],
        usage: UsageMetadataCallbackHandler,
        started: float,
    ) -> ChatResponse:
        sources = passages_to_sources(values.get("context", []))
        answer = values.get("answer", "")
        grounding = values.get("grounding")
        return ChatResponse(
            thread_id=thread_id,
            answer=answer,
            route=values.get("route", "answered"),
            sources=sources,
            citations=build_citations(answer, sources),
            grounding=Grounding.model_validate(grounding) if grounding else None,
            rewrites=values.get("rewrite_count", 0),
            regenerations=max(values.get("generation_count", 0) - 1, 0),
            usage=_usage(usage),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
