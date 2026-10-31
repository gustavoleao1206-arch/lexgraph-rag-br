"""Interfaces between the layers, so each one can be tested with simple fakes."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from lexgraph.config import RetrievalMode
from lexgraph.domain import LawId, Passage
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    ServiceInfo,
    Source,
)


class ChatServiceError(RuntimeError):
    """A request could not be answered; the message is safe to show to users."""


class Retriever(Protocol):
    """Search over the indexed legal passages."""

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[Passage]:
        """Return up to ``k`` passages for ``query``, best first.

        ``laws`` restricts the search to those laws; ``mode`` and ``rerank`` override the
        configured defaults (the evaluation harness uses them to compare configurations).
        """
        ...

    def get_article(self, article_id: str) -> list[Passage]:
        """Return every indexed chunk of one article (e.g. ``"CLT:482"``) in order.

        Returns an empty list when the article is not indexed (unknown, revoked or vetoed).
        """
        ...

    def count(self) -> int:
        """Number of indexed chunks."""
        ...

    def rerank_available(self) -> bool:
        """Whether the reranker model can be loaded (it is optional in the Docker image)."""
        ...


class ChatService(Protocol):
    """Runs the agent for one question, in one shot or as a stream of events."""

    async def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer ``request`` and return the complete response."""
        ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        """Answer ``request`` as a stream of events that ends with a final or error event."""
        ...

    async def history(self, thread_id: str) -> list[HistoryMessage]:
        """Return the stored messages of a conversation thread (empty if unknown)."""
        ...

    async def info(self) -> ServiceInfo:
        """Report the configuration and the size of the index."""
        ...

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[Source]:
        """Search the index directly, without any language model (``index`` = rank)."""
        ...
