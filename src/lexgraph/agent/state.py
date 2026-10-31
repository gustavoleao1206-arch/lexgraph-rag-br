"""State of the LangGraph agent.

The state is checkpointed per conversation thread, so it holds only JSON-friendly values.
``messages`` accumulates across turns; every other key belongs to the current turn and is
reset by :func:`new_turn` when a question arrives.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from lexgraph.schemas import Route


class AgentState(TypedDict):
    """Graph state; passage-like values are ``Passage.model_dump(mode="json")`` dicts."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    standalone_question: str
    search_query: str
    sub_queries: list[str]
    law_filter: list[str]
    article_refs: list[str]
    route: Route
    documents: list[dict[str, Any]]
    context: list[dict[str, Any]]
    rewrite_count: int
    generation_count: int
    answer: str
    answer_message_id: str | None
    feedback: list[str]
    grounding: dict[str, Any] | None


def new_turn(question: str, *, laws: list[str] | None = None) -> dict[str, Any]:
    """Input for a new turn: the user message plus every per-turn key reset.

    ``laws`` is an explicit filter chosen by the user; the agent may narrow it further but
    never widens it.
    """
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "standalone_question": question,
        "search_query": question,
        "sub_queries": [],
        "law_filter": list(laws or []),
        "article_refs": [],
        "route": "answered",
        "documents": [],
        "context": [],
        "rewrite_count": 0,
        "generation_count": 0,
        "answer": "",
        "answer_message_id": None,
        "feedback": [],
        "grounding": None,
    }
