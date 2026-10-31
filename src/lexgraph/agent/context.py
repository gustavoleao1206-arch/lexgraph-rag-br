"""Run-scoped dependencies injected into the graph nodes through LangGraph's ``Runtime``.

Clients live here, never in the state, because the state is checkpointed.
"""

from dataclasses import dataclass

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from lexgraph.agent.schemas import GroundingCheck, QueryAnalysis, QueryRewrite, RelevanceGrade
from lexgraph.protocols import Retriever


@dataclass(frozen=True)
class AgentOptions:
    """Behavior switches for one run (defaults come from :class:`lexgraph.config.Settings`)."""

    top_k: int = 10
    context_k: int = 5
    max_rewrites: int = 1
    max_regenerations: int = 1
    grading: bool = True
    verification: bool = True
    history_turns: int = 4
    rerank: bool | None = None


@dataclass(frozen=True)
class AgentContext:
    """Everything the nodes need besides the state.

    The structured runnables take a list of messages (or a prompt value) and return the
    parsed model; ``answerer`` returns an ``AIMessage`` and is the only one that streams.
    """

    retriever: Retriever
    analyzer: Runnable[LanguageModelInput, QueryAnalysis]
    grader: Runnable[LanguageModelInput, RelevanceGrade]
    rewriter: Runnable[LanguageModelInput, QueryRewrite]
    verifier: Runnable[LanguageModelInput, GroundingCheck]
    answerer: Runnable[LanguageModelInput, BaseMessage]
    options: AgentOptions = AgentOptions()
