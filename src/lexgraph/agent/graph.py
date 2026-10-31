"""The agent graph: a corrective retrieval loop over the legal corpus.

Edges (``|`` separates the alternatives of a branch)::

    START    → analyze
    analyze  → retrieve | respond (small talk, out of scope)
    retrieve → grade | generate (grading off) | rewrite, respond (nothing found)
    grade    → generate | rewrite (nothing relevant, rewrites left) | respond (none left)
    rewrite  → retrieve
    generate → verify | END (verification off)
    verify   → END | generate (unsupported claims and regenerations left, as feedback)
    respond  → END

Loop budgets come from :class:`AgentOptions` (``max_rewrites``, ``max_regenerations``).
"""

from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from lexgraph.agent import nodes
from lexgraph.agent.context import AgentContext
from lexgraph.agent.llm import TRANSIENT_ERRORS
from lexgraph.agent.nodes import GraphState

type AgentGraph = CompiledStateGraph[GraphState, AgentContext, GraphState, GraphState]

GENERATION_RETRY = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=TRANSIENT_ERRORS)
"""Generation is retried once on rate limits, server and connection errors."""


def build_graph(checkpointer: BaseCheckpointSaver[str] | None = None) -> AgentGraph:
    """Compile the agent; pass a checkpointer to keep conversation threads between turns."""
    builder = StateGraph(GraphState, context_schema=AgentContext)
    builder.add_node("analyze", nodes.analyze)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node(
        "generate",
        nodes.generate,
        retry_policy=GENERATION_RETRY,
        # LangGraph injects the NodeError argument; its type hints do not describe that yet.
        error_handler=cast("Any", nodes.generation_failed),
    )
    builder.add_node("verify", nodes.verify)
    builder.add_node("respond", nodes.respond)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", nodes.route_after_analyze, ["retrieve", "respond"])
    builder.add_conditional_edges(
        "retrieve", nodes.route_after_retrieve, ["grade", "generate", "rewrite", "respond"]
    )
    builder.add_conditional_edges(
        "grade", nodes.route_after_grade, ["generate", "rewrite", "respond"]
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges("generate", nodes.route_after_generate, ["verify", END])
    builder.add_conditional_edges("verify", nodes.route_after_verify, ["generate", END])
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer, name="lexgraph")
