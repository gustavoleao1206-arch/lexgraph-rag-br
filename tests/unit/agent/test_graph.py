"""Whole agent runs through every route, with fake roles and an in-memory checkpointer."""

from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from lexgraph.agent import graph as graph_module
from lexgraph.agent import prompts
from lexgraph.agent.context import AgentContext
from lexgraph.agent.graph import AgentGraph, build_graph
from lexgraph.agent.llm import TRANSIENT_ERRORS
from lexgraph.agent.nodes import CORPUS_DATE_KEY, GraphState
from lexgraph.agent.schemas import GroundingCheck
from lexgraph.agent.state import new_turn
from lexgraph.domain import LawId
from tests.unit.agent.fakes import (
    CDC_49,
    LGPD_18,
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
    from langchain_core.runnables import RunnableConfig


async def run(
    context: AgentContext,
    question: str,
    *,
    graph: AgentGraph | None = None,
    thread: str = "thread-1",
    laws: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one turn; return the final state and the custom events, in order."""
    graph = graph or build_graph(InMemorySaver())
    config: RunnableConfig = {
        "configurable": {"thread_id": thread, CORPUS_DATE_KEY: "30/10/2026"},
        "recursion_limit": 25,
    }
    turn = cast("GraphState", new_turn(question, laws=laws))
    parts = graph.astream(turn, config, context=context, stream_mode="custom", version="v2")
    events: list[dict[str, Any]] = [
        part["data"] async for part in parts if part["type"] == "custom"
    ]
    snapshot = await graph.aget_state(config)
    return snapshot.values, events


def stages(events: list[dict[str, Any]]) -> list[str]:
    return [event["stage"] for event in events if event["event"] == "status"]


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [event["event"] for event in events]


async def test_lookup_of_a_cited_article(retriever: FakeRetriever) -> None:
    context = make_context(
        retriever,
        grader=Script(relevant(1)).runnable,
        answerer=answer_model("O consumidor pode desistir em 7 dias [1]."),
    )

    state, events = await run(context, "O que diz o art. 49 do CDC?")

    assert stages(events) == ["analyze", "retrieve", "grade", "generate", "verify"]
    assert kinds(events)[3:5] == ["sources", "status"]
    assert retriever.lookups == ["CDC:49"]
    assert [passage["chunk_id"] for passage in state["context"]] == ["CDC:49#0"]
    assert state["answer"] == "O consumidor pode desistir em 7 dias [1]."
    assert state["route"] == "answered"
    assert state["grounding"] == {"grounded": True, "unsupported_claims": []}
    assert events[3]["sources"][0]["index"] == 1
    assert events[3]["sources"][0]["article_id"] == "CDC:49"


async def test_normal_question_is_searched_and_answered(retriever: FakeRetriever) -> None:
    context = make_context(
        retriever,
        analyzer=Script(analysis(query="duração do trabalho", laws=["CLT", "CF"])).runnable,
        grader=Script(relevant(1)).runnable,
        answerer=answer_model("A jornada normal é de 8 horas diárias [1]."),
    )

    state, _ = await run(context, "Quantas horas posso trabalhar por dia?")

    # Each law picked by the analysis is searched on its own, and every law is searched too.
    assert [search["laws"] for search in retriever.searches] == [[LawId.CLT], [LawId.CF], None]
    assert [passage["article_id"] for passage in state["context"]] == ["CLT:58"]
    assert state["messages"][-1].text == "A jornada normal é de 8 horas diárias [1]."
    assert state["rewrite_count"] == 0
    assert state["generation_count"] == 1


async def test_rewrite_rescues_a_search_that_found_nothing() -> None:
    retriever = FakeRetriever(by_query={"consulta ruim": [LGPD_18], "nova consulta": [CDC_49]})
    context = make_context(
        retriever,
        analyzer=Script(analysis(query="consulta ruim")).runnable,
        grader=Script(relevant(), relevant(1)).runnable,
        answerer=answer_model("Sim, em 7 dias [1]."),
    )

    state, events = await run(context, "Comprei pela internet e me arrependi. E agora?")

    assert stages(events) == [
        "analyze",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "verify",
    ]
    assert [search["query"] for search in retriever.searches] == ["consulta ruim", "nova consulta"]
    assert state["rewrite_count"] == 1
    assert state["route"] == "answered"
    assert [passage["article_id"] for passage in state["context"]] == ["CDC:49"]


async def test_a_rewrite_searches_the_new_query_alone(retriever: FakeRetriever) -> None:
    context = make_context(
        retriever,
        analyzer=Script(analysis(query="regra", sub_queries=["exceção", "outra lei"])).runnable,
        grader=Script(relevant(), relevant(1)).runnable,
    )

    state, _ = await run(context, "Comprei pela internet e me arrependi. E agora?")

    # The sub-queries found nothing useful either: searching them again would bring back
    # the passages just discarded.
    queries = [search["query"] for search in retriever.searches]
    assert queries == ["regra", "exceção", "outra lei", "nova consulta"]
    assert state["sub_queries"] == []
    assert state["route"] == "answered"


async def test_no_context_after_the_rewrite_budget(retriever: FakeRetriever) -> None:
    answerer = Script(AIMessage("não deveria responder"))
    context = make_context(
        retriever,
        grader=Script(relevant()).runnable,
        answerer=answerer.runnable,
        max_rewrites=1,
    )

    state, events = await run(context, "Qual a pena para furto simples?")

    assert stages(events) == [
        "analyze",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "respond",
    ]
    assert state["route"] == "no_context"
    assert state["answer"] == prompts.templated_reply("no_context")
    assert state["context"] == []
    assert answerer.inputs == []


@pytest.mark.parametrize("intent", ["smalltalk", "out_of_scope"])
async def test_messages_outside_the_law_get_a_fixed_reply(
    retriever: FakeRetriever, intent: str
) -> None:
    context = make_context(retriever, analyzer=Script(analysis(intent)).runnable)

    state, events = await run(context, "Bom dia! Qual a receita de bolo de fubá?")

    assert stages(events) == ["analyze", "respond"]
    assert state["route"] == intent
    assert state["answer"] == prompts.templated_reply(intent)  # type: ignore[arg-type]
    assert retriever.searches == []
    assert [type(message) for message in state["messages"]] == [HumanMessage, AIMessage]


async def test_unsupported_answer_is_regenerated_with_feedback(retriever: FakeRetriever) -> None:
    answerer = Script(AIMessage("O prazo é de 30 dias [1]."), AIMessage("O prazo é de 7 dias [1]."))
    context = make_context(
        retriever,
        grader=Script(relevant(1)).runnable,
        verifier=Script(
            unsupported("O prazo de 30 dias não consta da fonte."), grounded()
        ).runnable,
        answerer=answerer.runnable,
    )

    state, events = await run(context, "Qual o prazo para desistir?")

    assert stages(events)[-4:] == ["generate", "verify", "generate", "verify"]
    assert {"event": "reset", "reason": prompts.RESET_REGENERATION} in events
    assert kinds(events).count("sources") == 1
    assert state["answer"] == "O prazo é de 7 dias [1]."
    assert state["generation_count"] == 2
    assert state["grounding"]["grounded"] is True
    answers = [m for m in state["messages"] if isinstance(m, AIMessage)]
    assert [m.text for m in answers] == ["O prazo é de 7 dias [1]."]
    assert "- O prazo de 30 dias não consta da fonte." in answerer.inputs[1][-1].text
    assert "revisão" not in answerer.inputs[0][-1].text


async def test_regeneration_stops_at_the_budget(retriever: FakeRetriever) -> None:
    context = make_context(
        retriever,
        verifier=Script(unsupported("Afirmação sem apoio.")).runnable,
        answerer=Script(AIMessage("Resposta [1].")).runnable,
        max_regenerations=0,
    )

    state, _ = await run(context, "Qual o prazo para desistir?")

    assert state["generation_count"] == 1
    assert state["grounding"] == {"grounded": False, "unsupported_claims": ["Afirmação sem apoio."]}


async def test_generation_failure_ends_with_a_polite_reply(retriever: FakeRetriever) -> None:
    verifier = Script(grounded())
    context = make_context(
        retriever,
        grader=Script(relevant(1)).runnable,
        verifier=verifier.runnable,
        answerer=Script(ValueError("modelo indisponível")).runnable,
    )

    state, events = await run(context, "Qual o prazo para desistir?")

    assert state["answer"] == prompts.GENERATION_FAILED_REPLY
    assert state["messages"][-1].text == prompts.GENERATION_FAILED_REPLY
    assert state["route"] == "unavailable"
    assert [passage["article_id"] for passage in state["context"]] == ["CDC:49"]
    assert state["grounding"] is None
    assert "verify" not in stages(events)
    assert verifier.inputs == []


async def test_a_failed_regeneration_keeps_the_first_answer(retriever: FakeRetriever) -> None:
    context = make_context(
        retriever,
        verifier=Script(unsupported("Afirmação sem apoio.")).runnable,
        answerer=Script(AIMessage("Primeira [1]."), ValueError("modelo indisponível")).runnable,
    )

    state, _ = await run(context, "Qual o prazo para desistir?")

    assert state["answer"] == "Primeira [1]."
    assert state["route"] == "answered"
    assert state["grounding"] == {"grounded": False, "unsupported_claims": ["Afirmação sem apoio."]}


@pytest.mark.parametrize(
    ("verifier", "grounding"),
    [
        (grounded(), {"grounded": True, "unsupported_claims": []}),
        # The check failed: the regeneration was only meant to fix the marker.
        (OutputParserException("resposta inválida"), None),
    ],
)
async def test_a_failed_regeneration_removes_dangling_markers(
    retriever: FakeRetriever, verifier: GroundingCheck | Exception, grounding: dict[str, Any] | None
) -> None:
    context = make_context(
        retriever,
        grader=Script(relevant(1)).runnable,
        verifier=Script(verifier).runnable,
        answerer=Script(AIMessage("Regra [1] e outra [7]."), ValueError("indisponível")).runnable,
    )

    state, _ = await run(context, "Qual o prazo para desistir?")

    assert state["generation_count"] == 1
    assert state["answer"] == "Regra [1] e outra."
    assert state["messages"][-1].text == "Regra [1] e outra."
    assert state["grounding"] == grounding


async def test_transient_generation_errors_are_retried(
    retriever: FakeRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        graph_module,
        "GENERATION_RETRY",
        RetryPolicy(max_attempts=2, initial_interval=0.0, jitter=False, retry_on=TRANSIENT_ERRORS),
    )
    answerer = Script(httpx.ConnectError("conexão recusada"), AIMessage("Resposta [1]."))
    context = make_context(retriever, answerer=answerer.runnable)

    state, _ = await run(context, "Qual o prazo para desistir?")

    assert state["answer"] == "Resposta [1]."
    assert len(answerer.inputs) == 2


async def test_programming_errors_in_generation_propagate(retriever: FakeRetriever) -> None:
    context = make_context(retriever, answerer=Script(TypeError("bug")).runnable)

    with pytest.raises(TypeError, match="bug"):
        await run(context, "Qual o prazo para desistir?")


async def test_grading_and_verification_can_be_switched_off(retriever: FakeRetriever) -> None:
    grader, verifier = Script(relevant()), Script(grounded())
    context = make_context(
        retriever,
        grader=grader.runnable,
        verifier=verifier.runnable,
        grading=False,
        verification=False,
        context_k=2,
    )

    state, events = await run(context, "Qual o prazo para desistir?")

    assert stages(events) == ["analyze", "retrieve", "generate"]
    assert len(state["context"]) == 2
    assert state["grounding"] is None
    assert grader.inputs == []
    assert verifier.inputs == []


async def test_the_answer_prompt_carries_the_corpus_date(retriever: FakeRetriever) -> None:
    answerer = Script(AIMessage("Resposta [1]."))

    await run(make_context(retriever, answerer=answerer.runnable), "Qual o prazo?")

    system = answerer.inputs[0][0].text
    assert "vigente em 30/10/2026" in system


async def test_user_filter_restricts_search_and_cited_articles(retriever: FakeRetriever) -> None:
    context = make_context(retriever, analyzer=Script(analysis(laws=["CDC"])).runnable)

    state, _ = await run(context, "O art. 49 do CDC e o art. 58 da CLT", laws=["CLT"])

    assert state["article_refs"] == ["CLT:58"]
    assert all(search["laws"] == [LawId.CLT] for search in retriever.searches)
    assert {passage["law_id"] for passage in state["context"]} == {"CLT"}


async def test_threads_keep_the_conversation_between_turns(retriever: FakeRetriever) -> None:
    graph = build_graph(InMemorySaver())
    analyzer = Script(analysis())
    context = make_context(
        retriever,
        analyzer=analyzer.runnable,
        grader=Script(relevant(), relevant(1)).runnable,
        answerer=answer_model("Primeira resposta [1].", "Segunda resposta [1]."),
    )

    first, _ = await run(context, "Primeira pergunta?", graph=graph)
    second, _ = await run(context, "Segunda pergunta?", graph=graph)

    assert first["rewrite_count"] == 1
    assert second["rewrite_count"] == 0
    assert [m.text for m in second["messages"]] == [
        "Primeira pergunta?",
        "Primeira resposta [1].",
        "Segunda pergunta?",
        "Segunda resposta [1].",
    ]
    assert "Assistente: Primeira resposta." in analyzer.inputs[1][-1].text


async def test_sub_queries_belong_to_their_turn(retriever: FakeRetriever) -> None:
    graph = build_graph(InMemorySaver())
    analyzer = Script(analysis(query="regra", sub_queries=["exceção"]), analysis(query="outra"))
    context = make_context(
        retriever, analyzer=analyzer.runnable, answerer=answer_model("Uma [1].", "Outra [1].")
    )

    await run(context, "Primeira pergunta?", graph=graph)
    retriever.searches.clear()
    second, _ = await run(context, "Segunda pergunta?", graph=graph)

    assert [search["query"] for search in retriever.searches] == ["outra"]
    assert second["sub_queries"] == []


async def test_threads_are_isolated(retriever: FakeRetriever) -> None:
    graph = build_graph(InMemorySaver())
    context = make_context(retriever, answerer=answer_model("Uma [1].", "Outra [1]."))

    await run(context, "Pergunta A?", graph=graph, thread="a")
    other, _ = await run(context, "Pergunta B?", graph=graph, thread="b")

    assert [m.text for m in other["messages"]] == ["Pergunta B?", "Outra [1]."]


def test_graph_without_checkpointer_compiles_with_every_node() -> None:
    graph = build_graph()

    expected = {"analyze", "retrieve", "grade", "rewrite", "generate", "verify", "respond"}
    assert expected <= set(graph.get_graph().nodes)
