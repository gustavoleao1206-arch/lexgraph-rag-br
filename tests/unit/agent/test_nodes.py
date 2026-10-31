"""Graph nodes called one at a time, with fake model roles and an in-memory retriever."""

from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import NodeError
from langgraph.graph import END

from lexgraph.agent import nodes, prompts
from lexgraph.domain import LawId, Passage
from tests.unit.agent.fakes import (
    CDC_49,
    CF_7,
    CLT_58,
    LGPD_18,
    FakeRetriever,
    Script,
    analysis,
    dumped,
    grounded,
    make_context,
    make_passage,
    make_runtime,
    make_state,
    relevant,
    rewritten,
    unsupported,
)

MODEL_DOWN = OutputParserException("resposta inválida")


def statuses(events: list[Any]) -> list[str]:
    return [event["stage"] for event in events if event["event"] == "status"]


def chunk_ids(passages: list[dict[str, Any]]) -> list[str]:
    return [passage["chunk_id"] for passage in passages]


# --- analyze --------------------------------------------------------------------------------


async def test_analyze_applies_the_analysis(retriever: FakeRetriever) -> None:
    analyzer = Script(
        analysis(
            query="direito de arrependimento prazo",
            standalone="Posso desistir de uma compra online?",
            laws=["CDC"],
            articles=[{"law": "CDC", "article": "49º"}],
            sub_queries=["devolução dos valores pagos", " ", "a", "b"],
        )
    )
    runtime, events = make_runtime(make_context(retriever, analyzer=analyzer.runnable))

    # "o 49" escapes the patterns; the analyzer's reference holds because the number is there.
    question = "Posso desistir de uma compra online? Li algo sobre o 49."
    update = await nodes.analyze(make_state(question), runtime)

    assert update == {
        "standalone_question": "Posso desistir de uma compra online?",
        "search_query": "direito de arrependimento prazo",
        "sub_queries": ["devolução dos valores pagos", "a"],
        "law_filter": ["CDC"],
        "user_laws": [],
        "article_refs": ["CDC:49"],
        "route": "answered",
    }
    assert statuses(events) == ["analyze"]


async def test_analyze_unites_pattern_and_model_references(retriever: FakeRetriever) -> None:
    analyzer = Script(
        analysis(articles=[{"law": "CF", "article": "7"}, {"law": None, "article": "x"}])
    )
    runtime, _ = make_runtime(make_context(retriever, analyzer=analyzer.runnable))

    update = await nodes.analyze(make_state("O art. 482 da CLT e o art. 7º da CF"), runtime)

    assert update["article_refs"] == ["CLT:482", "CF:7"]


async def test_analyze_ignores_model_references_nobody_wrote(retriever: FakeRetriever) -> None:
    articles = [{"law": "CLT", "article": "129"}, {"law": "CLT", "article": "59-A"}]
    analyzer = Script(analysis(articles=articles), analysis(articles=articles))
    runtime, _ = make_runtime(make_context(retriever, analyzer=analyzer.runnable, history_turns=1))

    invented = await nodes.analyze(make_state("Posso dividir minhas férias?"), runtime)
    messages = [
        HumanMessage("Fale da jornada 12x36."),
        AIMessage("O art. 59-A da CLT permite [1]."),
        HumanMessage("E esse artigo vale para domésticos?"),
    ]
    state = make_state("E esse artigo vale para domésticos?", messages=messages)
    from_history = await nodes.analyze(state, runtime)

    assert invented["article_refs"] == []
    assert from_history["article_refs"] == ["CLT:59-A"]


async def test_analyze_gives_unnamed_articles_the_only_law_in_play(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(
        make_context(retriever, analyzer=Script(analysis(laws=["CLT"])).runnable)
    )

    update = await nodes.analyze(make_state("O que diz o art. 59-A?"), runtime)

    assert update["article_refs"] == ["CLT:59-A"]


async def test_analyze_keeps_the_user_filter(retriever: FakeRetriever) -> None:
    analyzer = Script(analysis(laws=["CF", "CLT"]))
    runtime, _ = make_runtime(make_context(retriever, analyzer=analyzer.runnable))
    state = make_state("O art. 7º da CF e o art. 58 da CLT", law_filter=["CLT"])

    update = await nodes.analyze(state, runtime)

    assert update["law_filter"] == ["CLT"]
    assert update["user_laws"] == ["CLT"]
    assert update["article_refs"] == ["CLT:58"]


async def test_analyze_routes_small_talk_and_out_of_scope(retriever: FakeRetriever) -> None:
    for intent in ("smalltalk", "out_of_scope"):
        runtime, _ = make_runtime(
            make_context(retriever, analyzer=Script(analysis(intent)).runnable)
        )

        update = await nodes.analyze(make_state("Oi!"), runtime)

        assert update["route"] == intent


async def test_a_cited_article_makes_the_message_a_legal_question(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(
        make_context(retriever, analyzer=Script(analysis("smalltalk")).runnable)
    )

    update = await nodes.analyze(make_state("Obrigado! E o art. 49 do CDC?"), runtime)

    assert update["route"] == "answered"


async def test_analyze_falls_back_to_a_heuristic_when_the_model_fails(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(make_context(retriever, analyzer=Script(MODEL_DOWN).runnable))
    question = "O que diz o art. 482 da CLT?"

    update = await nodes.analyze(make_state(question), runtime)

    assert update["route"] == "answered"
    assert update["search_query"] == question
    assert update["standalone_question"] == question
    assert update["law_filter"] == ["CLT"]
    assert update["article_refs"] == ["CLT:482"]


async def test_analyze_does_not_hide_programming_errors(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, analyzer=Script(TypeError("bug")).runnable))

    with pytest.raises(TypeError):
        await nodes.analyze(make_state(), runtime)


async def test_analyze_sends_previous_turns_to_the_model(retriever: FakeRetriever) -> None:
    analyzer = Script(analysis())
    runtime, _ = make_runtime(make_context(retriever, analyzer=analyzer.runnable, history_turns=1))
    messages = [
        HumanMessage("Primeira pergunta"),
        AIMessage("Primeira resposta"),
        HumanMessage("Segunda pergunta"),
        AIMessage("Segunda resposta [1]."),
        HumanMessage("E agora?"),
    ]

    await nodes.analyze(make_state("E agora?", messages=messages), runtime)

    prompt = analyzer.inputs[0][-1].text
    assert "Usuário: Segunda pergunta\nAssistente: Segunda resposta." in prompt
    assert "Primeira" not in prompt


# --- retrieve -------------------------------------------------------------------------------


async def test_retrieve_puts_cited_articles_first(retriever: FakeRetriever) -> None:
    runtime, events = make_runtime(make_context(retriever, top_k=3, context_k=2))
    state = make_state(search_query="jornada", article_refs=["LGPD:18"])

    update = await nodes.retrieve(state, runtime)

    assert chunk_ids(update["documents"]) == ["LGPD:18#0", "CDC:49#0", "CLT:58#0"]
    assert chunk_ids(update["context"]) == ["LGPD:18#0", "CDC:49#0"]
    assert retriever.lookups == ["LGPD:18"]
    assert events[0]["detail"] == {
        "query": "jornada",
        "laws": [],
        "articles": ["LGPD:18"],
        "sub_queries": [],
    }


async def test_retrieve_keeps_the_hit_version_of_a_cited_passage() -> None:
    hit = CDC_49.model_copy(update={"score": 0.99, "rerank_score": 4.2})
    retriever = FakeRetriever([CDC_49], by_query={"consulta": [hit]})
    runtime, _ = make_runtime(make_context(retriever))

    update = await nodes.retrieve(
        make_state(search_query="consulta", article_refs=["CDC:49"]), runtime
    )

    assert len(update["documents"]) == 1
    assert update["documents"][0]["rerank_score"] == 4.2


async def test_long_cited_articles_send_the_slices_the_search_found() -> None:
    xvii = CF_7[2]
    retriever = FakeRetriever([*CF_7, CDC_49], by_query={"férias": [xvii, CDC_49]})
    runtime, _ = make_runtime(make_context(retriever))

    update = await nodes.retrieve(make_state(search_query="férias", article_refs=["CF:7"]), runtime)

    assert chunk_ids(update["documents"]) == ["CF:7#0", "CF:7#2", "CDC:49#0"]


async def test_sub_queries_bring_each_legal_point_to_the_grader() -> None:
    retriever = FakeRetriever(
        [CDC_49, CLT_58, LGPD_18],
        by_query={"regra": [CLT_58, CDC_49], "exceção": [LGPD_18]},
    )
    runtime, _ = make_runtime(make_context(retriever, top_k=3))
    state = make_state(search_query="regra", sub_queries=["exceção"])

    update = await nodes.retrieve(state, runtime)

    assert [search["query"] for search in retriever.searches] == ["regra", "exceção"]
    # Rankings alternate rank by rank, so the sub-query's best hit is not pushed out.
    assert chunk_ids(update["documents"]) == ["CLT:58#0", "LGPD:18#0", "CDC:49#0"]


async def test_laws_picked_by_the_agent_are_only_a_preference(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, top_k=4))
    state = make_state(search_query="consulta", law_filter=["CF"])

    update = await nodes.retrieve(state, runtime)

    assert [search["laws"] for search in retriever.searches] == [[LawId.CF], None]
    # Filtered and unfiltered hits alternate, the filtered first, without repeats.
    assert chunk_ids(update["documents"]) == ["CF:7#0", "CDC:49#0", "CF:7#1", "CLT:58#0"]
    assert "law_filter" not in update


async def test_each_law_picked_by_the_agent_gets_a_share_of_the_top_ranks() -> None:
    retriever = FakeRetriever(
        [*CF_7, CLT_58, CDC_49],
        by_query={"férias": [CLT_58, CDC_49, *CF_7]},
    )
    runtime, _ = make_runtime(make_context(retriever, top_k=3))
    state = make_state(search_query="férias", law_filter=["CLT", "CF"])

    update = await nodes.retrieve(state, runtime)

    assert [search["laws"] for search in retriever.searches] == [[LawId.CLT], [LawId.CF], None]
    # The ranking of each law and the one across all laws alternate, without repeats.
    assert chunk_ids(update["documents"]) == ["CLT:58#0", "CF:7#0", "CF:7#1"]


async def test_a_wrong_pick_of_laws_still_finds_the_right_law(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever))
    state = make_state(search_query="consulta", law_filter=["MCI"])

    update = await nodes.retrieve(state, runtime)

    assert [search["laws"] for search in retriever.searches] == [[LawId.MCI], None]
    assert len(update["documents"]) == 6


async def test_retrieve_never_widens_the_user_filter(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever))
    state = make_state(search_query="consulta", law_filter=["MCI"], user_laws=["MCI"])

    update = await nodes.retrieve(state, runtime)

    assert [search["laws"] for search in retriever.searches] == [[LawId.MCI]]
    assert update["documents"] == []


async def test_retrieve_passes_the_search_options(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, top_k=4, rerank=True))
    state = make_state(search_query="consulta", law_filter=["CLT", "CF"], user_laws=["CLT", "CF"])

    await nodes.retrieve(state, runtime)

    assert retriever.searches == [
        {
            "query": "consulta",
            "k": 4,
            "laws": [LawId.CLT, LawId.CF],
            "mode": None,
            "rerank": True,
        }
    ]


async def test_retrieve_fetches_at_most_three_cited_articles(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever))
    refs = ["CDC:49", "CLT:58", "LGPD:18", "CF:7"]

    await nodes.retrieve(make_state(article_refs=refs), runtime)

    assert retriever.lookups == refs[:3]


# --- grade ----------------------------------------------------------------------------------


async def test_grade_keeps_relevant_and_cited_passages(retriever: FakeRetriever) -> None:
    grader = Script(relevant(3, 9))
    runtime, events = make_runtime(make_context(retriever, grader=grader.runnable))
    state = make_state(documents=dumped(CDC_49, CLT_58, LGPD_18, *CF_7), article_refs=["CF:7"])

    update = await nodes.grade(state, runtime)

    assert chunk_ids(update["context"]) == ["LGPD:18#0", "CF:7#0", "CF:7#1", "CF:7#2"]
    assert events[0]["detail"] == {"passages": 6}
    assert "[6] CF, art. 7" in grader.inputs[0][-1].text


async def test_grade_reads_an_article_number_as_its_passages(retriever: FakeRetriever) -> None:
    # "[2] CLT, art. 58" answered as 58; 99 matches neither a passage nor an article.
    grader = Script(relevant(1, 58, 99))
    runtime, _ = make_runtime(make_context(retriever, grader=grader.runnable))

    update = await nodes.grade(make_state(documents=dumped(CDC_49, CLT_58, LGPD_18)), runtime)

    assert chunk_ids(update["context"]) == ["CDC:49#0", "CLT:58#0"]


async def test_grade_respects_the_context_size(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, context_k=2))

    update = await nodes.grade(make_state(documents=dumped(CDC_49, CLT_58, LGPD_18)), runtime)

    assert chunk_ids(update["context"]) == ["CDC:49#0", "CLT:58#0"]


async def test_grade_can_find_nothing_relevant(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, grader=Script(relevant()).runnable))

    update = await nodes.grade(make_state(documents=dumped(CDC_49)), runtime)

    assert update["context"] == []


async def test_grade_fails_open(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, grader=Script(MODEL_DOWN).runnable))

    update = await nodes.grade(make_state(documents=dumped(CDC_49, CLT_58)), runtime)

    assert chunk_ids(update["context"]) == ["CDC:49#0", "CLT:58#0"]


# --- rewrite --------------------------------------------------------------------------------


async def test_rewrite_uses_the_new_query_and_laws(retriever: FakeRetriever) -> None:
    rewriter = Script(rewritten("rescisão por justa causa", "CLT", "CLT"))
    runtime, events = make_runtime(make_context(retriever, rewriter=rewriter.runnable))
    state = make_state(search_query="demitido", sub_queries=["aviso"], documents=dumped(CDC_49))

    update = await nodes.rewrite(state, runtime)

    # The new query is searched alone: the sub-queries found nothing useful either.
    assert update == {
        "search_query": "rescisão por justa causa",
        "sub_queries": [],
        "law_filter": ["CLT"],
        "rewrite_count": 1,
    }
    assert events[0]["detail"] == {"attempt": 1}
    assert "descartados: CDC, art. 49" in rewriter.inputs[0][-1].text


async def test_rewrite_keeps_the_user_filter(retriever: FakeRetriever) -> None:
    rewriter = Script(rewritten("nova consulta", "CDC"))
    runtime, _ = make_runtime(make_context(retriever, rewriter=rewriter.runnable))

    update = await nodes.rewrite(make_state(user_laws=["CLT"], rewrite_count=1), runtime)

    assert update["law_filter"] == ["CLT"]
    assert update["rewrite_count"] == 2


async def test_rewrite_falls_back_to_the_question(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, rewriter=Script(MODEL_DOWN).runnable))
    state = make_state(standalone_question="Pergunta completa?", law_filter=["CDC"])

    update = await nodes.rewrite(state, runtime)

    assert update["search_query"] == "Pergunta completa?"
    assert update["law_filter"] == []


# --- verify ---------------------------------------------------------------------------------


async def test_verify_records_a_grounded_answer(retriever: FakeRetriever) -> None:
    verifier = Script(grounded())
    runtime, events = make_runtime(make_context(retriever, verifier=verifier.runnable))
    state = make_state(answer="O prazo é de 7 dias [1].", context=dumped(CDC_49))

    update = await nodes.verify(state, runtime)

    assert update == {"grounding": {"grounded": True, "unsupported_claims": []}, "feedback": []}
    assert statuses(events) == ["verify"]
    assert "<resposta>\nO prazo é de 7 dias [1].\n</resposta>" in verifier.inputs[0][-1].text


async def test_verify_turns_unsupported_claims_into_feedback(retriever: FakeRetriever) -> None:
    verifier = Script(unsupported(" O prazo é de 30 dias. ", ""))
    runtime, _ = make_runtime(make_context(retriever, verifier=verifier.runnable))
    state = make_state(answer="O prazo é de 30 dias [1] [3].", context=dumped(CDC_49))

    update = await nodes.verify(state, runtime)

    claims = ["O prazo é de 30 dias.", prompts.dangling_marker_claim(3)]
    assert update == {
        "grounding": {"grounded": False, "unsupported_claims": claims},
        "feedback": claims,
    }


async def test_an_answer_that_cites_no_source_is_written_again(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever, verifier=Script(grounded()).runnable))
    state = make_state(answer="Ficam 24 dias de férias.", context=dumped(CLT_58))

    update = await nodes.verify(state, runtime)

    # Not an unsupported claim: the grounding stands, the regeneration is asked to cite.
    assert update == {
        "grounding": {"grounded": True, "unsupported_claims": []},
        "feedback": [prompts.MISSING_CITATIONS],
    }
    assert nodes.route_after_verify(make_state(generation_count=1, **update), runtime) == "generate"
    without_sources = make_state(answer="Os trechos não tratam disso.", context=[])
    assert (await nodes.verify(without_sources, runtime))["feedback"] == []


async def test_verify_leaves_the_answer_unverified_when_the_model_fails(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(make_context(retriever, verifier=Script(MODEL_DOWN).runnable))

    update = await nodes.verify(make_state(answer="Regra [1].", context=dumped(CDC_49)), runtime)

    assert update == {"grounding": None, "feedback": []}


async def test_dangling_markers_become_feedback_even_without_the_model(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(make_context(retriever, verifier=Script(MODEL_DOWN).runnable))

    update = await nodes.verify(make_state(answer="Regra [2].", context=dumped(CDC_49)), runtime)

    # The answer stays unverified, but the regeneration is told about the marker.
    assert update == {"grounding": None, "feedback": [prompts.dangling_marker_claim(2)]}
    assert nodes.route_after_verify(make_state(generation_count=1, **update), runtime) == "generate"


async def test_the_last_pass_removes_dangling_markers(retriever: FakeRetriever) -> None:
    verifier = Script(unsupported("Afirmação sem apoio."))
    runtime, _ = make_runtime(
        make_context(retriever, verifier=verifier.runnable, max_regenerations=0)
    )
    state = make_state(
        answer="Regra [1] e outra [7].",
        context=dumped(CDC_49),
        generation_count=1,
        answer_message_id="resposta-1",
    )

    update = await nodes.verify(state, runtime)

    assert update["answer"] == "Regra [1] e outra."
    assert update["messages"] == [AIMessage("Regra [1] e outra.", id="resposta-1")]
    assert update["grounding"] == {
        "grounded": False,
        "unsupported_claims": ["Afirmação sem apoio."],
    }
    assert update["feedback"] == ["Afirmação sem apoio."]


async def test_the_last_pass_without_the_model_leaves_the_answer_unverified(
    retriever: FakeRetriever,
) -> None:
    runtime, _ = make_runtime(
        make_context(retriever, verifier=Script(MODEL_DOWN).runnable, max_regenerations=0)
    )
    state = make_state(
        answer="Regra [1] e outra [7].",
        context=dumped(CDC_49),
        generation_count=1,
        answer_message_id="resposta-1",
    )

    update = await nodes.verify(state, runtime)

    # Removing the marker fixes the only problem found, but no check confirmed the rest.
    assert update["answer"] == "Regra [1] e outra."
    assert update["messages"] == [AIMessage("Regra [1] e outra.", id="resposta-1")]
    assert update["grounding"] is None
    assert update["feedback"] == []


# --- respond and generation failures --------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "expected"),
    [("smalltalk", "smalltalk"), ("out_of_scope", "out_of_scope"), ("answered", "no_context")],
)
def test_respond_sends_the_fixed_reply(retriever: FakeRetriever, route: str, expected: str) -> None:
    runtime, events = make_runtime(make_context(retriever))

    update = nodes.respond(make_state(route=route, context=dumped(CDC_49)), runtime)

    reply = prompts.templated_reply(expected)  # type: ignore[arg-type]
    assert update["route"] == expected
    assert update["answer"] == reply
    assert update["messages"][0].text == reply
    assert update["context"] == []
    assert statuses(events) == ["respond"]


def test_respond_mentions_the_user_filter(retriever: FakeRetriever) -> None:
    runtime, _ = make_runtime(make_context(retriever))

    update = nodes.respond(make_state(user_laws=["LGPD"]), runtime)

    assert "restrita a: LGPD." in update["answer"]


def test_generation_failure_ends_with_a_polite_reply() -> None:
    command = nodes.generation_failed(make_state(), NodeError("generate", MODEL_DOWN))

    assert command.goto == END
    assert isinstance(command.update, dict)
    assert command.update["answer"] == prompts.GENERATION_FAILED_REPLY
    assert command.update["grounding"] is None
    assert command.update["route"] == "unavailable"


def test_a_failed_regeneration_keeps_the_first_answer() -> None:
    state = make_state(generation_count=1, answer="Primeira resposta [1].", context=dumped(CDC_49))

    command = nodes.generation_failed(state, NodeError("generate", MODEL_DOWN))

    assert command.goto == END
    assert command.update is None


@pytest.mark.parametrize(
    ("grounding", "expected"),
    [
        (
            {
                "grounded": False,
                "unsupported_claims": ["Afirmação sem apoio.", prompts.dangling_marker_claim(7)],
            },
            {"grounded": False, "unsupported_claims": ["Afirmação sem apoio."]},
        ),
        (
            {"grounded": False, "unsupported_claims": [prompts.dangling_marker_claim(7)]},
            {"grounded": True, "unsupported_claims": []},
        ),
        # The check failed: only the marker was reported, and the answer stays unverified.
        (None, None),
    ],
)
def test_a_failed_regeneration_removes_the_dangling_markers_of_the_first_answer(
    grounding: dict[str, Any] | None, expected: dict[str, Any] | None
) -> None:
    state = make_state(
        generation_count=1,
        answer="Primeira [1] e outra [7].",
        answer_message_id="resposta-1",
        context=dumped(CDC_49),
        grounding=grounding,
    )

    command = nodes.generation_failed(state, NodeError("generate", MODEL_DOWN))

    assert command.goto == END
    assert isinstance(command.update, dict)
    assert command.update["answer"] == "Primeira [1] e outra."
    assert command.update["messages"] == [AIMessage("Primeira [1] e outra.", id="resposta-1")]
    assert command.update["grounding"] == expected


def test_generation_failure_does_not_hide_programming_errors() -> None:
    with pytest.raises(KeyError):
        nodes.generation_failed(make_state(), NodeError("generate", KeyError("context")))


# --- routing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "expected"),
    [("answered", "retrieve"), ("smalltalk", "respond"), ("out_of_scope", "respond")],
)
def test_route_after_analyze(route: str, expected: str) -> None:
    assert nodes.route_after_analyze(make_state(route=route)) == expected


@pytest.mark.parametrize(
    ("documents", "rewrites", "options", "expected"),
    [
        ([CDC_49], 0, {}, "grade"),
        ([CDC_49], 0, {"grading": False}, "generate"),
        ([], 0, {}, "rewrite"),
        ([], 1, {}, "respond"),
        ([], 0, {"max_rewrites": 0}, "respond"),
    ],
)
def test_route_after_retrieve(
    retriever: FakeRetriever,
    documents: list[Passage],
    rewrites: int,
    options: dict[str, Any],
    expected: str,
) -> None:
    runtime, _ = make_runtime(make_context(retriever, **options))
    state = make_state(documents=dumped(*documents), rewrite_count=rewrites)

    assert nodes.route_after_retrieve(state, runtime) == expected


@pytest.mark.parametrize(
    ("context", "rewrites", "expected"),
    [([CDC_49], 0, "generate"), ([], 0, "rewrite"), ([], 1, "respond")],
)
def test_route_after_grade(
    retriever: FakeRetriever, context: list[Passage], rewrites: int, expected: str
) -> None:
    runtime, _ = make_runtime(make_context(retriever))
    state = make_state(context=dumped(*context), rewrite_count=rewrites)

    assert nodes.route_after_grade(state, runtime) == expected


@pytest.mark.parametrize(("verification", "expected"), [(True, "verify"), (False, END)])
def test_route_after_generate(retriever: FakeRetriever, verification: bool, expected: str) -> None:
    runtime, _ = make_runtime(make_context(retriever, verification=verification))

    assert nodes.route_after_generate(make_state(), runtime) == expected


@pytest.mark.parametrize(
    ("feedback", "generations", "expected"),
    [([], 1, END), (["x"], 1, "generate"), (["x"], 2, END)],
)
def test_route_after_verify(
    retriever: FakeRetriever, feedback: list[str], generations: int, expected: str
) -> None:
    runtime, _ = make_runtime(make_context(retriever, max_regenerations=1))
    state = make_state(feedback=feedback, generation_count=generations)

    assert nodes.route_after_verify(state, runtime) == expected


# --- helpers --------------------------------------------------------------------------------


def test_previous_turns_take_the_last_exchanges_before_the_question() -> None:
    messages = [
        HumanMessage("p1"),
        AIMessage("r1"),
        HumanMessage("p2"),
        AIMessage("r2"),
        HumanMessage("atual"),
        AIMessage("resposta a ser reescrita"),
    ]

    assert [m.text for m in nodes.previous_turns(messages, 1)] == ["p2", "r2"]
    assert [m.text for m in nodes.previous_turns(messages, 5)] == ["p1", "r1", "p2", "r2"]
    assert nodes.previous_turns(messages, 0) == []
    assert nodes.previous_turns([HumanMessage("primeira")], 4) == []


def test_previous_turns_keep_only_answered_questions() -> None:
    messages = [
        AIMessage("órfã"),
        HumanMessage("p1"),
        AIMessage("r1"),
        HumanMessage("sem resposta: a execução falhou"),
        HumanMessage("p2"),
        AIMessage("r2"),
        HumanMessage("atual"),
    ]

    assert [m.text for m in nodes.previous_turns(messages, 3)] == ["p1", "r1", "p2", "r2"]


def test_heuristic_analysis_searches_the_question() -> None:
    result = nodes.heuristic_analysis("Quais os direitos do titular na LGPD?")

    assert result.intent == "legal_question"
    assert result.search_query == "Quais os direitos do titular na LGPD?"
    assert result.laws == ["LGPD"]


def test_passages_round_trip_through_the_state() -> None:
    passage = make_passage(LawId.MCI, "19")

    assert Passage.model_validate(dumped(passage)[0]) == passage
