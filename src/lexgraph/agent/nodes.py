"""Nodes and routing rules of the agent graph.

Each node reads the checkpointed state plus the run-scoped :class:`AgentContext` (retriever,
model roles and options) and returns a partial state update. Progress reaches clients as
custom stream events written through ``runtime.stream_writer``:

- ``{"event": "status", "stage": ..., "message": ..., "detail": {...}}`` when a step starts;
- ``{"event": "sources", "sources": [...]}`` with the passages given to the answer model;
- ``{"event": "reset", "reason": ...}`` when a streamed answer is about to be rewritten.

The auxiliary roles fail open: a failed analysis falls back to a heuristic, failed grading
keeps every passage, a failed rewrite searches the question itself and a failed check
leaves the answer unverified. Generation has no degraded mode, so the graph retries it and
then ends with :func:`generation_failed`.
"""

import asyncio
import itertools
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_config
from langgraph.errors import NodeError
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from lexgraph.agent import prompts
from lexgraph.agent.citations import (
    dangling_indices,
    drop_dangling_markers,
    normalize_markers,
    passages_to_sources,
)
from lexgraph.agent.context import AgentContext
from lexgraph.agent.llm import LLM_ERRORS
from lexgraph.agent.references import (
    ArticleMention,
    article_ids,
    detect_laws,
    find_article_mentions,
    normalize_article_number,
)
from lexgraph.agent.schemas import ArticleRef, QueryAnalysis
from lexgraph.agent.state import AgentState
from lexgraph.domain import LawId, Passage, split_article_id
from lexgraph.schemas import Grounding, Route, Stage

logger = logging.getLogger(__name__)

CORPUS_DATE_KEY = "corpus_date"
"""Key of ``config["configurable"]`` holding the date of the corpus texts."""

MAX_EXPLICIT_ARTICLES = 3
CHUNKS_PER_EXPLICIT_ARTICLE = 2
MAX_SUB_QUERIES = 2

_INTENT_ROUTES: dict[str, Route] = {"smalltalk": "smalltalk", "out_of_scope": "out_of_scope"}

type Detail = str | int | float | bool | list[str] | None


class GraphState(AgentState):
    """``AgentState`` plus the law filter chosen by the user, kept apart from the agent's.

    ``law_filter`` arrives holding the user's filter and the analysis may replace it with
    laws picked by the agent; ``user_laws`` remembers the user's choice, which later steps
    must never widen.
    """

    user_laws: list[str]


# --- helpers ------------------------------------------------------------------------------


def _status(runtime: Runtime[AgentContext], stage: Stage, message: str, **detail: Detail) -> None:
    runtime.stream_writer({"event": "status", "stage": stage, "message": message, "detail": detail})


def _passages(items: Sequence[Mapping[str, Any]]) -> list[Passage]:
    return [Passage.model_validate(item) for item in items]


def _dump(passages: Sequence[Passage]) -> list[dict[str, Any]]:
    return [passage.model_dump(mode="json") for passage in passages]


def _law_codes(codes: Sequence[str]) -> list[str]:
    """Valid law codes from ``codes``, in catalog order and without repeats."""
    wanted = set(codes)
    return [law_id.value for law_id in LawId if law_id.value in wanted]


def previous_turns(messages: Sequence[BaseMessage], turns: int) -> list[BaseMessage]:
    """The last ``turns`` question-answer pairs before the current question, oldest first.

    A question left without an answer (its run failed) is skipped, so the history always
    alternates between user and assistant, as some providers require.
    """
    if turns <= 0:
        return []
    current = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=len(messages),
    )
    exchanges = [
        (question, answer)
        for question, answer in itertools.pairwise(messages[:current])
        if isinstance(question, HumanMessage) and isinstance(answer, AIMessage)
    ]
    return [message for exchange in exchanges[-turns:] for message in exchange]


def heuristic_analysis(question: str) -> QueryAnalysis:
    """Analysis used when the analyzer model fails: search the question as written."""
    return QueryAnalysis.model_validate(
        {
            "intent": "legal_question",
            "standalone_question": question,
            "search_query": question,
            "sub_queries": [],
            "laws": [law_id.value for law_id in detect_laws(question)],
            "articles": [],
        }
    )


def _mention(ref: ArticleRef) -> ArticleMention | None:
    number = normalize_article_number(ref.article)
    if number is None:
        return None
    return ArticleMention(LawId(ref.law) if ref.law else None, number)


def _said(number: str, text: str) -> bool:
    """Whether the digits of article ``number`` ("59-A" → 59) are written in ``text``."""
    digits = re.match(r"\d+", number)
    return digits is not None and re.search(rf"(?<!\d){digits[0]}(?!\d)", text) is not None


def _explicit_refs(
    question: str, analysis: QueryAnalysis, laws: Sequence[str], history: str = ""
) -> list[str]:
    """Article ids cited in the question, found by pattern or by the analyzer.

    The analyzer's references count only when their number is written in the question or in
    the ``history``: models sometimes add articles nobody cited, and a cited article is
    always fetched and kept. A reference without a law gets the only law named in the
    question (or the only law being searched); references outside the user's filter are
    dropped later by the caller.
    """
    said = f"{question}\n{history}"
    mentions = find_article_mentions(question)
    mentions += [
        mention
        for ref in analysis.articles
        if (mention := _mention(ref)) and _said(mention.number, said)
    ]
    candidates = detect_laws(question) or [LawId(code) for code in laws]
    default_law = candidates[0] if len(candidates) == 1 else None
    return article_ids(mentions, default_law=default_law)


def _select_chunks(chunks: Sequence[Passage], hits: Sequence[Passage]) -> list[Passage]:
    """Slices of a cited article worth sending: those the search also found, else the first.

    Long articles (CF art. 5º has a dozen slices) would crowd out everything else.
    """
    if len(chunks) <= CHUNKS_PER_EXPLICIT_ARTICLE:
        return list(chunks)
    rank = {hit.chunk_id: position for position, hit in enumerate(hits)}
    best = sorted(
        chunks, key=lambda chunk: (rank.get(chunk.chunk_id, len(rank)), chunk.chunk_index)
    )
    return sorted(best[:CHUNKS_PER_EXPLICIT_ARTICLE], key=lambda chunk: chunk.chunk_index)


def _merge(explicit: Sequence[Passage], hits: Sequence[Passage], limit: int) -> list[Passage]:
    """Cited passages first, then search hits, without repeats (hits keep their scores)."""
    hit_by_id = {hit.chunk_id: hit for hit in hits}
    merged: dict[str, Passage] = {}
    for passage in [*explicit, *hits]:
        merged.setdefault(passage.chunk_id, hit_by_id.get(passage.chunk_id, passage))
    return list(merged.values())[:limit]


def interleave(*rankings: Sequence[Passage], limit: int) -> list[Passage]:
    """Alternate result lists rank by rank, earlier lists first, without repeats."""
    merged: dict[str, Passage] = {}
    for row in itertools.zip_longest(*rankings):
        for passage in row:
            if passage is not None:
                merged.setdefault(passage.chunk_id, passage)
    return list(merged.values())[:limit]


async def _search_laws(
    context: AgentContext, query: str, laws: Sequence[str], *, strict: bool
) -> list[Passage]:
    """Search ``laws``: together when the filter is ``strict``, else one by one, interleaved
    with a search across all laws.

    Laws picked by the agent are a preference. Searched alone, each one keeps a share of the
    top ranks instead of being crowded out by a law with many matching articles (the dozens
    of CLT articles on vacations would leave no room for CF art. 7º, XVII), and the search
    across all laws still finds a law the agent missed.
    """
    if strict or not laws:
        return await _search(context, query, laws)
    rankings = [await _search(context, query, [law]) for law in laws]
    rankings.append(await _search(context, query, []))
    return interleave(*rankings, limit=context.options.top_k)


async def _search(context: AgentContext, query: str, laws: Sequence[str]) -> list[Passage]:
    return await asyncio.to_thread(
        context.retriever.search,
        query,
        k=context.options.top_k,
        laws=[LawId(code) for code in laws] or None,
        rerank=context.options.rerank,
    )


def _grounding(claims: Sequence[str] | None) -> dict[str, Any] | None:
    """The grounding stored in the state; ``None`` (unverified) when the check did not run."""
    if claims is None:
        return None
    return Grounding(grounded=not claims, unsupported_claims=list(claims)).model_dump()


def _without_dangling_markers(state: GraphState, claims: Sequence[str] | None) -> dict[str, Any]:
    """Final update for ``state["answer"]`` without its markers that point at no source.

    ``claims`` are the problems found in the answer (``None`` when it was never checked); the
    ones raised by the removed markers are dropped, so the grounding describes what users see.
    """
    n_sources = len(state["context"])
    removed = {
        prompts.dangling_marker_claim(n) for n in dangling_indices(state["answer"], n_sources)
    }
    remaining = None if claims is None else [claim for claim in claims if claim not in removed]
    answer = drop_dangling_markers(state["answer"], n_sources)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer, id=state["answer_message_id"])],
        "grounding": _grounding(remaining),
        "feedback": remaining or [],
    }


# --- nodes --------------------------------------------------------------------------------


async def analyze(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Classify the message, make it standalone, plan the search and spot cited articles."""
    _status(runtime, "analyze", prompts.STATUS_ANALYZE)
    context = runtime.context
    question = state["question"]
    user_laws = _law_codes(state["law_filter"])
    history = previous_turns(state["messages"], context.options.history_turns)
    try:
        analysis = await context.analyzer.ainvoke(prompts.analyzer_messages(question, history))
    except LLM_ERRORS:
        logger.warning("query analysis failed; searching the question as written", exc_info=True)
        analysis = heuristic_analysis(question)
    laws = user_laws or _law_codes(analysis.laws)
    refs = _explicit_refs(
        question, analysis, laws, "\n".join(str(message.content) for message in history)
    )
    if user_laws:
        refs = [ref for ref in refs if split_article_id(ref)[0].value in user_laws]
    # A cited article makes the message a legal question whatever the analyzer said.
    route = "answered" if refs else _INTENT_ROUTES.get(analysis.intent, "answered")
    standalone = analysis.standalone_question.strip() or question
    return {
        "standalone_question": standalone,
        "search_query": analysis.search_query.strip() or standalone,
        "sub_queries": [q.strip() for q in analysis.sub_queries if q.strip()][:MAX_SUB_QUERIES],
        "law_filter": laws,
        "user_laws": user_laws,
        "article_refs": refs,
        "route": route,
    }


async def retrieve(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Fetch the cited articles, then search the laws in play.

    The user's law filter is strict. Laws picked by the agent are only a preference: dense
    and hybrid search fill every slot inside any filter, however poor the match, so a wrong
    pick (a takedown question sent to the CDC) would hide the right law. The search across
    every law therefore runs too, and its hits alternate with the filtered ones; when the
    pick is right, both searches mostly agree and little is lost.

    A question with several legal points also searches the analyzer's sub-queries (the
    exception next to the rule, the other law of a cross-law question), and their results
    alternate with the main query's, so each point reaches the grader. A rewrite drops them.
    """
    context = runtime.context
    query, laws, refs = state["search_query"], state["law_filter"], state["article_refs"]
    queries = [query, *state["sub_queries"]]
    _status(
        runtime,
        "retrieve",
        prompts.STATUS_RETRIEVE,
        query=query,
        laws=laws,
        articles=refs,
        sub_queries=state["sub_queries"],
    )
    cited = [
        await asyncio.to_thread(context.retriever.get_article, article_id)
        for article_id in refs[:MAX_EXPLICIT_ARTICLES]
    ]
    rankings = [
        await _search_laws(context, q, laws, strict=bool(state["user_laws"])) for q in queries
    ]
    hits = interleave(*rankings, limit=context.options.top_k)
    explicit = [passage for chunks in cited for passage in _select_chunks(chunks, hits)]
    documents = _merge(explicit, hits, limit=max(context.options.top_k, len(explicit)))
    return {
        "documents": _dump(documents),
        "context": _dump(documents[: context.options.context_k]),
    }


async def grade(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Keep the relevant passages; passages of explicitly cited articles are always kept."""
    context = runtime.context
    documents = _passages(state["documents"])
    _status(runtime, "grade", prompts.STATUS_GRADE, passages=len(documents))
    try:
        verdict = await context.grader.ainvoke(
            prompts.grader_messages(state["standalone_question"], documents)
        )
        relevant = _relevant_numbers(verdict.relevant, documents)
    except LLM_ERRORS:
        logger.warning("grading failed; keeping every passage", exc_info=True)
        relevant = set(range(1, len(documents) + 1))
    cited = set(state["article_refs"])
    kept = [
        passage
        for number, passage in enumerate(documents, start=1)
        if number in relevant or passage.article_id in cited
    ]
    return {"context": _dump(kept[: context.options.context_k])}


def _relevant_numbers(numbers: Sequence[int], documents: Sequence[Passage]) -> set[int]:
    """Passage numbers of the grader's verdict.

    Models sometimes give the article number instead of the passage number (143 for
    ``[2] CLT, art. 143``): a number that matches no passage is read as the passages of that
    article, if any.
    """
    resolved: set[int] = set()
    for number in numbers:
        if 1 <= number <= len(documents):
            resolved.add(number)
            continue
        same_article = [
            index
            for index, passage in enumerate(documents, start=1)
            if passage.article == str(number)
        ]
        if same_article:
            logger.info("grader gave article %d instead of its passage number", number)
        resolved.update(same_article)
    return resolved


async def rewrite(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Write a new search query (and laws) after a search that found nothing useful.

    The new query is searched alone: the sub-queries found nothing useful either, and
    searching them again would bring back the passages just discarded.
    """
    attempt = state["rewrite_count"] + 1
    _status(runtime, "rewrite", prompts.STATUS_REWRITE, attempt=attempt)
    question, user_laws = state["standalone_question"], state["user_laws"]
    try:
        proposal = await runtime.context.rewriter.ainvoke(
            prompts.rewriter_messages(
                question, state["search_query"], state["law_filter"], _passages(state["documents"])
            )
        )
        query, laws = proposal.search_query.strip() or question, _law_codes(proposal.laws)
    except LLM_ERRORS:
        logger.warning("query rewrite failed; searching the question itself", exc_info=True)
        query, laws = question, []
    return {
        "search_query": query,
        "sub_queries": [],
        "law_filter": user_laws or laws,
        "rewrite_count": attempt,
    }


async def generate(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Write the answer from the selected passages, streaming it token by token.

    A regeneration reuses the id of the first answer, so ``add_messages`` replaces it in the
    conversation instead of appending a second answer.
    """
    attempt = state["generation_count"]
    if attempt == 0:
        sources = passages_to_sources(state["context"])
        runtime.stream_writer(
            {"event": "sources", "sources": [source.model_dump(mode="json") for source in sources]}
        )
        message = prompts.STATUS_GENERATE
    else:
        runtime.stream_writer({"event": "reset", "reason": prompts.RESET_REGENERATION})
        message = prompts.STATUS_REGENERATE
    passages = _passages(state["context"])
    _status(runtime, "generate", message, sources=len(passages), attempt=attempt + 1)
    options = runtime.context.options
    prompt = prompts.answer_messages(
        question=state["question"],
        standalone_question=state["standalone_question"],
        passages=passages,
        history=previous_turns(state["messages"], options.history_turns),
        feedback=state["feedback"] if attempt else (),
        corpus_date=get_config().get("configurable", {}).get(CORPUS_DATE_KEY),
    )
    reply = await runtime.context.answerer.ainvoke(prompt)
    answer = normalize_markers(reply.text.strip())
    if not options.verification:
        # Without the check (which reports them first), markers without a source are dropped.
        answer = drop_dangling_markers(answer, len(passages))
    message_id = state["answer_message_id"] or reply.id or uuid.uuid4().hex
    return {
        "messages": [AIMessage(content=answer, id=message_id)],
        "answer": answer,
        "answer_message_id": message_id,
        "generation_count": attempt + 1,
    }


async def verify(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Check that every legal claim of the answer is supported by its sources.

    Markers that point at no source are found without the model, so they become feedback
    for a regeneration even when the check fails (the answer then stays unverified).
    """
    _status(runtime, "verify", prompts.STATUS_VERIFY)
    passages = _passages(state["context"])
    answer = state["answer"]
    dangling = [prompts.dangling_marker_claim(n) for n in dangling_indices(answer, len(passages))]
    claims: list[str] | None
    try:
        check = await runtime.context.verifier.ainvoke(prompts.verifier_messages(answer, passages))
    except LLM_ERRORS:
        logger.warning("grounding check failed; the answer stays unverified", exc_info=True)
        claims = None
    else:
        claims = [claim.strip() for claim in check.unsupported_claims if claim.strip()] + dangling
    if dangling and not _within_regeneration_budget(state, runtime):
        # Last pass: markers that point at no source are removed instead of shown to users.
        return _without_dangling_markers(state, claims)
    return {"grounding": _grounding(claims), "feedback": dangling if claims is None else claims}


def respond(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Fixed reply for small talk, out-of-scope messages and questions without sources."""
    _status(runtime, "respond", prompts.STATUS_RESPOND)
    route: Route = (
        state["route"] if state["route"] in ("smalltalk", "out_of_scope") else "no_context"
    )
    reply = prompts.templated_reply(route, user_laws=state["user_laws"])
    return {"messages": [AIMessage(content=reply)], "answer": reply, "route": route, "context": []}


def generation_failed(state: GraphState, error: NodeError) -> Command[str]:
    """End the run politely when generation still fails after its retries.

    A failed regeneration keeps the first answer (and its failed check), without the markers
    that point at no source, which the regeneration was meant to fix. Without any answer,
    the user gets a fixed message, with the passages found still listed as sources, and the
    route becomes ``unavailable`` so that clients can tell the outage from an answer. Errors
    that are not model failures are bugs and propagate.
    """
    if not isinstance(error.error, LLM_ERRORS):
        raise error.error
    logger.error("answer generation failed", exc_info=error.error)
    if state["generation_count"] > 0:
        if not dangling_indices(state["answer"], len(state["context"])):
            return Command(goto=END)
        grounding = state["grounding"]
        claims = None if grounding is None else grounding["unsupported_claims"]
        return Command(update=_without_dangling_markers(state, claims), goto=END)
    reply = prompts.GENERATION_FAILED_REPLY
    route: Route = "unavailable"
    return Command(
        update={
            "messages": [AIMessage(content=reply)],
            "answer": reply,
            "grounding": None,
            "route": route,
        },
        goto=END,
    )


# --- routing ------------------------------------------------------------------------------


def route_after_analyze(state: GraphState) -> str:
    """``respond`` to small talk and out-of-scope messages, ``retrieve`` otherwise."""
    return "respond" if state["route"] in ("smalltalk", "out_of_scope") else "retrieve"


def _rewrite_or_give_up(state: GraphState, runtime: Runtime[AgentContext]) -> str:
    return "rewrite" if state["rewrite_count"] < runtime.context.options.max_rewrites else "respond"


def route_after_retrieve(state: GraphState, runtime: Runtime[AgentContext]) -> str:
    """``grade`` (or ``generate`` with grading off); with nothing found, rewrite or give up."""
    if not state["documents"]:
        return _rewrite_or_give_up(state, runtime)
    return "grade" if runtime.context.options.grading else "generate"


def route_after_grade(state: GraphState, runtime: Runtime[AgentContext]) -> str:
    """``generate`` from the relevant passages; with none left, rewrite or give up."""
    return "generate" if state["context"] else _rewrite_or_give_up(state, runtime)


def route_after_generate(state: GraphState, runtime: Runtime[AgentContext]) -> str:
    """``verify`` the answer when verification is on, otherwise finish."""
    return "verify" if runtime.context.options.verification else END


def route_after_verify(state: GraphState, runtime: Runtime[AgentContext]) -> str:
    """Regenerate an answer with problems to fix while the regeneration budget lasts.

    The problems are the ``feedback`` of the check: its unsupported claims, or only the
    markers that point at no source when the check itself failed.
    """
    feedback = state["feedback"]
    return "generate" if feedback and _within_regeneration_budget(state, runtime) else END


def _within_regeneration_budget(state: GraphState, runtime: Runtime[AgentContext]) -> bool:
    return state["generation_count"] <= runtime.context.options.max_regenerations
