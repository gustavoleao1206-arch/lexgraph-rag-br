"""Batch runs over the golden set: retrieval ablation, agent answers and judge verdicts.

Results are JSON Lines files of the record models below, so long runs can be resumed and
reports can be rebuilt without calling any model again.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lexgraph.agent import prompts
from lexgraph.agent.context import AgentContext
from lexgraph.agent.llm import LLM_ERRORS
from lexgraph.agent.nodes import MAX_SUB_QUERIES, heuristic_analysis, interleave
from lexgraph.config import RetrievalMode
from lexgraph.domain import Passage
from lexgraph.eval.golden import GoldenItem, QuestionType
from lexgraph.eval.judge import (
    CorrectnessVerdict,
    GroundednessVerdict,
    Judge,
    TimedRateLimiter,
    shares_family,
    truthfulness,
)
from lexgraph.eval.metrics import article_ranking, retrieval_scores, unit_recall_at_k
from lexgraph.eval.rules import is_refusal
from lexgraph.protocols import ChatService, Retriever
from lexgraph.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    ResetEvent,
    Stage,
    StatusEvent,
    TokenEvent,
)

logger = logging.getLogger(__name__)

UNAVAILABLE_ERROR = "no language model could answer (route 'unavailable')"


class RetrievalConfig(BaseModel):
    """A retriever configuration compared in the ablation.

    With ``analyzed`` the search uses the query written by the agent's analyzer model instead
    of the raw question, which takes one language-model call per question.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    mode: RetrievalMode
    rerank: bool = False
    analyzed: bool = False


DEFAULT_CONFIGS = (
    RetrievalConfig(name="dense", mode="dense"),
    RetrievalConfig(name="sparse", mode="sparse"),
    RetrievalConfig(name="hybrid", mode="hybrid"),
    RetrievalConfig(name="hybrid+rerank", mode="hybrid", rerank=True),
)
ANALYZED_CONFIGS = (
    RetrievalConfig(name="dense+analyzer", mode="dense", analyzed=True),
    RetrievalConfig(name="hybrid+analyzer", mode="hybrid", analyzed=True),
)
"""Opt-in configurations that search the analyzer's queries, as the agent does."""
CONFIGS = (*DEFAULT_CONFIGS, *ANALYZED_CONFIGS)


def unit_recall_metric(k: int) -> str:
    """Name of the unit recall score, recorded only for items with expected units."""
    return f"unit_recall@{k}"


QueryAnalyzer = Callable[[str], list[str]]
"""Turns a question into the search queries the agent would use: the main one first."""


class RetrievalRecord(BaseModel):
    """Article ranking and scores of one question under one configuration."""

    config: str
    item_id: str
    question_type: QuestionType
    k: int
    ranking: list[str]
    scores: dict[str, float]
    latency_ms: float
    query: str | None = Field(
        default=None,
        description="The analyzer's queries joined by ' | '; None when the question was searched.",
    )
    rerank_candidates: int | None = Field(
        default=None,
        description="Passages the cross-encoder scored per search; None without reranking.",
    )


class AnswerRecord(BaseModel):
    """Outcome of asking the agent one golden question.

    ``latency_ms`` and ``ttft_ms`` (time to the first answer token) leave out ``paced_ms``, the
    time the evaluation's rate limiter held the agent's model calls back, so they measure the
    agent rather than the pacing.
    """

    item_id: str
    started_at: str
    latency_ms: float
    ttft_ms: float | None = None
    paced_ms: float = 0.0
    stages: list[Stage] = Field(default_factory=list)
    resets: int = 0
    response: ChatResponse | None = None
    error: str | None = None

    @property
    def result(self) -> ChatResponse | None:
        """The response when the question was answered without error, else ``None``.

        A response with an error (no model could answer) is kept for inspection only.
        """
        return self.response if self.error is None else None


class JudgmentRecord(BaseModel):
    """Judge verdicts and derived scores for one answer."""

    item_id: str
    answer_started_at: str = Field(description="``started_at`` of the judged answer record.")
    judge_model: str
    same_family: bool = Field(
        default=False,
        description="A model of the judge's family took part in producing the answer.",
    )
    refused: bool
    groundedness: GroundednessVerdict | None = None
    correctness: CorrectnessVerdict | None = None
    groundedness_score: float | None = None
    correctness_score: float | None = None
    truthfulness: float | None = None
    error: str | None = None


def read_records[RecordT: BaseModel](path: Path, model: type[RecordT]) -> list[RecordT]:
    """Read a JSON Lines file of ``model`` records.

    An unreadable last line, left by a run interrupted while writing it, is skipped with a
    warning; an unreadable line anywhere else is corruption and raises ``ValidationError``.
    """
    with path.open(encoding="utf-8") as handle:
        lines = [(number, line) for number, line in enumerate(handle, start=1) if line.strip()]
    records: list[RecordT] = []
    for position, (number, line) in enumerate(lines, start=1):
        try:
            records.append(model.model_validate_json(line))
        except ValidationError:
            if position < len(lines):
                raise
            logger.warning("Skipping the unfinished last line %s:%d", path, number)
    return records


def write_records(path: Path, records: Iterable[BaseModel]) -> None:
    """Write ``records`` as JSON Lines, replacing the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{record.model_dump_json()}\n" for record in records)
    path.write_text(text, encoding="utf-8", newline="\n")


def append_record(path: Path, record: BaseModel) -> None:
    """Append one record to a JSON Lines file, creating it when needed.

    A last line without its newline (an interrupted write) is completed when it holds a valid
    record and removed otherwise, so that it never merges with the new record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _finish_last_line(path, type(record))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{record.model_dump_json()}\n")


def _finish_last_line(path: Path, model: type[BaseModel]) -> None:
    with path.open("r+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        content = handle.read()
        start = content.rfind(b"\n") + 1
        try:
            model.model_validate_json(content[start:])
        except ValidationError:
            logger.warning("Removing the unfinished last line of %s", path)
            handle.truncate(start)
        else:
            handle.write(b"\n")


def latest_answers(records: Iterable[AnswerRecord]) -> dict[str, AnswerRecord]:
    """Last record of each item: a resumed run appends retries after failed attempts."""
    return {record.item_id: record for record in records}


def agent_query_analyzer(context: AgentContext) -> QueryAnalyzer:
    """Write search queries with the agent's analyzer model, as its ``analyze`` step does.

    The question is analyzed without conversation history. Only the queries are used (the
    main one and the sub-queries): the analyzer's intent and laws are ignored, so the
    configuration measures the query rewrite alone. A failed analysis falls back, like the
    agent, to the question as written.
    """

    def analyze(question: str) -> list[str]:
        try:
            analysis = context.analyzer.invoke(prompts.analyzer_messages(question, []))
        except LLM_ERRORS:
            logger.warning(
                "Query analysis failed; searching the question as written", exc_info=True
            )
            analysis = heuristic_analysis(question)
        main = analysis.search_query.strip() or question
        subs = [query.strip() for query in analysis.sub_queries if query.strip()]
        return [main, *subs[:MAX_SUB_QUERIES]]

    return analyze


def run_retrieval_eval(
    retriever: Retriever,
    items: Sequence[GoldenItem],
    configs: Sequence[RetrievalConfig] = DEFAULT_CONFIGS,
    k: int = 10,
    *,
    warmup: bool = True,
    analyzer: QueryAnalyzer | None = None,
    rerank_candidates: int | None = None,
) -> list[RetrievalRecord]:
    """Search every in-scope question with every configuration and score the rankings.

    Out-of-scope items have no relevant article and are skipped. With ``warmup`` one
    untimed search per configuration loads lazy models before latencies are measured.
    Configurations with ``analyzed`` need ``analyzer``; their latency excludes the analysis.
    ``rerank_candidates`` is the retriever's setting; the reranked configurations record the
    number of passages it makes the cross-encoder score, ``max(k, rerank_candidates)``.
    """
    if analyzer is None and any(config.analyzed for config in configs):
        msg = "configurations that search the analyzer's queries need a query analyzer"
        raise ValueError(msg)
    scored = [item for item in items if not item.should_refuse]
    records: list[RetrievalRecord] = []
    analyses: dict[str, list[str]] = {}  # one analysis per question, shared by the configs
    for config in configs:
        candidates: int | None = None
        if config.rerank and rerank_candidates is not None:
            candidates = max(k, rerank_candidates)  # what the retriever fetches for reranking
        if warmup and scored:
            _search(retriever, scored[0].question, config, k)
        for item in scored:
            queries = [item.question]
            if analyzer is not None and config.analyzed:
                if item.id not in analyses:
                    analyses[item.id] = analyzer(item.question)
                queries = analyses[item.id]
            started = time.perf_counter()
            # Like the agent: every query is searched and the rankings alternate rank by rank.
            rankings = [_search(retriever, query, config, k) for query in queries]
            passages = interleave(*rankings, limit=k)
            query = " | ".join(queries) if config.analyzed else None
            latency_ms = (time.perf_counter() - started) * 1000
            records.append(
                RetrievalRecord(
                    config=config.name,
                    item_id=item.id,
                    question_type=item.type,
                    k=k,
                    ranking=article_ranking(passage.article_id for passage in passages),
                    scores=_scores(item, passages, k),
                    latency_ms=latency_ms,
                    query=query,
                    rerank_candidates=candidates,
                )
            )
    return records


def _search(retriever: Retriever, query: str, config: RetrievalConfig, k: int) -> list[Passage]:
    return retriever.search(query, k=k, mode=config.mode, rerank=config.rerank)


def _scores(item: GoldenItem, passages: Sequence[Passage], k: int) -> dict[str, float]:
    scores = retrieval_scores(
        article_ranking(passage.article_id for passage in passages), item.relevance, k
    )
    if item.expected_units:
        chunks = [(passage.article_id, passage.units) for passage in passages]
        scores[unit_recall_metric(k)] = unit_recall_at_k(chunks, item.expected_units, k)
    return scores


async def run_answers(
    service: ChatService,
    items: Sequence[GoldenItem],
    out_path: Path,
    *,
    resume: bool = True,
    limit: int | None = None,
    delay_s: float = 0.0,
    limiter: TimedRateLimiter | None = None,
    on_record: Callable[[AnswerRecord], None] | None = None,
) -> list[AnswerRecord]:
    """Ask the agent every golden question and append one :class:`AnswerRecord` per question.

    Each question runs in a fresh conversation thread. With ``resume`` the items already
    answered without error in ``out_path`` are skipped (failed ones are retried); otherwise
    the file is replaced. ``limit`` caps the questions asked in this run and ``delay_s``
    spaces them out to respect provider rate limits. ``limiter`` is the rate limiter that
    paces the agent's model calls: the time spent waiting for it is left out of the timings.
    """
    records: list[AnswerRecord] = []
    for index, item in enumerate(_pending(items, out_path, resume=resume, limit=limit)):
        if index and delay_s > 0:
            await asyncio.sleep(delay_s)
        record = await _answer(service, item, limiter)
        append_record(out_path, record)
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def _pending(
    items: Sequence[GoldenItem], out_path: Path, *, resume: bool, limit: int | None
) -> list[GoldenItem]:
    if resume:
        done = _answered_ids(out_path)
    else:
        out_path.unlink(missing_ok=True)
        done = set()
    pending = [item for item in items if item.id not in done]
    return pending if limit is None else pending[:limit]


def _answered_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    records = latest_answers(read_records(path, AnswerRecord))
    return {item_id for item_id, record in records.items() if record.result is not None}


class _Stopwatch:
    """Time elapsed since creation, split into active time and rate-limiter waits."""

    def __init__(self, limiter: TimedRateLimiter | None) -> None:
        self._limiter = limiter
        self._waited_at_start = self._waited_s()
        self._started = time.perf_counter()

    def paced_ms(self) -> float:
        """Time the model calls spent waiting for the rate limiter so far."""
        return (self._waited_s() - self._waited_at_start) * 1000

    def active_ms(self) -> float:
        """Time elapsed so far, less :meth:`paced_ms`."""
        return (time.perf_counter() - self._started) * 1000 - self.paced_ms()

    def _waited_s(self) -> float:
        return self._limiter.waited_s if self._limiter is not None else 0.0


async def _answer(
    service: ChatService, item: GoldenItem, limiter: TimedRateLimiter | None = None
) -> AnswerRecord:
    """Ask one question; a response from the ``unavailable`` route counts as an error."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    stopwatch = _Stopwatch(limiter)
    ttft_ms: float | None = None
    stages: list[Stage] = []
    resets = 0
    response: ChatResponse | None = None
    error: str | None = None
    try:
        request = ChatRequest(question=item.question, thread_id=f"eval-{item.id}-{uuid4().hex[:8]}")
        async for event in service.stream(request):
            match event:
                case TokenEvent(text=text) if text and ttft_ms is None:
                    ttft_ms = stopwatch.active_ms()
                case StatusEvent(stage=stage) if not stages or stages[-1] != stage:
                    stages.append(stage)
                case ResetEvent():
                    resets += 1
                case FinalEvent(response=final):
                    response = final
                case ErrorEvent(message=message):
                    error = message
    except Exception as exc:
        logger.exception("Question %s failed", item.id)
        error = f"{type(exc).__name__}: {exc}"
    if response is None and error is None:
        error = "the stream ended without a final event"
    elif response is not None and response.route == "unavailable" and error is None:
        error = UNAVAILABLE_ERROR
    return AnswerRecord(
        item_id=item.id,
        started_at=started_at,
        latency_ms=stopwatch.active_ms(),
        ttft_ms=ttft_ms,
        paced_ms=stopwatch.paced_ms(),
        stages=stages,
        resets=resets,
        response=response,
        error=error,
    )


def run_judge(
    judge: Judge,
    items: Sequence[GoldenItem],
    answers: Iterable[AnswerRecord],
    out_path: Path,
    *,
    limit: int | None = None,
    on_record: Callable[[JudgmentRecord], None] | None = None,
) -> list[JudgmentRecord]:
    """Judge the latest answer of each golden item and write the verdicts to ``out_path``.

    Refusals need no model call: they make no claims and their truthfulness follows from
    whether the item should be refused. Answers to out-of-scope questions are only checked
    for groundedness, since answering them is already wrong. A failing call is recorded as an
    error and the run continues; cached verdicts make re-runs cheap. Failed answers are not
    judged.
    """
    latest = latest_answers(answers)
    answered = [
        (item, record.started_at, response)
        for item in items
        if (record := latest.get(item.id)) is not None and (response := record.result) is not None
    ]
    if limit is not None:
        answered = answered[:limit]
    records: list[JudgmentRecord] = []
    for item, started_at, response in answered:
        judgment = _judge_answer(judge, item, response, answer_started_at=started_at)
        records.append(judgment)
        if on_record is not None:
            on_record(judgment)
    write_records(out_path, records)
    return records


def _judge_answer(
    judge: Judge, item: GoldenItem, response: ChatResponse, *, answer_started_at: str
) -> JudgmentRecord:
    refused = is_refusal(response)
    groundedness: GroundednessVerdict | None = None
    correctness: CorrectnessVerdict | None = None
    error: str | None = None
    try:
        if not refused:
            groundedness = judge.groundedness(
                item.id, question=item.question, answer=response.answer, sources=response.sources
            )
            if not item.should_refuse:
                correctness = judge.correctness(
                    item.id,
                    question=item.question,
                    answer=response.answer,
                    reference=item.reference_answer,
                )
    except Exception as exc:
        logger.exception("Judging %s failed", item.id)
        error = f"{type(exc).__name__}: {exc}"
    return JudgmentRecord(
        item_id=item.id,
        answer_started_at=answer_started_at,
        judge_model=judge.model_name,
        same_family=shares_family(judge.model_name, response.usage.models),
        refused=refused,
        groundedness=groundedness,
        correctness=correctness,
        groundedness_score=groundedness.score if groundedness is not None else None,
        correctness_score=correctness.score if correctness is not None else None,
        truthfulness=truthfulness(
            should_refuse=item.should_refuse, refused=refused, correctness=correctness
        ),
        error=error,
    )
