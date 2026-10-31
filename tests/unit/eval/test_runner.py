"""Batch runners: retrieval ablation, agent answers (streamed, resumable) and judge verdicts."""

import logging
import math
import re
import time
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from lexgraph.agent.context import AgentContext
from lexgraph.agent.schemas import QueryAnalysis
from lexgraph.eval.golden import GoldenItem
from lexgraph.eval.judge import DEFAULT_JUDGE_MODEL, JudgeCache, rate_limiter
from lexgraph.eval.runner import (
    ANALYZED_CONFIGS,
    DEFAULT_CONFIGS,
    UNAVAILABLE_ERROR,
    AnswerRecord,
    JudgmentRecord,
    RetrievalConfig,
    RetrievalRecord,
    agent_query_analyzer,
    append_record,
    latest_answers,
    read_records,
    run_answers,
    run_judge,
    run_retrieval_eval,
    unit_recall_metric,
    write_records,
)
from lexgraph.schemas import (
    THREAD_ID_PATTERN,
    ChatEvent,
    ErrorEvent,
    ResetEvent,
    TokenEvent,
    Usage,
)
from tests.unit.eval.fakes import (
    FakeChatService,
    FakeRetriever,
    StubJudge,
    answer_events,
    correctness_verdict,
    groundedness_verdict,
    make_response,
)

CONFIGS = [
    RetrievalConfig(name="dense", mode="dense"),
    RetrievalConfig(name="hybrid+rerank", mode="hybrid", rerank=True),
]


@pytest.fixture
def retriever(golden_items: list[GoldenItem]) -> FakeRetriever:
    cdc, clt, lgpd, _ = (item.question for item in golden_items)
    return FakeRetriever(
        {
            ("dense", False): {
                cdc: ["CDC:49#0", "CDC:49#1", "CDC:18#0"],
                clt: ["CF:7#0", "CLT:59#2", "CLT:59#0"],
                lgpd: [],
            },
            ("hybrid", True): {cdc: ["CDC:49#0"], clt: ["CLT:59#0", "CF:7#0"], lgpd: ["LGPD:18#0"]},
        }
    )


def test_default_configs_cover_the_ablation() -> None:
    assert [(c.name, c.mode, c.rerank) for c in DEFAULT_CONFIGS] == [
        ("dense", "dense", False),
        ("sparse", "sparse", False),
        ("hybrid", "hybrid", False),
        ("hybrid+rerank", "hybrid", True),
    ]


def test_run_retrieval_eval_scores_article_rankings(
    retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    records = run_retrieval_eval(retriever, golden_items, CONFIGS, k=5)

    assert [(r.config, r.item_id) for r in records] == [
        ("dense", "cdc-001"),
        ("dense", "clt-001"),
        ("dense", "lgpd-001"),
        ("hybrid+rerank", "cdc-001"),
        ("hybrid+rerank", "clt-001"),
        ("hybrid+rerank", "lgpd-001"),
    ]
    dense_cdc, dense_clt, dense_lgpd, *_ = records
    assert dense_cdc.ranking == ["CDC:49", "CDC:18"]
    assert dense_cdc.scores["hit@1"] == 1.0
    assert dense_clt.ranking == ["CF:7", "CLT:59"]
    assert dense_clt.question_type == "cross_law"
    assert (dense_clt.scores["hit@1"], dense_clt.scores["mrr@5"]) == (0.0, 0.5)
    ideal = 2 + 1 / math.log2(3)
    assert dense_clt.scores["ndcg@5"] == pytest.approx((1 + 2 / math.log2(3)) / ideal)
    assert dense_lgpd.ranking == []
    assert dense_lgpd.scores["recall@5"] == 0.0
    assert all(record.k == 5 and record.latency_ms >= 0 for record in records)
    assert all(record.rerank_candidates is None for record in records)


@pytest.mark.parametrize(("k", "candidates"), [(5, 20), (30, 30)])
def test_run_retrieval_eval_records_the_passages_the_reranker_scored(
    retriever: FakeRetriever, golden_items: list[GoldenItem], k: int, candidates: int
) -> None:
    records = run_retrieval_eval(retriever, golden_items, CONFIGS, k=k, rerank_candidates=20)

    recorded = {(record.config, record.rerank_candidates) for record in records}
    assert recorded == {("dense", None), ("hybrid+rerank", candidates)}


def test_run_retrieval_eval_warms_up_each_configuration(
    retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    run_retrieval_eval(retriever, golden_items, CONFIGS, k=5)
    first_question = golden_items[0].question

    assert len(retriever.calls) == 8
    assert retriever.calls[0] == (first_question, 5, "dense", False)
    assert retriever.calls[4] == (first_question, 5, "hybrid", True)

    retriever.calls.clear()
    run_retrieval_eval(retriever, golden_items, CONFIGS, k=5, warmup=False)
    assert len(retriever.calls) == 6


def test_run_retrieval_eval_skips_out_of_scope_items(
    retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    assert run_retrieval_eval(retriever, golden_items[3:], CONFIGS) == []
    assert retriever.calls == []


def test_run_retrieval_eval_scores_the_units_of_long_articles(
    golden_items: list[GoldenItem],
) -> None:
    clt = golden_items[1].model_copy(update={"expected_units": {"CLT:59": ["caput", "§ 1º"]}})
    retriever = FakeRetriever(
        {("hybrid", False): {clt.question: ["CLT:59#1", "CF:7#0", "CLT:59#0"]}},
        units={"CLT:59#0": ["caput"], "CLT:59#1": ["§ 2º", "§ 3º"]},
    )

    (record,) = run_retrieval_eval(retriever, [clt], [DEFAULT_CONFIGS[2]], k=2)

    assert record.ranking == ["CLT:59", "CF:7"]
    assert record.scores["hit@1"] == 1.0
    assert record.scores[unit_recall_metric(2)] == 0.0
    (deeper,) = run_retrieval_eval(retriever, [clt], [DEFAULT_CONFIGS[2]], k=3)
    assert deeper.scores[unit_recall_metric(3)] == 0.5
    others = run_retrieval_eval(retriever, golden_items[:1], [DEFAULT_CONFIGS[2]], k=2)
    assert unit_recall_metric(2) not in others[0].scores


def test_run_retrieval_eval_searches_the_analyzer_query(
    golden_items: list[GoldenItem],
) -> None:
    cdc = golden_items[0]
    retriever = FakeRetriever(
        {
            ("hybrid", False): {
                "direito de arrependimento": ["CDC:49#0", "CDC:18#0"],
                "prazo de reflexão": ["CDC:26#0", "CDC:49#0"],
            }
        }
    )
    analyzed: list[str] = []

    def analyzer(question: str) -> list[str]:
        analyzed.append(question)
        return ["direito de arrependimento", "prazo de reflexão"]

    records = run_retrieval_eval(
        retriever, [cdc], [DEFAULT_CONFIGS[2], ANALYZED_CONFIGS[1]], k=5, analyzer=analyzer
    )

    raw, rewritten = records
    assert (raw.config, raw.query, raw.ranking) == ("hybrid", None, [])
    assert rewritten.config == "hybrid+analyzer"
    assert rewritten.query == "direito de arrependimento | prazo de reflexão"
    # Like the agent, the rankings alternate rank by rank without repeating an article.
    assert rewritten.ranking == ["CDC:49", "CDC:26", "CDC:18"]
    assert analyzed == [cdc.question]
    assert retriever.calls[-2:] == [
        ("direito de arrependimento", 5, "hybrid", False),
        ("prazo de reflexão", 5, "hybrid", False),
    ]


def test_run_retrieval_eval_needs_an_analyzer_for_analyzed_configs(
    retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    with pytest.raises(ValueError, match="query analyzer"):
        run_retrieval_eval(retriever, golden_items, [ANALYZED_CONFIGS[1]])
    assert retriever.calls == []


def _context(analyze: RunnableLambda[LanguageModelInput, QueryAnalysis]) -> AgentContext:
    def unused(_: LanguageModelInput) -> Any:
        msg = "only the analyzer is used"
        raise AssertionError(msg)

    return AgentContext(
        retriever=FakeRetriever({}),
        analyzer=analyze,
        grader=RunnableLambda(unused),
        rewriter=RunnableLambda(unused),
        verifier=RunnableLambda(unused),
        answerer=RunnableLambda(unused),
    )


def _analysis(search_query: str, sub_queries: list[str] | None = None) -> QueryAnalysis:
    return QueryAnalysis.model_validate(
        {
            "intent": "legal_question",
            "standalone_question": "Pergunta.",
            "search_query": search_query,
            "sub_queries": sub_queries or [],
            "laws": ["CDC"],
            "articles": [],
        }
    )


def test_agent_query_analyzer_uses_the_analyzer_query() -> None:
    prompts: list[LanguageModelInput] = []

    def analyze(messages: LanguageModelInput) -> QueryAnalysis:
        prompts.append(messages)
        return _analysis(
            "  direito de arrependimento compra fora do estabelecimento  ",
            [" prazo de reflexão ", " ", "devolução do valor pago", "vício do produto"],
        )

    analyzer = agent_query_analyzer(_context(RunnableLambda(analyze)))

    assert analyzer("Posso devolver?") == [
        "direito de arrependimento compra fora do estabelecimento",
        "prazo de reflexão",
        "devolução do valor pago",
    ]
    assert "Posso devolver?" in str(prompts[0])
    blank = agent_query_analyzer(_context(RunnableLambda(lambda _: _analysis(" "))))
    assert blank("Posso devolver?") == ["Posso devolver?"]


def test_agent_query_analyzer_falls_back_to_the_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_: LanguageModelInput) -> QueryAnalysis:
        msg = "no valid QueryAnalysis"
        raise OutputParserException(msg)

    analyzer = agent_query_analyzer(_context(RunnableLambda(fail)))

    with caplog.at_level(logging.WARNING):
        assert analyzer("Posso devolver?") == ["Posso devolver?"]
    assert "Query analysis failed" in caplog.text


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    record = RetrievalRecord(
        config="dense",
        item_id="cdc-001",
        question_type="lookup",
        k=5,
        ranking=["CDC:49"],
        scores={"hit@1": 1.0},
        latency_ms=1.5,
    )
    path = tmp_path / "out" / "retrieval.jsonl"
    write_records(path, [record])
    append_record(path, record)

    assert read_records(path, RetrievalRecord) == [record, record]
    assert b"\r\n" not in path.read_bytes()


def test_records_without_the_optional_fields_still_load(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        '{"config": "hybrid+rerank", "item_id": "cdc-001", "question_type": "lookup", "k": 5, '
        '"ranking": [], "scores": {}, "latency_ms": 1.5, "query": null}\n',
        encoding="utf-8",
    )
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        '{"item_id": "cdc-001", "started_at": "2026-10-30T12:00:00+00:00", "latency_ms": 9.0}\n',
        encoding="utf-8",
    )

    (retrieval,) = read_records(retrieval_path, RetrievalRecord)
    (answer,) = read_records(answers_path, AnswerRecord)

    assert retrieval.rerank_candidates is None
    assert (answer.latency_ms, answer.ttft_ms, answer.paced_ms) == (9.0, None, 0.0)


def _cut_record(item_id: str = "clt-001") -> AnswerRecord:
    return AnswerRecord(item_id=item_id, started_at="2026-10-30T12:00:00+00:00", latency_ms=1.0)


def test_read_records_skips_an_unfinished_last_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "answers.jsonl"
    write_records(path, [_cut_record("cdc-001")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_cut_record().model_dump_json()[:30])

    with caplog.at_level(logging.WARNING):
        records = read_records(path, AnswerRecord)

    assert [record.item_id for record in records] == ["cdc-001"]
    assert "unfinished last line" in caplog.text


def test_read_records_rejects_a_corrupt_line_elsewhere(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('{"truncated\n' + _cut_record().model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        read_records(path, AnswerRecord)


def test_append_record_removes_an_unfinished_last_line(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    write_records(path, [_cut_record("cdc-001")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_cut_record().model_dump_json()[:30])

    append_record(path, _cut_record("lgpd-001"))

    assert [r.item_id for r in read_records(path, AnswerRecord)] == ["cdc-001", "lgpd-001"]


def test_append_record_completes_a_last_line_without_its_newline(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(_cut_record("cdc-001").model_dump_json(), encoding="utf-8")

    append_record(path, _cut_record("lgpd-001"))

    assert [r.item_id for r in read_records(path, AnswerRecord)] == ["cdc-001", "lgpd-001"]
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    append_record(empty, _cut_record())
    assert read_records(empty, AnswerRecord) == [_cut_record()]


@pytest.fixture
def service(golden_items: list[GoldenItem]) -> FakeChatService:
    cdc, clt, lgpd, oos = golden_items
    good = make_response("Você pode desistir em 7 dias [1].", context=["CDC:49"], cited=["CDC:49"])
    regenerated = make_response("Até 2 horas extras [1].", context=["CLT:59"], regenerations=1)
    regenerated_events: list[ChatEvent] = [
        TokenEvent(text=""),
        TokenEvent(text="Rascunho"),
        ResetEvent(reason="ungrounded"),
        *answer_events(regenerated, "Até 2 horas extras [1]."),
    ]
    return FakeChatService(
        {
            cdc.question: answer_events(good, "Você pode ", "desistir em 7 dias [1]."),
            clt.question: regenerated_events,
            lgpd.question: [ErrorEvent(message="Falha temporária no provedor.")],
            oos.question: RuntimeError("connection reset"),
        }
    )


async def test_run_answers_records_streams_errors_and_timings(
    service: FakeChatService, golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    out_path = tmp_path / "answers.jsonl"
    seen: list[str] = []

    records = await run_answers(
        service, golden_items, out_path, on_record=lambda r: seen.append(r.item_id)
    )

    cdc, clt, lgpd, oos = records
    assert seen == ["cdc-001", "clt-001", "lgpd-001", "oos-001"]
    assert cdc.response is not None
    assert cdc.response.answer == "Você pode desistir em 7 dias [1]."
    assert cdc.stages == ["analyze", "retrieve"]
    assert cdc.ttft_ms is not None
    assert 0 <= cdc.ttft_ms <= cdc.latency_ms
    assert cdc.paced_ms == 0.0
    assert (cdc.resets, cdc.error) == (0, None)
    assert (clt.resets, clt.stages) == (1, ["analyze", "retrieve"])
    assert (lgpd.response, lgpd.error, lgpd.ttft_ms) == (
        None,
        "Falha temporária no provedor.",
        None,
    )
    assert (oos.response, oos.error) == (None, "RuntimeError: connection reset")
    assert read_records(out_path, AnswerRecord) == records


class _Clock:
    """Stand-in for ``time.perf_counter`` that moves only when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _PacedEvents(list[ChatEvent]):
    """Events that each take ``step_s`` on ``clock`` to arrive."""

    def __init__(self, events: list[ChatEvent], clock: _Clock, step_s: float) -> None:
        super().__init__(events)
        self.clock, self.step_s = clock, step_s

    def __iter__(self) -> Iterator[ChatEvent]:
        for event in super().__iter__():
            self.clock.now += self.step_s
            yield event


async def test_run_answers_leaves_the_rate_limiter_waits_out_of_the_timings(
    golden_items: list[GoldenItem], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()

    async def wait_for_a_token(self: InMemoryRateLimiter, *, blocking: bool = True) -> bool:
        clock.now += 2.0
        return True

    monkeypatch.setattr(time, "perf_counter", clock)
    monkeypatch.setattr(InMemoryRateLimiter, "aacquire", wait_for_a_token)
    limiter = rate_limiter(30)
    response = make_response("Você pode desistir em 7 dias [1].", context=["CDC:49"])
    # Three status events, two tokens and the final response, 0.25 s apart.
    events = answer_events(response, "Você pode ", "desistir em 7 dias [1].")
    paced = _PacedEvents(events, clock, 0.25)
    service = FakeChatService({item.question: paced for item in golden_items[:2]}, limiter=limiter)

    records = await run_answers(
        service, golden_items[:2], tmp_path / "answers.jsonl", limiter=limiter
    )

    assert limiter.waited_s == pytest.approx(4.0)
    for record in records:
        assert record.paced_ms == pytest.approx(2000)
        assert record.ttft_ms == pytest.approx(1000)
        assert record.latency_ms == pytest.approx(1500)


async def test_run_answers_uses_a_fresh_thread_per_question(
    service: FakeChatService, golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    await run_answers(service, golden_items, tmp_path / "answers.jsonl")

    thread_ids = [request.thread_id or "" for request in service.requests]
    assert len(set(thread_ids)) == 4
    assert thread_ids[0].startswith("eval-cdc-001-")
    assert all(re.fullmatch(THREAD_ID_PATTERN, thread_id) for thread_id in thread_ids)
    assert all(request.laws is None and request.rerank is None for request in service.requests)


async def test_run_answers_flags_streams_without_a_final_event(
    golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    silent = FakeChatService({golden_items[0].question: [TokenEvent(text="...")]})
    (record,) = await run_answers(silent, golden_items[:1], tmp_path / "answers.jsonl")
    assert record.error == "the stream ended without a final event"


async def test_run_answers_retries_answers_no_model_could_write(
    golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    question = golden_items[0].question
    degraded = make_response("Não consegui redigir a resposta agora.", route="unavailable")
    answered = make_response("Você pode desistir em 7 dias [1].", context=["CDC:49"])
    service = FakeChatService({question: answer_events(degraded, "Não consegui...")})
    out_path = tmp_path / "answers.jsonl"

    (failed,) = await run_answers(service, golden_items[:1], out_path)
    service.scripts = {question: answer_events(answered, "Você pode...")}
    (retried,) = await run_answers(service, golden_items[:1], out_path)

    assert failed.error == UNAVAILABLE_ERROR
    assert failed.response == degraded
    assert failed.result is None
    assert (retried.error, retried.result) == (None, answered)
    assert len(read_records(out_path, AnswerRecord)) == 2


async def test_run_answers_resumes_and_retries_failures(
    service: FakeChatService, golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    out_path = tmp_path / "answers.jsonl"
    await run_answers(service, golden_items, out_path)

    retried = await run_answers(service, golden_items, out_path)

    assert [record.item_id for record in retried] == ["lgpd-001", "oos-001"]
    assert len(service.requests) == 6
    stored = read_records(out_path, AnswerRecord)
    assert len(stored) == 6
    assert list(latest_answers(stored)) == ["cdc-001", "clt-001", "lgpd-001", "oos-001"]


async def test_run_answers_limit_and_restart(
    service: FakeChatService, golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    out_path = tmp_path / "answers.jsonl"
    first = await run_answers(service, golden_items, out_path, limit=2)
    assert [record.item_id for record in first] == ["cdc-001", "clt-001"]

    second = await run_answers(service, golden_items, out_path, limit=1)
    assert [record.item_id for record in second] == ["lgpd-001"]

    restarted = await run_answers(service, golden_items, out_path, resume=False, limit=1)
    assert [record.item_id for record in restarted] == ["cdc-001"]
    assert [r.item_id for r in read_records(out_path, AnswerRecord)] == ["cdc-001"]


async def test_run_answers_spaces_out_questions(
    service: FakeChatService, golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    await run_answers(service, golden_items[:3], tmp_path / "answers.jsonl", delay_s=0.1)

    gaps = [later - earlier for earlier, later in pairwise(service.started)]
    assert len(gaps) == 2
    # asyncio may fire a timer up to one clock tick early (about 16 ms on Windows).
    assert all(gap >= 0.08 for gap in gaps)


STARTED_AT = "2026-10-30T12:00:00+00:00"


def _answer(item_id: str, **response: Any) -> AnswerRecord:
    answer = response.pop("answer", "Resposta [1].")
    return AnswerRecord(
        item_id=item_id,
        started_at=STARTED_AT,
        latency_ms=100.0,
        response=make_response(answer, **response),
    )


@pytest.fixture
def answers() -> list[AnswerRecord]:
    return [
        _answer("cdc-001", answer="Desista em 7 dias [1].", context=["CDC:49"], cited=["CDC:49"]),
        _answer("clt-001", answer="Não encontrei base legal.", route="no_context"),
        _answer("lgpd-001", answer="Portabilidade e eliminação [1].", context=["LGPD:18"]),
        _answer("oos-001", answer="A pena é de reclusão de 1 a 4 anos."),
        AnswerRecord(
            item_id="zzz-999", started_at="2026-10-30T12:00:00+00:00", latency_ms=1.0, error="x"
        ),
    ]


def test_run_judge_applies_the_judging_policy(
    golden_items: list[GoldenItem], answers: list[AnswerRecord], tmp_path: Path
) -> None:
    lgpd_question = golden_items[2].question

    def correctness(inputs: dict[str, Any]) -> Any:
        if inputs["question"] == lgpd_question:
            msg = "quota exceeded"
            raise RuntimeError(msg)
        return correctness_verdict("partial")

    stub = StubJudge(
        groundedness=lambda _: groundedness_verdict("supported", "unsupported"),
        correctness=correctness,
    )
    out_path = tmp_path / "judgments.jsonl"

    records = run_judge(stub.judge, golden_items, answers, out_path)

    by_id = {record.item_id: record for record in records}
    assert list(by_id) == ["cdc-001", "clt-001", "lgpd-001", "oos-001"]
    cdc, clt, lgpd, oos = records
    assert (cdc.correctness_score, cdc.groundedness_score, cdc.truthfulness) == (0.5, 0.5, 0.5)
    assert (clt.refused, clt.groundedness, clt.correctness, clt.truthfulness) == (
        True,
        None,
        None,
        0.0,
    )
    assert lgpd.error == "RuntimeError: quota exceeded"
    assert (lgpd.groundedness_score, lgpd.correctness, lgpd.truthfulness) == (0.5, None, None)
    assert (oos.groundedness_score, oos.correctness, oos.truthfulness) == (0.5, None, -1.0)
    assert len(stub.groundedness_calls) == 3
    assert len(stub.correctness_calls) == 2
    assert all(record.judge_model == "google_genai:judge-test" for record in records)
    assert all(record.answer_started_at == STARTED_AT for record in records)
    assert not any(record.same_family for record in records)
    assert read_records(out_path, JudgmentRecord) == records


def test_run_judge_flags_answers_from_the_judge_family(
    golden_items: list[GoldenItem], tmp_path: Path
) -> None:
    usage = Usage(models=["openai/gpt-oss-20b", "gemini-3.5-flash-lite"])
    answers = [
        _answer("cdc-001", context=["CDC:49"], usage=usage),
        _answer("clt-001", context=["CLT:59"], usage=Usage(models=["openai/gpt-oss-120b"])),
    ]

    gemini_judge = StubJudge(model_name=DEFAULT_JUDGE_MODEL).judge

    records = run_judge(gemini_judge, golden_items, answers, tmp_path / "judgments.jsonl")

    assert [(record.item_id, record.same_family) for record in records] == [
        ("cdc-001", True),
        ("clt-001", False),
    ]


def test_run_judge_skips_failed_answers(golden_items: list[GoldenItem], tmp_path: Path) -> None:
    degraded = _answer("cdc-001", route="unavailable").model_copy(
        update={"error": UNAVAILABLE_ERROR}
    )
    stub = StubJudge()

    assert run_judge(stub.judge, golden_items, [degraded], tmp_path / "judgments.jsonl") == []
    assert stub.groundedness_calls == []


def test_run_judge_limit_and_cache(
    golden_items: list[GoldenItem], answers: list[AnswerRecord], tmp_path: Path
) -> None:
    cache = JudgeCache(tmp_path / "cache.jsonl")
    out_path = tmp_path / "judgments.jsonl"
    stub = StubJudge(cache=cache)

    assert [r.item_id for r in run_judge(stub.judge, golden_items, answers, out_path, limit=1)] == [
        "cdc-001"
    ]
    run_judge(stub.judge, golden_items, answers, out_path)
    calls = (len(stub.groundedness_calls), len(stub.correctness_calls))
    run_judge(stub.judge, golden_items, answers, out_path)

    assert calls == (3, 2)
    assert (len(stub.groundedness_calls), len(stub.correctness_calls)) == calls
    assert len(read_records(out_path, JudgmentRecord)) == 4


def test_analyzed_configurations_share_one_analysis_per_question(
    retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    calls: list[str] = []

    def analyzer(question: str) -> list[str]:
        calls.append(question)
        return ["direito de arrependimento"]

    in_scope = [item for item in golden_items if not item.should_refuse]

    records = run_retrieval_eval(retriever, golden_items, ANALYZED_CONFIGS, k=5, analyzer=analyzer)

    assert len(calls) == len(in_scope)
    assert {record.config for record in records} == {"dense+analyzer", "hybrid+analyzer"}
