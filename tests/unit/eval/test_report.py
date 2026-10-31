"""Report summaries and Markdown rendering, checked against hand-computed values."""

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexgraph.eval.golden import GoldenItem
from lexgraph.eval.metrics import retrieval_scores
from lexgraph.eval.report import (
    EvalSummary,
    Rate,
    build_summary,
    headline_metrics,
    render_markdown,
    retrieval_markdown,
    summarize_retrieval,
    write_report,
)
from lexgraph.eval.runner import (
    UNAVAILABLE_ERROR,
    AnswerRecord,
    JudgmentRecord,
    RetrievalRecord,
    unit_recall_metric,
)
from lexgraph.schemas import Route, Usage
from tests.unit.eval.fakes import correctness_verdict, groundedness_verdict, make_response

GENERATED_AT = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
STARTED_AT = "2026-10-30T12:00:00+00:00"
ANSWER_MODEL = "groq:openai/gpt-oss-120b"
FAST_MODEL = "groq:openai/gpt-oss-20b"
FALLBACK_MODEL = "google_genai:gemini-3.5-flash-lite"
NDCG_CLT_DENSE = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))


def _retrieval(
    golden: dict[str, GoldenItem],
    config: str,
    item_id: str,
    ranking: list[str],
    ms: float,
    *,
    query: str | None = None,
    unit_recall: float | None = None,
) -> RetrievalRecord:
    item = golden[item_id]
    scores = retrieval_scores(ranking, item.relevance, 5)
    if unit_recall is not None:
        scores[unit_recall_metric(5)] = unit_recall
    return RetrievalRecord(
        config=config,
        item_id=item_id,
        question_type=item.type,
        k=5,
        ranking=ranking,
        scores=scores,
        latency_ms=ms,
        query=query,
    )


@pytest.fixture
def retrieval(golden_items: list[GoldenItem]) -> list[RetrievalRecord]:
    golden = {item.id: item for item in golden_items}
    return [
        _retrieval(golden, "dense", "cdc-001", ["CDC:49"], 10),
        _retrieval(golden, "dense", "clt-001", ["CF:7", "CLT:59"], 20),
        _retrieval(golden, "dense", "lgpd-001", [], 30),
        _retrieval(golden, "hybrid", "cdc-001", ["CDC:49"], 40),
        _retrieval(golden, "hybrid", "clt-001", ["CLT:59", "CF:7"], 50),
        _retrieval(golden, "hybrid", "lgpd-001", ["LGPD:18"], 60),
    ]


def _answer(
    item_id: str,
    answer: str,
    *,
    latency_ms: float,
    ttft_ms: float | None = None,
    route: Route = "answered",
    context: tuple[str, ...] = (),
    cited: tuple[str, ...] = (),
    rewrites: int = 0,
    regenerations: int = 0,
    grounded: bool | None = None,
    tokens: tuple[int, int] = (0, 0),
    models: tuple[str, ...] = (FAST_MODEL,),
) -> AnswerRecord:
    usage = Usage(
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        total_tokens=sum(tokens),
        models=list(models),
    )
    response = make_response(
        answer,
        route=route,
        context=context,
        cited=cited,
        rewrites=rewrites,
        regenerations=regenerations,
        grounded=grounded,
        usage=usage,
    )
    return AnswerRecord(
        item_id=item_id,
        started_at=STARTED_AT,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        response=response,
    )


def _failure(item_id: str) -> AnswerRecord:
    return AnswerRecord(
        item_id=item_id, started_at="2026-10-30T11:00:00+00:00", latency_ms=5.0, error="timeout"
    )


@pytest.fixture
def answers() -> list[AnswerRecord]:
    return [
        _failure("cdc-001"),
        _answer(
            "cdc-001",
            "Desista em até 7 dias [1].",
            latency_ms=1000,
            ttft_ms=300,
            context=("CDC:49",),
            cited=("CDC:49",),
            grounded=True,
            tokens=(1000, 100),
            models=(FAST_MODEL, ANSWER_MODEL),
        ),
        _answer(
            "clt-001",
            "Até 2 horas extras por dia, com adicional de 50% [1].",
            latency_ms=3000,
            ttft_ms=900,
            context=("CLT:59", "CF:7"),
            cited=("CLT:59",),
            rewrites=1,
            regenerations=1,
            grounded=False,
            tokens=(2000, 200),
            models=(FAST_MODEL, ANSWER_MODEL, FALLBACK_MODEL),
        ),
        _answer(
            "lgpd-001",
            "Não encontrei base legal no acervo.",
            latency_ms=500,
            route="no_context",
            tokens=(300, 0),
        ),
        _answer(
            "oos-001",
            "A pergunta está fora do escopo do acervo.",
            latency_ms=200,
            route="out_of_scope",
            tokens=(200, 0),
        ),
        _failure("zzz-999"),
    ]


@pytest.fixture
def judgments() -> list[JudgmentRecord]:
    def record(item_id: str, **fields: object) -> JudgmentRecord:
        return JudgmentRecord.model_validate(
            {
                "item_id": item_id,
                "answer_started_at": STARTED_AT,
                "judge_model": "google_genai:judge-test",
                "refused": False,
            }
            | fields
        )

    return [
        record(
            "cdc-001",
            groundedness=groundedness_verdict("supported"),
            correctness=correctness_verdict("correct"),
            groundedness_score=1.0,
            correctness_score=1.0,
            truthfulness=1.0,
        ),
        record(
            "clt-001",
            groundedness=groundedness_verdict("supported", "unsupported"),
            correctness=correctness_verdict("partial"),
            groundedness_score=0.5,
            correctness_score=0.5,
            truthfulness=0.5,
            same_family=True,
        ),
        record("lgpd-001", refused=True, truthfulness=0.0),
        record("oos-001", refused=True, truthfulness=1.0),
        record("zzz-999", truthfulness=-1.0),
    ]


@pytest.fixture
def summary(
    golden_items: list[GoldenItem],
    retrieval: list[RetrievalRecord],
    answers: list[AnswerRecord],
    judgments: list[JudgmentRecord],
) -> EvalSummary:
    return build_summary(
        golden_items,
        retrieval=retrieval,
        answers=answers,
        judgments=judgments,
        generated_at=GENERATED_AT,
    )


def test_golden_stats(summary: EvalSummary) -> None:
    assert summary.generated_at == "2026-10-30T12:00:00+00:00"
    assert (summary.golden.items, summary.golden.in_scope, summary.golden.out_of_scope) == (4, 3, 1)
    assert summary.golden.by_type == {
        "conceptual": 1,
        "cross_law": 1,
        "lookup": 1,
        "out_of_scope": 1,
    }


def test_headline_metrics_follow_k() -> None:
    available = retrieval_scores(["CDC:49"], {"CDC:49": 2}, 10)
    assert headline_metrics(10, available) == [
        "hit@1",
        "hit@5",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
    ]
    assert headline_metrics(3, retrieval_scores([], {"CDC:49": 2}, 3)) == [
        "hit@1",
        "recall@3",
        "mrr@3",
        "ndcg@3",
    ]


def test_retrieval_summary(summary: EvalSummary) -> None:
    retrieval = summary.retrieval
    assert retrieval is not None
    dense, hybrid = retrieval.configs
    assert (retrieval.k, retrieval.headline_metric) == (5, "ndcg@5")
    assert (dense.name, dense.n, dense.latency_ms.p50) == ("dense", 3, 20.0)
    assert list(dense.metrics) == ["hit@1", "hit@5", "recall@5", "mrr@5", "ndcg@5"]
    assert dense.metrics["ndcg@5"].mean == pytest.approx((1 + NDCG_CLT_DENSE) / 3)
    assert dense.metrics["hit@1"].mean == pytest.approx(1 / 3)
    assert hybrid.metrics["ndcg@5"].mean == pytest.approx(1.0)

    (comparison,) = retrieval.comparisons
    assert (comparison.baseline, comparison.candidate) == ("dense", "hybrid")
    assert comparison.difference.mean == pytest.approx((1 - NDCG_CLT_DENSE + 1) / 3)
    assert comparison.difference.n == 3

    assert [(row.question_type, row.n) for row in retrieval.by_type] == [
        ("conceptual", 1),
        ("cross_law", 1),
        ("lookup", 1),
    ]
    assert retrieval.by_type[1].means == pytest.approx({"dense": NDCG_CLT_DENSE, "hybrid": 1.0})


def test_retrieval_summary_reports_unit_recall_and_analyzed_configs(
    golden_items: list[GoldenItem], retrieval: list[RetrievalRecord]
) -> None:
    golden = {item.id: item for item in golden_items}
    records = [
        retrieval[0],
        _retrieval(golden, "dense", "clt-001", ["CF:7", "CLT:59"], 20, unit_recall=0.5),
        _retrieval(golden, "hybrid+analyzer", "cdc-001", ["CDC:49"], 5, query="arrependimento"),
        _retrieval(
            golden, "hybrid+analyzer", "clt-001", ["CLT:59"], 5, query="hora", unit_recall=1.0
        ),
    ]

    summary = summarize_retrieval(records)

    dense, analyzer = summary.configs
    assert (dense.analyzed, analyzer.analyzed) == (False, True)
    assert dense.unit_recall is not None
    assert (dense.unit_recall.mean, dense.unit_recall.n) == (0.5, 1)
    assert analyzer.unit_recall is not None
    assert analyzer.unit_recall.mean == 1.0
    markdown = "\n".join(retrieval_markdown(summary))
    assert "without the agent's query analysis, except `hybrid+analyzer`" in markdown
    assert "| Config | n | Unit recall@5 |" in markdown
    assert "| `dense` | 1 | 0.50 [0.50, 0.50] |" in markdown
    plain = "\n".join(retrieval_markdown(summarize_retrieval(retrieval)))
    assert "Unit recall" not in plain
    assert "except" not in plain
    both = [*records, _retrieval(golden, "dense+analyzer", "cdc-001", ["CDC:49"], 5, query="x")]
    two = "\n".join(retrieval_markdown(summarize_retrieval(both)))
    assert "except `hybrid+analyzer` and `dense+analyzer`, which search" in two


def test_retrieval_summary_states_the_passages_the_reranker_scored(
    retrieval: list[RetrievalRecord],
) -> None:
    reranked = [
        record.model_copy(update={"config": "hybrid+rerank", "rerank_candidates": 20})
        for record in retrieval[3:]
    ]

    summary = summarize_retrieval([*retrieval, *reranked])

    assert [(config.name, config.rerank_candidates) for config in summary.configs] == [
        ("dense", None),
        ("hybrid", None),
        ("hybrid+rerank", 20),
    ]
    markdown = "\n".join(retrieval_markdown(summary))
    assert (
        "`hybrid+rerank` reranks the first 20 passages of its search with the cross-encoder "
        "and keeps the best 5."
    ) in markdown
    assert "reranks" not in "\n".join(retrieval_markdown(summarize_retrieval(retrieval)))


def test_single_configuration_has_no_comparison(retrieval: list[RetrievalRecord]) -> None:
    single = summarize_retrieval(retrieval[:3])
    assert single.comparisons == []
    with pytest.raises(ValueError, match="no retrieval records"):
        summarize_retrieval([])


def test_answer_quality_summary(summary: EvalSummary) -> None:
    answers = summary.answers
    assert answers is not None
    assert answers.scored == 4
    assert answers.must_include is not None
    assert answers.must_include.mean == pytest.approx(2 / 3)
    assert answers.must_include.n == 3
    assert answers.must_not_include is not None
    assert (answers.must_not_include.mean, answers.must_not_include.n) == (1.0, 1)
    assert answers.citation_precision is not None
    assert (answers.citation_precision.mean, answers.citation_precision.n) == (1.0, 2)
    assert answers.citation_recall is not None
    assert answers.citation_recall.mean == pytest.approx(2 / 3)
    assert answers.citations_in_context is not None
    assert answers.citations_in_context.mean == 1.0


def test_judge_summary_ignores_items_outside_the_golden_set(summary: EvalSummary) -> None:
    assert summary.answers is not None
    judge = summary.answers.judge
    assert judge is not None
    assert (judge.judged, judge.errors, judge.judge_models) == (4, 0, ["google_genai:judge-test"])
    assert (judge.refused, judge.same_family) == (2, 1)
    assert summary.answers.stale_judgments == 0
    assert judge.verdicts == {"correct": 1, "partial": 1, "incorrect": 0}
    assert judge.correctness is not None
    assert (judge.correctness.mean, judge.correctness.n) == (0.75, 2)
    assert judge.groundedness is not None
    assert judge.groundedness.mean == 0.75
    assert judge.truthfulness is not None
    assert (judge.truthfulness.mean, judge.truthfulness.n) == (0.625, 4)


def test_judgments_of_earlier_answers_are_left_out(
    golden_items: list[GoldenItem], answers: list[AnswerRecord], judgments: list[JudgmentRecord]
) -> None:
    earlier = judgments[0].model_copy(update={"answer_started_at": "2026-10-29T08:00:00+00:00"})
    summary = build_summary(golden_items, answers=answers, judgments=[earlier, *judgments[1:]])

    assert summary.answers is not None
    assert summary.answers.stale_judgments == 1
    judge = summary.answers.judge
    assert judge is not None
    assert judge.judged == 3
    assert judge.correctness is not None
    assert judge.correctness.n == 1
    markdown = render_markdown(summary)
    assert "1 judgment(s) of earlier answers were left out" in markdown

    only_judgments = build_summary(golden_items, judgments=judgments)
    assert only_judgments.answers is not None
    assert (only_judgments.answers.judge, only_judgments.answers.stale_judgments) == (None, 4)


def test_answers_no_model_could_write_count_as_errors(
    golden_items: list[GoldenItem], answers: list[AnswerRecord]
) -> None:
    unavailable = _answer("oos-001", "Não consegui redigir a resposta.", latency_ms=50.0)
    failed = unavailable.model_copy(update={"error": UNAVAILABLE_ERROR})
    summary = build_summary(golden_items, answers=[*answers, failed])

    assert summary.agent is not None
    assert (summary.agent.answered, summary.agent.errors) == (3, 1)
    assert summary.answers is not None
    assert summary.answers.scored == 3


def test_agent_summary(summary: EvalSummary) -> None:
    agent = summary.agent
    assert agent is not None
    assert (agent.answered, agent.errors) == (4, 0)
    assert agent.routes == {"answered": 2, "no_context": 1, "out_of_scope": 1}
    assert agent.rewrite_rate == Rate.of(1, 3)
    assert agent.mean_rewrites == pytest.approx(1 / 3)
    assert agent.regeneration_rate == Rate.of(1, 4)
    assert agent.self_check_pass_rate == Rate.of(1, 2)
    refusal = agent.refusal
    assert (refusal.true_positives, refusal.false_positives, refusal.false_negatives) == (1, 1, 0)
    assert (refusal.precision, refusal.recall) == (0.5, 1.0)
    assert refusal.false_refusal_rate == pytest.approx(1 / 3)
    assert agent.context_hit is not None
    assert agent.context_hit.mean == pytest.approx(2 / 3)
    assert agent.context_hit.n == 3
    assert agent.context_recall is not None
    assert agent.context_recall.mean == pytest.approx(2 / 3)
    assert agent.latency_ms is not None
    assert (agent.latency_ms.p50, agent.latency_ms.p95) == pytest.approx((750.0, 2700.0))
    assert agent.ttft_ms is not None
    assert (agent.ttft_ms.p50, agent.ttft_ms.p95) == pytest.approx((600.0, 870.0))
    assert agent.paced_ms is None
    assert agent.tokens_per_question is not None
    tokens = agent.tokens_per_question
    assert (tokens.input, tokens.output, tokens.total) == (875.0, 75.0, 950.0)
    assert agent.models == {ANSWER_MODEL: 2, FAST_MODEL: 4, FALLBACK_MODEL: 1}


def test_agent_summary_reports_the_pacing_left_out_of_the_timings(
    golden_items: list[GoldenItem], answers: list[AnswerRecord]
) -> None:
    paced = [
        record.model_copy(update={"paced_ms": 10_000.0 * position})
        for position, record in enumerate(answers)
    ]

    summary = build_summary(golden_items, answers=paced)

    agent = summary.agent
    assert agent is not None
    assert agent.paced_ms is not None
    assert (agent.paced_ms.p50, agent.paced_ms.p95, agent.paced_ms.n) == pytest.approx(
        (25_000.0, 38_500.0, 4)
    )
    assert agent.latency_ms is not None
    assert (agent.latency_ms.p50, agent.latency_ms.p95) == pytest.approx((750.0, 2700.0))
    markdown = render_markdown(summary)
    assert "| Latency p50 / p95 (ms) | 750 / 2700 |" in markdown
    assert (
        "| Rate-limit pacing left out of the two timings above, p50 / p95 (ms) | 25000 / 38500 |"
    ) in markdown


def test_agent_summary_counts_unanswered_questions(
    golden_items: list[GoldenItem], answers: list[AnswerRecord]
) -> None:
    summary = build_summary(golden_items, answers=[*answers, _failure("lgpd-001")])
    assert summary.agent is not None
    assert (summary.agent.answered, summary.agent.errors) == (3, 1)
    assert summary.retrieval is None


def test_summary_of_a_run_where_every_question_failed(golden_items: list[GoldenItem]) -> None:
    summary = build_summary(golden_items, answers=[_failure(item.id) for item in golden_items])

    agent = summary.agent
    assert agent is not None
    assert (agent.answered, agent.errors, agent.routes, agent.models) == (0, 4, {}, {})
    assert (agent.latency_ms, agent.ttft_ms, agent.tokens_per_question) == (None, None, None)
    assert (agent.context_hit, agent.context_recall) == (None, None)
    assert summary.answers is not None
    assert summary.answers.scored == 0
    markdown = render_markdown(summary)
    assert "| Questions answered | 0 (4 errors) |" in markdown
    assert "| Latency p50 / p95 (ms) | n/a |" in markdown
    assert "| Tokens per question (input / output / total) | n/a |" in markdown
    assert "| Models | n/a |" in markdown


def test_configurations_without_shared_questions_are_not_compared(
    retrieval: list[RetrievalRecord],
) -> None:
    dense_only_cdc = [r for r in retrieval if r.config == "dense" and r.item_id == "cdc-001"]
    hybrid_only_clt = [r for r in retrieval if r.config == "hybrid" and r.item_id == "clt-001"]
    summary = summarize_retrieval(dense_only_cdc + hybrid_only_clt)
    assert [config.name for config in summary.configs] == ["dense", "hybrid"]
    assert summary.comparisons == []


def test_rate_of_an_empty_total_is_undefined() -> None:
    assert Rate.of(0, 0).value is None
    assert Rate.of(1, 4).value == 0.25


def test_render_markdown(summary: EvalSummary) -> None:
    markdown = render_markdown(summary)

    for expected in (
        "# Evaluation report",
        "Golden set: 4 questions (3 in scope, 1 out of scope).",
        "| Config | n | Hit@1 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | p50 ms |",
        "| `dense` | 3 | 0.33 [",
        "Paired differences in nDCG@5 against `dense`:",
        "| `hybrid` | +0.380 |",
        "| conceptual | 1 | 0.00 | 1.00 |",
        "| Required facts present (must_include) | 0.67 [",
        "| Correctness of non-refused answers (judge: 1, 0.5, 0) | 0.75 [",
        "| Truthfulness (CRAG: 1, 0.5, 0, -1) | 0.62 [",
        (
            "Judge `google_genai:judge-test`: correct 1, partial 1, incorrect 0; 2 refusal(s) "
            "not judged; 0 error(s); 1 answer(s) produced with a model of the judge's family."
        ),
        "any slice of a long article counts as the article",
        "Configurations search the raw question, without the agent's query analysis.",
        "| Expected article in the context (in scope) | 0.67 [",
        "| Routes | answered 2, no_context 1, out_of_scope 1 |",
        "| Rewrite rate (in scope) | 0.33 (1/3) |",
        "| Refusal precision | 0.50 (1/2) |",
        "| Refusal recall | 1.00 (1/1) |",
        "| False refusal rate (in scope) | 0.33 (1/3) |",
        "| Latency p50 / p95 (ms) | 750 / 2700 |",
        "| Time to first token p50 / p95 (ms) | 600 / 870 |",
        "| Tokens per question (input / output / total) | 875 / 75 / 950 |",
    ):
        assert expected in markdown
    assert markdown.endswith("over questions._\n")
    assert "left out" not in markdown
    assert "pacing" not in markdown


def test_render_markdown_skips_missing_sections(golden_items: list[GoldenItem]) -> None:
    markdown = render_markdown(build_summary(golden_items))
    assert "## Retrieval" not in markdown
    assert "## Answer quality" not in markdown
    assert "## Agent behavior" not in markdown


def test_render_markdown_of_unjudged_answers(
    golden_items: list[GoldenItem], answers: list[AnswerRecord]
) -> None:
    markdown = render_markdown(build_summary(golden_items, answers=answers))
    assert "## Answer quality" in markdown
    assert "Correctness" not in markdown
    assert "| Citations within the given sources | 1.00 [1.00, 1.00] | 2 |" in markdown


def test_write_report(summary: EvalSummary, tmp_path: Path) -> None:
    markdown_path, json_path = write_report(summary, tmp_path / "reports" / "eval")

    assert markdown_path.read_text(encoding="utf-8") == render_markdown(summary)
    assert EvalSummary.model_validate_json(json_path.read_text(encoding="utf-8")) == summary
    assert b"\r\n" not in markdown_path.read_bytes() + json_path.read_bytes()
