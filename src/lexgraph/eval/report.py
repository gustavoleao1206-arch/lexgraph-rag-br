"""Summaries of the evaluation runs, as JSON (for tooling) and Markdown (for the README).

Each section appears only when its results exist:

* retrieval: per-configuration means with 95% bootstrap intervals, p50 latency, paired
  differences against the first configuration, a breakdown by question type and the recall
  of the specific units (incisos, paragraphs) that some questions need from long articles;
* answer quality: rule-based checks (required facts, forbidden claims, citations) and judge
  verdicts (correctness, groundedness, CRAG-style truthfulness) on the current answers;
* agent behavior: route mix, rewrites, regenerations, refusal precision and recall, whether
  the expected articles reached the answer model's context, latency and time-to-first-token
  percentiles (without the waits of the evaluation's rate limiter, reported apart), tokens
  per question.
"""

import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from lexgraph.eval.golden import GoldenItem
from lexgraph.eval.judge import CorrectnessLabel
from lexgraph.eval.metrics import (
    Estimate,
    LatencyStats,
    PairedDifference,
    estimate,
    latency_stats,
    paired_bootstrap,
)
from lexgraph.eval.rules import RefusalStats, is_refusal, refusal_stats, score_answer
from lexgraph.eval.runner import (
    AnswerRecord,
    JudgmentRecord,
    RetrievalRecord,
    latest_answers,
    unit_recall_metric,
)
from lexgraph.schemas import ChatResponse

SUMMARY_JSON = "summary.json"
SUMMARY_MARKDOWN = "summary.md"
CORRECTNESS_LABELS: tuple[CorrectnessLabel, ...] = ("correct", "partial", "incorrect")

_METRIC_LABELS = {"hit": "Hit", "recall": "Recall", "mrr": "MRR", "ndcg": "nDCG"}


class Rate(BaseModel):
    """A count out of a total; ``value`` is ``None`` when the total is zero."""

    count: int
    total: int
    value: float | None

    @classmethod
    def of(cls, count: int, total: int) -> Self:
        """Build the rate ``count / total``."""
        return cls(count=count, total=total, value=count / total if total else None)


class GoldenStats(BaseModel):
    """Composition of the golden set."""

    items: int
    in_scope: int
    out_of_scope: int
    by_type: dict[str, int]


class ConfigSummary(BaseModel):
    """Retrieval quality and latency of one configuration.

    ``analyzed`` tells that it searched the analyzer's queries instead of the questions;
    ``unit_recall`` covers only the questions that need specific units of long articles;
    ``rerank_candidates`` is the number of passages the cross-encoder scored per search
    (``None`` without reranking, or when the records do not say).
    """

    name: str
    analyzed: bool
    n: int
    metrics: dict[str, Estimate]
    unit_recall: Estimate | None
    latency_ms: LatencyStats
    rerank_candidates: int | None = None


class Comparison(BaseModel):
    """Paired difference between a configuration and the baseline."""

    metric: str
    baseline: str
    candidate: str
    difference: PairedDifference


class TypeBreakdown(BaseModel):
    """Mean of the headline metric per configuration for one question type."""

    question_type: str
    n: int
    means: dict[str, float]


class RetrievalSummary(BaseModel):
    """Retrieval ablation over the in-scope questions."""

    k: int
    headline_metric: str
    configs: list[ConfigSummary]
    comparisons: list[Comparison]
    by_type: list[TypeBreakdown]


class JudgeSummary(BaseModel):
    """Aggregated judge verdicts.

    Correctness is averaged over the judged (not refused) in-scope answers; refusals count in
    truthfulness instead. ``same_family`` counts answers produced with a model of the judge's
    family, whose verdicts may suffer from self-preference.
    """

    judge_models: list[str]
    judged: int
    refused: int
    errors: int
    same_family: int
    verdicts: dict[str, int]
    correctness: Estimate | None
    groundedness: Estimate | None
    truthfulness: Estimate | None


class AnswerSummary(BaseModel):
    """Quality of the final answers; ``stale_judgments`` counts judgments of earlier answers."""

    scored: int
    must_include: Estimate | None
    must_not_include: Estimate | None
    citation_precision: Estimate | None
    citation_recall: Estimate | None
    citations_in_context: Estimate | None
    judge: JudgeSummary | None
    stale_judgments: int


class TokenStats(BaseModel):
    """Mean tokens per answered question, summed over every model call."""

    input: float
    output: float
    total: float


class AgentSummary(BaseModel):
    """How the agent behaved while answering the golden questions.

    ``latency_ms`` and ``ttft_ms`` leave out the time the evaluation's rate limiter held model
    calls back; ``paced_ms`` gives that time per question (``None`` when nothing was paced).
    """

    answered: int
    errors: int
    routes: dict[str, int]
    rewrite_rate: Rate
    mean_rewrites: float | None
    regeneration_rate: Rate
    self_check_pass_rate: Rate
    refusal: RefusalStats
    context_hit: Estimate | None
    context_recall: Estimate | None
    latency_ms: LatencyStats | None
    ttft_ms: LatencyStats | None
    paced_ms: LatencyStats | None = None
    tokens_per_question: TokenStats | None
    models: dict[str, int]


class EvalSummary(BaseModel):
    """Everything the report shows."""

    generated_at: str
    golden: GoldenStats
    retrieval: RetrievalSummary | None = None
    answers: AnswerSummary | None = None
    agent: AgentSummary | None = None


def build_summary(
    items: Sequence[GoldenItem],
    *,
    retrieval: Sequence[RetrievalRecord] = (),
    answers: Sequence[AnswerRecord] = (),
    judgments: Sequence[JudgmentRecord] = (),
    generated_at: datetime | None = None,
) -> EvalSummary:
    """Summarize the recorded results; records of items no longer in the golden set are ignored."""
    by_id = {item.id: item for item in items}
    searched = [record for record in retrieval if record.item_id in by_id]
    judged = [record for record in judgments if record.item_id in by_id]
    return EvalSummary(
        generated_at=(generated_at or datetime.now(UTC)).isoformat(timespec="seconds"),
        golden=_golden_stats(items),
        retrieval=summarize_retrieval(searched) if searched else None,
        answers=summarize_answers(by_id, answers, judged) if answers or judged else None,
        agent=summarize_agent(by_id, answers) if answers else None,
    )


def _golden_stats(items: Sequence[GoldenItem]) -> GoldenStats:
    out_of_scope = sum(item.should_refuse for item in items)
    return GoldenStats(
        items=len(items),
        in_scope=len(items) - out_of_scope,
        out_of_scope=out_of_scope,
        by_type=dict(sorted(Counter(item.type for item in items).items())),
    )


def headline_metrics(k: int, available: Iterable[str]) -> list[str]:
    """Retrieval metrics shown in the report, among those that were computed."""
    candidates = ("hit@1", "hit@5", "recall@5", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}")
    present = set(available)
    return [name for name in dict.fromkeys(candidates) if name in present]


def summarize_retrieval(records: Sequence[RetrievalRecord]) -> RetrievalSummary:
    """Aggregate retrieval records; the first configuration is the comparison baseline."""
    if not records:
        msg = "no retrieval records to summarize"
        raise ValueError(msg)
    k = records[0].k
    by_config: dict[str, list[RetrievalRecord]] = {}
    for record in records:
        by_config.setdefault(record.config, []).append(record)
    metrics = headline_metrics(k, records[0].scores)
    headline = f"ndcg@{k}"
    unit_metric = unit_recall_metric(k)
    configs = [
        ConfigSummary(
            name=name,
            analyzed=any(row.query is not None for row in rows),
            n=len(rows),
            metrics={metric: estimate([row.scores[metric] for row in rows]) for metric in metrics},
            unit_recall=_estimate(
                [row.scores[unit_metric] for row in rows if unit_metric in row.scores]
            ),
            latency_ms=latency_stats([row.latency_ms for row in rows]),
            rerank_candidates=max(
                (row.rerank_candidates for row in rows if row.rerank_candidates is not None),
                default=None,
            ),
        )
        for name, rows in by_config.items()
    ]
    return RetrievalSummary(
        k=k,
        headline_metric=headline,
        configs=configs,
        comparisons=_comparisons(by_config, headline),
        by_type=_by_type(by_config, headline),
    )


def _comparisons(
    by_config: Mapping[str, Sequence[RetrievalRecord]], metric: str
) -> list[Comparison]:
    baseline_name, *candidates = by_config
    baseline = {row.item_id: row.scores[metric] for row in by_config[baseline_name]}
    comparisons: list[Comparison] = []
    for name in candidates:
        candidate = {row.item_id: row.scores[metric] for row in by_config[name]}
        shared = [item_id for item_id in baseline if item_id in candidate]
        if shared:
            difference = paired_bootstrap(
                [baseline[item_id] for item_id in shared],
                [candidate[item_id] for item_id in shared],
            )
            comparisons.append(
                Comparison(
                    metric=metric, baseline=baseline_name, candidate=name, difference=difference
                )
            )
    return comparisons


def _by_type(
    by_config: Mapping[str, Sequence[RetrievalRecord]], metric: str
) -> list[TypeBreakdown]:
    question_types = dict.fromkeys(row.question_type for rows in by_config.values() for row in rows)
    breakdown: list[TypeBreakdown] = []
    for question_type in sorted(question_types):
        values = {
            name: [row.scores[metric] for row in rows if row.question_type == question_type]
            for name, rows in by_config.items()
        }
        breakdown.append(
            TypeBreakdown(
                question_type=question_type,
                n=max(len(scores) for scores in values.values()),
                means={name: statistics.fmean(scores) for name, scores in values.items() if scores},
            )
        )
    return breakdown


def _answered(
    items: Mapping[str, GoldenItem], answers: Iterable[AnswerRecord]
) -> list[tuple[GoldenItem, AnswerRecord, ChatResponse]]:
    """Latest answer of each golden item, when it succeeded."""
    return [
        (items[record.item_id], record, response)
        for record in latest_answers(answers).values()
        if record.item_id in items and (response := record.result) is not None
    ]


def _estimate(values: Sequence[float]) -> Estimate | None:
    return estimate(values) if values else None


def summarize_answers(
    items: Mapping[str, GoldenItem],
    answers: Iterable[AnswerRecord],
    judgments: Iterable[JudgmentRecord] = (),
) -> AnswerSummary:
    """Rule-based scores of the latest answers plus the judge verdicts on those answers.

    A judgment of any other answer (one replaced by a later ``answers`` run) is left out and
    counted in ``stale_judgments``.
    """
    answered = _answered(items, answers)
    scores = [score_answer(item, response) for item, _, response in answered]
    started = {record.item_id: record.started_at for _, record, _ in answered}
    every_judgment = list(judgments)
    judged = [j for j in every_judgment if started.get(j.item_id) == j.answer_started_at]
    return AnswerSummary(
        scored=len(scores),
        must_include=_estimate(
            [float(s.must_include_passed) for s in scores if s.must_include_passed is not None]
        ),
        must_not_include=_estimate(
            [
                float(s.must_not_include_passed)
                for s in scores
                if s.must_not_include_passed is not None
            ]
        ),
        citation_precision=_estimate(
            [s.citations.precision for s in scores if s.citations.precision is not None]
        ),
        citation_recall=_estimate(
            [s.citations.recall for s in scores if s.citations.recall is not None]
        ),
        citations_in_context=_estimate(
            [s.citations.in_context for s in scores if s.citations.in_context is not None]
        ),
        judge=summarize_judgments(judged) if judged else None,
        stale_judgments=len(every_judgment) - len(judged),
    )


def summarize_judgments(judgments: Sequence[JudgmentRecord]) -> JudgeSummary:
    """Verdict counts and mean judge scores."""
    verdicts = Counter(r.correctness.verdict for r in judgments if r.correctness is not None)
    return JudgeSummary(
        judge_models=sorted({r.judge_model for r in judgments}),
        judged=len(judgments),
        refused=sum(r.refused for r in judgments),
        errors=sum(r.error is not None for r in judgments),
        same_family=sum(r.same_family for r in judgments),
        verdicts={label: verdicts[label] for label in CORRECTNESS_LABELS},
        correctness=_estimate(
            [r.correctness_score for r in judgments if r.correctness_score is not None]
        ),
        groundedness=_estimate(
            [r.groundedness_score for r in judgments if r.groundedness_score is not None]
        ),
        truthfulness=_estimate([r.truthfulness for r in judgments if r.truthfulness is not None]),
    )


def summarize_agent(
    items: Mapping[str, GoldenItem], answers: Iterable[AnswerRecord]
) -> AgentSummary:
    """Route mix, loop statistics, refusals, context, latency and token usage of the latest answers.

    The context scores check, for in-scope questions, whether the expected articles were
    among the sources given to the answer model. Latencies come without the rate limiter's
    waits, which are summarized apart when any call was paced.
    """
    latest = [record for record in latest_answers(answers).values() if record.item_id in items]
    answered = _answered(items, latest)
    responses = [response for _, _, response in answered]
    in_scope = [response for item, _, response in answered if not item.should_refuse]
    context = [_context_scores(item, r) for item, _, r in answered if not item.should_refuse]
    grounded = [r.grounding.grounded for r in responses if r.grounding is not None]
    ttfts = [record.ttft_ms for _, record, _ in answered if record.ttft_ms is not None]
    paced = [record.paced_ms for _, record, _ in answered]
    return AgentSummary(
        answered=len(answered),
        errors=len(latest) - len(answered),
        routes=dict(sorted(Counter(r.route for r in responses).items())),
        rewrite_rate=Rate.of(sum(r.rewrites > 0 for r in in_scope), len(in_scope)),
        mean_rewrites=statistics.fmean(r.rewrites for r in in_scope) if in_scope else None,
        regeneration_rate=Rate.of(sum(r.regenerations > 0 for r in responses), len(responses)),
        self_check_pass_rate=Rate.of(sum(grounded), len(grounded)),
        refusal=refusal_stats((item.should_refuse, is_refusal(r)) for item, _, r in answered),
        context_hit=_estimate([hit for hit, _ in context]),
        context_recall=_estimate([recall for _, recall in context]),
        latency_ms=latency_stats([rec.latency_ms for _, rec, _ in answered]) if answered else None,
        ttft_ms=latency_stats(ttfts) if ttfts else None,
        paced_ms=latency_stats(paced) if any(paced) else None,
        tokens_per_question=_token_stats(responses) if responses else None,
        models=dict(
            sorted(Counter(m for r in responses for m in dict.fromkeys(r.usage.models)).items())
        ),
    )


def _context_scores(item: GoldenItem, response: ChatResponse) -> tuple[float, float]:
    """Hit and recall of the expected articles among the sources of ``response``."""
    expected = set(item.expected_articles)
    found = len(expected & {source.article_id for source in response.sources})
    return float(found > 0), found / len(expected)


def _token_stats(responses: Sequence[ChatResponse]) -> TokenStats:
    return TokenStats(
        input=statistics.fmean(r.usage.input_tokens for r in responses),
        output=statistics.fmean(r.usage.output_tokens for r in responses),
        total=statistics.fmean(r.usage.total_tokens for r in responses),
    )


def render_markdown(summary: EvalSummary) -> str:
    """The report as Markdown, ready to paste into the README."""
    golden = summary.golden
    lines = [
        "# Evaluation report",
        "",
        (
            f"Golden set: {golden.items} questions ({golden.in_scope} in scope, "
            f"{golden.out_of_scope} out of scope)."
        ),
        "",
    ]
    if summary.retrieval is not None:
        lines += retrieval_markdown(summary.retrieval)
    if summary.answers is not None:
        lines += _answers_markdown(summary.answers)
    if summary.agent is not None:
        lines += _agent_markdown(summary.agent)
    lines.append(
        f"_Generated {summary.generated_at}. Values in brackets are 95% bootstrap confidence "
        "intervals over questions._"
    )
    return "\n".join(lines) + "\n"


def retrieval_markdown(summary: RetrievalSummary) -> list[str]:
    """Markdown lines of the retrieval section."""
    metrics = list(summary.configs[0].metrics)
    headline = _metric_label(summary.headline_metric)
    analyzed = [f"`{config.name}`" for config in summary.configs if config.analyzed]
    search = "Configurations search the raw question, without the agent's query analysis"
    if analyzed:
        listed = " and ".join(
            [", ".join(analyzed[:-1]), analyzed[-1]] if analyzed[1:] else analyzed
        )
        search += f", except {listed}, which search the analyzer's queries"
    notes = [
        f"{search}.",
        *(
            f"`{config.name}` reranks the first {config.rerank_candidates} passages of its "
            f"search with the cross-encoder and keeps the best {summary.k}."
            for config in summary.configs
            if config.rerank_candidates is not None
        ),
    ]
    lines = [
        "## Retrieval",
        "",
        (
            f"Top {summary.k} passages per question, collapsed to articles (any slice of a long "
            "article counts as the article); expected articles have grade 2 and acceptable "
            f"ones grade 1. {' '.join(notes)}"
        ),
        "",
        *_table(
            ["Config", "n", *(_metric_label(m) for m in metrics), "p50 ms"],
            (
                [
                    f"`{config.name}`",
                    str(config.n),
                    *(_format_estimate(config.metrics[m]) for m in metrics),
                    f"{config.latency_ms.p50:.0f}",
                ]
                for config in summary.configs
            ),
        ),
        "",
    ]
    if summary.comparisons:
        lines += [
            f"Paired differences in {headline} against `{summary.comparisons[0].baseline}`:",
            "",
            *_table(
                ["Config", "Mean diff", "95% CI", "P(diff <= 0)"],
                (
                    [
                        f"`{c.candidate}`",
                        f"{c.difference.mean:+.3f}",
                        f"[{c.difference.low:+.3f}, {c.difference.high:+.3f}]",
                        f"{c.difference.share_not_better:.3f}",
                    ]
                    for c in summary.comparisons
                ),
            ),
            "",
        ]
    names = [config.name for config in summary.configs]
    lines += [
        f"{headline} by question type:",
        "",
        *_table(
            ["Type", "n", *(f"`{name}`" for name in names)],
            (
                [row.question_type, str(row.n), *(_format(row.means.get(n)) for n in names)]
                for row in summary.by_type
            ),
        ),
        "",
    ]
    unit_rows = [config for config in summary.configs if config.unit_recall is not None]
    if unit_rows:
        lines += [
            (
                f"Unit recall@{summary.k} (questions that need a specific inciso or paragraph of "
                "a long article: share of those units among the retrieved slices):"
            ),
            "",
            *_table(
                ["Config", "n", f"Unit recall@{summary.k}"],
                (
                    [
                        f"`{config.name}`",
                        str(config.unit_recall.n),
                        _format_estimate(config.unit_recall),
                    ]
                    for config in unit_rows
                    if config.unit_recall is not None
                ),
            ),
            "",
        ]
    return lines


def _answers_markdown(summary: AnswerSummary) -> list[str]:
    rows = [
        ("Required facts present (must_include)", summary.must_include),
        ("Forbidden claims avoided (must_not_include)", summary.must_not_include),
        ("Citation precision", summary.citation_precision),
        ("Citation recall", summary.citation_recall),
        ("Citations within the given sources", summary.citations_in_context),
    ]
    judge = summary.judge
    if judge is not None:
        rows += [
            ("Correctness of non-refused answers (judge: 1, 0.5, 0)", judge.correctness),
            ("Groundedness (judge: share of supported claims)", judge.groundedness),
            ("Truthfulness (CRAG: 1, 0.5, 0, -1)", judge.truthfulness),
        ]
    lines = [
        "## Answer quality",
        "",
        *_table(
            ["Metric", "Mean [95% CI]", "n"],
            (
                [label, _format_estimate(value), str(value.n if value else 0)]
                for label, value in rows
            ),
        ),
        "",
    ]
    if judge is not None:
        verdicts = ", ".join(f"{label} {count}" for label, count in judge.verdicts.items())
        models = ", ".join(f"`{model}`" for model in judge.judge_models)
        lines += [
            (
                f"Judge {models}: {verdicts}; {judge.refused} refusal(s) not judged; "
                f"{judge.errors} error(s); {judge.same_family} answer(s) produced with a model "
                "of the judge's family."
            ),
            "",
        ]
    if summary.stale_judgments:
        lines += [
            (
                f"{summary.stale_judgments} judgment(s) of earlier answers were left out; run "
                "`lexgraph eval judge` again."
            ),
            "",
        ]
    return lines


def _agent_markdown(summary: AgentSummary) -> list[str]:
    refusal = summary.refusal
    rows = [
        ("Questions answered", f"{summary.answered} ({summary.errors} errors)"),
        ("Routes", _format_counts(summary.routes)),
        ("Rewrite rate (in scope)", _format_rate(summary.rewrite_rate)),
        ("Mean rewrites (in scope)", _format(summary.mean_rewrites)),
        ("Regeneration rate", _format_rate(summary.regeneration_rate)),
        ("Self-check pass rate", _format_rate(summary.self_check_pass_rate)),
        ("Expected article in the context (in scope)", _format_estimate(summary.context_hit)),
        (
            "Expected articles in the context, recall (in scope)",
            _format_estimate(summary.context_recall),
        ),
        (
            "Refusal precision",
            _format_rate(
                Rate.of(refusal.true_positives, refusal.true_positives + refusal.false_positives)
            ),
        ),
        (
            "Refusal recall",
            _format_rate(
                Rate.of(refusal.true_positives, refusal.true_positives + refusal.false_negatives)
            ),
        ),
        (
            "False refusal rate (in scope)",
            _format_rate(
                Rate.of(refusal.false_positives, refusal.false_positives + refusal.true_negatives)
            ),
        ),
        ("Latency p50 / p95 (ms)", _format_latency(summary.latency_ms)),
        ("Time to first token p50 / p95 (ms)", _format_latency(summary.ttft_ms)),
        *_pacing_rows(summary),
        ("Tokens per question (input / output / total)", _format_tokens(summary)),
        ("Models", _format_counts(summary.models)),
    ]
    return ["## Agent behavior", "", *_table(["Metric", "Value"], rows), ""]


def _pacing_rows(summary: AgentSummary) -> list[tuple[str, str]]:
    """The rate limiter's waits per question, when the answers were paced."""
    if summary.paced_ms is None:
        return []
    label = "Rate-limit pacing left out of the two timings above, p50 / p95 (ms)"
    return [(label, _format_latency(summary.paced_ms))]


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = [f"| {' | '.join(header)} |", "|---|" + "---:|" * (len(header) - 1)]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def _metric_label(name: str) -> str:
    kind, _, cutoff = name.partition("@")
    return f"{_METRIC_LABELS.get(kind, kind)}@{cutoff}"


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_estimate(value: Estimate | None) -> str:
    if value is None:
        return "n/a"
    return f"{value.mean:.2f} [{value.low:.2f}, {value.high:.2f}]"


def _format_rate(rate: Rate) -> str:
    return f"{_format(rate.value)} ({rate.count}/{rate.total})"


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items()) or "n/a"


def _format_latency(stats: LatencyStats | None) -> str:
    return "n/a" if stats is None else f"{stats.p50:.0f} / {stats.p95:.0f}"


def _format_tokens(summary: AgentSummary) -> str:
    tokens = summary.tokens_per_question
    if tokens is None:
        return "n/a"
    return f"{tokens.input:.0f} / {tokens.output:.0f} / {tokens.total:.0f}"


def write_report(summary: EvalSummary, out_dir: Path) -> tuple[Path, Path]:
    """Write ``summary.md`` and ``summary.json`` into ``out_dir``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / SUMMARY_MARKDOWN
    json_path = out_dir / SUMMARY_JSON
    markdown_path.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    json_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return markdown_path, json_path
