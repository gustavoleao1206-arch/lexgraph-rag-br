"""The ``lexgraph eval`` commands, driven through Typer's test runner with fakes."""

import json
import shutil
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, NoReturn

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from typer.testing import CliRunner

from lexgraph.agent.context import AgentContext
from lexgraph.agent.schemas import QueryAnalysis
from lexgraph.config import Settings
from lexgraph.eval import cli
from lexgraph.eval.golden import GOLDEN_FILE, GoldenItem
from lexgraph.eval.judge import DEFAULT_JUDGE_MODEL, Judge, TimedRateLimiter, rate_limiter
from lexgraph.eval.runner import (
    UNAVAILABLE_ERROR,
    AnswerRecord,
    JudgmentRecord,
    QueryAnalyzer,
    RetrievalRecord,
    read_records,
    write_records,
)
from lexgraph.protocols import ChatService, Retriever
from lexgraph.retrieval.store import StorageLockedError
from lexgraph.schemas import ErrorEvent, Usage
from lexgraph.service import LexGraphService
from tests.unit.eval.fakes import (
    CORPUS_IDS,
    GOLDEN_PATH,
    FakeChatService,
    FakeRetriever,
    StubJudge,
    answer_events,
    article_for,
    correctness_verdict,
    make_response,
    write_test_corpus,
)

runner = CliRunner()
OUT_DIR = Path("reports") / "eval"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> Settings:
    """Run from an empty folder with the golden fixture and a matching corpus in place.

    No provider key nor judge model is set, and any that a test's ``.env`` loads is removed
    afterwards.
    """
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", cli.JUDGE_MODEL_ENV):
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    settings.eval_dir.mkdir(parents=True)
    shutil.copy(GOLDEN_PATH, settings.eval_dir / GOLDEN_FILE)
    write_test_corpus(settings.corpus_dir, [article_for(article_id) for article_id in CORPUS_IDS])
    return settings


@pytest.fixture
def fake_retriever(
    monkeypatch: pytest.MonkeyPatch, golden_items: list[GoldenItem]
) -> FakeRetriever:
    rankings = {item.question: [f"{item.expected_articles[0]}#0"] for item in golden_items[:3]}
    retriever = FakeRetriever(
        {
            (mode, rerank): rankings
            for mode in ("dense", "sparse", "hybrid")
            for rerank in (False, True)
        }
    )

    @contextmanager
    def open_retriever(settings: Settings) -> Iterator[Retriever]:
        yield retriever

    monkeypatch.setattr(cli, "open_retriever", open_retriever)
    return retriever


def _serve(monkeypatch: pytest.MonkeyPatch, service: FakeChatService) -> list[BaseRateLimiter]:
    """Make the CLI use ``service``; returns the rate limiters it was opened with."""
    limiters: list[BaseRateLimiter] = []

    @asynccontextmanager
    async def open_chat_service(
        settings: Settings, *, limiter: BaseRateLimiter
    ) -> AsyncIterator[ChatService]:
        limiters.append(limiter)
        yield service

    monkeypatch.setattr(cli, "open_chat_service", open_chat_service)
    return limiters


@pytest.fixture
def fake_service(
    monkeypatch: pytest.MonkeyPatch, golden_items: list[GoldenItem]
) -> FakeChatService:
    usage = Usage(models=["openai/gpt-oss-120b", "gemini-3.5-flash-lite"])
    response = make_response(
        "Resposta com base no acervo [1].", context=["CDC:49"], cited=["CDC:49"], usage=usage
    )
    service = FakeChatService(
        {item.question: answer_events(response, "Resposta [1].") for item in golden_items}
    )
    _serve(monkeypatch, service)
    return service


def test_app_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("validate-golden", "retrieval", "answers", "judge", "report"):
        assert command in result.output


def test_validate_golden_accepts_a_consistent_set(workspace: Settings) -> None:
    result = runner.invoke(cli.app, ["validate-golden"])
    assert result.exit_code == 0, result.output
    assert "4 golden items are valid against 4 corpus articles." in result.output


def test_validate_golden_lists_problems(workspace: Settings, tmp_path: Path) -> None:
    corpus_dir = tmp_path / "partial-corpus"
    write_test_corpus(corpus_dir, [article_for("CDC:49"), article_for("CLT:59")])

    result = runner.invoke(cli.app, ["validate-golden", "--corpus-dir", str(corpus_dir)])

    assert result.exit_code == 1
    assert "- clt-001: law CF is not in the corpus" in result.output
    assert "- lgpd-001: LGPD:18 is not in the corpus" in result.output
    assert "4 problem(s) in 4 golden items." in result.output


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing_golden", "Golden set not found"),
        ("invalid_golden", "Invalid golden set"),
        ("missing_corpus", "Corpus not found"),
    ],
)
def test_validate_golden_reports_setup_errors(
    workspace: Settings, setup: str, message: str
) -> None:
    golden_path = workspace.eval_dir / GOLDEN_FILE
    if setup == "missing_golden":
        golden_path.unlink()
    elif setup == "invalid_golden":
        golden_path.write_text("{}\n", encoding="utf-8")
    else:
        shutil.rmtree(workspace.corpus_dir)

    result = runner.invoke(cli.app, ["validate-golden"])

    assert result.exit_code == 1
    assert message in result.output


def test_retrieval_writes_records_and_prints_the_table(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    result = runner.invoke(cli.app, ["retrieval", "--k", "5", "-c", "dense", "-c", "hybrid+rerank"])

    assert result.exit_code == 0, result.output
    records = read_records(OUT_DIR / cli.RETRIEVAL_FILE, RetrievalRecord)
    assert [record.config for record in records] == ["dense"] * 3 + ["hybrid+rerank"] * 3
    assert all(record.scores["hit@1"] == 1.0 for record in records)
    assert "## Retrieval" in result.output
    assert "| `hybrid+rerank` | 3 |" in result.output


def test_retrieval_measures_the_configured_reranking(
    workspace: Settings, fake_retriever: FakeRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = workspace.model_copy(update={"rerank_candidates": 8})
    monkeypatch.setattr(cli, "get_settings", lambda: configured)
    opened: list[Settings] = []

    @contextmanager
    def open_retriever(settings: Settings) -> Iterator[Retriever]:
        opened.append(settings)
        yield fake_retriever

    monkeypatch.setattr(cli, "open_retriever", open_retriever)

    result = runner.invoke(cli.app, ["retrieval", "--k", "5", "-c", "hybrid+rerank"])

    assert result.exit_code == 0, result.output
    assert opened == [configured]
    records = read_records(OUT_DIR / cli.RETRIEVAL_FILE, RetrievalRecord)
    assert {record.rerank_candidates for record in records} == {8}
    assert "`hybrid+rerank` reranks the first 8 passages of its search" in result.output


def test_retrieval_runs_every_configuration_by_default(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    result = runner.invoke(cli.app, ["retrieval"])

    assert result.exit_code == 0, result.output
    records = read_records(OUT_DIR / cli.RETRIEVAL_FILE, RetrievalRecord)
    assert list(dict.fromkeys(r.config for r in records)) == [
        "dense",
        "sparse",
        "hybrid",
        "hybrid+rerank",
    ]


@pytest.mark.parametrize("command", ["retrieval", "answers"])
def test_commands_explain_a_locked_vector_store(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    def locked(settings: Settings, **options: object) -> NoReturn:
        msg = "embedded Qdrant at data/qdrant is in use by another process"
        raise StorageLockedError(msg)

    monkeypatch.setattr(cli, "open_retriever", locked)
    monkeypatch.setattr(cli, "open_chat_service", locked)

    result = runner.invoke(cli.app, [command])

    assert result.exit_code == 1
    assert "is in use by another process. Stop the API server" in result.output


def test_retrieval_with_configs_replaces_only_their_results(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    assert runner.invoke(cli.app, ["retrieval", "--k", "5"]).exit_code == 0
    fake_retriever.rankings = {}

    result = runner.invoke(cli.app, ["retrieval", "--k", "5", "-c", "hybrid", "-c", "dense"])

    assert result.exit_code == 0, result.output
    records = read_records(OUT_DIR / cli.RETRIEVAL_FILE, RetrievalRecord)
    assert list(dict.fromkeys(r.config for r in records)) == [
        "dense",
        "sparse",
        "hybrid",
        "hybrid+rerank",
    ]
    hit = {r.config: r.scores["hit@1"] for r in records if r.item_id == "cdc-001"}
    assert hit == {"dense": 0.0, "sparse": 1.0, "hybrid": 0.0, "hybrid+rerank": 1.0}
    assert "Wrote 6 new records (12 in total)" in result.output


def test_retrieval_with_configs_needs_the_same_k(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    runner.invoke(cli.app, ["retrieval", "--k", "5"])
    fake_retriever.calls.clear()

    result = runner.invoke(cli.app, ["retrieval", "--k", "10", "-c", "hybrid"])

    assert result.exit_code == 1
    assert "holds results for k=5; pass the same --k" in result.output
    assert fake_retriever.calls == []


def test_retrieval_with_the_analyzer_needs_a_model_key(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    result = runner.invoke(cli.app, ["retrieval", "-c", "hybrid+analyzer"])
    assert result.exit_code == 1
    assert "set GROQ_API_KEY or GOOGLE_API_KEY" in result.output
    assert fake_retriever.calls == []


def test_retrieval_with_the_analyzer_searches_its_queries(
    workspace: Settings,
    fake_retriever: FakeRetriever,
    monkeypatch: pytest.MonkeyPatch,
    golden_items: list[GoldenItem],
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    rates: list[float] = []

    def build_query_analyzer(
        settings: Settings, retriever: Retriever, *, requests_per_minute: float
    ) -> QueryAnalyzer:
        rates.append(requests_per_minute)
        return lambda question: [f"consulta: {question}"]

    monkeypatch.setattr(cli, "build_query_analyzer", build_query_analyzer)
    fake_retriever.rankings = {
        ("hybrid", False): {
            f"consulta: {item.question}": [f"{item.expected_articles[0]}#0"]
            for item in golden_items[:3]
        }
    }

    result = runner.invoke(cli.app, ["retrieval", "-c", "hybrid+analyzer", "--rpm", "30"])

    assert result.exit_code == 0, result.output
    records = read_records(OUT_DIR / cli.RETRIEVAL_FILE, RetrievalRecord)
    assert [(r.config, r.scores["hit@1"]) for r in records] == [("hybrid+analyzer", 1.0)] * 3
    assert records[0].query == f"consulta: {golden_items[0].question}"
    assert rates == [30.0]
    assert "except `hybrid+analyzer`, which search the analyzer's queries" in result.output


def _agent_context(retriever: Retriever) -> AgentContext:
    analysis = QueryAnalysis.model_validate(
        {
            "intent": "legal_question",
            "standalone_question": "Pergunta.",
            "search_query": "direito de arrependimento",
            "sub_queries": [],
            "laws": [],
            "articles": [],
        }
    )

    def unused(_: LanguageModelInput) -> Any:
        msg = "not used by the evaluation"
        raise AssertionError(msg)

    return AgentContext(
        retriever=retriever,
        analyzer=RunnableLambda(lambda _: analysis),
        grader=RunnableLambda(unused),
        rewriter=RunnableLambda(unused),
        verifier=RunnableLambda(unused),
        answerer=RunnableLambda(unused),
    )


@pytest.fixture
def paced_contexts(monkeypatch: pytest.MonkeyPatch) -> list[BaseRateLimiter | None]:
    """Replace the agent's model wiring; returns the rate limiter of every context built."""
    limiters: list[BaseRateLimiter | None] = []

    def build_agent_context(
        settings: Settings, retriever: Retriever, *, rate_limiter: BaseRateLimiter | None = None
    ) -> AgentContext:
        limiters.append(rate_limiter)
        return _agent_context(retriever)

    monkeypatch.setattr(cli, "build_agent_context", build_agent_context)
    return limiters


def test_build_query_analyzer_paces_the_agent_analyzer(
    settings: Settings, paced_contexts: list[BaseRateLimiter | None]
) -> None:
    analyzer = cli.build_query_analyzer(settings, FakeRetriever({}), requests_per_minute=30)

    assert analyzer("Posso desistir da compra?") == ["direito de arrependimento"]
    (limiter,) = paced_contexts
    assert isinstance(limiter, InMemoryRateLimiter)
    assert limiter.requests_per_second == pytest.approx(0.5)


async def test_open_chat_service_paces_the_agent_and_keeps_threads_in_memory(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    paced_contexts: list[BaseRateLimiter | None],
) -> None:
    retriever = FakeRetriever({})
    opened: list[Settings] = []

    @contextmanager
    def open_retriever(current: Settings) -> Iterator[Retriever]:
        opened.append(current)
        yield retriever

    monkeypatch.setattr(cli, "open_retriever", open_retriever)

    limiter = rate_limiter(12)

    async with cli.open_chat_service(settings, limiter=limiter) as service:
        assert isinstance(service, LexGraphService)
        assert await service.history("eval-cdc-001-0000") == []

    assert opened == [settings]
    assert paced_contexts == [limiter]
    assert not settings.checkpoint_path.exists()


def test_retrieval_rejects_unknown_configurations(
    workspace: Settings, fake_retriever: FakeRetriever
) -> None:
    result = runner.invoke(cli.app, ["retrieval", "-c", "bm25"])
    assert result.exit_code == 1
    assert "Unknown configuration(s): bm25" in result.output
    assert fake_retriever.calls == []


def test_answers_records_every_question(
    workspace: Settings, fake_service: FakeChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    limiters = _serve(monkeypatch, fake_service)
    result = runner.invoke(cli.app, ["answers", "--limit", "3", "--rpm", "4"])

    assert result.exit_code == 0, result.output
    records = read_records(OUT_DIR / cli.ANSWERS_FILE, AnswerRecord)
    assert [record.item_id for record in records] == ["cdc-001", "clt-001", "lgpd-001"]
    assert "cdc-001: answered, 1 citation(s) (" in result.output
    assert "ms, plus 0 ms of pacing)" in result.output
    assert "Answered 3 question(s), 0 failed" in result.output

    resumed = runner.invoke(cli.app, ["answers"])
    assert "Answered 1 question(s), 0 failed" in resumed.output
    rates = [
        limiter.requests_per_second * 60
        for limiter in limiters
        if isinstance(limiter, TimedRateLimiter)
    ]
    assert rates == pytest.approx([4.0, cli.DEFAULT_ANSWER_RPM])


def test_answers_leave_the_pacing_out_of_the_timings(
    workspace: Settings, fake_service: FakeChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    limiters: list[TimedRateLimiter] = []

    def fast_rate_limiter(requests_per_minute: float) -> TimedRateLimiter:
        """A token every 0.1 s, which the fake service waits for like a paced model call."""
        limiter = rate_limiter(600)
        limiters.append(limiter)
        fake_service.limiter = limiter
        return limiter

    monkeypatch.setattr(cli, "rate_limiter", fast_rate_limiter)
    opened = _serve(monkeypatch, fake_service)

    result = runner.invoke(cli.app, ["answers", "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert opened == limiters
    records = read_records(OUT_DIR / cli.ANSWERS_FILE, AnswerRecord)
    # The waits of the agent's limiter are the ones recorded (the arithmetic is tested in
    # the runner's tests, on a fake clock).
    assert sum(record.paced_ms for record in records) == pytest.approx(limiters[0].waited_s * 1000)
    assert all(record.paced_ms > 0 for record in records)


def test_retrieval_needs_in_scope_questions(
    workspace: Settings, fake_retriever: FakeRetriever, golden_items: list[GoldenItem]
) -> None:
    golden_path = workspace.eval_dir / GOLDEN_FILE
    golden_path.write_text(golden_items[3].model_dump_json() + "\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["retrieval"])

    assert result.exit_code == 1
    assert "no in-scope questions" in result.output


def test_answers_and_judge_report_failures_and_refusals(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch, golden_items: list[GoldenItem]
) -> None:
    cdc, clt, lgpd, oos = golden_items
    answered = make_response("Resposta [1].", context=["CDC:49"], cited=["CDC:49"])
    refused = make_response("Fora do escopo do acervo.", route="out_of_scope")
    unavailable = make_response("Não consegui redigir a resposta agora.", route="unavailable")
    service = FakeChatService(
        {
            cdc.question: answer_events(answered, "Resposta [1]."),
            clt.question: answer_events(answered, "Resposta [1]."),
            lgpd.question: [ErrorEvent(message="Falha no provedor.")],
            oos.question: answer_events(refused),
        }
    )

    def correctness(inputs: dict[str, object]) -> object:
        if inputs["question"] == clt.question:
            msg = "quota exceeded"
            raise RuntimeError(msg)
        return correctness_verdict()

    stub = StubJudge(correctness=correctness)
    _serve(monkeypatch, service)
    monkeypatch.setattr(cli, "create_judge", lambda settings, **options: stub.judge)

    answers = runner.invoke(cli.app, ["answers"])
    judged = runner.invoke(cli.app, ["judge"])
    service.scripts = {**service.scripts, lgpd.question: answer_events(unavailable)}
    retried = runner.invoke(cli.app, ["answers"])

    assert "lgpd-001: error: Falha no provedor." in answers.output
    assert "Answered 4 question(s), 1 failed" in answers.output
    assert judged.exit_code == 0, judged.output
    assert "clt-001: error: RuntimeError: quota exceeded" in judged.output
    assert "oos-001: refused" in judged.output
    assert f"lgpd-001: error: {UNAVAILABLE_ERROR}" in retried.output
    assert "Answered 1 question(s), 1 failed" in retried.output


def test_judge_requires_recorded_answers(workspace: Settings) -> None:
    result = runner.invoke(cli.app, ["judge"])
    assert result.exit_code == 1
    assert "run the 'answers' command first" in result.output


def test_judge_writes_verdicts(
    workspace: Settings, fake_service: FakeChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(cli.app, ["answers"])
    stub = StubJudge()
    created: list[dict[str, object]] = []

    def create_judge(settings: Settings, **options: object) -> Judge:
        created.append(options)
        return stub.judge

    monkeypatch.setattr(cli, "create_judge", create_judge)

    result = runner.invoke(
        cli.app, ["judge", "--judge-model", "google_genai:judge-test", "--rpm", "6"]
    )

    assert result.exit_code == 0, result.output
    assert created == [
        {
            "model_spec": "google_genai:judge-test",
            "requests_per_minute": 6.0,
            "cache_path": OUT_DIR / cli.JUDGE_CACHE_FILE,
        }
    ]
    records = read_records(OUT_DIR / cli.JUDGMENTS_FILE, JudgmentRecord)
    assert len(records) == 4
    assert "cdc-001: correct, groundedness 1.0" in result.output
    assert "Judged 4 answer(s) with google_genai:judge-test (0 new tokens)" in result.output
    assert "judge's family" not in result.output


def test_judge_reads_the_model_from_the_environment_file(
    workspace: Settings, fake_service: FakeChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(cli.app, ["answers"])
    Path(".env").write_text(f"{cli.JUDGE_MODEL_ENV}={DEFAULT_JUDGE_MODEL}\n", encoding="utf-8")
    created: list[object] = []

    def create_judge(settings: Settings, **options: object) -> Judge:
        created.append(options["model_spec"])
        return StubJudge(model_name=DEFAULT_JUDGE_MODEL).judge

    monkeypatch.setattr(cli, "create_judge", create_judge)

    result = runner.invoke(cli.app, ["judge"])

    assert result.exit_code == 0, result.output
    assert created == [DEFAULT_JUDGE_MODEL]
    assert "4 answer(s) were produced with a model of the judge's family" in result.output


def test_judge_rejects_an_unknown_provider(
    workspace: Settings, fake_service: FakeChatService
) -> None:
    runner.invoke(cli.app, ["answers"])
    result = runner.invoke(cli.app, ["judge", "--judge-model", "foo:bar"])
    assert result.exit_code == 1
    assert "provider" in result.output


def test_report_requires_results(workspace: Settings) -> None:
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code == 1
    assert "No results in" in result.output


def test_report_writes_the_summaries(
    workspace: Settings, fake_retriever: FakeRetriever, fake_service: FakeChatService
) -> None:
    runner.invoke(cli.app, ["retrieval", "-c", "dense"])
    runner.invoke(cli.app, ["answers"])

    result = runner.invoke(cli.app, ["report"])

    assert result.exit_code == 0, result.output
    markdown = (OUT_DIR / "summary.md").read_text(encoding="utf-8")
    assert "## Retrieval" in markdown
    assert "## Agent behavior" in markdown
    summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["golden"]["items"] == 4
    assert summary["agent"]["answered"] == 4
    assert summary["answers"]["judge"] is None
    assert "left out" not in result.output


def test_report_warns_about_judgments_of_earlier_answers(
    workspace: Settings, fake_service: FakeChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(cli.app, ["answers"])
    monkeypatch.setattr(cli, "create_judge", lambda settings, **options: StubJudge().judge)
    runner.invoke(cli.app, ["judge"])
    answers_path = OUT_DIR / cli.ANSWERS_FILE
    later = {"started_at": "2026-10-31T09:00:00+00:00"}
    write_records(
        answers_path,
        [record.model_copy(update=later) for record in read_records(answers_path, AnswerRecord)],
    )

    result = runner.invoke(cli.app, ["report"])

    assert result.exit_code == 0, result.output
    assert "4 judgment(s) refer to earlier answers and were left out" in result.output
