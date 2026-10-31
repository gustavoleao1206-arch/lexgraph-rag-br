"""Command-line interface of the evaluation harness (mounted as ``lexgraph eval``).

Typical run, from the repository root::

    lexgraph eval validate-golden
    lexgraph eval retrieval        # no language model involved
    lexgraph eval answers --delay 5
    lexgraph eval judge
    lexgraph eval report           # reports/eval/summary.md and summary.json

``retrieval`` compares the default configurations on the raw questions; the opt-in
``-c dense+analyzer`` and ``-c hybrid+analyzer`` search the queries written by the agent's
analyzer model instead. The retriever keeps the configured settings, as in the application:
``hybrid+rerank`` lets the cross-encoder score ``max(k, LEXGRAPH_RERANK_CANDIDATES)`` passages.
``answers`` paces every model call of the agent (``--rpm``) and keeps its conversation threads
in memory; the recorded latencies leave out the time calls wait for that pacing. ``judge`` uses
``google_genai:gemini-3.8-flash`` unless that model writes answers in the current settings (see
:func:`lexgraph.eval.judge.default_judge_model`); choose another with ``--judge-model`` or the
``LEXGRAPH_JUDGE_MODEL`` environment variable. Commands read provider keys from ``.env``.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.rate_limiters import BaseRateLimiter
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from rich.console import Console

from lexgraph.agent.graph import build_graph
from lexgraph.agent.llm import build_agent_context, models_configured, provider_configured
from lexgraph.config import Settings, bootstrap_env, get_settings
from lexgraph.container import read_corpus_date
from lexgraph.corpus import Corpus
from lexgraph.eval.golden import (
    GOLDEN_FILE,
    GoldenItem,
    GoldenSetError,
    load_golden,
    validate_against_corpus,
)
from lexgraph.eval.judge import create_judge, rate_limiter
from lexgraph.eval.report import (
    build_summary,
    retrieval_markdown,
    summarize_retrieval,
    write_report,
)
from lexgraph.eval.rules import check_reference
from lexgraph.eval.runner import (
    CONFIGS,
    DEFAULT_CONFIGS,
    AnswerRecord,
    JudgmentRecord,
    QueryAnalyzer,
    RetrievalConfig,
    RetrievalRecord,
    agent_query_analyzer,
    read_records,
    run_answers,
    run_judge,
    run_retrieval_eval,
    write_records,
)
from lexgraph.protocols import ChatService, Retriever
from lexgraph.service import LexGraphService

DEFAULT_OUT_DIR = Path("reports/eval")
RETRIEVAL_FILE = "retrieval.jsonl"
ANSWERS_FILE = "answers.jsonl"
JUDGMENTS_FILE = "judgments.jsonl"
JUDGE_CACHE_FILE = "judge_cache.jsonl"
JUDGE_MODEL_ENV = "LEXGRAPH_JUDGE_MODEL"
# A question takes 4 to 6 model calls; Groq's free tier allows 8K tokens per model per minute.
DEFAULT_ANSWER_RPM = 6.0
DEFAULT_ANALYZER_RPM = 20.0

app = typer.Typer(
    help="Evaluate retrieval and answers against the golden set.", no_args_is_help=True
)
console = Console(markup=False, highlight=False, soft_wrap=True)

GoldenOption = Annotated[
    Path | None,
    typer.Option(
        "--golden", "-g", help="Golden set JSONL (default: <data dir>/eval/golden.jsonl)."
    ),
]
OutDirOption = Annotated[
    Path, typer.Option("--out-dir", "-o", help="Folder for the result files and the report.")
]
LimitOption = Annotated[
    int | None, typer.Option("--limit", "-n", min=1, help="Process at most this many questions.")
]


@app.callback()
def _load_env() -> None:
    """Load provider keys and options such as LEXGRAPH_JUDGE_MODEL from .env."""
    bootstrap_env()


@contextmanager
def open_retriever(settings: Settings) -> Iterator[Retriever]:
    """Open the configured vector store and yield the production retriever."""
    from lexgraph.retrieval.factory import build_retriever
    from lexgraph.retrieval.store import create_qdrant_client

    client = create_qdrant_client(settings)
    try:
        yield build_retriever(settings, client=client)
    finally:
        client.close()


@asynccontextmanager
async def open_chat_service(
    settings: Settings, *, limiter: BaseRateLimiter
) -> AsyncIterator[ChatService]:
    """The production agent with every model call paced by ``limiter``.

    Conversation threads stay in memory: evaluation questions never reach the thread store
    of the API.
    """
    with open_retriever(settings) as retriever:
        context = build_agent_context(settings, retriever, rate_limiter=limiter)
        yield LexGraphService(
            build_graph(InMemorySaver()),
            context,
            settings,
            corpus_date=read_corpus_date(settings.corpus_dir),
            models_configured=models_configured(settings),
        )


def build_query_analyzer(
    settings: Settings, retriever: Retriever, *, requests_per_minute: float
) -> QueryAnalyzer:
    """The agent's analyzer model, paced to ``requests_per_minute``, as a query writer."""
    context = build_agent_context(
        settings, retriever, rate_limiter=rate_limiter(requests_per_minute)
    )
    return agent_query_analyzer(context)


@contextmanager
def _vector_store_guard() -> Iterator[None]:
    """Turn the "embedded Qdrant already open" failure into an actionable message."""
    from lexgraph.retrieval.store import StorageLockedError

    try:
        yield
    except StorageLockedError as exc:
        _fail(f"{exc}. Stop the API server, or set LEXGRAPH_QDRANT_URL to use a Qdrant server.")


@app.command("validate-golden")
def validate_golden(
    golden: GoldenOption = None,
    corpus_dir: Annotated[
        Path | None, typer.Option(help="Parsed corpus folder (default: <data dir>/corpus).")
    ] = None,
) -> None:
    """Check the golden set: schema, article ids against the corpus and reference answers."""
    settings = get_settings()
    items = _load_items(golden, settings)
    corpus = _load_corpus(corpus_dir or settings.corpus_dir)
    problems = validate_against_corpus(items, corpus)
    problems += [problem for item in items for problem in check_reference(item)]
    for problem in problems:
        console.print(f"- {problem}")
    if problems:
        _fail(f"{len(problems)} problem(s) in {len(items)} golden items.")
    console.print(f"{len(items)} golden items are valid against {len(corpus)} corpus articles.")


@app.command()
def retrieval(
    golden: GoldenOption = None,
    k: Annotated[int, typer.Option(min=1, max=50, help="Passages fetched per question.")] = 10,
    config: Annotated[
        list[str] | None,
        typer.Option(
            "--config",
            "-c",
            help=(
                "Configuration to run (repeatable): dense, sparse, hybrid, hybrid+rerank, or "
                "dense+analyzer, hybrid+analyzer (search the agent analyzer's queries; need a "
                "model API key). Without it the first four run and replace the results file; "
                "with it the named configurations replace only their own earlier results."
            ),
        ),
    ] = None,
    rpm: Annotated[
        float,
        typer.Option(min=0.1, help="Analyzer requests per minute (analyzer configurations only)."),
    ] = DEFAULT_ANALYZER_RPM,
    out_dir: OutDirOption = DEFAULT_OUT_DIR,
) -> None:
    """Compare retrieval configurations on the in-scope questions."""
    settings = get_settings()
    items = _load_items(golden, settings)
    configs = _select_configs(config)
    path = out_dir / RETRIEVAL_FILE
    kept = _kept_results(path, configs, k) if config else []
    analyzed = any(selected.analyzed for selected in configs)
    if analyzed and not _fast_models_configured(settings):
        _fail("The analyzer configurations call a model: set GROQ_API_KEY or GOOGLE_API_KEY.")
    with _vector_store_guard(), open_retriever(settings) as retriever:
        analyzer = (
            build_query_analyzer(settings, retriever, requests_per_minute=rpm) if analyzed else None
        )
        records = run_retrieval_eval(
            retriever,
            items,
            configs,
            k=k,
            analyzer=analyzer,
            rerank_candidates=settings.rerank_candidates,
        )
    if not records:
        _fail("The golden set has no in-scope questions.")
    results = _in_config_order([*kept, *records])
    write_records(path, results)
    console.print("\n".join(retrieval_markdown(summarize_retrieval(results))))
    console.print(f"Wrote {len(records)} new records ({len(results)} in total) to {path}")


@app.command()
def answers(
    golden: GoldenOption = None,
    out_dir: OutDirOption = DEFAULT_OUT_DIR,
    resume: Annotated[
        bool, typer.Option(help="Skip questions already answered in a previous run.")
    ] = True,
    limit: LimitOption = None,
    rpm: Annotated[
        float,
        typer.Option(min=0.1, help="Model requests per minute, over every role of the agent."),
    ] = DEFAULT_ANSWER_RPM,
    delay: Annotated[
        float, typer.Option(min=0.0, help="Seconds between questions, for provider rate limits.")
    ] = 0.0,
) -> None:
    """Ask the agent every golden question and record responses, latency and first-token time.

    The timings leave out the time model calls wait for the --rpm pacing.
    """
    settings = get_settings()
    items = _load_items(golden, settings)
    path = out_dir / ANSWERS_FILE
    with _vector_store_guard():
        records = asyncio.run(
            _run_answers(settings, items, path, resume=resume, limit=limit, rpm=rpm, delay_s=delay)
        )
    failed = sum(record.error is not None for record in records)
    console.print(f"Answered {len(records)} question(s), {failed} failed: {path}")


async def _run_answers(
    settings: Settings,
    items: Sequence[GoldenItem],
    path: Path,
    *,
    resume: bool,
    limit: int | None,
    rpm: float,
    delay_s: float,
) -> list[AnswerRecord]:
    limiter = rate_limiter(rpm)
    async with open_chat_service(settings, limiter=limiter) as service:
        return await run_answers(
            service,
            items,
            path,
            resume=resume,
            limit=limit,
            delay_s=delay_s,
            limiter=limiter,
            on_record=_print_answer,
        )


def _print_answer(record: AnswerRecord) -> None:
    response = record.result
    if response is None:
        status = f"error: {record.error}"
    else:
        status = f"{response.route}, {len(response.citations)} citation(s)"
    timing = f"{record.latency_ms:.0f} ms, plus {record.paced_ms:.0f} ms of pacing"
    console.print(f"{record.item_id}: {status} ({timing})")


@app.command()
def judge(
    golden: GoldenOption = None,
    out_dir: OutDirOption = DEFAULT_OUT_DIR,
    judge_model: Annotated[
        str | None,
        typer.Option(
            envvar=JUDGE_MODEL_ENV,
            help=(
                "Judge as '<provider>:<model>' (default: google_genai:gemini-3.8-flash, or "
                "another configured model when that one writes answers)."
            ),
        ),
    ] = None,
    rpm: Annotated[
        float | None,
        typer.Option(min=0.1, help="Requests per minute (default: depends on the provider)."),
    ] = None,
    limit: LimitOption = None,
) -> None:
    """Judge the recorded answers for correctness and groundedness (cached verdicts)."""
    settings = get_settings()
    items = _load_items(golden, settings)
    answers_path = out_dir / ANSWERS_FILE
    if not answers_path.exists():
        _fail(f"No answers in {answers_path}; run the 'answers' command first.")
    try:
        evaluator = create_judge(
            settings,
            model_spec=judge_model,
            requests_per_minute=rpm,
            cache_path=out_dir / JUDGE_CACHE_FILE,
        )
    except ValueError as exc:
        _fail(str(exc))
    path = out_dir / JUDGMENTS_FILE
    with get_usage_metadata_callback() as usage:
        records = run_judge(
            evaluator,
            items,
            read_records(answers_path, AnswerRecord),
            path,
            limit=limit,
            on_record=_print_judgment,
        )
    tokens = sum(model_usage["total_tokens"] for model_usage in usage.usage_metadata.values())
    same_family = sum(record.same_family for record in records)
    console.print(
        f"Judged {len(records)} answer(s) with {evaluator.model_name} ({tokens} new tokens): {path}"
    )
    if same_family:
        console.print(
            f"{same_family} answer(s) were produced with a model of the judge's family; their "
            "verdicts may favor them."
        )


def _print_judgment(record: JudgmentRecord) -> None:
    if record.error is not None:
        status = f"error: {record.error}"
    elif record.refused:
        status = "refused"
    else:
        verdict = record.correctness.verdict if record.correctness is not None else "n/a"
        status = f"{verdict}, groundedness {record.groundedness_score}"
    console.print(f"{record.item_id}: {status}")


@app.command()
def report(golden: GoldenOption = None, out_dir: OutDirOption = DEFAULT_OUT_DIR) -> None:
    """Summarize the recorded results into summary.md and summary.json."""
    settings = get_settings()
    items = _load_items(golden, settings)
    retrieval_records = _read_optional(out_dir / RETRIEVAL_FILE, RetrievalRecord)
    answer_records = _read_optional(out_dir / ANSWERS_FILE, AnswerRecord)
    judgment_records = _read_optional(out_dir / JUDGMENTS_FILE, JudgmentRecord)
    if not (retrieval_records or answer_records or judgment_records):
        _fail(f"No results in {out_dir}; run 'retrieval', 'answers' or 'judge' first.")
    summary = build_summary(
        items, retrieval=retrieval_records, answers=answer_records, judgments=judgment_records
    )
    markdown_path, json_path = write_report(summary, out_dir)
    if summary.answers is not None and summary.answers.stale_judgments:
        console.print(
            f"{summary.answers.stale_judgments} judgment(s) refer to earlier answers and were "
            "left out; run 'judge' again."
        )
    console.print(f"Wrote {markdown_path} and {json_path}")


def _fail(message: str) -> NoReturn:
    console.print(message, style="red")
    raise typer.Exit(code=1)


def _load_items(golden: Path | None, settings: Settings) -> list[GoldenItem]:
    path = golden or settings.eval_dir / GOLDEN_FILE
    try:
        return load_golden(path)
    except FileNotFoundError:
        _fail(f"Golden set not found: {path}")
    except GoldenSetError as exc:
        _fail(f"Invalid golden set: {exc}")


def _load_corpus(corpus_dir: Path) -> Corpus:
    try:
        return Corpus.load(corpus_dir)
    except FileNotFoundError:
        _fail(f"Corpus not found in {corpus_dir}; build it before validating the golden set.")


def _select_configs(names: Sequence[str] | None) -> list[RetrievalConfig]:
    if not names:
        return list(DEFAULT_CONFIGS)
    known = {config.name: config for config in CONFIGS}
    if unknown := [name for name in names if name not in known]:
        _fail(f"Unknown configuration(s): {', '.join(unknown)}. Choose from {', '.join(known)}.")
    return [known[name] for name in dict.fromkeys(names)]


def _kept_results(path: Path, configs: Sequence[RetrievalConfig], k: int) -> list[RetrievalRecord]:
    """Earlier results of the configurations not run now; they must share this run's ``k``."""
    names = {config.name for config in configs}
    kept = [
        record for record in _read_optional(path, RetrievalRecord) if record.config not in names
    ]
    if other := sorted({record.k for record in kept} - {k}):
        _fail(
            f"{path} holds results for k={', '.join(map(str, other))}; pass the same --k or run "
            "every configuration again (without --config)."
        )
    return kept


def _in_config_order(records: Sequence[RetrievalRecord]) -> list[RetrievalRecord]:
    """Records grouped in the order of :data:`CONFIGS`, so ``dense`` stays the baseline."""
    order = {config.name: position for position, config in enumerate(CONFIGS)}
    return sorted(records, key=lambda record: order.get(record.config, len(order)))


def _fast_models_configured(settings: Settings) -> bool:
    specs = (settings.fast_llm_model, settings.fast_llm_fallback_model)
    return any(provider_configured(spec) for spec in specs if spec)


def _read_optional[RecordT: BaseModel](path: Path, model: type[RecordT]) -> list[RecordT]:
    return read_records(path, model) if path.exists() else []
