"""Builders and fakes shared by the evaluation harness tests (no models, no network)."""

import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableLambda

from lexgraph.config import RetrievalMode
from lexgraph.corpus import Corpus, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import Article, LawId, Passage, split_article_id
from lexgraph.eval.golden import GoldenItem
from lexgraph.eval.judge import (
    ClaimCheck,
    CorrectnessLabel,
    CorrectnessVerdict,
    GroundednessVerdict,
    Judge,
    JudgeCache,
)
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    Citation,
    FinalEvent,
    Grounding,
    HistoryMessage,
    Route,
    ServiceInfo,
    Source,
    StatusEvent,
    TokenEvent,
    Usage,
)
from tests.conftest import make_article, make_chunk

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.jsonl"
CORPUS_IDS = ("CDC:49", "CLT:59", "CF:7", "LGPD:18")


def make_item(**overrides: Any) -> GoldenItem:
    """A valid in-scope golden item, with ``overrides`` applied before validation."""
    data: dict[str, Any] = {
        "id": "cdc-001",
        "question": "Até quando posso desistir de uma compra feita pela internet?",
        "type": "lookup",
        "expected_laws": ["CDC"],
        "expected_articles": ["CDC:49"],
        "acceptable_articles": [],
        "reference_answer": "O consumidor pode desistir em 7 dias (CDC, art. 49).",
        "must_include": ["7 dias|sete dias"],
        "must_not_include": [],
        "should_refuse": False,
        "difficulty": "easy",
        "tags": [],
        "confidence": "high",
    }
    return GoldenItem.model_validate(data | overrides)


def article_for(article_id: str, *, status: str = "vigente", seq: int = 0) -> Article:
    """An article of the given id with placeholder text."""
    law_id, number = split_article_id(article_id)
    return make_article(law_id, number, f"Art. {number}. Texto de teste.", seq=seq, status=status)


def make_corpus(articles: Sequence[Article]) -> Corpus:
    """An in-memory corpus holding ``articles``."""
    return Corpus(articles, _manifest(articles))


def write_test_corpus(corpus_dir: Path, articles: Sequence[Article]) -> None:
    """Write ``articles`` as a corpus folder readable by :meth:`Corpus.load`."""
    by_law: dict[LawId, list[Article]] = {}
    for article in articles:
        by_law.setdefault(article.law_id, []).append(article)
    write_corpus(corpus_dir, by_law, _manifest(articles))


def _manifest(articles: Sequence[Article]) -> CorpusManifest:
    laws = {
        law_id: LawManifest(
            law_id=law_id,
            source=law_id.value.lower(),
            source_url="https://www.planalto.gov.br/example.htm",
            source_sha256="0" * 64,
            fetched_at="2026-10-30T00:00:00+00:00",
            articles=sum(a.law_id == law_id for a in articles),
            in_force=sum(a.law_id == law_id and a.status == "vigente" for a in articles),
            revoked=sum(a.law_id == law_id and a.status == "revogado" for a in articles),
            vetoed=sum(a.law_id == law_id and a.status == "vetado" for a in articles),
        )
        for law_id in {article.law_id for article in articles}
    }
    return CorpusManifest(
        generated_at="2026-10-30T00:00:00+00:00", parser_version="test", laws=laws
    )


def make_passage(chunk_id: str, score: float = 1.0, *, units: Sequence[str] = ("*",)) -> Passage:
    """A passage for ``"CDC:49#1"``-style chunk ids, covering ``units`` of its article."""
    article_id, _, index = chunk_id.partition("#")
    chunk_index = int(index or 0)
    chunk = make_chunk(article_for(article_id), index=chunk_index, n_chunks=chunk_index + 1)
    return Passage.model_validate(chunk.model_dump() | {"score": score, "units": list(units)})


def make_source(index: int, article_id: str) -> Source:
    """A source numbered ``index`` for ``article_id``."""
    law_id, number = split_article_id(article_id)
    return Source(
        index=index,
        chunk_id=f"{article_id}#0",
        article_id=article_id,
        law_id=law_id,
        citation=f"{law_id.value}, art. {number}",
        url=f"https://www.planalto.gov.br/example.htm#art{number}",
        text=f"Art. {number}. Texto de teste.",
        score=1.0,
    )


def make_response(
    answer: str,
    *,
    route: Route = "answered",
    context: Sequence[str] = (),
    cited: Sequence[str] = (),
    rewrites: int = 0,
    regenerations: int = 0,
    grounded: bool | None = None,
    usage: Usage | None = None,
) -> ChatResponse:
    """A response whose sources are ``context`` and whose ``[n]`` citations are ``cited``."""
    sources = [make_source(index, article_id) for index, article_id in enumerate(context, 1)]
    by_article = {source.article_id: source for source in sources}
    citations = [
        Citation(
            index=by_article[article_id].index,
            article_id=article_id,
            citation=by_article[article_id].citation,
            url=by_article[article_id].url,
        )
        for article_id in cited
    ]
    return ChatResponse(
        thread_id="eval-test",
        answer=answer,
        route=route,
        sources=sources,
        citations=citations,
        grounding=Grounding(grounded=grounded) if grounded is not None else None,
        rewrites=rewrites,
        regenerations=regenerations,
        usage=usage or Usage(),
    )


def answer_events(response: ChatResponse, *tokens: str) -> list[ChatEvent]:
    """A typical event stream: progress, streamed tokens and the final response."""
    return [
        StatusEvent(stage="analyze", message="Analisando a pergunta"),
        StatusEvent(stage="retrieve", message="Buscando dispositivos"),
        StatusEvent(stage="retrieve", message="Dispositivos encontrados"),
        *(TokenEvent(text=token) for token in tokens),
        FinalEvent(response=response),
    ]


class FakeRetriever:
    """Returns scripted chunk ids per ``(mode, rerank)`` configuration and query.

    ``units`` gives the units covered by a chunk id (default: the whole article).
    """

    def __init__(
        self,
        rankings: Mapping[tuple[str, bool], Mapping[str, Sequence[str]]],
        units: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.rankings = rankings
        self.units = units or {}
        self.calls: list[tuple[str, int | None, str | None, bool | None]] = []

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[Passage]:
        self.calls.append((query, k, mode, rerank))
        chunk_ids = self.rankings.get((mode or "hybrid", bool(rerank)), {}).get(query, [])
        return [
            make_passage(chunk_id, units=self.units.get(chunk_id, ("*",)))
            for chunk_id in chunk_ids[:k]
        ]

    def get_article(self, article_id: str) -> list[Passage]:
        return []

    def count(self) -> int:
        return 0

    def rerank_available(self) -> bool:
        return False


class FakeChatService:
    """Streams scripted events per question; a script may also be an exception to raise.

    With ``limiter`` every answer first waits for a token, like a paced model call.
    """

    def __init__(
        self,
        scripts: Mapping[str, Sequence[ChatEvent] | Exception],
        *,
        limiter: BaseRateLimiter | None = None,
    ) -> None:
        self.scripts = scripts
        self.limiter = limiter
        self.requests: list[ChatRequest] = []
        self.started: list[float] = []

    async def ask(self, request: ChatRequest) -> ChatResponse:
        async for event in self.stream(request):
            if isinstance(event, FinalEvent):
                return event.response
        msg = "the script has no final event"
        raise AssertionError(msg)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        self.started.append(time.perf_counter())
        script = self.scripts[request.question]
        if isinstance(script, Exception):
            raise script
        if self.limiter is not None:
            await self.limiter.aacquire()
        for event in script:
            yield event

    async def history(self, thread_id: str) -> list[HistoryMessage]:
        return []

    async def info(self) -> ServiceInfo:
        return ServiceInfo(
            version="test",
            collection="lexgraph",
            indexed_chunks=0,
            embedding_provider="fastembed",
            embedding_model="test",
            retrieval_mode="hybrid",
            rerank=False,
            rerank_available=False,
            llm_model="test",
            llm_fallback_model=None,
            fast_llm_model="test",
            models_configured=True,
            max_question_chars=2000,
        )

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[Source]:
        return []


def groundedness_verdict(*labels: str) -> GroundednessVerdict:
    """A verdict with one claim per label (``"supported"``, ``"unsupported"``...)."""
    return GroundednessVerdict(
        claims=[
            ClaimCheck.model_validate(
                {
                    "claim": f"Afirmação {n}",
                    "verdict": label,
                    "evidence": 1 if label == "supported" else None,
                }
            )
            for n, label in enumerate(labels, 1)
        ]
    )


def correctness_verdict(label: CorrectnessLabel = "correct") -> CorrectnessVerdict:
    """A correctness verdict with the given label."""
    return CorrectnessVerdict(
        reasoning="Comparação com a referência.",
        missing_points=[],
        contradictions=[],
        verdict=label,
    )


class StubJudge:
    """A :class:`Judge` whose prompts are replaced by functions; records every call."""

    def __init__(
        self,
        groundedness: Callable[[dict[str, Any]], Any] | None = None,
        correctness: Callable[[dict[str, Any]], Any] | None = None,
        *,
        model_name: str = "google_genai:judge-test",
        cache: JudgeCache | None = None,
    ) -> None:
        self.groundedness_calls: list[dict[str, Any]] = []
        self.correctness_calls: list[dict[str, Any]] = []
        on_groundedness = groundedness or (lambda _: groundedness_verdict("supported"))
        on_correctness = correctness or (lambda _: correctness_verdict())

        def run_groundedness(inputs: dict[str, Any]) -> Any:
            self.groundedness_calls.append(inputs)
            return on_groundedness(inputs)

        def run_correctness(inputs: dict[str, Any]) -> Any:
            self.correctness_calls.append(inputs)
            return on_correctness(inputs)

        self.judge = Judge(
            groundedness=RunnableLambda(run_groundedness),
            correctness=RunnableLambda(run_correctness),
            model_name=model_name,
            cache=cache,
        )
