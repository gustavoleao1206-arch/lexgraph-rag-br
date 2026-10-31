"""Test doubles for the agent: an in-memory retriever, scripted model roles and builders."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.runtime import Runtime

from lexgraph.agent.context import AgentContext, AgentOptions
from lexgraph.agent.llm import ANSWER_TAG
from lexgraph.agent.nodes import GraphState
from lexgraph.agent.schemas import GroundingCheck, QueryAnalysis, QueryRewrite, RelevanceGrade
from lexgraph.agent.state import new_turn
from lexgraph.config import RetrievalMode
from lexgraph.domain import LawId, Passage
from tests.conftest import make_article, make_chunk


def make_passage(
    law_id: LawId = LawId.CDC,
    number: str = "49",
    text: str = "Art. 49. O consumidor pode desistir do contrato no prazo de 7 dias.",
    *,
    chunk_index: int = 0,
    n_chunks: int = 1,
    score: float = 0.5,
) -> Passage:
    """A search result for one chunk of one article."""
    chunk = make_chunk(make_article(law_id, number, text), index=chunk_index, n_chunks=n_chunks)
    return Passage.model_validate({**chunk.model_dump(), "score": score})


CDC_49 = make_passage(
    LawId.CDC,
    "49",
    "Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias a contar de sua "
    "assinatura ou do ato de recebimento do produto ou serviço, sempre que a contratação "
    "ocorrer fora do estabelecimento comercial.",
    score=0.9,
)
CLT_58 = make_passage(
    LawId.CLT,
    "58",
    "Art. 58 - A duração normal do trabalho não excederá de 8 (oito) horas diárias.",
    score=0.7,
)
LGPD_18 = make_passage(
    LawId.LGPD,
    "18",
    "Art. 18. O titular dos dados pessoais tem direito a obter do controlador a confirmação "
    "da existência de tratamento e o acesso aos dados.",
    score=0.6,
)
CF_7 = [
    make_passage(
        LawId.CF,
        "7",
        f"Art. 7º São direitos dos trabalhadores urbanos e rurais: {units}",
        chunk_index=index,
        n_chunks=3,
        score=0.5 - index / 10,
    )
    for index, units in enumerate(
        (
            "I - relação de emprego protegida contra despedida arbitrária;",
            "XIII - duração do trabalho normal não superior a oito horas diárias;",
            "XVII - gozo de férias anuais remuneradas com, pelo menos, um terço a mais.",
        )
    )
]


class FakeRetriever:
    """Retriever over a fixed set of passages that records every search.

    ``by_query`` maps a query to the results it returns; other queries return every passage.
    """

    def __init__(
        self,
        passages: Sequence[Passage] = (),
        *,
        by_query: Mapping[str, Sequence[Passage]] | None = None,
        error: Exception | None = None,
        reranker: bool = False,
    ) -> None:
        self.passages = list(passages)
        self.by_query = dict(by_query or {})
        self.error = error
        self.reranker = reranker
        self.searches: list[dict[str, Any]] = []
        self.lookups: list[str] = []

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[Passage]:
        if self.error is not None:
            raise self.error
        self.searches.append({"query": query, "k": k, "laws": laws, "mode": mode, "rerank": rerank})
        found = list(self.by_query.get(query, self.passages))
        if laws:
            found = [passage for passage in found if passage.law_id in laws]
        return found[: k or 6]

    def get_article(self, article_id: str) -> list[Passage]:
        self.lookups.append(article_id)
        chunks = [passage for passage in self.passages if passage.article_id == article_id]
        return sorted(chunks, key=lambda passage: passage.chunk_index)

    def count(self) -> int:
        return len(self.passages)

    def rerank_available(self) -> bool:
        return self.reranker


class Script[T]:
    """A fake model role: returns (or raises) its outputs in turn, repeating the last one.

    Every input is recorded in ``inputs``.
    """

    def __init__(self, *outputs: T | Exception) -> None:
        self._outputs = list(outputs)
        self.inputs: list[Any] = []

    def __call__(self, messages: Any) -> T:
        self.inputs.append(messages)
        output = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        if isinstance(output, Exception):
            raise output
        return output

    @property
    def runnable(self) -> Runnable[Any, T]:
        return RunnableLambda(self.__call__)


def analysis(
    intent: str = "legal_question",
    query: str = "consulta",
    *,
    standalone: str = "",
    laws: Sequence[str] = (),
    articles: Sequence[Mapping[str, str | None]] = (),
    sub_queries: Sequence[str] = (),
) -> QueryAnalysis:
    return QueryAnalysis.model_validate(
        {
            "intent": intent,
            "standalone_question": standalone,
            "search_query": query,
            "sub_queries": list(sub_queries),
            "laws": list(laws),
            "articles": list(articles),
        }
    )


def relevant(*numbers: int) -> RelevanceGrade:
    return RelevanceGrade(reasoning="teste", relevant=list(numbers))


def rewritten(query: str, *laws: str) -> QueryRewrite:
    return QueryRewrite.model_validate({"search_query": query, "laws": list(laws)})


def grounded() -> GroundingCheck:
    return GroundingCheck(unsupported_claims=[], grounded=True)


def unsupported(*claims: str) -> GroundingCheck:
    return GroundingCheck(unsupported_claims=list(claims), grounded=False)


def answer_model(*replies: str) -> Runnable[LanguageModelInput, BaseMessage]:
    """A streaming answer model that gives ``replies`` in turn, tagged like the real one."""
    model = GenericFakeChatModel(messages=iter([AIMessage(reply) for reply in replies]))
    return model.with_config(tags=[ANSWER_TAG])


def make_context(
    retriever: FakeRetriever,
    *,
    analyzer: Runnable[Any, QueryAnalysis] | None = None,
    grader: Runnable[Any, RelevanceGrade] | None = None,
    rewriter: Runnable[Any, QueryRewrite] | None = None,
    verifier: Runnable[Any, GroundingCheck] | None = None,
    answerer: Runnable[LanguageModelInput, BaseMessage] | None = None,
    **options: Any,
) -> AgentContext:
    """Agent context with harmless defaults: every passage relevant, every answer grounded."""
    return AgentContext(
        retriever=retriever,
        analyzer=analyzer or Script(analysis()).runnable,
        grader=grader or Script(relevant(1, 2, 3, 4, 5, 6)).runnable,
        rewriter=rewriter or Script(rewritten("nova consulta")).runnable,
        verifier=verifier or Script(grounded()).runnable,
        answerer=answerer or answer_model("Resposta [1]."),
        options=AgentOptions(**options),
    )


def make_state(question: str = "Posso desistir de uma compra online?", **values: Any) -> GraphState:
    """A fresh turn, as the service starts it, with some keys overridden."""
    return cast("GraphState", {**new_turn(question), "user_laws": [], **values})


def make_runtime(context: AgentContext) -> tuple[Runtime[AgentContext], list[Any]]:
    """A runtime for calling nodes directly, plus the list that receives their events."""
    events: list[Any] = []
    return Runtime(context=context, stream_writer=events.append), events


def dumped(*passages: Passage) -> list[dict[str, Any]]:
    """Passages as stored in the agent state."""
    return [passage.model_dump(mode="json") for passage in passages]
