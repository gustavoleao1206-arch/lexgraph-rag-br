"""LegalRetriever: modes, filters, fusion, reranking, lookup, index checks and LangChain use."""

import logging
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient, models

from lexgraph.config import RetrievalMode
from lexgraph.domain import LawId
from lexgraph.retrieval import retriever as retriever_module
from lexgraph.retrieval.indexer import IndexReport
from lexgraph.retrieval.retriever import LOOKUP_SCORE, LegalRetriever, passage_to_document
from lexgraph.retrieval.store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    IndexMismatchError,
    create_collection,
    point_alias,
)
from tests.unit.retrieval.conftest import (
    ALIAS,
    DENSE_SIZE,
    FixedScoreReranker,
    HashingSparseEncoder,
    StopwordsOnlySparseEncoder,
    chunk_by_id,
)

if TYPE_CHECKING:
    from lexgraph.protocols import Retriever

CDC_49_WORDS = "desistir contrato prazo assinatura estabelecimento comercial"


@pytest.fixture
def query_calls(client: QdrantClient, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keyword arguments of every ``query_points`` call."""
    calls: list[dict[str, Any]] = []
    query_points = client.query_points

    def spy(*args: Any, **kwargs: Any) -> models.QueryResponse:
        calls.append(kwargs)
        return query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", spy)
    return calls


def with_options(retriever: LegalRetriever, **options: Any) -> LegalRetriever:
    return retriever.model_copy(update=options)


def test_retriever_satisfies_the_protocol(retriever: LegalRetriever) -> None:
    satisfied: Retriever = retriever  # checked by mypy

    assert satisfied.count() == len(satisfied.search("Art.", k=10)) == 10


def test_dense_search_finds_the_identical_passage(retriever: LegalRetriever) -> None:
    chunk = chunk_by_id("CLT:129#0")

    passages = retriever.search(chunk.embedding_text(), mode="dense", k=1)

    assert [passage.chunk_id for passage in passages] == ["CLT:129#0"]
    assert passages[0].score == pytest.approx(1.0)


def test_sparse_search_ranks_by_shared_words(retriever: LegalRetriever) -> None:
    passages = retriever.search(CDC_49_WORDS, mode="sparse")

    assert passages[0].article_id == "CDC:49"


def test_hybrid_search_fuses_both_branches(retriever: LegalRetriever) -> None:
    chunk = chunk_by_id("CDC:49#0")

    passages = retriever.search(chunk.embedding_text(), mode="hybrid", k=5)

    assert passages[0].chunk_id == "CDC:49#0"
    assert len(passages) == 5
    assert len({passage.chunk_id for passage in passages}) == 5
    scores = [passage.score for passage in passages]
    assert scores == sorted(scores, reverse=True)


def test_passages_carry_every_chunk_field(retriever: LegalRetriever) -> None:
    chunk = chunk_by_id("CDC:49#0")

    passage = retriever.search(CDC_49_WORDS, mode="sparse", k=1)[0]

    assert passage.model_dump(exclude={"score", "rerank_score"}) == chunk.model_dump()
    assert passage.rerank_score is None


def test_k_defaults_to_the_configured_value(retriever: LegalRetriever) -> None:
    assert len(retriever.search("Art.")) == retriever.k == 3
    assert len(retriever.search("Art.", k=7)) == 7


def test_k_must_be_positive(retriever: LegalRetriever) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("Art.", k=0)


def test_blank_query_returns_nothing(retriever: LegalRetriever) -> None:
    assert retriever.search("   ") == []


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_law_filter_applies_to_every_mode(retriever: LegalRetriever, mode: RetrievalMode) -> None:
    passages = retriever.search("Art. empregado", k=10, laws=[LawId.CLT], mode=mode)

    assert passages
    assert {passage.law_id for passage in passages} == {LawId.CLT}


def test_law_filter_accepts_several_laws(retriever: LegalRetriever) -> None:
    passages = retriever.search("Art.", k=10, laws=[LawId.CDC, LawId.MCI], mode="sparse")

    assert {passage.law_id for passage in passages} == {LawId.CDC, LawId.MCI}


def test_empty_law_filter_searches_every_law(retriever: LegalRetriever) -> None:
    passages = retriever.search("Art.", k=10, laws=[], mode="sparse")

    assert len({passage.law_id for passage in passages}) > 2


def test_mode_defaults_to_the_configured_one(
    retriever: LegalRetriever, query_calls: list[dict[str, Any]]
) -> None:
    with_options(retriever, mode="sparse").search(CDC_49_WORDS)
    with_options(retriever, mode="sparse").search(CDC_49_WORDS, mode="dense")

    assert [call["using"] for call in query_calls] == [SPARSE_VECTOR, DENSE_VECTOR]
    assert [call["prefetch"] for call in query_calls] == [None, None]


def test_hybrid_uses_default_rrf_and_wide_prefetches(
    retriever: LegalRetriever, query_calls: list[dict[str, Any]]
) -> None:
    with_options(retriever, prefetch_k=8).search(CDC_49_WORDS, k=3)
    with_options(retriever, prefetch_k=2).search(CDC_49_WORDS, k=5)

    first, second = query_calls
    assert first["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert [(p.using, p.limit) for p in first["prefetch"]] == [
        (DENSE_VECTOR, 8),
        (SPARSE_VECTOR, 8),
    ]
    assert [p.limit for p in second["prefetch"]] == [5, 5]


def test_hybrid_uses_the_configured_rrf_constant(
    retriever: LegalRetriever, query_calls: list[dict[str, Any]]
) -> None:
    passages = with_options(retriever, rrf_k=60).search(CDC_49_WORDS)

    assert query_calls[0]["query"] == models.RrfQuery(rrf=models.Rrf(k=60))
    assert 0 < passages[0].score <= 2 / 60


def test_hybrid_weights_the_branches(
    retriever: LegalRetriever, query_calls: list[dict[str, Any]]
) -> None:
    with_options(retriever, rrf_weights=(2.0, 1.0)).search(CDC_49_WORDS)
    with_options(retriever, rrf_k=60, rrf_weights=(1.0, 0.5)).search(CDC_49_WORDS)

    first, second = query_calls
    assert first["query"] == models.RrfQuery(rrf=models.Rrf(weights=[2.0, 1.0]))
    assert second["query"] == models.RrfQuery(rrf=models.Rrf(k=60, weights=[1.0, 0.5]))


def test_hybrid_weights_without_bm25_terms_keep_the_dense_weight(
    client: QdrantClient,
    dense: DeterministicFakeEmbedding,
    indexed: IndexReport,
    query_calls: list[dict[str, Any]],
) -> None:
    retriever = LegalRetriever(
        client=client,
        collection=ALIAS,
        dense=dense,
        sparse=StopwordsOnlySparseEncoder(),
        rrf_weights=(2.0, 1.0),
        k=3,
    )

    assert len(retriever.search("de que os", mode="hybrid")) == 3
    assert query_calls[0]["query"] == models.RrfQuery(rrf=models.Rrf(weights=[2.0]))


def test_rrf_weights_must_be_positive(
    client: QdrantClient, dense: DeterministicFakeEmbedding
) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        LegalRetriever(
            client=client,
            collection=ALIAS,
            dense=dense,
            sparse=HashingSparseEncoder(),
            rrf_weights=(1.0, 0.0),
        )


def test_query_without_bm25_terms(
    client: QdrantClient, dense: DeterministicFakeEmbedding, indexed: IndexReport
) -> None:
    retriever = LegalRetriever(
        client=client, collection=ALIAS, dense=dense, sparse=StopwordsOnlySparseEncoder(), k=3
    )

    assert retriever.search("de que os", mode="sparse") == []
    assert len(retriever.search("de que os", mode="hybrid")) == 3


def test_rerank_fetches_candidates_and_keeps_the_best_k(retriever: LegalRetriever) -> None:
    reranker = FixedScoreReranker({"CLT:58": 3.0, "LGPD:18": 2.0})
    reranking = with_options(retriever, reranker=reranker, rerank_candidates=10)

    passages = reranking.search("Art.", k=2, mode="sparse", rerank=True)

    (query, candidates, top_n) = reranker.calls[0]
    assert (query, len(candidates), top_n) == ("Art.", 10, 2)
    assert [passage.article_id for passage in passages] == ["CLT:58", "LGPD:18"]
    assert [passage.rerank_score for passage in passages] == [3.0, 2.0]


def test_rerank_fetches_at_least_k_candidates(retriever: LegalRetriever) -> None:
    reranker = FixedScoreReranker({})
    reranking = with_options(retriever, reranker=reranker, rerank_candidates=2)

    passages = reranking.search("Art.", k=5, mode="sparse", rerank=True)

    assert len(reranker.calls[0][1]) == 5
    assert len(passages) == 5


def test_rerank_follows_the_configured_default(retriever: LegalRetriever) -> None:
    reranker = FixedScoreReranker({})
    reranking = with_options(retriever, reranker=reranker, rerank=True)

    reranking.search("Art.")
    reranking.search("Art.", rerank=False)

    assert len(reranker.calls) == 1


def test_rerank_without_a_reranker_is_an_error(retriever: LegalRetriever) -> None:
    with pytest.raises(ValueError, match="no reranker is configured"):
        retriever.search("Art.", rerank=True)


@pytest.mark.parametrize(
    ("reranker", "available"),
    [
        (None, False),
        (FixedScoreReranker({}), True),
        (FixedScoreReranker({}, available=False), False),
    ],
)
def test_rerank_available_needs_a_reranker_with_its_model(
    retriever: LegalRetriever, reranker: FixedScoreReranker | None, available: bool
) -> None:
    assert with_options(retriever, reranker=reranker).rerank_available() is available


def test_check_index_accepts_the_encoders_that_built_it(retriever: LegalRetriever) -> None:
    retriever.check_index()


def test_check_index_rejects_other_encoders(
    retriever: LegalRetriever, dense: DeterministicFakeEmbedding
) -> None:
    class OtherSparseEncoder(HashingSparseEncoder):
        def fingerprint(self) -> dict[str, str]:
            return {"model": "other-bm25"}

    other = with_options(retriever, sparse=OtherSparseEncoder())

    with pytest.raises(IndexMismatchError) as error:
        other.check_index()

    message = str(error.value)
    assert message.startswith(f"the index {ALIAS} was built with other encoders (")
    assert "sparse.model: None indexed, 'other-bm25' configured" in message
    assert "sparse.class: 'tests.unit.retrieval.conftest.HashingSparseEncoder' indexed" in message
    assert message.endswith("re-run `lexgraph ingest`")


def test_check_index_before_the_first_ingestion_passes(retriever: LegalRetriever) -> None:
    with_options(retriever, collection="missing").check_index()


def test_check_index_only_logs_an_index_without_metadata(
    client: QdrantClient,
    dense: DeterministicFakeEmbedding,
    caplog: pytest.LogCaptureFixture,
) -> None:
    create_collection(client, "legacy-20261030120005", dim=DENSE_SIZE)
    point_alias(client, "legacy", "legacy-20261030120005")
    retriever = LegalRetriever(
        client=client, collection="legacy", dense=dense, sparse=HashingSparseEncoder()
    )

    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        retriever.check_index()

    assert "the index legacy does not record its encoders" in caplog.text


def test_get_article_returns_its_chunks_in_order(retriever: LegalRetriever) -> None:
    passages = retriever.get_article("CF:5")

    assert [passage.chunk_id for passage in passages] == ["CF:5#0", "CF:5#1", "CF:5#2"]
    assert {passage.score for passage in passages} == {LOOKUP_SCORE}


def test_get_article_reads_every_page(
    retriever: LegalRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retriever_module, "_SCROLL_PAGE", 1)

    assert [passage.chunk_index for passage in retriever.get_article("CF:5")] == [0, 1, 2]


def test_get_article_of_an_unknown_article_is_empty(retriever: LegalRetriever) -> None:
    assert retriever.get_article("CF:250") == []


def test_count_is_the_number_of_indexed_chunks(retriever: LegalRetriever) -> None:
    assert retriever.count() == 10


def test_count_before_the_first_ingestion_is_zero(retriever: LegalRetriever) -> None:
    assert with_options(retriever, collection="missing").count() == 0


def test_invoke_returns_langchain_documents(retriever: LegalRetriever) -> None:
    documents = retriever.invoke(chunk_by_id("CDC:49#0").embedding_text())

    assert len(documents) == 3
    first = documents[0]
    assert first.id == "CDC:49#0"
    assert first.page_content == chunk_by_id("CDC:49#0").text
    assert first.metadata["citation"] == "CDC, art. 49"
    assert first.metadata["score"] > 0


def test_invoke_forwards_search_options(retriever: LegalRetriever) -> None:
    documents = retriever.invoke("Art.", k=4, laws=[LawId.CLT], mode="sparse")

    assert len(documents) == 3
    assert {document.metadata["law_id"] for document in documents} == {"CLT"}


def test_passage_to_document_keeps_the_scores(retriever: LegalRetriever) -> None:
    passage = retriever.search(CDC_49_WORDS, mode="sparse", k=1)[0]
    reranked = passage.model_copy(update={"rerank_score": 1.5})

    document = passage_to_document(reranked)

    assert document.metadata["score"] == passage.score
    assert document.metadata["rerank_score"] == 1.5
    assert "text" not in document.metadata


def test_retriever_rejects_invalid_settings(
    client: QdrantClient, dense: DeterministicFakeEmbedding
) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        LegalRetriever(
            client=client, collection=ALIAS, dense=dense, sparse=HashingSparseEncoder(), k=0
        )
