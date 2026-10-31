"""LangChain retriever over the Qdrant index: dense, sparse (BM25) or hybrid search."""

import logging
import time
from collections.abc import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PositiveFloat
from qdrant_client import QdrantClient, models

from lexgraph.config import RetrievalMode
from lexgraph.domain import LawId, Passage
from lexgraph.retrieval.reranker import Reranker
from lexgraph.retrieval.sparse import SparseEncoder
from lexgraph.retrieval.store import (
    ARTICLE_ID_KEY,
    DENSE_VECTOR,
    LAW_ID_KEY,
    SPARSE_VECTOR,
    IndexMismatchError,
    collection_exists,
    encoder_fingerprint,
    passage_from_payload,
    read_index_metadata,
)

logger = logging.getLogger(__name__)

# Score of the passages found by an exact article lookup rather than by a search.
LOOKUP_SCORE = 1.0

_SCROLL_PAGE = 64

_QueryInput = list[float] | models.SparseVector | models.FusionQuery | models.RrfQuery


class LegalRetriever(BaseRetriever):
    """Search over the legal passages indexed in Qdrant (implements ``protocols.Retriever``).

    Modes:

    - ``dense``: cosine similarity between the query and passage embeddings;
    - ``sparse``: BM25 on the ``bm25`` sparse vectors;
    - ``hybrid``: each branch fetches ``prefetch_k`` candidates and Reciprocal Rank Fusion
      merges them (Qdrant's default constant, or ``rrf_k`` when set), weighting both branches
      equally unless ``rrf_weights`` gives the dense and sparse weights.

    With reranking, ``max(k, rerank_candidates)`` passages are fetched and the cross-encoder
    keeps the best ``k``; with ``k >= rerank_candidates`` it only reorders the passages the
    search found. A law filter restricts every branch, since Qdrant applies the top-level
    filter to the prefetches too.

    One instance can serve FastAPI's worker threads: it keeps no per-request state, the Qdrant
    client is only read, and the embedders and the reranker load their models under a lock.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: QdrantClient
    collection: str = Field(description="Collection or alias to search.")
    dense: Embeddings
    sparse: SparseEncoder
    reranker: Reranker | None = None
    mode: RetrievalMode = "hybrid"
    k: int = Field(default=6, ge=1, description="Passages returned by default.")
    prefetch_k: int = Field(default=30, ge=1, description="Candidates per hybrid branch.")
    rrf_k: int | None = Field(default=None, ge=1, description="RRF constant; None = default.")
    rrf_weights: tuple[PositiveFloat, PositiveFloat] | None = Field(
        default=None, description="RRF weights of the dense and sparse branches; None = equal."
    )
    rerank: bool = False
    rerank_candidates: int = Field(default=10, ge=1, description="Passages to rerank.")

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[Passage]:
        """Return up to ``k`` passages for ``query``, best first.

        ``laws`` restricts the search to those laws (``None`` or empty: every law); ``k``,
        ``mode`` and ``rerank`` override the configured defaults. A blank query returns no
        passages, and a query made only of stopwords has no BM25 terms: it finds nothing in
        sparse mode and only uses the dense branch in hybrid mode.
        """
        top_k = self.k if k is None else k
        if top_k < 1:
            msg = f"k must be at least 1, got {top_k}"
            raise ValueError(msg)
        reranker = self._active_reranker(rerank)
        if not query.strip():
            return []
        started = time.perf_counter()
        search_mode = mode or self.mode
        limit = top_k if reranker is None else max(top_k, self.rerank_candidates)
        points = self._query(query, search_mode, limit=limit, query_filter=_law_filter(laws))
        passages = [passage_from_payload(point.payload, score=point.score) for point in points]
        if reranker is not None:
            passages = reranker.rerank(query, passages, top_n=top_k)
        logger.debug(
            "search mode=%s k=%d laws=%s rerank=%s: %d passages in %.0f ms",
            search_mode,
            top_k,
            laws,
            reranker is not None,
            len(passages),
            (time.perf_counter() - started) * 1000,
        )
        return passages

    def get_article(self, article_id: str) -> list[Passage]:
        """Return every indexed chunk of one article (e.g. ``"CLT:482"``) in order.

        Returns an empty list when the article is not indexed (unknown, revoked or vetoed).
        Passages get :data:`LOOKUP_SCORE`.
        """
        article_filter = models.Filter(
            must=[
                models.FieldCondition(key=ARTICLE_ID_KEY, match=models.MatchValue(value=article_id))
            ]
        )
        records = self._scroll(article_filter)
        passages = [passage_from_payload(record.payload, score=LOOKUP_SCORE) for record in records]
        return sorted(passages, key=lambda passage: passage.chunk_index)

    def count(self) -> int:
        """Number of indexed chunks (0 before the first ingestion)."""
        if not collection_exists(self.client, self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def rerank_available(self) -> bool:
        """Whether reranking can run: a reranker is configured and its model is available."""
        return self.reranker is not None and self.reranker.is_available()

    def check_index(self) -> None:
        """Raise :class:`IndexMismatchError` if the index was built with other encoders.

        Vectors of another model live in another space, so searching them would return noise
        without any error. Nothing is checked before the first ingestion, and an index that
        records no metadata (built by an earlier version) is only logged.
        """
        if not collection_exists(self.client, self.collection):
            return
        metadata = read_index_metadata(self.client, self.collection)
        if metadata is None:
            logger.warning(
                "the index %s does not record its encoders; re-run `lexgraph ingest` to add them",
                self.collection,
            )
            return
        indexed, configured = metadata.encoders, encoder_fingerprint(self.dense, self.sparse)
        changed = sorted(
            key
            for key in indexed.keys() | configured.keys()
            if indexed.get(key) != configured.get(key)
        )
        if changed:
            detail = "; ".join(
                f"{key}: {indexed.get(key)!r} indexed, {configured.get(key)!r} configured"
                for key in changed
            )
            msg = (
                f"the index {self.collection} was built with other encoders ({detail}); "
                "re-run `lexgraph ingest`"
            )
            raise IndexMismatchError(msg)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
        laws: Sequence[LawId] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[Document]:
        passages = self.search(query, k=k, laws=laws, mode=mode, rerank=rerank)
        return [passage_to_document(passage) for passage in passages]

    def _active_reranker(self, rerank: bool | None) -> Reranker | None:
        if not (self.rerank if rerank is None else rerank):
            return None
        if self.reranker is None:
            msg = "reranking was requested but no reranker is configured"
            raise ValueError(msg)
        return self.reranker

    def _query(
        self, query: str, mode: RetrievalMode, *, limit: int, query_filter: models.Filter | None
    ) -> list[models.ScoredPoint]:
        if mode == "dense":
            dense_vector = self.dense.embed_query(query)
            return self._points(
                dense_vector, using=DENSE_VECTOR, limit=limit, query_filter=query_filter
            )
        sparse_vector = self.sparse.encode_query(query)
        if mode == "sparse":
            if not sparse_vector.indices:
                return []
            return self._points(
                sparse_vector, using=SPARSE_VECTOR, limit=limit, query_filter=query_filter
            )
        branch_limit = max(self.prefetch_k, limit)
        prefetch = [
            models.Prefetch(
                query=self.dense.embed_query(query), using=DENSE_VECTOR, limit=branch_limit
            )
        ]
        if sparse_vector.indices:
            prefetch.append(
                models.Prefetch(query=sparse_vector, using=SPARSE_VECTOR, limit=branch_limit)
            )
        return self._points(
            self._fusion(len(prefetch)), prefetch=prefetch, limit=limit, query_filter=query_filter
        )

    def _points(
        self,
        query: _QueryInput,
        *,
        limit: int,
        query_filter: models.Filter | None,
        using: str | None = None,
        prefetch: list[models.Prefetch] | None = None,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            self.collection,
            query=query,
            using=using,
            prefetch=prefetch,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _fusion(self, branches: int) -> models.FusionQuery | models.RrfQuery:
        """RRF over the prefetches, dense first; a query without BM25 terms has one branch."""
        if self.rrf_k is None and self.rrf_weights is None:
            return models.FusionQuery(fusion=models.Fusion.RRF)
        weights = None if self.rrf_weights is None else list(self.rrf_weights[:branches])
        return models.RrfQuery(rrf=models.Rrf(k=self.rrf_k, weights=weights))

    def _scroll(self, scroll_filter: models.Filter) -> list[models.Record]:
        records, offset = self.client.scroll(
            self.collection, scroll_filter=scroll_filter, limit=_SCROLL_PAGE, with_payload=True
        )
        while offset is not None:
            page, offset = self.client.scroll(
                self.collection,
                scroll_filter=scroll_filter,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
            )
            records.extend(page)
        return records


def passage_to_document(passage: Passage) -> Document:
    """LangChain document for a passage: its text as content, the other fields as metadata."""
    return Document(
        page_content=passage.text,
        id=passage.chunk_id,
        metadata=passage.model_dump(mode="json", exclude={"text"}),
    )


def _law_filter(laws: Sequence[LawId] | None) -> models.Filter | None:
    if not laws:
        return None
    law_ids = sorted({LawId(law).value for law in laws})
    return models.Filter(
        must=[models.FieldCondition(key=LAW_ID_KEY, match=models.MatchAny(any=law_ids))]
    )
