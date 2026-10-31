"""Composition root of the retrieval layer."""

from qdrant_client import QdrantClient

from lexgraph.config import Settings
from lexgraph.retrieval.embeddings import build_dense_embedder
from lexgraph.retrieval.reranker import CrossEncoderReranker
from lexgraph.retrieval.retriever import LegalRetriever
from lexgraph.retrieval.sparse import build_sparse_encoder
from lexgraph.retrieval.store import create_qdrant_client


def build_retriever(settings: Settings, *, client: QdrantClient | None = None) -> LegalRetriever:
    """Assemble the retriever configured by ``settings``.

    Pass ``client`` to share an open Qdrant client (embedded Qdrant allows one per folder);
    otherwise a client is created and the caller owns it as ``retriever.client`` (close it on
    shutdown). Models load on the first search, so building the retriever is cheap. Raises
    :class:`~lexgraph.retrieval.store.IndexMismatchError` when the existing index was built
    with other encoders than the ones ``settings`` select.
    """
    dense = build_dense_embedder(settings)
    sparse = build_sparse_encoder(settings)
    reranker = CrossEncoderReranker(settings.reranker_model, cache_dir=settings.models_dir)
    retriever = LegalRetriever(
        client=client if client is not None else create_qdrant_client(settings),
        collection=settings.collection,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        mode=settings.retrieval_mode,
        k=settings.top_k,
        prefetch_k=settings.prefetch_k,
        rrf_k=settings.rrf_k,
        rerank=settings.rerank,
        rerank_candidates=settings.rerank_candidates,
    )
    try:
        retriever.check_index()
    except BaseException:
        if client is None:
            retriever.client.close()
        raise
    return retriever
