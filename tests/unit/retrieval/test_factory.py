"""Composition of the retriever from the settings (no model is loaded)."""

import pytest
from qdrant_client import QdrantClient

from lexgraph.config import Settings
from lexgraph.retrieval.embeddings import DenseEmbedder, build_dense_embedder
from lexgraph.retrieval.factory import build_retriever
from lexgraph.retrieval.reranker import CrossEncoderReranker
from lexgraph.retrieval.sparse import Bm25Encoder, build_sparse_encoder
from lexgraph.retrieval.store import (
    IndexMetadata,
    IndexMismatchError,
    create_collection,
    point_alias,
)
from tests.unit.retrieval.conftest import CHUNKS


def index_built_with(settings: Settings, client: QdrantClient, **changes: object) -> None:
    """An (empty) index whose metadata records the encoders of ``settings`` with ``changes``."""
    built_with = settings.model_copy(update=changes)
    metadata = IndexMetadata.describe(
        CHUNKS, dense=build_dense_embedder(built_with), sparse=build_sparse_encoder(built_with)
    )
    create_collection(client, f"{settings.collection}-20261030120005", dim=4, metadata=metadata)
    point_alias(client, settings.collection, f"{settings.collection}-20261030120005")


def test_build_retriever_wires_the_settings(settings: Settings, client: QdrantClient) -> None:
    custom = settings.model_copy(
        update={
            "collection": "laws",
            "retrieval_mode": "sparse",
            "top_k": 4,
            "prefetch_k": 12,
            "rrf_k": 60,
            "rerank": True,
            "rerank_candidates": 8,
        }
    )

    retriever = build_retriever(custom, client=client)

    assert retriever.client is client
    assert retriever.collection == "laws"
    assert (retriever.mode, retriever.k, retriever.prefetch_k, retriever.rrf_k) == (
        "sparse",
        4,
        12,
        60,
    )
    assert (retriever.rerank, retriever.rerank_candidates) == (True, 8)


def test_build_retriever_uses_the_configured_models(
    settings: Settings, client: QdrantClient
) -> None:
    retriever = build_retriever(settings, client=client)

    assert isinstance(retriever.dense, DenseEmbedder)
    assert retriever.dense.model_name == settings.embedding_model
    assert retriever.dense.cache_dir == settings.models_dir
    assert isinstance(retriever.sparse, Bm25Encoder)
    assert retriever.sparse.model_name == settings.sparse_model
    assert isinstance(retriever.reranker, CrossEncoderReranker)
    assert retriever.reranker.model_name == settings.reranker_model
    assert retriever.reranker.cache_dir == settings.models_dir


def test_build_retriever_accepts_an_index_of_the_same_encoders(
    settings: Settings, client: QdrantClient
) -> None:
    index_built_with(settings, client)

    assert build_retriever(settings, client=client).collection == settings.collection


def test_build_retriever_rejects_an_index_of_other_encoders(
    settings: Settings, client: QdrantClient
) -> None:
    index_built_with(settings, client, embedding_model="intfloat/multilingual-e5-large")

    with pytest.raises(IndexMismatchError, match=r"dense\.model: 'intfloat/multilingual-e5-large'"):
        build_retriever(settings, client=client)

    assert client.collection_exists(f"{settings.collection}-20261030120005")  # still usable


def test_build_retriever_closes_its_own_client_on_a_mismatch(settings: Settings) -> None:
    owned = QdrantClient(path=str(settings.qdrant_path))
    index_built_with(settings, owned, embedding_model="intfloat/multilingual-e5-large")
    owned.close()

    with pytest.raises(IndexMismatchError):
        build_retriever(settings)

    reopened = QdrantClient(path=str(settings.qdrant_path))  # the folder lock was released
    reopened.close()


def test_build_retriever_opens_embedded_qdrant_by_default(settings: Settings) -> None:
    retriever = build_retriever(settings)
    try:
        assert settings.qdrant_path.is_dir()
        assert retriever.count() == 0
    finally:
        retriever.client.close()
