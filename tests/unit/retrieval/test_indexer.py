"""Index building: blue/green collections behind an alias, ids, payloads, metadata, failures."""

import re
from collections.abc import Sequence

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from pydantic import Field
from qdrant_client import QdrantClient

from lexgraph.domain import Chunk
from lexgraph.retrieval.indexer import IndexReport, index_chunks, index_is_current
from lexgraph.retrieval.store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    IndexMetadata,
    chunk_payload,
    create_collection,
    physical_collections,
    point_alias,
    point_id,
    read_index_metadata,
    resolve_alias,
)
from tests.unit.retrieval.conftest import ALIAS, DENSE_SIZE, HashingSparseEncoder


class ChunkAwareEmbedding(DeterministicFakeEmbedding):
    """Fake embedder that builds document vectors from chunks and records them."""

    seen: list[str] = Field(default_factory=list)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        self.seen.extend(chunk.chunk_id for chunk in chunks)
        return self.embed_documents([chunk.citation for chunk in chunks])


class FailingEmbedding(DeterministicFakeEmbedding):
    """Fake embedder that fails on batch number ``fail_on`` (the second by default)."""

    fail_on: int = 2
    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on:
            msg = "embedding service unavailable"
            raise RuntimeError(msg)
        return super().embed_documents(texts)


def index(
    client: QdrantClient, chunks: Sequence[Chunk], *, dense: Embeddings | None = None
) -> IndexReport:
    return index_chunks(
        client,
        chunks,
        alias=ALIAS,
        dense=dense or DeterministicFakeEmbedding(size=DENSE_SIZE),
        sparse=HashingSparseEncoder(),
        batch_size=4,
    )


def test_index_builds_a_collection_behind_the_alias(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    report = index(client, chunks)

    assert re.fullmatch(rf"{ALIAS}-\d{{14}}", report.collection)
    assert report.alias == ALIAS
    assert report.chunks == len(chunks)
    assert report.seconds >= 0
    assert report.dropped == ()
    assert resolve_alias(client, ALIAS) == report.collection
    assert client.count(ALIAS, exact=True).count == len(chunks)


def test_points_have_stable_ids_both_vectors_and_the_payload(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    index(client, chunks)
    chunk = chunks[0]

    (record,) = client.retrieve(ALIAS, [point_id(chunk.chunk_id)], with_vectors=True)

    assert record.payload == chunk_payload(chunk)
    assert isinstance(record.vector, dict)
    assert set(record.vector) == {DENSE_VECTOR, SPARSE_VECTOR}
    dense_vector = record.vector[DENSE_VECTOR]
    assert isinstance(dense_vector, list)
    assert len(dense_vector) == DENSE_SIZE


def test_index_records_its_chunks_and_encoders(client: QdrantClient, chunks: list[Chunk]) -> None:
    index(client, chunks)

    assert read_index_metadata(client, ALIAS) == IndexMetadata.describe(
        chunks, dense=DeterministicFakeEmbedding(size=DENSE_SIZE), sparse=HashingSparseEncoder()
    )


def is_current(client: QdrantClient, chunks: Sequence[Chunk], dense: Embeddings) -> bool:
    return index_is_current(client, chunks, alias=ALIAS, dense=dense, sparse=HashingSparseEncoder())


def test_index_is_current_for_the_same_chunks_and_encoders(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    dense = DeterministicFakeEmbedding(size=DENSE_SIZE)
    assert not is_current(client, chunks, dense)  # nothing indexed yet

    index(client, chunks)

    assert is_current(client, chunks, dense)
    assert not is_current(client, chunks[:-1], dense)
    edited = chunks[0].model_copy(update={"text": "Art. 49. Texto alterado."})
    assert not is_current(client, [edited, *chunks[1:]], dense)
    assert not is_current(client, chunks, ChunkAwareEmbedding(size=DENSE_SIZE))


def test_index_without_metadata_is_never_current(client: QdrantClient, chunks: list[Chunk]) -> None:
    create_collection(client, f"{ALIAS}-20261030120005", dim=DENSE_SIZE)
    point_alias(client, ALIAS, f"{ALIAS}-20261030120005")

    assert not is_current(client, chunks, DeterministicFakeEmbedding(size=DENSE_SIZE))


def test_progress_is_reported_after_each_batch(client: QdrantClient, chunks: list[Chunk]) -> None:
    progress: list[tuple[int, int]] = []

    index_chunks(
        client,
        chunks,
        alias=ALIAS,
        dense=DeterministicFakeEmbedding(size=DENSE_SIZE),
        sparse=HashingSparseEncoder(),
        batch_size=4,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(4, 10), (8, 10), (10, 10)]


def test_reindexing_swaps_the_alias_and_drops_the_old_collection(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    first = index(client, chunks)
    ids_before = {str(record.id) for record in client.scroll(ALIAS, limit=100)[0]}

    second = index(client, chunks)

    assert second.collection != first.collection
    assert second.dropped == (first.collection,)
    assert resolve_alias(client, ALIAS) == second.collection
    assert physical_collections(client, ALIAS) == [second.collection]
    assert client.count(ALIAS, exact=True).count == len(chunks)
    assert {str(record.id) for record in client.scroll(ALIAS, limit=100)[0]} == ids_before


def test_reindexing_a_subset_replaces_the_whole_index(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    index(client, chunks)

    index(client, chunks[:2])

    assert client.count(ALIAS, exact=True).count == 2


def test_chunk_aware_embedders_receive_the_chunks(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    embedder = ChunkAwareEmbedding(size=DENSE_SIZE)

    index(client, chunks, dense=embedder)

    assert embedder.seen == [chunk.chunk_id for chunk in chunks]


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_first_index_leaves_nothing_behind(
    client: QdrantClient, chunks: list[Chunk], fail_on: int
) -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        index(client, chunks, dense=FailingEmbedding(size=DENSE_SIZE, fail_on=fail_on))

    assert resolve_alias(client, ALIAS) is None
    assert physical_collections(client, ALIAS) == []


def test_failed_reindex_keeps_serving_the_previous_index(
    client: QdrantClient, chunks: list[Chunk]
) -> None:
    previous = index(client, chunks)

    with pytest.raises(RuntimeError, match="unavailable"):
        index(client, chunks, dense=FailingEmbedding(size=DENSE_SIZE))

    assert resolve_alias(client, ALIAS) == previous.collection
    assert physical_collections(client, ALIAS) == [previous.collection]
    assert client.count(ALIAS, exact=True).count == len(chunks)


def test_empty_input_is_rejected(client: QdrantClient) -> None:
    with pytest.raises(ValueError, match="no chunks"):
        index(client, [])

    assert client.get_collections().collections == []


def test_duplicate_chunk_ids_are_rejected(client: QdrantClient, chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="duplicate chunk ids: CDC:49#0"):
        index(client, [*chunks, chunks[0]])

    assert client.get_collections().collections == []


def test_batch_size_must_be_positive(client: QdrantClient, chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        index_chunks(
            client,
            chunks,
            alias=ALIAS,
            dense=DeterministicFakeEmbedding(size=DENSE_SIZE),
            sparse=HashingSparseEncoder(),
            batch_size=0,
        )
