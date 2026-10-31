"""Index building: embed chunks into a new collection, then move the alias to it."""

import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from lexgraph.domain import Chunk
from lexgraph.retrieval.embeddings import ChunkEmbedder
from lexgraph.retrieval.sparse import SparseEncoder
from lexgraph.retrieval.store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    IndexMetadata,
    chunk_payload,
    create_collection,
    drop_collection,
    new_collection_name,
    physical_collections,
    point_alias,
    point_id,
    read_index_metadata,
)

logger = logging.getLogger(__name__)

# Called after each batch with (chunks indexed so far, total chunks).
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class IndexReport:
    """Outcome of :func:`index_chunks`."""

    alias: str
    collection: str
    chunks: int
    seconds: float
    dropped: tuple[str, ...] = ()


def index_chunks(
    client: QdrantClient,
    chunks: Sequence[Chunk],
    *,
    alias: str,
    dense: Embeddings,
    sparse: SparseEncoder,
    batch_size: int = 32,
    on_progress: ProgressCallback | None = None,
) -> IndexReport:
    """Index ``chunks`` into a new physical collection and point ``alias`` at it.

    The collection behind the alias keeps serving searches until the new one is complete (a
    blue/green swap). If indexing fails, the new collection is deleted and the alias is left
    as it was. After the swap, older physical collections of the alias are dropped. Point ids
    derive from the chunk ids, so a chunk keeps its id across re-indexing. The collection
    records its :class:`~lexgraph.retrieval.store.IndexMetadata`.
    """
    _check_chunks(chunks, batch_size)
    started = time.perf_counter()
    metadata = IndexMetadata.describe(chunks, dense=dense, sparse=sparse)
    collection = new_collection_name(client, alias)
    try:
        _fill(client, collection, chunks, dense, sparse, batch_size, on_progress, metadata)
    except BaseException:
        if client.collection_exists(collection):
            drop_collection(client, collection)
        raise
    point_alias(client, alias, collection)
    dropped = tuple(name for name in physical_collections(client, alias) if name != collection)
    for name in dropped:
        drop_collection(client, name)
    report = IndexReport(
        alias=alias,
        collection=collection,
        chunks=len(chunks),
        seconds=time.perf_counter() - started,
        dropped=dropped,
    )
    logger.debug(
        "indexed %d chunks into %s (alias %s) in %.1f s",
        report.chunks,
        report.collection,
        report.alias,
        report.seconds,
    )
    return report


def index_is_current(
    client: QdrantClient,
    chunks: Sequence[Chunk],
    *,
    alias: str,
    dense: Embeddings,
    sparse: SparseEncoder,
) -> bool:
    """Whether ``alias`` already serves exactly ``chunks``, built with the same encoders.

    Indexing them again would rebuild the same index. An index without metadata (built by an
    earlier version) is never current.
    """
    current = read_index_metadata(client, alias)
    return current == IndexMetadata.describe(chunks, dense=dense, sparse=sparse)


def _check_chunks(chunks: Sequence[Chunk], batch_size: int) -> None:
    if batch_size < 1:
        msg = f"batch_size must be at least 1, got {batch_size}"
        raise ValueError(msg)
    if not chunks:
        msg = "there are no chunks to index"
        raise ValueError(msg)
    counts = Counter(chunk.chunk_id for chunk in chunks)
    duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if duplicates:
        msg = f"duplicate chunk ids: {', '.join(duplicates)}"
        raise ValueError(msg)


def _fill(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    dense: Embeddings,
    sparse: SparseEncoder,
    batch_size: int,
    on_progress: ProgressCallback | None,
    metadata: IndexMetadata,
) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        dense_vectors = _dense_vectors(dense, batch)
        sparse_vectors = sparse.encode_documents([chunk.embedding_text() for chunk in batch])
        if start == 0:  # the first vectors give the dimension of the collection
            create_collection(client, collection, dim=len(dense_vectors[0]), metadata=metadata)
        points = [
            _point(chunk, dense_vector, sparse_vector)
            for chunk, dense_vector, sparse_vector in zip(
                batch, dense_vectors, sparse_vectors, strict=True
            )
        ]
        client.upsert(collection, points=points)
        if on_progress is not None:
            on_progress(start + len(batch), len(chunks))


def _dense_vectors(dense: Embeddings, chunks: Sequence[Chunk]) -> list[list[float]]:
    if isinstance(dense, ChunkEmbedder):
        return dense.embed_chunks(chunks)
    return dense.embed_documents([chunk.embedding_text() for chunk in chunks])


def _point(
    chunk: Chunk, dense_vector: list[float], sparse_vector: models.SparseVector
) -> models.PointStruct:
    return models.PointStruct(
        id=point_id(chunk.chunk_id),
        vector={DENSE_VECTOR: dense_vector, SPARSE_VECTOR: sparse_vector},
        payload=chunk_payload(chunk),
    )
