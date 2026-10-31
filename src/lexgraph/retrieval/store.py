"""Qdrant access: client creation, collection schema, payload layout, aliases and metadata.

Each point has a named dense vector ``"dense"`` (cosine) and a named sparse vector ``"bm25"``
(Qdrant applies the IDF). The payload follows the ``langchain-qdrant`` layout, so its
``QdrantVectorStore`` can read the same collection::

    {
        "page_content": "<chunk text>",
        "metadata": {"chunk_id": ..., "law_id": ..., "content_hash": ...},
    }

Searches go through an alias (``settings.collection``) that points at a physical collection
named ``"{alias}-{UTC yyyymmddHHMMSS}"``; re-indexing builds a new one and moves the alias.

The collection metadata records what the index was built from (:class:`IndexMetadata`): the
fingerprint of the dense and sparse encoders, since a query embedded by another model lands in
another vector space and finds noise without any error, and a digest of the indexed chunks.
"""

import hashlib
import re
import sqlite3
import uuid
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.local.persistence import CollectionPersistence
from qdrant_client.local.qdrant_local import QdrantLocal

from lexgraph.config import Settings
from lexgraph.domain import Chunk, Passage

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
LAW_ID_KEY = "metadata.law_id"
ARTICLE_ID_KEY = "metadata.article_id"

# Namespace of the uuid5 point ids; changing it changes every id.
POINT_ID_NAMESPACE = uuid.UUID("742929ce-e7e6-47f0-92e1-fd1389002312")

# Key of the :class:`IndexMetadata` in the collection metadata.
INDEX_METADATA_KEY = "lexgraph"


class StorageLockedError(RuntimeError):
    """The embedded Qdrant folder is already open in another client or process."""


class IndexMismatchError(RuntimeError):
    """The index was built with other encoders than the ones configured for searching it."""


@runtime_checkable
class Fingerprinted(Protocol):
    """An encoder that reports the settings its vectors depend on."""

    def fingerprint(self) -> dict[str, str]:
        """Settings that determine the vectors (provider, model, prompts, size...)."""
        ...


def encoder_fingerprint(dense: object, sparse: object) -> dict[str, str]:
    """Fingerprint of a dense and a sparse encoder, as ``"dense.<key>"``/``"sparse.<key>"``.

    An encoder without a ``fingerprint()`` method is identified by its class.
    """
    return {
        f"{role}.{key}": value
        for role, encoder in (("dense", dense), ("sparse", sparse))
        for key, value in _fingerprint(encoder).items()
    }


class IndexMetadata(BaseModel):
    """What an index was built from, stored in the metadata of its collection."""

    model_config = ConfigDict(frozen=True)

    encoders: dict[str, str] = Field(description="Fingerprint of the dense and sparse encoders.")
    chunks: int = Field(ge=0, description="Number of indexed chunks.")
    chunks_sha256: str = Field(description="Digest of the indexed chunks, in order.")

    @classmethod
    def describe(cls, chunks: Sequence[Chunk], *, dense: object, sparse: object) -> Self:
        """Metadata of an index of ``chunks`` built with the ``dense`` and ``sparse`` encoders."""
        return cls(
            encoders=encoder_fingerprint(dense, sparse),
            chunks=len(chunks),
            chunks_sha256=chunks_digest(chunks),
        )


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Connect to the Qdrant server at ``settings.qdrant_url``, or open embedded Qdrant.

    Embedded Qdrant keeps its data in ``settings.qdrant_path`` and allows one client per
    folder across all processes; opening a second one raises :class:`StorageLockedError`.
    """
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    probe_sqlite_threading()
    try:
        return QdrantClient(path=str(settings.qdrant_path))
    except RuntimeError as exc:
        if "already accessed" not in str(exc):
            raise
        msg = f"embedded Qdrant at {settings.qdrant_path} is in use by another client or process"
        raise StorageLockedError(msg) from exc


def probe_sqlite_threading() -> None:
    """Answer embedded Qdrant's question about SQLite threading before it asks it.

    The first embedded collection of a process checks whether SQLite connections may cross
    threads through a throwaway connection that is never closed, which Python 3.13 reports
    as an unclosed database. The same check runs here on a connection that is closed, and
    its answer is recorded where embedded Qdrant looks for it.
    """
    if CollectionPersistence.CHECK_SAME_THREAD is not None:
        return
    with closing(sqlite3.connect(":memory:")) as connection:
        (option,) = connection.execute(
            "SELECT * FROM pragma_compile_options WHERE compile_options LIKE 'THREADSAFE=%'"
        ).fetchone()
    CollectionPersistence.CHECK_SAME_THREAD = option != "THREADSAFE=1"


def create_collection(
    client: QdrantClient, name: str, dim: int, *, metadata: IndexMetadata | None = None
) -> None:
    """Create a collection for chunks, with keyword indexes on the law and article ids.

    ``metadata`` is stored with the collection (see :func:`read_index_metadata`). Payload
    indexes only take effect on a Qdrant server; embedded Qdrant ignores them and warns, so
    that warning is silenced.
    """
    client.create_collection(
        name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata={INDEX_METADATA_KEY: metadata.model_dump()} if metadata is not None else None,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Payload indexes have no effect", category=UserWarning
        )
        for key in (LAW_ID_KEY, ARTICLE_ID_KEY):
            client.create_payload_index(name, key, field_schema=models.PayloadSchemaType.KEYWORD)


def point_id(chunk_id: str) -> str:
    """Stable point id of a chunk, so the same chunk always gets the same id."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def content_hash(chunk: Chunk) -> str:
    """SHA-256 of the chunk as JSON, to tell whether an indexed chunk has changed."""
    return hashlib.sha256(chunk.model_dump_json().encode("utf-8")).hexdigest()


def chunks_digest(chunks: Iterable[Chunk]) -> str:
    """SHA-256 over the content hashes of ``chunks``, in order: equal iff the chunks are."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(content_hash(chunk).encode("ascii"))
    return digest.hexdigest()


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    """Payload stored with a chunk: its text plus every other field as metadata."""
    metadata = chunk.model_dump(mode="json", exclude={"text"})
    return {
        "page_content": chunk.text,
        "metadata": {**metadata, "content_hash": content_hash(chunk)},
    }


def passage_from_payload(payload: Mapping[str, Any] | None, *, score: float) -> Passage:
    """Rebuild an indexed chunk from its payload, as a passage with ``score``."""
    if payload is None:
        msg = "the point has no payload"
        raise ValueError(msg)
    fields = {key: value for key, value in payload["metadata"].items() if key != "content_hash"}
    return Passage.model_validate({**fields, "text": payload["page_content"], "score": score})


def new_collection_name(client: QdrantClient, alias: str, now: datetime | None = None) -> str:
    """Name of a new physical collection for ``alias``: ``"{alias}-{UTC yyyymmddHHMMSS}"``.

    A ``-2``, ``-3``... suffix is added when that name is taken (two builds in one second).
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    base = f"{alias}-{stamp}"
    taken = set(_collection_names(client))
    name, attempt = base, 1
    while name in taken:
        attempt += 1
        name = f"{base}-{attempt}"
    return name


def physical_collections(client: QdrantClient, alias: str) -> list[str]:
    """Physical collections built for ``alias`` by :func:`new_collection_name`, sorted."""
    pattern = re.compile(rf"{re.escape(alias)}-\d{{14}}(?:-\d+)?")
    return sorted(name for name in _collection_names(client) if pattern.fullmatch(name))


def resolve_alias(client: QdrantClient, alias: str) -> str | None:
    """Collection that ``alias`` points at, or ``None`` when there is no such alias."""
    for description in client.get_aliases().aliases:
        if description.alias_name == alias:
            return description.collection_name
    return None


def point_alias(client: QdrantClient, alias: str, collection: str) -> None:
    """Point ``alias`` at ``collection`` in one atomic operation (creating it if needed)."""
    operations: list[models.AliasOperations] = []
    if resolve_alias(client, alias) is not None:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def drop_collection(client: QdrantClient, name: str) -> None:
    """Delete the physical collection ``name`` and its files.

    Embedded Qdrant forgets a deleted collection without closing its SQLite storage: the file
    stays open until garbage collection, so Windows cannot remove the folder (it is left
    behind) and Python 3.13 warns about an unclosed database. The storage is closed first.
    """
    local = client._client
    if isinstance(local, QdrantLocal) and (collection := local.collections.get(name)):
        collection.close()
    client.delete_collection(name)


def collection_exists(client: QdrantClient, name: str) -> bool:
    """Whether ``name`` is an alias or a collection."""
    return resolve_alias(client, name) is not None or client.collection_exists(name)


def read_index_metadata(client: QdrantClient, name: str) -> IndexMetadata | None:
    """Metadata of the collection that ``name`` (an alias or a collection) designates.

    ``None`` when there is no such collection or it records no valid :class:`IndexMetadata`
    (an index built before the metadata existed, or by another tool).
    """
    collection = resolve_alias(client, name) or name
    if not client.collection_exists(collection):
        return None
    stored = (client.get_collection(collection).config.metadata or {}).get(INDEX_METADATA_KEY)
    if stored is None:
        return None
    try:
        return IndexMetadata.model_validate(stored)
    except ValidationError:
        return None


def _fingerprint(encoder: object) -> dict[str, str]:
    if isinstance(encoder, Fingerprinted):
        return encoder.fingerprint()
    kind = type(encoder)
    return {"class": f"{kind.__module__}.{kind.__qualname__}"}


def _collection_names(client: QdrantClient) -> list[str]:
    return [description.name for description in client.get_collections().collections]
