"""Qdrant client creation, collection schema, payload layout, aliases and index metadata."""

import gc
import json
import sqlite3
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import SecretStr
from qdrant_client import QdrantClient, models
from qdrant_client.local.persistence import CollectionPersistence
from qdrant_client.local.qdrant_local import QdrantLocal

from lexgraph.config import Settings
from lexgraph.domain import Chunk
from lexgraph.retrieval import store as store_module
from lexgraph.retrieval.store import (
    DENSE_VECTOR,
    INDEX_METADATA_KEY,
    SPARSE_VECTOR,
    Fingerprinted,
    IndexMetadata,
    StorageLockedError,
    chunk_payload,
    chunks_digest,
    collection_exists,
    content_hash,
    create_collection,
    create_qdrant_client,
    drop_collection,
    encoder_fingerprint,
    new_collection_name,
    passage_from_payload,
    physical_collections,
    point_alias,
    point_id,
    probe_sqlite_threading,
    read_index_metadata,
    resolve_alias,
)
from tests.unit.retrieval.conftest import CHUNKS, HashingSparseEncoder, chunk_by_id

NOW = datetime(2026, 10, 30, 12, 0, 5, tzinfo=UTC)


def test_embedded_client_creates_its_folder(settings: Settings) -> None:
    client = create_qdrant_client(settings)
    try:
        create_collection(client, "laws", dim=4)
        assert settings.qdrant_path.is_dir()
        assert client.collection_exists("laws")
    finally:
        client.close()


def test_locked_embedded_folder_is_reported(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The error qdrant-client raises when another client holds the folder. Opening the folder
    # twice for real would leak the lock file handle inside qdrant-client.
    def locked(**options: Any) -> QdrantClient:
        msg = (
            f"Storage folder {options['path']} is already accessed by another instance of "
            "Qdrant client. If you require concurrent access, use Qdrant server instead."
        )
        raise RuntimeError(msg)

    monkeypatch.setattr(store_module, "QdrantClient", locked)

    with pytest.raises(StorageLockedError, match="in use by another client or process"):
        create_qdrant_client(settings)


class RecordingClient:
    """Stands in for ``QdrantClient``: records the constructor arguments."""

    def __init__(self, **options: Any) -> None:
        self.options = options


def test_server_client_uses_url_and_api_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "QdrantClient", RecordingClient)
    server = settings.model_copy(
        update={"qdrant_url": "http://qdrant:6333", "qdrant_api_key": SecretStr("s3cret")}
    )

    client = create_qdrant_client(server)

    assert isinstance(client, RecordingClient)
    assert client.options == {"url": "http://qdrant:6333", "api_key": "s3cret"}
    assert not settings.qdrant_path.exists()


def test_server_client_without_api_key(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "QdrantClient", RecordingClient)

    client = create_qdrant_client(settings.model_copy(update={"qdrant_url": "http://qdrant:6333"}))

    assert isinstance(client, RecordingClient)
    assert client.options == {"url": "http://qdrant:6333", "api_key": None}


def test_other_embedded_errors_are_not_reported_as_a_lock(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(**options: Any) -> QdrantClient:
        msg = "storage is corrupted"
        raise RuntimeError(msg)

    monkeypatch.setattr(store_module, "QdrantClient", broken)

    with pytest.raises(RuntimeError, match="corrupted") as error:
        create_qdrant_client(settings)
    assert not isinstance(error.value, StorageLockedError)


def test_collection_has_named_dense_and_idf_sparse_vectors(client: QdrantClient) -> None:
    create_collection(client, "laws", dim=32)

    params = client.get_collection("laws").config.params
    assert isinstance(params.vectors, dict)
    assert params.vectors[DENSE_VECTOR].size == 32
    assert params.vectors[DENSE_VECTOR].distance == models.Distance.COSINE
    assert params.sparse_vectors is not None
    assert params.sparse_vectors[SPARSE_VECTOR].modifier == models.Modifier.IDF


def test_point_id_is_a_stable_uuid5() -> None:
    first = point_id("CDC:49#0")

    assert first == point_id("CDC:49#0")
    assert uuid.UUID(first).version == 5
    assert first != point_id("CDC:49#1")


def test_payload_follows_the_langchain_qdrant_layout() -> None:
    chunk = chunk_by_id("CF:5#1")

    payload = chunk_payload(chunk)

    assert set(payload) == {"page_content", "metadata"}
    assert payload["page_content"] == chunk.text
    metadata = payload["metadata"]
    assert metadata["content_hash"] == content_hash(chunk)
    assert {key: value for key, value in metadata.items() if key != "content_hash"} == (
        chunk.model_dump(mode="json", exclude={"text"})
    )
    assert metadata["law_id"] == "CF"
    assert json.loads(json.dumps(payload)) == payload


def test_content_hash_changes_with_the_content() -> None:
    chunk = chunk_by_id("CDC:49#0")
    changed = chunk.model_copy(update={"text": chunk.text + " Parágrafo único."})

    assert len(content_hash(chunk)) == 64
    assert content_hash(chunk) == content_hash(Chunk.model_validate(chunk.model_dump()))
    assert content_hash(changed) != content_hash(chunk)


def test_passage_round_trips_through_the_payload() -> None:
    chunk = chunk_by_id("CF:5#2")

    passage = passage_from_payload(chunk_payload(chunk), score=0.75)

    assert passage.score == 0.75
    assert passage.rerank_score is None
    assert passage.model_dump(exclude={"score", "rerank_score"}) == chunk.model_dump()


def test_passage_needs_a_payload() -> None:
    with pytest.raises(ValueError, match="no payload"):
        passage_from_payload(None, score=1.0)


def test_new_collection_name_uses_the_utc_timestamp(client: QdrantClient) -> None:
    assert new_collection_name(client, "lexgraph", NOW) == "lexgraph-20261030120005"


def test_new_collection_name_avoids_taken_names(client: QdrantClient) -> None:
    create_collection(client, "lexgraph-20261030120005", dim=4)
    create_collection(client, "lexgraph-20261030120005-2", dim=4)

    assert new_collection_name(client, "lexgraph", NOW) == "lexgraph-20261030120005-3"


def test_physical_collections_belong_to_one_alias(client: QdrantClient) -> None:
    for name in (
        "lexgraph-20261030120005",
        "lexgraph-20261029090000-2",
        "lexgraph-test-20261030120005",
        "lexgraph",
        "other",
    ):
        create_collection(client, name, dim=4)

    assert physical_collections(client, "lexgraph") == [
        "lexgraph-20261029090000-2",
        "lexgraph-20261030120005",
    ]


def test_sqlite_threading_is_probed_on_a_closed_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CollectionPersistence, "CHECK_SAME_THREAD", None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        probe_sqlite_threading()
        gc.collect()

    assert isinstance(CollectionPersistence.CHECK_SAME_THREAD, bool)


def test_sqlite_threading_probe_keeps_an_earlier_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CollectionPersistence, "CHECK_SAME_THREAD", True)

    probe_sqlite_threading()

    assert CollectionPersistence.CHECK_SAME_THREAD is True


def test_drop_collection_closes_the_embedded_storage(tmp_path: Path) -> None:
    embedded = QdrantClient(path=str(tmp_path))
    create_collection(embedded, "lexgraph-20261030120005", dim=4)
    local = embedded._client
    assert isinstance(local, QdrantLocal)
    storage = local.collections["lexgraph-20261030120005"].storage
    assert storage is not None

    drop_collection(embedded, "lexgraph-20261030120005")

    assert not embedded.collection_exists("lexgraph-20261030120005")
    with pytest.raises(sqlite3.ProgrammingError):
        storage.storage.execute("SELECT 1")
    # With the file closed, the folder can be removed on every platform.
    assert not (tmp_path / "collection" / "lexgraph-20261030120005").exists()
    embedded.close()


def test_drop_collection_works_without_embedded_storage(client: QdrantClient) -> None:
    create_collection(client, "other", dim=4)

    drop_collection(client, "other")

    assert not client.collection_exists("other")


def test_point_alias_creates_and_moves_the_alias(client: QdrantClient) -> None:
    create_collection(client, "lexgraph-1", dim=4)
    create_collection(client, "lexgraph-2", dim=4)
    assert resolve_alias(client, "lexgraph") is None

    point_alias(client, "lexgraph", "lexgraph-1")
    assert resolve_alias(client, "lexgraph") == "lexgraph-1"

    point_alias(client, "lexgraph", "lexgraph-2")
    assert resolve_alias(client, "lexgraph") == "lexgraph-2"
    assert [alias.alias_name for alias in client.get_aliases().aliases] == ["lexgraph"]


def test_collection_exists_accepts_aliases_and_collections(client: QdrantClient) -> None:
    create_collection(client, "lexgraph-1", dim=4)
    point_alias(client, "lexgraph", "lexgraph-1")

    assert collection_exists(client, "lexgraph")
    assert collection_exists(client, "lexgraph-1")
    assert not collection_exists(client, "missing")


class FingerprintedEncoder:
    """An encoder that reports its own fingerprint."""

    def fingerprint(self) -> dict[str, str]:
        return {"provider": "fake", "model": "fake-model"}


def test_encoder_fingerprint_prefixes_each_role() -> None:
    assert isinstance(FingerprintedEncoder(), Fingerprinted)

    fingerprint = encoder_fingerprint(FingerprintedEncoder(), FingerprintedEncoder())

    assert fingerprint == {
        "dense.provider": "fake",
        "dense.model": "fake-model",
        "sparse.provider": "fake",
        "sparse.model": "fake-model",
    }


def test_encoders_without_a_fingerprint_are_identified_by_their_class() -> None:
    fingerprint = encoder_fingerprint(DeterministicFakeEmbedding(size=4), HashingSparseEncoder())

    assert fingerprint == {
        "dense.class": "langchain_core.embeddings.fake.DeterministicFakeEmbedding",
        "sparse.class": "tests.unit.retrieval.conftest.HashingSparseEncoder",
    }


def test_chunks_digest_depends_on_content_and_order() -> None:
    first, second = CHUNKS[:2]
    changed = first.model_copy(update={"text": first.text + " Parágrafo único."})

    assert chunks_digest([first, second]) == chunks_digest([first, second])
    assert chunks_digest([first, second]) != chunks_digest([second, first])
    assert chunks_digest([changed, second]) != chunks_digest([first, second])
    assert len(chunks_digest([])) == 64


def test_index_metadata_describes_chunks_and_encoders() -> None:
    dense, sparse = FingerprintedEncoder(), HashingSparseEncoder()

    metadata = IndexMetadata.describe(CHUNKS, dense=dense, sparse=sparse)

    assert metadata.encoders == encoder_fingerprint(dense, sparse)
    assert metadata.chunks == len(CHUNKS)
    assert metadata.chunks_sha256 == chunks_digest(CHUNKS)


def test_index_metadata_round_trips_through_the_collection(client: QdrantClient) -> None:
    metadata = IndexMetadata(encoders={"dense.model": "m"}, chunks=3, chunks_sha256="0" * 64)
    create_collection(client, "lexgraph-1", dim=4, metadata=metadata)
    point_alias(client, "lexgraph", "lexgraph-1")

    assert read_index_metadata(client, "lexgraph") == metadata
    assert read_index_metadata(client, "lexgraph-1") == metadata


def test_index_metadata_is_none_when_absent_or_foreign(client: QdrantClient) -> None:
    create_collection(client, "plain", dim=4)
    client.create_collection(
        "foreign",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        metadata={INDEX_METADATA_KEY: {"encoders": "not a mapping"}, "owner": "other tool"},
    )

    assert read_index_metadata(client, "plain") is None
    assert read_index_metadata(client, "foreign") is None
    assert read_index_metadata(client, "missing") is None
