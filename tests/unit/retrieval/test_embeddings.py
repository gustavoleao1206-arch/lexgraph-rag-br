"""Dense embedders: model prompts, lazy FastEmbed loading and the Gemini API client."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from google.genai import errors as genai_errors
from numpy.typing import NDArray
from tenacity import wait_none

from lexgraph.config import Settings
from lexgraph.retrieval import embeddings as embeddings_module
from lexgraph.retrieval.embeddings import (
    E5_PROMPTS,
    EMBEDDINGGEMMA_PROMPTS,
    GEMINI_EMBEDDING_MODEL,
    ChunkEmbedder,
    DenseEmbedder,
    GoogleDenseEmbedder,
    ModelPrompts,
    build_dense_embedder,
    embedding_model_name,
    prompts_for,
)
from lexgraph.retrieval.store import Fingerprinted
from tests.unit.retrieval.conftest import chunk_by_id


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("google/embeddinggemma-300m", EMBEDDINGGEMMA_PROMPTS),
        ("Google/EmbeddingGemma-300m", EMBEDDINGGEMMA_PROMPTS),
        ("intfloat/multilingual-e5-large", E5_PROMPTS),
        ("intfloat/e5-base-v2", E5_PROMPTS),
        ("BAAI/bge-small-en-v1.5", ModelPrompts()),
    ],
)
def test_prompts_for_model_family(model_name: str, expected: ModelPrompts) -> None:
    assert prompts_for(model_name) == expected


def test_document_prompt_takes_a_title_or_none() -> None:
    assert EMBEDDINGGEMMA_PROMPTS.document_text("texto", title="CDC, art. 49") == (
        "title: CDC, art. 49 | text: texto"
    )
    assert EMBEDDINGGEMMA_PROMPTS.document_text("texto") == "title: none | text: texto"
    assert E5_PROMPTS.document_text("texto", title="ignored") == "passage: texto"
    assert ModelPrompts().query_text("pergunta") == "pergunta"


class FakeTextEmbedding:
    """Stands in for FastEmbed's ``TextEmbedding``: vector = (text length, 1)."""

    def __init__(self, model_name: str, cache_dir: str | None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.texts: list[str] = []

    def embed(self, documents: list[str], batch_size: int = 256) -> Iterator[NDArray[np.float32]]:
        self.texts.extend(documents)
        for text in documents:
            yield np.array([len(text), 1.0], dtype=np.float32)


@pytest.fixture
def text_models(monkeypatch: pytest.MonkeyPatch) -> list[FakeTextEmbedding]:
    """Replace ``TextEmbedding`` with :class:`FakeTextEmbedding`; returns the instances."""
    created: list[FakeTextEmbedding] = []

    def factory(model_name: str, *, cache_dir: str | None = None) -> FakeTextEmbedding:
        model = FakeTextEmbedding(model_name, cache_dir)
        created.append(model)
        return model

    monkeypatch.setattr(embeddings_module, "TextEmbedding", factory)
    return created


def test_dense_embedder_loads_the_model_once_on_first_use(
    text_models: list[FakeTextEmbedding], tmp_path: Path
) -> None:
    embedder = DenseEmbedder("google/embeddinggemma-300m", cache_dir=tmp_path)
    assert text_models == []

    embedder.embed_query("férias")
    embedder.embed_documents(["texto"])

    assert len(text_models) == 1
    assert text_models[0].model_name == "google/embeddinggemma-300m"
    assert text_models[0].cache_dir == str(tmp_path)


def test_dense_embedder_adds_the_query_prompt(text_models: list[FakeTextEmbedding]) -> None:
    vector = DenseEmbedder("google/embeddinggemma-300m").embed_query("férias")

    assert text_models[0].texts == ["task: search result | query: férias"]
    assert vector == [float(len("task: search result | query: férias")), 1.0]


def test_dense_embedder_adds_the_document_prompt(text_models: list[FakeTextEmbedding]) -> None:
    vectors = DenseEmbedder("google/embeddinggemma-300m").embed_documents(["a", "bb"])

    assert text_models[0].texts == ["title: none | text: a", "title: none | text: bb"]
    assert len(vectors) == 2


def test_dense_embedder_titles_chunks_with_their_citation(
    text_models: list[FakeTextEmbedding],
) -> None:
    chunk = chunk_by_id("CDC:49#0")

    vectors = DenseEmbedder("google/embeddinggemma-300m").embed_chunks([chunk])

    assert text_models[0].texts == [f"title: CDC, art. 49 | text: {chunk.embedding_text()}"]
    assert len(vectors) == 1


def test_dense_embedder_without_known_prompts_embeds_raw_text(
    text_models: list[FakeTextEmbedding],
) -> None:
    embedder = DenseEmbedder("BAAI/bge-small-en-v1.5")

    embedder.embed_query("q")
    embedder.embed_documents(["d"])

    assert text_models[0].texts == ["q", "d"]


def test_dense_embedder_accepts_explicit_prompts(text_models: list[FakeTextEmbedding]) -> None:
    embedder = DenseEmbedder("custom/model", prompts=ModelPrompts(query="Q: ", document="D: "))

    embedder.embed_query("q")
    embedder.embed_documents(["d"])

    assert text_models[0].texts == ["Q: q", "D: d"]


GeminiHandler = Callable[[httpx.Request], httpx.Response]


def gemini_api(requests: list[httpx.Request], statuses: list[int]) -> GeminiHandler:
    """Fake batchEmbedContents endpoint: answers with ``statuses`` in turn, then 200."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = statuses.pop(0) if statuses else 200
        if status != httpx.codes.OK:
            error = {"code": status, "message": "simulated failure", "status": "UNAVAILABLE"}
            return httpx.Response(status, json={"error": error})
        batch = json.loads(request.content)["requests"]
        vectors = [{"values": [float(index), 0.5, 0.25]} for index in range(len(batch))]
        return httpx.Response(200, json={"embeddings": vectors})

    return handle


def gemini_embedder(handler: GeminiHandler, *, max_attempts: int = 3) -> GoogleDenseEmbedder:
    return GoogleDenseEmbedder(
        dimensions=3,
        api_key="test-key",
        max_attempts=max_attempts,
        wait=wait_none(),
        client_args={"transport": httpx.MockTransport(handler)},
    )


def request_body(request: httpx.Request) -> list[dict[str, object]]:
    batch: list[dict[str, object]] = json.loads(request.content)["requests"]
    return batch


def test_gemini_query_uses_the_query_task_type() -> None:
    requests: list[httpx.Request] = []

    vector = gemini_embedder(gemini_api(requests, [])).embed_query("férias")

    assert vector == [0.0, 0.5, 0.25]
    (request,) = requests
    assert request.url.path.endswith(f"/models/{GEMINI_EMBEDDING_MODEL}:batchEmbedContents")
    assert request.headers["x-goog-api-key"] == "test-key"
    assert request.extensions["timeout"]["read"] == 30.0
    (item,) = request_body(request)
    assert item["taskType"] == "RETRIEVAL_QUERY"
    assert item["outputDimensionality"] == 3
    assert item["content"] == {"parts": [{"text": "férias"}], "role": "user"}


def test_gemini_documents_are_sent_in_batches_of_100() -> None:
    requests: list[httpx.Request] = []

    vectors = gemini_embedder(gemini_api(requests, [])).embed_documents(
        [f"texto {n}" for n in range(101)]
    )

    assert len(vectors) == 101
    assert [len(request_body(request)) for request in requests] == [100, 1]
    assert {item["taskType"] for item in request_body(requests[0])} == {"RETRIEVAL_DOCUMENT"}


def test_gemini_chunks_embed_header_and_text() -> None:
    requests: list[httpx.Request] = []
    chunk = chunk_by_id("CDC:49#0")

    gemini_embedder(gemini_api(requests, [])).embed_chunks([chunk])

    (item,) = request_body(requests[0])
    assert item["content"] == {"parts": [{"text": chunk.embedding_text()}], "role": "user"}


@pytest.mark.parametrize("status", [429, 500, 503])
def test_gemini_retries_transient_errors(status: int) -> None:
    requests: list[httpx.Request] = []

    vector = gemini_embedder(gemini_api(requests, [status, status])).embed_query("férias")

    assert vector == [0.0, 0.5, 0.25]
    assert len(requests) == 3


def test_gemini_retries_network_errors() -> None:
    attempts: list[httpx.Request] = []
    answer = gemini_api([], [])

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return answer(request)

    assert gemini_embedder(flaky).embed_query("férias") == [0.0, 0.5, 0.25]
    assert len(attempts) == 2


def test_gemini_gives_up_after_max_attempts() -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(genai_errors.ServerError):
        gemini_embedder(gemini_api(requests, [503] * 5), max_attempts=2).embed_query("férias")

    assert len(requests) == 2


def test_gemini_does_not_retry_client_errors() -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(genai_errors.ClientError):
        gemini_embedder(gemini_api(requests, [400])).embed_query("férias")

    assert len(requests) == 1


def test_gemini_rejects_a_response_with_missing_vectors() -> None:
    def short_answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0, 0.0]}]})

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        gemini_embedder(short_answer).embed_documents(["a", "b"])


def test_embedders_build_vectors_from_chunks() -> None:
    assert isinstance(DenseEmbedder("google/embeddinggemma-300m"), ChunkEmbedder)
    assert isinstance(gemini_embedder(gemini_api([], [])), ChunkEmbedder)


def test_embedders_report_what_their_vectors_depend_on() -> None:
    fastembed = DenseEmbedder("google/embeddinggemma-300m")
    gemini = gemini_embedder(gemini_api([], []))

    assert isinstance(fastembed, Fingerprinted)
    assert fastembed.fingerprint() == {
        "provider": "fastembed",
        "model": "google/embeddinggemma-300m",
        "query_prompt": EMBEDDINGGEMMA_PROMPTS.query,
        "document_prompt": EMBEDDINGGEMMA_PROMPTS.document,
    }
    assert isinstance(gemini, Fingerprinted)
    assert gemini.fingerprint() == {
        "provider": "google",
        "model": GEMINI_EMBEDDING_MODEL,
        "dimensions": "3",
    }


def test_build_dense_embedder_defaults_to_fastembed(
    settings: Settings, text_models: list[FakeTextEmbedding]
) -> None:
    embedder = build_dense_embedder(settings)

    assert isinstance(embedder, DenseEmbedder)
    assert embedder.model_name == settings.embedding_model
    assert embedder.cache_dir == settings.models_dir
    assert text_models == []


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("google/embeddinggemma-300m", GEMINI_EMBEDDING_MODEL),
        ("gemini-embedding-2", "gemini-embedding-2"),
        ("models/gemini-embedding-001", "models/gemini-embedding-001"),
    ],
)
def test_build_dense_embedder_for_google(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, configured: str, expected: str
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    google_settings = settings.model_copy(
        update={"embedding_provider": "google", "embedding_model": configured, "embedding_dim": 256}
    )

    embedder = build_dense_embedder(google_settings)

    assert isinstance(embedder, GoogleDenseEmbedder)
    assert embedder.model == expected == embedding_model_name(google_settings)
    assert embedder.dimensions == 256


def test_embedding_model_name_with_fastembed(settings: Settings) -> None:
    gemini_named = settings.model_copy(update={"embedding_model": "gemini-embedding-001"})

    assert embedding_model_name(settings) == settings.embedding_model
    assert embedding_model_name(gemini_named) == "gemini-embedding-001"
