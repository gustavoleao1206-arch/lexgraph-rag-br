"""Dense embedders: a local FastEmbed model by default, or the Gemini embedding API."""

import itertools
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from fastembed import TextEmbedding
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from langchain_core.embeddings import Embeddings
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import WaitBaseT

from lexgraph.config import Settings
from lexgraph.domain import Chunk

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

_GEMINI_MAX_BATCH = 100  # texts per batchEmbedContents request (API limit)
_GEMINI_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_GEMINI_BACKOFF = wait_exponential_jitter(initial=1, max=30)


@runtime_checkable
class ChunkEmbedder(Protocol):
    """Embeddings that build document vectors from chunks, using more than their text."""

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Return one document vector per chunk."""
        ...


@dataclass(frozen=True)
class ModelPrompts:
    """Instruction prefixes a retrieval model expects in front of queries and documents.

    ``document`` may contain a ``{title}`` placeholder, filled with the chunk citation (or
    ``"none"``, as the EmbeddingGemma model card prescribes for untitled documents).
    """

    query: str = ""
    document: str = ""

    def query_text(self, text: str) -> str:
        """Prefix a search query."""
        return f"{self.query}{text}"

    def document_text(self, text: str, title: str | None = None) -> str:
        """Prefix a document, with its title when the prompt has a place for one."""
        return self.document.replace("{title}", title or "none") + text


EMBEDDINGGEMMA_PROMPTS = ModelPrompts(
    query="task: search result | query: ", document="title: {title} | text: "
)
E5_PROMPTS = ModelPrompts(query="query: ", document="passage: ")


def prompts_for(model_name: str) -> ModelPrompts:
    """Prompts of the model family of ``model_name``; unknown models get no prompts."""
    name = model_name.lower().rsplit("/", maxsplit=1)[-1]
    if name.startswith("embeddinggemma"):
        return EMBEDDINGGEMMA_PROMPTS
    if name.startswith(("e5-", "multilingual-e5-")):
        return E5_PROMPTS
    return ModelPrompts()


class DenseEmbedder(Embeddings):
    """LangChain embeddings over a local FastEmbed model, with the prompts the model expects.

    FastEmbed embeds raw text, while retrieval models such as EmbeddingGemma and E5 are
    trained with instruction prefixes, so queries and documents get the prefixes from
    :func:`prompts_for`. The ONNX model is loaded on first use, under a lock, so building the
    embedder is cheap and concurrent first requests load it once.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        prompts: ModelPrompts | None = None,
    ) -> None:
        self.model_name = model_name
        self.prompts = prompts if prompts is not None else prompts_for(model_name)
        self.cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed untitled documents."""
        return self._embed([self.prompts.document_text(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed([self.prompts.query_text(text)])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed chunks (header plus text), titled with their citation."""
        return self._embed(
            [
                self.prompts.document_text(chunk.embedding_text(), title=chunk.citation)
                for chunk in chunks
            ]
        )

    def fingerprint(self) -> dict[str, str]:
        """Model and prompts: the settings the vectors depend on."""
        return {
            "provider": "fastembed",
            "model": self.model_name,
            "query_prompt": self.prompts.query,
            "document_prompt": self.prompts.document,
        }

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._text_embedding().embed(texts, batch_size=self._batch_size)
        return [vector.tolist() for vector in vectors]

    def _text_embedding(self) -> TextEmbedding:
        with self._lock:
            if self._model is None:
                cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
                self._model = TextEmbedding(self.model_name, cache_dir=cache_dir)
            return self._model


def _is_transient(error: BaseException) -> bool:
    """Rate limits, server errors and network failures are worth retrying."""
    if isinstance(error, genai_errors.APIError):
        return error.code in _GEMINI_TRANSIENT_STATUS
    return isinstance(error, httpx.TransportError)


class GoogleDenseEmbedder(Embeddings):
    """Gemini API embeddings (``gemini-embedding-001`` by default).

    Queries and documents use the ``RETRIEVAL_QUERY`` and ``RETRIEVAL_DOCUMENT`` task types
    and a reduced output size. Gemini does not normalize vectors below 3072 dimensions, which
    is harmless because the collection uses cosine distance. Every request has a timeout, and
    rate limits, server errors and network failures are retried with exponential backoff.
    The API key defaults to the ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) environment variable.
    """

    def __init__(
        self,
        model: str = GEMINI_EMBEDDING_MODEL,
        *,
        dimensions: int = 768,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_attempts: int = 5,
        wait: WaitBaseT = _GEMINI_BACKOFF,
        client_args: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        http_options = genai_types.HttpOptions(
            timeout=round(timeout_s * 1000), client_args=client_args
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait,
            retry=retry_if_exception(_is_transient),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for retrieval."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed chunks (header plus text)."""
        return self.embed_documents([chunk.embedding_text() for chunk in chunks])

    def fingerprint(self) -> dict[str, str]:
        """Model and output size: the settings the vectors depend on."""
        return {"provider": "google", "model": self.model, "dimensions": str(self.dimensions)}

    def _embed(self, texts: Sequence[str], *, task_type: str) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimensions
        )
        vectors: list[list[float]] = []
        for batch in itertools.batched(texts, _GEMINI_MAX_BATCH):
            contents: list[genai_types.ContentUnion] = list(batch)
            response = self._retrying(
                self._client.models.embed_content,
                model=self.model,
                contents=contents,
                config=config,
            )
            vectors.extend(_vectors(response, expected=len(batch)))
        return vectors


def _vectors(response: genai_types.EmbedContentResponse, *, expected: int) -> list[list[float]]:
    vectors = [values for embedding in response.embeddings or [] if (values := embedding.values)]
    if len(vectors) != expected:
        msg = f"the embedding API returned {len(vectors)} vectors for {expected} texts"
        raise RuntimeError(msg)
    return vectors


def embedding_model_name(settings: Settings) -> str:
    """Name of the dense model that ``settings`` select.

    With the ``google`` provider, ``embedding_model`` is used when it names a Gemini model and
    :data:`GEMINI_EMBEDDING_MODEL` otherwise: the default setting names a FastEmbed model, so
    switching only the provider selects the default Gemini model.
    """
    configured = settings.embedding_model
    names_gemini = configured.removeprefix("models/").startswith("gemini-")
    if settings.embedding_provider == "google" and not names_gemini:
        return GEMINI_EMBEDDING_MODEL
    return configured


def build_dense_embedder(settings: Settings) -> Embeddings:
    """Dense embedder selected by ``settings.embedding_provider``."""
    model = embedding_model_name(settings)
    if settings.embedding_provider == "google":
        return GoogleDenseEmbedder(model, dimensions=settings.embedding_dim)
    return DenseEmbedder(model, cache_dir=settings.models_dir)
