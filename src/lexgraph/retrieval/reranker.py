"""Cross-encoder reranking of retrieved passages."""

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from fastembed.rerank.cross_encoder import TextCrossEncoder

from lexgraph.domain import Passage

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Re-orders passages by their relevance to a query."""

    def rerank(self, query: str, passages: Sequence[Passage], top_n: int) -> list[Passage]:
        """Return the ``top_n`` most relevant passages, best first, with ``rerank_score`` set."""
        ...

    def is_available(self) -> bool:
        """Whether the model can be used without downloading anything."""
        ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a FastEmbed cross-encoder; the model loads on first use.

    A cross-encoder reads the query and the passage together, which ranks better than
    comparing separate embeddings but costs one model run per passage (about 0.3 s for the
    default multilingual model on a laptop CPU), so only a few fused candidates are reranked.
    """

    def __init__(
        self, model_name: str, *, cache_dir: Path | None = None, batch_size: int = 16
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: TextCrossEncoder | None = None
        self._installed: bool | None = None  # model files found in the local cache
        self._lock = threading.Lock()

    def rerank(self, query: str, passages: Sequence[Passage], top_n: int) -> list[Passage]:
        """Return the ``top_n`` passages with the highest cross-encoder scores, best first.

        Passages keep their search ``score``; the raw cross-encoder logit goes to
        ``rerank_score``.
        """
        if not passages or top_n < 1:
            return []
        texts = [passage.embedding_text() for passage in passages]
        scores = self._cross_encoder().rerank(query, texts, batch_size=self._batch_size)
        ranked = sorted(zip(scores, passages, strict=True), key=lambda pair: pair[0], reverse=True)
        return [
            passage.model_copy(update={"rerank_score": float(score)})
            for score, passage in ranked[:top_n]
        ]

    def is_available(self) -> bool:
        """Whether the model is loaded or its files are in the local cache; never downloads.

        The reranker is optional in the Docker image. The cache is looked up once, since
        FastEmbed logs an error each time the files are missing.
        """
        with self._lock:
            if self._model is not None:
                return True
            if self._installed is None:
                self._installed = self._in_local_cache()
                if not self._installed:
                    logger.info("reranker model %s is not installed", self.model_name)
            return self._installed

    def _in_local_cache(self) -> bool:
        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        try:
            TextCrossEncoder(
                self.model_name, cache_dir=cache_dir, lazy_load=True, local_files_only=True
            )
        except ValueError:  # FastEmbed's error for a model it cannot load
            return False
        return True

    def _cross_encoder(self) -> TextCrossEncoder:
        with self._lock:
            if self._model is None:
                cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
                self._model = TextCrossEncoder(self.model_name, cache_dir=cache_dir)
            return self._model
