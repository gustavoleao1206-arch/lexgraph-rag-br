"""Sparse BM25 vectors for Brazilian Portuguese legal text.

Vectors are computed on the client with FastEmbed's ``Qdrant/bm25`` model (Snowball stemming,
the bundled Portuguese stopword list and BM25 term-frequency weights). The collection stores
them with ``Modifier.IDF``, so Qdrant applies the inverse document frequency at query time.
"""

import re
import threading
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from py_rust_stemmers import SnowballStemmer  # type: ignore[import-untyped]
from qdrant_client import models

from lexgraph.config import Settings

LANGUAGE = "portuguese"

# Mean number of indexed tokens per chunk (header plus text, stopwords removed) measured on
# the corpus. FastEmbed's default of 256 assumes much longer documents and would weaken the
# length normalization of BM25. Query vectors do not depend on it: changing it only requires
# re-indexing.
DEFAULT_AVG_LEN = 80.0

# Version of the text analysis: normalize(), the stopword list and AccentInsensitiveStemmer.
# Bump it when any of them changes, so that indexes built with the old one are rebuilt.
ANALYSIS_VERSION = "1"

_ORDINAL_SIGN = re.compile(r"(?<=\d)\.?[ºª°]")

# Unaccented spellings of the nasal endings whose accents the Snowball suffix rules rely on,
# most specific first so that "indenizacoes" becomes "indenizações", not "indenizacões".
_NASAL_ENDINGS = (("coes", "ções"), ("cao", "ção"), ("oes", "ões"), ("aes", "ães"), ("ao", "ão"))


@runtime_checkable
class SparseEncoder(Protocol):
    """Turns documents and queries into sparse vectors for the ``bm25`` named vector."""

    def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Return one document vector per text."""
        ...

    def encode_query(self, text: str) -> models.SparseVector:
        """Return the query vector (empty when every word is a stopword)."""
        ...


def normalize(text: str) -> str:
    """Prepare text for BM25; applied to documents and queries alike.

    Composes Unicode (NFC), so accented letters are single code points, and removes ordinal
    signs after digits (``7º`` -> ``7``, ``1ª`` -> ``1``, ``2°`` -> ``2``, ``1.º`` -> ``1``), so
    a question about "artigo 7" matches "Art. 7º".
    """
    return _ORDINAL_SIGN.sub("", unicodedata.normalize("NFC", text))


def fold_accents(text: str) -> str:
    """Strip diacritics: ``"Indenização"`` -> ``"Indenizacao"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def restore_nasal_ending(word: str) -> str:
    """Put back the tilde and cedilla of a nasal ending typed without them.

    ``"indenizacao"`` -> ``"indenização"``, ``"informacoes"`` -> ``"informações"``,
    ``"maes"`` -> ``"mães"``. Words with other endings are returned unchanged.
    """
    for plain, accented in _NASAL_ENDINGS:
        if word.endswith(plain):
            return word[: -len(plain)] + accented
    return word


def read_stopwords(path: Path) -> set[str]:
    """Read a stopword file as UTF-8 and add the unaccented spelling of each word."""
    words = path.read_text(encoding="utf-8").split()
    return {*words, *(fold_accents(word) for word in words)}


class AccentInsensitiveStemmer:
    """Snowball Portuguese stemmer whose stems do not depend on accents.

    The word is stemmed with its nasal ending restored (the suffix rules need the accents)
    and the stem is then folded, so "indenizacao", "indenização" and "indenizações" share the
    stem "indeniz".
    """

    def __init__(self) -> None:
        self._snowball = SnowballStemmer(LANGUAGE)

    def stem_word(self, word: str) -> str:
        """Return the accent-free stem of a lowercase word."""
        return fold_accents(self._snowball.stem_word(restore_nasal_ending(word)))


class Bm25Encoder:
    """BM25 vectors for Portuguese with accent-insensitive stems; the model loads on first use.

    Accent folding is decided from measurements on the corpus, where accented words are 29% of
    the indexed tokens. Official texts are always accented, users often are not, and plain
    BM25 treats "ferias" and "férias" as different terms: an unaccented query word reaches its
    accented corpus form only 10% of the time. Stemming and then folding reaches 76%, missing
    the "-ção/-ções" family, whose Snowball suffix rules need the accent; folding before
    stemming reaches 100% but breaks 15% of the Snowball conflations ("indenização" no longer
    matches "indenizações" or "indenizar"). Restoring common nasal endings before stemming
    and folding afterwards (:class:`AccentInsensitiveStemmer`) reaches 92% and keeps every
    conflation, so that is the strategy used.

    FastEmbed reads the bundled stopword file with the platform encoding, which garbles the
    accented Portuguese stopwords on Windows; the list is read as UTF-8 and passed explicitly,
    together with its unaccented spellings, so vectors are the same on every platform.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        *,
        cache_dir: Path | None = None,
        avg_len: float = DEFAULT_AVG_LEN,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._avg_len = avg_len
        self._model: Bm25 | None = None
        self._lock = threading.Lock()

    def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Return one BM25 document vector (term-frequency weights) per text."""
        embeddings = self._bm25().embed([normalize(text) for text in texts])
        return [_to_sparse_vector(embedding) for embedding in embeddings]

    def encode_query(self, text: str) -> models.SparseVector:
        """Return the query vector: weight 1 for each distinct stem, empty for stopwords only."""
        embedding = next(iter(self._bm25().query_embed(normalize(text))))
        return _to_sparse_vector(embedding)

    def fingerprint(self) -> dict[str, str]:
        """Model, text analysis and length normalization: what the vectors depend on."""
        return {
            "model": self.model_name,
            "language": LANGUAGE,
            "analysis": ANALYSIS_VERSION,
            "avg_len": str(self._avg_len),
        }

    def _bm25(self) -> Bm25:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def _load(self) -> Bm25:
        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        # A first instance only locates (or downloads) the model files with the stopword list.
        bundled = Bm25(self.model_name, cache_dir=cache_dir, language=LANGUAGE)
        stopwords = read_stopwords(bundled._model_dir / f"{LANGUAGE}.txt")
        return Bm25(
            self.model_name,
            cache_dir=cache_dir,
            language=LANGUAGE,
            avg_len=self._avg_len,
            stemmer=AccentInsensitiveStemmer(),
            stopwords=stopwords,
        )


def build_sparse_encoder(settings: Settings) -> Bm25Encoder:
    """BM25 encoder for ``settings.sparse_model``, cached under ``settings.models_dir``."""
    return Bm25Encoder(settings.sparse_model, cache_dir=settings.models_dir)


def _to_sparse_vector(embedding: SparseEmbedding) -> models.SparseVector:
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
