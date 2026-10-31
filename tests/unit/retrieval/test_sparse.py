"""BM25 text normalization, accent-insensitive stemming and the encoder wiring (no download)."""

import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import models

from lexgraph.config import Settings
from lexgraph.retrieval import sparse as sparse_module
from lexgraph.retrieval.sparse import (
    ANALYSIS_VERSION,
    DEFAULT_AVG_LEN,
    AccentInsensitiveStemmer,
    Bm25Encoder,
    SparseEncoder,
    build_sparse_encoder,
    fold_accents,
    normalize,
    read_stopwords,
    restore_nasal_ending,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Art. 7º São direitos", "Art. 7 São direitos"),
        ("a 1ª Turma", "a 1 Turma"),
        ("§ 2° Os anúncios", "§ 2 Os anúncios"),
        ("art. 1.º desta Lei", "art. 1 desta Lei"),
        ("arts. 5º, 6º e 7º", "arts. 5, 6 e 7"),
        ("Lei nº 8.078", "Lei nº 8.078"),
        ("sem ordinais", "sem ordinais"),
    ],
)
def test_normalize_drops_ordinal_signs_after_digits(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_normalize_composes_decomposed_accents() -> None:
    decomposed = unicodedata.normalize("NFD", "férias e rescisão")

    assert normalize(decomposed) == "férias e rescisão"


def test_normalize_is_idempotent() -> None:
    text = "Art. 7º, § 1º: férias"

    assert normalize(normalize(text)) == normalize(text)


def test_fold_accents_strips_every_diacritic() -> None:
    assert fold_accents("Indenização às AÇÕES, órgãos públicos") == (
        "Indenizacao as ACOES, orgaos publicos"
    )


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("indenizacao", "indenização"),
        ("indenizacoes", "indenizações"),
        ("informacoes", "informações"),
        ("rescisao", "rescisão"),
        ("orgao", "orgão"),
        ("maes", "mães"),
        ("indenização", "indenização"),
        ("trabalho", "trabalho"),
        ("ferias", "ferias"),
    ],
)
def test_restore_nasal_ending(word: str, expected: str) -> None:
    assert restore_nasal_ending(word) == expected


@pytest.mark.parametrize(
    "spellings",
    [
        ("férias", "ferias"),
        ("rescisão", "rescisao"),
        ("salário", "salario"),
        ("indenização", "indenizacao", "indenizações", "indenizacoes"),
        ("informações", "informacoes", "informação"),
        ("mães", "maes"),
        ("trabalhador", "trabalhadores", "trabalho"),
    ],
)
def test_stemmer_gives_accented_and_plain_spellings_the_same_stem(
    spellings: tuple[str, ...],
) -> None:
    stemmer = AccentInsensitiveStemmer()

    stems = {stemmer.stem_word(word) for word in spellings}

    assert len(stems) == 1
    assert stems.pop().isascii()


def test_stemmer_keeps_distinct_words_apart() -> None:
    stemmer = AccentInsensitiveStemmer()

    assert stemmer.stem_word("férias") != stemmer.stem_word("salário")


def test_read_stopwords_decodes_utf8_and_adds_plain_spellings(tmp_path: Path) -> None:
    path = tmp_path / "portuguese.txt"
    path.write_bytes("não\né\naté\nde\n".encode())

    assert read_stopwords(path) == {"não", "nao", "é", "e", "até", "ate", "de"}


class FakeBm25:
    """Stands in for FastEmbed's ``Bm25``: records its options and the texts it encodes."""

    def __init__(
        self, model_name: str, *, model_dir: Path, cache_dir: str | None, **options: Any
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.options = options
        self._model_dir = model_dir
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed(self, documents: list[str]) -> list[SparseEmbedding]:
        self.documents.extend(documents)
        return [
            SparseEmbedding(values=np.array([0.5, 1.5]), indices=np.array([3, 1]))
            for _ in documents
        ]

    def query_embed(self, query: str) -> Iterator[SparseEmbedding]:
        self.queries.append(query)
        yield SparseEmbedding(values=np.array([1.0]), indices=np.array([7]))


@pytest.fixture
def bm25_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[FakeBm25]:
    """Replace ``Bm25`` with :class:`FakeBm25`; returns the instances created."""
    (tmp_path / "portuguese.txt").write_bytes("não\né\nde\n".encode())
    created: list[FakeBm25] = []

    def factory(model_name: str, *, cache_dir: str | None = None, **options: Any) -> FakeBm25:
        model = FakeBm25(model_name, model_dir=tmp_path, cache_dir=cache_dir, **options)
        created.append(model)
        return model

    monkeypatch.setattr(sparse_module, "Bm25", factory)
    return created


def test_encoder_loads_the_model_on_first_use_only(
    bm25_models: list[FakeBm25], tmp_path: Path
) -> None:
    encoder = Bm25Encoder(cache_dir=tmp_path)
    assert bm25_models == []

    encoder.encode_query("férias")
    loaded = len(bm25_models)
    encoder.encode_documents(["férias"])

    assert loaded > 0
    assert len(bm25_models) == loaded


def test_encoder_passes_utf8_stopwords_and_the_folding_stemmer(
    bm25_models: list[FakeBm25], tmp_path: Path
) -> None:
    Bm25Encoder("Qdrant/bm25", cache_dir=tmp_path).encode_query("férias")
    model = bm25_models[-1]

    assert model.model_name == "Qdrant/bm25"
    assert model.cache_dir == str(tmp_path)
    assert model.options["language"] == "portuguese"
    assert model.options["stopwords"] == {"não", "nao", "é", "e", "de"}
    assert isinstance(model.options["stemmer"], AccentInsensitiveStemmer)
    assert model.options["avg_len"] == DEFAULT_AVG_LEN


def test_encoder_normalizes_documents_and_converts_vectors(bm25_models: list[FakeBm25]) -> None:
    vectors = Bm25Encoder().encode_documents(["Art. 7º Férias", "§ 2° Prazo"])

    assert bm25_models[-1].documents == ["Art. 7 Férias", "§ 2 Prazo"]
    assert vectors == [models.SparseVector(indices=[3, 1], values=[0.5, 1.5])] * 2


def test_encoder_normalizes_queries_and_converts_vectors(bm25_models: list[FakeBm25]) -> None:
    vector = Bm25Encoder().encode_query("artigo 5º")

    assert bm25_models[-1].queries == ["artigo 5"]
    assert vector == models.SparseVector(indices=[7], values=[1.0])


def test_encoder_satisfies_the_sparse_encoder_protocol() -> None:
    assert isinstance(Bm25Encoder(), SparseEncoder)


def test_encoder_reports_what_its_vectors_depend_on() -> None:
    assert Bm25Encoder("Qdrant/bm25", avg_len=60.0).fingerprint() == {
        "model": "Qdrant/bm25",
        "language": "portuguese",
        "analysis": ANALYSIS_VERSION,
        "avg_len": "60.0",
    }


def test_build_sparse_encoder_uses_the_settings(settings: Settings) -> None:
    encoder = build_sparse_encoder(settings)

    assert encoder.model_name == settings.sparse_model
    assert encoder.cache_dir == settings.models_dir
