"""Checks on the committed corpus in ``data/corpus``, the text the application ships with.

The expected counts were verified against the official pages fetched on 2026-10-30; a rebuild
from newer pages must update them deliberately.
"""

from collections import Counter

import pytest

from lexgraph.catalog import LAWS, SOURCES
from lexgraph.corpus import Corpus
from lexgraph.domain import LawId
from lexgraph.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_corpus
from lexgraph.ingestion.planalto import PARSER_VERSION
from lexgraph.ingestion.validate import validate_corpus
from tests.conftest import REPO_ROOT

CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# records, in force, revoked, vetoed
EXPECTED_COUNTS = {
    LawId.CLT: (1027, 841, 182, 4),
    LawId.CDC: (130, 118, 0, 12),
    LawId.LGPD: (80, 73, 1, 6),
    LawId.MCI: (32, 32, 0, 0),
    LawId.CF: (277, 274, 3, 0),
    LawId.ADCT: (148, 138, 10, 0),
}
# Chunks of the articles in force at the default size; changes with the corpus or the chunker.
EXPECTED_CHUNKS = 2545


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load(CORPUS_DIR)


def test_counts_per_law(corpus: Corpus) -> None:
    for law_id, expected in EXPECTED_COUNTS.items():
        statuses = Counter(article.status for article in corpus.by_law(law_id))
        counts = (statuses.total(), statuses["vigente"], statuses["revogado"], statuses["vetado"])
        law = corpus.manifest.laws[law_id]

        assert counts == expected, law_id
        assert (law.articles, law.in_force, law.revoked, law.vetoed) == expected, law_id


def test_every_law_passes_validation(corpus: Corpus) -> None:
    assert validate_corpus({law_id: corpus.by_law(law_id) for law_id in LawId}) == {}


def test_manifest_matches_the_catalog(corpus: Corpus) -> None:
    manifest = corpus.manifest

    assert manifest.parser_version == PARSER_VERSION
    assert list(manifest.laws) == list(LawId)
    for law_id, law in manifest.laws.items():
        assert law.source == LAWS[law_id].source
        assert law.source_url == SOURCES[law.source].url
        assert len(law.source_sha256) == 64


def test_in_force_articles_have_text_and_a_deep_link(corpus: Corpus) -> None:
    for article in corpus.in_force():
        assert article.text.strip(), article.article_id
        assert article.url.startswith("https://www.planalto.gov.br/ccivil_03/"), article.article_id


@pytest.mark.parametrize(
    ("article_id", "fragment"),
    [
        ("CLT:482", "a) ato de improbidade;"),
        ("CLT:51", "Art. 51 - Incorrerá em multa"),
        ("CDC:49", "O consumidor pode desistir do contrato, no prazo de 7 dias"),
        ("LGPD:18", "O titular dos dados pessoais tem direito a obter do controlador"),
        ("MCI:19", "Com o intuito de assegurar a liberdade de expressão"),
        ("CF:5", "LXXVIII - a todos, no âmbito judicial e administrativo"),
        ("CF:preambulo", "Nós, representantes do povo brasileiro"),
        ("ADCT:1", "Art. 1º. O Presidente da República"),
    ],
)
def test_landmark_articles(corpus: Corpus, article_id: str, fragment: str) -> None:
    article = corpus.get(article_id)

    assert article is not None
    assert article.status == "vigente"
    assert fragment in article.text


def test_revoked_article_with_its_paragraph(corpus: Corpus) -> None:
    article = corpus.get("CF:117")  # "Art. 117. e Parágrafo único. (Revogados ...)"

    assert article is not None
    assert (article.status, article.text) == ("revogado", "Art. 117.")


def test_notes_on_constitutional_review_are_kept(corpus: Corpus) -> None:
    # CLT art. 879, § 7º still reads "Taxa Referencial (TR)": the STF rulings that replaced
    # it survive only as notes.
    article = corpus.get("CLT:879")

    assert article is not None
    assert {"Vide ADC 58", "Vide ADC 59", "Vide ADI 6021"} <= set(article.annotations)


def test_chunks_cover_the_corpus_without_losing_text(corpus: Corpus) -> None:
    chunks = chunk_corpus(corpus)
    in_force = list(corpus.in_force())
    lines_by_article: dict[str, set[str]] = {}
    for chunk in chunks:
        lines_by_article.setdefault(chunk.article_id, set()).update(chunk.text.splitlines())

    assert len(chunks) == EXPECTED_CHUNKS
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len(in_force) <= len(chunks) <= 2 * len(in_force)
    oversized = [chunk for chunk in chunks if len(chunk.text) > DEFAULT_MAX_CHARS]
    assert len(oversized) <= 0.005 * len(chunks)
    assert all(len(chunk.text) <= 2 * DEFAULT_MAX_CHARS for chunk in chunks)
    for article in in_force:
        missing = set(article.text.splitlines()) - lines_by_article[article.article_id]
        assert not missing, article.article_id


def test_caput_chunks_cite_the_incisos_they_hold(corpus: Corpus) -> None:
    chunks = {chunk.chunk_id: chunk for chunk in chunk_corpus(corpus)}

    assert chunks["CF:8#0"].citation == "CF, art. 8º, caput e I a VIII"
    assert chunks["CF:8#1"].citation == "CF, art. 8º, parágrafo único"


def test_no_unit_is_cited_twice_in_an_article(corpus: Corpus) -> None:
    units_by_article: dict[str, list[str]] = {}
    for chunk in chunk_corpus(corpus):
        units_by_article.setdefault(chunk.article_id, []).extend(chunk.units)
    repeated = {
        article_id: units
        for article_id, units in units_by_article.items()
        if len(set(units)) < len(units)
    }

    assert repeated == {}
    # CF art. 109 prints "V-A as causas ..." without the dash of its label.
    assert "V-A" in units_by_article["CF:109"]
