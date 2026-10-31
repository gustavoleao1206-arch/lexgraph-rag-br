"""``ingest`` and ``search`` commands, with fake encoders and a stubbed chunker."""

import sys
import types
from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from lexgraph.config import Settings
from lexgraph.corpus import Corpus, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import Chunk, LawId
from lexgraph.retrieval import cli, embeddings, factory, sparse, store
from lexgraph.retrieval.retriever import LegalRetriever
from lexgraph.retrieval.store import StorageLockedError, resolve_alias
from tests.conftest import make_article, make_chunk
from tests.unit.retrieval.conftest import DENSE_SIZE, HashingSparseEncoder

runner = CliRunner()

PRODUCTION_BUILD_RETRIEVER = factory.build_retriever

ARTICLES = [
    make_article(),
    make_article(
        LawId.CDC,
        "18",
        "Art. 18. Os fornecedores respondem solidariamente pelos vícios de qualidade.",
        seq=1,
    ),
    make_article(
        LawId.LGPD,
        "18",
        "Art. 18. O titular dos dados pessoais tem direito a obter do controlador o acesso.",
        seq=0,
    ),
    make_article(LawId.LGPD, "19", "Art. 19. (revogado)", seq=1, status="revogado"),
]


def write_test_corpus(settings: Settings) -> None:
    manifest = CorpusManifest(
        generated_at="2026-10-30T12:00:00Z",
        parser_version="test",
        laws={
            law_id: LawManifest(
                law_id=law_id,
                source=law_id.value.lower(),
                source_url="https://www.planalto.gov.br/example.htm",
                source_sha256="0" * 64,
                fetched_at="2026-10-30T12:00:00Z",
                articles=2,
                in_force=sum(1 for a in ARTICLES if a.law_id == law_id and a.status == "vigente"),
                revoked=sum(1 for a in ARTICLES if a.law_id == law_id and a.status == "revogado"),
                vetoed=0,
            )
            for law_id in (LawId.CDC, LawId.LGPD)
        },
    )
    by_law = {
        law_id: [article for article in ARTICLES if article.law_id == law_id]
        for law_id in manifest.laws
    }
    write_corpus(settings.corpus_dir, by_law, manifest)


@pytest.fixture
def chunker_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub ``lexgraph.ingestion.chunking`` with one whole-article chunk per article in force."""
    calls: list[dict[str, Any]] = []

    def chunk_corpus(
        corpus: Corpus,
        *,
        law_ids: Iterable[LawId] | None = None,
        max_chars: int = 1500,
        pack_chars: int = 700,
    ) -> list[Chunk]:
        selected = list(law_ids) if law_ids is not None else None
        calls.append({"law_ids": selected, "max_chars": max_chars, "pack_chars": pack_chars})
        return [make_chunk(article) for article in corpus.in_force(selected)]

    module = types.ModuleType("lexgraph.ingestion.chunking")
    module.chunk_corpus = chunk_corpus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lexgraph.ingestion.chunking", module)
    return calls


@pytest.fixture
def cli_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, chunker_calls: list[dict[str, Any]]
) -> Settings:
    """Run the commands against ``settings`` with fake encoders and no global side effects."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "bootstrap_env", lambda: None)
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(
        embeddings, "build_dense_embedder", lambda _: DeterministicFakeEmbedding(size=DENSE_SIZE)
    )
    monkeypatch.setattr(sparse, "build_sparse_encoder", lambda _: HashingSparseEncoder())

    def build_retriever(config: Settings, *, client: QdrantClient | None = None) -> LegalRetriever:
        assert client is not None
        return LegalRetriever(
            client=client,
            collection=config.collection,
            dense=DeterministicFakeEmbedding(size=DENSE_SIZE),
            sparse=HashingSparseEncoder(),
            mode=config.retrieval_mode,
            k=config.top_k,
        )

    monkeypatch.setattr(factory, "build_retriever", build_retriever)
    return settings


def indexed_chunks(settings: Settings) -> int:
    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        assert resolve_alias(client, settings.collection) is not None
        return client.count(settings.collection, exact=True).count
    finally:
        client.close()


def test_ingest_indexes_the_articles_in_force(
    cli_settings: Settings, chunker_calls: list[dict[str, Any]]
) -> None:
    write_test_corpus(cli_settings)

    result = runner.invoke(cli.app, ["ingest", "--batch-size", "2"])

    assert result.exit_code == 0, result.output
    assert "Indexed 3 chunks" in result.output
    assert chunker_calls == [
        {
            "law_ids": None,
            "max_chars": cli_settings.chunk_max_chars,
            "pack_chars": cli_settings.chunk_pack_chars,
        }
    ]
    assert indexed_chunks(cli_settings) == 3


def test_ingest_can_select_laws(
    cli_settings: Settings, chunker_calls: list[dict[str, Any]]
) -> None:
    write_test_corpus(cli_settings)

    result = runner.invoke(cli.app, ["ingest", "--law", "lgpd"])

    assert result.exit_code == 0, result.output
    assert chunker_calls[0]["law_ids"] == [LawId.LGPD]
    assert indexed_chunks(cli_settings) == 1


def test_reingest_reports_the_replaced_collection(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert "Removed previous collections" in result.output
    assert indexed_chunks(cli_settings) == 3


def test_ingest_if_missing_keeps_a_current_index(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["ingest", "--if-missing"])

    assert result.exit_code == 0, result.output
    assert "The index lexgraph is up to date (3 chunks); nothing to do." in result.output
    assert "Indexed" not in result.output
    assert indexed_chunks(cli_settings) == 3


def test_ingest_if_missing_builds_a_missing_index(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)

    result = runner.invoke(cli.app, ["ingest", "--if-missing"])

    assert result.exit_code == 0, result.output
    assert "Indexed 3 chunks" in result.output


def test_ingest_if_missing_rebuilds_an_index_of_other_chunks(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest", "--law", "LGPD"])

    result = runner.invoke(cli.app, ["ingest", "--if-missing"])

    assert result.exit_code == 0, result.output
    assert "Indexed 3 chunks" in result.output
    assert indexed_chunks(cli_settings) == 3


def test_ingest_without_corpus_fails(cli_settings: Settings) -> None:
    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 1
    assert "lexgraph build-corpus" in result.output


def test_ingest_without_articles_in_force_fails(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)

    result = runner.invoke(cli.app, ["ingest", "--law", "MCI"])

    assert result.exit_code == 1
    assert "no articles in force" in result.output


def test_ingest_reports_a_locked_store(
    cli_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_corpus(cli_settings)

    def locked(config: Settings) -> QdrantClient:
        msg = f"embedded Qdrant at {config.qdrant_path} is in use by another client or process"
        raise StorageLockedError(msg)

    monkeypatch.setattr(store, "create_qdrant_client", locked)

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 1
    assert "in use by another client or process" in result.output
    assert "LEXGRAPH_QDRANT_URL" in result.output


def test_search_prints_the_passages(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(
        cli.app, ["search", "Art. consumidor desistir", "--mode", "sparse", "-k", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "2 passages (sparse search)" in result.output
    assert "CDC, art. 49" in result.output


def test_search_title_counts_a_single_passage(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["search", "desistir", "--mode", "sparse"])

    assert result.exit_code == 0, result.output
    assert "1 passage (sparse search)" in result.output


def test_search_filters_by_law(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["search", "Art.", "--law", "LGPD", "--mode", "sparse"])

    assert result.exit_code == 0, result.output
    assert "LGPD, art. 18" in result.output
    assert "CDC, art." not in result.output


def test_search_uses_the_configured_mode_by_default(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["search", "consumidor"])

    assert result.exit_code == 0, result.output
    assert f"({cli_settings.retrieval_mode} search)" in result.output


def test_search_without_matches(cli_settings: Settings) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])

    result = runner.invoke(cli.app, ["search", "inexistente", "--mode", "sparse"])

    assert result.exit_code == 0, result.output
    assert "No passages found." in result.output


def test_search_with_an_index_of_other_models_fails(
    cli_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_corpus(cli_settings)
    runner.invoke(cli.app, ["ingest"])  # indexed with the fake encoders
    monkeypatch.setattr(factory, "build_retriever", PRODUCTION_BUILD_RETRIEVER)

    result = runner.invoke(cli.app, ["search", "férias"])

    assert result.exit_code == 1
    assert "was built with other encoders" in result.output
    assert "re-run `lexgraph ingest`." in result.output


def test_search_without_index_fails(cli_settings: Settings) -> None:
    result = runner.invoke(cli.app, ["search", "férias"])

    assert result.exit_code == 1
    assert "run `lexgraph ingest` first" in result.output
