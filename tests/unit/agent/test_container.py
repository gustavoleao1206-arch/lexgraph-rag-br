"""Composition root: corpus date and the lifecycle of the service resources."""

import logging
from pathlib import Path

import pytest

from lexgraph.agent import prompts
from lexgraph.config import Settings
from lexgraph.container import open_service, read_corpus_date
from lexgraph.corpus import MANIFEST_FILE, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import LawId
from lexgraph.schemas import ChatRequest, ErrorEvent
from lexgraph.service import LexGraphService
from tests.unit.agent.fakes import FakeRetriever

GENERATED_AT = "2026-10-30T12:00:00+00:00"


def law_manifest(law_id: LawId, fetched_at: str) -> LawManifest:
    return LawManifest(
        law_id=law_id,
        source=law_id.value.lower(),
        source_url="https://www.planalto.gov.br/exemplo.htm",
        source_sha256="0" * 64,
        fetched_at=fetched_at,
        articles=1,
        in_force=1,
        revoked=0,
        vetoed=0,
    )


def write_manifest(corpus_dir: Path, *fetches: tuple[LawId, str]) -> None:
    laws = {law_id: law_manifest(law_id, fetched_at) for law_id, fetched_at in fetches}
    manifest = CorpusManifest(generated_at=GENERATED_AT, parser_version="test", laws=laws)
    write_corpus(corpus_dir, {}, manifest)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_corpus_date_is_the_oldest_fetch(tmp_path: Path) -> None:
    write_manifest(
        tmp_path, (LawId.CLT, "2026-10-30T11:52:03+00:00"), (LawId.CDC, "2026-10-29T08:00:00Z")
    )

    assert read_corpus_date(tmp_path) == "29/10/2026"


def test_corpus_date_falls_back_to_the_generation_time(tmp_path: Path) -> None:
    write_manifest(tmp_path)

    assert read_corpus_date(tmp_path) == "30/10/2026"


def test_corpus_date_is_unknown_without_a_manifest(tmp_path: Path) -> None:
    assert read_corpus_date(tmp_path) is None


def test_corpus_date_is_unknown_when_the_manifest_is_unreadable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / MANIFEST_FILE).write_text("{ não é JSON", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="lexgraph.container"):
        assert read_corpus_date(tmp_path) is None

    assert "could not read the corpus date" in caplog.text


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch, provider_keys: None) -> FakeQdrantClient:
    """Replace the Qdrant client and the retriever with in-memory fakes."""
    store = FakeQdrantClient()

    def build_retriever(settings: Settings, *, client: object = None) -> FakeRetriever:
        assert client is store, "the retriever must share the client opened by the service"
        return FakeRetriever()

    monkeypatch.setattr("lexgraph.retrieval.store.create_qdrant_client", lambda settings: store)
    monkeypatch.setattr("lexgraph.retrieval.factory.build_retriever", build_retriever)
    return store


async def test_open_service_wires_a_persistent_service(
    settings: Settings, fake_store: FakeQdrantClient
) -> None:
    async with open_service(settings) as service:
        assert isinstance(service, LexGraphService)
        assert await service.history("nenhuma") == []
        info = await service.info()
        closed_while_open = fake_store.closed

    assert info.indexed_chunks == 0
    assert info.models_configured is True
    assert closed_while_open is False
    assert fake_store.closed
    assert settings.checkpoint_path.is_file()


async def test_open_service_without_keys_starts_and_explains_how_to_configure_them(
    settings: Settings,
    fake_store: FakeQdrantClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with caplog.at_level(logging.WARNING, logger="lexgraph.container"):
        async with open_service(settings) as service:
            info = await service.info()
            events = [event async for event in service.stream(ChatRequest(question="Oi?"))]

    assert info.models_configured is False
    assert events == [ErrorEvent(message=prompts.MODELS_NOT_CONFIGURED)]
    assert "has no provider API key" in caplog.text


async def test_open_service_closes_the_client_when_setup_fails(
    settings: Settings, fake_store: FakeQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(settings: Settings, *, client: object = None) -> FakeRetriever:
        raise RuntimeError("coleção inexistente")

    monkeypatch.setattr("lexgraph.retrieval.factory.build_retriever", broken)

    with pytest.raises(RuntimeError, match="coleção inexistente"):
        async with open_service(settings):
            pass

    assert fake_store.closed
