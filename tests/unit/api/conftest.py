"""Fixtures for the HTTP API tests: a scripted chat service and a small corpus on disk."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lexgraph.api.app import create_app
from lexgraph.config import Settings
from lexgraph.corpus import Corpus, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import Article, LawId
from tests.conftest import make_article
from tests.unit.api.fakes import FakeChatService

ApiFactory = Callable[..., TestClient]

BASE_URL = "http://localhost"
"""Base URL of the test clients: its host is one of the default ``allowed_hosts``."""


def write_sample_corpus(corpus_dir: Path, articles: list[Article]) -> None:
    """Write ``articles`` (plus a revoked one) as a corpus with a consistent manifest."""
    revoked = make_article(LawId.CLT, "59-A", "Art. 59-A. (Revogado).", seq=5, status="revogado")
    by_law: dict[LawId, list[Article]] = {}
    for article in [*articles, revoked]:
        by_law.setdefault(article.law_id, []).append(article)
    manifest = CorpusManifest(
        generated_at="2026-10-30T12:00:00+00:00",
        parser_version="test",
        laws={
            law_id: LawManifest(
                law_id=law_id,
                source=law_id.value.lower(),
                source_url=f"https://www.planalto.gov.br/{law_id.value.lower()}.htm",
                source_sha256="0" * 64,
                fetched_at="2026-10-30T11:58:00+00:00",
                articles=len(items),
                in_force=sum(item.status == "vigente" for item in items),
                revoked=sum(item.status == "revogado" for item in items),
                vetoed=sum(item.status == "vetado" for item in items),
            )
            for law_id, items in by_law.items()
        },
    )
    write_corpus(corpus_dir, by_law, manifest)


@pytest.fixture
def corpus(tmp_path: Path, sample_articles: list[Article]) -> Corpus:
    """The sample articles loaded back from a corpus written to disk."""
    corpus_dir = tmp_path / "corpus"
    write_sample_corpus(corpus_dir, sample_articles)
    return Corpus.load(corpus_dir)


@pytest.fixture
def service() -> FakeChatService:
    return FakeChatService()


@pytest.fixture
def make_api(settings: Settings, service: FakeChatService, corpus: Corpus) -> Iterator[ApiFactory]:
    """Build a started test client; keyword arguments override settings fields."""
    with ExitStack() as stack:

        def factory(**overrides: Any) -> TestClient:
            app = create_app(settings.model_copy(update=overrides), service=service, corpus=corpus)
            client = TestClient(app, base_url=BASE_URL, raise_server_exceptions=False)
            return stack.enter_context(client)

        yield factory


@pytest.fixture
def api(make_api: ApiFactory) -> TestClient:
    """Test client of an API with default settings, the fake service and the sample corpus."""
    return make_api()
