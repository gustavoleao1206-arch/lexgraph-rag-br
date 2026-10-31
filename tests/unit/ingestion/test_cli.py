"""The ``fetch`` and ``build-corpus`` commands, run through Typer's test runner."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from lexgraph.catalog import SOURCES, SourcePage
from lexgraph.config import Settings
from lexgraph.corpus import MANIFEST_FILE, Corpus, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import LawId
from lexgraph.ingestion import cli
from lexgraph.ingestion.fetch import FetchMeta, FetchResult, html_path, meta_path
from lexgraph.ingestion.planalto import PARSER_VERSION
from tests.conftest import make_article

runner = CliRunner()


def marco_civil_page(last: int = 32) -> bytes:
    """A page shaped like the official one, with articles 1..last."""
    articles = "".join(
        f'<p><a name="art{n}"></a>Art. {n}{"º" if n < 10 else "."} Texto do artigo {n}.</p>'
        for n in range(1, last + 1)
    )
    return (
        "<html><body><p>Faço saber que o Congresso Nacional decreta e eu sanciono a seguinte "
        f"Lei:</p><p>CAPÍTULO I</p><p>DISPOSIÇÕES PRELIMINARES</p>{articles}"
        "<p>Brasília, 23 de abril de 2014; 193º da Independência.</p></body></html>"
    ).encode("cp1252")


def store_page(raw_dir: Path, html: bytes) -> FetchMeta:
    source = SOURCES["marco-civil"]
    meta = FetchMeta(
        url=source.url,
        final_url=source.url,
        status=200,
        sha256=hashlib.sha256(html).hexdigest(),
        etag='"1ce91-650291064ef51"',
        last_modified="Thu, 23 Apr 2026 23:32:39 GMT",
        fetched_at="2026-10-30T14:29:41+00:00",
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path(raw_dir, source.slug).write_bytes(html)
    meta_path(raw_dir, source.slug).write_text(meta.model_dump_json(), encoding="utf-8")
    return meta


def write_existing_corpus(corpus_dir: Path, parser_version: str = PARSER_VERSION) -> None:
    law = LawManifest(
        law_id=LawId.CDC,
        source="cdc",
        source_url=SOURCES["cdc"].url,
        source_sha256="f" * 64,
        fetched_at="2026-10-30T14:29:35+00:00",
        articles=1,
        in_force=1,
        revoked=0,
        vetoed=0,
    )
    manifest = CorpusManifest(
        generated_at="2026-10-30T15:00:00+00:00",
        parser_version=parser_version,
        laws={LawId.CDC: law},
    )
    write_corpus(corpus_dir, {LawId.CDC: [make_article()]}, manifest)


@pytest.fixture
def app_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def test_build_corpus_writes_the_articles_and_the_manifest(app_settings: Settings) -> None:
    meta = store_page(app_settings.raw_dir, marco_civil_page())

    result = runner.invoke(cli.app, ["build-corpus", "--source", "marco-civil"])

    assert result.exit_code == 0, result.output
    assert "Wrote 32 articles" in result.output
    corpus = Corpus.load(app_settings.corpus_dir)
    assert len(corpus.by_law(LawId.MCI)) == 32
    assert corpus.get("MCI:19") is not None
    law = corpus.manifest.laws[LawId.MCI]
    assert (law.source, law.source_sha256, law.fetched_at) == (
        "marco-civil",
        meta.sha256,
        meta.fetched_at,
    )
    assert (law.articles, law.in_force, law.revoked, law.vetoed) == (32, 32, 0, 0)
    assert corpus.manifest.parser_version == PARSER_VERSION


def test_build_corpus_refuses_to_write_when_validation_fails(app_settings: Settings) -> None:
    store_page(app_settings.raw_dir, marco_civil_page(last=31))

    result = runner.invoke(cli.app, ["build-corpus", "-s", "marco-civil"])

    assert result.exit_code == 1
    assert "missing articles: 32" in result.output
    assert not (app_settings.corpus_dir / MANIFEST_FILE).exists()


def test_build_corpus_can_write_despite_problems(app_settings: Settings) -> None:
    store_page(app_settings.raw_dir, marco_civil_page(last=31))

    result = runner.invoke(cli.app, ["build-corpus", "-s", "marco-civil", "--allow-problems"])

    assert result.exit_code == 0, result.output
    assert len(Corpus.load(app_settings.corpus_dir).by_law(LawId.MCI)) == 31


def test_build_corpus_needs_fetched_pages(app_settings: Settings) -> None:
    result = runner.invoke(cli.app, ["build-corpus", "-s", "marco-civil"])

    assert result.exit_code == 1
    assert "run fetch first" in result.output


def test_unknown_source_is_a_usage_error(app_settings: Settings) -> None:
    result = runner.invoke(cli.app, ["build-corpus", "-s", "codigo-civil"])

    assert result.exit_code == 2
    assert "unknown source codigo-civil" in result.output


def test_partial_build_keeps_the_other_laws(app_settings: Settings) -> None:
    write_existing_corpus(app_settings.corpus_dir)
    store_page(app_settings.raw_dir, marco_civil_page())

    result = runner.invoke(cli.app, ["build-corpus", "-s", "marco-civil"])

    assert result.exit_code == 0, result.output
    corpus = Corpus.load(app_settings.corpus_dir)
    assert corpus.law_ids == [LawId.CDC, LawId.MCI]
    assert list(corpus.manifest.laws) == [LawId.CDC, LawId.MCI]


def test_partial_build_refuses_a_corpus_from_another_parser(app_settings: Settings) -> None:
    write_existing_corpus(app_settings.corpus_dir, parser_version="0.0.1")
    store_page(app_settings.raw_dir, marco_civil_page())

    result = runner.invoke(cli.app, ["build-corpus", "-s", "marco-civil"])

    assert result.exit_code == 1
    assert "rebuild every source" in result.output
    assert Corpus.load(app_settings.corpus_dir).law_ids == [LawId.CDC]


def test_fetch_downloads_the_selected_pages(
    app_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = marco_civil_page()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=page))
    monkeypatch.setattr(cli, "build_client", lambda: httpx.Client(transport=transport))

    result = runner.invoke(cli.app, ["fetch", "-s", "marco-civil"])

    assert result.exit_code == 0, result.output
    assert "downloaded" in result.output
    assert html_path(app_settings.raw_dir, "marco-civil").read_bytes() == page


def test_fetch_reports_download_failures(
    app_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(404))
    monkeypatch.setattr(cli, "build_client", lambda: httpx.Client(transport=transport))

    result = runner.invoke(cli.app, ["fetch", "-s", "marco-civil"])

    assert result.exit_code == 1
    assert "Download failed" in result.output


def test_fetch_defaults_to_every_source(
    app_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []

    def fake_fetch_all(
        sources: Sequence[SourcePage], raw_dir: Path, **_: object
    ) -> list[FetchResult]:
        requested.extend(source.slug for source in sources)
        return []

    monkeypatch.setattr(cli, "fetch_all", fake_fetch_all)

    result = runner.invoke(cli.app, ["fetch", "--force"])

    assert result.exit_code == 0, result.output
    assert requested == list(SOURCES)
