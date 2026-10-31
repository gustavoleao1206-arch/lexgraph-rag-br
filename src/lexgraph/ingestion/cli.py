"""Ingestion commands: ``fetch`` downloads the official pages, ``build-corpus`` parses them."""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from lexgraph.catalog import SOURCES, SourcePage
from lexgraph.config import get_settings
from lexgraph.corpus import MANIFEST_FILE, Corpus, CorpusManifest, LawManifest, write_corpus
from lexgraph.domain import Article, LawId
from lexgraph.ingestion.fetch import FetchMeta, build_client, fetch_all, html_path, read_meta
from lexgraph.ingestion.planalto import PARSER_VERSION, parse_page
from lexgraph.ingestion.validate import validate_corpus

app = typer.Typer(help="Download and parse the official law pages.", no_args_is_help=True)
console = Console()

SourceOption = Annotated[
    list[str] | None,
    typer.Option(
        "--source",
        "-s",
        help=f"Source to process (repeatable): {', '.join(SOURCES)}. Default: all.",
        show_default=False,
    ),
]


@app.command()
def fetch(
    source: SourceOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Download even when the stored copy is current.")
    ] = False,
) -> None:
    """Download the official pages into the raw data folder (conditional requests)."""
    settings = get_settings()
    sources = _select_sources(source)
    try:
        with build_client() as client:
            results = fetch_all(sources, settings.raw_dir, client=client, force=force)
    except httpx.HTTPError as exc:
        console.print(f"[red]Download failed:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"Pages in {escape(str(settings.raw_dir))}")
    for column in ("Source", "Result", "Last modified", "SHA-256"):
        table.add_column(column)
    for result in results:
        table.add_row(
            result.slug,
            "not modified" if result.not_modified else "downloaded",
            result.meta.last_modified or "-",
            result.meta.sha256[:12],
        )
    console.print(table)


@app.command("build-corpus")
def build_corpus(
    source: SourceOption = None,
    allow_problems: Annotated[
        bool,
        typer.Option("--allow-problems", help="Write the corpus even if validation fails."),
    ] = False,
) -> None:
    """Parse the raw pages, validate every law and write the corpus with its manifest.

    With --source, the other laws of an existing corpus are kept as they are.
    """
    settings = get_settings()
    articles, laws = _parse_sources(_select_sources(source), settings.raw_dir)
    problems = validate_corpus(articles)
    _print_summary(laws, problems)
    if problems and not allow_problems:
        console.print("[red]Validation failed: corpus not written[/red] (see --allow-problems).")
        raise typer.Exit(code=1)
    if source:
        _keep_other_laws(settings.corpus_dir, articles, laws)
    manifest = CorpusManifest(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        parser_version=PARSER_VERSION,
        laws=_in_catalog_order(laws),
    )
    write_corpus(settings.corpus_dir, _in_catalog_order(articles), manifest)
    total = sum(len(law_articles) for law_articles in articles.values())
    console.print(f"Wrote {total} articles to {escape(str(settings.corpus_dir))}")


def _select_sources(slugs: Sequence[str] | None) -> list[SourcePage]:
    if not slugs:
        return list(SOURCES.values())
    if unknown := [slug for slug in slugs if slug not in SOURCES]:
        msg = f"unknown source {', '.join(unknown)}; choose from {', '.join(SOURCES)}"
        raise typer.BadParameter(msg, param_hint="--source")
    return [SOURCES[slug] for slug in dict.fromkeys(slugs)]


def _parse_sources(
    sources: Sequence[SourcePage], raw_dir: Path
) -> tuple[dict[LawId, list[Article]], dict[LawId, LawManifest]]:
    articles: dict[LawId, list[Article]] = {}
    laws: dict[LawId, LawManifest] = {}
    for source in sources:
        try:
            meta = read_meta(raw_dir, source.slug)
            html = html_path(raw_dir, source.slug).read_bytes()
        except FileNotFoundError as exc:
            console.print(f"[red]Missing {escape(str(exc.filename))}[/red]: run fetch first.")
            raise typer.Exit(code=1) from exc
        for law_id, law_articles in parse_page(html, source, meta).items():
            articles[law_id] = law_articles
            laws[law_id] = _law_manifest(law_id, source, meta, law_articles)
    return articles, laws


def _law_manifest(
    law_id: LawId, source: SourcePage, meta: FetchMeta, articles: Sequence[Article]
) -> LawManifest:
    statuses = Counter(article.status for article in articles)
    return LawManifest(
        law_id=law_id,
        source=source.slug,
        source_url=meta.final_url,
        source_sha256=meta.sha256,
        last_modified=meta.last_modified,
        fetched_at=meta.fetched_at,
        articles=len(articles),
        in_force=statuses["vigente"],
        revoked=statuses["revogado"],
        vetoed=statuses["vetado"],
    )


def _keep_other_laws(
    corpus_dir: Path, articles: dict[LawId, list[Article]], laws: dict[LawId, LawManifest]
) -> None:
    """Add the laws of the existing corpus that were not rebuilt in this run."""
    if not (corpus_dir / MANIFEST_FILE).is_file():
        return
    existing = Corpus.load(corpus_dir)
    if existing.manifest.parser_version != PARSER_VERSION:
        console.print(
            f"[red]The corpus was built by parser {existing.manifest.parser_version}; "
            f"rebuild every source with parser {PARSER_VERSION}.[/red]"
        )
        raise typer.Exit(code=1)
    for law_id, law in existing.manifest.laws.items():
        if law_id not in articles:
            articles[law_id] = existing.by_law(law_id)
            laws[law_id] = law


def _print_summary(laws: Mapping[LawId, LawManifest], problems: Mapping[LawId, list[str]]) -> None:
    table = Table(title="Parsed laws")
    for column in ("Law", "Articles", "In force", "Revoked", "Vetoed", "Problems"):
        table.add_column(column, justify="left" if column == "Law" else "right")
    for law_id, law in _in_catalog_order(laws).items():
        counts = (
            law.articles,
            law.in_force,
            law.revoked,
            law.vetoed,
            len(problems.get(law_id, [])),
        )
        table.add_row(law_id.value, *(str(count) for count in counts))
    console.print(table)
    for law_id, law_problems in problems.items():
        for problem in law_problems:
            console.print(f"[yellow]{law_id.value}[/yellow]: {escape(problem)}")


def _in_catalog_order[T](items: Mapping[LawId, T]) -> dict[LawId, T]:
    return {law_id: items[law_id] for law_id in LawId if law_id in items}
