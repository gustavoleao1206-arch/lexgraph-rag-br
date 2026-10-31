"""Commands that build the search index and query it from the terminal.

Model and vector-store modules are imported inside the commands, so the CLI starts quickly.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from lexgraph.config import RetrievalMode, Settings, bootstrap_env, get_settings
from lexgraph.corpus import Corpus
from lexgraph.domain import LawId, Passage
from lexgraph.logging_setup import configure_logging

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

app = typer.Typer(help="Build the search index and query it.", no_args_is_help=True)

console = Console()

LawsOption = Annotated[
    list[LawId] | None,
    typer.Option("--law", "-l", case_sensitive=False, help="Restrict to this law (repeatable)."),
]

_SNIPPET_CHARS = 280


@app.command()
def ingest(
    laws: LawsOption = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Chunks embedded per batch.")
    ] = 32,
    if_missing: Annotated[
        bool,
        typer.Option(
            "--if-missing",
            help="Do nothing when the index already holds these chunks, built with the "
            "configured models.",
        ),
    ] = False,
) -> None:
    """Rebuild the search index from the corpus.

    The articles in force are chunked, embedded and indexed into a new collection, and the
    alias then moves to it: the previous index keeps serving until the new one is complete.
    With --law, the new index holds only those laws. With --if-missing, an index that already
    holds the same chunks, embedded by the same models, is kept (Docker Compose runs this
    command on every start). Embedded Qdrant allows one process at a time, so stop the API
    first or use a Qdrant server.
    """
    from lexgraph.ingestion.chunking import chunk_corpus
    from lexgraph.retrieval.embeddings import build_dense_embedder
    from lexgraph.retrieval.indexer import index_chunks, index_is_current
    from lexgraph.retrieval.sparse import build_sparse_encoder

    settings = _setup()
    corpus = _load_corpus(settings)
    chunks = chunk_corpus(
        corpus,
        law_ids=laws or None,
        max_chars=settings.chunk_max_chars,
        pack_chars=settings.chunk_pack_chars,
    )
    if not chunks:
        _fail("The corpus has no articles in force for the selected laws.")
    dense = build_dense_embedder(settings)
    sparse = build_sparse_encoder(settings)
    alias = settings.collection
    with _qdrant(settings) as client:
        if if_missing and index_is_current(client, chunks, alias=alias, dense=dense, sparse=sparse):
            typer.echo(f"The index {alias} is up to date ({len(chunks)} chunks); nothing to do.")
            return
        with _progress() as progress:
            task = progress.add_task("Indexing chunks", total=len(chunks))
            report = index_chunks(
                client,
                chunks,
                alias=alias,
                dense=dense,
                sparse=sparse,
                batch_size=batch_size,
                on_progress=lambda done, _total: progress.update(task, completed=done),
            )
    typer.echo(
        f"Indexed {report.chunks} chunks into {report.collection} "
        f"(alias {report.alias}) in {report.seconds:.1f} s."
    )
    if report.dropped:
        typer.echo(f"Removed previous collections: {', '.join(report.dropped)}.")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Question or keywords to search for.")],
    k: Annotated[
        int | None,
        typer.Option("--k", "-k", min=1, help="Passages to return (default: LEXGRAPH_TOP_K)."),
    ] = None,
    mode: Annotated[
        RetrievalMode | None,
        typer.Option("--mode", "-m", help="Search mode (default: LEXGRAPH_RETRIEVAL_MODE)."),
    ] = None,
    laws: LawsOption = None,
    rerank: Annotated[
        bool | None,
        typer.Option(
            "--rerank/--no-rerank", help="Rerank with the cross-encoder (default: LEXGRAPH_RERANK)."
        ),
    ] = None,
) -> None:
    """Search the index and print the passages found, best first."""
    from lexgraph.retrieval.factory import build_retriever
    from lexgraph.retrieval.store import IndexMismatchError, collection_exists

    settings = _setup()
    with _qdrant(settings) as client:
        if not collection_exists(client, settings.collection):
            _fail(f"There is no index named {settings.collection}; run `lexgraph ingest` first.")
        try:
            retriever = build_retriever(settings, client=client)
        except IndexMismatchError as exc:
            _fail(f"{exc}.")
        passages = retriever.search(query, k=k, laws=laws, mode=mode, rerank=rerank)
    _print_passages(passages, mode=mode or settings.retrieval_mode)


def _setup() -> Settings:
    bootstrap_env()
    settings = get_settings()
    configure_logging(settings.log_level)
    return settings


def _load_corpus(settings: Settings) -> Corpus:
    try:
        return Corpus.load(settings.corpus_dir)
    except FileNotFoundError:
        _fail(f"There is no corpus in {settings.corpus_dir}; run `lexgraph build-corpus` first.")


@contextmanager
def _qdrant(settings: Settings) -> Iterator["QdrantClient"]:
    from lexgraph.retrieval.store import StorageLockedError, create_qdrant_client

    try:
        client = create_qdrant_client(settings)
    except StorageLockedError as exc:
        _fail(
            f"{exc}. Stop the API (`lexgraph serve`) or the other command using it, "
            "or set LEXGRAPH_QDRANT_URL to use a Qdrant server."
        )
    try:
        yield client
    finally:
        client.close()


def _progress() -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _print_passages(passages: Sequence[Passage], *, mode: RetrievalMode) -> None:
    if not passages:
        typer.echo("No passages found.")
        return
    noun = "passage" if len(passages) == 1 else "passages"
    table = Table(title=f"{len(passages)} {noun} ({mode} search)", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Citation", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Rerank", justify="right")
    table.add_column("Passage")
    for rank, passage in enumerate(passages, start=1):
        rerank_score = "-" if passage.rerank_score is None else f"{passage.rerank_score:.2f}"
        table.add_row(
            str(rank),
            Text(passage.citation, style=f"link {passage.url}"),
            f"{passage.score:.3f}",
            rerank_score,
            Text(_snippet(passage.text)),
        )
    console.print(table)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else f"{flat[: _SNIPPET_CHARS - 1]}…"


def _fail(message: str) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1)
