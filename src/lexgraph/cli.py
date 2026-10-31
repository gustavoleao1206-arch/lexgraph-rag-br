"""Command-line interface: ``lexgraph <command>``.

Each area defines its commands in its own ``cli`` module; this one mounts them::

    lexgraph fetch / build-corpus     download and parse the official pages
    lexgraph ingest / search          build the search index and query it
    lexgraph ask                      ask a question in the terminal
    lexgraph serve / ui               run the HTTP API and the web interface
    lexgraph eval ...                 evaluate retrieval and answers
"""

from typing import Annotated

import typer

from lexgraph import __version__
from lexgraph.agent.cli import app as agent_app
from lexgraph.api.cli import app as api_app
from lexgraph.eval.cli import app as eval_app
from lexgraph.ingestion.cli import app as ingestion_app
from lexgraph.retrieval.cli import app as retrieval_app

app = typer.Typer(
    name="lexgraph",
    help="Agentic RAG over Brazilian federal legislation.",
    no_args_is_help=True,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"lexgraph {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Agentic RAG over Brazilian federal legislation."""


for sub_app in (ingestion_app, retrieval_app, agent_app, api_app):
    app.add_typer(sub_app)
app.add_typer(eval_app, name="eval")
