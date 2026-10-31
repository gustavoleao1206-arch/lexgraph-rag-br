"""Commands that run the HTTP API and the Streamlit interface."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from lexgraph.config import bootstrap_env, get_settings
from lexgraph.logging_setup import configure_logging

UI_SCRIPT = Path(__file__).resolve().parent.parent / "ui" / "app.py"

app = typer.Typer(help="Run the HTTP API and the web interface.", no_args_is_help=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Network interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Run the HTTP API.

    Always one worker and no auto-reload: embedded Qdrant allows a single process.
    """
    bootstrap_env()
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "lexgraph.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
        log_config=None,
        log_level=settings.log_level.lower(),
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ui(
    ctx: typer.Context,
    host: Annotated[str, typer.Option(help="Network interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8501,
) -> None:
    """Run the Streamlit interface; extra options are passed on to ``streamlit run``.

    The interface talks to the API at ``LEXGRAPH_API_URL`` (default http://127.0.0.1:8000).
    """
    if importlib.util.find_spec("streamlit") is None:
        typer.echo(
            "Streamlit is not installed; install the 'ui' extra: uv sync --extra ui", err=True
        )
        raise typer.Exit(code=1)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(UI_SCRIPT),
        "--server.address",
        host,
        "--server.port",
        str(port),
        *ctx.args,
    ]
    # The command is built from this interpreter, a packaged script and the CLI options.
    completed = subprocess.run(command, check=False)  # noqa: S603
    raise typer.Exit(code=completed.returncode)
