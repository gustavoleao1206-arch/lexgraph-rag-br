"""``ask`` command: questions to the legal assistant from the terminal, answers streamed."""

import asyncio
import re
from collections.abc import Callable
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

from lexgraph.config import Settings, bootstrap_env, get_settings
from lexgraph.container import open_service
from lexgraph.domain import LawId
from lexgraph.protocols import ChatService
from lexgraph.schemas import (
    THREAD_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    ResetEvent,
    StatusEvent,
    TokenEvent,
)
from lexgraph.service import ChatServiceError

app = typer.Typer(help="Ask the legal assistant.")

EXIT_WORDS = frozenset({"sair", "exit", "quit"})


class InvalidQuestionError(ValueError):
    """A question the service would reject; the message is meant for the user."""


def check_question(text: str, max_chars: int) -> str:
    """``text`` without surrounding blanks; raises :class:`InvalidQuestionError` if unusable."""
    question = text.strip()
    if not question:
        msg = "A pergunta está vazia."
        raise InvalidQuestionError(msg)
    if len(question) > max_chars:
        msg = (
            f"A pergunta é longa demais: tem {len(question)} caracteres, e o limite é {max_chars}."
        )
        raise InvalidQuestionError(msg)
    return question


class _Printer:
    """Renders streamed answers: progress in grey, tokens as they arrive, then citations."""

    def __init__(self, console: Console, *, show_sources: bool) -> None:
        self._console = console
        self._show_sources = show_sources
        self._mid_line = False

    def _end_line(self) -> None:
        if self._mid_line:
            self._console.print()
            self._mid_line = False

    def status(self, event: StatusEvent) -> None:
        self._end_line()
        self._console.print(f"[dim]· {escape(event.message)}[/dim]")

    def token(self, event: TokenEvent) -> None:
        self._console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)
        self._mid_line = True

    def reset(self, event: ResetEvent) -> None:
        self._end_line()
        self._console.print(f"[yellow]Resposta reiniciada:[/yellow] {escape(event.reason)}")

    def error(self, event: ErrorEvent) -> None:
        self._end_line()
        self._console.print(f"[red]{escape(event.message)}[/red]")

    def final(self, response: ChatResponse) -> None:
        self._end_line()
        if response.citations:
            self._console.print("\n[bold]Fontes citadas[/bold]")
            for citation in response.citations:
                line = f"[{citation.index}] {citation.citation} — {citation.url}"
                self._console.print(line, markup=False, highlight=False)
        if response.grounding is not None and not response.grounding.grounded:
            self._console.print(
                "[yellow]A revisão apontou afirmações sem apoio nas fontes:[/yellow]"
            )
            for claim in response.grounding.unsupported_claims:
                self._console.print(f"  - {claim}", markup=False, highlight=False)
        if self._show_sources:
            for source in response.sources:
                self._console.rule(f"[{source.index}] {escape(source.citation)}")
                self._console.print(source.text, markup=False, highlight=False)


async def _stream_answer(service: ChatService, request: ChatRequest, printer: _Printer) -> bool:
    """Print the answer to ``request`` as it streams; False when no answer was produced."""
    ok = True
    async for event in service.stream(request):
        if isinstance(event, StatusEvent):
            printer.status(event)
        elif isinstance(event, TokenEvent):
            printer.token(event)
        elif isinstance(event, ResetEvent):
            printer.reset(event)
        elif isinstance(event, FinalEvent):
            printer.final(event.response)
            ok = event.response.route != "unavailable"
        elif isinstance(event, ErrorEvent):
            printer.error(event)
            ok = False
    return ok


async def _print_json(service: ChatService, request: ChatRequest, console: Console) -> bool:
    try:
        response = await service.ask(request)
    except ChatServiceError as error:
        console.print(f"[red]{escape(str(error))}[/red]")
        return False
    typer.echo(response.model_dump_json(indent=2))
    return True


async def _chat(
    service: ChatService,
    new_request: Callable[[str], ChatRequest],
    printer: _Printer,
    console: Console,
) -> None:
    """Interactive session on one thread, until an empty line or an exit word."""
    console.print("[dim]Digite sua pergunta; uma linha vazia ou “sair” encerra a conversa.[/dim]")
    while True:
        text = Prompt.ask("[bold]Pergunta[/bold]", console=console, default="", show_default=False)
        if not text.strip() or text.strip().lower() in EXIT_WORDS:
            return
        try:
            request = new_request(text)
        except InvalidQuestionError as error:
            console.print(f"[red]{escape(str(error))}[/red]")
            continue
        await _stream_answer(service, request, printer)
        console.print()


async def _run(
    settings: Settings,
    request: ChatRequest | None,
    new_request: Callable[[str], ChatRequest],
    *,
    as_json: bool,
    show_sources: bool,
    console: Console,
) -> bool:
    """Answer ``request``, or chat when there is none; False when no answer was produced."""
    async with open_service(settings) as service:
        printer = _Printer(console, show_sources=show_sources)
        if request is None:
            await _chat(service, new_request, printer, console)
            return True
        if as_json:
            return await _print_json(service, request, console)
        return await _stream_answer(service, request, printer)


@app.command()
def ask(
    question: Annotated[
        str | None, typer.Argument(help="Question; omit it for an interactive session.")
    ] = None,
    thread: Annotated[
        str | None, typer.Option("--thread", "-t", help="Conversation thread to continue.")
    ] = None,
    laws: Annotated[
        list[LawId] | None,
        typer.Option("--law", "-l", help="Search only this law (repeatable)."),
    ] = None,
    rerank: Annotated[
        bool | None,
        typer.Option("--rerank/--no-rerank", help="Override the configured reranking."),
    ] = None,
    sources: Annotated[
        bool, typer.Option("--sources", "-s", help="Also print the passages given to the model.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the final response as JSON, without streaming.")
    ] = False,
) -> None:
    """Ask a question about the laws in the corpus, or chat when no question is given."""
    if as_json and question is None:
        raise typer.BadParameter("needs a question", param_hint="--json")
    if thread is not None and not re.fullmatch(THREAD_ID_PATTERN, thread):
        msg = "use up to 64 letters, digits, '-' or '_'"
        raise typer.BadParameter(msg, param_hint="--thread")
    thread_id = thread or uuid4().hex
    bootstrap_env()
    settings = get_settings()
    console = Console()

    def new_request(text: str) -> ChatRequest:
        checked = check_question(text, settings.max_question_chars)
        return ChatRequest(question=checked, thread_id=thread_id, laws=laws, rerank=rerank)

    request = None
    if question is not None:
        try:
            request = new_request(question)
        except InvalidQuestionError as error:
            console.print(f"[red]{escape(str(error))}[/red]")
            raise typer.Exit(code=1) from None
    ok = asyncio.run(
        _run(
            settings,
            request,
            new_request,
            as_json=as_json,
            show_sources=sources,
            console=console,
        )
    )
    if not ok:
        raise typer.Exit(code=1)
