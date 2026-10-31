"""FastAPI application factory.

Run it with ``lexgraph serve`` or ``uvicorn lexgraph.api.app:create_app --factory``. Keep a
single worker process: embedded Qdrant allows only one process per storage folder.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from lexgraph import __version__
from lexgraph.api.routes import AppState, router, v1_router
from lexgraph.api.security import (
    API_KEY_HEADER,
    MAX_BODY_BYTES,
    RequestGuardMiddleware,
    api_key_dependencies,
)
from lexgraph.config import Settings, bootstrap_env, get_settings
from lexgraph.corpus import MANIFEST_FILE, Corpus
from lexgraph.protocols import ChatService

logger = logging.getLogger(__name__)

DESCRIPTION = f"""\
Questions and answers over Brazilian federal legislation (CLT, CDC, LGPD, Marco Civil da
Internet, Constituição Federal and ADCT), grounded in the official compiled texts and cited
article by article.

When the server sets `LEXGRAPH_API_KEY`, every `/v1` endpoint requires it in the `X-API-Key`
header. Requests whose `Host` header is not in `LEXGRAPH_ALLOWED_HOSTS` are rejected with 400,
and request bodies over {MAX_BODY_BYTES // 1024} KiB with 413.
"""

OPENAPI_TAGS = [
    {"name": "chat", "description": "Ask questions, in one response or as a stream of events."},
    {"name": "corpus", "description": "Laws and articles of the corpus, and passage search."},
    {"name": "system", "description": "Health and service configuration."},
]

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[AppState]]


def create_app(
    settings: Settings | None = None,
    *,
    service: ChatService | None = None,
    corpus: Corpus | None = None,
) -> FastAPI:
    """Build the API.

    ``service`` and ``corpus`` let tests inject fakes; when omitted, the lifespan opens the
    real chat service and loads the corpus from ``settings.corpus_dir``.
    """
    resolved = settings if settings is not None else get_settings()
    app = FastAPI(
        title="LexGraph API",
        summary="Agentic RAG over Brazilian federal legislation.",
        description=DESCRIPTION,
        version=__version__,
        openapi_tags=OPENAPI_TAGS,
        license_info={"name": "MIT", "identifier": "MIT"},
        lifespan=_lifespan(resolved, service, corpus),
        exception_handlers={
            RequestValidationError: _validation_error_handler,
            Exception: _unhandled_error_handler,
        },
    )
    # Added first so it runs after the Host and CORS checks: CORS answers preflight requests,
    # which never carry the key, and adds its headers to the 401 and 413 responses.
    app.add_middleware(
        RequestGuardMiddleware, api_key=resolved.api_key, protected_prefix=v1_router.prefix
    )
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", API_KEY_HEADER],
        )
    # Added last so it runs first: a page that rebinds its own domain name to this server
    # (DNS rebinding) sends a foreign Host header and is turned away before any route runs.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=resolved.allowed_hosts)
    app.include_router(router)
    app.include_router(v1_router, dependencies=api_key_dependencies(resolved.api_key))
    return app


def load_corpus(corpus_dir: Path) -> Corpus | None:
    """Load the corpus served by ``/v1/laws`` and ``/v1/articles``; ``None`` if not built."""
    if not (corpus_dir / MANIFEST_FILE).is_file():
        logger.warning("No corpus in %s: the corpus endpoints will be empty", corpus_dir)
        return None
    return Corpus.load(corpus_dir)


def _lifespan(settings: Settings, service: ChatService | None, corpus: Corpus | None) -> Lifespan:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[AppState]:
        async with AsyncExitStack() as stack:
            if service is None:
                chat_service = await stack.enter_async_context(_open_service(settings))
            else:
                chat_service = service
            if corpus is None:
                loaded_corpus = await asyncio.to_thread(load_corpus, settings.corpus_dir)
            else:
                loaded_corpus = corpus
            yield {"settings": settings, "service": chat_service, "corpus": loaded_corpus}

    return lifespan


def _open_service(settings: Settings) -> AbstractAsyncContextManager[ChatService]:
    # Imported here so the API (and its tests) load without the retrieval and agent stack.
    from lexgraph.container import open_service

    bootstrap_env()
    return open_service(settings)


async def _validation_error_handler(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    """Report what is invalid without echoing the submitted values back."""
    errors = [
        {"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]}
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors}
    )


async def _unhandled_error_handler(_request: Request, _error: Exception) -> JSONResponse:
    """Hide internal details; Starlette re-raises the error afterwards so the server logs it."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
