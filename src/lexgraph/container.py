"""Composition root: wires the vector store, retriever, models and checkpointer into the service.

The embedded Qdrant store and the SQLite checkpointer belong to one process at a time, so
open the service once per process: in the FastAPI lifespan, or once per CLI command.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from lexgraph.agent.graph import build_graph
from lexgraph.agent.llm import build_agent_context, models_configured
from lexgraph.config import Settings
from lexgraph.corpus import MANIFEST_FILE, CorpusManifest
from lexgraph.service import LexGraphService

logger = logging.getLogger(__name__)


def read_corpus_date(corpus_dir: Path) -> str | None:
    """Date (``dd/mm/yyyy``) of the oldest fetch of the corpus texts, from its manifest.

    Returns None when there is no readable manifest; the answers then omit the date.
    """
    path = corpus_dir / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
        stamps = [law.fetched_at for law in manifest.laws.values()] or [manifest.generated_at]
        oldest = min(datetime.fromisoformat(stamp) for stamp in stamps)
    except (OSError, ValueError, TypeError):
        logger.warning("could not read the corpus date from %s", path, exc_info=True)
        return None
    return f"{oldest:%d/%m/%Y}"


@asynccontextmanager
async def open_service(settings: Settings) -> AsyncIterator[LexGraphService]:
    """Open the chat service; the Qdrant client and the checkpointer close on exit.

    Provider API keys are read from the environment (see ``lexgraph.config.bootstrap_env``).
    When a model role has no key, the service still starts (search and the API work) and
    every question gets an explanation of how to configure one.
    """
    # Imported here so that importing this module does not load the embedding models' stack.
    from lexgraph.retrieval.factory import build_retriever
    from lexgraph.retrieval.store import create_qdrant_client

    configured = models_configured(settings)
    if not configured:
        logger.warning("a model role has no provider API key; questions will not be answered")
    client = create_qdrant_client(settings)
    try:
        context = build_agent_context(settings, build_retriever(settings, client=client))
        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
            yield LexGraphService(
                build_graph(saver),
                context,
                settings,
                corpus_date=read_corpus_date(settings.corpus_dir),
                models_configured=configured,
            )
    finally:
        client.close()
