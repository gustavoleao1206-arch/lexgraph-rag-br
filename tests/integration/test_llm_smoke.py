"""One real question, end to end: the configured models, the real index and the agent graph.

Deselected by the usual test run (``-m "not integration and not llm"``). It needs a provider
key exported before pytest starts (for example ``uv run --env-file .env pytest -m llm``) and
an index built with ``lexgraph ingest``, and it spends a few requests of the free-tier quota.
"""

import os
from pathlib import Path

import pytest

from lexgraph.agent.llm import PROVIDER_KEYS, models_configured
from lexgraph.config import Settings
from lexgraph.container import open_service
from lexgraph.schemas import ChatRequest

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not any(os.environ.get(key) for keys in PROVIDER_KEYS.values() for key in keys),
        reason="needs a provider API key (GROQ_API_KEY or GOOGLE_API_KEY) in the environment",
    ),
]

QUESTION = "Comprei um produto pela internet e me arrependi. Posso desistir? Qual é o prazo?"


@pytest.fixture
def live_settings(tmp_path: Path) -> Settings:
    """The configured settings, with a throwaway conversation store."""
    settings = Settings(checkpoint_path=tmp_path / "threads.sqlite")
    if not models_configured(settings):
        pytest.skip("some model role has no provider API key")
    if settings.qdrant_url is None and not settings.qdrant_path.is_dir():
        pytest.skip("no index: run `lexgraph ingest` first")
    return settings


async def test_a_real_question_is_answered_from_the_cited_law(live_settings: Settings) -> None:
    async with open_service(live_settings) as service:
        response = await service.ask(ChatRequest(question=QUESTION))

    assert response.route == "answered", response.answer
    assert "CDC:49" in {source.article_id for source in response.sources}
    assert response.citations, response.answer
    assert response.usage.total_tokens > 0
