"""Fixtures of the agent tests."""

import pytest

from lexgraph.domain import Passage
from tests.unit.agent.fakes import CDC_49, CF_7, CLT_58, LGPD_18, FakeRetriever


@pytest.fixture
def passages() -> list[Passage]:
    """Indexed passages: three single-chunk articles and CF art. 7º in three slices."""
    return [CDC_49, CLT_58, LGPD_18, *CF_7]


@pytest.fixture
def retriever(passages: list[Passage]) -> FakeRetriever:
    return FakeRetriever(passages)


@pytest.fixture
def provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy provider keys, enough to build the real model clients without network calls."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
