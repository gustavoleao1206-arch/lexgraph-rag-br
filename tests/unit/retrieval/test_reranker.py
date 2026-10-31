"""Cross-encoder reranking with a fake FastEmbed model (no download)."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from lexgraph.domain import Passage
from lexgraph.retrieval import reranker as reranker_module
from lexgraph.retrieval.reranker import CrossEncoderReranker, Reranker
from lexgraph.retrieval.store import chunk_payload, passage_from_payload
from tests.unit.retrieval.conftest import CHUNKS


class FakeCrossEncoder:
    """Stands in for FastEmbed's ``TextCrossEncoder``: scores = query words in the passage."""

    def __init__(self, model_name: str, cache_dir: str | None, options: dict[str, Any]) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.options = options
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        texts = list(documents)
        self.calls.append((query, texts, batch_size))
        words = query.lower().split()
        return [float(sum(text.lower().count(word) for word in words)) for text in texts]


@pytest.fixture
def cross_encoders(monkeypatch: pytest.MonkeyPatch) -> list[FakeCrossEncoder]:
    """Replace ``TextCrossEncoder`` with :class:`FakeCrossEncoder`; returns the instances."""
    created: list[FakeCrossEncoder] = []

    def factory(
        model_name: str, *, cache_dir: str | None = None, **options: Any
    ) -> FakeCrossEncoder:
        model = FakeCrossEncoder(model_name, cache_dir, options)
        created.append(model)
        return model

    monkeypatch.setattr(reranker_module, "TextCrossEncoder", factory)
    return created


@pytest.fixture
def missing_model(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """``TextCrossEncoder`` that fails as FastEmbed does without the model files."""
    attempts: list[dict[str, Any]] = []

    def factory(model_name: str, **options: Any) -> FakeCrossEncoder:
        attempts.append(options)
        msg = f"Could not load model {model_name} from any source."
        raise ValueError(msg)

    monkeypatch.setattr(reranker_module, "TextCrossEncoder", factory)
    return attempts


@pytest.fixture
def passages() -> list[Passage]:
    """The test chunks as passages with decreasing search scores."""
    return [
        passage_from_payload(chunk_payload(chunk), score=1.0 - rank / 10)
        for rank, chunk in enumerate(CHUNKS)
    ]


def test_rerank_orders_by_cross_encoder_score(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage]
) -> None:
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3-int8")

    ranked = reranker.rerank("férias gozo duração", passages, top_n=2)

    assert [passage.article_id for passage in ranked] == ["CLT:129", "CLT:58"]
    assert [passage.rerank_score for passage in ranked] == [2.0, 1.0]
    original = {passage.chunk_id: passage.score for passage in passages}
    assert all(passage.score == original[passage.chunk_id] for passage in ranked)


def test_rerank_scores_header_and_text(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage]
) -> None:
    CrossEncoderReranker("model", batch_size=4).rerank("férias", passages[:2], top_n=2)

    (query, texts, batch_size) = cross_encoders[0].calls[0]
    assert query == "férias"
    assert texts == [passage.embedding_text() for passage in passages[:2]]
    assert batch_size == 4


def test_rerank_keeps_every_passage_when_top_n_is_large(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage]
) -> None:
    ranked = CrossEncoderReranker("model").rerank("férias", passages, top_n=50)

    assert len(ranked) == len(passages)


def test_rerank_loads_the_model_once_on_first_use(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage], tmp_path: Path
) -> None:
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3-int8", cache_dir=tmp_path)
    assert cross_encoders == []

    reranker.rerank("férias", passages, top_n=1)
    reranker.rerank("contrato", passages, top_n=1)

    assert len(cross_encoders) == 1
    assert cross_encoders[0].model_name == "BAAI/bge-reranker-v2-m3-int8"
    assert cross_encoders[0].cache_dir == str(tmp_path)


@pytest.mark.parametrize("top_n", [0, 3])
def test_nothing_to_rerank_skips_the_model(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage], top_n: int
) -> None:
    candidates = passages if top_n == 0 else []

    assert CrossEncoderReranker("model").rerank("férias", candidates, top_n=top_n) == []
    assert cross_encoders == []


def test_cross_encoder_reranker_satisfies_the_protocol() -> None:
    assert isinstance(CrossEncoderReranker("model"), Reranker)


def test_available_when_the_model_is_in_the_local_cache(
    cross_encoders: list[FakeCrossEncoder], tmp_path: Path
) -> None:
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3-int8", cache_dir=tmp_path)

    assert reranker.is_available()
    assert reranker.is_available()

    (probe,) = cross_encoders  # looked up once, offline, without loading the model
    assert probe.cache_dir == str(tmp_path)
    assert probe.options == {"lazy_load": True, "local_files_only": True}


def test_unavailable_when_the_model_is_not_installed(
    missing_model: list[dict[str, Any]],
) -> None:
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3-int8")

    assert not reranker.is_available()
    assert not reranker.is_available()
    assert missing_model == [{"cache_dir": None, "lazy_load": True, "local_files_only": True}]


def test_a_loaded_model_is_available_without_lookup(
    cross_encoders: list[FakeCrossEncoder], passages: list[Passage]
) -> None:
    reranker = CrossEncoderReranker("model")
    reranker.rerank("férias", passages, top_n=1)

    assert reranker.is_available()
    assert len(cross_encoders) == 1
