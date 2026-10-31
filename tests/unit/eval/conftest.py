"""Fixtures for the evaluation harness tests."""

import pytest

from lexgraph.corpus import Corpus
from lexgraph.eval.golden import GoldenItem, load_golden
from tests.unit.eval.fakes import CORPUS_IDS, GOLDEN_PATH, article_for, make_corpus


@pytest.fixture
def golden_items() -> list[GoldenItem]:
    """The four-item golden set of ``data/golden.jsonl`` (three in scope, one out of scope)."""
    return load_golden(GOLDEN_PATH)


@pytest.fixture
def corpus() -> Corpus:
    """A corpus with every article the golden fixture refers to, all in force."""
    return make_corpus([article_for(article_id, seq=n) for n, article_id in enumerate(CORPUS_IDS)])
