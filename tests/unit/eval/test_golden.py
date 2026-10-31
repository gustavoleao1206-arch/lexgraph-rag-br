"""Golden set schema, loading and validation against the corpus."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lexgraph.corpus import Corpus
from lexgraph.domain import LawId
from lexgraph.eval.golden import GoldenItem, GoldenSetError, load_golden, validate_against_corpus
from tests.unit.eval.fakes import GOLDEN_PATH, article_for, make_corpus, make_item


def test_load_golden_reads_every_item(golden_items: list[GoldenItem]) -> None:
    assert [item.id for item in golden_items] == ["cdc-001", "clt-001", "lgpd-001", "oos-001"]
    assert golden_items[1].expected_laws == [LawId.CLT, LawId.CF]
    assert golden_items[3].should_refuse


def test_relevance_grades_expected_and_acceptable_articles(golden_items: list[GoldenItem]) -> None:
    assert golden_items[1].relevance == {"CLT:59": 2, "CF:7": 1}
    assert golden_items[3].relevance == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_laws": ["ADCT"], "expected_articles": ["ADCT:10"]},
        {"expected_laws": ["CLT"], "expected_articles": ["CLT:59-A"]},
        {"expected_laws": ["CF"], "expected_articles": ["CF:preambulo"]},
        {"expected_laws": ["CDC", "CF"], "acceptable_articles": ["CF:5"]},
        {"acceptable_articles": ["CLT:58"]},
        {"expected_units": {"CDC:49": ["caput", "parágrafo único"]}},
    ],
)
def test_item_accepts_valid_variants(overrides: dict[str, Any]) -> None:
    assert make_item(**overrides).id == "cdc-001"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "cdc_1"}, "should match pattern"),
        ({"expected_articles": ["CDC-49"]}, "invalid article id"),
        ({"expected_articles": ["CDC:49", "CDC:49"]}, "duplicate article ids"),
        ({"expected_laws": ["CDC", "CDC"]}, "duplicate laws"),
        ({"expected_laws": ["CP"]}, "Input should be"),
        ({"must_include": ["(7 dias"]}, "invalid regular expression"),
        ({"should_refuse": True}, "should_refuse must be true exactly"),
        ({"type": "out_of_scope"}, "should_refuse must be true exactly"),
        ({"type": "out_of_scope", "should_refuse": True}, "cannot list laws or articles"),
        ({"expected_laws": [], "expected_articles": []}, "at least one expected article"),
        ({"acceptable_articles": ["CDC:49"]}, "both expected and acceptable"),
        ({"expected_laws": ["CLT"]}, "lacks the laws of expected articles"),
        ({"expected_laws": ["CDC", "LGPD"]}, "without any expected or acceptable article"),
        ({"expected_units": {"CDC:18": ["I"]}}, "expected_units for articles that are not"),
        ({"expected_units": {"CDC:49": []}}, "lists no unit"),
        (
            {
                "type": "out_of_scope",
                "should_refuse": True,
                "expected_laws": [],
                "expected_articles": [],
                "expected_units": {"CDC:49": ["I"]},
            },
            "cannot list laws or articles",
        ),
        ({"answer": "texto"}, "Extra inputs are not permitted"),
    ],
)
def test_item_rejects_inconsistent_data(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_item(**overrides)


def _write(path: Path, *lines: str) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    line = make_item().model_dump_json()
    items = load_golden(_write(tmp_path / "golden.jsonl", "", line, "   "))
    assert [item.id for item in items] == ["cdc-001"]


def test_load_golden_names_the_invalid_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "golden.jsonl", make_item().model_dump_json(), "{not json")
    with pytest.raises(GoldenSetError, match=r"golden\.jsonl, line 2"):
        load_golden(path)


def test_load_golden_reports_schema_errors(tmp_path: Path) -> None:
    bad = json.dumps(make_item().model_dump(mode="json") | {"type": "trivia"})
    path = _write(tmp_path / "golden.jsonl", make_item(id="cdc-002").model_dump_json(), bad)
    with pytest.raises(GoldenSetError, match=r"line 2[\s\S]*type"):
        load_golden(path)


def test_load_golden_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = make_item().model_dump_json()
    with pytest.raises(GoldenSetError, match="duplicate item ids: cdc-001"):
        load_golden(_write(tmp_path / "golden.jsonl", line, line))


def test_load_golden_requires_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "missing.jsonl")


def test_fixture_golden_set_is_valid_against_its_corpus(
    golden_items: list[GoldenItem], corpus: Corpus
) -> None:
    assert GOLDEN_PATH.is_file()
    assert validate_against_corpus(golden_items, corpus) == []


def test_validate_against_corpus_reports_every_problem() -> None:
    items = [
        make_item(),
        make_item(
            id="clt-001",
            expected_laws=["CLT", "CF"],
            expected_articles=["CLT:59"],
            acceptable_articles=["CF:7"],
        ),
    ]
    corpus = make_corpus([article_for("CDC:49", status="revogado"), article_for("CLT:59")])

    assert validate_against_corpus(items, corpus) == [
        "cdc-001: CDC:49 is revogado, not in force",
        "clt-001: law CF is not in the corpus",
        "clt-001: CF:7 is not in the corpus",
    ]
