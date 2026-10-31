"""Golden evaluation set: schema, loading and checks against the parsed corpus.

The golden set is a JSON Lines file (``data/eval/golden.jsonl`` by default) with one
:class:`GoldenItem` per line. Article ids use the corpus format (``"CLT:482"``, ``"CF:5"``,
``"ADCT:10"``), so retrieval rankings and answer citations compare with them directly.
"""

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from lexgraph.corpus import Corpus
from lexgraph.domain import LawId, split_article_id

GOLDEN_FILE = "golden.jsonl"
EXPECTED_GRADE = 2
ACCEPTABLE_GRADE = 1

QuestionType = Literal[
    "lookup", "conceptual", "multi_article", "cross_law", "out_of_scope", "adversarial", "temporal"
]
Difficulty = Literal["easy", "medium", "hard"]
Confidence = Literal["high", "medium-high", "medium", "low"]


class GoldenSetError(ValueError):
    """The golden set file is malformed."""


class GoldenItem(BaseModel):
    """One evaluation question with its reference answer and graded relevance judgments.

    ``must_include`` / ``must_not_include`` are regular expressions (``|`` separates
    alternatives) matched against answers ignoring accents, letter case and extra whitespace;
    numbers are written with digits only (``"2 horas"`` also matches the statute style
    ``"2 (duas) horas"``) and ``must_not_include`` is matched sentence by sentence (see
    :mod:`lexgraph.eval.rules`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-\d{3}$", description='Stable id such as "cdc-001".')
    question: str = Field(min_length=10)
    type: QuestionType
    expected_laws: list[LawId] = Field(description="Laws a good answer relies on.")
    expected_articles: list[str] = Field(description="Articles an answer needs (grade 2).")
    acceptable_articles: list[str] = Field(
        default_factory=list, description="Helpful but optional articles (grade 1)."
    )
    expected_units: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Units an answer needs inside long expected articles, labeled as in the chunks "
            '(e.g. {"CF:5": ["XI"]}): a slice of the article without them does not help.'
        ),
    )
    reference_answer: str = Field(min_length=1, description="Answer written from the corpus.")
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    confidence: Confidence

    @field_validator("expected_articles", "acceptable_articles")
    @classmethod
    def _check_article_ids(cls, article_ids: list[str]) -> list[str]:
        for article_id in article_ids:
            split_article_id(article_id)
        if len(set(article_ids)) != len(article_ids):
            msg = "duplicate article ids"
            raise ValueError(msg)
        return article_ids

    @field_validator("expected_laws")
    @classmethod
    def _check_laws(cls, laws: list[LawId]) -> list[LawId]:
        if len(set(laws)) != len(laws):
            msg = "duplicate laws"
            raise ValueError(msg)
        return laws

    @field_validator("must_include", "must_not_include")
    @classmethod
    def _check_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                msg = f"invalid regular expression {pattern!r}: {exc}"
                raise ValueError(msg) from exc
        return patterns

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.should_refuse != (self.type == "out_of_scope"):
            msg = "should_refuse must be true exactly for out_of_scope items"
            raise ValueError(msg)
        if self.should_refuse:
            if (
                self.expected_laws
                or self.expected_articles
                or self.acceptable_articles
                or self.expected_units
            ):
                msg = "out_of_scope items cannot list laws or articles"
                raise ValueError(msg)
            return self
        if not self.expected_articles:
            msg = "in-scope items need at least one expected article"
            raise ValueError(msg)
        if both := set(self.expected_articles) & set(self.acceptable_articles):
            msg = f"articles both expected and acceptable: {sorted(both)}"
            raise ValueError(msg)
        if stray := set(self.expected_units) - set(self.expected_articles):
            msg = f"expected_units for articles that are not expected: {sorted(stray)}"
            raise ValueError(msg)
        if any(not units for units in self.expected_units.values()):
            msg = "expected_units lists no unit for an article"
            raise ValueError(msg)
        required = {_law_of(article_id) for article_id in self.expected_articles}
        if missing := required - set(self.expected_laws):
            msg = f"expected_laws lacks the laws of expected articles: {sorted(missing)}"
            raise ValueError(msg)
        covered = required | {_law_of(article_id) for article_id in self.acceptable_articles}
        if unused := set(self.expected_laws) - covered:
            msg = f"expected_laws without any expected or acceptable article: {sorted(unused)}"
            raise ValueError(msg)
        return self

    @property
    def relevance(self) -> dict[str, int]:
        """Graded relevance per article id: 2 for expected, 1 for acceptable articles."""
        grades = dict.fromkeys(self.acceptable_articles, ACCEPTABLE_GRADE)
        grades.update(dict.fromkeys(self.expected_articles, EXPECTED_GRADE))
        return grades


def _law_of(article_id: str) -> LawId:
    return split_article_id(article_id)[0]


def load_golden(path: Path) -> list[GoldenItem]:
    """Read and validate a golden set; raises :class:`GoldenSetError` naming the bad line."""
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValidationError as exc:
                msg = f"{path.name}, line {number}: {exc}"
                raise GoldenSetError(msg) from exc
    duplicates = sorted(item_id for item_id, n in Counter(i.id for i in items).items() if n > 1)
    if duplicates:
        msg = f"duplicate item ids: {', '.join(duplicates)}"
        raise GoldenSetError(msg)
    return items


def validate_against_corpus(items: Sequence[GoldenItem], corpus: Corpus) -> list[str]:
    """List problems of the golden set against ``corpus``; an empty list means it is usable.

    Every expected and acceptable article must exist and be in force, and every expected law
    must be part of the corpus.
    """
    problems: list[str] = []
    corpus_laws = set(corpus.law_ids)
    for item in items:
        problems.extend(
            f"{item.id}: law {law} is not in the corpus"
            for law in item.expected_laws
            if law not in corpus_laws
        )
        for article_id in (*item.expected_articles, *item.acceptable_articles):
            article = corpus.get(article_id)
            if article is None:
                problems.append(f"{item.id}: {article_id} is not in the corpus")
            elif article.status != "vigente":
                problems.append(f"{item.id}: {article_id} is {article.status}, not in force")
    return problems
