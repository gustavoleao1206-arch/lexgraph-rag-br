"""Consistency checks run on every parsed law before the corpus is written.

The official pages are edited by hand, so a new typo can silently drop, duplicate or merge
articles. Each check below catches one such failure mode; :func:`validate_law` returns a
human-readable problem per finding and an empty list for a clean law.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise

from lexgraph.catalog import law_info
from lexgraph.domain import Article, LawId, article_sort_key
from lexgraph.ingestion.planalto import ARTICLE_RE, DASHES, UNITS_ONLY_BODY_RE

# Amendment notes that must never survive in the text (see planalto.strip_annotations).
_RESIDUAL_NOTE_RE = re.compile(
    r"\((?:Reda[çc][ãa]o dada|Inclu[íi]d[oa]|Vide|Vig[êe]ncia|Revogad|Renumerad"
    r"|Produ[çc][ãa]o de efeito)",
    re.IGNORECASE,
)
# Parser markers, C1 controls (undecoded "&#150;") and U+FFFD (undecodable bytes).
_BAD_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ufffd]")
_MAX_LISTED = 10


def validate_law(law_id: LawId, articles: Sequence[Article]) -> list[str]:
    """Return the problems found in the parsed articles of ``law_id`` (empty when clean).

    Checks: articles ``1..expected_last_article`` all present, no duplicate numbers, document
    order matching the numbering, no residual amendment notes or stray control characters,
    every in-force article has a body that is more than a list of units, and every article
    has a URL.
    """
    if not articles:
        return ["no articles parsed"]
    ordered = sorted(articles, key=lambda article: article.seq)
    numbered = [article for article in ordered if article.number != "preambulo"]
    problems = [
        f"{article.article_id} belongs to {law_id}"
        for article in ordered
        if article.law_id != law_id
    ]
    problems += _check_numbering(law_id, [article.number for article in numbered])
    for article in ordered:
        problems += _check_article(article)
    return problems


def validate_corpus(articles_by_law: Mapping[LawId, Sequence[Article]]) -> dict[LawId, list[str]]:
    """Validate every law; only laws with problems appear in the result."""
    report = {
        law_id: validate_law(law_id, articles) for law_id, articles in articles_by_law.items()
    }
    return {law_id: problems for law_id, problems in report.items() if problems}


def _check_numbering(law_id: LawId, numbers: list[str]) -> list[str]:
    problems: list[str] = []
    expected_last = law_info(law_id).expected_last_article
    bases = {article_sort_key(number)[0] for number in numbers}
    if missing := [n for n in range(1, expected_last + 1) if n not in bases]:
        problems.append(f"missing articles: {_listing(missing)}")
    if extra := sorted(n for n in bases if n > expected_last):
        problems.append(f"articles beyond the expected last ({expected_last}): {_listing(extra)}")
    duplicates = [number for number, count in Counter(numbers).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate articles: {_listing(duplicates)}")
    breaks = [
        f"{before} -> {after}"
        for before, after in pairwise(numbers)
        if article_sort_key(after) <= article_sort_key(before)
    ]
    if breaks:
        problems.append(f"articles out of order: {_listing(breaks)}")
    return problems


def _check_article(article: Article) -> list[str]:
    problems: list[str] = []
    if residual := _RESIDUAL_NOTE_RE.search(article.text):
        problems.append(f"{article.article_id}: residual amendment note {residual[0]!r}")
    if bad := _BAD_CHARS_RE.search(article.text):
        problems.append(f"{article.article_id}: unexpected character U+{ord(bad[0]):04X}")
    if not article.text.strip():
        problems.append(f"{article.article_id}: empty text")
    elif article.status == "vigente" and _without_text(article):
        problems.append(f"{article.article_id}: in force but without text")
    if not article.url.startswith("https://"):
        problems.append(f"{article.article_id}: missing URL")
    return problems


def _without_text(article: Article) -> bool:
    """Whether the body is empty or only lists units ("Art. 117. e Parágrafo único.")."""
    body = _body(article).strip(" .;,:" + DASHES)
    return not body or UNITS_ONLY_BODY_RE.fullmatch(body) is not None


def _body(article: Article) -> str:
    """Article text without its label."""
    if article.number == "preambulo":
        return article.text
    return ARTICLE_RE.sub("", article.text, count=1)


def _listing(items: Sequence[object]) -> str:
    shown = ", ".join(str(item) for item in items[:_MAX_LISTED])
    hidden = len(items) - _MAX_LISTED
    return f"{shown} (+{hidden} more)" if hidden > 0 else shown
