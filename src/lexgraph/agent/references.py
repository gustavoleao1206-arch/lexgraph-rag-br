"""Deterministic detection of law names and explicit article references in Portuguese text.

The query analyzer also extracts references, but a regular expression never misses a
plainly written one such as "art. 7º da CLT" and costs nothing, so the agent uses both.
Law names are matched regardless of case and accents; besides the catalog aliases, the law
numbers ("Lei nº 8.078/1990", "Decreto-Lei 5.452") are recognized.

A reference takes the law written right after it ("art. 5º, inciso XI, da Constituição") or
right before it ("CDC, art. 49"); otherwise its law is left unknown for the caller to decide.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from lexgraph.catalog import LAWS
from lexgraph.domain import ARTICLE_ID_RE, LawId, make_article_id


@dataclass(frozen=True)
class ArticleMention:
    """An article cited in a text; ``law`` is None when the text does not say which law."""

    law: LawId | None
    number: str


_EXTRA_ALIASES: dict[LawId, tuple[str, ...]] = {
    LawId.CLT: ("consolidação",),
    LawId.CF: ("crfb", "crfb/88"),
}
_LAW_NUMBERS: dict[str, LawId] = {
    "5452": LawId.CLT,
    "8078": LawId.CDC,
    "13709": LawId.LGPD,
    "12965": LawId.MCI,
}
_MAX_RANGE = 10

# Article numbers as written: "7", "7º", "7°", "7o", "59-A" (hyphen or en dash), "401A".
_NUMBER = (
    r"\d{1,4}(?:\s*[-\N{EN DASH}]\s*[A-Za-z]{1,2}(?![A-Za-z])|[A-Z](?![A-Za-z]))?"
    r"(?:\.?\s?[º°]|o(?![A-Za-z]))?"
)
_NUMBER_RE = re.compile(_NUMBER)
_SEPARATOR = r"(?:\s*,\s*|\s+(?:e|ou|a)\s+)"
_ARTICLES_RE = re.compile(
    rf"(?<![^\W\d_])(?i:art(?:igo)?s?)\.?\s*(?P<numbers>{_NUMBER}(?:{_SEPARATOR}{_NUMBER})*)"
)
# Units that may sit between an article number and its law: "art. 5º, inciso XI, da CF".
_QUALIFIER = (
    r"caput|paragrafo unico|§+\s*\d+\s*[o°]?|inc(?:iso)?s?\.?\s+[ivxlcdm]+"
    r"|al(?:inea)?s?\.?\s+\W?[a-z]\W?|[ivxlcdm]+(?![a-z0-9])|e(?![a-z])|ou(?![a-z])"
)
_LAW_AFTER_RE = re.compile(rf"(?:\s*,?\s*(?:{_QUALIFIER}))*\s*,?\s*(?:d[aoe]s?|n[ao]s?)\s+")
_LAW_NUMBER_RE = re.compile(
    r"(?<![a-z0-9])(?:decreto-)?lei\s*(?:n[\s.o°]*)?(\d{1,2}\.?\d{3})(?:/\d{2,4})?(?!\d)"
)
# On folded text, where "º" became "o"; an unhyphenated suffix letter is never "o".
_RAW_NUMBER_RE = re.compile(
    r"(?:art(?:igo)?s?\.?\s*)?(\d{1,4})"
    r"(?:\s*[-\N{EN DASH}]\s*([a-z]{1,2})(?![a-z])|([a-np-z])(?![a-z]))?"
)


def fold(text: str) -> str:
    """Lowercase ``text`` and strip accents, keeping exactly one character per character.

    Keeping the length lets a match on the folded text be mapped back to the original.
    """
    return "".join(unicodedata.normalize("NFKD", char)[0].lower()[0] for char in text)


def _alias_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    alternatives = sorted({fold(alias) for alias in aliases}, key=len, reverse=True)
    body = "|".join(re.escape(alias) for alias in alternatives)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


_ALIAS_PATTERNS: dict[LawId, re.Pattern[str]] = {
    law_id: _alias_pattern((*info.aliases, *_EXTRA_ALIASES.get(law_id, ())))
    for law_id, info in LAWS.items()
}


def _law_mentions(folded: str) -> list[tuple[int, int, LawId]]:
    """Spans ``(start, end, law)`` of every law name or law number in ``folded``."""
    spans = [
        (match.start(), match.end(), law_id)
        for law_id, pattern in _ALIAS_PATTERNS.items()
        for match in pattern.finditer(folded)
    ]
    for match in _LAW_NUMBER_RE.finditer(folded):
        law_id = _LAW_NUMBERS.get(match.group(1).replace(".", ""))
        if law_id is not None:
            spans.append((match.start(), match.end(), law_id))
    return sorted(spans)


def detect_laws(text: str) -> list[LawId]:
    """Laws named in ``text``, in catalog order."""
    found = {law_id for _, _, law_id in _law_mentions(fold(text))}
    return [law_id for law_id in LawId if law_id in found]


def normalize_article_number(raw: str) -> str | None:
    """Canonical article number (``"7"``, ``"59-A"``, ``"preambulo"``) or None if unreadable.

    Accepts what people and models write: ``"7º"``, ``"art. 7o"``, ``"59-a"``, ``"401A"``,
    ``"5º, XI"`` (units after the number are ignored).
    """
    folded = fold(raw).strip()
    if folded.startswith("preambulo"):
        return "preambulo"
    match = _RAW_NUMBER_RE.match(folded)
    if match is None:
        return None
    number = str(int(match.group(1)))
    suffix = match.group(2) or match.group(3)
    return f"{number}-{suffix.upper()}" if suffix else number


def _numbers(group: str) -> list[str]:
    """Article numbers of one reference, expanding short ranges ("arts. 7º a 9º")."""
    numbers: list[str] = []
    previous: re.Match[str] | None = None
    for token in _NUMBER_RE.finditer(group):
        number = normalize_article_number(token.group())
        if number is None:
            continue
        is_range = previous is not None and group[previous.end() : token.start()].strip() == "a"
        if is_range and numbers and numbers[-1].isdigit() and number.isdigit():
            start, end = int(numbers[-1]), int(number)
            if start < end <= start + _MAX_RANGE:
                numbers.extend(str(n) for n in range(start + 1, end))
        numbers.append(number)
        previous = token
    return numbers


def _law_after(folded: str, position: int, starts: dict[int, LawId]) -> LawId | None:
    match = _LAW_AFTER_RE.match(folded, position)
    return starts.get(match.end()) if match else None


def _law_before(folded: str, position: int, mentions: list[tuple[int, int, LawId]]) -> LawId | None:
    for _, end, law_id in reversed(mentions):
        if end <= position:
            gap = folded[end:position]
            return law_id if len(gap) <= 3 and not gap.strip(" ,:") else None
    return None


def find_article_mentions(text: str) -> list[ArticleMention]:
    """Explicit article references in ``text``, in order and without repeats."""
    folded = fold(text)
    mentions = _law_mentions(folded)
    starts = {start: law_id for start, _, law_id in mentions}
    found: list[ArticleMention] = []
    for match in _ARTICLES_RE.finditer(text):
        law = _law_after(folded, match.end(), starts) or _law_before(
            folded, match.start(), mentions
        )
        found.extend(ArticleMention(law, number) for number in _numbers(match.group("numbers")))
    return list(dict.fromkeys(found))


def article_ids(
    mentions: Iterable[ArticleMention], *, default_law: LawId | None = None
) -> list[str]:
    """Canonical ids (``"CLT:482"``) of the mentions, giving unknown laws ``default_law``.

    Mentions whose law stays unknown are dropped, as are repeats.
    """
    ids: list[str] = []
    for mention in mentions:
        law = mention.law or default_law
        if law is None:
            continue
        article_id = make_article_id(law, mention.number)
        if ARTICLE_ID_RE.match(article_id) and article_id not in ids:
            ids.append(article_id)
    return ids
