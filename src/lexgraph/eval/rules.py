"""Deterministic checks of answers against golden items; no language model involved.

* ``must_include`` / ``must_not_include`` are regular expressions matched ignoring accents,
  letter case and runs of whitespace. A number spelled out in parentheses after its digits, as
  the statutes write it (``"2 (duas) horas"``, ``"20% (vinte por cento)"``), is reduced to the
  digits first, so patterns are written as ``"2 horas"``.
* ``must_include`` patterns are matched against the whole answer; ``must_not_include`` patterns
  against each sentence, so that ``^`` anchors a sentence and a negative look-ahead such as
  ``^(?!.*nao)`` only looks at the sentence that makes the forbidden claim.
* A response is a refusal when the agent routed it to ``no_context`` or ``out_of_scope``, or
  when an answer without citations says that it found no legal basis and makes no legal claim
  of its own (no article number, no quantity such as ``"4 anos"``).
* The cited articles are those behind the ``[n]`` markers plus the articles referenced
  explicitly in the text (``"art. 7º, XIII, da Constituição"``). They are compared with the
  expected articles (precision and recall) and with the sources that were given to the model,
  which exposes invented citations.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

from lexgraph.catalog import LAWS, LawInfo
from lexgraph.domain import LawId, make_article_id
from lexgraph.eval.golden import GoldenItem
from lexgraph.schemas import ChatResponse, Route

REFUSAL_ROUTES: frozenset[Route] = frozenset({"no_context", "out_of_scope"})

# Matched against accent-folded text, so the Portuguese words carry no diacritics here.
_REFUSAL_WORDING = re.compile(
    r"\bnao (?:encontrei|localizei|identifiquei)\b"
    r"|\bnao (?:ha|existe|consta)\b.{0,40}\b(?:acervo|fontes|trechos|base legal)\b"
    r"|\b(?:fontes|trechos|documentos)\b.{0,30}\bnao (?:tratam|abordam|mencionam|contem)\b"
    r"|\bfora do (?:escopo|acervo|ambito)\b"
    r"|\bnao (?:consigo|posso|e possivel) responder\b",
    re.IGNORECASE,
)
# A legal claim made despite refusal wording: an article number or a quantity with its unit.
_LEGAL_CLAIM = re.compile(
    r"\b(?:arts?\.?|artigos?)\s*\d"
    r"|\d\s*(?:%|(?:dias?|semanas?|meses|mes|anos?|horas?|minutos)\b)|R\$",
    re.IGNORECASE,
)

_NUMBER_WORD = (
    "um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|quatorze"
    "|catorze|quinze|dezesseis|dezessete|dezoito|dezenove|vinte|trinta|quarenta|cinquenta"
    "|sessenta|setenta|oitenta|noventa|cem|cento|duzent[oa]s|trezent[oa]s|quatrocent[oa]s"
    "|quinhent[oa]s|seiscent[oa]s|setecent[oa]s|oitocent[oa]s|novecent[oa]s|mil|milh(?:ao|oes)"
    "|bilh(?:ao|oes)"
)
# "2 (duas)", "20 % (vinte por cento)", "50.000.000,00 (cinquenta milhoes de reais)".
_SPELLED_NUMBER = re.compile(
    rf"(\d(?: ?%)?) ?\((?:{_NUMBER_WORD})(?: (?:e|de|por|reais|centavos|{_NUMBER_WORD}))*\)",
    re.IGNORECASE,
)
# After ".", "!", "?" or ";" (but not inside "art. 52" or "1.000") and at line breaks.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])(?!\s*\d)|\n")

_NUMBER = r"\d{1,4}[o°]?(?:-[A-Z]{1,2}(?![A-Z]))?"
_ARTICLES = re.compile(
    rf"\b(?:arts?\.?|artigos?)\s*({_NUMBER}(?:\s*(?:,|e)\s*{_NUMBER}(?![\dIVXLC]))*)",
    re.IGNORECASE,
)
_NUMBER_PARTS = re.compile(r"(\d+)[o°]?(?:-([A-Z]{1,2}))?", re.IGNORECASE)
# Units between an article number and the law named after it: "art. 10, II, b, do ADCT".
_QUALIFIERS = re.compile(
    r"(?:\s*,?\s*(?:caput|(?-i:[IVXLC]+)|§+\s*\d+[o°]?(?:\s*e\s*\d+[o°]?)?|paragrafo unico"
    r"|inc(?:isos?)?\.?\s*(?-i:[IVXLC]+)|(?:alineas?\s*)?(?-i:\"?[a-z]\"?)(?![a-z])))*"
    r"\s*,?\s*d[aoe]s?\s+",
    re.IGNORECASE,
)
_ARTICLE_WORD = re.compile(r"\b(?:arts?|artigos?)\b", re.IGNORECASE)
_ACRONYM_MAX_LENGTH = 5
_BACK_WINDOW = 25
_NEAR_WINDOW = 60


class CitationScores(BaseModel):
    """Citation quality of one answer; ``None`` where the ratio is undefined."""

    model_config = ConfigDict(frozen=True)

    precision: float | None
    recall: float | None
    in_context: float | None


class RuleScores(BaseModel):
    """Rule-based scores of one answer."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    refused: bool
    missing_patterns: list[str]
    forbidden_patterns: list[str]
    must_include_passed: bool | None
    must_not_include_passed: bool | None
    cited: list[str]
    citations: CitationScores


class RefusalStats(BaseModel):
    """How well the agent refuses out-of-scope questions without refusing in-scope ones."""

    model_config = ConfigDict(frozen=True)

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    recall: float | None
    false_refusal_rate: float | None


def fold(text: str) -> str:
    """Strip accents and compatibility forms: ``"Proteção, art. 7º"`` -> ``"Protecao, art. 7o"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _searchable(text: str) -> str:
    """``text`` folded, on one line, with spelled-out numbers reduced to their digits."""
    return _SPELLED_NUMBER.sub(r"\1", " ".join(fold(text).split()))


def sentences(text: str) -> list[str]:
    """The sentences of ``text``, each prepared for matching like a whole answer.

    Sentences end after ``.``, ``!``, ``?`` or ``;`` (unless a digit follows, as in
    ``"art. 52"`` or ``"R$ 1.000"``) and at line breaks, so list items stand on their own.
    """
    return [sentence for part in _SENTENCE_BREAK.split(text) if (sentence := _searchable(part))]


@lru_cache(maxsize=1024)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(fold(pattern), re.IGNORECASE)


def matches(pattern: str, text: str) -> bool:
    """Whether ``pattern`` occurs in ``text``, ignoring accents, case and extra whitespace."""
    return _compile(pattern).search(_searchable(text)) is not None


def missing_patterns(patterns: Iterable[str], text: str) -> list[str]:
    """Patterns that do not occur in ``text``."""
    return [pattern for pattern in patterns if not matches(pattern, text)]


def found_patterns(patterns: Iterable[str], text: str) -> list[str]:
    """Patterns that occur in ``text``."""
    return [pattern for pattern in patterns if matches(pattern, text)]


def forbidden_claims(patterns: Iterable[str], text: str) -> list[str]:
    """``must_not_include`` patterns that occur in some sentence of ``text``."""
    parts = sentences(text)
    return [
        pattern
        for pattern in patterns
        if any(_compile(pattern).search(sentence) for sentence in parts)
    ]


def check_reference(item: GoldenItem) -> list[str]:
    """Problems of an item whose reference answer fails its own pattern checks."""
    problems = [
        f"{item.id}: must_include {pattern!r} does not match the reference answer"
        for pattern in missing_patterns(item.must_include, item.reference_answer)
    ]
    problems.extend(
        f"{item.id}: must_not_include {pattern!r} matches the reference answer"
        for pattern in forbidden_claims(item.must_not_include, item.reference_answer)
    )
    return problems


def is_refusal(response: ChatResponse) -> bool:
    """Whether the agent declined to answer (see the module docstring for the rule)."""
    if response.route in REFUSAL_ROUTES:
        return True
    if response.citations:
        return False
    text = _searchable(response.answer)
    return _REFUSAL_WORDING.search(text) is not None and _LEGAL_CLAIM.search(text) is None


def _alias_pattern(alias: str) -> str:
    folded = fold(alias)
    if " " in folded or len(folded) > _ACRONYM_MAX_LENGTH:
        return rf"\b{re.escape(folded)}\b"
    # Acronyms (CLT, LGPD, CF/88) only count in upper case: "cf." is not the Constitution.
    return rf"(?-i:\b{re.escape(folded.upper())}\b)"


def _law_pattern(law: LawInfo) -> re.Pattern[str]:
    # The full name first, so "Ato das Disposições..." is found where the name starts.
    alternatives = [_alias_pattern(alias) for alias in (law.name, *law.aliases)]
    if number := re.search(r"n[º°]\s*([\d.]*\d)", law.official_name):
        digits = re.escape(number.group(1)).replace(r"\.", r"\.?")
        alternatives.append(rf"\b(?:lei|decreto-lei)\s+(?:n[o°.]*\s*)?{digits}\b")
    return re.compile("|".join(alternatives), re.IGNORECASE)


_LAW_PATTERNS = [(law.law_id, _law_pattern(law)) for law in LAWS.values()]


def extract_article_refs(text: str) -> set[str]:
    """Article ids referenced explicitly in ``text``, such as ``"CF:7"`` or ``"CLT:611-A"``.

    The law of a reference is the first that applies: the law named right after the article
    and its units (``"art. 7º, XIII, da Constituição"``), the law named right before it
    (``"CDC, art. 49"``), the nearest law mentioned within 60 characters, or the last law
    mentioned earlier in the text. References whose law cannot be told are ignored.
    """
    folded = fold(text)
    mentions = sorted(
        (match.start(), law_id)
        for law_id, pattern in _LAW_PATTERNS
        for match in pattern.finditer(folded)
    )
    refs: set[str] = set()
    for match in _ARTICLES.finditer(folded):
        law = _resolve_law(folded, match, mentions)
        if law is not None:
            refs.update(make_article_id(law, number) for number in _numbers(match.group(1)))
    return refs


def _resolve_law(
    text: str, match: re.Match[str], mentions: Sequence[tuple[int, LawId]]
) -> LawId | None:
    qualifiers = _QUALIFIERS.match(text, match.end())
    if qualifiers is not None:
        for position, law in mentions:
            if position == qualifiers.end():
                return law
    start = match.start()
    before = [m for m in mentions if start - _BACK_WINDOW <= m[0] < start]
    if before and not _ARTICLE_WORD.search(text, before[-1][0], start):
        return before[-1][1]
    near = [(abs(pos - start), law) for pos, law in mentions if abs(pos - start) <= _NEAR_WINDOW]
    if near:
        return min(near)[1]
    earlier = [law for position, law in mentions if position < start]
    return earlier[-1] if earlier else None


def _numbers(raw: str) -> list[str]:
    return [
        f"{int(digits)}-{suffix.upper()}" if suffix else str(int(digits))
        for digits, suffix in _NUMBER_PARTS.findall(raw)
    ]


def cited_articles(response: ChatResponse) -> set[str]:
    """Articles cited by an answer: resolved ``[n]`` markers plus explicit references."""
    markers = {citation.article_id for citation in response.citations}
    return markers | extract_article_refs(response.answer)


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def citation_scores(
    cited: AbstractSet[str],
    *,
    expected: AbstractSet[str],
    acceptable: AbstractSet[str],
    context: AbstractSet[str],
) -> CitationScores:
    """Precision against expected or acceptable articles, recall of the expected ones and
    the share of citations that point to a source the model actually received."""
    return CitationScores(
        precision=_ratio(len(cited & (expected | acceptable)), len(cited)),
        recall=_ratio(len(cited & expected), len(expected)),
        in_context=_ratio(len(cited & context), len(cited)),
    )


def score_answer(item: GoldenItem, response: ChatResponse) -> RuleScores:
    """Apply every rule-based check of ``item`` to ``response``."""
    missing = missing_patterns(item.must_include, response.answer)
    forbidden = forbidden_claims(item.must_not_include, response.answer)
    cited = cited_articles(response)
    return RuleScores(
        item_id=item.id,
        refused=is_refusal(response),
        missing_patterns=missing,
        forbidden_patterns=forbidden,
        must_include_passed=not missing if item.must_include else None,
        must_not_include_passed=not forbidden if item.must_not_include else None,
        cited=sorted(cited),
        citations=citation_scores(
            cited,
            expected=set(item.expected_articles),
            acceptable=set(item.acceptable_articles),
            context={source.article_id for source in response.sources},
        ),
    )


def refusal_stats(outcomes: Iterable[tuple[bool, bool]]) -> RefusalStats:
    """Confusion counts and rates from ``(should_refuse, refused)`` pairs."""
    counts = Counter(outcomes)
    tp, fn = counts[True, True], counts[True, False]
    fp, tn = counts[False, True], counts[False, False]
    return RefusalStats(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        false_refusal_rate=_ratio(fp, fp + tn),
    )
