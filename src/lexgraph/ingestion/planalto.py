"""Parse the compiled law pages of planalto.gov.br into :class:`~lexgraph.domain.Article` records.

The pages are hand-maintained HTML: windows-1252 without a declared charset, struck-through
former wording kept next to the current text, amendment notes after almost every unit, quoted
articles of other laws, signatures and annexes. :func:`parse_page` turns a page into articles in
four steps:

1. :func:`decode_page` decodes the bytes and drops premature ``</body>``/``</html>`` tags;
2. :func:`linearize` flattens the DOM into logical lines, keeping anchors, links and struck
   text as inline markers;
3. a line-level state machine skips the page header, tracks headings, opens an article at each
   label, attaches its units, removes amendment notes and stops at the signature or an annex
   (on the Constitution page it then switches to the ADCT);
4. every finished article gets its status (in force, revoked or vetoed) and a deep link.

Each tolerance below exists because of a defect observed on the real pages; the unit tests cut
a fixture from the pages for every one of them. Unit kinds keep their Brazilian legal names:
``paragraph`` (§), ``inciso`` (roman numeral), ``alinea`` (letter) and ``item`` (number).
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from urllib.parse import quote

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PreformattedString, Tag

from lexgraph.catalog import SourcePage
from lexgraph.domain import Article, ArticleStatus, LawId
from lexgraph.ingestion.fetch import FetchMeta

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.1.1"

# Inline markers written by the linearizer (control characters never present in the pages).
ANCHOR_START, ANCHOR_END = "\x01", "\x02"  # <a name="...">
LINK_START, LINK_END = "\x03", "\x04"  # text of <a href="...">
STRUCK_START, STRUCK_END = "\x05", "\x06"  # struck-through text
# Hyphen, en dash and em dash: the pages use all three as label separators.
DASHES = "-\u2013\u2014"


# --------------------------------------------------------------------------------------------
# Page profiles
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PageProfile:
    """Where the law text is on one source page."""

    start: re.Pattern[str]
    """Line after which the law text begins (enacting clause or title)."""
    keep_preamble: bool = False
    """Store the text between ``start`` and the first heading as article ``preambulo``."""
    second_law: tuple[re.Pattern[str], LawId] | None = None
    """Heading that starts a second law on the same page (the ADCT, after the Constitution)."""


_ENACTING_CLAUSE_RE = re.compile(r"sanciono a seguinte lei", re.IGNORECASE)

PROFILES: dict[str, PageProfile] = {
    # The decree that approved the CLT has two articles and signatures of its own first.
    "clt": PageProfile(start=re.compile(r"^CONSOLIDAÇÃO DAS LEIS DO TRABALHO$", re.IGNORECASE)),
    "cdc": PageProfile(start=_ENACTING_CLAUSE_RE),
    "lgpd": PageProfile(start=_ENACTING_CLAUSE_RE),
    "marco-civil": PageProfile(start=_ENACTING_CLAUSE_RE),
    "cf88": PageProfile(
        start=re.compile(r"^PREÂMBULO$", re.IGNORECASE),
        keep_preamble=True,
        second_law=(re.compile(r"^ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS$"), LawId.ADCT),
    ),
}

# Base-page anchor prefix used when an article has no anchor of its own (CDC/LGPD: "#art49").
_FALLBACK_ANCHOR_PREFIX = {LawId.ADCT: "adctart"}


# --------------------------------------------------------------------------------------------
# Decoding and linearization
# --------------------------------------------------------------------------------------------
_DOCUMENT_END_TAGS_RE = re.compile(r"</(?:html|body)\s*>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_CELL_SEPARATORS_RE = re.compile(r"^(?:\|\s*)+")
_ARTICLE_MENTION_RE = re.compile(r"\bArt\.\s*\d")
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "blockquote", "li", "ul", "ol", "table", "tbody", "thead", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6", "center", "header", "footer", "section", "article",
        "dl", "dt", "dd", "pre", "hr", "form",
    }
)  # fmt: skip
_SKIPPED_TAGS = frozenset(
    {"script", "style", "head", "noscript", "title", "meta", "link", "input", "button", "select"}
)
_STRUCK_TAGS = frozenset({"strike", "s", "del"})
_ORDINAL_SUPERSCRIPTS = {"o": "º", "º": "º", "°": "º", "a": "ª", "ª": "ª"}
_INLINE_CELL_MAX_CHARS = 300
# lxml decodes "&#150;" to the C1 control U+0096, while browsers map numeric references 128-159
# through windows-1252 (U+2013 EN DASH); translate them back.
_C1_TO_CP1252 = {
    code: bytes([code]).decode("cp1252", errors="ignore") or " " for code in range(0x80, 0xA0)
}


def decode_page(html: bytes) -> str:
    """Decode a page as windows-1252 and drop every ``</body>`` and ``</html>`` tag.

    The pages declare no charset (the HTTP header is a bare ``text/html``) yet are windows-1252.
    Some carry a premature ``</body></html>`` mid-text, after which lxml discards the rest of
    the document. Undecodable bytes become U+FFFD, which the validator reports.
    """
    return _DOCUMENT_END_TAGS_RE.sub("", html.decode("cp1252", errors="replace"))


def linearize(html: str) -> list[str]:
    """Flatten a decoded page into its non-empty logical lines.

    Block elements and ``<br>`` delimit lines; small table cells stay on one line, separated by
    ``" | "``. Anchors, link texts and struck-through text are wrapped in the module's marker
    characters; ``<sup>o</sup>`` becomes ``º``.
    """
    soup = BeautifulSoup(html, "lxml")
    text = _Linearizer().run(soup.body or soup).translate(_C1_TO_CP1252)
    lines = (_WHITESPACE_RE.sub(" ", raw).strip() for raw in text.split("\n"))
    cleaned = (_LEADING_CELL_SEPARATORS_RE.sub("", line).strip() for line in lines)
    return [line for line in cleaned if line]


class _Linearizer:
    """Depth-first DOM walk that emits text with line breaks at block boundaries."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._inline_depth = 0  # > 0 inside a small table cell, where blocks become spaces

    def run(self, root: Tag) -> str:
        self._walk(root)
        return "".join(self._parts)

    def _walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Tag):
                self._element(child)
            elif isinstance(child, NavigableString) and not isinstance(child, PreformattedString):
                self._parts.append(_WHITESPACE_RE.sub(" ", str(child)))

    def _element(self, tag: Tag) -> None:
        name = tag.name
        if name in _SKIPPED_TAGS:
            return
        if _is_struck(tag):
            text = _WHITESPACE_RE.sub(" ", tag.get_text(" ")).strip()
            self._parts.append(f" {STRUCK_START}{text}{STRUCK_END} " if text else " ")
        elif name == "br":
            self._parts.append(" " if self._inline_depth else "\n")
        elif name in ("td", "th") and _is_inline_cell(tag):
            self._parts.append(" | ")
            self._inline_depth += 1
            self._walk(tag)
            self._inline_depth -= 1
        elif self._inline_depth and name in _BLOCK_TAGS:
            self._parts.append(" ")
            self._walk(tag)
            self._parts.append(" ")
        elif name == "sup" and (sign := _ORDINAL_SUPERSCRIPTS.get(tag.get_text().strip())):
            self._parts.append(sign)
        else:
            self._container(tag)

    def _container(self, tag: Tag) -> None:
        if tag.name == "a" and (anchor := _attribute(tag, "name")):
            self._parts.append(f"{ANCHOR_START}{anchor}{ANCHOR_END}")
        is_link = tag.name == "a" and bool(_attribute(tag, "href"))
        boundary = "\n" if tag.name in _BLOCK_TAGS else ""
        self._parts.append((LINK_START if is_link else "") + boundary)
        self._walk(tag)
        self._parts.append(boundary + (LINK_END if is_link else ""))


def _attribute(tag: Tag, name: str) -> str:
    value = tag.get(name)
    if value is None:
        return ""
    return value if isinstance(value, str) else " ".join(value)


def _is_struck(tag: Tag) -> bool:
    return tag.name in _STRUCK_TAGS or "line-through" in _attribute(tag, "style").lower()


def _is_inline_cell(cell: Tag) -> bool:
    text = cell.get_text(" ")
    return len(text) <= _INLINE_CELL_MAX_CHARS and not _ARTICLE_MENTION_RE.search(text)


# --------------------------------------------------------------------------------------------
# Line grammar
# --------------------------------------------------------------------------------------------
_ORD = r"(?:[º°ª]|o(?![a-zà-ú]))"

ARTICLE_RE = re.compile(
    # Tolerates real typos: "Art. . 154" (CLT), "Art.184", "Art 15.", "Art. 401A." (suffix
    # without hyphen) and "Art. 5 7." (LGPD art. 57 split across two <span>s). In "Art. 1º A
    # República" the "A" is the Portuguese article, so a suffix must touch the number or follow
    # a hyphen.
    r"^Art(?:igo)?\s*\.?\s*\.?\s*(?P<num>\d{1,3}(?:\.\d{3})+|\d(?:\s?\d)*)\s*\.?\s*"
    + _ORD
    + r"?\s*\.?(?:(?:\s?-|(?<=\d)|(?<=[º°ª]))(?P<suf>[A-Z]{1,2})(?![A-Za-zà-ú]))?\s*[.\-\u2013\u2014:]?\s*"
)
PARAGRAPH_RE = re.compile(
    # The ordinal sign is case-sensitive: in "§ 2 O fornecedor" (CDC, struck "º") the "O"
    # starts the sentence.
    r"^(?:§\s*(?P<num>\d+)(?:\s*\.?\s*[º°ª]|(?-i:o)(?![a-zà-ú]))?(?:-(?P<suf>[A-Z]{1,2}))?"
    r"|(?P<unico>Par[áa]grafo\s+[úu]nico))\s*[.\-\u2013\u2014:]?\s*",
    re.IGNORECASE,
)
# "Il -" and "VlIl -" (lowercase L for I) are real typos on the CLT page. The hyphen of a
# suffix is never the separator: "V-A as causas" is inciso V-A without its dash, not inciso V.
INCISO_RE = re.compile(
    r"^(?P<num>(?=[IVXLCDMl]*[IVXLCDM])[IVXLCDMl]+)(?:-(?P<suf>[A-Z]))?"
    r"\s*(?!-[A-Z]\b)[-\u2013\u2014]\s*"
)
# The Constitution page has incisos whose dash is missing ("XV a distribuição de processos",
# "V-A as causas relativas a direitos humanos", "I-A o Conselho Nacional de Justiça").
INCISO_NO_DASH_RE = re.compile(
    r"^(?P<num>(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))(?:-(?P<suf>[A-Z]))?"
    r"\s+(?=[a-zà-ú])"
)
PARAGRAPH_NO_SIGN_RE = re.compile(r"^(?P<num>\d{1,2})\s*[º°]\s+(?=[A-ZÀ-Ý])")  # CF 215 "2º A lei"
MULTI_PARAGRAPH_RE = re.compile(r"^§§\s*\d+\s*[º°]?\s*(?:a|e|,)\s*\d+\s*[º°]?\s*\.?\s*$")
ALINEA_RE = re.compile(r"^(?P<num>[a-z])(?:-(?P<suf>[a-z]))?\)\s*")
ITEM_RE = re.compile(r"^(?P<num>\d{1,2})[.)]\s+")
HEADING_RE = re.compile(
    r"^(?P<kind>PARTE|LIVRO|T[ÍI]TULO|CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O)\s+"
    r"(?P<num>[IVXLCDM]+(?:-[A-Z])?|[ÚU]NIC[OA]|GERAL|ESPECIAL|PRELIMINAR)\b\.?\s*[-\u2013\u2014:]?\s*"
    r"(?P<title>.*)$",
    re.IGNORECASE,
)
SIGNATURE_RE = re.compile(
    r"^(?:Bras[íi]lia|Rio de Janeiro)\s*,\s*(?:em\s+)?\d{1,2}"
    + _ORD
    + r"?\s+de\s+[A-Za-zçÇ]+\s+de\s+\d{4}"
)
ANNEX_RE = re.compile(r"^ANEXOS?\b(?:\s+[IVX]+)?$")
# Article epigraph printed in bold before some CLT articles ("Trabalho aos domingos").
_RUBRIC_RE = re.compile(r"^(?=[A-ZÀ-Ý])[^|\d§]{3,120}(?<![.;:,])$")

_HEADING_LEVELS = {"PARTE": 0, "LIVRO": 1, "TITULO": 2, "CAPITULO": 3, "SECAO": 4, "SUBSECAO": 5}
_UNLABELED_HEADING_LEVEL = 6  # "DISPOSIÇÕES GERAIS" without a TÍTULO/CAPÍTULO label
_SECOND_LAW_HEADING_LEVEL = -1  # the ADCT title stays above every heading of its own
_HEADING_FOLD = str.maketrans("ÍÇÃ", "ICA")

UnitKind = Literal["paragraph", "inciso", "alinea", "item", "text"]
_UNIT_KINDS: frozenset[UnitKind] = frozenset({"paragraph", "inciso", "alinea", "item"})


class UnitLabel(NamedTuple):
    """Classification of one article line: unit kind, normalized label and the text after it."""

    kind: UnitKind
    label: str | None
    body: str


class ArticleLabel(NamedTuple):
    """Article heading line split into its normalized parts."""

    number: str  # "1", "59-A", "1000"
    label: str  # "Art. 1º", "Art. 59-A", "Art. 1.000"
    separator: str  # separator printed after the label, normalized: " ", ". ", " - " or ": "
    body: str


def ordinal_label(prefix: str, number: int) -> str:
    """Label in the official style: ordinal up to 9 (``§ 9º``), cardinal from 10 (``§ 10``)."""
    shown = f"{number:,}".replace(",", ".")
    return f"{prefix} {shown}{'º' if number < 10 else ''}"


def match_article_label(line: str) -> ArticleLabel | None:
    """Parse an article heading line such as ``"Art. . 154 - A observância..."``."""
    match = ARTICLE_RE.match(line)
    if match is None:
        return None
    number = int(re.sub(r"[.\s]", "", match["num"]))
    suffix = f"-{match['suf']}" if match["suf"] else ""
    return ArticleLabel(
        number=f"{number}{suffix}",
        label=ordinal_label("Art.", number) + suffix,
        separator=_separator(line[: match.end()]),
        body=line[match.end() :].strip(),
    )


def classify_line(line: str) -> UnitLabel:
    """Classify a cleaned article line as a §, inciso, alínea, item or plain text."""
    if match := PARAGRAPH_RE.match(line):
        if match["unico"]:
            label = "Parágrafo único"
        else:
            suffix = f"-{match['suf'].upper()}" if match["suf"] else ""
            label = ordinal_label("§", int(match["num"])) + suffix
        return UnitLabel("paragraph", label, line[match.end() :])
    if match := INCISO_RE.match(line):
        suffix = f"-{match['suf']}" if match["suf"] else ""
        return UnitLabel("inciso", match["num"].replace("l", "I") + suffix, line[match.end() :])
    if match := PARAGRAPH_NO_SIGN_RE.match(line):
        return UnitLabel("paragraph", ordinal_label("§", int(match["num"])), line[match.end() :])
    if (match := INCISO_NO_DASH_RE.match(line)) and match["num"]:
        suffix = f"-{match['suf']}" if match["suf"] else ""
        return UnitLabel("inciso", match["num"] + suffix, line[match.end() :])
    if MULTI_PARAGRAPH_RE.match(line):
        return UnitLabel("paragraph", line.rstrip(". "), "")
    if match := ALINEA_RE.match(line):
        suffix = f"-{match['suf']}" if match["suf"] else ""
        return UnitLabel("alinea", match["num"] + suffix, line[match.end() :])
    if match := ITEM_RE.match(line):
        return UnitLabel("item", match["num"], line[match.end() :])
    return UnitLabel("text", None, line)


def _separator(label_prefix: str) -> str:
    """Normalized separator ending a matched label prefix ("Art. 62 - " -> " - ")."""
    match = re.search(r"([.\-\u2013\u2014:])\s*$", label_prefix)
    if match is None:
        return " "
    return {".": ". ", ":": ": "}.get(match[1], " - ")


# --------------------------------------------------------------------------------------------
# Amendment notes
# --------------------------------------------------------------------------------------------
_ANNOTATION_KEYWORDS = (
    r"Reda[çc][ãa]o\s+(?:dada|restabelecida|original|anterior)|Com\s+(?:a\s+)?reda[çc][ãa]o|"
    r"Inclu[íi]d[oa]s?|Par[áa]grafo\s+(?:[úu]nico\s+)?inclu[íi]do|Inciso\s+inclu[íi]do|"
    r"Al[íi]nea\s+inclu[íi]da|Acrescid[oa]|Acrescentad[oa]|Revogad[oa]s?|Vide|Vig[êe]ncia|"
    r"Em\s+vigor|Produ[çc][ãa]o\s+de\s+efeitos?|Regulamento|Regulamentad[oa]|Renumerad[oa]|"
    r"Restaurad[oa]|Restabelecid[oa]|Suprimid[oa]|Rejeitad[oa]|Promulga[çc][ãa]o|Declarad[oa]|"
    r"Express[ãa]o|Execu[çc][ãa]o\s+suspensa|Suspens[oa]|Mantid[oa]|Transformad[oa]|"
    r"Convertid[oa]|Retificad[oa]|Alterad[oa]|Modificad[oa]|Inclus[ãa]o|NR|AC|"
    r"Texto\s+compilado|Vetad[oa]\s+(?:pel|o\s)|Vide\s+ADIN?|ADIN?|ADPF|ADC"
)
_LINK_PREFIX = f"[{LINK_START}{LINK_END}\\s]*"
# A parenthetical whose text starts with a keyword (never a generic one: "(trinta)" stays);
# one level of nested parentheses is allowed inside.
_ANNOTATION_RE = re.compile(
    rf"\(\s*{_LINK_PREFIX}((?:{_ANNOTATION_KEYWORDS})\b[^()]*(?:\([^()]*\)[^()]*)*)\)",
    re.IGNORECASE,
)
# The same with the closing parenthesis missing at the end of the line (CLT art. 139).
_TRAILING_ANNOTATION_RE = re.compile(
    rf"\(\s*{_LINK_PREFIX}((?:{_ANNOTATION_KEYWORDS})\b[^()]*)$", re.IGNORECASE
)
# A link whose whole text is a note, without parentheses: "Vigência", "Vide art. 96 - ADCT".
_BARE_ANNOTATION_LINK_RE = re.compile(
    LINK_START
    + r"\s*((?:Vig[êe]ncia|Regulamento|Vide\b|Produ[çc][ãa]o\s+de\s+efeito|Mensagem\s+de\s+veto|"
    r"Texto\s+compilado|Convers[ãa]o|Reda[çc][ãa]o\s+dada|Inclu[íi]d[oa]|Revogad[oa])"
    rf"[^{LINK_END}]*){LINK_END}",
    re.IGNORECASE,
)
# The CLT page splits "(Vide ADC 80)" into two links: <a>(V</a><a>ide ADC 80)</a>.
_ADJACENT_LINKS_RE = re.compile(rf"{LINK_END}(\s*){LINK_START}")
_LINK_MARKERS_RE = re.compile(f"[{LINK_START}{LINK_END}]")


def strip_annotations(line: str) -> tuple[str, list[str]]:
    """Remove amendment notes ("(Redação dada pela Lei nº ...)", "Vigência") from one line.

    Returns the cleaned line and the removed notes, in order of removal.
    """
    notes: list[str] = []

    def keep_note(match: re.Match[str]) -> str:
        notes.append(_clean_text(match[1]))
        return " "

    text = _ADJACENT_LINKS_RE.sub(r"\1", line)
    previous = None
    while previous != text:  # chains such as "(Redação dada ...) (Vigência)"
        previous = text
        text = _ANNOTATION_RE.sub(keep_note, text)
        text = _BARE_ANNOTATION_LINK_RE.sub(keep_note, text)
    text = _TRAILING_ANNOTATION_RE.sub(keep_note, text)
    return _clean_text(text), notes


def _clean_text(text: str) -> str:
    text = _LINK_MARKERS_RE.sub("", text)
    text = re.sub(r"\.{4,}", "…", text)  # dotted leaders in quoted amendments
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)  # left behind when struck words are cut out
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# --------------------------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------------------------
_ANCHOR_RE = re.compile(f"{ANCHOR_START}([^{ANCHOR_END}]*){ANCHOR_END}")
_STRUCK_RE = re.compile(f"{STRUCK_START}([^{STRUCK_END}]*){STRUCK_END}")
_VETOED_BODY_RE = re.compile(r"^\(?\s*VETAD[OA]\s*\)?\s*[.;,]?\s*(?:e|ou)?\s*$", re.IGNORECASE)
_REVOKED_BODY_RE = re.compile(r"^\(?\s*Revogad[oa]s?\s*\)?\s*[.;,]?\s*(?:e|ou)?\s*$", re.IGNORECASE)
# What remains of an article line whose note revoked the article together with the units it
# lists: "Art. 117. e Parágrafo único. (Revogados pela Emenda ...)" keeps "e Parágrafo único.".
UNITS_ONLY_BODY_RE = re.compile(
    r"^(?:(?:e|ou)\s+)?(?P<units>Par[áa]grafo\s+[úu]nico|§§?\s*\d[\dº°ª ,ea]*?)\s*\.?$",
    re.IGNORECASE,
)
_ARTICLE_GONE_NOTE_RE = re.compile(r"Revogad|Suprimid|Rejeitad", re.IGNORECASE)
_UNIT_GONE_NOTE_RE = re.compile(r"Revogad|Suprimid", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.;:!?]\W*$")
_QUOTE_CHARS = '“”"«»'
_OPENING_QUOTES = '“"«'
_MAX_QUOTED_LINES = 60  # safety valve if a closing quote is lost


class _Line(NamedTuple):
    """A linearized line with its markers separated out."""

    text: str  # still holds link markers; anchors and struck text removed
    plain: str  # text without any marker
    anchors: list[str]
    struck: list[str]

    @classmethod
    def split(cls, raw: str) -> "_Line":
        struck = [text.strip() for text in _STRUCK_RE.findall(raw) if text.strip()]
        text = _STRUCK_RE.sub(" ", _ANCHOR_RE.sub("", raw))
        return cls(text, _clean_text(text), _ANCHOR_RE.findall(raw), struck)


@dataclass(eq=False)
class _Draft:
    """An article whose lines are still being collected."""

    law_id: LawId
    number: str
    label: str
    heading_path: list[str]
    lines: list[str]
    rubric: str | None = None
    anchors: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    status: ArticleStatus = "vigente"


class _PageParser:
    """Line-by-line state machine that assembles the articles of one page."""

    def __init__(self, profile: PageProfile, law_id: LawId) -> None:
        self._profile = profile
        self._law_id = law_id
        self._started = False
        self._stopped = False  # after a signature or an annex, until a second law begins
        self._preamble: list[str] | None = None  # collecting while not None
        self._headings: dict[int, str] = {}
        self._untitled_heading: int | None = None  # heading level whose title is on the next line
        self._pending_anchors: list[str] = []
        self._pending_text: list[str] = []  # continuation of the article, or the next rubric
        self._in_quote = False
        self._quoted_lines = 0
        self._current: _Draft | None = None
        self._drafts: list[_Draft] = []
        self.stats: Counter[str] = Counter()

    def feed(self, raw: str) -> None:
        """Consume one linearized line."""
        line = _Line.split(raw)
        if not line.plain:
            self._on_markup_only(line)
        elif not self._started:
            self._try_start(line.plain)
        elif (second_law := self._second_law_starting_at(line.plain)) is not None:
            self._switch_law(second_law, line.plain)
        elif self._stopped:
            return
        elif SIGNATURE_RE.match(line.plain) or ANNEX_RE.match(line.plain):
            self.stats["stop:" + ("signature" if SIGNATURE_RE.match(line.plain) else "annex")] += 1
            self._close()
            self._stopped = True
        else:
            self._on_text(line)

    def finish(self) -> list[_Draft]:
        """Close the last article and return every article in document order."""
        self._close()
        if self._preamble:
            self.stats["preamble_without_body"] += 1
        return self._drafts

    # -- page structure ----------------------------------------------------------------------
    def _on_markup_only(self, line: _Line) -> None:
        if line.struck and self._current is not None:
            self._current.removed += line.struck
            self._current.anchors += line.anchors  # a struck block's anchor must not leak forward
        else:
            self._pending_anchors += line.anchors

    def _try_start(self, plain: str) -> None:
        if self._profile.start.search(plain):
            self._started = True
            self._pending_anchors = []
            self._preamble = [] if self._profile.keep_preamble else None

    def _second_law_starting_at(self, plain: str) -> LawId | None:
        second = self._profile.second_law
        if second is not None and second[0].match(plain):
            return second[1]
        return None

    def _switch_law(self, law_id: LawId, title: str) -> None:
        self._close()
        self.stats["second_law"] += 1
        self._stopped = False
        self._law_id = law_id
        self._headings = {_SECOND_LAW_HEADING_LEVEL: title}
        self._untitled_heading = None
        self._pending_anchors = []

    # -- text lines --------------------------------------------------------------------------
    def _on_text(self, line: _Line) -> None:
        clean, notes = strip_annotations(line.text)
        if self._in_quote or (clean and clean[0] in _OPENING_QUOTES):
            self._on_quoted(clean, notes, line.struck)
        elif not clean:
            self._on_annotation_only(notes, line)
        elif not self._consumed_by_preamble(clean):
            self._on_body_line(clean, notes, line)

    def _on_body_line(self, clean: str, notes: list[str], line: _Line) -> None:
        draft = self._current
        if heading := HEADING_RE.match(clean):
            self._on_heading(heading)
        elif article := match_article_label(clean):
            self._on_article(article, notes, line)
        elif draft is None:
            self._on_text_between_articles(clean)
        else:
            self._on_article_line(draft, clean, notes, line)

    def _on_quoted(self, clean: str, notes: list[str], struck: list[str]) -> None:
        """Quoted text (an amending article quoting another law) never opens a unit."""
        self._flush_pending_text()
        if self._current is not None:
            if clean:
                self._current.lines.append(clean)
            self._current.annotations += notes
            self._current.removed += struck
        self.stats["quoted_lines"] += 1
        odd_quotes = sum(clean.count(char) for char in _QUOTE_CHARS) % 2 == 1
        if not self._in_quote:
            self._in_quote = odd_quotes  # "“Art. 7º ......" opens; a one-line quote does not
            self._quoted_lines = 0
        elif odd_quotes:
            self._in_quote = False  # '......” (NR)'
        self._quoted_lines += 1
        if self._in_quote and self._quoted_lines > _MAX_QUOTED_LINES:
            self._in_quote = False
            self.stats["quote_state_reset"] += 1

    def _on_annotation_only(self, notes: list[str], line: _Line) -> None:
        if self._current is not None:
            self._current.annotations += notes
            self._current.removed += line.struck
        self._pending_anchors += line.anchors
        self.stats["annotation_only_lines"] += 1

    def _consumed_by_preamble(self, clean: str) -> bool:
        """Collect preamble text until the first heading or article, which ends the preamble."""
        preamble = self._preamble
        if preamble is None:
            return False
        if not (HEADING_RE.match(clean) or ARTICLE_RE.match(clean)):
            preamble.append(clean)
            return True
        if preamble:
            self._drafts.append(_Draft(self._law_id, "preambulo", "Preâmbulo", [], lines=preamble))
        self._preamble = None
        return False

    def _on_heading(self, match: re.Match[str]) -> None:
        self._close()
        level = _HEADING_LEVELS[match["kind"].upper().translate(_HEADING_FOLD)]
        self._headings = {k: v for k, v in self._headings.items() if k < level}
        label = f"{match['kind'].upper()} {match['num'].upper()}"
        title = match["title"].strip()
        self._headings[level] = f"{label} - {title}" if title else label
        self._untitled_heading = None if title else level
        self.stats["headings"] += 1

    def _on_article(self, article: ArticleLabel, notes: list[str], line: _Line) -> None:
        rubric = self._take_rubric()
        self._close()
        self._current = _Draft(
            law_id=self._law_id,
            number=article.number,
            label=article.label,
            heading_path=[self._headings[level] for level in sorted(self._headings)],
            lines=[f"{article.label}{article.separator}{article.body}".strip()],
            rubric=rubric,
            anchors=[*self._pending_anchors, *line.anchors],
            annotations=notes,
            removed=list(line.struck),
        )
        self._pending_anchors = []
        self._untitled_heading = None

    def _take_rubric(self) -> str | None:
        if self._pending_text and _RUBRIC_RE.match(self._pending_text[-1]):
            self.stats["rubrics"] += 1
            return self._pending_text.pop()
        return None

    def _on_text_between_articles(self, clean: str) -> None:
        if self._untitled_heading is not None:
            self._headings[self._untitled_heading] += f" - {clean}"
            self._untitled_heading = None
        elif self._headings and clean[:1].islower() and not self._pending_text:
            self._headings[max(self._headings)] += f" {clean}"  # title broken over two lines
        elif clean.upper() == clean and len(clean) < 160:
            self._headings[_UNLABELED_HEADING_LEVEL] = clean
            self.stats["unlabeled_headings"] += 1
        else:
            self._pending_text.append(clean)

    def _on_article_line(self, draft: _Draft, clean: str, notes: list[str], line: _Line) -> None:
        draft.anchors += line.anchors
        draft.annotations += notes
        draft.removed += line.struck
        unit = classify_line(clean)
        if unit.kind in _UNIT_KINDS:
            self._flush_pending_text()
            self._add_unit(draft, unit, clean, notes)
        elif self._continues_last_line(draft, clean):
            # lxml closes the <p> early on a misnested "</font></font>" (CF: "...os diretores
            # do" / "banco central ..."): the line completes an unfinished sentence.
            draft.lines[-1] = f"{draft.lines[-1]} {clean}"
            self.stats["rejoined_lines"] += 1
        else:
            self._pending_text.append(clean)

    def _add_unit(self, draft: _Draft, unit: UnitLabel, clean: str, notes: list[str]) -> None:
        body = unit.body.strip()
        if _VETOED_BODY_RE.match(body):
            reason = "vetado"
        elif _REVOKED_BODY_RE.match(body) or not body.strip(" .;,:"):
            # "I - (revogado);" loses its note to strip_annotations and keeps only ";"
            gone = body or any(_UNIT_GONE_NOTE_RE.match(note) for note in notes)
            reason = "revogado" if gone else "sem texto"
        else:
            draft.lines.append(self._normalized_unit_line(unit, clean))
            self.stats[f"unit:{unit.kind}"] += 1
            return
        draft.removed.append(f"{unit.label} ({reason})")
        self.stats[f"dropped_unit:{reason}"] += 1

    def _normalized_unit_line(self, unit: UnitLabel, clean: str) -> str:
        """Rewrite the label of a unit line in its canonical form, keeping its text."""
        body = unit.body.strip()
        if unit.kind == "inciso":
            strict = INCISO_RE.match(clean)
            if strict is None or "l" in strict["num"]:  # missing dash or "Il -" typo
                self.stats["inciso_label_fixed"] += 1
                return f"{unit.label} - {body}"
        elif unit.kind == "paragraph" and not MULTI_PARAGRAPH_RE.match(clean):
            label_prefix = clean[: len(clean) - len(unit.body)]
            return f"{unit.label}{_separator(label_prefix)}{body}"
        return clean

    def _continues_last_line(self, draft: _Draft, clean: str) -> bool:
        last = draft.lines[-1]
        return (
            not self._pending_text
            and " | " not in clean
            and " | " not in last
            and _SENTENCE_END_RE.search(last) is None
        )

    def _flush_pending_text(self) -> None:
        if self._current is not None:
            self._current.lines += self._pending_text
            self.stats["continuation_lines"] += len(self._pending_text)
        elif self._pending_text:
            logger.debug("dropping text outside articles: %s", self._pending_text)
        self._pending_text = []

    def _close(self) -> None:
        self._flush_pending_text()
        if self._current is not None:
            _settle_status(self._current)
            self._drafts.append(self._current)
            self._current = None


def _settle_status(draft: _Draft) -> None:
    """Set the status of a finished article from its remaining text and notes."""
    caput = ARTICLE_RE.sub("", draft.lines[0], count=1).strip()
    single_line = len(draft.lines) == 1
    gone_note = any(_ARTICLE_GONE_NOTE_RE.match(note) for note in draft.annotations)
    if single_line and gone_note and (units := UNITS_ONLY_BODY_RE.match(caput)):
        draft.removed.append(f"{units['units']} (revogado)")
        caput = ""
    # "Art. 235-H. (Revogado)." is left as "Art. 235-H. ." once its note is removed.
    if not caput.strip(" .;,:" + DASHES):
        draft.lines[0] = f"{draft.label}."
        caput = ""
    if not single_line:
        return
    if _VETOED_BODY_RE.match(caput):
        draft.status = "vetado"
    elif not caput or _REVOKED_BODY_RE.match(caput):
        # Without any revocation evidence the article stays "vigente" with an empty body,
        # which the validator reports.
        draft.status = "revogado" if caput or gone_note else "vigente"


# --------------------------------------------------------------------------------------------
# Articles
# --------------------------------------------------------------------------------------------
def parse_page(html: bytes, source: SourcePage, meta: FetchMeta) -> dict[LawId, list[Article]]:
    """Parse one stored page into the articles of every law it holds, in document order.

    ``cf88`` yields the Constitution (including its ``preambulo``) and the ADCT. Deep links use
    the article's own anchor on the page served at ``meta.final_url``; articles without one
    link to ``source.anchor_base_url`` (whose anchors follow ``#artN``), else to the page.
    """
    profile = PROFILES.get(source.slug)
    if profile is None:
        msg = f"no parsing profile for source {source.slug!r}"
        raise ValueError(msg)
    parser = _PageParser(profile, source.law_ids[0])
    for line in linearize(decode_page(html)):
        parser.feed(line)
    articles: dict[LawId, list[Article]] = {law_id: [] for law_id in source.law_ids}
    for draft in parser.finish():
        law_articles = articles[draft.law_id]
        law_articles.append(_to_article(draft, len(law_articles), meta.final_url, source))
    logger.debug("%s: %s", source.slug, dict(parser.stats))
    return articles


def _to_article(draft: _Draft, seq: int, page_url: str, source: SourcePage) -> Article:
    anchor = _pick_anchor(draft.number, draft.anchors)
    if anchor is not None:
        url = f"{page_url}#{quote(anchor, safe='')}"
    elif source.anchor_base_url and draft.number != "preambulo":
        prefix = _FALLBACK_ANCHOR_PREFIX.get(draft.law_id, "art")
        url = f"{source.anchor_base_url}#{prefix}{draft.number.replace('-', '').lower()}"
    else:
        url = page_url
    return Article(
        law_id=draft.law_id,
        number=draft.number,
        label=draft.label,
        seq=seq,
        heading_path=draft.heading_path,
        rubric=draft.rubric,
        text="\n".join(draft.lines),
        status=draft.status,
        annotations=draft.annotations,
        removed=draft.removed,
        anchor=anchor,
        url=url,
    )


def _pick_anchor(number: str, anchors: list[str]) -> str | None:
    """Prefer the anchor named after the article number, else the first one in the article.

    Anchor names are not derivable from numbers (CLT art. 51 is ``art923``), so the anchor that
    physically precedes the article wins; one that encodes the number (``art482``,
    ``sadctart1``, ``...-artigo-5``) is preferred when the article has several.
    """
    if not anchors:
        return None
    base, _, suffix = number.partition("-")
    suffix = suffix.lower()
    own = re.compile(
        rf"^(?:(?:cf)?art{base}{suffix}[.\d]*|s?adctart{base}{suffix}[.\d]*"
        rf"|.*-artigo-{base}(?:-{suffix})?(?:\.\d+)?)$"
    )
    return next((anchor for anchor in anchors if own.match(anchor)), anchors[0])
