"""Split articles into retrieval chunks that follow the structure of the law.

An article up to ``max_chars`` is one chunk. A longer one is split along its units, never
inside one: first into the caput block and its paragraphs (§), then a block that is still too
long into its incisos, then an inciso into its alíneas. Consecutive slices that share the same
context are packed greedily up to ``max_chars``, and every chunk starts with the caput and the
leads of the enclosing § / inciso (truncated, and together kept to half of a packed chunk), so
a slice such as "LXXVIII - a todos ... são assegurados a razoável duração do processo" still
says which article it belongs to.

Articles that quote text of another law (amending articles such as "Art. 60. A Lei nº 12.965
... passa a vigorar com as seguintes alterações:") are never split: their quoted incisos and
paragraphs belong to the other law and must not be cited as units of the host article.
"""

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from lexgraph.catalog import law_info
from lexgraph.corpus import Corpus
from lexgraph.domain import Article, Chunk, LawId
from lexgraph.ingestion.planalto import classify_line

DEFAULT_MAX_CHARS = 1500
DEFAULT_PACK_CHARS = 700
DEFAULT_CAPUT_CHARS = 400
WHOLE_ARTICLE = "*"
_ELLIPSIS = " …"
_QUOTED_LINE_RE = re.compile(r"^[“\"«]", re.MULTILINE)
CONTROL_NOTE_RE = re.compile(
    r"\bAD(?:I|IN|C|PF|O)\b|inconstitucional|execu[çc][ãa]o\s+suspensa", re.IGNORECASE
)
"""Amendment notes that record constitutional review by the STF; they travel with every chunk."""

type _Path = tuple[str, ...]
"""Unit labels from the article down: ("caput",), ("§ 1º",), ("§ 1º", "II"), ("II", "a")."""


@dataclass(eq=False)
class _Unit:
    """A unit of an article (caput, §, inciso or alínea) with its own lines and sub-units."""

    path: _Path
    lines: list[str]
    children: list["_Unit"] = field(default_factory=list)

    @property
    def lead(self) -> str:
        return "\n".join(self.lines)

    def render(self) -> str:
        return "\n".join([self.lead, *(child.render() for child in self.children)])


@dataclass(frozen=True, eq=False)
class _Slice:
    """Consecutive text of one context: the enclosing units whose leads prefix the chunk."""

    context: tuple[_Unit, ...]
    body: str
    paths: tuple[_Path, ...]
    opens_context: bool = False  # the body starts with the full lead of context[-1]


@dataclass(frozen=True)
class _LeadBudget:
    """How much of the enclosing leads a chunk repeats: ``per_lead`` each, ``total`` in all."""

    per_lead: int
    total: int

    def prefix(self, context: Sequence[_Unit]) -> list[str]:
        """The leads of ``context``, truncated so that together they fit in ``total``.

        Leads within an equal share of what is left stay whole and the longer ones split the
        rest: under a long caput, a long § and the short lead of an inciso ("I - implementar
        programa de governança em privacidade que, no mínimo:"), the inciso keeps its lead and
        the caput and the § are cut to about the same length.
        """
        leads: dict[int, str] = {}
        left = self.total
        by_length = sorted(range(len(context)), key=lambda index: len(context[index].lead))
        for rank, index in enumerate(by_length):
            share = min(self.per_lead, left // (len(context) - rank))
            leads[index] = _truncate(context[index].lead, share)
            left -= len(leads[index])
        return [leads[index] for index in range(len(context))]


def chunk_article(
    article: Article,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    pack_chars: int = DEFAULT_PACK_CHARS,
    caput_chars: int = DEFAULT_CAPUT_CHARS,
) -> list[Chunk]:
    """Split one article into chunks of at most ``max_chars`` characters where possible.

    An article up to ``max_chars`` stays whole. A longer one is split along its units, and
    consecutive units are packed only up to ``pack_chars`` (capped at ``max_chars``): long
    lists of incisos, such as the rights of CF art. 7º, then become chunks of a few incisos
    each, whose vectors are not diluted by a dozen unrelated rights. Only a single unit
    longer than ``max_chars`` on its own, or an article quoting another law, yields a longer
    chunk. Each context lead is truncated to ``min(caput_chars, pack_chars // 3)`` characters
    and the leads together to ``pack_chars // 2``, so a chunk under a caput, a § and an inciso
    still has room for several alíneas.
    """
    if max_chars <= 0 or pack_chars <= 0 or caput_chars <= 0:
        msg = "max_chars, pack_chars and caput_chars must be positive"
        raise ValueError(msg)
    pack_chars = min(pack_chars, max_chars)
    budget = _LeadBudget(per_lead=min(caput_chars, pack_chars // 3), total=pack_chars // 2)
    groups: list[list[_Slice]] = []
    if len(article.text) > max_chars and not _QUOTED_LINE_RE.search(article.text):
        caput, paragraphs = _structure(article.text)
        groups = _pack(_slices(caput, paragraphs, max_chars, budget), pack_chars, budget)
    if len(groups) <= 1:  # short, or nothing to split along
        return [_make_chunk(article, 0, 1, None, article.text)]
    return [
        _make_chunk(
            article,
            index,
            len(groups),
            [path for piece in group for path in piece.paths],
            _render(group, budget),
        )
        for index, group in enumerate(groups)
    ]


def chunk_corpus(
    corpus: Corpus,
    *,
    law_ids: Iterable[LawId] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    pack_chars: int = DEFAULT_PACK_CHARS,
) -> list[Chunk]:
    """Chunk every article in force (status ``vigente``), optionally only of ``law_ids``."""
    return [
        chunk
        for article in corpus.in_force(law_ids)
        for chunk in chunk_article(article, max_chars=max_chars, pack_chars=pack_chars)
    ]


# --------------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------------
def _structure(text: str) -> tuple[_Unit, list[_Unit]]:
    """Build the unit tree: the caput block and the § blocks, each with incisos and alíneas.

    Lines that are not unit labels (items, tables, "Pena - ..." lines, quoted text) stay with
    the unit opened last, so the document order is preserved.
    """
    first, *rest = text.split("\n")
    caput = _Unit(("caput",), [first])
    paragraphs: list[_Unit] = []
    block, inciso, last = caput, None, caput
    for line in rest:
        kind, label, _ = classify_line(line)
        if kind == "paragraph" and label is not None:
            block = _Unit((_unit_name(label),), [line])
            paragraphs.append(block)
            inciso, last = None, block
        elif kind == "inciso" and label is not None:
            inciso = _Unit(_child_path(block, label), [line])
            block.children.append(inciso)
            last = inciso
        elif kind == "alinea" and label is not None:
            parent = inciso or block
            last = _Unit(_child_path(parent, label), [line])
            parent.children.append(last)
        else:
            last.lines.append(line)
    return caput, paragraphs


def _child_path(parent: _Unit, label: str) -> _Path:
    return (*parent.path, label) if parent.path != ("caput",) else (label,)


def _unit_name(label: str) -> str:
    return "parágrafo único" if label == "Parágrafo único" else label


# --------------------------------------------------------------------------------------------
# Slicing and packing
# --------------------------------------------------------------------------------------------
def _slices(
    caput: _Unit, paragraphs: list[_Unit], max_chars: int, budget: _LeadBudget
) -> list[_Slice]:
    root = (caput,)
    whole = caput.render()
    if len(whole) <= max_chars:
        # The caput's own incisos are cited apart from it: "caput e I a VIII", not "caput".
        paths = (caput.path, *(child.path for child in caput.children))
        slices = [_Slice(root, whole, paths, opens_context=True)]
    else:
        slices = [_Slice(root, caput.lead, (caput.path,), opens_context=True)]
        for child in caput.children:
            slices += _split(child, root, max_chars, budget)
    for paragraph in paragraphs:
        slices += _split(paragraph, root, max_chars, budget)
    return slices


def _split(
    unit: _Unit, context: tuple[_Unit, ...], max_chars: int, budget: _LeadBudget
) -> Iterator[_Slice]:
    """Yield ``unit`` whole when it fits under its context, else its lead and its sub-units."""
    whole = unit.render()
    if not unit.children or _prefix_chars(context, budget) + len(whole) <= max_chars:
        yield _Slice(context, whole, (unit.path,))
        return
    inner = (*context, unit)
    yield _Slice(inner, unit.lead, (unit.path,), opens_context=True)
    for child in unit.children:
        yield from _split(child, inner, max_chars, budget)


def _pack(slices: list[_Slice], max_chars: int, budget: _LeadBudget) -> list[list[_Slice]]:
    """Group consecutive slices greedily while the rendered chunk fits.

    Slices come in document order, so the prefix of a group's first slice, plus any lead
    opened inside the group, introduces every later slice of the group.
    """
    groups: list[list[_Slice]] = []
    for piece in slices:
        if groups and len(_render([*groups[-1], piece], budget)) <= max_chars:
            groups[-1].append(piece)
        else:
            groups.append([piece])
    return groups


def _render(group: list[_Slice], budget: _LeadBudget) -> str:
    first = group[0]
    context = first.context[:-1] if first.opens_context else first.context
    return "\n".join([*budget.prefix(context), *(piece.body for piece in group)])


def _prefix_chars(context: tuple[_Unit, ...], budget: _LeadBudget) -> int:
    return sum(len(lead) + 1 for lead in budget.prefix(context))


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` at a word boundary so that, with the ellipsis, it fits in ``limit``."""
    if len(text) <= limit:
        return text
    cut = text[: max(limit - len(_ELLIPSIS), 1)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + _ELLIPSIS


# --------------------------------------------------------------------------------------------
# Chunk metadata
# --------------------------------------------------------------------------------------------
def _make_chunk(
    article: Article, index: int, total: int, paths: Sequence[_Path] | None, text: str
) -> Chunk:
    """Build a chunk; ``paths`` lists the covered units, None meaning the whole article."""
    info = law_info(article.law_id)
    headings = [heading for heading in article.heading_path if heading.strip()]
    citation = f"{info.short_name}, {article.label[:1].lower()}{article.label[1:]}"
    if paths:
        citation = f"{citation}, {_cite_units(paths)}"
    return Chunk(
        chunk_id=f"{article.article_id}#{index}",
        article_id=article.article_id,
        law_id=article.law_id,
        article=article.number,
        chunk_index=index,
        n_chunks=total,
        units=[_unit_label(path) for path in paths] if paths else [WHOLE_ARTICLE],
        heading_path=article.heading_path,
        citation=citation,
        header=f"{' > '.join([info.short_name, *headings])} — {article.label}",
        text=text,
        url=article.url,
        control_notes=control_notes(article),
    )


def control_notes(article: Article) -> list[str]:
    """Distinct annotations of ``article`` that record constitutional review by the STF."""
    return [note for note in dict.fromkeys(article.annotations) if CONTROL_NOTE_RE.search(note)]


def _unit_label(path: _Path) -> str:
    return ", ".join(path)


def _cite_units(paths: Sequence[_Path]) -> str:
    """Covered units in citation style: ``caput e I a XXXIV``, ``IV, a a h``, ``§ 1º``."""
    first, rest = paths[0], paths[1:]
    if not rest:
        return _unit_label(first)
    if first == ("caput",):
        return f"caput e {_cite_units(rest)}"
    if all(path[: len(first)] == first for path in rest):
        return _cite_units(rest)  # the lead of ``first`` only introduces its own sub-units
    return _cite_span(paths)


def _cite_span(paths: Sequence[_Path]) -> str:
    """Two or more units: ``X e Y`` or ``X a Z``, factoring out their common parent."""
    first, last = paths[0], paths[-1]
    shared = 0
    while shared < min(len(first), len(last)) - 1 and first[shared] == last[shared]:
        shared += 1
    joiner = " e " if len(paths) == 2 else " a "
    span = f"{_unit_label(first[shared:])}{joiner}{_unit_label(last[shared:])}"
    return f"{_unit_label(first[:shared])}, {span}" if shared else span
