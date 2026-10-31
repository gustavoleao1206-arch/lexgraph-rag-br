"""Citation markers in answers (``[1]``, ``[1, 2]``, ``[1][2]``, ``[1-3]``) and their sources.

Sources are numbered from 1 in the order they are given to the answer model, so marker
``[n]`` resolves to ``sources[n - 1]``.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from lexgraph.schemas import Citation, Source

# Source numbers inside a marker; ranges may be written with an en dash as well as a hyphen.
_NUMBERS = r"\d{1,2}(?:\s*(?:[,;]|[-\N{EN DASH}]|e)\s*\d{1,2})*"
_MARKER_RE = re.compile(r"\[\s*(" + _NUMBERS + r")\s*\]")
_SPACED_MARKER_RE = re.compile(r"[ \t]*" + _MARKER_RE.pattern)
_RANGE_RE = re.compile(r"(\d+)\s*[-\N{EN DASH}]\s*(\d+)")
_NUMBER_RE = re.compile(r"\d+")
_MAX_RANGE = 20


def _expand(group: str) -> list[int]:
    """Numbers in the inside of one marker: ``"1, 3-5"`` gives ``[1, 3, 4, 5]``."""
    numbers: list[int] = []
    position = 0
    for match in _RANGE_RE.finditer(group):
        numbers.extend(int(n) for n in _NUMBER_RE.findall(group[position : match.start()]))
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end <= start + _MAX_RANGE:
            numbers.extend(range(start, end + 1))
        else:
            numbers.extend((start, end))
        position = match.end()
    numbers.extend(int(n) for n in _NUMBER_RE.findall(group[position:]))
    return numbers


def cited_indices(text: str) -> list[int]:
    """Every source number cited in ``text``, in order of first appearance, without repeats."""
    seen: dict[int, None] = {}
    for match in _MARKER_RE.finditer(text):
        for number in _expand(match.group(1)):
            seen.setdefault(number, None)
    return list(seen)


def dangling_indices(text: str, n_sources: int) -> list[int]:
    """Numbers cited in ``text`` that do not match any of the ``n_sources`` sources."""
    return [index for index in cited_indices(text) if not 1 <= index <= n_sources]


# Units a model may name after the source numbers, in any case: "§ 3", "Art. 7º", "inc. IV",
# "alínea b", "parágrafo único", "par. único", "caput", "item 2".
_UNIT = (
    r"(?i:§|art(?:igo)?s?\b\.?|inc(?:iso)?s?\b\.?|al[íi]neas?\b|al\.|par[áa]grafos?\b|par\."
    r"|caput\b|ite(?:m|ns)\b)"
)
_DETAILED_MARKER_RE = re.compile(r"\[\s*(" + _NUMBERS + r")\s*[,;]?\s*" + _UNIT + r"[^\[\]]*\]")
# A paragraph glued to the closing bracket: "[2]§ XVII", "[1]§ 1º", "[3]§§ 2º e 3º".
_UNIT_LABEL = r"(?:\d{1,3}\s*[º°o]?|[IVXLCDM]+(?:-[A-Z])?\b|[úu]nico\b)"
_GLUED_UNIT_RE = re.compile(
    r"(" + _MARKER_RE.pattern + r")§+\s*" + _UNIT_LABEL + r"(?:\s*(?:,|e)\s*" + _UNIT_LABEL + r")*"
)
# gpt-oss's own citation style: "【1】", "【1†§ 1】", "【2†L1-L4】" (lenticular brackets).
_LENTICULAR_MARKER_RE = re.compile(r"(\S?)【\s*(" + _NUMBERS + r")\s*(?:†[^】]*)?】")
_REPEATED_MARKER_RE = re.compile(r"\[(\d{1,2})\](?:\s*\[\1\])+")


def _lenticular(match: re.Match[str]) -> str:
    before = match.group(1)
    space = " " if before and before not in "([]" else ""
    return f"{before}{space}[{match.group(2)}]"


def normalize_markers(text: str) -> str:
    """Reduce markers to their source numbers, in the ``[n]`` style.

    ``[1§ 3]`` and ``[2, inciso IV]`` become ``[1]`` and ``[2]``; ``[1, 2, parágrafo único]``
    becomes ``[1, 2]``, and ``[2]§ XVII``, with the paragraph glued after the marker, becomes
    ``[2]``. Models sometimes write the paragraph in or next to the marker; the unit is
    already in the source's citation, and only the numbers link the sentence to it. The
    lenticular markers that gpt-oss writes on its own (``consumidor【1†§ 1】``) become ``[1]``
    too, after a space, and a marker repeated right after itself (``[1][1]``) is kept once.
    """
    text = _LENTICULAR_MARKER_RE.sub(_lenticular, text)
    text = _GLUED_UNIT_RE.sub(r"\1", _DETAILED_MARKER_RE.sub(r"[\1]", text))
    return _REPEATED_MARKER_RE.sub(r"[\1]", text)


def drop_dangling_markers(text: str, n_sources: int) -> str:
    """``text`` without the numbers that match none of the ``n_sources`` sources.

    A marker that keeps some valid numbers is rewritten as ``[1][2]``; one left with none
    disappears together with the space before it.
    """

    def rewrite(match: re.Match[str]) -> str:
        numbers = _expand(match.group(1))
        valid = list(dict.fromkeys(n for n in numbers if 1 <= n <= n_sources))
        if len(valid) == len(numbers):
            return match.group(0)
        leading = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip(" \t"))]
        return leading + "".join(f"[{n}]" for n in valid) if valid else ""

    return _SPACED_MARKER_RE.sub(rewrite, text)


def build_citations(answer: str, sources: Sequence[Source]) -> list[Citation]:
    """Resolve the markers of ``answer`` against ``sources``, ignoring dangling ones."""
    by_index = {source.index: source for source in sources}
    return [
        Citation(
            index=index,
            article_id=by_index[index].article_id,
            citation=by_index[index].citation,
            url=by_index[index].url,
        )
        for index in cited_indices(answer)
        if index in by_index
    ]


def strip_markers(text: str) -> str:
    """Remove citation markers, e.g. from earlier answers whose sources are gone."""
    return _SPACED_MARKER_RE.sub("", text)


def passages_to_sources(passages: Sequence[Mapping[str, Any]]) -> list[Source]:
    """Turn passage dicts (as stored in the agent state) into sources numbered from 1."""
    return [
        Source.model_validate({**passage, "index": index})
        for index, passage in enumerate(passages, start=1)
    ]
