"""Domain models shared by ingestion, retrieval, the agent, the API and the evaluation code."""

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class LawId(StrEnum):
    """Canonical identifiers of the laws in the corpus."""

    CLT = "CLT"
    CDC = "CDC"
    LGPD = "LGPD"
    MCI = "MCI"
    CF = "CF"
    ADCT = "ADCT"


ArticleStatus = Literal["vigente", "revogado", "vetado"]

ARTICLE_ID_RE = re.compile(r"^(CLT|CDC|LGPD|MCI|CF|ADCT):(\d+(?:-[A-Z]{1,2})?|preambulo)$")


def make_article_id(law_id: LawId | str, number: str) -> str:
    """Build the canonical article id, e.g. ``make_article_id("CLT", "482") -> "CLT:482"``."""
    return f"{LawId(law_id)}:{number}"


def split_article_id(article_id: str) -> tuple[LawId, str]:
    """Split ``"CLT:59-A"`` into ``(LawId.CLT, "59-A")``; raises ``ValueError`` if malformed."""
    match = ARTICLE_ID_RE.match(article_id)
    if match is None:
        msg = f"invalid article id: {article_id!r}"
        raise ValueError(msg)
    return LawId(match.group(1)), match.group(2)


def article_sort_key(number: str) -> tuple[int, str]:
    """Order article numbers naturally: ``"59" < "59-A" < "59-B" < "60"``."""
    if number == "preambulo":
        return (0, "")
    base, _, suffix = number.partition("-")
    return (int(base), suffix)


class Article(BaseModel):
    """One article as published on the official compiled page, after cleaning."""

    model_config = ConfigDict(frozen=True)

    law_id: LawId
    number: str = Field(description='Normalized number: "1", "59-A" or "preambulo".')
    label: str = Field(description='Label as printed, e.g. "Art. 59-A".')
    seq: int = Field(ge=0, description="Position of the article within its law.")
    heading_path: list[str] = Field(default_factory=list, description="TÍTULO/CAPÍTULO/Seção path.")
    rubric: str | None = None
    text: str = Field(description="Current text only: struck text and amendment notes removed.")
    status: ArticleStatus
    annotations: list[str] = Field(default_factory=list, description="Removed amendment notes.")
    removed: list[str] = Field(default_factory=list, description="Dropped struck/revoked units.")
    anchor: str | None = Field(default=None, description="Anchor name on the source page.")
    url: str = Field(description="Deep link to the article on the official page.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def article_id(self) -> str:
        """Canonical id such as ``"CLT:482"``."""
        return make_article_id(self.law_id, self.number)


class Chunk(BaseModel):
    """A retrieval unit: a whole article or a slice of a long article."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description='Article id plus chunk index, e.g. "CF:5#3".')
    article_id: str
    law_id: LawId
    article: str = Field(description='Article number, e.g. "482".')
    chunk_index: int = Field(ge=0)
    n_chunks: int = Field(ge=1)
    units: list[str] = Field(
        default_factory=list, description='Covered units, e.g. ["caput", "I", "II"]; ["*"] = all.'
    )
    heading_path: list[str] = Field(default_factory=list)
    citation: str = Field(description='Short citation, e.g. "CF, art. 5º, LXXVIII".')
    header: str = Field(description='Context header, e.g. "CLT > TÍTULO IV > ... — Art. 482".')
    text: str = Field(
        description="Passage text, prefixed with the caput when the article is split."
    )
    url: str
    control_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Notes on constitutional review by the STF (ADI/ADC/ADPF, unconstitutionality, "
            "suspended execution) attached to the article; not part of the indexed text."
        ),
    )

    def embedding_text(self) -> str:
        """Text indexed by both the dense and the sparse (BM25) vectors."""
        return f"{self.header}\n{self.text}"


class Passage(Chunk):
    """A chunk returned by a search, with its scores."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(description="Fusion (or single-branch) score from the vector store.")
    rerank_score: float | None = Field(default=None, description="Cross-encoder score, if used.")
