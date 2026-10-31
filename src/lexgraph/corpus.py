"""Read and write the parsed corpus: one JSON Lines file per law plus a manifest.

Layout of ``data/corpus``::

    manifest.json  # provenance and counts per law
    clt.jsonl  # one Article per line, in document order
    cdc.jsonl
    ...
"""

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from lexgraph.domain import Article, LawId

MANIFEST_FILE = "manifest.json"


class LawManifest(BaseModel):
    """Provenance and counts for one law in the corpus."""

    law_id: LawId
    source: str
    source_url: str
    source_sha256: str
    last_modified: str | None = None
    fetched_at: str
    articles: int
    in_force: int
    revoked: int
    vetoed: int


class CorpusManifest(BaseModel):
    """Manifest describing the whole corpus."""

    generated_at: str
    parser_version: str
    laws: dict[LawId, LawManifest]


def law_file(corpus_dir: Path, law_id: LawId) -> Path:
    """Path of the JSON Lines file holding the articles of ``law_id``."""
    return corpus_dir / f"{law_id.value.lower()}.jsonl"


def write_corpus(
    corpus_dir: Path,
    articles_by_law: Mapping[LawId, Sequence[Article]],
    manifest: CorpusManifest,
) -> None:
    """Write one JSONL file per law and the manifest (UTF-8, LF line endings)."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for law_id, articles in articles_by_law.items():
        lines = (
            json.dumps(article.model_dump(mode="json", exclude={"article_id"}), ensure_ascii=False)
            for article in sorted(articles, key=lambda a: a.seq)
        )
        law_file(corpus_dir, law_id).write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
        )
    (corpus_dir / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def read_articles(path: Path) -> list[Article]:
    """Read the articles stored in one JSON Lines file."""
    with path.open(encoding="utf-8") as handle:
        return [Article.model_validate_json(line) for line in handle if line.strip()]


class Corpus:
    """In-memory view of the corpus, indexed by article id."""

    def __init__(self, articles: Iterable[Article], manifest: CorpusManifest) -> None:
        self.manifest = manifest
        self._by_law: dict[LawId, list[Article]] = {}
        self._by_id: dict[str, Article] = {}
        for article in articles:
            self._by_law.setdefault(article.law_id, []).append(article)
            self._by_id[article.article_id] = article
        for items in self._by_law.values():
            items.sort(key=lambda a: a.seq)

    @classmethod
    def load(cls, corpus_dir: Path) -> "Corpus":
        """Load the corpus written by :func:`write_corpus`."""
        manifest = CorpusManifest.model_validate_json(
            (corpus_dir / MANIFEST_FILE).read_text(encoding="utf-8")
        )
        articles: list[Article] = []
        for law_id in manifest.laws:
            articles.extend(read_articles(law_file(corpus_dir, law_id)))
        return cls(articles, manifest)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, article_id: object) -> bool:
        return article_id in self._by_id

    @property
    def law_ids(self) -> list[LawId]:
        """Laws present in the corpus, in catalog order."""
        return [law_id for law_id in LawId if law_id in self._by_law]

    def get(self, article_id: str) -> Article | None:
        """Return the article with id ``article_id`` (e.g. ``"CLT:482"``), if any."""
        return self._by_id.get(article_id)

    def by_law(self, law_id: LawId) -> list[Article]:
        """All articles of ``law_id`` in document order (any status)."""
        return list(self._by_law.get(law_id, []))

    def in_force(self, law_ids: Iterable[LawId] | None = None) -> Iterator[Article]:
        """Articles currently in force, optionally restricted to ``law_ids``."""
        selected = list(law_ids) if law_ids is not None else self.law_ids
        for law_id in selected:
            yield from (a for a in self._by_law.get(law_id, []) if a.status == "vigente")
