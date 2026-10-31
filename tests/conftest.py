"""Fixtures shared by every test package."""

import os
from pathlib import Path

import pytest

from lexgraph.config import Settings
from lexgraph.domain import Article, Chunk, LawId

# Model downloads on Windows emit a symlink warning, which the strict warning filter turns
# into an error; it is irrelevant for the tests.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings isolated in a temporary folder (embedded Qdrant, no server, no keys needed)."""
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        models_dir=REPO_ROOT / "data" / "models",
        qdrant_path=data_dir / "qdrant",
        checkpoint_path=data_dir / "checkpoints" / "threads.sqlite",
        qdrant_url=None,
    )


def make_article(
    law_id: LawId = LawId.CDC,
    number: str = "49",
    text: str = (
        "Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias a contar de sua "
        "assinatura ou do ato de recebimento do produto ou serviço, sempre que a contratação "
        "de fornecimento de produtos e serviços ocorrer fora do estabelecimento comercial."
    ),
    *,
    seq: int = 0,
    status: str = "vigente",
    heading_path: list[str] | None = None,
) -> Article:
    """Build an :class:`Article` with sensible defaults for tests."""
    return Article.model_validate(
        {
            "law_id": law_id,
            "number": number,
            "label": f"Art. {number}",
            "seq": seq,
            "heading_path": heading_path or ["CAPÍTULO VI - DA PROTEÇÃO CONTRATUAL"],
            "text": text,
            "status": status,
            "url": f"https://www.planalto.gov.br/example.htm#art{number}",
        }
    )


def make_chunk(article: Article, *, index: int = 0, n_chunks: int = 1) -> Chunk:
    """Build a :class:`Chunk` covering the whole ``article``."""
    citation = f"{article.law_id.value}, art. {article.number}"
    return Chunk(
        chunk_id=f"{article.article_id}#{index}",
        article_id=article.article_id,
        law_id=article.law_id,
        article=article.number,
        chunk_index=index,
        n_chunks=n_chunks,
        units=["*"],
        heading_path=article.heading_path,
        citation=citation,
        header=f"{article.law_id.value} > {' > '.join(article.heading_path)} — {article.label}",
        text=article.text,
        url=article.url,
    )


@pytest.fixture
def sample_articles() -> list[Article]:
    """A handful of real-looking articles from different laws."""
    return [
        make_article(),
        make_article(
            LawId.CLT,
            "58",
            "Art. 58 - A duração normal do trabalho, para os empregados em qualquer atividade "
            "privada, não excederá de 8 (oito) horas diárias, desde que não seja fixado "
            "expressamente outro limite.",
            seq=1,
            heading_path=["TÍTULO II - DAS NORMAS GERAIS DE TUTELA DO TRABALHO"],
        ),
        make_article(
            LawId.LGPD,
            "18",
            "Art. 18. O titular dos dados pessoais tem direito a obter do controlador, em relação "
            "aos dados do titular por ele tratados, a qualquer momento e mediante requisição: "
            "I - confirmação da existência de tratamento; II - acesso aos dados;",
            seq=2,
            heading_path=["CAPÍTULO III - DOS DIREITOS DO TITULAR"],
        ),
    ]
