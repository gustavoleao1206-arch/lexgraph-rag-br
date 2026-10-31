"""Fakes and fixtures for the retrieval tests: in-memory Qdrant and no model downloads."""

import re
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient, models

from lexgraph.domain import Chunk, LawId, Passage
from lexgraph.retrieval.indexer import IndexReport, index_chunks
from lexgraph.retrieval.retriever import LegalRetriever

ALIAS = "lexgraph"
DENSE_SIZE = 32


class HashingSparseEncoder:
    """Bag-of-words sparse vectors: each lowercase word gets a CRC32 index and its count."""

    def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self.encode_query(text) for text in texts]

    def encode_query(self, text: str) -> models.SparseVector:
        counts = Counter(zlib.crc32(word.encode()) for word in re.findall(r"\w+", text.lower()))
        indices = sorted(counts)
        return models.SparseVector(indices=indices, values=[float(counts[i]) for i in indices])


class StopwordsOnlySparseEncoder(HashingSparseEncoder):
    """Encodes documents normally but every query as an empty vector (all stopwords)."""

    def encode_query(self, text: str) -> models.SparseVector:
        return models.SparseVector(indices=[], values=[])


class FixedScoreReranker:
    """Ranks passages by a fixed score per article id and records every call."""

    def __init__(self, scores: Mapping[str, float], *, available: bool = True) -> None:
        self.scores = dict(scores)
        self.available = available
        self.calls: list[tuple[str, list[Passage], int]] = []

    def is_available(self) -> bool:
        return self.available

    def rerank(self, query: str, passages: Sequence[Passage], top_n: int) -> list[Passage]:
        self.calls.append((query, list(passages), top_n))
        ranked = sorted(passages, key=lambda p: self.scores.get(p.article_id, 0.0), reverse=True)
        return [
            passage.model_copy(update={"rerank_score": self.scores.get(passage.article_id, 0.0)})
            for passage in ranked[:top_n]
        ]


def make_legal_chunk(
    law_id: LawId,
    number: str,
    text: str,
    *,
    index: int = 0,
    n_chunks: int = 1,
    units: Sequence[str] = ("*",),
) -> Chunk:
    """Build a chunk of article ``number`` of ``law_id``."""
    article_id = f"{law_id.value}:{number}"
    return Chunk(
        chunk_id=f"{article_id}#{index}",
        article_id=article_id,
        law_id=law_id,
        article=number,
        chunk_index=index,
        n_chunks=n_chunks,
        units=list(units),
        heading_path=["CAPÍTULO I - Disposições Gerais"],
        citation=f"{law_id.value}, art. {number}",
        header=f"{law_id.value} > CAPÍTULO I - Disposições Gerais — Art. {number}",
        text=text,
        url=f"https://www.planalto.gov.br/ccivil_03/example.htm#art{number}",
    )


CF_5_CAPUT = (
    "Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se "
    "aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito à vida."
)

# The CF art. 5 slices are listed out of order on purpose: get_article must sort them.
CHUNKS = (
    make_legal_chunk(
        LawId.CDC,
        "49",
        "Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias a contar de sua "
        "assinatura ou do ato de recebimento do produto ou serviço, sempre que a contratação "
        "ocorrer fora do estabelecimento comercial, especialmente por telefone ou a domicílio.",
    ),
    make_legal_chunk(
        LawId.CDC,
        "18",
        "Art. 18. Os fornecedores de produtos de consumo duráveis ou não duráveis respondem "
        "solidariamente pelos vícios de qualidade ou quantidade que os tornem impróprios.",
    ),
    make_legal_chunk(
        LawId.CLT,
        "58",
        "Art. 58 - A duração normal do trabalho, para os empregados em qualquer atividade "
        "privada, não excederá de 8 (oito) horas diárias.",
    ),
    make_legal_chunk(
        LawId.CLT,
        "129",
        "Art. 129 - Todo empregado terá direito anualmente ao gozo de um período de férias, "
        "sem prejuízo da remuneração.",
    ),
    make_legal_chunk(
        LawId.CLT,
        "482",
        "Art. 482 - Constituem justa causa para rescisão do contrato de trabalho pelo "
        "empregador: a) ato de improbidade; b) incontinência de conduta ou mau procedimento;",
    ),
    make_legal_chunk(
        LawId.LGPD,
        "18",
        "Art. 18. O titular dos dados pessoais tem direito a obter do controlador, a qualquer "
        "momento e mediante requisição: I - confirmação da existência de tratamento;",
    ),
    make_legal_chunk(
        LawId.MCI,
        "19",
        "Art. 19. O provedor de aplicações de internet somente poderá ser responsabilizado "
        "civilmente por danos decorrentes de conteúdo gerado por terceiros se, após ordem "
        "judicial específica, não tornar indisponível o conteúdo apontado como infringente.",
    ),
    make_legal_chunk(
        LawId.CF,
        "5",
        f"{CF_5_CAPUT}\nXI - a casa é asilo inviolável do indivíduo;",
        index=2,
        n_chunks=3,
        units=["XI"],
    ),
    make_legal_chunk(LawId.CF, "5", CF_5_CAPUT, index=0, n_chunks=3, units=["caput"]),
    make_legal_chunk(
        LawId.CF,
        "5",
        f"{CF_5_CAPUT}\nI - homens e mulheres são iguais em direitos e obrigações;",
        index=1,
        n_chunks=3,
        units=["I"],
    ),
)


def chunk_by_id(chunk_id: str) -> Chunk:
    """The test chunk with ``chunk_id``."""
    return next(chunk for chunk in CHUNKS if chunk.chunk_id == chunk_id)


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    """An in-memory Qdrant client."""
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def dense() -> DeterministicFakeEmbedding:
    """Deterministic pseudo-embeddings: equal texts get equal vectors."""
    return DeterministicFakeEmbedding(size=DENSE_SIZE)


@pytest.fixture
def sparse() -> HashingSparseEncoder:
    """Bag-of-words sparse encoder."""
    return HashingSparseEncoder()


@pytest.fixture
def chunks() -> list[Chunk]:
    """Chunks of several laws, including an article split in three."""
    return list(CHUNKS)


@pytest.fixture
def indexed(
    client: QdrantClient,
    chunks: list[Chunk],
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparseEncoder,
) -> IndexReport:
    """Index :data:`CHUNKS` behind the alias :data:`ALIAS`."""
    return index_chunks(client, chunks, alias=ALIAS, dense=dense, sparse=sparse, batch_size=4)


@pytest.fixture
def retriever(
    client: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparseEncoder,
    indexed: IndexReport,
) -> LegalRetriever:
    """A retriever over the indexed chunks, without reranker."""
    return LegalRetriever(client=client, collection=ALIAS, dense=dense, sparse=sparse, k=3)
