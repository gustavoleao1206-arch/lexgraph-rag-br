"""Retrieval with the real models cached in ``data/models``: EmbeddingGemma, BM25 and a reranker.

Fifteen real articles (some shortened) are indexed into in-memory Qdrant once per module.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from lexgraph.catalog import law_info
from lexgraph.config import RetrievalMode
from lexgraph.domain import Chunk, LawId
from lexgraph.retrieval.embeddings import DenseEmbedder
from lexgraph.retrieval.indexer import index_chunks
from lexgraph.retrieval.reranker import CrossEncoderReranker
from lexgraph.retrieval.retriever import LOOKUP_SCORE, LegalRetriever
from lexgraph.retrieval.sparse import Bm25Encoder
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

MODELS_DIR = REPO_ROOT / "data" / "models"
DENSE_MODEL = "google/embeddinggemma-300m"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3-int8"

CDC_RIGHTS = "TÍTULO I - Dos Direitos do Consumidor"
CLT_TUTELA = "TÍTULO II - DAS NORMAS GERAIS DE TUTELA DO TRABALHO"
CLT_CONTRATO = "TÍTULO IV - DO CONTRATO INDIVIDUAL DO TRABALHO"


def legal_chunk(law_id: LawId, label: str, headings: Sequence[str], text: str) -> Chunk:
    """Whole-article chunk built the way the ingestion chunker does."""
    number = label.removeprefix("Art. ").rstrip("º")
    short_name = law_info(law_id).short_name
    citation_number = label.removeprefix("Art. ")
    return Chunk(
        chunk_id=f"{law_id.value}:{number}#0",
        article_id=f"{law_id.value}:{number}",
        law_id=law_id,
        article=number,
        chunk_index=0,
        n_chunks=1,
        units=["*"],
        heading_path=list(headings),
        citation=f"{short_name}, art. {citation_number}",
        header=f"{short_name} > {' > '.join(headings)} — {label}",
        text=text,
        url=f"https://www.planalto.gov.br/ccivil_03/example.htm#art{number}",
    )


CHUNKS = [
    legal_chunk(
        LawId.CDC,
        "Art. 2º",
        [CDC_RIGHTS, "CAPÍTULO I - Disposições Gerais"],
        "Art. 2º Consumidor é toda pessoa física ou jurídica que adquire ou utiliza produto ou "
        "serviço como destinatário final.\nParágrafo único. Equipara-se a consumidor a "
        "coletividade de pessoas, ainda que indetermináveis, que haja intervindo nas relações "
        "de consumo.",
    ),
    legal_chunk(
        LawId.CDC,
        "Art. 18",
        [CDC_RIGHTS, "SEÇÃO III - Da Responsabilidade por Vício do Produto e do Serviço"],
        "Art. 18. Os fornecedores de produtos de consumo duráveis ou não duráveis respondem "
        "solidariamente pelos vícios de qualidade ou quantidade que os tornem impróprios ou "
        "inadequados ao consumo a que se destinam ou lhes diminuam o valor.\n§ 1° Não sendo o "
        "vício sanado no prazo máximo de trinta dias, pode o consumidor exigir, "
        "alternativamente e à sua escolha:\nI - a substituição do produto por outro da mesma "
        "espécie, em perfeitas condições de uso;\nII - a restituição imediata da quantia paga, "
        "monetariamente atualizada, sem prejuízo de eventuais perdas e danos;\nIII - o "
        "abatimento proporcional do preço.",
    ),
    legal_chunk(
        LawId.CDC,
        "Art. 26",
        [CDC_RIGHTS, "SEÇÃO IV - Da Decadência e da Prescrição"],
        "Art. 26. O direito de reclamar pelos vícios aparentes ou de fácil constatação caduca "
        "em:\nI - trinta dias, tratando-se de fornecimento de serviço e de produtos não "
        "duráveis;\nII - noventa dias, tratando-se de fornecimento de serviço e de produtos "
        "duráveis.\n§ 1° Inicia-se a contagem do prazo decadencial a partir da entrega efetiva "
        "do produto ou do término da execução dos serviços.",
    ),
    legal_chunk(
        LawId.CDC,
        "Art. 42",
        [CDC_RIGHTS, "SEÇÃO V - Da Cobrança de Dívidas"],
        "Art. 42. Na cobrança de débitos, o consumidor inadimplente não será exposto a "
        "ridículo, nem será submetido a qualquer tipo de constrangimento ou ameaça.\nParágrafo "
        "único. O consumidor cobrado em quantia indevida tem direito à repetição do indébito, "
        "por valor igual ao dobro do que pagou em excesso, acrescido de correção monetária e "
        "juros legais, salvo hipótese de engano justificável.",
    ),
    legal_chunk(
        LawId.CDC,
        "Art. 49",
        [CDC_RIGHTS, "CAPÍTULO VI - Da Proteção Contratual", "SEÇÃO I - Disposições Gerais"],
        "Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias a contar de sua "
        "assinatura ou do ato de recebimento do produto ou serviço, sempre que a contratação "
        "de fornecimento de produtos e serviços ocorrer fora do estabelecimento comercial, "
        "especialmente por telefone ou a domicílio.\nParágrafo único. Se o consumidor exercitar "
        "o direito de arrependimento previsto neste artigo, os valores eventualmente pagos, a "
        "qualquer título, durante o prazo de reflexão, serão devolvidos, de imediato, "
        "monetariamente atualizados.",
    ),
    legal_chunk(
        LawId.CDC,
        "Art. 51",
        [CDC_RIGHTS, "CAPÍTULO VI - Da Proteção Contratual", "SEÇÃO II - Das Cláusulas Abusivas"],
        "Art. 51. São nulas de pleno direito, entre outras, as cláusulas contratuais relativas "
        "ao fornecimento de produtos e serviços que:\nI - impossibilitem, exonerem ou atenuem a "
        "responsabilidade do fornecedor por vícios de qualquer natureza dos produtos e "
        "serviços ou impliquem renúncia ou disposição de direitos;\nII - subtraiam ao "
        "consumidor a opção de reembolso da quantia já paga, nos casos previstos neste "
        "código;\nIV - estabeleçam obrigações consideradas iníquas, abusivas, que coloquem o "
        "consumidor em desvantagem exagerada, ou sejam incompatíveis com a boa-fé ou a "
        "eqüidade;",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 58",
        [CLT_TUTELA, "CAPÍTULO II - DA DURAÇÃO DO TRABALHO", "SEÇÃO II - DA JORNADA DE TRABALHO"],
        "Art. 58 - A duração normal do trabalho, para os empregados em qualquer atividade "
        "privada, não excederá de 8 (oito) horas diárias, desde que não seja fixado "
        "expressamente outro limite.\n§ 1º Não serão descontadas nem computadas como jornada "
        "extraordinária as variações de horário no registro de ponto não excedentes de cinco "
        "minutos, observado o limite máximo de dez minutos diários.",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 59",
        [CLT_TUTELA, "CAPÍTULO II - DA DURAÇÃO DO TRABALHO", "SEÇÃO II - DA JORNADA DE TRABALHO"],
        "Art. 59. A duração diária do trabalho poderá ser acrescida de horas extras, em número "
        "não excedente de duas, por acordo individual, convenção coletiva ou acordo coletivo "
        "de trabalho.\n§ 1º A remuneração da hora extra será, pelo menos, 50% (cinquenta por "
        "cento) superior à da hora normal.",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 129",
        [CLT_TUTELA, "CAPÍTULO IV - DAS FÉRIAS ANUAIS", "SEÇÃO I - DO DIREITO A FÉRIAS"],
        "Art. 129 - Todo empregado terá direito anualmente ao gozo de um período de férias, "
        "sem prejuízo da remuneração.",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 134",
        [CLT_TUTELA, "CAPÍTULO IV - DAS FÉRIAS ANUAIS", "SEÇÃO II - DA CONCESSÃO DAS FÉRIAS"],
        "Art. 134 - As férias serão concedidas por ato do empregador, em um só período, nos 12 "
        "(doze) meses subseqüentes à data em que o empregado tiver adquirido o direito.\n§ 1º "
        "Desde que haja concordância do empregado, as férias poderão ser usufruídas em até "
        "três períodos, sendo que um deles não poderá ser inferior a quatorze dias corridos e "
        "os demais não poderão ser inferiores a cinco dias corridos, cada um.",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 477",
        [CLT_CONTRATO, "CAPÍTULO V - DA RESCISÃO"],
        "Art. 477. Na extinção do contrato de trabalho, o empregador deverá proceder à "
        "anotação na Carteira de Trabalho e Previdência Social, comunicar a dispensa aos órgãos "
        "competentes e realizar o pagamento das verbas rescisórias no prazo e na forma "
        "estabelecidos neste artigo.\n§ 4º O pagamento a que fizer jus o empregado será "
        "efetuado:\nI - em dinheiro, depósito bancário ou cheque visado, conforme acordem as "
        "partes; ou\nII - em dinheiro ou depósito bancário quando o empregado for analfabeto.",
    ),
    legal_chunk(
        LawId.CLT,
        "Art. 482",
        [CLT_CONTRATO, "CAPÍTULO V - DA RESCISÃO"],
        "Art. 482 - Constituem justa causa para rescisão do contrato de trabalho pelo "
        "empregador:\na) ato de improbidade;\nb) incontinência de conduta ou mau "
        "procedimento;\nd) condenação criminal do empregado, passada em julgado, caso não tenha "
        "havido suspensão da execução da pena;\ne) desídia no desempenho das respectivas "
        "funções;\nf) embriaguez habitual ou em serviço;\ng) violação de segredo da empresa;\nh) "
        "ato de indisciplina ou de insubordinação;\ni) abandono de emprego;",
    ),
    legal_chunk(
        LawId.LGPD,
        "Art. 7º",
        ["CAPÍTULO II - DO TRATAMENTO DE DADOS PESSOAIS"],
        "Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes "
        "hipóteses:\nI - mediante o fornecimento de consentimento pelo titular;\nII - para o "
        "cumprimento de obrigação legal ou regulatória pelo controlador;\nV - quando necessário "
        "para a execução de contrato ou de procedimentos preliminares relacionados a contrato "
        "do qual seja parte o titular, a pedido do titular dos dados;",
    ),
    legal_chunk(
        LawId.LGPD,
        "Art. 18",
        ["CAPÍTULO III - DOS DIREITOS DO TITULAR"],
        "Art. 18. O titular dos dados pessoais tem direito a obter do controlador, em relação "
        "aos dados do titular por ele tratados, a qualquer momento e mediante requisição:\nI - "
        "confirmação da existência de tratamento;\nII - acesso aos dados;\nIII - correção de "
        "dados incompletos, inexatos ou desatualizados;\nIV - anonimização, bloqueio ou "
        "eliminação de dados desnecessários, excessivos ou tratados em desconformidade com o "
        "disposto nesta Lei;\nV - portabilidade dos dados a outro fornecedor de serviço ou "
        "produto;",
    ),
    legal_chunk(
        LawId.MCI,
        "Art. 19",
        ["SEÇÃO III - Da Responsabilidade por Danos Decorrentes de Conteúdo Gerado por Terceiros"],
        "Art. 19. Com o intuito de assegurar a liberdade de expressão e impedir a censura, o "
        "provedor de aplicações de internet somente poderá ser responsabilizado civilmente por "
        "danos decorrentes de conteúdo gerado por terceiros se, após ordem judicial "
        "específica, não tomar as providências para, no âmbito e nos limites técnicos do seu "
        "serviço e dentro do prazo assinalado, tornar indisponível o conteúdo apontado como "
        "infringente, ressalvadas as disposições legais em contrário.",
    ),
]

ONLINE_PURCHASE = (
    "Comprei um celular pela internet e me arrependi. Posso desistir da compra e pedir o "
    "dinheiro de volta?"
)


@pytest.fixture(scope="module")
def dense() -> DenseEmbedder:
    return DenseEmbedder(DENSE_MODEL, cache_dir=MODELS_DIR)


@pytest.fixture(scope="module")
def sparse() -> Bm25Encoder:
    return Bm25Encoder(cache_dir=MODELS_DIR)


@pytest.fixture(scope="module")
def retriever(dense: DenseEmbedder, sparse: Bm25Encoder) -> Iterator[LegalRetriever]:
    client = QdrantClient(":memory:")
    index_chunks(client, CHUNKS, alias="lexgraph", dense=dense, sparse=sparse)
    yield LegalRetriever(
        client=client,
        collection="lexgraph",
        dense=dense,
        sparse=sparse,
        reranker=CrossEncoderReranker(RERANKER_MODEL, cache_dir=MODELS_DIR),
        rerank_candidates=5,
    )
    client.close()


def test_colloquial_question_finds_the_right_of_withdrawal_first(
    retriever: LegalRetriever,
) -> None:
    passages = retriever.search(ONLINE_PURCHASE, mode="hybrid")

    assert passages[0].article_id == "CDC:49"


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
@pytest.mark.parametrize(
    ("question", "article_id"),
    [
        ("Quantas horas por dia o empregado pode trabalhar?", "CLT:58"),
        ("Quais atitudes do funcionário permitem a demissão por justa causa?", "CLT:482"),
        ("O provedor de aplicações responde por conteúdo publicado por terceiros?", "MCI:19"),
        ("Quais direitos o titular tem sobre os seus dados pessoais?", "LGPD:18"),
    ],
)
def test_every_mode_finds_the_article_in_the_top_3(
    retriever: LegalRetriever, question: str, article_id: str, mode: RetrievalMode
) -> None:
    passages = retriever.search(question, mode=mode, k=3)

    assert article_id in [passage.article_id for passage in passages]


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_semantic_branch_finds_paraphrases(retriever: LegalRetriever, mode: RetrievalMode) -> None:
    # The question shares no indexed term with Marco Civil, art. 19: only embeddings match it.
    question = "A rede social responde por um post ofensivo publicado por usuário?"

    passages = retriever.search(question, mode=mode, k=3)

    assert "MCI:19" in [passage.article_id for passage in passages]


def test_bm25_matches_questions_typed_without_accents(retriever: LegalRetriever) -> None:
    passages = retriever.search("ferias anuais do empregado", mode="sparse", k=2)

    assert {passage.article_id for passage in passages} == {"CLT:129", "CLT:134"}


def test_bm25_stems_ignore_accents(sparse: Bm25Encoder) -> None:
    accented = sparse.encode_query("férias rescisão indenização salário informações")
    plain = sparse.encode_query("ferias rescisao indenizacao salario informacoes")

    assert sorted(plain.indices) == sorted(accented.indices)
    assert len(plain.indices) == 5


def test_bm25_ignores_ordinal_signs(sparse: Bm25Encoder) -> None:
    document = sparse.encode_documents(["Art. 7º O tratamento de dados pessoais"])[0]
    query = sparse.encode_query("art 7")

    assert set(query.indices) <= set(document.indices)


def test_bm25_drops_accented_stopwords(sparse: Bm25Encoder) -> None:
    assert sparse.encode_query("não é até").indices == []


def test_dense_vectors_use_the_model_prompts(dense: DenseEmbedder) -> None:
    query = dense.embed_query("férias")
    document = dense.embed_documents(["férias"])[0]

    assert len(query) == len(document) == 768
    assert query != document


def test_reranker_keeps_the_best_passage_first(retriever: LegalRetriever) -> None:
    passages = retriever.search(ONLINE_PURCHASE, k=3, rerank=True)

    assert passages[0].article_id == "CDC:49"
    scores = [passage.rerank_score for passage in passages]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, key=lambda score: score or 0.0, reverse=True)


def test_index_records_the_encoders_of_the_retriever(retriever: LegalRetriever) -> None:
    retriever.check_index()


def test_reranker_is_available_only_from_the_local_cache(
    retriever: LegalRetriever, tmp_path: Path
) -> None:
    assert retriever.rerank_available()
    assert not CrossEncoderReranker(RERANKER_MODEL, cache_dir=tmp_path).is_available()
    assert list(tmp_path.iterdir()) == []  # nothing was downloaded


def test_article_lookup_and_count(retriever: LegalRetriever) -> None:
    (passage,) = retriever.get_article("CDC:49")

    assert passage.score == LOOKUP_SCORE
    assert passage.citation == "CDC, art. 49"
    assert retriever.count() == len(CHUNKS)
