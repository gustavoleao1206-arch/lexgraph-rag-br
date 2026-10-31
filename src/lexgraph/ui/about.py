"""Sobre page: what the assistant does, how the pipeline works and its limits."""

import streamlit as st

from lexgraph.catalog import LAWS
from lexgraph.schemas import ServiceInfo
from lexgraph.ui.client import ApiClient, ApiError
from lexgraph.ui.formatting import describe_api_error, format_int, retrieval_mode_label

DISCLAIMER = (
    "Projeto de demonstração. As respostas são produzidas automaticamente a partir dos textos "
    "compilados e podem conter erros ou omissões. Elas não substituem orientação jurídica nem "
    "o texto publicado no DOU."
)

INTRO = """\
O **LexGraph** responde perguntas sobre a legislação federal brasileira usando apenas os textos
compilados publicados no portal do Planalto. Cada afirmação jurídica da resposta indica o artigo
de onde foi tirada, com link para o texto oficial, e uma checagem final, quando ativada no
servidor, confere se a resposta está de fato apoiada nas fontes.
"""

STEPS = """\
1. **Análise** — classifica a pergunta (jurídica, conversa ou fora do escopo), identifica as
   leis e os artigos citados e reescreve a busca com o vocabulário da lei.
2. **Busca híbrida** — combina busca semântica (embeddings) e lexical (BM25) no Qdrant, com
   fusão por RRF; artigos citados explicitamente entram direto no contexto.
3. **Avaliação** — um modelo separa os trechos realmente úteis; se nenhum servir, a busca é
   reformulada e repetida.
4. **Resposta** — o modelo responde somente com base nos trechos numerados e cita cada
   afirmação com `[n]`.
5. **Autoverificação** — quando ativada, uma segunda checagem compara a resposta com as fontes;
   afirmações sem apoio fazem a resposta ser gerada de novo. Uma resposta que não passou pela
   checagem aparece marcada como "Sem autoverificação".
"""

PIPELINE_DOT = """\
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#E8EEF7", color="#1F4E8C",
        fontname="Helvetica", fontsize=16, fontcolor="#1B2430", margin="0.15,0.06"];
  edge [color="#7A8699", fontname="Helvetica", fontsize=13, fontcolor="#7A8699",
        arrowsize=0.7];
  analise [label="Análise"];
  busca [label="Busca\\nhíbrida"];
  avaliacao [label="Avaliação"];
  resposta [label="Resposta\\ncom citações"];
  verificacao [label="Auto-\\nverificação"];
  reformulacao [label="Reformulação"];
  padrao [label="Resposta padrão", style="rounded,dashed"];
  analise -> busca -> avaliacao -> resposta -> verificacao;
  avaliacao -> reformulacao [label="nada útil"];
  reformulacao -> busca;
  verificacao -> resposta [label="sem apoio"];
  analise -> padrao [label="fora do escopo"];
  avaliacao -> padrao [label="sem base"];
}
"""


def render(client: ApiClient) -> None:
    """Render the Sobre page."""
    st.set_page_config(layout="centered")
    st.title("Sobre o LexGraph", anchor=False)
    st.markdown(INTRO)
    st.warning(DISCLAIMER, icon=":material/gavel:")
    st.subheader("Como funciona", anchor=False)
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown(STEPS)
    st.subheader("Configuração do serviço", anchor=False)
    try:
        st.markdown(_service_table(client.info()))
    except ApiError as error:
        st.info(describe_api_error(error), icon=":material/cloud_off:")
    st.subheader("Leis do acervo", anchor=False)
    st.markdown(
        "\n".join(
            f"- **{law.short_name}** — {law.name} ({law.official_name})" for law in LAWS.values()
        )
    )


def _service_table(info: ServiceInfo) -> str:
    rows = [
        ("Modelo de resposta", f"`{info.llm_model}`"),
        ("Modelo auxiliar (análise, avaliação e checagem)", f"`{info.fast_llm_model}`"),
        ("Modelo reserva", f"`{info.llm_fallback_model}`" if info.llm_fallback_model else "—"),
        ("Embeddings", f"`{info.embedding_model}` ({info.embedding_provider})"),
        ("Busca", retrieval_mode_label(info.retrieval_mode)),
        ("Reranking", _rerank_label(info)),
        ("Índice", f"`{info.collection}` · {format_int(info.indexed_chunks)} trechos"),
        ("Versão", info.version),
    ]
    lines = [
        "| Item | Valor |",
        "| --- | --- |",
        *(f"| {item} | {value} |" for item, value in rows),
    ]
    return "\n".join(lines)


def _rerank_label(info: ServiceInfo) -> str:
    if not info.rerank_available:
        return "indisponível (modelo não instalado)"
    return "ativado por padrão" if info.rerank else "desativado por padrão"
