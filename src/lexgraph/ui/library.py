"""Acervo page: the laws in the corpus and a lookup of individual articles."""

from collections.abc import Sequence

import streamlit as st

from lexgraph.catalog import law_info
from lexgraph.domain import LawId, make_article_id, split_article_id
from lexgraph.schemas import LawSummary
from lexgraph.ui.client import ApiClient, ApiError
from lexgraph.ui.components import render_article
from lexgraph.ui.formatting import (
    describe_api_error,
    format_date,
    format_int,
    normalize_article_number,
)

LOOKUP_KEY = "library_article_id"


def render(client: ApiClient) -> None:
    """Render the Acervo page."""
    st.set_page_config(layout="wide")
    st.title("Acervo", anchor=False)
    st.caption(
        "Leis federais consultadas pelo assistente, extraídas dos textos compilados "
        "publicados no portal da legislação do Planalto."
    )
    try:
        laws = client.laws()
    except ApiError as error:
        st.error(describe_api_error(error), icon=":material/cloud_off:")
        return
    if not laws:
        st.info(
            "O acervo ainda não foi gerado no servidor da API.", icon=":material/hourglass_empty:"
        )
        return
    _overview(laws)
    _laws_table(laws)
    _article_lookup(client, laws)


def _overview(laws: Sequence[LawSummary]) -> None:
    fetched = [law.fetched_at for law in laws if law.fetched_at]
    columns = st.columns(4)
    columns[0].metric("Leis", len(laws), border=True)
    columns[1].metric("Artigos", format_int(sum(law.articles for law in laws)), border=True)
    columns[2].metric("Em vigor", format_int(sum(law.in_force for law in laws)), border=True)
    columns[3].metric("Última coleta", format_date(max(fetched, default=None)), border=True)


def _laws_table(laws: Sequence[LawSummary]) -> None:
    rows = [
        {
            "Lei": law.short_name,
            "Nome": law.name,
            "Norma": law.official_name,
            "Artigos": law.articles,
            "Em vigor": law.in_force,
            "Revogados": law.revoked,
            "Vetados": law.vetoed,
            "Coleta": format_date(law.fetched_at),
            "Texto oficial": law.source_url,
        }
        for law in laws
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "Texto oficial": st.column_config.LinkColumn(display_text="Abrir no Planalto"),
            "Artigos": st.column_config.NumberColumn(format="localized"),
            "Em vigor": st.column_config.NumberColumn(format="localized"),
        },
    )


def _article_lookup(client: ApiClient, laws: Sequence[LawSummary]) -> None:
    st.subheader("Consultar artigo", anchor=False)
    with st.form("article_lookup"):
        law_column, number_column = st.columns([3, 2], vertical_alignment="bottom")
        law_id = law_column.selectbox(
            "Lei",
            [law.law_id for law in laws],
            format_func=lambda law_id: f"{law_info(law_id).short_name} — {law_info(law_id).name}",
        )
        number = number_column.text_input("Artigo", placeholder="Ex.: 7, 59-A ou 482")
        submitted = st.form_submit_button("Consultar", icon=":material/search:", type="primary")
    if submitted:
        _remember_lookup(law_id, number)
    article_id: str | None = st.session_state.get(LOOKUP_KEY)
    if article_id is not None:
        _show_article(client, article_id)


def _remember_lookup(law_id: LawId | None, number: str | None) -> None:
    normalized = normalize_article_number(number or "")
    if law_id is None or normalized is None:
        st.session_state[LOOKUP_KEY] = None
        st.warning("Informe o número do artigo, como 7, 59-A ou 482.", icon=":material/edit:")
        return
    st.session_state[LOOKUP_KEY] = make_article_id(law_id, normalized)


def _show_article(client: ApiClient, article_id: str) -> None:
    try:
        article = client.article(article_id)
    except ApiError as error:
        st.error(describe_api_error(error), icon=":material/cloud_off:")
        return
    if article is None:
        law_id, number = split_article_id(article_id)
        st.warning(
            f"O artigo {number} não está no acervo de {law_info(law_id).short_name}.",
            icon=":material/search_off:",
        )
        return
    render_article(article)
