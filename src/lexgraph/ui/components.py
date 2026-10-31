"""Streamlit building blocks shared by the pages: answer card, sources and article card."""

from collections.abc import Sequence

import streamlit as st

from lexgraph.catalog import law_info
from lexgraph.domain import Article
from lexgraph.schemas import ChatResponse, Source
from lexgraph.ui.formatting import (
    Badge,
    article_status_badge,
    escape_markdown,
    format_int,
    format_score,
    format_seconds,
    link_citations,
    paragraphs,
    response_badges,
)

ASSISTANT_AVATAR = ":material/balance:"
USER_AVATAR = ":material/person:"
PLANALTO_LABEL = "Ler no Planalto"
BREADCRUMB = " \u203a "


def badge_row(badges: Sequence[Badge]) -> None:
    """Render badges side by side."""
    if not badges:
        return
    with st.container(horizontal=True, gap="small"):
        for badge in badges:
            st.badge(badge.label, icon=badge.icon, color=badge.color, help=badge.help)


def render_response(response: ChatResponse) -> None:
    """Answer with linked citations, how it was produced, its sources and its cost."""
    st.markdown(link_citations(escape_markdown(response.answer), response.sources))
    badge_row(response_badges(response))
    grounding = response.grounding
    if grounding is not None and grounding.unsupported_claims:
        claims = "\n".join(f"- {escape_markdown(claim)}" for claim in grounding.unsupported_claims)
        st.warning(f"Afirmações sem apoio claro nas fontes:\n\n{claims}", icon=":material/report:")
    if response.sources:
        cited = {citation.index for citation in response.citations}
        render_sources(response.sources, cited=cited)
    st.caption(_usage_line(response))


def render_sources(sources: Sequence[Source], *, cited: set[int]) -> None:
    """Sources cited by the answer first, then the other passages given to the model."""
    cited_sources = [source for source in sources if source.index in cited]
    other_sources = [source for source in sources if source.index not in cited]
    if cited_sources:
        st.caption(":material/bookmark: Fontes citadas")
        for source in cited_sources:
            _source_expander(source, icon=":material/gavel:")
    if other_sources:
        st.caption(":material/library_books: Também consultadas")
        for source in other_sources:
            _source_expander(source, icon=":material/article:")


def render_article(article: Article) -> None:
    """An article with its status, location in the law, text, notes and official link."""
    law = law_info(article.law_id)
    with st.container(border=True):
        st.markdown(f"#### {law.short_name}, {escape_markdown(article.label)}", anchors=False)
        badge_row([article_status_badge(article.status)])
        if article.heading_path:
            st.caption(BREADCRUMB.join(escape_markdown(part) for part in article.heading_path))
        if article.rubric:
            st.markdown(f"*{escape_markdown(article.rubric)}*")
        st.markdown(paragraphs(escape_markdown(article.text)))
        st.link_button(PLANALTO_LABEL, article.url, icon=":material/open_in_new:")
        if article.annotations:
            with st.expander(f"Notas de alteração ({len(article.annotations)})"):
                st.markdown("\n".join(f"- {escape_markdown(note)}" for note in article.annotations))


def _source_expander(source: Source, *, icon: str) -> None:
    with st.expander(f"[{source.index}] {source.citation}", icon=icon):
        st.markdown(paragraphs(escape_markdown(source.text)))
        scores = f"Pontuação da busca: {format_score(source.score)}"
        if source.rerank_score is not None:
            scores += f" · reranking: {format_score(source.rerank_score)}"
        st.caption(scores)
        st.link_button(PLANALTO_LABEL, source.url, icon=":material/open_in_new:")


def _usage_line(response: ChatResponse) -> str:
    parts = [f":material/schedule: {format_seconds(response.latency_ms)}"]
    if response.usage.total_tokens:
        parts.append(f":material/token: {format_int(response.usage.total_tokens)} tokens")
    if response.usage.models:
        parts.append(":material/memory: " + ", ".join(response.usage.models))
    return " · ".join(parts)
