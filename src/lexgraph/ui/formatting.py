"""Text helpers for the interface: Markdown escaping, citation links, badges and PT-BR copy.

Everything here is pure, so the presentation rules are unit-tested without Streamlit.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from lexgraph.agent.citations import cited_indices
from lexgraph.domain import ArticleStatus
from lexgraph.schemas import ChatResponse, Source
from lexgraph.ui.client import ApiConnectionError, ApiError

BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "primary"]

RETRIEVAL_MODES = {"hybrid": "híbrida (densa + BM25)", "dense": "densa", "sparse": "lexical (BM25)"}

# Text in square brackets, without nested brackets: a candidate citation marker.
_BRACKETS_RE = re.compile(r"\[[^\[\]]*\]")
_ARTICLE_NUMBER_RE = re.compile(
    r"^(?:art(?:igo)?\.?\s*)?(?P<number>\d+)\s*[º°o]?\s*[-\u2013]?\s*(?P<suffix>[a-z]{1,2})?\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Badge:
    """A small colored label shown under an answer."""

    label: str
    icon: str
    color: BadgeColor
    help: str


def escape_markdown(text: str) -> str:
    """Escape ``$`` so Streamlit does not render amounts such as ``R$ 100`` as LaTeX."""
    return text.replace("$", r"\$")


def paragraphs(text: str) -> str:
    """Keep the line breaks of a legal text when rendered as Markdown."""
    return "\n\n".join(line.strip() for line in text.splitlines() if line.strip())


def link_citations(answer: str, sources: Sequence[Source]) -> str:
    """Turn the ``[n]`` markers of an answer into links to the cited articles.

    Markers are read with the agent's own grammar (``[1, 2]``, ``[1; 3]``, ``[1 e 2]``,
    ``[1-3]``), so the links always match the citations the API reports. A marker naming
    several sources becomes one link per source; numbers without a matching source are left
    out, and a marker with none stays plain text, so it never points to the wrong article.
    """
    by_index = {source.index: source for source in sources}

    def replace(match: re.Match[str]) -> str:
        # The candidate has a single pair of brackets, so the agent finds numbers in it only
        # when the whole candidate is a marker.
        indices = cited_indices(match.group())
        links = [_citation_link(by_index[index]) for index in indices if index in by_index]
        return "".join(links) if links else match.group()

    return _BRACKETS_RE.sub(replace, answer)


def response_badges(response: ChatResponse) -> list[Badge]:
    """Badges that summarize how an answer was produced."""
    badges = _route_badges(response)
    if response.rewrites:
        badges.append(
            Badge(
                "Busca reformulada",
                ":material/manage_search:",
                "blue",
                "A consulta foi reescrita para encontrar trechos mais relevantes.",
            )
        )
    if response.regenerations:
        badges.append(
            Badge(
                "Resposta revisada",
                ":material/autorenew:",
                "violet",
                "A primeira versão foi descartada pela autoverificação e gerada novamente.",
            )
        )
    return badges


def article_status_badge(status: ArticleStatus) -> Badge:
    """Badge for the status of an article."""
    match status:
        case "vigente":
            return Badge("Em vigor", ":material/check_circle:", "green", "Texto vigente.")
        case "revogado":
            return Badge("Revogado", ":material/cancel:", "red", "Artigo revogado.")
        case "vetado":
            return Badge("Vetado", ":material/block:", "gray", "Dispositivo vetado.")


def normalize_article_number(raw: str) -> str | None:
    """Normalize what a user types as an article number, or ``None`` if it is not one.

    ``"7"``, ``"7º"`` and ``"art. 7"`` give ``"7"``; ``"59a"`` and ``"59-A"`` (with a hyphen
    or an en dash) give ``"59-A"``; ``"preâmbulo"`` gives ``"preambulo"``.
    """
    text = raw.strip()
    if text.casefold() in {"preambulo", "preâmbulo"}:
        return "preambulo"
    match = _ARTICLE_NUMBER_RE.match(text)
    if match is None:
        return None
    number = str(int(match["number"]))
    suffix = match["suffix"]
    return f"{number}-{suffix.upper()}" if suffix else number


def retrieval_mode_label(mode: str) -> str:
    """Portuguese name of a retrieval mode reported by the API."""
    return RETRIEVAL_MODES.get(mode, mode)


def format_int(value: int) -> str:
    """Brazilian thousands separator: ``12345 -> "12.345"``."""
    return f"{value:,}".replace(",", ".")


def format_seconds(milliseconds: int) -> str:
    """Duration with one decimal and a comma: ``3240 -> "3,2 s"``."""
    return f"{milliseconds / 1000:.1f} s".replace(".", ",")


def format_score(value: float) -> str:
    """Retrieval score with three decimals and a comma: ``0.03125 -> "0,031"``."""
    return f"{value:.3f}".replace(".", ",")


def format_date(timestamp: str | None) -> str:
    """ISO 8601 timestamp as ``dd/mm/yyyy``; the input itself if it cannot be parsed."""
    if not timestamp:
        return "—"
    try:
        return datetime.fromisoformat(timestamp).strftime("%d/%m/%Y")
    except ValueError:
        return timestamp


def describe_api_error(error: ApiError) -> str:
    """Message in Portuguese for a failed call to the API."""
    if isinstance(error, ApiConnectionError):
        return (
            "Não foi possível falar com a API (conexão recusada, interrompida ou lenta demais). "
            "Verifique se o servidor está em execução e tente novamente."
        )
    match error.status_code:
        case None:
            return "A API enviou uma resposta inesperada. Tente novamente."
        case 401 | 403:
            return "A API recusou a chave de acesso. Confira a variável LEXGRAPH_API_KEY."
        case 422:
            return "A API não aceitou a pergunta. Verifique o tamanho do texto e tente de novo."
        case 429:
            return "Muitas perguntas em sequência. Aguarde alguns instantes e tente novamente."
        case code if code >= 500:
            return "A API encontrou um erro interno. Tente novamente em instantes."
        case code:
            return f"A API recusou a solicitação (erro {code})."


def _citation_link(source: Source) -> str:
    title = source.citation.replace('"', "'")
    return f'[[{source.index}]](<{source.url}> "{title}")'


def _route_badges(response: ChatResponse) -> list[Badge]:
    match response.route:
        case "answered":
            return [_grounding_badge(response)]
        case "no_context":
            return [
                Badge(
                    "Sem base no acervo",
                    ":material/search_off:",
                    "orange",
                    "Nenhum trecho relevante foi encontrado nas leis do acervo.",
                )
            ]
        case "out_of_scope":
            return [
                Badge(
                    "Fora do escopo",
                    ":material/block:",
                    "gray",
                    "A pergunta não trata da legislação coberta pelo acervo.",
                )
            ]
        case "unavailable":
            return [
                Badge(
                    "Modelo indisponível",
                    ":material/cloud_off:",
                    "red",
                    "Nenhum modelo de linguagem conseguiu redigir a resposta agora.",
                )
            ]
        case "smalltalk":
            return []


def _grounding_badge(response: ChatResponse) -> Badge:
    grounding = response.grounding
    if grounding is None:
        return Badge(
            "Sem autoverificação",
            ":material/help:",
            "gray",
            "A resposta não passou pela checagem automática contra as fontes: ela está "
            "desativada no servidor ou não pôde ser concluída desta vez.",
        )
    if grounding.grounded:
        return Badge(
            "Fundamentada nas fontes",
            ":material/verified:",
            "green",
            "Uma segunda checagem confirmou que as afirmações estão apoiadas nas fontes.",
        )
    return Badge(
        "Confira as fontes",
        ":material/warning:",
        "orange",
        "A checagem automática encontrou afirmações sem apoio claro nas fontes.",
    )
