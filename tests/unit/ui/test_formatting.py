"""Presentation rules of the interface: citations, numbers, badges and error messages."""

import re

import pytest

from lexgraph.agent.citations import build_citations
from lexgraph.domain import ArticleStatus
from lexgraph.schemas import Grounding, Route, Source
from lexgraph.ui.client import ApiConnectionError, ApiError
from lexgraph.ui.formatting import (
    article_status_badge,
    describe_api_error,
    escape_markdown,
    format_date,
    format_int,
    format_score,
    format_seconds,
    link_citations,
    normalize_article_number,
    paragraphs,
    response_badges,
    retrieval_mode_label,
)
from tests.unit.api.fakes import sample_response, sample_source


def source(index: int, citation: str = "") -> Source:
    return sample_source(index).model_copy(
        update={
            "citation": citation or f"CDC, art. {index}",
            "url": f"https://planalto.test/l8078.htm#art{index}",
        }
    )


def link(index: int) -> str:
    return f'[[{index}]](<https://planalto.test/l8078.htm#art{index}> "CDC, art. {index}")'


SOURCES = [source(1), source(2), source(3)]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Pode desistir [1].", f"Pode desistir {link(1)}."),
        ("Ver [1, 3].", f"Ver {link(1)}{link(3)}."),
        ("Ver [1][2].", f"Ver {link(1)}{link(2)}."),
        ("Ver [1-3].", f"Ver {link(1)}{link(2)}{link(3)}."),
        ("Ver [2\u20133].", f"Ver {link(2)}{link(3)}."),
        ("Ver [3-1].", f"Ver {link(3)}{link(1)}."),
        ("Ver [2, 9].", f"Ver {link(2)}."),
        ("Ver [2-9].", f"Ver {link(2)}{link(3)}."),
        ("Ver [1; 3].", f"Ver {link(1)}{link(3)}."),
        ("Ver [1 e 2].", f"Ver {link(1)}{link(2)}."),
        ("Ver [ 2 ].", f"Ver {link(2)}."),
        ("Ver [1, 1].", f"Ver {link(1)}."),
        ("Ver [9].", "Ver [9]."),
        ("Ver [2-99999999].", "Ver [2-99999999]."),
        ("Sem marcadores.", "Sem marcadores."),
        ("Lista [a], [1a] e [nota 1].", "Lista [a], [1a] e [nota 1]."),
    ],
)
def test_link_citations(answer: str, expected: str) -> None:
    assert link_citations(answer, SOURCES) == expected


def test_linked_sources_match_the_citations_the_api_reports() -> None:
    answer = "Pode [1; 3]. Também [2 e 9], [ 2 ] e [1-2]. Nunca [12345] nem [x]."

    linked = re.findall(r"\[\[(\d+)\]\]", link_citations(answer, SOURCES))

    cited = [citation.index for citation in build_citations(answer, SOURCES)]
    assert sorted(set(map(int, linked))) == sorted(cited) == [1, 2, 3]


def test_link_citations_without_sources_leaves_markers() -> None:
    assert link_citations("Ver [1].", []) == "Ver [1]."


def test_link_citations_keeps_titles_valid() -> None:
    quoted = source(1, citation='CF, art. 5º, "caput"')

    assert link_citations("[1]", [quoted]) == (
        "[[1]](<https://planalto.test/l8078.htm#art1> \"CF, art. 5º, 'caput'\")"
    )


def test_escape_markdown_protects_currency_from_latex() -> None:
    assert escape_markdown("Multa de R$ 100 a R$ 200.") == r"Multa de R\$ 100 a R\$ 200."


def test_paragraphs_keeps_line_breaks() -> None:
    assert paragraphs("Art. 1º\n  I - um;\n\nII - dois.") == "Art. 1º\n\nI - um;\n\nII - dois."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", "7"),
        (" 7º ", "7"),
        ("7°", "7"),
        ("7o", "7"),
        ("art. 7º", "7"),
        ("Art 482", "482"),
        ("artigo 5", "5"),
        ("059", "59"),
        ("59-A", "59-A"),
        ("59\u2013A", "59-A"),
        ("59a", "59-A"),
        ("59 a", "59-A"),
        ("art. 611-B", "611-B"),
        ("preâmbulo", "preambulo"),
        ("Preambulo", "preambulo"),
        ("", None),
        ("abc", None),
        ("5, XI", None),
        ("59-ABC", None),
    ],
)
def test_normalize_article_number(raw: str, expected: str | None) -> None:
    assert normalize_article_number(raw) == expected


def test_number_formats_follow_brazilian_conventions() -> None:
    assert format_int(1234567) == "1.234.567"
    assert format_int(12) == "12"
    assert format_seconds(3240) == "3,2 s"
    assert format_seconds(0) == "0,0 s"
    assert format_score(0.03125) == "0,031"


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-10-30T11:58:00+00:00", "30/10/2026"),
        ("2026-01-02", "02/01/2026"),
        (None, "—"),
        ("", "—"),
        ("ontem", "ontem"),
    ],
)
def test_format_date(timestamp: str | None, expected: str) -> None:
    assert format_date(timestamp) == expected


def test_retrieval_mode_label() -> None:
    assert retrieval_mode_label("hybrid") == "híbrida (densa + BM25)"
    assert retrieval_mode_label("sparse") == "lexical (BM25)"
    assert retrieval_mode_label("experimental") == "experimental"


@pytest.mark.parametrize(
    ("route", "grounding", "labels"),
    [
        ("answered", Grounding(grounded=True), ["Fundamentada nas fontes"]),
        ("answered", Grounding(grounded=False, unsupported_claims=["x"]), ["Confira as fontes"]),
        ("answered", None, ["Sem autoverificação"]),
        ("no_context", None, ["Sem base no acervo"]),
        ("out_of_scope", None, ["Fora do escopo"]),
        ("unavailable", None, ["Modelo indisponível"]),
        ("smalltalk", None, []),
    ],
)
def test_response_badges_by_route(
    route: Route, grounding: Grounding | None, labels: list[str]
) -> None:
    response = sample_response().model_copy(update={"route": route, "grounding": grounding})

    assert [badge.label for badge in response_badges(response)] == labels


def test_unverified_answers_do_not_claim_that_verification_is_off() -> None:
    response = sample_response().model_copy(update={"grounding": None})

    [badge] = response_badges(response)

    assert badge.help == (
        "A resposta não passou pela checagem automática contra as fontes: ela está "
        "desativada no servidor ou não pôde ser concluída desta vez."
    )


def test_response_badges_show_rewrites_and_regenerations() -> None:
    response = sample_response().model_copy(update={"rewrites": 1, "regenerations": 2})

    assert [badge.label for badge in response_badges(response)] == [
        "Fundamentada nas fontes",
        "Busca reformulada",
        "Resposta revisada",
    ]


@pytest.mark.parametrize(
    ("status", "label", "color"),
    [
        ("vigente", "Em vigor", "green"),
        ("revogado", "Revogado", "red"),
        ("vetado", "Vetado", "gray"),
    ],
)
def test_article_status_badge(status: ArticleStatus, label: str, color: str) -> None:
    badge = article_status_badge(status)

    assert (badge.label, badge.color) == (label, color)


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (ApiConnectionError("refused"), "Não foi possível falar com a API"),
        (ApiError("bad event"), "resposta inesperada"),
        (ApiError("401", status_code=401), "LEXGRAPH_API_KEY"),
        (ApiError("403", status_code=403), "LEXGRAPH_API_KEY"),
        (ApiError("422", status_code=422), "tamanho do texto"),
        (ApiError("429", status_code=429), "Aguarde"),
        (ApiError("503", status_code=503), "erro interno"),
        (ApiError("404", status_code=404), "erro 404"),
    ],
)
def test_describe_api_error(error: ApiError, fragment: str) -> None:
    assert fragment in describe_api_error(error)
