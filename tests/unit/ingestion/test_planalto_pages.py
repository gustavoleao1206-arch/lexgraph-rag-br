"""Parsing of whole pages: excerpts of the official pages plus small synthetic pages.

Each file under ``tests/fixtures/planalto`` keeps the original windows-1252 bytes of a few
dozen lines of the page named by its prefix, cut around one defect the parser works around.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from lexgraph.catalog import SOURCES, SourcePage
from lexgraph.domain import Article, LawId
from lexgraph.ingestion.fetch import FetchMeta
from lexgraph.ingestion.planalto import parse_page

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "planalto"
FIXTURE_SOURCES = {
    "clt": "clt",
    "cdc": "cdc",
    "lgpd": "lgpd",
    "marco": "marco-civil",
    "cf88": "cf88",
}
ENACTING_CLAUSE = "O PRESIDENTE DA REPÚBLICA Faço saber que eu sanciono a seguinte Lei:"


def meta_for(source: SourcePage, final_url: str | None = None) -> FetchMeta:
    return FetchMeta(
        url=source.url,
        final_url=final_url or source.url,
        status=200,
        sha256="0" * 64,
        fetched_at="2026-10-30T14:29:32+00:00",
    )


def parse_fixture(name: str) -> dict[LawId, list[Article]]:
    source = SOURCES[FIXTURE_SOURCES[name.split("_", 1)[0]]]
    return parse_page((FIXTURES / name).read_bytes(), source, meta_for(source))


def parse_synthetic(*paragraphs: str, slug: str = "cdc") -> dict[LawId, list[Article]]:
    """Parse a page made of one ``<p>`` per paragraph (raw HTML allowed inside)."""
    source = SOURCES[slug]
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    html = f"<html><body>{body}</body></html>".encode("cp1252")
    return parse_page(html, source, meta_for(source))


def numbers(articles: list[Article]) -> list[str]:
    return [article.number for article in articles]


def by_number(articles: list[Article]) -> dict[str, Article]:
    return {article.number: article for article in articles}


# --------------------------------------------------------------------------------------------
# Excerpts of the official pages
# --------------------------------------------------------------------------------------------
def test_clt_skips_the_articles_of_the_approving_decree() -> None:
    clt = parse_fixture("clt_page_header.html")[LawId.CLT]

    assert numbers(clt) == ["1"]
    article = clt[0]
    assert article.text.startswith("Art. 1º - Esta Consolidação estatui as normas")
    assert article.heading_path == ["TÍTULO I - INTRODUÇÃO"]
    assert article.anchor == "art1."  # not "art1", the anchor of the decree's own article
    assert article.seq == 0


def test_clt_label_typos_are_normalized() -> None:
    clt = by_number(parse_fixture("clt_label_typos.html")[LawId.CLT])

    assert list(clt) == ["154", "158", "401-A", "401-B"]
    assert clt["154"].text.startswith("Art. 154 - A observância")
    assert clt["154"].heading_path == ["SEÇÃO I - DISPOSIÇÕES GERAIS"]
    assert clt["154"].annotations == ["Redação dada pela Lei nº 6.514, de 22.12.1977"]
    assert "II - colaborar com a empresa" in clt["158"].text.splitlines()[2]
    assert (clt["401-A"].label, clt["401-A"].status) == ("Art. 401-A", "vetado")


def test_clt_rubric_and_physical_anchor() -> None:
    clt = by_number(parse_fixture("clt_rubric_and_anchor.html")[LawId.CLT])

    assert clt["51"].rubric == "Falsificação de carteira de trabalho"
    assert clt["51"].anchor == "art923"  # anchors are not derived from the number
    assert clt["51"].url == f"{SOURCES['clt'].url}#art923"
    assert clt["50"].rubric is None
    assert "(Redação" not in clt["139"].text  # note with a missing ")"
    assert "Redação dada pelo Decreto-lei nº 1.535, de 13.4.1977" in clt["139"].annotations


def test_clt_struck_text_is_removed_but_kept_aside() -> None:
    (article,) = parse_fixture("clt_struck_text.html")[LawId.CLT]

    assert article.number == "790-B"
    assert article.text.splitlines()[0].endswith("pretensão objeto da perícia,")
    assert "justiça gratuita" not in article.text
    assert article.removed == ["ainda que beneficiária da justiça gratuita.", "§ 4º (sem texto)"]
    assert "Vide ADC 80" in article.annotations  # note split across two links
    assert article.text.splitlines()[-1].startswith("§ 3º O juízo não poderá")


def test_clt_parsing_stops_at_the_annex() -> None:
    (article,) = parse_fixture("clt_annex.html")[LawId.CLT]

    assert article.number == "922"
    assert article.text == (
        "Art. 922 - O disposto no art. 301 regerá somente as relações de empregos iniciadas "
        "depois da vigência desta Consolidação."
    )


def test_cdc_quoted_article_never_opens_an_article() -> None:
    cdc = parse_fixture("cdc_quoted_article.html")[LawId.CDC]

    assert numbers(cdc) == ["117", "118", "119"]
    assert cdc[0].text.splitlines()[1].startswith('"Art. 21. Aplicam-se à defesa dos direitos')
    assert cdc[0].anchor is None
    assert cdc[0].url == "https://www.planalto.gov.br/ccivil_03/leis/l8078.htm#art117"


def test_cdc_struck_ordinal_keeps_canonical_paragraph_labels() -> None:
    cdc = by_number(parse_fixture("cdc_struck_ordinal.html")[LawId.CDC])

    lines = cdc["8"].text.splitlines()
    assert lines[1].startswith("§ 1º Em se tratando de produto industrial")
    assert lines[2].startswith("§ 2º O fornecedor deverá higienizar")
    assert cdc["8"].removed == ["º", "º"]
    assert cdc["8"].heading_path == [
        "CAPÍTULO IV - Da Qualidade de Produtos e Serviços, da Prevenção e da Reparação dos Danos",
        "SEÇÃO I - Da Proteção à Saúde e Segurança",
    ]
    assert list(cdc) == ["8", "9"]


def test_lgpd_article_label_split_across_spans() -> None:
    lgpd = parse_fixture("lgpd_split_label.html")[LawId.LGPD]

    assert numbers(lgpd) == ["56", "57", "58"]
    assert {article.status for article in lgpd} == {"vetado"}
    assert lgpd[2].heading_path == [
        "SEÇÃO II - Do Conselho Nacional de Proteção de Dados Pessoais e da Privacidade"
    ]
    assert lgpd[1].url == f"{SOURCES['lgpd'].anchor_base_url}#art57"


def test_lgpd_quoted_amendment_with_nr_note() -> None:
    lgpd = by_number(parse_fixture("lgpd_quoted_amendment.html")[LawId.LGPD])

    assert list(lgpd) == ["60", "61"]
    assert "“Art. 7º …" in lgpd["60"].text.splitlines()
    assert "(Marco Civil da Internet)" in lgpd["60"].text
    assert lgpd["60"].annotations == ["NR"]


def test_marco_civil_ignores_struck_articles_of_a_rejected_measure() -> None:
    mci = by_number(parse_fixture("marco_civil_rejected_mp.html")[LawId.MCI])

    assert list(mci) == ["8", "9"]
    assert any(text.startswith("Art. 8º-A Aos usuários") for text in mci["8"].removed)
    assert mci["8"].annotations == ["Rejeitada", "Rejeitada"]
    assert mci["8"].url == f"{SOURCES['marco-civil'].url}#art8"


def test_cf_preamble_is_stored_as_an_article() -> None:
    parsed = parse_fixture("cf88_preamble.html")
    preamble, first = parsed[LawId.CF]

    assert (preamble.number, preamble.label, preamble.seq) == ("preambulo", "Preâmbulo", 0)
    assert preamble.text.startswith("Nós, representantes do povo brasileiro")
    assert preamble.url == SOURCES["cf88"].url
    assert first.heading_path == ["TÍTULO I - Dos Princípios Fundamentais"]
    assert first.anchor == "cf-88-parte-1-titulo-1-artigo-1"
    assert first.text.splitlines()[1:] == ["I - a soberania;", "II - a cidadania;"]
    assert parsed[LawId.ADCT] == []


def test_cf_dash_written_as_a_numeric_reference() -> None:
    (article,) = parse_fixture("cf88_dash_entity.html")[LawId.CF]

    lines = article.text.splitlines()
    assert len(lines) == 7  # caput and six incisos, none glued to the previous one
    assert "IX \u2013 remuneração do trabalho noturno superior à do diurno;" in lines


def test_cf_units_missing_their_sign_or_dash() -> None:
    articles = by_number(parse_fixture("cf88_units_without_signs.html")[LawId.CF])

    lines = articles["215"].text.splitlines()
    assert lines[2].startswith("§ 2º A lei disporá sobre a fixação")
    assert lines[3].startswith("§ 3º A lei estabelecerá o Plano Nacional de Cultura")
    assert lines[4] == "I - defesa e valorização do patrimônio cultural brasileiro;"
    assert articles["215"].annotations[0] == "Regulamento"
    assert list(articles) == ["215", "216"]


def test_cf_article_revoked_together_with_its_paragraph() -> None:
    cf = by_number(parse_fixture("cf88_revoked_with_units.html")[LawId.CF])

    assert list(cf) == ["116", "117", "118"]
    # "Art. 117. e Parágrafo único. (Revogados ...)": the note revokes the paragraph too.
    assert (cf["117"].status, cf["117"].text) == ("revogado", "Art. 117.")
    assert cf["117"].removed == ["Parágrafo único (revogado)"]
    assert cf["117"].annotations == ["Revogados pela Emenda Constitucional nº 24, de 1999"]
    # Art. 116 lost only its paragraph and keeps its caput.
    assert cf["116"].status == "vigente"
    assert cf["116"].text.startswith("Art. 116. Nas Varas do Trabalho")
    assert cf["116"].removed == ["Parágrafo único (revogado)"]
    assert cf["118"].heading_path == ["SEÇÃO VI - DOS TRIBUNAIS E JUÍZES ELEITORAIS"]


def test_cf_line_closed_early_by_misnested_font_is_rejoined() -> None:
    (article,) = parse_fixture("cf88_misnested_font.html")[LawId.CF]

    assert "d) Presidente e diretores do banco central;" in article.text.splitlines()
    assert article.heading_path == ["SEÇÃO IV - DO SENADO FEDERAL"]


def test_cf_page_switches_to_the_adct() -> None:
    parsed = parse_fixture("cf88_adct_switch.html")

    assert numbers(parsed[LawId.CF]) == ["250"]
    first, second = parsed[LawId.ADCT]
    assert (first.number, first.label, first.seq) == ("1", "Art. 1º", 0)
    assert first.heading_path == ["ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS"]
    assert first.url == f"{SOURCES['cf88'].url}#sadctart1"
    assert second.text.splitlines()[1].startswith("§ 1º - Será assegurada gratuidade")


def test_premature_document_end_does_not_truncate_the_page() -> None:
    raw = (FIXTURES / "cf88_base_premature_end.html").read_bytes()
    assert "Art. 25" not in BeautifulSoup(raw.decode("cp1252"), "lxml").get_text()

    cf = parse_fixture("cf88_base_premature_end.html")[LawId.CF]

    assert numbers(cf) == ["24", "25"]
    assert cf[1].heading_path == ["CAPÍTULO III - DOS ESTADOS FEDERADOS"]


# --------------------------------------------------------------------------------------------
# Synthetic pages
# --------------------------------------------------------------------------------------------
def test_unknown_source_is_rejected() -> None:
    source = SourcePage(slug="decreto", url="https://example.gov.br/d.htm", law_ids=(LawId.CLT,))

    with pytest.raises(ValueError, match="no parsing profile"):
        parse_page(b"<p>Art. 1. Texto.</p>", source, meta_for(source))


def test_text_before_the_start_marker_is_ignored() -> None:
    parsed = parse_synthetic("Art. 1º Texto de outra norma.", "Brasília, 1 de maio de 2000.")

    assert parsed == {LawId.CDC: []}


@pytest.mark.parametrize(
    ("line", "status", "text"),
    [
        ("Art. 1º (VETADO).", "vetado", "Art. 1º (VETADO)."),
        ("Art. 1º (Revogado pela Lei nº 9.999, de 2000)", "revogado", "Art. 1º."),
        ("Art. 1º (Revogado).", "revogado", "Art. 1º."),
        ("Art. 1º Texto em vigor.", "vigente", "Art. 1º Texto em vigor."),
        ("Art. 1º", "vigente", "Art. 1º."),  # no evidence of revocation: left to the validator
        ("Art. 1º e §§ 1º e 2º (Revogados pela Lei nº 9.999, de 2000)", "revogado", "Art. 1º."),
        ("Art. 1º e parágrafo único (Suprimido pela Lei nº 2, de 2001)", "revogado", "Art. 1º."),
        # Without a note revoking it, a list of units is not evidence: left to the validator.
        ("Art. 1º e parágrafo único.", "vigente", "Art. 1º e parágrafo único."),
    ],
)
def test_article_status(line: str, status: str, text: str) -> None:
    (article,) = parse_synthetic(ENACTING_CLAUSE, line)[LawId.CDC]

    assert (article.status, article.text) == (status, text)


def test_revoked_and_vetoed_units_are_dropped() -> None:
    (article,) = parse_synthetic(
        ENACTING_CLAUSE,
        "Art. 1º São direitos:",
        "I - (revogado);",
        "II - (VETADO);",
        "III - <a href='L2.htm'>(Revogado pela Lei nº 2, de 2001)</a>",
        "IV - o acesso;",
        "§ 1º (VETADO).",
    )[LawId.CDC]

    assert article.text == "Art. 1º São direitos:\nIV - o acesso;"
    assert article.removed == ["I (revogado)", "II (vetado)", "III (revogado)", "§ 1º (vetado)"]
    assert article.status == "vigente"


def test_suffixed_inciso_missing_its_dash_keeps_its_suffix() -> None:
    # CF art. 109: "V-A as causas" is inciso V-A without its dash, not a second inciso V.
    (article,) = parse_synthetic(
        ENACTING_CLAUSE,
        "Art. 1º Aos juízes federais compete processar e julgar:",
        "V - os crimes previstos em tratado ou convenção internacional;",
        "V-A as causas relativas a direitos humanos a que se refere o § 5º deste artigo;",
        "VI - os crimes contra a organização do trabalho;",
    )[LawId.CDC]

    assert article.text.splitlines()[1:] == [
        "V - os crimes previstos em tratado ou convenção internacional;",
        "V-A - as causas relativas a direitos humanos a que se refere o § 5º deste artigo;",
        "VI - os crimes contra a organização do trabalho;",
    ]


def test_headings_build_the_heading_path() -> None:
    cdc = parse_synthetic(
        ENACTING_CLAUSE,
        "TÍTULO I - Dos Direitos",
        "CAPÍTULO I",
        "Disposições Gerais",
        "Art. 1º Texto.",
        "CAPÍTULO II",
        "Da Política Nacional",
        "DISPOSIÇÕES PRELIMINARES",
        "Art. 2º Texto.",
        "TÍTULO II DAS INFRAÇÕES",
        "Art. 3º Texto.",
        "Seção I",
        "Do Tribunal Superior",
        "do Trabalho",  # title broken over two lines
        "Art. 4º Texto.",
    )[LawId.CDC]

    assert [article.heading_path for article in cdc] == [
        ["TÍTULO I - Dos Direitos", "CAPÍTULO I - Disposições Gerais"],
        [
            "TÍTULO I - Dos Direitos",
            "CAPÍTULO II - Da Política Nacional",
            "DISPOSIÇÕES PRELIMINARES",
        ],
        ["TÍTULO II - DAS INFRAÇÕES"],
        ["TÍTULO II - DAS INFRAÇÕES", "SEÇÃO I - Do Tribunal Superior do Trabalho"],
    ]


def test_text_outside_articles_is_a_rubric_or_dropped() -> None:
    first, second = parse_synthetic(
        ENACTING_CLAUSE,
        "<a href='D1.htm'>(Vide Decreto nº 1, de 1991)</a>",  # a note about the whole page
        "CAPÍTULO V",
        "Do Repouso",
        "Trabalho aos domingos",
        "Art. 67 - Será assegurado um descanso semanal.",
        "Art. 68 - O trabalho em domingo depende de permissão.",
        "CAPÍTULO VI",
        "Das Disposições Finais",
        "Texto solto que não pertence a nenhum artigo.",
        "Brasília, 1 de maio de 1943.",
    )[LawId.CDC]

    assert first.rubric == "Trabalho aos domingos"
    assert first.annotations == []
    assert second.rubric is None
    assert second.text == "Art. 68 - O trabalho em domingo depende de permissão."


def test_unfinished_sentence_is_rejoined_but_new_sentences_are_not() -> None:
    (article,) = parse_synthetic(
        ENACTING_CLAUSE, "Art. 1º O presidente e os diretores do", "banco central.", "Pena - multa."
    )[LawId.CDC]

    assert article.text == "Art. 1º O presidente e os diretores do banco central.\nPena - multa."


def test_unclosed_quote_is_reset_after_sixty_lines() -> None:
    filler = [f"linha {n} do texto citado" for n in range(70)]
    cdc = parse_synthetic(
        ENACTING_CLAUSE, "Art. 1º Altera a lei:", "“Art. 9º Novo", *filler, "Art. 2º Fim."
    )[LawId.CDC]

    assert numbers(cdc) == ["1", "2"]
    assert cdc[0].text.splitlines()[1] == "“Art. 9º Novo"


def test_anchors_before_the_law_text_do_not_leak() -> None:
    (article,) = parse_synthetic('<a name="topo"></a>', ENACTING_CLAUSE, "Art. 1º Texto.")[
        LawId.CDC
    ]

    assert article.anchor is None
    assert article.url == f"{SOURCES['cdc'].anchor_base_url}#art1"


def test_links_point_to_the_page_actually_served() -> None:
    source = SOURCES["marco-civil"]
    html = f'<p>{ENACTING_CLAUSE}</p><p><a name="art1"></a>Art. 1º Texto.</p>'.encode("cp1252")
    moved = "https://www.planalto.gov.br/ccivil_03/novo/l12965.htm"

    (article,) = parse_page(html, source, meta_for(source, final_url=moved))[LawId.MCI]

    assert article.url == f"{moved}#art1"


def test_anchor_names_are_percent_encoded() -> None:
    source = SOURCES["clt"]
    html = (
        "<p>CONSOLIDAÇÃO DAS LEIS DO TRABALHO</p><p>Art. 75-C. Texto:</p>"
        '<p><a name="art75c§1"></a>§ 1º Texto.</p>'
    ).encode("cp1252")

    (article,) = parse_page(html, source, meta_for(source))[LawId.CLT]

    assert article.anchor == "art75c§1"
    assert article.url == f"{source.url}#art75c%C2%A71"


def test_adct_fallback_link_uses_the_adct_anchor_prefix() -> None:
    parsed = parse_synthetic(
        "PREÂMBULO",
        "Art. 1º Texto.",
        "Brasília, 5 de outubro de 1988.",
        "ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS",
        "Art. 1º Texto transitório.",
        "Art. 2º-A Outro texto.",
        slug="cf88",
    )

    base = SOURCES["cf88"].anchor_base_url
    assert [article.url for article in parsed[LawId.CF]] == [f"{base}#art1"]
    assert [article.url for article in parsed[LawId.ADCT]] == [
        f"{base}#adctart1",
        f"{base}#adctart2a",
    ]
