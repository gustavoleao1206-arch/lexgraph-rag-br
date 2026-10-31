"""Catalog of the laws in the corpus and of the official pages they are parsed from.

Texts come from the compiled ("compilado") pages of planalto.gov.br, the official portal of
Brazilian federal legislation. Legislative texts are not protected by copyright in Brazil
(Lei 9.610/1998, art. 8º, IV). URNs follow the LexML naming scheme.
"""

from pydantic import BaseModel, ConfigDict

from lexgraph.domain import LawId


class LawInfo(BaseModel):
    """Static metadata about one law."""

    model_config = ConfigDict(frozen=True)

    law_id: LawId
    short_name: str
    name: str
    official_name: str
    urn: str
    source: str
    expected_last_article: int
    aliases: tuple[str, ...] = ()


class SourcePage(BaseModel):
    """One official page; the Constitution page holds two laws (CF and ADCT)."""

    model_config = ConfigDict(frozen=True)

    slug: str
    url: str
    anchor_base_url: str | None = None
    law_ids: tuple[LawId, ...]


SOURCES: dict[str, SourcePage] = {
    page.slug: page
    for page in (
        SourcePage(
            slug="clt",
            url="https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm",
            law_ids=(LawId.CLT,),
        ),
        SourcePage(
            slug="cdc",
            url="https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
            anchor_base_url="https://www.planalto.gov.br/ccivil_03/leis/l8078.htm",
            law_ids=(LawId.CDC,),
        ),
        SourcePage(
            slug="lgpd",
            url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm",
            anchor_base_url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
            law_ids=(LawId.LGPD,),
        ),
        SourcePage(
            slug="marco-civil",
            url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm",
            law_ids=(LawId.MCI,),
        ),
        SourcePage(
            slug="cf88",
            url="https://www.planalto.gov.br/ccivil_03/constituicao/constituicaocompilado.htm",
            anchor_base_url="https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
            law_ids=(LawId.CF, LawId.ADCT),
        ),
    )
}

LAWS: dict[LawId, LawInfo] = {
    law.law_id: law
    for law in (
        LawInfo(
            law_id=LawId.CLT,
            short_name="CLT",
            name="Consolidação das Leis do Trabalho",
            official_name="Decreto-Lei nº 5.452, de 1º de maio de 1943",
            urn="urn:lex:br:federal:decreto.lei:1943-05-01;5452",
            source="clt",
            expected_last_article=922,
            aliases=("clt", "consolidação das leis do trabalho", "leis do trabalho"),
        ),
        LawInfo(
            law_id=LawId.CDC,
            short_name="CDC",
            name="Código de Defesa do Consumidor",
            official_name="Lei nº 8.078, de 11 de setembro de 1990",
            urn="urn:lex:br:federal:lei:1990-09-11;8078",
            source="cdc",
            expected_last_article=119,
            aliases=("cdc", "código de defesa do consumidor", "código do consumidor"),
        ),
        LawInfo(
            law_id=LawId.LGPD,
            short_name="LGPD",
            name="Lei Geral de Proteção de Dados Pessoais",
            official_name="Lei nº 13.709, de 14 de agosto de 2018",
            urn="urn:lex:br:federal:lei:2018-08-14;13709",
            source="lgpd",
            expected_last_article=65,
            aliases=("lgpd", "lei geral de proteção de dados", "lei de proteção de dados"),
        ),
        LawInfo(
            law_id=LawId.MCI,
            short_name="Marco Civil",
            name="Marco Civil da Internet",
            official_name="Lei nº 12.965, de 23 de abril de 2014",
            urn="urn:lex:br:federal:lei:2014-04-23;12965",
            source="marco-civil",
            expected_last_article=32,
            aliases=("marco civil", "marco civil da internet", "mci"),
        ),
        LawInfo(
            law_id=LawId.CF,
            short_name="CF",
            name="Constituição Federal",
            official_name="Constituição da República Federativa do Brasil de 1988",
            urn="urn:lex:br:federal:constituicao:1988-10-05;1988",
            source="cf88",
            expected_last_article=250,
            aliases=("constituição", "constituição federal", "cf", "cf/88", "carta magna"),
        ),
        LawInfo(
            law_id=LawId.ADCT,
            short_name="ADCT",
            name="Ato das Disposições Constitucionais Transitórias",
            official_name="ADCT da Constituição de 1988",
            urn="urn:lex:br:federal:ato.disposicoes.constitucionais.transitorias:1988-10-05;1988",
            source="cf88",
            expected_last_article=138,
            aliases=("adct", "disposições constitucionais transitórias"),
        ),
    )
}


def law_info(law_id: LawId | str) -> LawInfo:
    """Return the catalog entry for ``law_id`` (accepts the enum or its string value)."""
    return LAWS[LawId(law_id)]


def scope_description() -> str:
    """Human-readable list of the laws in the corpus, used in prompts and refusals."""
    return "; ".join(f"{law.short_name} ({law.name})" for law in LAWS.values())
