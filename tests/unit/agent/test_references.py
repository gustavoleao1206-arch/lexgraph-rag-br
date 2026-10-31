"""Detection of laws and explicit article references in questions."""

import pytest

from lexgraph.agent.references import (
    ArticleMention,
    article_ids,
    detect_laws,
    find_article_mentions,
    fold,
    normalize_article_number,
)
from lexgraph.domain import LawId


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("O que diz o art. 7º da CLT?", [("CLT", "7")]),
        ("artigo 482 da Consolidação", [("CLT", "482")]),
        ("artigo 482 da Consolidação das Leis do Trabalho", [("CLT", "482")]),
        ("CDC art 49", [("CDC", "49")]),
        ("art. 5º, inciso XI, da Constituição", [("CF", "5")]),
        ("arts. 7 e 8 da LGPD", [("LGPD", "7"), ("LGPD", "8")]),
        ("o art. 59-A trata de quê?", [(None, "59-A")]),
        ("art. 10 do ADCT", [("ADCT", "10")]),
        ("Marco Civil, art. 19", [("MCI", "19")]),
        ("art. 19 do Marco Civil da Internet", [("MCI", "19")]),
        ("LGPD, arts. 7º, 8º e 11", [("LGPD", "7"), ("LGPD", "8"), ("LGPD", "11")]),
        ("arts. 482 a 485 da CLT", [("CLT", n) for n in ("482", "483", "484", "485")]),
        ("art. 7º, XVII, da CF", [("CF", "7")]),
        ("art. 5º, LXXIX, da CF/88", [("CF", "5")]),
        ("art. 611-A, III, da CLT", [("CLT", "611-A")]),
        ("art. 59\N{EN DASH}A da CLT", [("CLT", "59-A")]),
        ("Art. 401A da CLT", [("CLT", "401-A")]),
        ("art. 1o da LGPD", [("LGPD", "1")]),
        ("art. 5.º da CF", [("CF", "5")]),
        ("ARTIGO 5 DA CONSTITUIÇÃO FEDERAL", [("CF", "5")]),
        ("art. 18 da Lei 13.709/2018", [("LGPD", "18")]),
        ("Lei nº 8.078/1990, art. 49", [("CDC", "49")]),
        ("Decreto-Lei nº 5.452, art. 3", [("CLT", "3")]),
        ("§ 2º do art. 7º da CF", [("CF", "7")]),
        ("incisos I e II do art. 7º da CF", [("CF", "7")]),
        ("Pela CLT, qual o prazo do art. 477?", [(None, "477")]),
        ("art. 7 o empregado tem direito?", [(None, "7")]),
        (
            "os arts. 445 e 477 da CLT e o art. 7º, XIII, da Constituição",
            [("CLT", "445"), ("CLT", "477"), ("CF", "7")],
        ),
        ("art. 7º da CLT e de novo o art. 7º da CLT", [("CLT", "7")]),
    ],
)
def test_find_article_mentions(text: str, expected: list[tuple[str | None, str]]) -> None:
    mentions = find_article_mentions(text)

    assert [(m.law.value if m.law else None, m.number) for m in mentions] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Quero saber sobre artesanato e arte",
        "a parte 2 do contrato",
        "o artista recebe 10 por cento",
        "Quais são os direitos do consumidor?",
        "",
    ],
)
def test_find_article_mentions_ignores_text_without_references(text: str) -> None:
    assert find_article_mentions(text) == []


def test_ranges_are_expanded_only_when_short() -> None:
    mentions = find_article_mentions("arts. 1 a 300 da CLT")

    assert [m.number for m in mentions] == ["1", "300"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Pela CLT e pela Constituição Federal", [LawId.CLT, LawId.CF]),
        ("a lei geral de proteção de dados pessoais", [LawId.LGPD]),
        ("o código do consumidor", [LawId.CDC]),
        ("a Carta Magna e o ADCT", [LawId.CF, LawId.ADCT]),
        ("Marco Civil da Internet", [LawId.MCI]),
        ("a CF/88", [LawId.CF]),
        ("a lei 8.078 e a Lei nº 13.709/2018", [LawId.CDC, LawId.LGPD]),
        ("a Lei 9.999/2020 não está no acervo", []),
        ("conforme o clt", [LawId.CLT]),
        ("a palavra clteste não é a lei", []),
        ("direito constitucional", []),
    ],
)
def test_detect_laws(text: str, expected: list[LawId]) -> None:
    assert detect_laws(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", "7"),
        ("7º", "7"),
        ("7°", "7"),
        ("7o", "7"),
        ("art. 7o", "7"),
        ("Artigo 482", "482"),
        ("59-a", "59-A"),
        ("59 - A", "59-A"),
        ("401A", "401-A"),
        ("10-AB", "10-AB"),
        ("007", "7"),
        ("5º, XI", "5"),
        ("Preâmbulo", "preambulo"),
        ("xyz", None),
        ("", None),
    ],
)
def test_normalize_article_number(raw: str, expected: str | None) -> None:
    assert normalize_article_number(raw) == expected


def test_article_ids_resolve_unknown_laws_with_the_default() -> None:
    mentions = [
        ArticleMention(LawId.CLT, "7"),
        ArticleMention(None, "59-A"),
        ArticleMention(LawId.CLT, "7"),
    ]

    assert article_ids(mentions, default_law=LawId.CLT) == ["CLT:7", "CLT:59-A"]


def test_article_ids_drop_mentions_without_a_law() -> None:
    mentions = [ArticleMention(None, "59-A"), ArticleMention(LawId.CDC, "49")]

    assert article_ids(mentions) == ["CDC:49"]


def test_fold_keeps_one_character_per_character() -> None:
    text = "Constituição, Ação nº 1ª — §"

    folded = fold(text)

    assert folded == "constituicao, acao no 1a — §"
    assert len(folded) == len(text)
