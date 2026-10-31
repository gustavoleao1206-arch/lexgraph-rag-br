"""Rule-based answer checks: patterns, refusals, citation extraction and citation scores."""

import pytest

from lexgraph.eval.golden import GoldenItem
from lexgraph.eval.rules import (
    check_reference,
    citation_scores,
    cited_articles,
    extract_article_refs,
    fold,
    forbidden_claims,
    found_patterns,
    is_refusal,
    matches,
    missing_patterns,
    refusal_stats,
    score_answer,
    sentences,
)
from lexgraph.schemas import Route
from tests.unit.eval.fakes import make_item, make_response


def test_fold_strips_accents_and_ordinal_signs() -> None:
    assert fold("Proteção, art. 7º, § 1º") == "Protecao, art. 7o, § 1o"


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        ("proteção de dados", "PROTECAO   DE\nDADOS pessoais", True),
        ("7 dias|sete dias", "O prazo é de sete dias.", True),
        ("7 dias|sete dias", "O prazo é de 30 dias.", False),
        ("reclus[aã]o", "pena de reclusão de 1 a 4 anos", True),
        (r"^\s*SIM\.?\s*$", " sim. ", True),
        (r"^\s*SIM\.?\s*$", "Sim, a multa pode ser cobrada.", False),
        ("50%", "acréscimo de 50% sobre a hora normal", True),
    ],
)
def test_matches_ignores_accents_case_and_spacing(pattern: str, text: str, expected: bool) -> None:
    assert matches(pattern, text) is expected


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        (r"\b2 ?(horas|h\b)|duas horas", "reduzido em 2 (duas) horas diárias"),
        (r"20 ?%|vinte por cento", "acréscimo de 20 % (vinte por cento), pelo menos"),
        (r"\b1 ano\b|\bum ano\b", "pelo prazo de 1 (um) ano, nos termos"),
        ("120 dias", "licença de 120 (cento e vinte) dias"),
        ("52 minutos e 30 segundos", "52 (cinqüenta e dois) minutos e 30 (trinta) segundos"),
        (r"50\.000\.000,00 por", "R$ 50.000.000,00 (cinquenta milhões de reais) por infração"),
        (r"\(art\. 59, § 1o\)", "acréscimo de 50% (art. 59, § 1º)"),
        (r"art\. 59 \(banco de horas\)", "CLT, art. 59 (banco de horas)"),
    ],
)
def test_numbers_spelled_out_in_parentheses_match_their_digits(pattern: str, text: str) -> None:
    assert matches(pattern, text)


def test_sentences_split_at_sentence_ends_and_line_breaks() -> None:
    text = "Não. Pelo art. 52 do CDC, a multa é de até 2%; R$ 1.000,00 seria\nabusivo! Certo?"
    assert sentences(text) == [
        "Nao.",
        "Pelo art. 52 do CDC, a multa e de ate 2%;",
        "R$ 1.000,00 seria",
        "abusivo!",
        "Certo?",
    ]


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("Sim, o CDC garante 30 dias para desistir. Não esqueça de guardar a nota.", True),
        ("- Prazo: 30 dias\n- Não esqueça a nota.", True),
        ("Não. O prazo é de 7 dias, e não de 30 dias.", False),
        ("O prazo de 30 dias mencionado está equivocado: são 7 dias.", False),
    ],
)
def test_forbidden_claims_are_matched_sentence_by_sentence(text: str, forbidden: bool) -> None:
    pattern = r"^(?!.*\b(nao|incorret|equivoc)).*\b(30|trinta) dias"
    assert forbidden_claims([pattern, "reclusao"], text) == ([pattern] if forbidden else [])


def test_missing_and_found_patterns() -> None:
    patterns = ["7 dias", "estabelecimento comercial", "multa"]
    text = "Até 7 dias quando a compra ocorre fora do estabelecimento comercial."
    assert missing_patterns(patterns, text) == ["multa"]
    assert found_patterns(patterns, text) == ["7 dias", "estabelecimento comercial"]


def test_check_reference_accepts_consistent_items(golden_items: list[GoldenItem]) -> None:
    assert [problem for item in golden_items for problem in check_reference(item)] == []


def test_check_reference_reports_patterns_the_reference_contradicts() -> None:
    item = make_item(must_include=["30 dias"], must_not_include=["7 dias"])
    assert check_reference(item) == [
        "cdc-001: must_include '30 dias' does not match the reference answer",
        "cdc-001: must_not_include '7 dias' matches the reference answer",
    ]


def test_check_reference_applies_forbidden_claims_per_sentence() -> None:
    reference = "O prazo é de 7 dias, e não de 30 dias. Não esqueça de guardar a nota."
    item = make_item(
        reference_answer=reference,
        must_not_include=[r"^(?!.*\bnao\b).*\b30 dias", r"^(?!.*\bnao\b).*\b7 dias"],
    )
    assert check_reference(item) == []


@pytest.mark.parametrize("route", ["no_context", "out_of_scope"])
def test_refusal_routes_are_refusals(route: Route) -> None:
    assert is_refusal(make_response("Posso ajudar com outra pergunta?", route=route))


@pytest.mark.parametrize(
    "answer",
    [
        "Não encontrei base legal no acervo para responder a essa pergunta.",
        "As fontes recuperadas não tratam do crime de furto.",
        "Essa pergunta está fora do escopo do acervo.",
        "Não há, nas fontes disponíveis, informação sobre esse tema.",
        "Não é possível responder com base nos trechos encontrados.",
    ],
)
def test_uncited_refusal_wording_is_a_refusal(answer: str) -> None:
    assert is_refusal(make_response(answer))


@pytest.mark.parametrize(
    ("answer", "route", "cited"),
    [
        ("O prazo de arrependimento é de 7 dias [1].", "answered", ["CDC:49"]),
        ("Não encontrei previsão de multa de 10%; o limite é 2% [1].", "answered", ["CDC:49"]),
        ("Não há multa quando o consumidor desiste no prazo.", "answered", []),
        ("Olá! Posso ajudar com dúvidas sobre a legislação.", "smalltalk", []),
        (
            (
                "Não encontrei esse tema nas fontes, mas pelo Código Penal o furto simples tem "
                "pena de reclusão de 1 a 4 anos (art. 155)."
            ),
            "answered",
            [],
        ),
        ("Não encontrei previsão expressa, mas a multa é limitada a 2%.", "answered", []),
    ],
)
def test_answers_are_not_refusals(answer: str, route: Route, cited: list[str]) -> None:
    response = make_response(answer, route=route, context=["CDC:49"], cited=cited)
    assert not is_refusal(response)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            (
                "Conforme os arts. 445 e 477 da CLT e o art. 7º, XIII, da Constituição; "
                "ver também Lei nº 8.078/1990, art. 49."
            ),
            {"CLT:445", "CLT:477", "CF:7", "CDC:49"},
        ),
        ("O art. 18 da LGPD e o art. 49 do CDC.", {"LGPD:18", "CDC:49"}),
        ("(CLT, art. 71, caput). A não concessão gera pagamento (art. 71, § 4º).", {"CLT:71"}),
        (
            "nos termos do art. 5º, LXXIX, da CF/88 e do art. 611-A, III, da CLT",
            {"CF:5", "CLT:611-A"},
        ),
        ("O art. 10, II, do ADCT veda a dispensa arbitrária.", {"ADCT:10"}),
        (
            "Segundo o artigo 482 da Consolidação das Leis do Trabalho e o Marco Civil, art. 19.",
            {"CLT:482", "MCI:19"},
        ),
        ("O art. 4º-A da Lei 13.709/2018 trata do tema.", {"LGPD:4-A"}),
        ("O art. 482 da Consolidação trabalhista (CLT) lista as faltas graves.", {"CLT:482"}),
        ("Veja o art. 5 sobre o tema.", set()),
        ("Na CLT, cf. art. 7º, o tema é regulado.", {"CLT:7"}),
        (
            (
                "A licença de 120 dias (art. 7º, XVIII, da CF) e a estabilidade do art. 10, II, "
                "b, do ADCT."
            ),
            {"CF:7", "ADCT:10"},
        ),
        (
            'Veja o art. 10, II, "b", do Ato das Disposições Constitucionais Transitórias.',
            {"ADCT:10"},
        ),
        ("Conforme o art. 483, alíneas b, d e e, da CLT.", {"CLT:483"}),
        ("O art. 18, incisos I e II, da LGPD.", {"LGPD:18"}),
        ("CDC, art. 49, o consumidor pode desistir.", {"CDC:49"}),
    ],
)
def test_extract_article_refs(text: str, expected: set[str]) -> None:
    assert extract_article_refs(text) == expected


def test_cited_articles_join_markers_and_explicit_references() -> None:
    response = make_response(
        "Você pode desistir em 7 dias [1]; a multa segue o art. 52 do CDC.",
        context=["CDC:49"],
        cited=["CDC:49"],
    )
    assert cited_articles(response) == {"CDC:49", "CDC:52"}


def test_citation_scores_hand_computed() -> None:
    scores = citation_scores(
        {"CDC:49", "CDC:52", "CF:7"},
        expected={"CDC:49", "CDC:18"},
        acceptable={"CF:7"},
        context={"CDC:49", "CF:7", "LGPD:18"},
    )
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(1 / 2)
    assert scores.in_context == pytest.approx(2 / 3)


def test_citation_scores_are_undefined_without_citations_or_expectations() -> None:
    no_citations = citation_scores(set(), expected={"CDC:49"}, acceptable=set(), context=set())
    assert (no_citations.precision, no_citations.recall, no_citations.in_context) == (
        None,
        0.0,
        None,
    )
    no_expected = citation_scores({"CDC:49"}, expected=set(), acceptable=set(), context=set())
    assert (no_expected.precision, no_expected.recall, no_expected.in_context) == (0.0, None, 0.0)


def test_score_answer_of_a_good_answer(golden_items: list[GoldenItem]) -> None:
    response = make_response(
        "Você pode desistir da compra em até 7 dias do recebimento [1].",
        context=["CDC:49", "CDC:26"],
        cited=["CDC:49"],
    )
    scores = score_answer(golden_items[0], response)

    assert scores.item_id == "cdc-001"
    assert not scores.refused
    assert (scores.must_include_passed, scores.must_not_include_passed) == (True, None)
    assert scores.cited == ["CDC:49"]
    assert (scores.citations.precision, scores.citations.recall, scores.citations.in_context) == (
        1.0,
        1.0,
        1.0,
    )


def test_score_answer_of_a_wrong_answer_with_an_invented_citation(
    golden_items: list[GoldenItem],
) -> None:
    response = make_response("O prazo é de 30 dias, conforme o art. 26 do CDC.")
    scores = score_answer(golden_items[0], response)

    assert scores.must_include_passed is False
    assert scores.missing_patterns == ["7 dias|sete dias"]
    assert scores.cited == ["CDC:26"]
    assert (scores.citations.precision, scores.citations.recall, scores.citations.in_context) == (
        0.0,
        0.0,
        0.0,
    )


def test_score_answer_flags_forbidden_claims(golden_items: list[GoldenItem]) -> None:
    scores = score_answer(golden_items[3], make_response("A pena é de reclusão de 1 a 4 anos."))
    assert scores.must_not_include_passed is False
    assert scores.forbidden_patterns == ["reclus[aã]o"]
    assert not scores.refused


def test_refusal_stats() -> None:
    stats = refusal_stats(
        [(True, True), (True, False), (False, True), (False, False), (False, False)]
    )
    assert (
        stats.true_positives,
        stats.false_negatives,
        stats.false_positives,
        stats.true_negatives,
    ) == (1, 1, 1, 2)
    assert (stats.precision, stats.recall) == (0.5, 0.5)
    assert stats.false_refusal_rate == pytest.approx(1 / 3)


def test_refusal_stats_without_outcomes_are_undefined() -> None:
    stats = refusal_stats([])
    assert (stats.precision, stats.recall, stats.false_refusal_rate) == (None, None, None)
