"""Line grammar of the planalto pages: labels, headings, signatures and amendment notes.

Every case below was seen on the real pages (or is the closest synthetic equivalent).
"""

import pytest

from lexgraph.ingestion.planalto import (
    ANNEX_RE,
    HEADING_RE,
    LINK_END,
    LINK_START,
    SIGNATURE_RE,
    UnitKind,
    classify_line,
    match_article_label,
    ordinal_label,
    strip_annotations,
)


@pytest.mark.parametrize(
    ("line", "number", "label", "body"),
    [
        ("Art. 1º Fica aprovada a Consolidação", "1", "Art. 1º", "Fica aprovada a Consolidação"),
        ("Art. 1° O presente código estabelece", "1", "Art. 1º", "O presente código estabelece"),
        ("Art. 2o O presente decreto-lei", "2", "Art. 2º", "O presente decreto-lei"),
        ("Art. 10. O fornecedor não poderá", "10", "Art. 10", "O fornecedor não poderá"),
        ("Art. 62 - Não são abrangidos", "62", "Art. 62", "Não são abrangidos"),
        ("Art. 4º - A empresa", "4", "Art. 4º", "A empresa"),
        ("Art. 1º A República Federativa", "1", "Art. 1º", "A República Federativa"),
        ("Art. 59-A. Em exceção ao disposto", "59-A", "Art. 59-A", "Em exceção ao disposto"),
        ("Art. 8º-A Aos usuários", "8-A", "Art. 8º-A", "Aos usuários"),
        ("Art. 401A. (VETADO)", "401-A", "Art. 401-A", "(VETADO)"),
        ("Art. . 154 - A observância", "154", "Art. 154", "A observância"),
        ("Art.184 - As máquinas", "184", "Art. 184", "As máquinas"),
        ("Art 15. Texto", "15", "Art. 15", "Texto"),
        ("Art. 5 7. (VETADO).", "57", "Art. 57", "(VETADO)."),
        ("Art. 1º. O Presidente da República", "1", "Art. 1º", "O Presidente da República"),
        ("Art. 1.000. Texto hipotético", "1000", "Art. 1.000", "Texto hipotético"),
        ("Artigo 5º texto", "5", "Art. 5º", "texto"),
    ],
)
def test_article_labels_tolerate_the_variants_of_the_pages(
    line: str, number: str, label: str, body: str
) -> None:
    parsed = match_article_label(line)

    assert parsed is not None
    assert (parsed.number, parsed.label, parsed.body) == (number, label, body)


@pytest.mark.parametrize(
    ("line", "separator"),
    [
        ("Art. 62 - Não são abrangidos", " - "),
        ("Art. 7º \u2013 Texto", " - "),
        ("Art. 10. O fornecedor", ". "),
        ("Art. 1º A República", " "),
        ("Art. 5º: Texto", ": "),
    ],
)
def test_article_label_separator_is_normalized(line: str, separator: str) -> None:
    parsed = match_article_label(line)

    assert parsed is not None
    assert parsed.separator == separator


@pytest.mark.parametrize(
    "line",
    ["Arts. 1º e 2º revogados", "“Art. 7º ..........", "Artesanato local", "Parágrafo único."],
)
def test_lines_that_are_not_article_labels(line: str) -> None:
    assert match_article_label(line) is None


@pytest.mark.parametrize(
    ("line", "kind", "label", "body"),
    [
        ("§ 1º O direito comum", "paragraph", "§ 1º", "O direito comum"),
        ("§ 2o Súmulas", "paragraph", "§ 2º", "Súmulas"),
        ("§ 1º-A Texto", "paragraph", "§ 1º-A", "Texto"),
        ("§ 4.° O requisito", "paragraph", "§ 4º", "O requisito"),
        ("§ 10. Texto", "paragraph", "§ 10", "Texto"),
        ("§ 2 O fornecedor deverá", "paragraph", "§ 2º", "O fornecedor deverá"),
        ("Parágrafo único - Continuam", "paragraph", "Parágrafo único", "Continuam"),
        ("Parágrafo Único. Texto", "paragraph", "Parágrafo único", "Texto"),
        ("§§ 1º a 3º.", "paragraph", "§§ 1º a 3º", ""),
        ("2º A lei disporá", "paragraph", "§ 2º", "A lei disporá"),
        ("I - a soberania;", "inciso", "I", "a soberania;"),
        ("XIX-A - texto", "inciso", "XIX-A", "texto"),
        ("V-A - Procuradoria;", "inciso", "V-A", "Procuradoria;"),
        ("I-A \u2013 dia 1º de agosto de 2021", "inciso", "I-A", "dia 1º de agosto de 2021"),
        ("V\u2013 doações, de qualquer natureza", "inciso", "V", "doações, de qualquer natureza"),
        ("V-texto", "inciso", "V", "texto"),
        ("I - Teto de Referência da União", "inciso", "I", "Teto de Referência da União"),
        ("IX \u2013 remuneração", "inciso", "IX", "remuneração"),
        ("Il - durante o licenciamento", "inciso", "II", "durante o licenciamento"),
        ("VlIl - apresentar", "inciso", "VIII", "apresentar"),
        ("XV a distribuição de processos", "inciso", "XV", "a distribuição de processos"),
        # Suffixed incisos without their dash (CF arts. 109 and 92): the hyphen of the suffix
        # is not a separator, so these are not a second inciso V and a second inciso I.
        (
            "V-A as causas relativas a direitos humanos a que se refere o § 5º deste artigo;",
            "inciso",
            "V-A",
            "as causas relativas a direitos humanos a que se refere o § 5º deste artigo;",
        ),
        ("I-A o Conselho Nacional de Justiça;", "inciso", "I-A", "o Conselho Nacional de Justiça;"),
        ("a) ato de improbidade;", "alinea", "a", "ato de improbidade;"),
        ("b-a) texto", "alinea", "b-a", "texto"),
        ("1. até 150 vezes", "item", "1", "até 150 vezes"),
        ("DISPOSIÇÕES GERAIS", "text", None, "DISPOSIÇÕES GERAIS"),
        ("Pena - Detenção de seis meses", "text", None, "Pena - Detenção de seis meses"),
    ],
)
def test_classify_line(line: str, kind: UnitKind, label: str | None, body: str) -> None:
    assert tuple(classify_line(line)) == (kind, label, body)


@pytest.mark.parametrize(
    ("prefix", "number", "expected"),
    [("§", 1, "§ 1º"), ("§", 9, "§ 9º"), ("§", 10, "§ 10"), ("Art.", 1000, "Art. 1.000")],
)
def test_ordinal_up_to_nine_cardinal_from_ten(prefix: str, number: int, expected: str) -> None:
    assert ordinal_label(prefix, number) == expected


@pytest.mark.parametrize(
    ("line", "kind", "number", "title"),
    [
        ("TÍTULO II-A", "TÍTULO", "II-A", ""),
        ("CAPÍTULO III-A", "CAPÍTULO", "III-A", ""),
        ("Seção I", "Seção", "I", ""),
        ("Subseção II", "Subseção", "II", ""),
        ("SEÇÃO VIII DAS PENALIDADES", "SEÇÃO", "VIII", "DAS PENALIDADES"),
        ("CAPITULO V", "CAPITULO", "V", ""),
        ("TÍTULO ÚNICO - Disposições Gerais", "TÍTULO", "ÚNICO", "Disposições Gerais"),
    ],
)
def test_headings(line: str, kind: str, number: str, title: str) -> None:
    match = HEADING_RE.match(line)

    assert match is not None
    assert (match["kind"], match["num"], match["title"]) == (kind, number, title)


@pytest.mark.parametrize(
    "line",
    [
        "Brasília, 11 de setembro de 1990; 169° da Independência e 102° da República.",
        "Brasília , 14 de agosto de 2018; 197º da Independência e 130º da República.",
        "Rio de Janeiro, 1 de maio de 1943, 122º da Independência e 55º da República.",
        "Brasília, 5 de outubro de 1988.",
    ],
)
def test_signature_lines_end_the_law(line: str) -> None:
    assert SIGNATURE_RE.match(line)


@pytest.mark.parametrize(
    ("line", "is_annex"),
    [("ANEXO", True), ("ANEXO II", True), ("ANEXOS", True), ("ANEXO DA LEI", False)],
)
def test_annex_headings(line: str, is_annex: bool) -> None:
    assert bool(ANNEX_RE.match(line)) is is_annex


@pytest.mark.parametrize(
    ("line", "text", "notes"),
    [
        (
            (
                f"§ 2º Sempre que {LINK_START}(Redação dada pela Lei nº 13.467, de 2017){LINK_END}"
                f" {LINK_START}(Vigência){LINK_END}"
            ),
            "§ 2º Sempre que",
            ["Redação dada pela Lei nº 13.467, de 2017", "Vigência"],
        ),
        (  # one note split across two links: <a>(V</a><a>ide ADC 80)</a>
            (
                f"§ 3º O juízo não poderá. {LINK_START}(Incluído pela Lei nº 13.467, de 2017)"
                f"{LINK_END} {LINK_START} (V{LINK_END}{LINK_START}ide ADC 80){LINK_END}"
            ),
            "§ 3º O juízo não poderá.",
            ["Incluído pela Lei nº 13.467, de 2017", "Vide ADC 80"],
        ),
        (  # closing parenthesis missing on the page
            (
                "Art. 139 - Poderão ser concedidas férias coletivas. "
                "(Redação dada pelo Decreto-lei nº 1.535, de 13.4.1977"
            ),
            "Art. 139 - Poderão ser concedidas férias coletivas.",
            ["Redação dada pelo Decreto-lei nº 1.535, de 13.4.1977"],
        ),
        (  # generic parentheses are part of the text
            "XXII - salário de 40% (quarenta por cento) do limite (Vide Lei nº 13.105, de 2015)",
            "XXII - salário de 40% (quarenta por cento) do limite",
            ["Vide Lei nº 13.105, de 2015"],
        ),
        (  # a note with a nested parenthetical
            "IV - texto (Redação dada pela Lei nº 9.528, de 1997 (Vide ADI 3.395))",
            "IV - texto",
            ["Redação dada pela Lei nº 9.528, de 1997 (Vide ADI 3.395)"],
        ),
        (  # a bare link is a note only when its text is one
            f"Art. 1º Esta Lei entra em vigor. {LINK_START}Vigência{LINK_END}",
            "Art. 1º Esta Lei entra em vigor.",
            ["Vigência"],
        ),
        (
            f"Art. 60. A {LINK_START}Lei nº 12.965 (Marco Civil da Internet){LINK_END} , passa",
            "Art. 60. A Lei nº 12.965 (Marco Civil da Internet), passa",
            [],
        ),
        (
            "III - recursos da Lei de Acesso (Lei de Acesso à Informação); (VETADO)",
            "III - recursos da Lei de Acesso (Lei de Acesso à Informação); (VETADO)",
            [],
        ),
    ],
)
def test_strip_annotations(line: str, text: str, notes: list[str]) -> None:
    assert strip_annotations(line) == (text, notes)
