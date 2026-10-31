"""Structure-aware chunking of articles."""

import pytest

from lexgraph.corpus import Corpus, CorpusManifest
from lexgraph.domain import Article, Chunk, LawId
from lexgraph.ingestion.chunking import chunk_article, chunk_corpus
from lexgraph.ingestion.planalto import ordinal_label
from tests.conftest import make_article

CAPUT = "Art. 5º Todos são iguais perante a lei, nos termos seguintes:"
ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def sized(head: str, size: int) -> str:
    """A unit line starting with ``head`` and exactly ``size`` characters long."""
    return head + ("texto " * size)[: size - len(head) - 1] + ";"


def article(*lines: str, heading_path: list[str] | None = None) -> Article:
    return Article(
        law_id=LawId.CF,
        number="5",
        label="Art. 5º",
        seq=0,
        heading_path=heading_path if heading_path is not None else ["TÍTULO II - Dos Direitos"],
        text="\n".join(lines),
        status="vigente",
        url="https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm#art5",
    )


def covers_every_line(source: Article, chunks: list[Chunk]) -> bool:
    lines = {line for chunk in chunks for line in chunk.text.splitlines()}
    return all(line in lines for line in source.text.splitlines())


def test_short_article_is_a_single_chunk() -> None:
    source = make_article()

    (chunk,) = chunk_article(source)

    assert chunk.chunk_id == "CDC:49#0"
    assert (chunk.article_id, chunk.law_id, chunk.article) == ("CDC:49", LawId.CDC, "49")
    assert (chunk.chunk_index, chunk.n_chunks, chunk.units) == (0, 1, ["*"])
    assert chunk.citation == "CDC, art. 49"
    assert chunk.header == "CDC > CAPÍTULO VI - DA PROTEÇÃO CONTRATUAL — Art. 49"
    assert (chunk.text, chunk.url, chunk.heading_path) == (
        source.text,
        source.url,
        source.heading_path,
    )


def test_header_uses_the_short_name_and_skips_empty_headings() -> None:
    source = make_article(LawId.MCI, "19", "Art. 19. Texto.", heading_path=["", "CAPÍTULO III"])

    (chunk,) = chunk_article(source)

    assert chunk.header == "Marco Civil > CAPÍTULO III — Art. 19"
    assert chunk.citation == "Marco Civil, art. 19"


def test_preamble_citation_and_header() -> None:
    source = article("Nós, representantes do povo brasileiro...", heading_path=[])
    source = source.model_copy(update={"number": "preambulo", "label": "Preâmbulo"})

    (chunk,) = chunk_article(source)

    assert (chunk.chunk_id, chunk.citation, chunk.header) == (
        "CF:preambulo#0",
        "CF, preâmbulo",
        "CF — Preâmbulo",
    )


def test_long_article_is_split_by_inciso_with_the_caput_in_every_chunk() -> None:
    source = article(CAPUT, *(sized(f"{roman} - ", 100) for roman in ROMANS))
    max_chars = len(CAPUT) + 3 * 101  # the caput plus three incisos

    chunks = chunk_article(source, max_chars=max_chars)

    assert [chunk.units for chunk in chunks] == [
        ["caput", "I", "II", "III"],
        ["IV", "V", "VI"],
        ["VII", "VIII", "IX"],
        ["X", "XI", "XII"],
    ]
    assert [chunk.citation for chunk in chunks] == [
        "CF, art. 5º, caput e I a III",
        "CF, art. 5º, IV a VI",
        "CF, art. 5º, VII a IX",
        "CF, art. 5º, X a XII",
    ]
    assert [chunk.chunk_id for chunk in chunks] == ["CF:5#0", "CF:5#1", "CF:5#2", "CF:5#3"]
    assert {chunk.n_chunks for chunk in chunks} == {4}
    assert all(chunk.text.startswith(f"{CAPUT}\n") for chunk in chunks)
    assert all(len(chunk.text) <= max_chars for chunk in chunks)
    assert covers_every_line(source, chunks)


def test_split_paragraph_repeats_its_lead() -> None:
    lead = "§ 1º São requisitos:"
    last = sized("§ 2º ", 60)
    source = article(CAPUT, lead, *(sized(f"{roman} - ", 100) for roman in ROMANS[:6]), last)

    chunks = chunk_article(source, max_chars=400)

    assert [chunk.citation for chunk in chunks] == [
        "CF, art. 5º, caput e § 1º, I a III",
        "CF, art. 5º, § 1º, IV a VI",
        "CF, art. 5º, § 2º",
    ]
    assert chunks[1].text.startswith(f"{CAPUT}\n{lead}\nIV - ")
    assert chunks[2].text == f"{CAPUT}\n{last}"
    assert covers_every_line(source, chunks)


def test_short_tail_joins_the_previous_chunk() -> None:
    lead = "§ 1º São requisitos:"
    source = article(CAPUT, lead, *(sized(f"{roman} - ", 100) for roman in ROMANS[:6]), "§ 2º Fim.")

    chunks = chunk_article(source, max_chars=400)

    assert chunks[-1].units == ["§ 1º, IV", "§ 1º, V", "§ 1º, VI", "§ 2º"]
    assert chunks[-1].citation == "CF, art. 5º, § 1º, IV a § 2º"


def test_inciso_with_many_alineas_is_split_by_alinea() -> None:
    lead = "IV - para a composição das Câmaras, o limite máximo de:"
    alineas = [sized(f"{letter}) ", 100) for letter in "abcdefgh"]
    source = article(CAPUT, sized("I - ", 60), lead, *alineas, sized("V - ", 60))

    chunks = chunk_article(source, max_chars=450)

    assert [chunk.citation for chunk in chunks] == [
        "CF, art. 5º, caput e I a IV, b",
        "CF, art. 5º, IV, c a e",
        "CF, art. 5º, IV, f a h",
        "CF, art. 5º, V",
    ]
    assert chunks[1].units == ["IV, c", "IV, d", "IV, e"]
    assert chunks[1].text.startswith(f"{CAPUT}\n{lead}\nc) ")
    assert all(len(chunk.text) <= 450 for chunk in chunks)
    assert covers_every_line(source, chunks)


def test_alineas_under_a_paragraph_and_an_inciso_are_packed_together() -> None:
    # LGPD art. 50, § 2º, I, a to h: the leads of the caput, the § and the inciso must leave
    # room for more than one alínea per chunk.
    caput = "Art. 5º " + "palavra " * 75 + "fim:"
    lead = "§ 2º " + "termo " * 50 + "deverão:"
    inciso = "I - implementar programa de governança em privacidade que, no mínimo:"
    alineas = [sized(f"{letter}) ", 150) for letter in "abcdefgh"]
    source = article(caput, lead, inciso, *alineas, sized("II - ", 100))

    chunks = chunk_article(source, max_chars=1500, pack_chars=700)

    assert [chunk.citation for chunk in chunks] == [
        "CF, art. 5º, caput",
        "CF, art. 5º, § 2º, I",
        "CF, art. 5º, § 2º, I, a e b",
        "CF, art. 5º, § 2º, I, c e d",
        "CF, art. 5º, § 2º, I, e e f",
        "CF, art. 5º, § 2º, I, g e h",
        "CF, art. 5º, § 2º, II",
    ]
    caput_lead, paragraph_lead, inciso_lead, first_alinea, *_ = chunks[2].text.splitlines()
    assert caput.startswith(caput_lead.removesuffix(" …"))
    assert lead.startswith(paragraph_lead.removesuffix(" …"))
    assert (caput_lead[-2:], paragraph_lead[-2:]) == (" …", " …")
    assert (inciso_lead, first_alinea) == (inciso, alineas[0])  # a short lead stays whole
    assert len(caput_lead) + len(paragraph_lead) + len(inciso_lead) <= 700 // 2
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert covers_every_line(source, chunks)


def test_suffixed_inciso_missing_its_dash_is_cited_with_its_suffix() -> None:
    # CF art. 109 prints inciso V-A without its dash; it is neither a second V nor left out.
    source = article(
        "Art. 109. Aos juízes federais compete processar e julgar:",
        sized("IV - os crimes políticos ", 263),
        sized("V - os crimes previstos em tratado ", 176),
        "V-A as causas relativas a direitos humanos a que se refere o § 5º deste artigo;",
        sized("VI - os crimes contra a organização do trabalho ", 142),
    ).model_copy(update={"number": "109", "label": "Art. 109"})

    first, second = chunk_article(source, max_chars=600)

    assert first.units == ["caput", "IV", "V", "V-A"]
    assert first.citation == "CF, art. 109, caput e IV a V-A"
    assert second.units == ["VI"]


def test_long_caput_is_whole_in_the_first_chunk_and_truncated_after() -> None:
    caput = "Art. 5º " + "palavra " * 80 + "fim:"
    source = article(caput, *(sized(f"{roman} - ", 300) for roman in ROMANS[:6]))

    first, second, *_ = chunk_article(source, caput_chars=200)

    assert first.text.startswith(caput)
    prefix = second.text.splitlines()[0]
    assert prefix.endswith(" …")
    assert len(prefix) <= 200
    assert caput.startswith(prefix.removesuffix(" …"))


def test_lead_truncation_scales_down_with_small_chunks() -> None:
    caput = "Art. 5º " + "palavra " * 30 + "fim:"
    source = article(caput, *(sized(f"{roman} - ", 120) for roman in ROMANS[:4]))

    chunks = chunk_article(source, max_chars=300)

    assert all(len(chunk.text.splitlines()[0]) <= 100 for chunk in chunks[1:])


def test_a_unit_longer_than_max_chars_is_never_cut() -> None:
    giant = sized("II - ", 900)
    source = article(CAPUT, sized("I - ", 100), giant, sized("III - ", 100))

    chunks = chunk_article(source, max_chars=500)

    assert [chunk.units for chunk in chunks] == [["caput", "I"], ["II"], ["III"]]
    assert chunks[1].text == f"{CAPUT}\n{giant}"


def test_article_without_units_stays_whole() -> None:
    source = article("Art. 5º " + "palavra " * 300)

    (chunk,) = chunk_article(source, max_chars=500)

    assert (chunk.units, chunk.text, chunk.citation) == (["*"], source.text, "CF, art. 5º")


def test_paragrafo_unico_in_units_and_citation() -> None:
    sole = "Parágrafo único. " + "texto " * 40
    source = article(CAPUT, sized("I - ", 200), sized("II - ", 200), sole)

    chunks = chunk_article(source, max_chars=400)

    assert chunks[-1].units == ["parágrafo único"]
    assert chunks[-1].citation == "CF, art. 5º, parágrafo único"


def test_whole_caput_is_cited_with_its_incisos() -> None:
    incisos = [sized(f"{roman} - ", 60) for roman in ROMANS[:3]]
    paragraph = sized("§ 1º ", 300)
    source = article(CAPUT, *incisos, paragraph)

    first, second = chunk_article(source, max_chars=320)

    assert first.text == "\n".join([CAPUT, *incisos])
    assert first.units == ["caput", "I", "II", "III"]
    assert first.citation == "CF, art. 5º, caput e I a III"
    assert (second.units, second.citation) == (["§ 1º"], "CF, art. 5º, § 1º")


def test_whole_caput_without_incisos_is_cited_as_the_caput() -> None:
    source = article(CAPUT, sized("§ 1º ", 300), sized("§ 2º ", 300))

    chunks = chunk_article(source, max_chars=400)

    assert [chunk.citation for chunk in chunks] == [
        "CF, art. 5º, caput e § 1º",
        "CF, art. 5º, § 2º",
    ]


def test_article_quoting_another_law_is_never_split() -> None:
    # LGPD, art. 60: the quoted incisos belong to Marco Civil, art. 7º, not to art. 60.
    source = article(
        "Art. 5º A Lei nº 12.965, de 23 de abril de 2014, passa a vigorar com as seguintes "
        "alterações:",
        "“Art. 7º …",
        "…",
        sized("X - exclusão definitiva dos dados pessoais ", 150),
        "…”",
        "“Art. 16. …",
        "…",
        sized("II - de dados pessoais que sejam excessivos ", 150) + "”",
    )

    (chunk,) = chunk_article(source, max_chars=300)

    assert (chunk.units, chunk.citation, chunk.text) == (["*"], "CF, art. 5º", source.text)


def test_paragraph_labels_are_ordinal_up_to_nine() -> None:
    paragraphs = [f"{ordinal_label('§', n)} {'texto ' * 30}" for n in range(1, 12)]
    source = article(CAPUT, *paragraphs)

    chunks = chunk_article(source, max_chars=700)

    assert [unit for chunk in chunks for unit in chunk.units] == [
        "caput",
        *(ordinal_label("§", n) for n in range(1, 12)),
    ]
    assert chunks[-1].citation == "CF, art. 5º, § 10 e § 11"


@pytest.mark.parametrize(("max_chars", "caput_chars"), [(0, 400), (1500, 0), (-1, -1)])
def test_invalid_sizes_are_rejected(max_chars: int, caput_chars: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        chunk_article(make_article(), max_chars=max_chars, caput_chars=caput_chars)


def test_chunk_corpus_keeps_articles_in_force_of_the_selected_laws() -> None:
    long_text = "\n".join([CAPUT, *(sized(f"{roman} - ", 100) for roman in ROMANS)])
    corpus = Corpus(
        [
            make_article(LawId.CDC, "49", seq=0),
            make_article(LawId.CDC, "50", "Art. 50. (VETADO).", seq=1, status="vetado"),
            make_article(LawId.LGPD, "18", long_text, seq=0),
        ],
        CorpusManifest(generated_at="2026-10-30T00:00:00+00:00", parser_version="test", laws={}),
    )

    everything = chunk_corpus(corpus)
    lgpd = chunk_corpus(corpus, law_ids=[LawId.LGPD], max_chars=500)

    assert [chunk.chunk_id for chunk in everything] == ["CDC:49#0", "LGPD:18#0"]
    assert {chunk.article_id for chunk in lgpd} == {"LGPD:18"}
    assert len(lgpd) > 1


def test_constitutional_review_notes_travel_with_every_chunk() -> None:
    article = make_article(
        LawId.CLT,
        "879",
        "Art. 879 - Sendo ilíquida a sentença exequenda, ordenar-se-á, previamente, a sua "
        "liquidação.",
    ).model_copy(
        update={
            "annotations": [
                "Redação dada pela Lei nº 13.467, de 2017",
                "Vide ADC 58",
                "Vide ADI 6021",
                "Vide ADC 58",
            ]
        }
    )

    chunks = chunk_article(article)

    assert all(chunk.control_notes == ["Vide ADC 58", "Vide ADI 6021"] for chunk in chunks)
    assert all("ADC" not in chunk.embedding_text() for chunk in chunks)


def test_long_lists_of_incisos_are_packed_into_small_chunks() -> None:
    source = article("Art. 7º São direitos:", *(sized(f"{roman} - ", 200) for roman in ROMANS[:9]))

    chunks = chunk_article(source, max_chars=1500, pack_chars=700)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert covers_every_line(source, chunks)
