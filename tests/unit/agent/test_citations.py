"""Citation markers in answers and the sources they resolve to."""

import pytest

from lexgraph.agent.citations import (
    build_citations,
    cited_indices,
    dangling_indices,
    drop_dangling_markers,
    normalize_markers,
    passages_to_sources,
    strip_markers,
)
from tests.unit.agent.fakes import CDC_49, CLT_58, LGPD_18, dumped


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("O prazo é de 7 dias [1].", [1]),
        ("Vale para ambos [1, 2].", [1, 2]),
        ("Vale para ambos [1][2].", [1, 2]),
        ("Vale para todos [1-3].", [1, 2, 3]),
        ("Vale para todos [1\N{EN DASH}3].", [1, 2, 3]),
        ("Vale [2; 4] e [1 e 3].", [2, 4, 1, 3]),
        ("Repetido [2], de novo [2] e [1].", [2, 1]),
        ("Misturado [1, 3-4].", [1, 3, 4]),
        ("Faixa invertida [3-1].", [3, 1]),
        ("Sem marcadores.", []),
        ("Ano [2023] e lista [a] não contam.", []),
    ],
)
def test_cited_indices(text: str, expected: list[int]) -> None:
    assert cited_indices(text) == expected


def test_dangling_indices_are_the_ones_without_a_source() -> None:
    assert dangling_indices("Regra [1], outra [4] e [0].", n_sources=3) == [4, 0]


def test_build_citations_resolves_markers_in_order_of_appearance() -> None:
    sources = passages_to_sources(dumped(CDC_49, CLT_58, LGPD_18))

    citations = build_citations("Primeiro [3], depois [1] e um inexistente [9].", sources)

    assert [(c.index, c.article_id) for c in citations] == [(3, "LGPD:18"), (1, "CDC:49")]
    assert citations[1].citation == CDC_49.citation
    assert citations[1].url == CDC_49.url


def test_passages_to_sources_numbers_from_one() -> None:
    sources = passages_to_sources(dumped(CDC_49, CLT_58))

    assert [source.index for source in sources] == [1, 2]
    assert sources[0].chunk_id == CDC_49.chunk_id
    assert sources[0].score == CDC_49.score
    assert sources[1].text == CLT_58.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("O prazo é de 7 dias [1].", "O prazo é de 7 dias."),
        ("Ambos [1][2], com exceções [3, 4].", "Ambos, com exceções."),
        ("Sem marcadores.", "Sem marcadores."),
    ],
)
def test_strip_markers(text: str, expected: str) -> None:
    assert strip_markers(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("O prazo é de 7 dias [1].", "O prazo é de 7 dias [1]."),
        ("O prazo é de 7 dias [4].", "O prazo é de 7 dias."),
        ("Há multa [1][4].", "Há multa [1]."),
        ("Há multa [1, 4].", "Há multa [1]."),
        ("Há multa [2-5].", "Há multa [2][3]."),
    ],
)
def test_drop_dangling_markers_keeps_only_known_sources(text: str, expected: str) -> None:
    assert drop_dangling_markers(text, 3) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Exceção prevista [1§ 3].", "Exceção prevista [1]."),
        ("Ver [2, inciso IV] e [3].", "Ver [2] e [3]."),
        ("Ver [1, art. 5].", "Ver [1]."),
        ("Ver [1, Art. 7º].", "Ver [1]."),
        ("Ver [1, artigo 7º].", "Ver [1]."),
        ("Ver [1, Inciso IV].", "Ver [1]."),
        ("Ver [1, inc. IV].", "Ver [1]."),
        ("Ver [1; alínea b].", "Ver [1]."),
        ("Ver [1, al. b].", "Ver [1]."),
        ("Ver [1, parágrafo único].", "Ver [1]."),
        ("Ver [1, Parágrafo Único].", "Ver [1]."),
        ("Ver [3, par. único].", "Ver [3]."),
        ("Ver [2, caput].", "Ver [2]."),
        ("Ver [2, item 3].", "Ver [2]."),
        ("Ver [ 1 , §3].", "Ver [1]."),
        ("Ver [1, 2, § 3].", "Ver [1, 2]."),
        ("Ver [1 e 2, incisos I e II].", "Ver [1 e 2]."),
        ("Ver [1-3, § 2].", "Ver [1-3]."),
        ("Sem detalhe [1][2].", "Sem detalhe [1][2]."),
        ("Sem detalhe [1, 2].", "Sem detalhe [1, 2]."),
        ("Sem detalhe [1-3].", "Sem detalhe [1-3]."),
        ("Sem detalhe [1].", "Sem detalhe [1]."),
        ("Ano [2023, art. 5] e lista [a].", "Ano [2023, art. 5] e lista [a]."),
        ("Comentário [1, artesanal].", "Comentário [1, artesanal]."),
        ("Um terço a mais [2]§ XVII.", "Um terço a mais [2]."),
        ("Até 15 dias antes [1]§ 1º.", "Até 15 dias antes [1]."),
        ("Ver [3]§§ 2º e 3º do artigo.", "Ver [3] do artigo."),
        ("Ver [1, 2]§ único.", "Ver [1, 2]."),
        ("Ver [1]§ V-A.", "Ver [1]."),
        ("O [1] § 2º trata disso.", "O [1] § 2º trata disso."),
    ],
)
def test_normalize_markers_keeps_only_the_source_numbers(text: str, expected: str) -> None:
    assert normalize_markers(text) == expected


def test_normalized_markers_are_checked_against_the_sources() -> None:
    answer = normalize_markers("Regra [1, 2, § 3] e exceção [9, parágrafo único].")

    assert cited_indices(answer) == [1, 2, 9]
    assert dangling_indices(answer, n_sources=3) == [9]
    assert drop_dangling_markers(answer, 3) == "Regra [1, 2] e exceção."
