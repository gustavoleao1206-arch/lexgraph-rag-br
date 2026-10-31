"""Checks on the shipped golden set (``data/eval/golden.jsonl``) against the committed corpus.

They run what ``lexgraph eval validate-golden`` checks and also test the answer patterns: the
required facts must match answers that write numbers like the statutes (``"2 (duas) horas"``)
and every forbidden-claim guard must catch an answer that accepts the false premise of its
question, without flagging the answer that rejects it.
"""

import re

import pytest

from lexgraph.corpus import Corpus
from lexgraph.eval.golden import GOLDEN_FILE, GoldenItem, load_golden, validate_against_corpus
from lexgraph.eval.rules import check_reference, forbidden_claims, missing_patterns
from lexgraph.ingestion.chunking import chunk_article
from tests.conftest import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "data" / "eval" / GOLDEN_FILE
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
GOLDEN_SIZE = 55

pytestmark = pytest.mark.skipif(
    not (GOLDEN_PATH.is_file() and CORPUS_DIR.is_dir()),
    reason="the golden set or the corpus is not in the checkout",
)

_UNITS = [
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
]
_TENS = {
    2: "vinte",
    3: "trinta",
    4: "quarenta",
    5: "cinqüenta",
    6: "sessenta",
    7: "setenta",
    8: "oitenta",
    9: "noventa",
}
_QUANTITY = re.compile(
    r"\b(\d{1,3})( ?%| (?:dias?|horas?|semanas?|meses|mês|anos?|minutos|segundos)\b)"
)


def spelled(number: int) -> str:
    """Portuguese words for ``number`` (0 to 199), in the masculine form."""
    if number < len(_UNITS):
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] + (f" e {_UNITS[unit]}" if unit else "")
    if number == 100:
        return "cem"
    if number < 200:
        return f"cento e {spelled(number - 100)}"
    msg = f"no words for {number}"
    raise ValueError(msg)


def statute_style(text: str) -> str:
    """``text`` with every quantity written like the statutes: ``"2 (dois) horas"``."""

    def spell(match: re.Match[str]) -> str:
        digits, unit = match.groups()
        words = spelled(int(digits))
        if unit.strip() == "%":
            return f"{digits}{unit} ({words} por cento)"
        return f"{digits} ({words}){unit}"

    return _QUANTITY.sub(spell, text)


# Answers that accept the false premise (or the outdated rule) of each guarded item.
PREMISE_ACCEPTED = {
    "adv-001": [
        "SIM.",
        "Sim. A multa de 10% é permitida, pois não há limite para ela.",
        "Sim, a multa de dez por cento pode ser cobrada [1].",
    ],
    "adv-002": [
        "Sim, o CDC garante 30 dias para desistir. Não esqueça de guardar a nota fiscal.",
        "Correto: são 30 (trinta) dias, então ainda dá tempo de devolver o tênis.",
    ],
    "adv-003": ["Com 10 faltas injustificadas, você fica com 20 dias de férias."],
    "adv-004": [
        (
            "Você está certo: as redes sociais guardam os registros por 5 anos. Já a operadora "
            "não tem prazo definido."
        ),
    ],
    "cf-003": [
        "O limite é de 44 horas semanais; não houve mudança nesse ponto.",
        "A jornada normal é de 8 (oito) horas diárias e 44 (quarenta e quatro) horas semanais.",
    ],
    "lgpd-009": [
        "A ANPD é a Autoridade Nacional de Proteção de Dados, que não tem autonomia financeira.",
    ],
}
# Answers that reject the premise (or mention the old rule as past) and must not be flagged.
PREMISE_REJECTED = {
    "adv-001": [
        "Não. A multa de mora não pode superar 2% da prestação (CDC, art. 52, § 1º); 10% é abusivo.",
        "Não. Pelo art. 52, § 1º, do CDC, a multa de 10% fica limitada a 2% [1].",
    ],
    "adv-002": [
        "Não. O prazo é de 7 dias, e não de 30 dias [1].",
        "O prazo de 30 dias mencionado está equivocado: são 7 dias, contados do recebimento.",
    ],
    "adv-003": ["Com 10 faltas injustificadas, você tem direito a 24 dias corridos de férias [1]."],
    "adv-004": ["Não: o prazo não é de 5 anos, mas de 6 meses (MCI, art. 15) [1]."],
    "cf-003": [
        "O limite atual é de 40 horas semanais [1]. Antes da EC nº 141/2026, era de 44 horas.",
        "O limite anterior, de 44 horas, foi reduzido para 40 horas semanais.",
    ],
    "lgpd-009": [
        (
            "Hoje a ANPD é a Agência Nacional de Proteção de Dados; antes, era a Autoridade "
            "Nacional de Proteção de Dados."
        ),
    ],
}


@pytest.fixture(scope="module")
def items() -> dict[str, GoldenItem]:
    return {item.id: item for item in load_golden(GOLDEN_PATH)}


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load(CORPUS_DIR)


def test_golden_set_is_valid_against_the_corpus(
    items: dict[str, GoldenItem], corpus: Corpus
) -> None:
    assert len(items) == GOLDEN_SIZE
    assert validate_against_corpus(list(items.values()), corpus) == []
    assert [problem for item in items.values() for problem in check_reference(item)] == []


def test_expected_units_are_units_of_their_articles(
    items: dict[str, GoldenItem], corpus: Corpus
) -> None:
    with_units = [item for item in items.values() if item.expected_units]
    assert with_units
    for item in with_units:
        for article_id, units in item.expected_units.items():
            article = corpus.get(article_id)
            assert article is not None
            chunked = {unit for chunk in chunk_article(article) for unit in chunk.units}
            assert set(units) <= chunked, (item.id, article_id, sorted(set(units) - chunked))


def test_statute_style_spells_out_quantities() -> None:
    assert statute_style("até 2 horas, 50% e 120 dias") == (
        "até 2 (dois) horas, 50% (cinqüenta por cento) e 120 (cento e vinte) dias"
    )
    assert statute_style("art. 59 e 10 % de 1.000") == "art. 59 e 10 % (dez por cento) de 1.000"


def test_required_facts_match_answers_written_in_statute_style(
    items: dict[str, GoldenItem],
) -> None:
    rewritten = {item.id: statute_style(item.reference_answer) for item in items.values()}
    for item_id in ("clt-001", "clt-002", "clt-004", "clt-006", "cross-001", "adv-003", "adv-004"):
        assert rewritten[item_id] != items[item_id].reference_answer
    for item in items.values():
        assert missing_patterns(item.must_include, rewritten[item.id]) == [], item.id


def test_every_adversarial_guard_has_cases(items: dict[str, GoldenItem]) -> None:
    guarded = {i.id for i in items.values() if i.type == "adversarial" and i.must_not_include}
    assert guarded <= set(PREMISE_ACCEPTED)
    assert set(PREMISE_ACCEPTED) == set(PREMISE_REJECTED)


@pytest.mark.parametrize(
    ("item_id", "answer"),
    [(item_id, answer) for item_id, answers in PREMISE_ACCEPTED.items() for answer in answers],
)
def test_guards_catch_answers_that_accept_the_premise(
    items: dict[str, GoldenItem], item_id: str, answer: str
) -> None:
    assert forbidden_claims(items[item_id].must_not_include, answer)


@pytest.mark.parametrize(
    ("item_id", "answer"),
    [(item_id, answer) for item_id, answers in PREMISE_REJECTED.items() for answer in answers],
)
def test_guards_accept_answers_that_reject_the_premise(
    items: dict[str, GoldenItem], item_id: str, answer: str
) -> None:
    assert forbidden_claims(items[item_id].must_not_include, answer) == []
