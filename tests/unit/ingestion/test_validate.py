"""Consistency checks run before the corpus is written."""

from lexgraph.catalog import law_info
from lexgraph.domain import Article, LawId
from lexgraph.ingestion.validate import validate_corpus, validate_law
from tests.conftest import make_article

LAST = law_info(LawId.MCI).expected_last_article


def mci(*numbers: str, status: str = "vigente") -> list[Article]:
    """Marco Civil articles numbered as given, in that document order."""
    return [
        make_article(LawId.MCI, number, f"Art. {number}. Texto do artigo.", seq=seq, status=status)
        for seq, number in enumerate(numbers)
    ]


def complete(*extra: str) -> list[str]:
    return [str(n) for n in range(1, LAST + 1)] + list(extra)


def test_complete_law_has_no_problems() -> None:
    assert validate_law(LawId.MCI, mci(*complete())) == []


def test_suffixed_articles_and_preamble_are_accepted() -> None:
    numbers = ["preambulo", "1", "2", "2-A", "2-B", *complete()[2:]]

    assert validate_law(LawId.MCI, mci(*numbers)) == []


def test_no_articles() -> None:
    assert validate_law(LawId.MCI, []) == ["no articles parsed"]


def test_missing_articles_are_listed() -> None:
    numbers = [n for n in complete() if n not in {"5", "6"}]

    assert validate_law(LawId.MCI, mci(*numbers)) == ["missing articles: 5, 6"]


def test_long_listings_are_truncated() -> None:
    assert validate_law(LawId.MCI, mci("1")) == [
        f"missing articles: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (+{LAST - 11} more)"
    ]


def test_articles_beyond_the_expected_last() -> None:
    problems = validate_law(LawId.MCI, mci(*complete(str(LAST + 1))))

    assert problems == [f"articles beyond the expected last ({LAST}): {LAST + 1}"]


def test_duplicate_articles() -> None:
    numbers = complete()
    numbers.insert(7, "7")

    assert "duplicate articles: 7" in validate_law(LawId.MCI, mci(*numbers))


def test_articles_out_of_order() -> None:
    numbers = complete()
    numbers[8], numbers[9] = numbers[9], numbers[8]

    assert validate_law(LawId.MCI, mci(*numbers)) == ["articles out of order: 10 -> 9"]


def test_order_follows_seq_not_list_position() -> None:
    articles = mci(*complete())

    assert validate_law(LawId.MCI, list(reversed(articles))) == []


def test_residual_amendment_note() -> None:
    articles = mci(*complete())
    articles[2] = make_article(LawId.MCI, "3", "Art. 3. Texto (Redação dada pela Lei nº 1).", seq=2)

    assert validate_law(LawId.MCI, articles) == ["MCI:3: residual amendment note '(Redação dada'"]


def test_control_and_replacement_characters() -> None:
    articles = mci(*complete())
    articles[0] = make_article(LawId.MCI, "1", "Art. 1. IX \x96 texto.", seq=0)
    articles[1] = make_article(LawId.MCI, "2", "Art. 2. Prote\ufffdo.", seq=1)

    assert validate_law(LawId.MCI, articles) == [
        "MCI:1: unexpected character U+0096",
        "MCI:2: unexpected character U+FFFD",
    ]


def test_in_force_article_needs_a_body() -> None:
    articles = mci(*complete())
    articles[3] = make_article(LawId.MCI, "4", "Art. 4.", seq=3)
    articles[4] = make_article(LawId.MCI, "5", "Art. 5.", seq=4, status="revogado")
    articles[5] = make_article(LawId.MCI, "6", "", seq=5, status="vetado")
    articles[6] = make_article(LawId.MCI, "7", "Art. 7. e Parágrafo único.", seq=6)
    articles[7] = make_article(LawId.MCI, "8", "Art. 8. e §§ 1º e 2º", seq=7)
    articles[8] = make_article(LawId.MCI, "9", "Art. 9. Parágrafo único é o texto.", seq=8)

    assert validate_law(LawId.MCI, articles) == [
        "MCI:4: in force but without text",
        "MCI:6: empty text",
        "MCI:7: in force but without text",
        "MCI:8: in force but without text",
    ]


def test_every_article_needs_a_url() -> None:
    articles = mci(*complete())
    articles[0] = articles[0].model_copy(update={"url": ""})

    assert validate_law(LawId.MCI, articles) == ["MCI:1: missing URL"]


def test_article_of_another_law() -> None:
    articles = mci(*complete())
    articles.append(make_article(LawId.CDC, "49", seq=LAST))

    assert "CDC:49 belongs to MCI" in validate_law(LawId.MCI, articles)


def test_validate_corpus_reports_only_laws_with_problems() -> None:
    report = validate_corpus({LawId.MCI: mci(*complete()), LawId.LGPD: []})

    assert report == {LawId.LGPD: ["no articles parsed"]}
