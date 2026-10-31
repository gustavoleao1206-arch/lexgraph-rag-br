"""Decoding and linearization of the raw pages."""

from lexgraph.ingestion.planalto import (
    ANCHOR_END,
    ANCHOR_START,
    LINK_END,
    LINK_START,
    STRUCK_END,
    STRUCK_START,
    decode_page,
    linearize,
)


def test_decode_page_reads_windows_1252() -> None:
    assert decode_page("<p>Art. 1º Proteção “integral”</p>".encode("cp1252")) == (
        "<p>Art. 1º Proteção “integral”</p>"
    )


def test_decode_page_drops_premature_document_end() -> None:
    html = b"<html><body><p>Art. 24. Texto.</p></div></body></html><p>Art. 25. Texto.</p>"

    assert linearize(decode_page(html)) == ["Art. 24. Texto.", "Art. 25. Texto."]


def test_decode_page_replaces_undecodable_bytes() -> None:
    assert decode_page(b"<p>Art. 1\x81</p>") == "<p>Art. 1\ufffd</p>"


def test_linearize_breaks_lines_at_blocks_and_br() -> None:
    html = "<p>TÍTULO I<br>INTRODUÇÃO</p><div>Art. 1º   Esta\nConsolidação</div><p> </p>"

    assert linearize(html) == ["TÍTULO I", "INTRODUÇÃO", "Art. 1º Esta Consolidação"]


def test_linearize_marks_anchors_links_and_struck_text() -> None:
    html = (
        '<p><a name="art5"></a>Art. 5º Texto <strike>antigo</strike> '
        '<span style="text-decoration: line-through">velho</span> '
        '<a href="L1.htm">(Vide Lei)</a></p>'
    )

    expected = (
        f"{ANCHOR_START}art5{ANCHOR_END}Art. 5º Texto {STRUCK_START}antigo{STRUCK_END} "
        f"{STRUCK_START}velho{STRUCK_END} {LINK_START}(Vide Lei){LINK_END}"
    )
    assert linearize(html) == [expected]


def test_linearize_turns_superscript_o_into_an_ordinal() -> None:
    html = "<p>Art. 2<sup>o</sup> Texto do § 1<u><sup>o</sup></u> e da 1<sup>a</sup> parte</p>"

    assert linearize(html) == ["Art. 2º Texto do § 1º e da 1ª parte"]


def test_linearize_maps_c1_references_back_to_windows_1252() -> None:
    assert linearize("<p>IX &#150; remuneração &#147;noturna&#148;</p>") == [
        "IX \u2013 remuneração “noturna”"
    ]


def test_linearize_keeps_small_table_cells_on_one_line() -> None:
    html = "<table><tr><td>Classe</td><td><p>Valor</p><p>R$ 1,00</p></td></tr></table>"

    assert linearize(html) == ["Classe | Valor R$ 1,00"]


def test_linearize_treats_cells_holding_articles_as_blocks() -> None:
    html = "<table><tr><td><p>Art. 1º Texto.</p><p>Art. 2º Texto.</p></td></tr></table>"

    assert linearize(html) == ["Art. 1º Texto.", "Art. 2º Texto."]


def test_linearize_skips_scripts_styles_and_comments() -> None:
    html = (
        "<html><head><title>Lei</title><style>p {}</style></head><body>"
        '<script id="f5_cspm">bot()</script><!-- nota --><p>Art. 1º Texto.</p></body></html>'
    )

    assert linearize(html) == ["Art. 1º Texto."]
