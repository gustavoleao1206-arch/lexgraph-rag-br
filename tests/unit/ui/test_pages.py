"""Acervo and Sobre pages and the multipage entry point, run with Streamlit's ``AppTest``.

``AppTest.from_function`` runs the source of each page function below as a standalone script,
which is why those functions import the page themselves.
"""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

from lexgraph.config import Settings
from lexgraph.domain import LawId
from lexgraph.ui import app as app_module
from lexgraph.ui.about import DISCLAIMER
from lexgraph.ui.library import LOOKUP_KEY
from tests.conftest import make_article
from tests.unit.api.fakes import SERVICE_INFO
from tests.unit.ui.fake_api import FakeApi

if TYPE_CHECKING:
    from lexgraph.ui.client import ApiClient


def library_page(client: "ApiClient") -> None:
    from lexgraph.ui import library

    library.render(client)


def about_page(client: "ApiClient") -> None:
    from lexgraph.ui import about

    about.render(client)


def entry_point(client: "ApiClient") -> None:
    from lexgraph.ui import app

    app.main(client)


@pytest.fixture
def fake_api() -> FakeApi:
    return FakeApi()


def run(page: Callable[["ApiClient"], None], fake_api: FakeApi) -> AppTest:
    script = AppTest.from_function(page, kwargs={"client": fake_api.client()}, default_timeout=30)
    return script.run()


def look_up(app: AppTest, law_id: LawId, number: str) -> AppTest:
    app.selectbox[0].set_value(law_id)
    app.text_input[0].set_value(number)
    return app.button[0].click().run()


def test_library_lists_the_laws_with_totals(fake_api: FakeApi) -> None:
    app = run(library_page, fake_api)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {
        "Leis": "2",
        "Artigos": "1.157",
        "Em vigor": "972",
        "Última coleta": "30/10/2026",
    }
    table = app.dataframe[0].value
    assert list(table["Lei"]) == ["CLT", "CDC"]
    assert list(table["Coleta"]) == ["30/10/2026", "29/10/2026"]
    assert table["Texto oficial"][1].endswith("l8078compilado.htm")


def test_library_looks_up_an_article(fake_api: FakeApi) -> None:
    app = run(library_page, fake_api)

    look_up(app, LawId.CDC, "art. 49º")

    assert app.session_state[LOOKUP_KEY] == "CDC:49"
    texts = [element.value for element in app.markdown]
    assert "#### CDC, Art. 49" in texts
    assert ":green-badge[:material/check_circle: Em vigor]" in texts
    assert any(text.startswith("Art. 49. O consumidor pode desistir") for text in texts)
    assert [button.label for button in app.link_button] == ["Ler no Planalto"]


def test_library_shows_revoked_articles_with_their_notes(fake_api: FakeApi) -> None:
    revoked = make_article(LawId.CLT, "59-A", "Art. 59-A. (Revogado).", status="revogado")
    fake_api.articles["CLT:59-A"] = revoked.model_copy(
        update={"annotations": ["(Revogado pela Lei nº 13.467, de 2017)"]}
    )
    app = run(library_page, fake_api)

    look_up(app, LawId.CLT, "59a")

    texts = [element.value for element in app.markdown]
    assert ":red-badge[:material/cancel: Revogado]" in texts
    assert [expander.label for expander in app.expander] == ["Notas de alteração (1)"]


def test_library_shows_the_article_rubric(fake_api: FakeApi) -> None:
    fake_api.articles["CDC:49"] = make_article().model_copy(
        update={"rubric": "Direito de arrependimento"}
    )
    app = run(library_page, fake_api)

    look_up(app, LawId.CDC, "49")

    assert "*Direito de arrependimento*" in [element.value for element in app.markdown]


def test_library_reports_failed_article_lookups(fake_api: FakeApi) -> None:
    fake_api.failures["/v1/articles/CDC:49"] = 500
    app = run(library_page, fake_api)

    look_up(app, LawId.CDC, "49")

    assert "erro interno" in app.error[0].value


def test_library_reports_unknown_articles(fake_api: FakeApi) -> None:
    app = run(library_page, fake_api)

    look_up(app, LawId.CLT, "9999")

    assert app.warning[0].value == "O artigo 9999 não está no acervo de CLT."


def test_library_rejects_invalid_article_numbers(fake_api: FakeApi) -> None:
    app = run(library_page, fake_api)

    look_up(app, LawId.CLT, "inciso XI")

    assert app.session_state[LOOKUP_KEY] is None
    assert "Informe o número do artigo" in app.warning[0].value


def test_library_explains_an_empty_corpus(fake_api: FakeApi) -> None:
    fake_api.laws = []

    app = run(library_page, fake_api)

    assert "ainda não foi gerado" in app.info[0].value
    assert len(app.dataframe) == 0


def test_library_reports_api_failures(fake_api: FakeApi) -> None:
    fake_api.failures["/v1/laws"] = 401

    app = run(library_page, fake_api)

    assert "LEXGRAPH_API_KEY" in app.error[0].value


def test_about_page_explains_the_pipeline_and_its_limits(fake_api: FakeApi) -> None:
    app = run(about_page, fake_api)

    assert not app.exception
    assert app.warning[0].value == DISCLAIMER
    assert "não substituem orientação jurídica nem o texto publicado no DOU" in DISCLAIMER
    texts = "\n".join(element.value for element in app.markdown)
    assert "`groq:openai/gpt-oss-120b`" in texts
    assert "| Embeddings | `google/embeddinggemma-300m` (fastembed) |" in texts
    assert "| Reranking | desativado por padrão |" in texts
    assert "`lexgraph` · 4.321 trechos" in texts
    assert "**ADCT** — Ato das Disposições Constitucionais Transitórias" in texts


def test_about_page_reports_a_missing_reranker(fake_api: FakeApi) -> None:
    fake_api.info = SERVICE_INFO.model_copy(update={"rerank": True, "rerank_available": False})

    app = run(about_page, fake_api)

    texts = "\n".join(element.value for element in app.markdown)
    assert "| Reranking | indisponível (modelo não instalado) |" in texts


def test_about_page_works_without_the_api(fake_api: FakeApi) -> None:
    fake_api.offline = True

    app = run(about_page, fake_api)

    assert not app.exception
    assert "Não foi possível falar com a API" in app.info[0].value


def test_entry_point_opens_the_chat_page(fake_api: FakeApi) -> None:
    app = run(entry_point, fake_api)

    assert not app.exception
    assert [title.value for title in app.title] == ["Pergunte à legislação"]


@pytest.fixture
def empty_client_cache() -> Iterator[None]:
    app_module.default_client.clear()
    yield
    app_module.default_client.clear()


@pytest.mark.usefixtures("empty_client_cache")
def test_default_client_is_built_once_from_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    configured = settings.model_copy(
        update={"api_url": "http://api.internal:8000", "api_key": SecretStr("s3cret")}
    )
    created: list[tuple[str, str | None]] = []

    def fake_client(base_url: str, api_key: str | None) -> FakeApi:
        created.append((base_url, api_key))
        return FakeApi()

    monkeypatch.setattr(app_module, "get_settings", lambda: configured)
    monkeypatch.setattr(app_module, "ApiClient", fake_client)

    app_module.default_client()
    app_module.default_client()

    assert created == [("http://api.internal:8000", "s3cret")]
