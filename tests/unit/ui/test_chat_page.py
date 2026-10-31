"""Chat page, run with Streamlit's ``AppTest`` against the in-memory API.

``AppTest.from_function`` runs the source of :func:`chat_page` as a standalone script, which is
why that function imports the page itself.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from lexgraph.agent import prompts
from lexgraph.domain import LawId
from lexgraph.schemas import (
    QUESTION_MAX_CHARS,
    ChatResponse,
    FinalEvent,
    Grounding,
    ResetEvent,
    StatusEvent,
    TokenEvent,
)
from lexgraph.ui.chat import (
    EXAMPLES,
    INTERRUPTED_ANSWER,
    MODELS_MISSING_HELP,
    RERANK_MISSING_HELP,
    RESET_ICON,
    THREAD_KEY,
    TURNS_KEY,
    Turn,
)
from tests.unit.api.fakes import SERVICE_INFO, error_events, sample_response, sample_source
from tests.unit.ui.fake_api import FakeApi, sse_body

if TYPE_CHECKING:
    from lexgraph.ui.client import ApiClient

QUESTION = "Comprei pela internet e me arrependi. Posso devolver?"
CDC_49_LINK = '[[1]](<https://www.planalto.gov.br/ccivil_03/leis/l8078.htm#art49> "CDC, art. 49")'


def chat_page(client: "ApiClient") -> None:
    from lexgraph.ui import chat

    chat.render(client)


@pytest.fixture
def fake_api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def app(fake_api: FakeApi) -> AppTest:
    """The chat page after its first run."""
    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}, default_timeout=30)
    return app.run()


def ask(app: AppTest, question: str = QUESTION) -> AppTest:
    return app.chat_input[0].set_value(question).run()


def markdown(app: AppTest) -> list[str]:
    return [element.value for element in app.markdown]


def test_first_visit_shows_the_welcome_screen(app: AppTest) -> None:
    assert not app.exception
    assert [title.value for title in app.title] == ["Pergunte à legislação"]
    assert [button.label for button in app.main.button] == list(EXAMPLES)
    assert len(app.chat_message) == 0


def test_sidebar_shows_the_api_status_and_search_options(app: AppTest) -> None:
    sidebar = [element.value for element in app.sidebar.markdown]

    assert ":green-badge[:material/cloud_done: API online]" in sidebar
    assert [caption.value for caption in app.sidebar.caption] == [
        "4.321 trechos indexados · busca híbrida (densa + BM25)"
    ]
    assert app.sidebar.button[0].label == "Nova conversa"
    assert app.pills[0].options == ["CLT", "CDC", "LGPD", "Marco Civil", "CF", "ADCT"]
    assert app.toggle[0].value is False


def test_question_streams_an_answer_with_citations(app: AppTest, fake_api: FakeApi) -> None:
    ask(app)

    assert not app.exception
    user, assistant = app.chat_message
    assert user.markdown[0].value == QUESTION
    answer_texts = [element.value for element in assistant.markdown]
    assert f"Sim. Você pode desistir da compra em até 7 dias {CDC_49_LINK}." in answer_texts
    assert ":green-badge[:material/verified: Fundamentada nas fontes]" in answer_texts
    assert [expander.label for expander in app.expander] == ["[1] CDC, art. 49"]
    assert (app.status[0].label, app.status[0].state) == ("Concluído em 1,2 s", "complete")
    steps = [element.value for element in app.status[0].markdown]
    assert steps == [
        ":material/psychology: Analisando a pergunta",
        ":material/search: Buscando nas leis",
    ]
    assert "1.020 tokens" in app.main.caption[-1].value


def test_question_is_sent_with_the_conversation_thread(app: AppTest, fake_api: FakeApi) -> None:
    ask(app)

    [request] = fake_api.chat_requests()
    assert request.question == QUESTION
    assert request.thread_id == app.session_state[THREAD_KEY]
    assert (request.laws, request.rerank) == (None, False)


def test_conversation_is_kept_across_reruns(app: AppTest, fake_api: FakeApi) -> None:
    ask(app).run()

    assert len(app.chat_message) == 2
    assert app.status[0].label == "Concluído em 1,2 s"
    assert len(fake_api.chat_requests()) == 1
    turns: list[Turn] = app.session_state[TURNS_KEY]
    assert [turn.response for turn in turns] == [sample_response()]


def test_example_buttons_ask_their_question(app: AppTest, fake_api: FakeApi) -> None:
    app.main.button[1].click().run()

    assert [request.question for request in fake_api.chat_requests()] == [EXAMPLES[1]]
    assert app.chat_message[0].markdown[0].value == EXAMPLES[1]


def test_search_options_are_sent_with_the_question(app: AppTest, fake_api: FakeApi) -> None:
    app.pills[0].set_value([LawId.CDC, LawId.CLT]).run()
    app.toggle[0].set_value(True).run()
    ask(app)

    [request] = fake_api.chat_requests()
    assert request.laws == [LawId.CDC, LawId.CLT]
    assert request.rerank is True


def test_rerank_defaults_to_the_server_setting(fake_api: FakeApi) -> None:
    fake_api.info = SERVICE_INFO.model_copy(update={"rerank": True})

    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}).run()

    assert app.toggle[0].value is True
    assert not app.toggle[0].disabled


def test_rerank_is_switched_off_when_the_server_has_no_reranker(
    app: AppTest, fake_api: FakeApi
) -> None:
    app.toggle[0].set_value(True).run()
    fake_api.info = SERVICE_INFO.model_copy(update={"rerank": True, "rerank_available": False})

    ask(app)

    toggle = app.toggle[0]
    assert (toggle.value, toggle.disabled, toggle.help) == (False, True, RERANK_MISSING_HELP)
    [request] = fake_api.chat_requests()
    assert request.rerank is False


def test_question_limit_comes_from_the_api(fake_api: FakeApi) -> None:
    fake_api.info = SERVICE_INFO.model_copy(update={"max_question_chars": 3000})
    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}).run()

    ask(app, "x" * 2500)

    assert not app.exception
    assert app.chat_input[0].proto.max_chars == 3000
    assert [len(request.question) for request in fake_api.chat_requests()] == [2500]


def test_question_limit_falls_back_to_the_hard_cap_without_the_api(fake_api: FakeApi) -> None:
    fake_api.offline = True

    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}).run()

    assert app.chat_input[0].proto.max_chars == QUESTION_MAX_CHARS


@pytest.mark.parametrize(
    "reason", [prompts.RESET_REGENERATION, prompts.RESET_MODEL_SWITCH, prompts.RESET_FINAL]
)
def test_reset_discards_the_draft_and_says_why(
    app: AppTest, fake_api: FakeApi, reason: str
) -> None:
    fake_api.stream = sse_body(
        StatusEvent(stage="generate", message="Escrevendo a resposta"),
        TokenEvent(text="Rascunho descartado"),
        ResetEvent(reason=reason),
        TokenEvent(text="Sim. Você pode desistir da compra em até 7 dias [1]."),
        FinalEvent(response=sample_response()),
    )

    ask(app)

    assert not any("Rascunho" in text for text in markdown(app))
    steps = [element.value for element in app.status[0].markdown]
    assert steps == [":material/draw: Escrevendo a resposta", f"{RESET_ICON} {reason}"]


def test_regenerated_answers_are_badged(app: AppTest, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(
        FinalEvent(response=sample_response().model_copy(update={"regenerations": 1}))
    )

    ask(app)

    assert ":violet-badge[:material/autorenew: Resposta revisada]" in markdown(app)


def test_question_survives_a_rerun_during_the_answer(app: AppTest, fake_api: FakeApi) -> None:
    def interrupted_stream() -> Iterator[bytes]:
        yield sse_body(
            StatusEvent(stage="analyze", message="Analisando a pergunta"),
            TokenEvent(text="Sim"),
            done=False,
        )
        # What a click on a sidebar option or on another page does while the answer streams.
        st.rerun()

    fake_api.stream = interrupted_stream()

    ask(app)

    assert not app.exception
    assert app.session_state[TURNS_KEY] == [Turn(QUESTION, error=INTERRUPTED_ANSWER)]
    assert app.chat_message[0].markdown[0].value == QUESTION
    assert [error.value for error in app.error] == [INTERRUPTED_ANSWER]
    assert len(fake_api.chat_requests()) == 1


def test_unsupported_claims_and_other_sources_are_shown(app: AppTest, fake_api: FakeApi) -> None:
    other = sample_source(2).model_copy(
        update={"chunk_id": "CDC:26#0", "citation": "CDC, art. 26", "rerank_score": 3.21}
    )
    response = sample_response().model_copy(
        update={
            "sources": [sample_source(), other],
            "grounding": Grounding(grounded=False, unsupported_claims=["O prazo é de 30 dias."]),
        }
    )
    fake_api.stream = sse_body(FinalEvent(response=response))

    ask(app)

    assert ":orange-badge[:material/warning: Confira as fontes]" in markdown(app)
    assert "- O prazo é de 30 dias." in app.warning[0].value
    assert [expander.label for expander in app.expander] == ["[1] CDC, art. 49", "[2] CDC, art. 26"]
    captions = [caption.value for caption in app.main.caption]
    assert ":material/library_books: Também consultadas" in captions
    assert "Pontuação da busca: 0,033 · reranking: 3,210" in captions


def test_smalltalk_answers_have_no_badges_or_sources(app: AppTest, fake_api: FakeApi) -> None:
    reply = ChatResponse(
        thread_id="t", answer="Olá! Pergunte sobre a legislação.", route="smalltalk"
    )
    fake_api.stream = sse_body(
        StatusEvent(stage="respond", message="Respondendo"), FinalEvent(response=reply)
    )

    ask(app)

    texts = [element.value for element in app.chat_message[1].markdown]
    assert "Olá! Pergunte sobre a legislação." in texts
    assert not any("-badge[" in text for text in texts)
    assert len(app.expander) == 0


def test_unavailable_answers_keep_their_sources_but_fail_the_status(
    app: AppTest, fake_api: FakeApi
) -> None:
    reply = sample_response().model_copy(
        update={
            "answer": prompts.GENERATION_FAILED_REPLY,
            "route": "unavailable",
            "citations": [],
            "grounding": None,
        }
    )
    fake_api.stream = sse_body(
        StatusEvent(stage="generate", message="Redigindo a resposta"), FinalEvent(response=reply)
    )

    ask(app)

    texts = markdown(app)
    assert prompts.GENERATION_FAILED_REPLY in texts
    assert ":red-badge[:material/cloud_off: Modelo indisponível]" in texts
    assert not any("autoverificação" in text for text in texts)
    assert [expander.label for expander in app.expander] == ["[1] CDC, art. 49"]
    assert (app.status[0].label, app.status[0].state) == ("Não foi possível responder", "error")


def test_sidebar_warns_when_the_api_has_no_language_model(fake_api: FakeApi) -> None:
    fake_api.info = SERVICE_INFO.model_copy(update={"models_configured": False})

    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}).run()

    [badge] = [element for element in app.sidebar.markdown if "key_off" in element.value]
    assert badge.value == ":orange-badge[:material/key_off: Modelos não configurados]"
    assert badge.proto.help == MODELS_MISSING_HELP


def test_service_errors_are_shown_to_the_user(app: AppTest, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(*error_events())

    ask(app)

    assert [error.value for error in app.error] == ["O provedor de linguagem está indisponível."]
    assert (app.status[0].label, app.status[0].state) == ("Não foi possível responder", "error")
    turns: list[Turn] = app.session_state[TURNS_KEY]
    assert turns[0].error == "O provedor de linguagem está indisponível."


def test_cut_stream_is_reported_as_incomplete(app: AppTest, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(TokenEvent(text="Sim"), done=False)

    ask(app)

    assert "Não foi possível falar com a API" in app.error[0].value
    assert len(app.status) == 0


def test_failed_turn_looks_the_same_when_rerendered(app: AppTest, fake_api: FakeApi) -> None:
    fake_api.stream = sse_body(*error_events())
    ask(app)
    live = [(status.label, status.state) for status in app.status], app.error[0].value

    app.run()

    assert ([(status.label, status.state) for status in app.status], app.error[0].value) == live


def test_offline_api_is_reported(fake_api: FakeApi) -> None:
    fake_api.offline = True

    app = AppTest.from_function(chat_page, kwargs={"client": fake_api.client()}).run()
    ask(app)

    sidebar = [element.value for element in app.sidebar.markdown]
    assert ":red-badge[:material/cloud_off: API indisponível]" in sidebar
    assert "Não foi possível falar com a API" in app.sidebar.caption[0].value
    assert "Não foi possível falar com a API" in app.error[0].value
    assert len(app.status) == 0


def test_currency_in_questions_is_not_rendered_as_latex(app: AppTest) -> None:
    ask(app, "A multa vai de R$ 100 a R$ 200?")

    assert app.chat_message[0].markdown[0].value == r"A multa vai de R\$ 100 a R\$ 200?"


def test_new_conversation_starts_a_new_thread(app: AppTest) -> None:
    ask(app)
    first_thread = app.session_state[THREAD_KEY]

    app.sidebar.button[0].click().run()

    assert app.session_state[TURNS_KEY] == []
    assert app.session_state[THREAD_KEY] != first_thread
    assert len(app.chat_message) == 0
    assert [title.value for title in app.title] == ["Pergunte à legislação"]
