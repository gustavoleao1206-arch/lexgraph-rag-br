"""Chat page: questions with live agent progress, streamed answers and cited sources.

The conversation lives in the session state as a list of :class:`Turn`; the server keeps its
own copy under the same ``thread_id``, which gives the agent the context of follow-up
questions.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import streamlit as st

from lexgraph.catalog import law_info
from lexgraph.domain import LawId
from lexgraph.schemas import (
    QUESTION_MAX_CHARS,
    ChatEvent,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    ResetEvent,
    ServiceInfo,
    SourcesEvent,
    Stage,
    StatusEvent,
    TokenEvent,
)
from lexgraph.ui.client import ApiClient, ApiError
from lexgraph.ui.components import ASSISTANT_AVATAR, USER_AVATAR, render_response
from lexgraph.ui.formatting import (
    describe_api_error,
    escape_markdown,
    format_int,
    format_seconds,
    retrieval_mode_label,
)

TURNS_KEY = "chat_turns"
THREAD_KEY = "chat_thread_id"
PENDING_KEY = "chat_pending_question"
LAWS_KEY = "chat_laws"
RERANK_KEY = "chat_rerank"

CURSOR = " ▌"
INPUT_PLACEHOLDER = "Escreva sua pergunta sobre a legislação…"
FIRST_STEP_LABEL = "Enviando a pergunta…"
FAILED_LABEL = "Não foi possível responder"
INCOMPLETE_ANSWER = "A resposta não foi concluída. Tente novamente."
INTERRUPTED_ANSWER = "A resposta foi interrompida antes de terminar. Envie a pergunta de novo."
RESET_ICON = ":material/restart_alt:"
RERANK_HELP = (
    "Reavalia os trechos encontrados com um modelo de reranking: mais preciso, porém mais lento."
)
RERANK_MISSING_HELP = "Indisponível: o modelo de reranking não está instalado no servidor da API."
MODELS_MISSING_HELP = (
    "A API não tem chave de nenhum provedor de modelos de linguagem (GROQ_API_KEY ou "
    "GOOGLE_API_KEY), então as perguntas não serão respondidas."
)
SCOPE = (
    "CLT, Código de Defesa do Consumidor, LGPD, Marco Civil da Internet "
    "e Constituição Federal (com o ADCT)"
)
EXAMPLES = (
    "Comprei um produto pela internet e me arrependi. Posso devolver?",
    "Qual é a jornada máxima de trabalho prevista na CLT?",
    "Quais direitos o titular tem sobre seus dados pessoais?",
    "O provedor responde por conteúdo publicado por terceiros?",
    "A Constituição garante a duração razoável do processo?",
)
STAGE_ICONS: dict[Stage, str] = {
    "analyze": ":material/psychology:",
    "retrieve": ":material/search:",
    "grade": ":material/filter_alt:",
    "rewrite": ":material/edit_note:",
    "generate": ":material/draw:",
    "verify": ":material/fact_check:",
    "respond": ":material/chat:",
}


@dataclass(frozen=True)
class Turn:
    """A question and what the assistant produced for it."""

    question: str
    response: ChatResponse | None = None
    error: str | None = None
    steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatOptions:
    """Search options chosen in the sidebar."""

    laws: list[LawId]
    rerank: bool


def render(client: ApiClient) -> None:
    """Render the chat page."""
    st.set_page_config(layout="centered")
    _init_session()
    info, problem = _service_info(client)
    options = _sidebar(info, problem)
    turns: list[Turn] = st.session_state[TURNS_KEY]
    typed = st.chat_input(INPUT_PLACEHOLDER, max_chars=_question_limit(info), submit_mode="disable")
    question = (typed or st.session_state.pop(PENDING_KEY, None) or "").strip()
    # A fixed slot replaces the welcome block at once; otherwise its stale buttons would
    # stay on screen until the streamed answer finishes.
    welcome_slot = st.empty()
    if not turns and not question:
        with welcome_slot.container():
            _welcome()
    for turn in turns:
        _render_turn(turn)
    if question:
        # Stored before the answer streams: a click elsewhere on the page (a sidebar option,
        # another page) stops this run to start a new one, and the question must survive it.
        turns.append(Turn(question, error=INTERRUPTED_ANSWER))
        turns[-1] = _ask(client, question, options)


def _init_session() -> None:
    st.session_state.setdefault(TURNS_KEY, [])
    st.session_state.setdefault(THREAD_KEY, uuid4().hex)


def _new_conversation() -> None:
    st.session_state[TURNS_KEY] = []
    st.session_state[THREAD_KEY] = uuid4().hex


def _ask_example(question: str) -> None:
    st.session_state[PENDING_KEY] = question


def _sidebar(info: ServiceInfo | None, problem: str | None) -> ChatOptions:
    rerank_missing = info is not None and not info.rerank_available
    if rerank_missing:
        st.session_state[RERANK_KEY] = False
    st.session_state.setdefault(RERANK_KEY, info.rerank if info is not None else False)
    with st.sidebar:
        st.button(
            "Nova conversa",
            icon=":material/add_comment:",
            type="primary",
            width="stretch",
            on_click=_new_conversation,
        )
        st.subheader("Busca", anchor=False)
        laws = st.pills(
            "Leis consultadas",
            list(LawId),
            selection_mode="multi",
            format_func=lambda law_id: law_info(law_id).short_name,
            key=LAWS_KEY,
            persist_state="session",
            help="Sem seleção, o assistente escolhe as leis a partir da pergunta.",
        )
        rerank = st.toggle(
            "Reordenar com cross-encoder",
            key=RERANK_KEY,
            persist_state="session",
            disabled=rerank_missing,
            help=RERANK_MISSING_HELP if rerank_missing else RERANK_HELP,
        )
        st.divider()
        _service_status(info, problem)
    return ChatOptions(laws=laws, rerank=rerank)


def _service_info(client: ApiClient) -> tuple[ServiceInfo | None, str | None]:
    try:
        return client.info(), None
    except ApiError as error:
        return None, describe_api_error(error)


def _question_limit(info: ServiceInfo | None) -> int:
    """Longest question the API accepts; the schema's hard cap while the API is unreachable."""
    if info is None:
        return QUESTION_MAX_CHARS
    return min(info.max_question_chars, QUESTION_MAX_CHARS)


def _service_status(info: ServiceInfo | None, problem: str | None) -> None:
    if info is None:
        st.badge("API indisponível", icon=":material/cloud_off:", color="red")
        st.caption(problem or "")
        return
    st.badge("API online", icon=":material/cloud_done:", color="green")
    if not info.models_configured:
        st.badge(
            "Modelos não configurados",
            icon=":material/key_off:",
            color="orange",
            help=MODELS_MISSING_HELP,
        )
    mode = retrieval_mode_label(info.retrieval_mode)
    st.caption(f"{format_int(info.indexed_chunks)} trechos indexados · busca {mode}")


def _welcome() -> None:
    st.title("Pergunte à legislação", anchor=False)
    st.markdown(
        f"Respostas com citação de artigo, baseadas nos textos oficiais compilados de {SCOPE}."
    )
    st.caption("Experimente uma destas perguntas:")
    with st.container(horizontal=True, gap="small"):
        for example in EXAMPLES:
            st.button(
                example,
                icon=":material/lightbulb:",
                on_click=_ask_example,
                args=(example,),
                wrap=True,
            )


def _render_turn(turn: Turn) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(escape_markdown(turn.question))
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if turn.steps:
            label, state = _outcome_status(turn)
            box = st.status(label, state=state)
            for step in turn.steps:
                box.markdown(step)
        _render_outcome(turn)


def _outcome_status(turn: Turn) -> tuple[str, Literal["complete", "error"]]:
    response = turn.response
    if response is None or response.route == "unavailable":
        return FAILED_LABEL, "error"
    return f"Concluído em {format_seconds(response.latency_ms)}", "complete"


def _render_outcome(turn: Turn) -> None:
    if turn.response is not None:
        render_response(turn.response)
    else:
        st.error(turn.error or INCOMPLETE_ANSWER, icon=":material/error:")


def _ask(client: ApiClient, question: str, options: ChatOptions) -> Turn:
    request = ChatRequest(
        question=question,
        thread_id=st.session_state[THREAD_KEY],
        laws=options.laws or None,
        rerank=options.rerank,
    )
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer = LiveAnswer()
        try:
            for event in client.stream_chat(request):
                answer.handle(event)
        except ApiError as error:
            answer.fail(describe_api_error(error))
        return answer.finish(question)


class LiveAnswer:
    """Shows one answer while it streams: agent steps in a status box, text below it.

    Once finished it looks exactly like the same turn rendered again from the history.
    """

    def __init__(self) -> None:
        self._status_slot = st.empty()
        self._status = self._status_slot.status(FIRST_STEP_LABEL)
        self._body = st.empty()
        self._text = ""
        self._steps: list[str] = []
        self._response: ChatResponse | None = None
        self._error: str | None = None

    def handle(self, event: ChatEvent) -> None:
        """Update the display with one event from the stream."""
        match event:
            case StatusEvent():
                self._step(f"{STAGE_ICONS[event.stage]} {event.message}")
                self._status.update(label=event.message)
            case TokenEvent():
                self._text += event.text
                self._body.markdown(escape_markdown(self._text) + CURSOR)
            case ResetEvent():
                self._text = ""
                self._body.empty()
                self._step(f"{RESET_ICON} {escape_markdown(event.reason)}")
            case FinalEvent():
                self._response = event.response
            case ErrorEvent():
                self._error = event.message
            case SourcesEvent():
                pass  # The final response carries the sources with their citations.

    def fail(self, message: str) -> None:
        """Record a failure that interrupted the stream."""
        self._error = message

    def finish(self, question: str) -> Turn:
        """Replace the streamed text with the final answer (or the error) and close the box."""
        steps = tuple(self._steps)
        if self._response is not None:
            turn = Turn(question, response=self._response, steps=steps)
        else:
            turn = Turn(question, error=self._error or INCOMPLETE_ANSWER, steps=steps)
        if steps:
            label, state = _outcome_status(turn)
            self._status.update(label=label, state=state)
        else:
            self._status_slot.empty()
        with self._body.container():
            _render_outcome(turn)
        return turn

    def _step(self, line: str) -> None:
        self._steps.append(line)
        self._status.markdown(line)
