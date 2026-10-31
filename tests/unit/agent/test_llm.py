"""Model roles built from settings, with provider HTTP mocked (no keys, no network)."""

import importlib.util
import json
from collections.abc import Callable
from typing import Any

import groq
import httpx
import pytest
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableSequence,
    RunnableWithFallbacks,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.constants import TAG_NOSTREAM

from lexgraph.agent import llm
from lexgraph.agent.schemas import QueryAnalysis
from lexgraph.config import Settings
from tests.unit.agent.fakes import FakeRetriever

type Handler = Callable[[httpx.Request], httpx.Response]


def tags(role: Runnable[Any, Any]) -> list[str] | None:
    """Tags a role attaches to every run, set with ``with_config``."""
    assert isinstance(role, RunnableBinding)
    return role.config.get("tags")


def answer_models(role: Runnable[Any, Any]) -> list[BaseChatModel]:
    """Chat models of the answer role, primary first (each one checks its answers)."""
    assert isinstance(role, RunnableBinding)
    bound = role.bound
    chains = (
        [bound.runnable, *bound.fallbacks] if isinstance(bound, RunnableWithFallbacks) else [bound]
    )
    models = []
    for chain in chains:
        assert isinstance(chain, RunnableSequence)
        assert isinstance(chain.first, BaseChatModel)
        models.append(chain.first)
    return models


ANALYSIS = {
    "intent": "legal_question",
    "standalone_question": "Posso desistir de uma compra feita pela internet?",
    "search_query": "direito de arrependimento compra fora do estabelecimento",
    "sub_queries": [],
    "laws": ["CDC"],
    "articles": [{"law": "CDC", "article": "49"}],
}
QUESTION = [HumanMessage("Posso desistir de uma compra feita pela internet?")]


class FakeProvider:
    """Mocked provider endpoint that records requests and answers with ``handler``."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


type Installer = Callable[[Handler, Handler], tuple[FakeProvider, FakeProvider]]


@pytest.fixture
def mocked_http(monkeypatch: pytest.MonkeyPatch, provider_keys: None) -> Installer:
    """Route every Groq and Gemini client built by ``llm`` to the given handlers."""

    def install(
        groq_handler: Handler, gemini_handler: Handler
    ) -> tuple[FakeProvider, FakeProvider]:
        groq_api, gemini_api = FakeProvider(groq_handler), FakeProvider(gemini_handler)

        def factory(model: str, *, model_provider: str, **kwargs: Any) -> Any:
            if model_provider == "groq":
                kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(groq_api))
                kwargs["http_async_client"] = httpx.AsyncClient(
                    transport=httpx.MockTransport(groq_api)
                )
            else:
                kwargs["client_args"] = {"transport": httpx.MockTransport(gemini_api)}
            return init_chat_model(model, model_provider=model_provider, **kwargs)

        monkeypatch.setattr(llm, "init_chat_model", factory)
        return groq_api, gemini_api

    return install


def groq_reply(content: str, finish_reason: str = "stop") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    return handler


def groq_error(status: int) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "falha simulada", "type": "tokens"}},
        )

    return handler


def groq_stream(*contents: str, reasoning: str = "", finish_reason: str = "stop") -> Handler:
    """Streamed Groq reply: optional reasoning, then ``contents`` as separate deltas."""

    def chunk(delta: dict[str, str], finish: str | None = None) -> str:
        payload = {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        deltas = [{"role": "assistant", "reasoning": reasoning}] if reasoning else []
        deltas += [{"content": content} for content in contents]
        events = "".join(
            chunk(delta, finish_reason if index == len(deltas) - 1 else None)
            for index, delta in enumerate(deltas)
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=events + "data: [DONE]\n\n",
        )

    return handler


def gemini_reply(text: str, finish_reason: str = "STOP") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": text}], "role": "model"},
                        "finishReason": finish_reason,
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

    return handler


def unreachable(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request to {request.url}")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("groq:openai/gpt-oss-120b", ("groq", "openai/gpt-oss-120b")),
        ("google_genai:gemini-3.5-flash-lite", ("google_genai", "gemini-3.5-flash-lite")),
        ("openai:gpt-6-luna", ("openai", "gpt-6-luna")),
    ],
)
def test_parse_model_spec(spec: str, expected: tuple[str, str]) -> None:
    assert llm.parse_model_spec(spec) == expected


@pytest.mark.parametrize("spec", ["gemini-3.5-flash-lite", "groq:", "mistral:large", ""])
def test_parse_model_spec_rejects_specs_without_a_known_provider(spec: str) -> None:
    with pytest.raises(ValueError, match="provider:model"):
        llm.parse_model_spec(spec)


@pytest.mark.usefixtures("provider_keys")
def test_groq_gpt_oss_hides_reasoning_and_bounds_every_call(settings: Settings) -> None:
    model = llm.build_chat_model(
        "groq:openai/gpt-oss-120b", settings, streaming=True, temperature=0.2
    )

    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.reasoning_effort == settings.llm_reasoning_effort
    assert model.model_kwargs == {"include_reasoning": False}
    assert model.reasoning_format is None
    assert model.max_retries == 0  # a rate limit goes straight to the fallback
    assert model.request_timeout == settings.llm_timeout
    assert model.max_tokens == settings.llm_max_tokens
    assert model.temperature == 0.2
    assert model.disable_streaming is False


@pytest.mark.usefixtures("provider_keys")
def test_groq_structured_model_does_not_stream(settings: Settings) -> None:
    model = llm.build_chat_model(
        "groq:openai/gpt-oss-20b", settings, streaming=False, temperature=0.0
    )

    assert model.disable_streaming is True


@pytest.mark.usefixtures("provider_keys")
def test_other_groq_models_get_no_reasoning_parameters(settings: Settings) -> None:
    model = llm.build_chat_model("groq:qwen/qwen3.8-27b", settings, streaming=True, temperature=0)

    assert isinstance(model, ChatGroq)
    assert model.reasoning_effort is None
    assert model.model_kwargs == {}


@pytest.mark.usefixtures("provider_keys")
def test_gemini_gets_attempts_and_timeout_but_no_temperature(settings: Settings) -> None:
    model = llm.build_chat_model(
        "google_genai:gemini-3.5-flash-lite", settings, streaming=True, temperature=0.2
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.max_retries == llm.GEMINI_ATTEMPTS
    assert model.timeout == settings.llm_timeout
    assert model.max_output_tokens == settings.llm_max_tokens
    assert model.temperature is None


@pytest.mark.usefixtures("provider_keys")
def test_openai_specs_use_the_optional_extra(settings: Settings) -> None:
    pytest.importorskip("langchain_openai")

    model = llm.build_chat_model("openai:gpt-6-luna", settings, streaming=True, temperature=0.2)

    assert type(model).__name__ == "ChatOpenAI"
    assert model.max_retries == llm.OPENAI_RETRIES  # type: ignore[attr-defined]


def test_openai_specs_explain_how_to_install_the_extra(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    find_spec = importlib.util.find_spec

    def without_openai(name: str, package: str | None = None) -> Any:
        return None if name == "langchain_openai" else find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", without_openai)

    with pytest.raises(ImportError, match="--extra openai"):
        llm.build_chat_model("openai:gpt-6-luna", settings, streaming=True, temperature=0.2)


async def test_structured_role_sends_a_strict_schema_to_gpt_oss(
    settings: Settings, mocked_http: Installer
) -> None:
    groq_api, _ = mocked_http(groq_reply(json.dumps(ANALYSIS)), unreachable)
    context = llm.build_agent_context(settings, FakeRetriever())

    result = await context.analyzer.ainvoke(QUESTION)

    assert result == QueryAnalysis.model_validate(ANALYSIS)
    (body,) = groq_api.bodies
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == settings.llm_max_tokens
    assert body["stream"] is False


async def test_structured_role_falls_back_to_gemini_at_once_on_rate_limit(
    settings: Settings, mocked_http: Installer
) -> None:
    groq_api, gemini_api = mocked_http(groq_error(429), gemini_reply(json.dumps(ANALYSIS)))
    context = llm.build_agent_context(settings, FakeRetriever())

    result = await context.analyzer.ainvoke(QUESTION)

    assert result.search_query == ANALYSIS["search_query"]
    assert len(groq_api.requests) == 1
    assert len(gemini_api.requests) == 1
    assert "gemini-3.5-flash-lite:generateContent" in str(gemini_api.requests[0].url)


async def test_structured_role_falls_back_when_the_reply_is_not_valid_json(
    settings: Settings, mocked_http: Installer
) -> None:
    groq_api, gemini_api = mocked_http(groq_reply("não é JSON"), gemini_reply(json.dumps(ANALYSIS)))
    context = llm.build_agent_context(settings, FakeRetriever())

    result = await context.analyzer.ainvoke(QUESTION)

    assert result.intent == "legal_question"
    assert len(groq_api.requests) == 1
    assert len(gemini_api.requests) == 1


async def test_tool_calling_without_a_tool_call_triggers_the_fallback(
    settings: Settings, mocked_http: Installer
) -> None:
    groq_api, gemini_api = mocked_http(
        groq_reply("Prefiro responder em texto."), gemini_reply(json.dumps(ANALYSIS))
    )
    primary = llm.build_chat_model(
        "groq:qwen/qwen3.8-27b", settings, streaming=False, temperature=0
    )
    fallback = llm.build_chat_model(
        "google_genai:gemini-3.5-flash-lite", settings, streaming=False, temperature=0
    )
    role = llm.structured_role(QueryAnalysis, primary, fallback, name="analyze_query")

    result = await role.ainvoke(QUESTION)

    assert result.laws == ["CDC"]
    assert groq_api.bodies[0]["tool_choice"]["function"]["name"] == "QueryAnalysis"
    assert len(gemini_api.requests) == 1


async def test_answer_role_falls_back_to_gemini_on_server_errors(
    settings: Settings, mocked_http: Installer
) -> None:
    groq_api, gemini_api = mocked_http(groq_error(503), gemini_reply("Resposta da reserva [1]."))
    context = llm.build_agent_context(settings, FakeRetriever())

    reply = await context.answerer.ainvoke(QUESTION)

    assert isinstance(reply, AIMessage)
    assert reply.text == "Resposta da reserva [1]."
    assert groq_api.bodies[0]["model"] == "openai/gpt-oss-120b"
    assert len(groq_api.requests) == 1
    assert len(gemini_api.requests) == 1


@pytest.mark.parametrize(
    "groq_answer",
    [groq_reply("  "), groq_reply("O prazo para desistir é de", finish_reason="length")],
    ids=["empty", "cut-off"],
)
async def test_an_incomplete_answer_is_replaced_by_the_fallback(
    settings: Settings, mocked_http: Installer, groq_answer: Handler
) -> None:
    groq_api, gemini_api = mocked_http(groq_answer, gemini_reply("Resposta da reserva [1]."))
    context = llm.build_agent_context(settings, FakeRetriever())

    reply = await context.answerer.ainvoke(QUESTION)

    assert reply.text == "Resposta da reserva [1]."
    assert len(groq_api.requests) == 1
    assert len(gemini_api.requests) == 1


async def test_a_gemini_answer_cut_off_at_the_token_limit_is_rejected(
    settings: Settings, mocked_http: Installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_keys(monkeypatch, "GOOGLE_API_KEY")
    mocked_http(unreachable, gemini_reply("O prazo é", finish_reason="MAX_TOKENS"))
    context = llm.build_agent_context(settings, FakeRetriever())

    with pytest.raises(OutputParserException, match="cut off"):
        await context.answerer.ainvoke(QUESTION)


async def test_a_streamed_answer_cut_off_at_the_token_limit_is_rejected(
    settings: Settings, mocked_http: Installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_keys(monkeypatch, "GROQ_API_KEY")
    mocked_http(groq_stream("O prazo ", "é de", finish_reason="length"), unreachable)
    context = llm.build_agent_context(settings, FakeRetriever())

    with pytest.raises(OutputParserException, match="cut off"):
        async for _ in context.answerer.astream(QUESTION):
            pass


async def test_without_a_fallback_the_provider_error_propagates(
    settings: Settings, mocked_http: Installer
) -> None:
    mocked_http(groq_error(429), unreachable)
    lone = settings.model_copy(update={"llm_fallback_model": None, "fast_llm_fallback_model": None})
    context = llm.build_agent_context(lone, FakeRetriever())

    with pytest.raises(groq.RateLimitError):
        await context.answerer.ainvoke(QUESTION)


async def test_streamed_reasoning_never_reaches_the_answer_text(
    settings: Settings, mocked_http: Installer
) -> None:
    stream = groq_stream("O prazo é de 7 dias", " [1].", reasoning="Vou pensar no art. 49...")
    mocked_http(stream, unreachable)
    context = llm.build_agent_context(settings, FakeRetriever())

    chunks = [chunk async for chunk in context.answerer.astream(QUESTION)]

    assert "".join(chunk.text for chunk in chunks) == "O prazo é de 7 dias [1]."


@pytest.mark.usefixtures("provider_keys")
def test_agent_context_tags_roles_and_takes_options_from_settings(settings: Settings) -> None:
    custom = settings.model_copy(update={"top_k": 9, "max_rewrites": 0, "verification": False})

    context = llm.build_agent_context(custom, FakeRetriever())

    roles = (context.analyzer, context.grader, context.rewriter, context.verifier)
    assert all(tags(role) == [TAG_NOSTREAM] for role in roles)
    assert tags(context.answerer) == [llm.ANSWER_TAG]
    assert context.options == llm.agent_options(custom)
    assert context.options.top_k == 9
    assert context.options.max_rewrites == 0
    assert context.options.verification is False
    assert context.options.rerank is None


def only_keys(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Leave exactly the provider keys in ``names`` set in the environment."""
    for keys in llm.PROVIDER_KEYS.values():
        for key in keys:
            monkeypatch.delenv(key, raising=False)
    for name in names:
        monkeypatch.setenv(name, f"test-{name.lower()}")


@pytest.mark.parametrize(
    ("keys", "spec", "expected"),
    [
        (("GROQ_API_KEY",), "groq:openai/gpt-oss-120b", True),
        (("GOOGLE_API_KEY",), "google_genai:gemini-3.5-flash-lite", True),
        (("GEMINI_API_KEY",), "google_genai:gemini-3.5-flash-lite", True),
        (("GOOGLE_API_KEY",), "groq:openai/gpt-oss-120b", False),
        ((), "openai:gpt-6-luna", False),
    ],
)
def test_provider_configured_reads_the_provider_keys(
    monkeypatch: pytest.MonkeyPatch, keys: tuple[str, ...], spec: str, expected: bool
) -> None:
    only_keys(monkeypatch, *keys)

    assert llm.provider_configured(spec) is expected


def test_a_missing_fallback_key_leaves_only_the_primary(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    only_keys(monkeypatch, "GROQ_API_KEY")

    context = llm.build_agent_context(settings, FakeRetriever())

    (model,) = answer_models(context.answerer)
    assert isinstance(model, ChatGroq)
    assert llm.models_configured(settings)


def test_the_fallback_answers_alone_when_only_its_key_is_set(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    only_keys(monkeypatch, "GOOGLE_API_KEY")

    context = llm.build_agent_context(settings, FakeRetriever())

    (model,) = answer_models(context.answerer)
    assert isinstance(model, ChatGoogleGenerativeAI)


@pytest.mark.usefixtures("provider_keys")
def test_the_answer_role_keeps_its_fallback(settings: Settings) -> None:
    context = llm.build_agent_context(settings, FakeRetriever())

    primary, fallback = answer_models(context.answerer)
    assert isinstance(primary, ChatGroq)
    assert isinstance(fallback, ChatGoogleGenerativeAI)


@pytest.mark.usefixtures("provider_keys")
def test_a_rate_limiter_paces_every_model(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[BaseChatModel] = []

    def recording(model: str, **kwargs: Any) -> Any:
        chat_model = init_chat_model(model, **kwargs)
        built.append(chat_model)
        return chat_model

    monkeypatch.setattr(llm, "init_chat_model", recording)
    limiter = InMemoryRateLimiter(requests_per_second=1)

    llm.build_agent_context(settings, FakeRetriever(), rate_limiter=limiter)

    assert len(built) == 4  # fast model, main model and the fallback of each
    assert all(model.rate_limiter is limiter for model in built)


@pytest.mark.parametrize(
    ("keys", "changes", "expected"),
    [
        (("GROQ_API_KEY",), {}, True),
        (("GOOGLE_API_KEY",), {}, True),
        (("GROQ_API_KEY",), {"fast_llm_model": "google_genai:gemini-3.5-flash-lite"}, False),
        (
            ("OPENAI_API_KEY",),
            {"llm_model": "openai:gpt-6-luna", "llm_fallback_model": None},
            False,
        ),
        ((), {}, False),
    ],
    ids=["groq", "gemini", "fast-role-without-key", "answer-model-only", "no-key"],
)
def test_models_are_configured_only_when_every_role_has_a_key(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    keys: tuple[str, ...],
    changes: dict[str, Any],
    expected: bool,
) -> None:
    only_keys(monkeypatch, *keys)

    assert llm.models_configured(settings.model_copy(update=changes)) is expected


async def test_roles_without_any_key_explain_how_to_configure_one(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    only_keys(monkeypatch)

    context = llm.build_agent_context(settings, FakeRetriever())

    assert not llm.models_configured(settings)
    with pytest.raises(llm.ModelNotConfiguredError, match="GROQ_API_KEY"):
        await context.analyzer.ainvoke(QUESTION)
    with pytest.raises(llm.ModelNotConfiguredError):
        await context.answerer.ainvoke(QUESTION)
