"""Chat models for the agent roles, built from settings, each with a fallback provider.

The fast model (``fast_llm_model``) serves the four structured roles: query analysis,
relevance grading, query rewriting and the grounding check. The main model (``llm_model``)
writes the answers, the only output streamed to users. Model specs carry an explicit
provider prefix: ``groq:``, ``google_genai:`` or ``openai:`` (the last one needs the
``openai`` extra).

Provider notes:

- Groq's ``max_retries`` counts retries inside its SDK, which honors ``retry-after`` and may
  sleep up to 60 s first. On the free tier a 429 usually means the model's tokens per minute
  are spent, which a quick retry does not fix, so the SDK never retries: the fallback provider
  takes over at once (the graph still retries an answer that fails transiently).
- Gemini's ``max_retries`` counts attempts, and Gemini 3.x ignores sampling parameters, so
  no temperature is sent to it.
- gpt-oss models reason before answering. Groq does not support ``reasoning_format`` for
  them; ``include_reasoning=False`` keeps the reasoning out of the response altogether
  (langchain-groq would otherwise put it in ``additional_kwargs["reasoning_content"]``, so it
  never reaches the streamed ``content`` either way).
- No client has a default timeout, so ``llm_timeout`` is always set.
- An answer that comes back empty or cut off at ``llm_max_tokens`` (reasoning tokens count
  towards it) is treated as a failure of the model that wrote it, so the fallback takes over.
- A model whose provider API key is missing is left out, so the service still starts with
  only one provider configured. When a role has no usable model at all, the service starts
  anyway and every request explains how to configure a key.
"""

import importlib.util
import logging
import os
from collections.abc import Callable
from typing import Any, NoReturn, cast

import groq
import httpx
from google.genai import errors as genai_errors
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelError, ModelRateLimitError, OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_groq import ChatGroq
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel

from lexgraph.agent import prompts
from lexgraph.agent.context import AgentContext, AgentOptions
from lexgraph.agent.schemas import GroundingCheck, QueryAnalysis, QueryRewrite, RelevanceGrade
from lexgraph.config import Settings
from lexgraph.protocols import ChatServiceError, Retriever

logger = logging.getLogger(__name__)

PROVIDERS = frozenset({"groq", "google_genai", "openai"})
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}
"""Environment variables holding each provider's API key (any one of them is enough)."""
# langchain-groq keeps ``strict=True`` only for these models and drops it silently otherwise.
STRICT_SCHEMA_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
ANSWER_TAG = "final_answer"
# Deterministic routing and grading; answers stay close to the wording of the sources.
STRUCTURED_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2
GROQ_RETRIES = 0
GEMINI_ATTEMPTS = 2
OPENAI_RETRIES = 1
# Finish reasons of a reply cut off at the token limit (Groq and OpenAI, Gemini). Streamed
# metadata is merged by concatenating strings, so they are matched as substrings.
TRUNCATED_FINISH_REASONS = ("length", "MAX_TOKENS")


def _openai_errors(*names: str) -> tuple[type[Exception], ...]:
    """Error classes of the optional ``openai`` package, or nothing when it is absent."""
    try:
        import openai
    except ImportError:
        return ()
    return tuple(getattr(openai, name) for name in names)


LLM_ERRORS: tuple[type[Exception], ...] = (
    groq.APIError,
    genai_errors.APIError,
    ChatGoogleGenerativeAIError,
    ModelError,
    httpx.HTTPError,
    ValueError,  # also OutputParserException and pydantic's ValidationError
    *_openai_errors("APIError"),
)
"""Failures of a model call worth recovering from: provider, transport and parsing errors."""

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
    ModelRateLimitError,
    genai_errors.ServerError,
    httpx.TransportError,
    *_openai_errors("RateLimitError", "InternalServerError", "APIConnectionError"),
)
"""Failures that may succeed on a later attempt: rate limits, 5xx and connection errors."""


class ModelNotConfiguredError(ChatServiceError):
    """No model of a role has its provider API key set."""

    def __init__(self) -> None:
        super().__init__(prompts.MODELS_NOT_CONFIGURED)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``"groq:openai/gpt-oss-120b"`` into ``("groq", "openai/gpt-oss-120b")``."""
    provider, separator, model = spec.partition(":")
    if not separator or not model or provider not in PROVIDERS:
        msg = f"model spec {spec!r} must be 'provider:model' with provider in {sorted(PROVIDERS)}"
        raise ValueError(msg)
    return provider, model


def provider_configured(spec: str) -> bool:
    """Whether the API key of the provider of ``spec`` is set in the environment."""
    provider, _ = parse_model_spec(spec)
    return any(os.environ.get(name) for name in PROVIDER_KEYS[provider])


def models_configured(settings: Settings) -> bool:
    """Whether every role has a usable model (one whose provider key is set).

    The answer writer needs the main model or its fallback; the structured roles need the
    fast model or its fallback.
    """
    roles = (
        (settings.llm_model, settings.llm_fallback_model),
        (settings.fast_llm_model, settings.fast_llm_fallback_model),
    )
    return all(any(provider_configured(spec) for spec in specs if spec) for specs in roles)


def build_chat_model(
    spec: str,
    settings: Settings,
    *,
    streaming: bool,
    temperature: float,
    rate_limiter: BaseRateLimiter | None = None,
) -> BaseChatModel:
    """Chat model for ``spec`` with timeouts, retries and token limits from ``settings``.

    ``temperature`` applies to Groq only (see the module notes); ``streaming=False`` makes
    the model answer in one piece even inside a streamed graph run; ``rate_limiter`` paces
    its calls.
    """
    provider, model = parse_model_spec(spec)
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout,
        "max_tokens": settings.llm_max_tokens,
        "disable_streaming": not streaming,
        "rate_limiter": rate_limiter,
    }
    if provider == "groq":
        kwargs |= {"temperature": temperature, "max_retries": GROQ_RETRIES}
        if model.startswith("openai/gpt-oss"):
            kwargs |= {
                "reasoning_effort": settings.llm_reasoning_effort,
                "model_kwargs": {"include_reasoning": False},
            }
    elif provider == "google_genai":
        kwargs["max_retries"] = GEMINI_ATTEMPTS
    else:
        if importlib.util.find_spec("langchain_openai") is None:
            msg = f"{spec!r} needs the optional 'openai' dependencies: uv sync --extra openai"
            raise ImportError(msg)
        kwargs["max_retries"] = OPENAI_RETRIES
    chat_model: BaseChatModel = init_chat_model(model, model_provider=provider, **kwargs)
    return chat_model


def _require[SchemaT: BaseModel](schema: type[SchemaT]) -> Callable[[object], SchemaT]:
    def check(value: object) -> SchemaT:
        # Tool calling returns None, without an error, when the model skips the tool call.
        if not isinstance(value, schema):
            msg = f"the model returned no valid {schema.__name__}"
            raise OutputParserException(msg)
        return value

    return check


def structured_output[SchemaT: BaseModel](
    model: BaseChatModel, schema: type[SchemaT]
) -> Runnable[LanguageModelInput, SchemaT]:
    """``model`` answering with a ``schema`` instance, by the best method for its provider.

    Groq's gpt-oss models get strict JSON-schema decoding; other Groq models use tool
    calling; Gemini and OpenAI use their native JSON-schema mode.
    """
    if isinstance(model, ChatGroq):
        if model.model_name in STRICT_SCHEMA_MODELS:
            parsed = model.with_structured_output(schema, method="json_schema", strict=True)
        else:
            parsed = model.with_structured_output(schema, method="function_calling")
    else:
        parsed = model.with_structured_output(schema)
    return parsed | RunnableLambda(_require(schema), name=f"require_{schema.__name__}")


def _not_configured(_: object) -> NoReturn:
    raise ModelNotConfiguredError


def structured_role[SchemaT: BaseModel](
    schema: type[SchemaT],
    primary: BaseChatModel | None,
    fallback: BaseChatModel | None = None,
    *,
    name: str,
) -> Runnable[LanguageModelInput, SchemaT]:
    """A structured role that switches to ``fallback`` when ``primary`` fails.

    Tagged ``nostream`` so its output never mixes with the answer tokens of a graph run.
    Without any model it raises :class:`ModelNotConfiguredError` when called.
    """
    if primary is None:
        unavailable: Runnable[object, NoReturn] = RunnableLambda(_not_configured, name=name)
        return cast("Runnable[LanguageModelInput, SchemaT]", unavailable)
    role = structured_output(primary, schema)
    if fallback is not None:
        role = role.with_fallbacks(
            [structured_output(fallback, schema)], exceptions_to_handle=LLM_ERRORS
        )
    return role.with_config(tags=[TAG_NOSTREAM], run_name=name)


def _complete_answer(message: BaseMessage) -> BaseMessage:
    """``message`` when it holds a whole answer; raises when it is empty or was cut off."""
    if not message.text.strip():
        msg = "the model returned an empty answer"
        raise OutputParserException(msg)
    finish_reason = str(message.response_metadata.get("finish_reason") or "")
    if any(reason in finish_reason for reason in TRUNCATED_FINISH_REASONS):
        msg = f"the answer was cut off at the token limit (finish reason {finish_reason!r})"
        raise OutputParserException(msg)
    return message


def _with_answer_check(model: BaseChatModel) -> Runnable[LanguageModelInput, BaseMessage]:
    """``model``, failing on an empty or truncated answer instead of returning it."""
    return model | RunnableLambda(_complete_answer, name="require_complete_answer")


def answer_role(
    primary: BaseChatModel | None, fallback: BaseChatModel | None = None
) -> Runnable[LanguageModelInput, BaseMessage]:
    """The answer writer, tagged ``final_answer``; switches to ``fallback`` on failure.

    An empty or truncated answer counts as a failure, so it is never shown as an answer.
    """
    if primary is None:
        unavailable: Runnable[object, NoReturn] = RunnableLambda(
            _not_configured, name="write_answer"
        )
        return cast("Runnable[LanguageModelInput, BaseMessage]", unavailable)
    role = _with_answer_check(primary)
    if fallback is not None:
        role = role.with_fallbacks([_with_answer_check(fallback)], exceptions_to_handle=LLM_ERRORS)
    return role.with_config(tags=[ANSWER_TAG], run_name="write_answer")


def _model_pair(
    primary_spec: str,
    fallback_spec: str | None,
    settings: Settings,
    *,
    streaming: bool,
    temperature: float,
    rate_limiter: BaseRateLimiter | None,
) -> tuple[BaseChatModel | None, BaseChatModel | None]:
    """The models of a role whose provider key is set: (primary, fallback), in that order.

    When only the fallback is configured it becomes the primary; when neither is, both are
    ``None``.
    """
    usable: list[BaseChatModel] = []
    for spec in (primary_spec, fallback_spec):
        if spec is None:
            continue
        if not provider_configured(spec):
            logger.warning("Skipping model %s: its provider API key is not set", spec)
            continue
        usable.append(
            build_chat_model(
                spec,
                settings,
                streaming=streaming,
                temperature=temperature,
                rate_limiter=rate_limiter,
            )
        )
    primary = usable[0] if usable else None
    fallback = usable[1] if len(usable) > 1 else None
    return primary, fallback


def agent_options(settings: Settings) -> AgentOptions:
    """Agent behavior configured in ``settings``; reranking follows the retriever default."""
    return AgentOptions(
        top_k=settings.top_k,
        context_k=settings.context_k,
        max_rewrites=settings.max_rewrites,
        max_regenerations=settings.max_regenerations,
        grading=settings.grading,
        verification=settings.verification,
        history_turns=settings.history_turns,
    )


def build_agent_context(
    settings: Settings,
    retriever: Retriever,
    options: AgentOptions | None = None,
    *,
    rate_limiter: BaseRateLimiter | None = None,
) -> AgentContext:
    """Every model role plus the retriever, ready to be passed to a graph run.

    ``rate_limiter`` paces every model call, of every role and provider alike. The service
    runs without one; ``lexgraph eval answers`` passes one (``--rpm``) so that a batch of
    questions stays within the free-tier request quotas.
    """
    fast, fast_fallback = _model_pair(
        settings.fast_llm_model,
        settings.fast_llm_fallback_model,
        settings,
        streaming=False,
        temperature=STRUCTURED_TEMPERATURE,
        rate_limiter=rate_limiter,
    )
    main, main_fallback = _model_pair(
        settings.llm_model,
        settings.llm_fallback_model,
        settings,
        streaming=True,
        temperature=ANSWER_TEMPERATURE,
        rate_limiter=rate_limiter,
    )
    return AgentContext(
        retriever=retriever,
        analyzer=structured_role(QueryAnalysis, fast, fast_fallback, name="analyze_query"),
        grader=structured_role(RelevanceGrade, fast, fast_fallback, name="grade_passages"),
        rewriter=structured_role(QueryRewrite, fast, fast_fallback, name="rewrite_query"),
        verifier=structured_role(GroundingCheck, fast, fast_fallback, name="check_grounding"),
        answerer=answer_role(main, main_fallback),
        options=options or agent_options(settings),
    )
