"""Wire models shared by the chat service, the HTTP API, the CLI and the UI.

The streaming endpoint sends Server-Sent Events whose ``event`` field is the ``type`` of one
of the :data:`ChatEvent` models below and whose ``data`` field is that model as JSON. The
stream always ends with a ``final`` (or ``error``) event followed by ``done``.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from lexgraph.domain import LawId

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
QUESTION_MAX_CHARS = 8000
"""Hard upper bound; the server enforces its configurable (lower) limit on top of it."""

Route = Literal["answered", "no_context", "smalltalk", "out_of_scope", "unavailable"]
"""How a question was handled; ``unavailable`` means no language model could answer it."""

Stage = Literal["analyze", "retrieve", "grade", "rewrite", "generate", "verify", "respond"]


class ChatRequest(BaseModel):
    """A user question, optionally continuing a conversation thread."""

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)
    laws: list[LawId] | None = Field(
        default=None,
        max_length=len(LawId),
        description="Restrict the search to these laws (default: let the agent decide).",
    )
    rerank: bool | None = Field(
        default=None, description="Override the server default for cross-encoder reranking."
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, question: str) -> str:
        stripped = question.strip()
        if not stripped:
            msg = "question must not be blank"
            raise ValueError(msg)
        return stripped

    @field_validator("laws")
    @classmethod
    def _dedupe_laws(cls, laws: list[LawId] | None) -> list[LawId] | None:
        return list(dict.fromkeys(laws)) if laws else laws


class Source(BaseModel):
    """A passage that was given to the model; ``index`` is the ``[n]`` marker used in answers."""

    index: int = Field(ge=1)
    chunk_id: str
    article_id: str
    law_id: LawId
    citation: str
    url: str
    text: str
    score: float
    rerank_score: float | None = None
    control_notes: list[str] = Field(
        default_factory=list,
        description="STF constitutional-review notes of the article, shown to the model too.",
    )


class Citation(BaseModel):
    """A ``[n]`` marker found in the answer, resolved to its source."""

    index: int = Field(ge=1)
    article_id: str
    citation: str
    url: str


class Grounding(BaseModel):
    """Result of the self-check that compares the answer against its sources."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Token usage summed over every model call made for one answer."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    models: list[str] = Field(default_factory=list, description="Models that were called.")


class ChatResponse(BaseModel):
    """Final result for one question."""

    thread_id: str
    answer: str
    route: Route
    sources: list[Source] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    grounding: Grounding | None = None
    rewrites: int = 0
    regenerations: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0


class StatusEvent(BaseModel):
    """Progress of the agent through its steps."""

    type: Literal["status"] = "status"
    stage: Stage
    message: str
    detail: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)


class SourcesEvent(BaseModel):
    """The passages selected as context for the answer."""

    type: Literal["sources"] = "sources"
    sources: list[Source]


class TokenEvent(BaseModel):
    """A piece of the answer text, streamed as it is generated."""

    type: Literal["token"] = "token"
    text: str


class ResetEvent(BaseModel):
    """The answer streamed so far was discarded and will be generated again."""

    type: Literal["reset"] = "reset"
    reason: str


class FinalEvent(BaseModel):
    """The complete response, sent once at the end of the stream."""

    type: Literal["final"] = "final"
    response: ChatResponse


class ErrorEvent(BaseModel):
    """A failure that ended the stream; the message is safe to show to users."""

    type: Literal["error"] = "error"
    message: str


ChatEvent = Annotated[
    StatusEvent | SourcesEvent | TokenEvent | ResetEvent | FinalEvent | ErrorEvent,
    Field(discriminator="type"),
]


class HistoryMessage(BaseModel):
    """One message of a stored conversation thread."""

    role: Literal["user", "assistant"]
    content: str


class LawSummary(BaseModel):
    """A law in the corpus, with article counts."""

    law_id: LawId
    short_name: str
    name: str
    official_name: str
    urn: str
    source_url: str
    articles: int
    in_force: int
    revoked: int
    vetoed: int
    fetched_at: str | None = None


class ServiceInfo(BaseModel):
    """Configuration and index status reported by ``GET /v1/info``."""

    version: str
    collection: str
    indexed_chunks: int
    embedding_provider: str
    embedding_model: str
    retrieval_mode: str
    rerank: bool
    rerank_available: bool = Field(
        description="Whether the reranker model is installed, so per-request reranking works."
    )
    llm_model: str
    llm_fallback_model: str | None
    fast_llm_model: str
    models_configured: bool = Field(description="Whether every model role has a provider API key.")
    max_question_chars: int
