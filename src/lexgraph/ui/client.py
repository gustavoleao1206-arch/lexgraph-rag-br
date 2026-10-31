"""HTTP client for the LexGraph API, used by the Streamlit interface.

It is synchronous on purpose: Streamlit runs each script in a worker thread without an event
loop, and a blocking iterator plugs straight into its rerun model.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter, ValidationError

from lexgraph.domain import Article
from lexgraph.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    LawSummary,
    ServiceInfo,
)

API_KEY_HEADER = "X-API-Key"
"""Header checked by the API when ``LEXGRAPH_API_KEY`` is set (see ``lexgraph.api.security``)."""

DONE_EVENT = "done"

_CONNECT_TIMEOUT = 5.0
_LOOKUP_TIMEOUT = httpx.Timeout(10.0, connect=_CONNECT_TIMEOUT)

_INFO = TypeAdapter(ServiceInfo)
_LAWS = TypeAdapter(list[LawSummary])
_ARTICLE = TypeAdapter(Article)
_HISTORY = TypeAdapter(list[HistoryMessage])
_RESPONSE = TypeAdapter(ChatResponse)
_EVENT: TypeAdapter[ChatEvent] = TypeAdapter(ChatEvent)


class ApiError(Exception):
    """The API answered with an error status or broke the response protocol."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiConnectionError(ApiError):
    """The API could not be reached, timed out or dropped the connection."""


@dataclass(frozen=True)
class SseMessage:
    """One event dispatched from a ``text/event-stream`` body."""

    event: str
    data: str


def parse_sse(lines: Iterable[str]) -> Iterator[SseMessage]:
    """Parse Server-Sent Events following the WHATWG rules.

    Comment lines (such as the ``: ping`` keep-alive) are skipped, consecutive ``data`` lines
    are joined with newlines, an event is dispatched at each blank line, events without data
    are ignored and an unterminated event at the end of the stream is dropped.
    """
    event = ""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield SseMessage(event=event or "message", data="\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)


class ApiClient:
    """Typed access to the LexGraph HTTP API.

    Every method raises :class:`ApiError` (or :class:`ApiConnectionError`) on failure, except
    :meth:`health`, which reports failures as ``False``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._answer_timeout = httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT)
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self._answer_timeout,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def health(self) -> bool:
        """Whether the API answers its health check."""
        try:
            response = self._http.get("/health", timeout=_LOOKUP_TIMEOUT)
        except httpx.HTTPError:
            return False
        return response.is_success

    def info(self) -> ServiceInfo:
        """Models, retrieval configuration and index size reported by the API."""
        return _decode(_INFO, self._request("GET", "/v1/info").content)

    def laws(self) -> list[LawSummary]:
        """Laws in the corpus with their article counts."""
        return _decode(_LAWS, self._request("GET", "/v1/laws").content)

    def article(self, article_id: str) -> Article | None:
        """One article by id (e.g. ``"CLT:482"``), or ``None`` when it is not in the corpus."""
        try:
            response = self._request("GET", f"/v1/articles/{quote(article_id, safe=':')}")
        except ApiError as error:
            if error.status_code == httpx.codes.NOT_FOUND:
                return None
            raise
        return _decode(_ARTICLE, response.content)

    def history(self, thread_id: str) -> list[HistoryMessage]:
        """Stored messages of a conversation thread."""
        return _decode(_HISTORY, self._request("GET", f"/v1/threads/{quote(thread_id)}").content)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Ask a question and wait for the complete answer."""
        response = self._request(
            "POST", "/v1/chat", json=_body(request), timeout=self._answer_timeout
        )
        return _decode(_RESPONSE, response.content)

    def stream_chat(self, request: ChatRequest) -> Iterator[ChatEvent]:
        """Ask a question and yield the answer events as they arrive.

        The last event is a ``final`` or an ``error`` event. A stream that ends before the
        API's ``done`` marker raises :class:`ApiConnectionError`.
        """
        try:
            with self._http.stream(
                "POST",
                "/v1/chat/stream",
                json=_body(request),
                headers={"Accept": "text/event-stream"},
            ) as response:
                if not response.is_success:
                    response.read()
                    _raise_for_status(response)
                for message in parse_sse(response.iter_lines()):
                    if message.event == DONE_EVENT:
                        return
                    yield _decode(_EVENT, message.data)
        except httpx.HTTPError as error:
            raise ApiConnectionError(f"Connection to {self.base_url} failed: {error}") from error
        raise ApiConnectionError("The answer stream ended before it was complete.")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: httpx.Timeout = _LOOKUP_TIMEOUT,
    ) -> httpx.Response:
        try:
            response = self._http.request(method, path, json=json, timeout=timeout)
        except httpx.HTTPError as error:
            raise ApiConnectionError(f"Connection to {self.base_url} failed: {error}") from error
        _raise_for_status(response)
        return response


def _body(request: ChatRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude_none=True)


def _decode[T](adapter: TypeAdapter[T], content: bytes | str) -> T:
    try:
        return adapter.validate_json(content)
    except ValidationError as error:
        msg = "The API sent a response that does not match its schema."
        raise ApiError(msg) from error


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    msg = f"The API answered {response.status_code}: {_error_detail(response)}"
    raise ApiError(msg, status_code=response.status_code)


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.reason_phrase
    return detail if isinstance(detail, str) else response.reason_phrase
