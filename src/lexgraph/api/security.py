"""Request guards: optional API-key protection for the ``/v1`` endpoints and a body size limit."""

import secrets
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import SecretStr
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

API_KEY_HEADER = "X-API-Key"

MAX_BODY_BYTES = 128 * 1024
"""Largest request body accepted, in bytes.

A chat request with the longest question allowed fits even when every character is escaped
as a ``\\u`` surrogate pair (12 bytes), so only oversized bodies are turned away.
"""

INVALID_KEY_DETAIL = "Missing or invalid API key."
BODY_TOO_LARGE_DETAIL = "Request body too large."

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="ApiKey",
    description="Required when the server sets `LEXGRAPH_API_KEY`.",
    auto_error=False,
)


class ApiKeyGuard:
    """Dependency that rejects requests without the configured key in ``X-API-Key``.

    The comparison runs in constant time so response timing does not leak the key.
    """

    def __init__(self, api_key: SecretStr) -> None:
        self._expected = api_key.get_secret_value().encode()

    def __call__(self, provided: Annotated[str | None, Security(api_key_header)]) -> None:
        if not _key_matches(provided, self._expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=INVALID_KEY_DETAIL,
                headers={"WWW-Authenticate": "APIKey"},
            )


def api_key_dependencies(api_key: SecretStr | None) -> list[Depends]:
    """Router dependencies that enforce ``api_key``.

    An unset or blank key (``LEXGRAPH_API_KEY=`` in ``.env``) leaves the API open.
    """
    if api_key is None or not api_key.get_secret_value():
        return []
    return [Security(ApiKeyGuard(api_key))]


class RequestGuardMiddleware:
    """ASGI middleware that screens HTTP requests before any route reads their body.

    With an API key configured, a request under ``protected_prefix`` without it gets 401
    right away, so unauthenticated clients cannot make the server buffer and parse a body.
    A body over ``max_body_bytes`` gets 413: at once when ``Content-Length`` announces it,
    otherwise (chunked bodies) as soon as the bytes read pass the limit. The router keeps
    :func:`api_key_dependencies`, which documents the key in OpenAPI.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: SecretStr | None,
        protected_prefix: str,
        max_body_bytes: int = MAX_BODY_BYTES,
    ) -> None:
        self.app = app
        self._expected = _expected_key(api_key)
        self._protected_prefix = protected_prefix
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if self._lacks_key(_route_path(scope), headers):
            response = JSONResponse(
                {"detail": INVALID_KEY_DETAIL},
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "APIKey"},
            )
        elif _declared_length(headers) > self._max_body_bytes:
            response = JSONResponse(
                {"detail": BODY_TOO_LARGE_DETAIL}, status_code=status.HTTP_413_CONTENT_TOO_LARGE
            )
        else:
            await self.app(scope, self._capped(receive), send)
            return
        await response(scope, receive, send)

    def _lacks_key(self, path: str, headers: Headers) -> bool:
        """Whether ``path`` needs the API key and the request does not carry it."""
        prefix = self._protected_prefix
        if self._expected is None or not (path == prefix or path.startswith(f"{prefix}/")):
            return False
        return not _key_matches(headers.get(API_KEY_HEADER), self._expected)

    def _capped(self, receive: Receive) -> Receive:
        """``receive`` that raises a 413 error once the body read so far passes the limit."""
        received = 0

        async def capped_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body_bytes:
                    # FastAPI re-raises HTTP errors met while it reads the body, so the client
                    # gets this 413 instead of a generic 400.
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=BODY_TOO_LARGE_DETAIL,
                    )
            return message

        return capped_receive


def _expected_key(api_key: SecretStr | None) -> bytes | None:
    """The key requests must present, or ``None`` when it is unset or blank."""
    if api_key is None or not api_key.get_secret_value():
        return None
    return api_key.get_secret_value().encode()


def _route_path(scope: Scope) -> str:
    """The path the router matches: behind ``--root-path`` the server puts it before the path."""
    path: str = scope["path"]
    root_path: str = scope.get("root_path", "")
    if root_path and path.startswith(f"{root_path}/"):
        return path.removeprefix(root_path)
    return path


def _key_matches(provided: str | None, expected: bytes) -> bool:
    return provided is not None and secrets.compare_digest(provided.encode(), expected)


def _declared_length(headers: Headers) -> int:
    """Body size announced by ``Content-Length``; 0 without one (the server rejects bad values)."""
    try:
        return int(headers.get("content-length", "0"))
    except ValueError:
        return 0
