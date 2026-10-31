"""The compose stack must be reachable from this machine only, and not from other web pages.

Neither the API (without ``LEXGRAPH_API_KEY``) nor Qdrant asks for a key, so their ports stay
on the loopback interface, and Qdrant must not allow cross-origin requests from a browser.
"""

from tests.unit.devops.repo_files import compose_environment, compose_published_ports


def test_published_ports_are_bound_to_localhost() -> None:
    ports = compose_published_ports()

    # CI reads the Qdrant aliases from the host, on this address.
    assert "127.0.0.1:6333:6333" in ports
    assert [port for port in ports if not port.startswith("127.0.0.1:")] == []


def test_qdrant_sends_no_cors_headers() -> None:
    assert compose_environment("qdrant")["QDRANT__SERVICE__ENABLE_CORS"] == '"false"'
