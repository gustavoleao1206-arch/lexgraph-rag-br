"""The ``.dockerignore`` allowlist must admit what the Dockerfile reads, and nothing private.

A file missing from the build context only surfaces as a failed ``docker build`` in CI, so
these checks catch it early. Allowlist entries are plain paths, so prefix matching is exact.
"""

import pytest

from tests.unit.devops.repo_files import context_sources, read_repo_file


def dockerignore_rules() -> list[str]:
    """Non-comment, non-blank lines of ``.dockerignore``, in order."""
    lines = read_repo_file(".dockerignore").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def is_allowlisted(path: str) -> bool:
    """Whether a ``!`` rule re-includes ``path`` (the rules start by excluding everything)."""
    entries = [rule[1:].rstrip("/") for rule in dockerignore_rules() if rule.startswith("!")]
    return any(path == entry or path.startswith(f"{entry}/") for entry in entries)


def test_the_context_starts_from_nothing() -> None:
    assert dockerignore_rules()[0] == "*"


def test_parser_finds_the_dockerfile_sources() -> None:
    expected = {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "src",
        "src/lexgraph/config.py",
        "scripts/download_models.py",
        "data/corpus",
    }
    assert expected <= context_sources()


def test_parser_ignores_mounts_and_copies_from_other_stages() -> None:
    sources = context_sources()

    assert not {"/uv", "/app/dist", "/app/.venv", "/app/data/models"} & sources


def test_every_context_source_is_allowlisted() -> None:
    assert sorted(path for path in context_sources() if not is_allowlisted(path)) == []


@pytest.mark.parametrize(
    "path", [".env", ".venv", ".git", "data/models", "data/qdrant", "data/raw", "tests"]
)
def test_local_state_and_secrets_stay_out_of_the_context(path: str) -> None:
    assert not is_allowlisted(path)
