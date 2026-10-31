"""Versions that several files pin must agree, although Dependabot updates each file alone.

A failure here typically shows up on a Dependabot pull request: the assertion names the
file to update in the same pull request.
"""

import re
import tomllib

from tests.unit.devops.repo_files import read_repo_file

_HOOK_ID = re.compile(r"^ +- id: (\S+)", re.MULTILINE)


def search(pattern: str, file_name: str) -> re.Match[str]:
    """First match of ``pattern`` (multiline) in a repository file."""
    match = re.search(pattern, read_repo_file(file_name), flags=re.MULTILINE)
    assert match is not None, f"{file_name} has no match for {pattern!r}"
    return match


def locked_version(package: str) -> str:
    """Version of ``package`` in uv.lock."""
    lock = tomllib.loads(read_repo_file("uv.lock"))
    version: str = next(entry["version"] for entry in lock["package"] if entry["name"] == package)
    return version


def test_ci_installs_the_uv_version_of_the_dockerfile() -> None:
    image_uv = search(r"^FROM ghcr\.io/astral-sh/uv:([\w.]+)[@\s]", "Dockerfile")[1]
    ci_uv = search(r'^\s+UV_VERSION: "([\w.]+)"$', ".github/workflows/ci.yml")[1]

    assert ci_uv == image_uv, "update UV_VERSION in .github/workflows/ci.yml"


def test_the_image_uses_the_python_version_of_the_project() -> None:
    image_python = search(r"^FROM python:(\d+\.\d+)-", "Dockerfile")[1]

    assert image_python == read_repo_file(".python-version").strip()


def test_the_locked_qdrant_client_supports_the_qdrant_server() -> None:
    # qdrant-client warns about servers of another major version or more than one minor away.
    server = search(r"image: qdrant/qdrant:v(\d+)\.(\d+)\.\d+", "compose.yaml")
    client_major, client_minor, *_ = locked_version("qdrant-client").split(".")

    assert server[1] == client_major, "update qdrant-client in uv.lock"
    assert abs(int(server[2]) - int(client_minor)) <= 1, "update qdrant-client in uv.lock"


def test_pre_commit_runs_the_locked_ruff_and_mypy() -> None:
    # A hook from a remote repository would pin a second version next to the one in uv.lock.
    config = read_repo_file(".pre-commit-config.yaml")
    repos = re.split(r"^  - repo: ", config, flags=re.MULTILINE)[1:]
    hooks = {repo.split()[0]: set(_HOOK_ID.findall(repo)) for repo in repos}

    assert {"ruff-check", "ruff-format", "mypy"} <= hooks["local"]
    assert not any("ruff" in url or "mypy" in url for url in hooks.keys() - {"local"})
