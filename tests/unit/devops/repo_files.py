"""Readers for the deployment files the devops tests inspect: Dockerfile and compose.yaml.

They parse only the subset of each format the repository uses, and fail loudly (an empty
result or a missing key) rather than guess when a file no longer looks as expected.
"""

import re

from tests.conftest import REPO_ROOT

_LINE_CONTINUATION = re.compile(r"\\\n")
_COMPOSE_BUILD_ARG = re.compile(r" {6}([A-Z][A-Z0-9_]*):")
_COMPOSE_VARIABLE = re.compile(r" {6}([A-Z][A-Z0-9_]*): (.+)")
_COMPOSE_LIST_ITEM = re.compile(r" {6}- (.+)")


def read_repo_file(name: str) -> str:
    """Text of a repository file, by its path relative to the repository root."""
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def dockerfile_instructions() -> list[list[str]]:
    """Words of each Dockerfile instruction, with continuation lines joined."""
    text = _LINE_CONTINUATION.sub(" ", read_repo_file("Dockerfile"))
    return [line.split() for line in text.splitlines() if line.strip() and line[0] != "#"]


def dockerfile_stage_args() -> dict[str, set[str]]:
    """Names of the ``ARG`` instructions of each stage (``FROM ... AS <stage>``)."""
    args: dict[str, set[str]] = {}
    stage: set[str] | None = None
    for words in dockerfile_instructions():
        if words[0] == "FROM":
            stage = args.setdefault(words[-1], set())
        elif words[0] == "ARG" and stage is not None:
            stage.update(word.partition("=")[0] for word in words[1:])
    return args


def context_sources() -> set[str]:
    """Paths the Dockerfile reads from the build context: local ``COPY`` and bind mounts."""
    sources: set[str] = set()
    for instruction, *words in dockerfile_instructions():
        options = [word for word in words if word.startswith("--")]
        if instruction == "RUN":
            mounts = [_mount(option) for option in options if option.startswith("--mount=")]
            sources.update(
                mount["source"]
                for mount in mounts
                if mount.get("type", "bind") == "bind" and "from" not in mount
            )
        elif instruction == "COPY" and not any(word.startswith("--from=") for word in options):
            paths = [word for word in words if not word.startswith("--")]
            sources.update(path.rstrip("/") for path in paths[:-1])
    return sources


def compose_build_args() -> list[str]:
    """Keys of the ``build.args`` block of compose.yaml, shared by the lexgraph services."""
    lines = read_repo_file("compose.yaml").splitlines()
    names: list[str] = []
    for line in lines[lines.index("    args:") + 1 :]:
        match = _COMPOSE_BUILD_ARG.match(line)
        if match is None:
            break
        names.append(match[1])
    return names


def compose_published_ports() -> list[str]:
    """Items of every ``ports`` list of compose.yaml, unquoted, such as ``127.0.0.1:8000:8000``.

    A long-syntax item comes back as its first line (``target: 8000``), which no check accepts.
    """
    lines = read_repo_file("compose.yaml").splitlines()
    starts = [index + 1 for index, line in enumerate(lines) if line == "    ports:"]
    ports: list[str] = []
    for start in starts:
        for line in lines[start:]:
            match = _COMPOSE_LIST_ITEM.fullmatch(line)
            if match is None:
                break
            ports.append(match[1].strip("\"'"))
    return ports


def compose_environment(service: str) -> dict[str, str]:
    """The ``environment`` block of a compose.yaml service, with values as written."""
    lines = read_repo_file("compose.yaml").splitlines()
    start = lines.index("    environment:", lines.index(f"  {service}:")) + 1
    variables: dict[str, str] = {}
    for line in lines[start:]:
        if line.lstrip().startswith("#"):
            continue
        match = _COMPOSE_VARIABLE.fullmatch(line)
        if match is None:
            break
        variables[match[1]] = match[2]
    return variables


def _mount(option: str) -> dict[str, str]:
    """Options of a ``--mount=key=value,...`` flag of ``RUN``."""
    pairs = (item.partition("=") for item in option.removeprefix("--mount=").split(","))
    return {key: value for key, _, value in pairs}
