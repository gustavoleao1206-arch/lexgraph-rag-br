"""A plain ``pytest`` must leave out the tests that need downloaded models or provider keys.

A throwaway test module is collected in a subprocess under the repository's configuration,
so the check sees ``addopts`` exactly as a contributor's ``pytest`` does.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

MARKED_TESTS = """\
import pytest


@pytest.mark.integration
def test_needs_models():
    pass


@pytest.mark.llm
def test_needs_keys():
    pass


def test_offline():
    pass
"""


def selected_tests(folder: Path, *options: str) -> list[str]:
    """Names of the tests of MARKED_TESTS that pytest selects with ``options``."""
    module = folder / "test_markers.py"
    module.write_text(MARKED_TESTS, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pytest",
        f"--config-file={REPO_ROOT / 'pyproject.toml'}",
        # The configuration lives in another tree: keep collection inside this folder.
        f"--rootdir={folder}",
        f"--confcutdir={folder}",
        "--collect-only",
        "--quiet",
        "-p",
        "no:cacheprovider",
        *options,
        str(module),
    ]
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, errors="replace", check=False, cwd=folder, env=env
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    return sorted(line.rpartition("::")[2] for line in result.stdout.splitlines() if "::" in line)


def test_plain_pytest_runs_only_the_offline_tests(tmp_path: Path) -> None:
    assert selected_tests(tmp_path) == ["test_offline"]


def test_a_marker_expression_on_the_command_line_replaces_the_default(tmp_path: Path) -> None:
    assert selected_tests(tmp_path, "-m", "") == [
        "test_needs_keys",
        "test_needs_models",
        "test_offline",
    ]
