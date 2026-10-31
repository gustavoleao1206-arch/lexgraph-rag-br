"""Fixtures for the tests of the deployment files: scripts, Docker build and ``.env`` template."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import REPO_ROOT


@pytest.fixture(scope="session")
def download_models_script() -> ModuleType:
    """``scripts/download_models.py`` imported as a module (``scripts`` is not a package)."""
    name = "download_models"
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot import {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Work in an empty folder without ``LEXGRAPH_*`` variables, so settings take defaults.

    The folder has no ``.env`` file, so neither :class:`~lexgraph.config.Settings` nor
    :func:`~lexgraph.config.bootstrap_env` can pick up the developer's local configuration.
    """
    for key in list(os.environ):
        if key.upper().startswith("LEXGRAPH_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    return tmp_path
