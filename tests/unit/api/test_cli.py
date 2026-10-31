"""The ``serve`` and ``ui`` commands, with uvicorn and the subprocess replaced."""

import subprocess
import sys
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from lexgraph.api import cli
from lexgraph.config import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Keep the commands away from the developer's ``.env`` and logging setup."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "bootstrap_env", lambda: None)
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)


def test_serve_runs_one_uvicorn_worker_with_the_app_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--port", "9000"])  # noqa: S104

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            ("lexgraph.api.app:create_app",),
            {
                "factory": True,
                "host": "0.0.0.0",  # noqa: S104
                "port": 9000,
                "workers": 1,
                "log_config": None,
                "log_level": "info",
            },
        )
    ]


def test_serve_binds_to_localhost_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    assert (calls[0]["host"], calls[0]["port"]) == ("127.0.0.1", 8000)


def test_ui_runs_streamlit_on_the_packaged_app(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, returncode=3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(cli.app, ["ui", "--port", "8600", "--server.headless", "true"])

    assert result.exit_code == 3
    assert commands == [
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(cli.UI_SCRIPT),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8600",
            "--server.headless",
            "true",
        ]
    ]
    assert cli.UI_SCRIPT.is_file()


def test_ui_explains_how_to_install_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", None)

    result = runner.invoke(cli.app, ["ui"])

    assert result.exit_code == 1
    assert "uv sync --extra ui" in result.output
