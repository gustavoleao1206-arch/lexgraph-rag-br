"""``.env.example`` documents every setting with its real default and holds no secret."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import dotenv_values

from lexgraph.config import Settings
from lexgraph.eval.cli import JUDGE_MODEL_ENV

ENV_EXAMPLE = Path(__file__).resolve().parents[3] / ".env.example"
PREFIX = "LEXGRAPH_"
SETTING_VARIABLES = {f"{PREFIX}{name.upper()}": name for name in Settings.model_fields}
UNSET_BY_DEFAULT = {
    variable
    for variable, name in SETTING_VARIABLES.items()
    if Settings.model_fields[name].default is None
}
COMMAND_VARIABLES = {JUDGE_MODEL_ENV}
"""Variables read by a command rather than by :class:`Settings`."""
_ASSIGNMENT = re.compile(r"^(?P<comment>#\s*)?(?P<assignment>[A-Z][A-Z0-9_]*=.*)$")


@dataclass(frozen=True)
class EnvTemplate:
    """Variables of the template: parsed values (commented out or not) and the active ones."""

    values: dict[str, str]
    active: frozenset[str]


@pytest.fixture(scope="module")
def template() -> EnvTemplate:
    """Parse ``.env.example`` with python-dotenv, treating commented-out assignments as set."""
    assignments: list[str] = []
    active: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT.match(line)
        if match is None:
            continue
        assignments.append(match["assignment"])
        if match["comment"] is None:
            active.add(match["assignment"].partition("=")[0])
    parsed = dotenv_values(stream=io.StringIO("\n".join(assignments)))
    return EnvTemplate({key: value or "" for key, value in parsed.items()}, frozenset(active))


def test_every_setting_is_documented(template: EnvTemplate) -> None:
    missing = sorted(set(SETTING_VARIABLES) - set(template.values))
    assert missing == []


def test_no_unknown_setting_is_documented(template: EnvTemplate) -> None:
    documented = {key for key in template.values if key.startswith(PREFIX)}
    assert sorted(documented - set(SETTING_VARIABLES) - COMMAND_VARIABLES) == []


@pytest.mark.usefixtures("clean_env")
def test_documented_values_are_the_built_in_defaults(
    template: EnvTemplate, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = Settings()
    for variable in SETTING_VARIABLES.keys() & template.values.keys():
        monkeypatch.setenv(variable, template.values[variable])

    documented = Settings()

    differing = sorted(
        name
        for variable, name in SETTING_VARIABLES.items()
        if variable not in UNSET_BY_DEFAULT and getattr(documented, name) != getattr(defaults, name)
    )
    assert differing == []


def test_settings_unset_by_default_are_commented_out(template: EnvTemplate) -> None:
    # An empty value would set them to "" instead of leaving them unset.
    assert sorted(UNSET_BY_DEFAULT & template.active) == []


@pytest.mark.parametrize("variable", ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"])
def test_provider_keys_are_documented(template: EnvTemplate, variable: str) -> None:
    assert variable in template.values


def test_secrets_are_left_empty(template: EnvTemplate) -> None:
    filled = sorted(
        key
        for key, value in template.values.items()
        if key.endswith(("_KEY", "_TOKEN", "_SECRET")) and value
    )
    assert filled == []
