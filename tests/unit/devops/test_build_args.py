"""The Docker image must hold the models that the configuration selects.

compose forwards the model settings of ``.env`` as build arguments, the Dockerfile declares
them in the stage that plans the download, and the download script reads them as settings.
A name lost on the way would bake the default model into the image, and the containers,
which run offline, would fail on the first search.
"""

from types import ModuleType
from typing import cast

from lexgraph.config import Settings
from tests.unit.devops.repo_files import compose_build_args, dockerfile_stage_args

SETTING_VARIABLES = {f"LEXGRAPH_{name.upper()}" for name in Settings.model_fields}


class AttributeRecorder:
    """Stands in for an object and records the names of the attributes read from it."""

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped
        self.names: set[str] = set()

    def __getattr__(self, name: str) -> object:
        self.names.add(name)
        return getattr(self._wrapped, name)


def test_compose_forwards_every_setting_that_selects_a_model(
    download_models_script: ModuleType,
) -> None:
    recorder = AttributeRecorder(Settings.model_construct())
    download_models_script.plan_downloads(cast("Settings", recorder), include_reranker=True)

    needed = {f"LEXGRAPH_{name.upper()}" for name in recorder.names}

    assert "LEXGRAPH_EMBEDDING_MODEL" in needed
    assert sorted(needed - set(compose_build_args())) == []


def test_compose_forwards_only_existing_settings() -> None:
    forwarded = {name for name in compose_build_args() if name.startswith("LEXGRAPH_")}

    assert forwarded
    assert sorted(forwarded - SETTING_VARIABLES) == []


def test_the_planning_stage_declares_every_build_argument() -> None:
    # Docker ignores, with a mere warning, a build argument that the stage does not declare.
    build_args = compose_build_args()

    assert "INCLUDE_RERANKER" in build_args
    assert sorted(set(build_args) - dockerfile_stage_args()["model-plan"]) == []
