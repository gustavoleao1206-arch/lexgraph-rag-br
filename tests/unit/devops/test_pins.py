"""Code fetched from elsewhere is pinned to an immutable reference, which a tag is not.

pre-commit hooks run next to the developer's ``.env`` and workflow actions next to the CI
secrets, so both are pinned to a commit, with the release in a comment, and images to a
digest. Dependabot moves the action and image pins; ``pre-commit autoupdate --freeze`` the
hook pins.
"""

import re

from tests.conftest import REPO_ROOT
from tests.unit.devops.repo_files import dockerfile_instructions, read_repo_file

_COMMIT = "[0-9a-f]{40}"
_LOCAL_IMAGE = "lexgraph:local"
"""The image compose builds from the Dockerfile."""


def test_pre_commit_hooks_are_frozen_to_a_commit() -> None:
    config = read_repo_file(".pre-commit-config.yaml")
    revs = re.findall(r"^ +rev: (.+)$", config, flags=re.MULTILINE)

    assert revs
    assert [rev for rev in revs if not re.fullmatch(rf"{_COMMIT}  # frozen: v\S+", rev)] == []


def test_workflow_actions_are_pinned_to_a_commit() -> None:
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    actions = [
        action
        for workflow in workflows
        for action in re.findall(
            r"^ +(?:- )?uses: (.+)$", workflow.read_text(encoding="utf-8"), flags=re.MULTILINE
        )
    ]

    assert actions
    assert [
        action for action in actions if not re.fullmatch(rf"\S+@{_COMMIT} # v\S+", action)
    ] == []


def test_external_images_are_pinned_by_digest() -> None:
    stages: set[str] = set()
    images: list[str] = []
    for words in dockerfile_instructions():
        if words[0] != "FROM":
            continue
        # FROM <image> AS <stage>: a later FROM may name an earlier stage instead of an image.
        if words[1] not in stages:
            images.append(words[1])
        stages.add(words[-1])
    compose = re.findall(r"^ +image: (\S+)$", read_repo_file("compose.yaml"), flags=re.MULTILINE)
    images += [image for image in compose if image != _LOCAL_IMAGE]

    assert sorted(image.split(":")[0] for image in images) == [
        "ghcr.io/astral-sh/uv",
        "python",
        "qdrant/qdrant",
    ]
    assert [image for image in images if not re.search("@sha256:[0-9a-f]{64}$", image)] == []
