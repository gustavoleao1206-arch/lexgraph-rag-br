"""Tests for ``scripts/download_models.py``: fake loaders stand in for FastEmbed (no download)."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from lexgraph.config import Settings

KINDS = ("dense", "sparse", "reranker")


class FakeFastEmbed:
    """Fake FastEmbed model classes that record every model they are asked to fetch."""

    def __init__(self, failing_kind: str | None = None) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []
        self.failing_kind = failing_kind

    def loaders(self) -> dict[str, Callable[..., object]]:
        """One loader per model kind, as returned by ``fastembed_loaders()``."""
        return {kind: self._loader(kind) for kind in KINDS}

    def kinds(self) -> list[str]:
        """Model kinds requested so far, in call order."""
        return [kind for kind, *_ in self.calls]

    def _loader(self, kind: str) -> Callable[..., object]:
        def load(model_name: str, *, cache_dir: str, lazy_load: bool) -> object:
            self.calls.append((kind, model_name, cache_dir, lazy_load))
            if kind == self.failing_kind:
                msg = f"Could not load model {model_name} from any source."
                raise ValueError(msg)
            return object()

        return load


@pytest.fixture
def model_settings(clean_env: Path) -> Settings:
    """Settings with recognizable model names and no influence from the environment."""
    return Settings(
        embedding_model="dense-model", sparse_model="sparse-model", reranker_model="reranker-model"
    )


def test_plan_lists_the_dense_then_the_sparse_model(
    download_models_script: ModuleType, model_settings: Settings
) -> None:
    script = download_models_script

    specs = script.plan_downloads(model_settings, include_reranker=False)

    assert specs == [
        script.ModelSpec("dense", "dense-model"),
        script.ModelSpec("sparse", "sparse-model"),
    ]


def test_plan_adds_the_reranker_on_request(
    download_models_script: ModuleType, model_settings: Settings
) -> None:
    specs = download_models_script.plan_downloads(model_settings, include_reranker=True)

    assert [spec.kind for spec in specs] == ["dense", "sparse", "reranker"]
    assert specs[-1].name == "reranker-model"


@pytest.mark.usefixtures("clean_env")
def test_plan_skips_the_dense_model_when_google_computes_embeddings(
    download_models_script: ModuleType,
) -> None:
    settings = Settings(embedding_provider="google")

    specs = download_models_script.plan_downloads(settings, include_reranker=False)

    assert [spec.kind for spec in specs] == ["sparse"]


def test_download_creates_the_cache_folder_and_loads_lazily(
    download_models_script: ModuleType, model_settings: Settings, tmp_path: Path
) -> None:
    script = download_models_script
    fake = FakeFastEmbed()
    cache_dir = tmp_path / "nested" / "models"
    specs = script.plan_downloads(model_settings, include_reranker=True)

    script.download_models(specs, cache_dir, fake.loaders())

    assert cache_dir.is_dir()
    assert fake.calls == [
        ("dense", "dense-model", str(cache_dir), True),
        ("sparse", "sparse-model", str(cache_dir), True),
        ("reranker", "reranker-model", str(cache_dir), True),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        ("Yes", True),
        ("1", True),
        (" on ", True),
        ("false", False),
        ("NO", False),
        ("0", False),
        ("off", False),
    ],
)
def test_parse_bool_accepts_common_spellings(
    download_models_script: ModuleType, text: str, expected: bool
) -> None:
    assert download_models_script.parse_bool(text) is expected


@pytest.mark.parametrize("text", ["", "maybe", "2", "truee"])
def test_parse_bool_rejects_anything_else(download_models_script: ModuleType, text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected true or false"):
        download_models_script.parse_bool(text)


def test_main_downloads_into_the_configured_models_dir(
    download_models_script: ModuleType,
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models_dir = clean_env / "models-from-env"
    monkeypatch.setenv("LEXGRAPH_MODELS_DIR", str(models_dir))
    fake = FakeFastEmbed()

    exit_code = download_models_script.main([], loaders=fake.loaders())

    assert exit_code == 0
    assert fake.kinds() == ["dense", "sparse"]
    assert {cache_dir for _, _, cache_dir, _ in fake.calls} == {str(models_dir)}
    assert f"Downloading 2 model(s) into {models_dir}" in capsys.readouterr().out


def test_main_cache_dir_option_overrides_the_setting(
    download_models_script: ModuleType, clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEXGRAPH_MODELS_DIR", str(clean_env / "ignored"))
    target = clean_env / "target"
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(["--cache-dir", str(target)], loaders=fake.loaders())

    assert exit_code == 0
    assert {cache_dir for _, _, cache_dir, _ in fake.calls} == {str(target)}
    assert not (clean_env / "ignored").exists()


@pytest.mark.parametrize(
    ("argv", "rerank_setting", "expected_kinds"),
    [
        pytest.param([], None, ["dense", "sparse"], id="default-off"),
        pytest.param([], "true", ["dense", "sparse", "reranker"], id="follows-LEXGRAPH_RERANK"),
        pytest.param(["--include-reranker"], None, ["dense", "sparse", "reranker"], id="flag"),
        pytest.param(
            ["--include-reranker", "yes"], None, ["dense", "sparse", "reranker"], id="flag-value"
        ),
        pytest.param(
            ["--include-reranker=false"], "true", ["dense", "sparse"], id="explicit-false-wins"
        ),
    ],
)
def test_main_reranker_selection(
    download_models_script: ModuleType,
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    rerank_setting: str | None,
    expected_kinds: list[str],
) -> None:
    if rerank_setting is not None:
        monkeypatch.setenv("LEXGRAPH_RERANK", rerank_setting)
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        [*argv, "--cache-dir", str(clean_env / "models")], loaders=fake.loaders()
    )

    assert exit_code == 0
    assert fake.kinds() == expected_kinds


def test_main_skips_the_dense_model_for_google_embeddings(
    download_models_script: ModuleType, clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEXGRAPH_EMBEDDING_PROVIDER", "google")
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--cache-dir", str(clean_env / "models")], loaders=fake.loaders()
    )

    assert exit_code == 0
    assert fake.kinds() == ["sparse"]


def test_main_stops_and_reports_a_failed_download(
    download_models_script: ModuleType, clean_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFastEmbed(failing_kind="dense")

    exit_code = download_models_script.main(
        ["--cache-dir", str(clean_env / "models")], loaders=fake.loaders()
    )

    assert exit_code == 1
    assert fake.kinds() == ["dense"]
    assert "error: Could not load model" in capsys.readouterr().err


def test_main_reports_an_unusable_cache_dir(
    download_models_script: ModuleType, clean_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_folder = clean_env / "models"
    not_a_folder.write_text("occupied", encoding="utf-8")
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--cache-dir", str(not_a_folder)], loaders=fake.loaders()
    )

    assert exit_code == 1
    assert fake.calls == []
    assert capsys.readouterr().err.startswith("error: ")


def test_main_rejects_an_invalid_boolean(
    download_models_script: ModuleType, clean_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFastEmbed()

    with pytest.raises(SystemExit) as exit_info:
        download_models_script.main(["--include-reranker=maybe"], loaders=fake.loaders())

    assert exit_info.value.code == 2
    assert "expected true or false, got 'maybe'" in capsys.readouterr().err
    assert fake.calls == []


def test_fastembed_loaders_map_each_kind_to_its_fastembed_class(
    download_models_script: ModuleType,
) -> None:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    assert download_models_script.fastembed_loaders() == {
        "dense": TextEmbedding,
        "sparse": SparseTextEmbedding,
        "reranker": TextCrossEncoder,
    }


def test_empty_variables_keep_the_default_models(
    download_models_script: ModuleType, clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = Settings()
    # Docker passes the model build arguments that compose leaves blank as empty strings.
    for variable in ("LEXGRAPH_EMBEDDING_PROVIDER", "LEXGRAPH_EMBEDDING_MODEL", "LEXGRAPH_RERANK"):
        monkeypatch.setenv(variable, "")
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--cache-dir", str(clean_env / "models")], loaders=fake.loaders()
    )

    assert exit_code == 0
    assert [(kind, name) for kind, name, _, _ in fake.calls] == [
        ("dense", defaults.embedding_model),
        ("sparse", defaults.sparse_model),
    ]


def test_main_reports_invalid_settings(
    download_models_script: ModuleType,
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LEXGRAPH_EMBEDDING_PROVIDER", "elsewhere")
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--cache-dir", str(clean_env / "models")], loaders=fake.loaders()
    )

    assert exit_code == 1
    assert fake.calls == []
    assert "embedding_provider" in capsys.readouterr().err


def test_plan_file_round_trip(download_models_script: ModuleType, tmp_path: Path) -> None:
    script = download_models_script
    specs = [
        script.ModelSpec("dense", "org/dense model"),
        script.ModelSpec("sparse", "Qdrant/bm25"),
        script.ModelSpec("reranker", "org/reranker"),
    ]
    path = tmp_path / "models.txt"

    script.write_plan(specs, path)

    assert path.read_text(encoding="utf-8").splitlines()[0] == "dense org/dense model"
    assert script.read_plan(path) == specs


@pytest.mark.parametrize(
    "line", ["dense", "dense ", "colbert some/model", "Dense some/model", " some/model"]
)
def test_read_plan_rejects_malformed_lines(
    download_models_script: ModuleType, tmp_path: Path, line: str
) -> None:
    path = tmp_path / "models.txt"
    path.write_text(f"sparse Qdrant/bm25\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2: expected '<kind> <model name>'"):
        download_models_script.read_plan(path)


def test_save_plan_records_the_selection_without_downloading(
    download_models_script: ModuleType,
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = download_models_script
    monkeypatch.setenv("LEXGRAPH_EMBEDDING_MODEL", "dense-from-env")
    plan = clean_env / "models.txt"
    fake = FakeFastEmbed()

    exit_code = script.main(
        ["--include-reranker", "--save-plan", str(plan)], loaders=fake.loaders()
    )

    assert exit_code == 0
    assert fake.calls == []
    assert [spec.kind for spec in script.read_plan(plan)] == ["dense", "sparse", "reranker"]
    assert script.read_plan(plan)[0].name == "dense-from-env"
    assert f"Saved a plan of 3 model(s) to {plan}" in capsys.readouterr().out


def test_plan_downloads_the_listed_models_without_lexgraph(
    download_models_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Docker stage that downloads the models has FastEmbed but not the lexgraph package.
    monkeypatch.setitem(sys.modules, "lexgraph", None)
    monkeypatch.setitem(sys.modules, "lexgraph.config", None)
    plan = tmp_path / "models.txt"
    plan.write_text("sparse sparse-model\nreranker reranker-model\n", encoding="utf-8")
    cache_dir = tmp_path / "models"
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--plan", str(plan), "--cache-dir", str(cache_dir)], loaders=fake.loaders()
    )

    assert exit_code == 0
    assert fake.calls == [
        ("sparse", "sparse-model", str(cache_dir), True),
        ("reranker", "reranker-model", str(cache_dir), True),
    ]


def test_main_reports_a_missing_plan(
    download_models_script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFastEmbed()

    exit_code = download_models_script.main(
        ["--plan", str(tmp_path / "missing.txt"), "--cache-dir", str(tmp_path)],
        loaders=fake.loaders(),
    )

    assert exit_code == 1
    assert fake.calls == []
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--plan", "models.txt"], id="plan-without-cache-dir"),
        pytest.param(
            ["--plan", "models.txt", "--cache-dir", "models", "--include-reranker"],
            id="plan-with-include-reranker",
        ),
        pytest.param(["--plan", "a.txt", "--save-plan", "b.txt"], id="plan-and-save-plan"),
    ],
)
def test_main_rejects_conflicting_plan_options(
    download_models_script: ModuleType, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFastEmbed()

    with pytest.raises(SystemExit) as exit_info:
        download_models_script.main(argv, loaders=fake.loaders())

    assert exit_info.value.code == 2
    assert "--plan" in capsys.readouterr().err
    assert fake.calls == []
