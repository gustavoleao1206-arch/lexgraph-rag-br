"""Download the FastEmbed models used by lexgraph into a local cache folder.

The Docker build runs this script so containers start with ``HF_HUB_OFFLINE=1`` and never
download a model at runtime; on a workstation it warms ``data/models`` before the first
``lexgraph ingest``. Model names come from :class:`lexgraph.config.Settings`, so the
``LEXGRAPH_*`` variables select the models here as they do in the application. Empty
variables count as unset: Docker passes build arguments left blank as empty strings.

The Docker build works in two steps. ``--save-plan`` records the models the settings select
in a small text file, and ``--plan`` downloads the models listed in such a file without
reading the settings (nor importing lexgraph). The download step thus depends on the plan,
not on the code, so Docker reruns it only when the selection of models changes.

Usage::

    python scripts/download_models.py                       # dense + BM25 into data/models
    python scripts/download_models.py --include-reranker    # also the cross-encoder
    python scripts/download_models.py --include-reranker=false --save-plan models.txt
    python scripts/download_models.py --plan models.txt --cache-dir /app/data/models
"""

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, get_args

if TYPE_CHECKING:
    from lexgraph.config import Settings

ModelKind = Literal["dense", "sparse", "reranker"]

_KINDS: dict[str, ModelKind] = {kind: kind for kind in get_args(ModelKind)}
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ModelLoader(Protocol):
    """A FastEmbed model class: creating an instance downloads the files into ``cache_dir``."""

    def __call__(self, model_name: str, *, cache_dir: str, lazy_load: bool) -> object: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model to download."""

    kind: ModelKind
    name: str


def load_settings() -> "Settings":
    """Read the settings as the application does, except that empty variables are ignored.

    lexgraph is imported here, so ``--plan`` runs where only FastEmbed is installed.
    """
    from lexgraph.config import Settings, bootstrap_env

    bootstrap_env()
    return Settings(_env_ignore_empty=True)


def plan_downloads(settings: "Settings", *, include_reranker: bool) -> list[ModelSpec]:
    """List the models the application loads with ``settings``.

    The dense model is skipped when embeddings come from the Google API, and the reranker
    is only listed on request because it is used only when reranking is enabled.
    """
    specs: list[ModelSpec] = []
    if settings.embedding_provider == "fastembed":
        specs.append(ModelSpec("dense", settings.embedding_model))
    specs.append(ModelSpec("sparse", settings.sparse_model))
    if include_reranker:
        specs.append(ModelSpec("reranker", settings.reranker_model))
    return specs


def write_plan(specs: Sequence[ModelSpec], path: Path) -> None:
    """Record ``specs`` in ``path``, one ``<kind> <model name>`` line per model."""
    path.write_text("".join(f"{spec.kind} {spec.name}\n" for spec in specs), encoding="utf-8")


def read_plan(path: Path) -> list[ModelSpec]:
    """Read the models recorded by :func:`write_plan`.

    Raises:
        ValueError: a line is not ``<kind> <model name>`` with a known kind.
    """
    specs: list[ModelSpec] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        word, _, name = line.strip().partition(" ")
        kind = _KINDS.get(word)
        if kind is None or not name.strip():
            msg = f"{path}, line {number}: expected '<kind> <model name>', got {line!r}"
            raise ValueError(msg)
        specs.append(ModelSpec(kind, name.strip()))
    return specs


def fastembed_loaders() -> dict[ModelKind, ModelLoader]:
    """Map each model kind to its FastEmbed class; imported here because the import is slow."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return {"dense": TextEmbedding, "sparse": SparseTextEmbedding, "reranker": TextCrossEncoder}


def download_models(
    specs: Sequence[ModelSpec], cache_dir: Path, loaders: Mapping[ModelKind, ModelLoader]
) -> None:
    """Fetch every model in ``specs`` into ``cache_dir``.

    ``lazy_load=True`` makes FastEmbed download (or find in the cache) the model files
    without creating an inference session, so the step needs little memory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        print(f"[{spec.kind}] {spec.name}", flush=True)
        started = time.perf_counter()
        loaders[spec.kind](spec.name, cache_dir=str(cache_dir), lazy_load=True)
        print(f"[{spec.kind}] ready in {time.perf_counter() - started:.1f} s", flush=True)


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean such as ``true``, ``0`` or ``yes``."""
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    msg = f"expected true or false, got {value!r}"
    raise argparse.ArgumentTypeError(msg)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line (``sys.argv[1:]`` when ``argv`` is None)."""
    parser = argparse.ArgumentParser(
        description="Download the FastEmbed models used by lexgraph into a cache folder."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="destination folder (default: LEXGRAPH_MODELS_DIR, i.e. data/models)",
    )
    parser.add_argument(
        "--include-reranker",
        nargs="?",
        const=True,
        default=None,
        type=parse_bool,
        metavar="BOOL",
        help="also download the cross-encoder reranker (default: LEXGRAPH_RERANK)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--save-plan",
        type=Path,
        metavar="FILE",
        help="only record the models the settings select in FILE, without downloading them",
    )
    source.add_argument(
        "--plan",
        type=Path,
        metavar="FILE",
        help="download the models listed in FILE instead of reading the settings",
    )
    args = parser.parse_args(argv)
    if args.plan is not None and (args.cache_dir is None or args.include_reranker is not None):
        parser.error("--plan requires --cache-dir and excludes --include-reranker")
    return args


def main(
    argv: Sequence[str] | None = None, *, loaders: Mapping[ModelKind, ModelLoader] | None = None
) -> int:
    """Run the script and return its exit code; ``loaders`` replaces FastEmbed in tests."""
    args = parse_args(argv)
    cache_dir: Path
    try:
        if args.plan is not None:
            specs, cache_dir = read_plan(args.plan), args.cache_dir
        else:
            settings = load_settings()
            include_reranker: bool = (
                settings.rerank if args.include_reranker is None else args.include_reranker
            )
            specs = plan_downloads(settings, include_reranker=include_reranker)
            cache_dir = args.cache_dir or settings.models_dir
        if args.save_plan is not None:
            write_plan(specs, args.save_plan)
            print(f"Saved a plan of {len(specs)} model(s) to {args.save_plan}", flush=True)
            return 0
        print(f"Downloading {len(specs)} model(s) into {cache_dir}", flush=True)
        download_models(specs, cache_dir, loaders or fastembed_loaders())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
