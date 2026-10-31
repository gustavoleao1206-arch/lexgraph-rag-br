"""Application settings, read from environment variables and an optional ``.env`` file.

Every setting can be overridden with an environment variable prefixed with ``LEXGRAPH_``
(for example ``LEXGRAPH_RETRIEVAL_MODE=dense``). Provider credentials keep their standard
names (``GROQ_API_KEY``, ``GOOGLE_API_KEY``, ``OPENAI_API_KEY``) because the provider SDKs
read them directly from the process environment.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "sparse", "hybrid"]
EmbeddingProvider = Literal["fastembed", "google"]
ReasoningEffort = Literal["low", "medium", "high"]


def bootstrap_env(env_file: str | Path = ".env") -> None:
    """Load ``.env`` into ``os.environ`` without overriding variables that are already set.

    pydantic-settings reads ``.env`` for its own fields only; provider SDKs (Groq, Gemini,
    LangSmith) look at ``os.environ``, so entry points call this before building clients.
    """
    load_dotenv(env_file, override=False)


class Settings(BaseSettings):
    """Runtime configuration for ingestion, retrieval, the agent, the API and the UI."""

    model_config = SettingsConfigDict(
        env_prefix="LEXGRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -------------------------------------------------------------------------
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    checkpoint_path: Path = Path("data/checkpoints/threads.sqlite")

    # --- vector store --------------------------------------------------------------------
    qdrant_url: str | None = Field(
        default=None, description="Qdrant server URL. When unset, Qdrant runs embedded."
    )
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Field(
        default=Path("data/qdrant"), description="Storage folder for embedded Qdrant."
    )
    collection: str = Field(default="lexgraph", description="Collection alias used for search.")

    # --- embeddings and reranking --------------------------------------------------------
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_dim: int = 768
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3-int8"

    # --- retrieval -----------------------------------------------------------------------
    retrieval_mode: RetrievalMode = "hybrid"
    prefetch_k: int = Field(default=30, ge=1, le=200, description="Candidates per hybrid branch.")
    top_k: int = Field(default=10, ge=1, le=50, description="Passages returned by a search.")
    context_k: int = Field(default=5, ge=1, le=20, description="Passages sent to the model.")
    rrf_k: int | None = Field(
        default=None, ge=1, description="RRF constant; None = Qdrant default."
    )
    rerank: bool = False
    rerank_candidates: int = Field(
        default=20, ge=2, le=50, description="Candidates scored by the cross-encoder."
    )
    chunk_max_chars: int = Field(default=1500, ge=300)
    chunk_pack_chars: int = Field(
        default=700, ge=200, description="Size of the slices of a split article."
    )

    # --- language models -----------------------------------------------------------------
    llm_model: str = "groq:openai/gpt-oss-120b"
    llm_fallback_model: str | None = "google_genai:gemini-3.5-flash-lite"
    fast_llm_model: str = "groq:openai/gpt-oss-20b"
    fast_llm_fallback_model: str | None = "google_genai:gemini-3.5-flash-lite"
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_reasoning_effort: ReasoningEffort = "low"

    # --- agent ---------------------------------------------------------------------------
    max_rewrites: int = Field(default=1, ge=0, le=3)
    max_regenerations: int = Field(default=1, ge=0, le=2)
    grading: bool = True
    verification: bool = True
    history_turns: int = Field(default=4, ge=0, le=20)

    # --- API and UI ----------------------------------------------------------------------
    api_url: str = "http://127.0.0.1:8000"
    api_key: SecretStr | None = Field(
        default=None, description="When set, the API requires it in the X-API-Key header."
    )
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    allowed_hosts: list[str] = Field(
        default=["localhost", "127.0.0.1", "[::1]", "api"],
        description="Host headers the API accepts (guards against DNS rebinding); '*' allows all.",
    )
    max_question_chars: int = Field(default=2000, ge=10, le=8000)
    log_level: str = "INFO"

    @property
    def corpus_dir(self) -> Path:
        """Folder with the parsed, versioned corpus (``*.jsonl`` + ``manifest.json``)."""
        return self.data_dir / "corpus"

    @property
    def raw_dir(self) -> Path:
        """Folder with downloaded source pages (not versioned)."""
        return self.data_dir / "raw"

    @property
    def eval_dir(self) -> Path:
        """Folder with the golden evaluation set."""
        return self.data_dir / "eval"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
