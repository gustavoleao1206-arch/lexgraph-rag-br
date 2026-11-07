# syntax=docker/dockerfile:1

# One image runs every service of compose.yaml: the API (default command), the one-shot
# `lexgraph ingest` job and the Streamlit UI (`lexgraph ui`).
#
#   docker build -t lexgraph:local .
#   docker build -t lexgraph:local --build-arg INCLUDE_RERANKER=true .
#   docker build -t lexgraph:local --build-arg LEXGRAPH_EMBEDDING_MODEL=<fastembed model> .
#   docker build -t lexgraph:local --secret id=hf_token,env=HF_TOKEN .
#
# The LEXGRAPH_* model arguments select the models baked into the image; unset or empty ones
# keep the defaults of lexgraph.config.Settings. The optional hf_token secret authenticates
# the model downloads. A code change never reruns the model download (about 1.3 GB), and the
# runtime layers go from the least to the most volatile, so it replaces only the last one.

# The external images are pinned by digest, each named in a single FROM line, and Dependabot
# keeps their tags and digests current. The syntax line at the top is not pinned: it follows
# docker/dockerfile:1, the stable channel of the BuildKit frontend.
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv
# Every stage builds on this image: the virtual environment built in one stage is copied to
# another and must find its interpreter at the same path.
FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base

# --- deps: third-party packages from uv.lock (rebuilt only when the lock changes) ---------
FROM base AS deps
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra ui --no-editable --no-install-project

# --- model-plan: the models the build arguments select, as a small text file ------------
FROM deps AS model-plan
ARG LEXGRAPH_EMBEDDING_PROVIDER
ARG LEXGRAPH_EMBEDDING_MODEL
ARG LEXGRAPH_SPARSE_MODEL
ARG LEXGRAPH_RERANKER_MODEL
# The cross-encoder is optional (about 0.6 GB). Without it, the API reports reranking as
# unavailable and the UI does not offer it.
ARG INCLUDE_RERANKER=false
COPY src/lexgraph/__init__.py src/lexgraph/config.py src/lexgraph/
COPY scripts/download_models.py scripts/
RUN PYTHONPATH=/app/src /app/.venv/bin/python scripts/download_models.py \
        --include-reranker="${INCLUDE_RERANKER}" --save-plan /app/models.txt

# --- models: FastEmbed models baked into the image, so containers never download one ------
# COPY --from compares file contents, so the download reruns only when the plan, the script
# or the locked packages change, not when the code (the settings module included) does.
FROM deps AS models
COPY scripts/download_models.py scripts/
COPY --from=model-plan /app/models.txt ./
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    /app/.venv/bin/python scripts/download_models.py \
        --plan models.txt --cache-dir /app/data/models

# --- builder: the project as a wheel ------------------------------------------------------
FROM deps AS builder
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /app/dist

# --- runtime: models, corpus and packages; no uv, no sources, no build tools -------------
FROM base AS runtime
LABEL org.opencontainers.image.title="lexgraph" \
      org.opencontainers.image.description="Agentic RAG over Brazilian federal legislation" \
      org.opencontainers.image.licenses="MIT"

# Unprivileged user. It owns only what the application writes: data/ (embedded Qdrant,
# conversation checkpoints, whose volume inherits this ownership) and the model cache.
RUN groupadd --system --gid 999 lexgraph \
 && useradd --system --gid 999 --uid 999 --create-home lexgraph \
 && install --directory --owner=999 --group=999 /app/data /app/data/checkpoints

WORKDIR /app
COPY --from=models --chown=999:999 /app/data/models /app/data/models
COPY data/corpus /app/data/corpus
COPY .streamlit /app/.streamlit
COPY --from=deps /app/.venv /app/.venv
# The project goes last, without its dependencies (already copied) and with uv mounted for
# this step only; bytecode compilation skips the copied packages, whose .pyc are current.
RUN --mount=type=bind,from=uv,source=/uv,target=/bin/uv \
    --mount=type=bind,from=builder,source=/app/dist,target=/tmp/dist \
    uv pip install --python /app/.venv/bin/python --no-deps --no-cache \
        --link-mode=copy --compile-bytecode /tmp/dist/*.whl

# Streamlit reads .streamlit/config.toml from the working directory; the bind address and
# port come from the `lexgraph ui --host/--port` options.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    STREAMLIT_SERVER_HEADLESS=true

USER 999:999
EXPOSE 8000 8501

# The image has no curl; urlopen raises (exit code 1) on errors and non-2xx responses.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --start-interval=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["lexgraph", "serve", "--host", "0.0.0.0"]
