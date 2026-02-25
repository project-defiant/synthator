# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1: build — install dependencies into an isolated venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# git: resolve the alphagenome-research git dependency
# build-essential / python3-dev: compile C extensions (ncls via pyranges)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy lock file and project metadata first for layer caching
COPY uv.lock pyproject.toml README.md ./
COPY src/ src/

# Install production dependencies only into /build/.venv
RUN uv sync --frozen --no-group dev

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image with the installed venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the fully-built venv from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Put the venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["synthator"]
