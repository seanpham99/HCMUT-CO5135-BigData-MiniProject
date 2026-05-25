# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build the webclaw CLI from source
# ──────────────────────────────────────────────────────────────────────────────
FROM rust:1-slim-bookworm AS webclaw-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    g++ \
    make \
    perl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install --git https://github.com/0xMassi/webclaw.git webclaw-cli

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Production Airflow image
# ──────────────────────────────────────────────────────────────────────────────
FROM apache/airflow:latest 

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install webclaw with proper execute permissions
COPY --from=webclaw-builder /usr/local/cargo/bin/webclaw /usr/local/bin/webclaw
RUN chmod 755 /usr/local/bin/webclaw

WORKDIR /opt/airflow

# Copy dependency/project metadata files needed for wheel build
# These are copied before source for optimal layer caching
COPY pyproject.toml uv.lock README.md /opt/airflow/

# Install Python packages using uv with proper cache mount path
RUN --mount=type=cache,id=uv,target=/root/.cache/uv \
    uv export --format requirements-txt --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

USER airflow
