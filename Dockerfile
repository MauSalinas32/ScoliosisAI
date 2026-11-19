# ==============================================================================
# Multi-Stage Production Dockerfile — Medical AI System
# ==============================================================================
# Stage 1: Dependency Builder
# Stage 2: Minimal Production Runtime with Non-Root Security Context
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies into isolated wheels directory
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Final Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Metadata labels
LABEL maintainer="Equipo 3 - Software Engineering II" \
      version="3.3.0" \
      description="Medical Image Classification System with Transfer Learning & Vertebral Tracking"

WORKDIR /app

# Install minimal runtime OS dependencies with security patches (OpenCV headless & image processing)
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the wheels built in stage 1. They are bind-mounted (not COPY'd) so the ~2 GB of wheels
# never become an image layer; only the installed packages remain.
COPY requirements.txt .
RUN --mount=type=bind,from=builder,source=/build/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Create dedicated non-root application user for clinical data security compliance
# Numeric UID/GID 10001: identical to the Kubernetes securityContext (runAsUser/fsGroup) so volume ownership matches
RUN groupadd --gid 10001 medicalgroup && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /app --shell /usr/sbin/nologin medicaluser

# Copy application source code
COPY . /app/

# Create persistent storage directories with proper permissions
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/models /app/data/reports /app/logs && \
    chown -R medicaluser:medicalgroup /app

# Switch to non-root user
USER 10001:10001

# Expose internal API service port
EXPOSE 8000

# Container healthcheck for orchestrator readiness / liveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# One worker per container: TensorFlow already uses the pod's cores for one forward pass and every worker would
# load its own copy of the model (about 0.8 GB). Capacity grows with replicas (HPA), not with workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
