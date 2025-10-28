# Dockerfile for pipeguru-ssr-api - FastAPI application for Cloud Run
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
FROM base AS deps

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install PyTorch CPU-only version first to avoid downloading CUDA libraries
# This significantly reduces build time and image size for Cloud Run (CPU-only)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies with cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Production image
FROM base AS runner

WORKDIR /app

# Copy installed dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY default_embeddings/ ./default_embeddings/

# Copy production environment file
COPY .env.production .env

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run uvicorn server
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
