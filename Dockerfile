# Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build — keeps the final image small and secure.
#
# Stage 1 (builder): Install all deps including build tools
# Stage 2 (runtime): Copy only what's needed to run the app
#
# WHY multi-stage?
#   - Build tools (gcc, pip build deps) don't belong in production
#   - Final image is ~300MB smaller = faster pull on EC2 at deploy time
#   - Smaller attack surface (fewer packages = fewer CVEs)
# ─────────────────────────────────────────────────────────────

# ── STAGE 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Copy requirements first — Docker caches this layer.
# Only invalidated when requirements.txt changes (not when code changes).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── STAGE 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-root user — NEVER run as root in production
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/preprocess.py  ./src/preprocess.py
COPY app/main.py        ./app/main.py

# Add src to PYTHONPATH so app/main.py can import from src/
ENV PYTHONPATH="/app/src"

# MLflow config — override via docker run -e or docker-compose
ENV MLFLOW_TRACKING_URI="mlruns"
ENV MODEL_NAME="cancer-classifier"
ENV MODEL_STAGE="Production"
ENV TRANSFORMERS_DIR="/app/transformers"

# Create directory for transformers (mounted or copied in at deploy)
RUN mkdir -p /app/transformers

# Switch to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start FastAPI with Uvicorn
# --workers 2: 2 worker processes (adjust based on EC2 instance size)
# --host 0.0.0.0: listen on all interfaces (needed inside Docker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
