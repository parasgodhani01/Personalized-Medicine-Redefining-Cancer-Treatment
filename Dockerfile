# Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build — keeps the final image small and secure.
# ─────────────────────────────────────────────────────────────

# ── STAGE 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── STAGE 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY --from=builder /install /usr/local

# Flat project structure — files live at project root, not src/ or app/
COPY preprocess.py .
COPY main.py .

# MLflow config — overridden by docker-compose environment
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV MODEL_NAME="cancer-classifier"
ENV MODEL_STAGE="Production"
ENV TRANSFORMERS_DIR="/app/transformers"

RUN mkdir -p /app/transformers

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Flat structure — module is main:app, not app.main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]