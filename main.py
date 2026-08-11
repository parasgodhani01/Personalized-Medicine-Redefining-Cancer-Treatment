# app/main.py
# ─────────────────────────────────────────────────────────────
# FastAPI inference server.
#
# WHY FastAPI over Flask?
#   - Async support (handles concurrent prediction requests)
#   - Auto-generates OpenAPI docs at /docs
#   - Pydantic validation catches bad inputs before they hit the model
#   - Production-grade, used by Uber, Netflix, etc.
#
# The model is loaded ONCE at startup from MLflow Model Registry.
# This is the correct pattern — not loading on every request.
# ─────────────────────────────────────────────────────────────

import os
import sys
import joblib
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.sparse import hstack
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Add src/ to path so we can import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import build_combined_feature

# ── PROMETHEUS METRICS ───────────────────────────────────────
# Counter: monotonically increasing count (requests, errors)
# Histogram: buckets of observed values (latency) — gives us
# percentiles (p50/p95/p99) when queried in Prometheus/Grafana.

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of requests received",
    ["endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

PREDICTION_CLASS_COUNT = Counter(
    "predictions_by_class_total",
    "Count of predictions made per mutation class",
    ["predicted_class"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence score distribution of predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ── CONFIG ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME          = os.getenv("MODEL_NAME", "cancer-classifier")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")   # or "latest"
TRANSFORMERS_DIR    = os.getenv("TRANSFORMERS_DIR", "/app/transformers")

CLASS_DESCRIPTIONS = {
    1: "Likely Loss-of-function",
    2: "Likely Gain-of-function",
    3: "Neutral",
    4: "Loss-of-function",
    5: "Likely Neutral",
    6: "Inconclusive",
    7: "Gain-of-function",
    8: "Likely Switch-of-function",
    9: "Switch-of-function",
}

# Global model + transformers (loaded once at startup)
_model       = None
_tfidf       = None
_gene_tfidf  = None
_var_tfidf   = None


# ── LIFESPAN: Load model at startup ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — replaces deprecated @app.on_event("startup").
    Load the model from MLflow once when the server starts, then load the
    matching transformers from whichever run produced that model — whether
    we got the model via the registry (Production stage) or the fallback
    (latest run search).
    """
    global _model, _tfidf, _gene_tfidf, _var_tfidf

    print(f"[STARTUP] Loading model from MLflow...")
    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Model Name   : {MODEL_NAME}")
    print(f"  Stage        : {MODEL_STAGE}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    run_id = None

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        _model    = mlflow.sklearn.load_model(model_uri)
        print(f"  [✓] Model loaded from: {model_uri}")

        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        run_id = versions[0].run_id

    except Exception as e:
        print(f"  [!] Could not load from registry: {e}")
        print(f"  [!] Falling back to local 'latest' run...")
        experiment = client.get_experiment_by_name("personalized-medicine")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_log_loss ASC"],
                max_results=1,
            )
            if runs:
                run_id    = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                _model    = mlflow.sklearn.load_model(model_uri)
                print(f"  [✓] Loaded best run model: {run_id}")

    if run_id:
        artifacts_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="transformers"
        )
        _tfidf      = joblib.load(os.path.join(artifacts_path, "tfidf.joblib"))
        _gene_tfidf = joblib.load(os.path.join(artifacts_path, "gene_tfidf.joblib"))
        _var_tfidf  = joblib.load(os.path.join(artifacts_path, "var_tfidf.joblib"))
        print(f"  [✓] Transformers loaded from run artifacts")

    if _tfidf is None and os.path.exists(TRANSFORMERS_DIR):
        _tfidf      = joblib.load(os.path.join(TRANSFORMERS_DIR, "tfidf.joblib"))
        _gene_tfidf = joblib.load(os.path.join(TRANSFORMERS_DIR, "gene_tfidf.joblib"))
        _var_tfidf  = joblib.load(os.path.join(TRANSFORMERS_DIR, "var_tfidf.joblib"))
        print(f"  [✓] Transformers loaded from {TRANSFORMERS_DIR}")

    print("[STARTUP] Server ready ✓\n")
    yield
    print("[SHUTDOWN] Cleaning up...")


# ── FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(
    title="Personalized Medicine — Cancer Mutation Classifier",
    description="Classifies genetic mutations into 9 classes using NLP + ML",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REQUEST / RESPONSE SCHEMAS ────────────────────────────────
class PredictionRequest(BaseModel):
    gene: str = Field(..., example="BRCA1", description="Gene name")
    variation: str = Field(..., example="R1699Q", description="Variation identifier")
    clinical_text: str = Field(..., min_length=10,
                               example="The BRCA1 gene plays a critical role in DNA repair...",
                               description="Clinical literature text")

class ClassProbability(BaseModel):
    class_id: int
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    confidence: float
    all_probabilities: list[ClassProbability]
    gene: str
    variation: str


# ── ROUTES ────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "model": MODEL_NAME, "stage": MODEL_STAGE}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus scrapes this endpoint on a schedule (e.g. every 10s).
    Returns all counters/histograms in Prometheus's plain-text format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["Health"])
def health():
    model_loaded = _model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "transformers_loaded": _tfidf is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Predict the mutation class for a given gene, variation, and clinical text.
    Returns predicted class + confidence + all 9 class probabilities.
    """
    start_time = time.time()
    status_code = "200"

    try:
        if _model is None:
            status_code = "503"
            raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
        if _tfidf is None:
            status_code = "503"
            raise HTTPException(status_code=503, detail="Transformers not loaded.")

        combined = build_combined_feature(request.gene, request.variation, request.clinical_text)

        X_text = _tfidf.transform([combined])
        X_gene = _gene_tfidf.transform([request.gene.lower()])
        X_var  = _var_tfidf.transform([request.variation.lower()])
        X      = hstack([X_text, X_gene, X_var])

        probabilities = _model.predict_proba(X)

        if hasattr(probabilities, "values"):
            probabilities = probabilities.values
        probabilities = np.array(probabilities).flatten()

        predicted_idx   = int(np.argmax(probabilities))
        predicted_class = predicted_idx + 1
        confidence      = float(probabilities[predicted_idx])

        all_probs = [
            ClassProbability(
                class_id=i + 1,
                class_name=CLASS_DESCRIPTIONS.get(i + 1, f"Class {i+1}"),
                probability=round(float(p), 4),
            )
            for i, p in enumerate(probabilities)
        ]
        all_probs.sort(key=lambda x: x.probability, reverse=True)

        # Record prediction-specific metrics
        PREDICTION_CLASS_COUNT.labels(predicted_class=str(predicted_class)).inc()
        PREDICTION_CONFIDENCE.observe(confidence)

        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_class_name=CLASS_DESCRIPTIONS.get(predicted_class, f"Class {predicted_class}"),
            confidence=round(confidence, 4),
            all_probabilities=all_probs,
            gene=request.gene,
            variation=request.variation,
        )
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    finally:
        # Always record request count + latency, success or failure
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=status_code).inc()


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the currently loaded model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            v = versions[0]
            return {
                "model_name": MODEL_NAME,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "source": v.source,
            }
    except Exception as e:
        return {"error": str(e)}
    return {"model_name": MODEL_NAME, "stage": MODEL_STAGE}