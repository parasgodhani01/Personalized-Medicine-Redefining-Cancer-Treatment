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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.sparse import hstack

# Add src/ to path so we can import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import build_combined_feature

# ── CONFIG ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
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
    Load the model from MLflow once when the server starts.
    """
    global _model, _tfidf, _gene_tfidf, _var_tfidf

    print(f"[STARTUP] Loading model from MLflow...")
    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Model Name   : {MODEL_NAME}")
    print(f"  Stage        : {MODEL_STAGE}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        _model    = mlflow.sklearn.load_model(model_uri)
        print(f"  [✓] Model loaded from: {model_uri}")
    except Exception as e:
        print(f"  [!] Could not load from registry: {e}")
        print(f"  [!] Falling back to local 'latest' run...")
        # Fallback: load from latest run artifact (useful in dev/testing)
        client = mlflow.tracking.MlflowClient()
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

                # Load transformers from that run's artifacts
                artifacts_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="transformers"
                )
                _tfidf      = joblib.load(os.path.join(artifacts_path, "tfidf.joblib"))
                _gene_tfidf = joblib.load(os.path.join(artifacts_path, "gene_tfidf.joblib"))
                _var_tfidf  = joblib.load(os.path.join(artifacts_path, "var_tfidf.joblib"))
                print(f"  [✓] Transformers loaded from run artifacts")

    # Load transformers from disk if not already loaded via MLflow
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
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    if _tfidf is None:
        raise HTTPException(status_code=503, detail="Transformers not loaded.")

    # Build features — MUST use the same logic as training
    combined = build_combined_feature(request.gene, request.variation, request.clinical_text)

    X_text = _tfidf.transform([combined])
    X_gene = _gene_tfidf.transform([request.gene.lower()])
    X_var  = _var_tfidf.transform([request.variation.lower()])
    X      = hstack([X_text, X_gene, X_var])

    # sklearn model loaded directly — predict_proba returns all class probabilities
    probabilities = _model.predict_proba(X)

    # Handle both ndarray and DataFrame outputs
    if hasattr(probabilities, "values"):
        probabilities = probabilities.values
    probabilities = np.array(probabilities).flatten()

    predicted_idx   = int(np.argmax(probabilities))
    predicted_class = predicted_idx + 1   # back to 1-indexed
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

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_class_name=CLASS_DESCRIPTIONS.get(predicted_class, f"Class {predicted_class}"),
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
        gene=request.gene,
        variation=request.variation,
    )


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