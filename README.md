# Personalized Medicine - Cancer Mutation Classifier


A production-style ML system that classifies genetic mutations into 9 clinical
categories from the MSK "Personalized Medicine: Redefining Cancer Treatment"
Kaggle dataset — built to prove real MLOps engineering judgment, not just
model accuracy.



## Why This Project Exists

Most ML portfolio projects stop at "I trained a model and got X% accuracy."
This one goes further: it trains a model, **registers** it, **serves** it
behind a real API, **containerizes** the whole stack, **load-tests** it under
concurrency, and **monitors** it live with the same tools (Prometheus +
Grafana) that production ML systems use in industry.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training (local)                                       │
│  train.py → scikit-learn (LogReg / RF / XGBoost)        │
│  TF-IDF feature engineering on clinical text + gene/var │
└───────────────────────┬─────────────────────────────────┘
                         │ logs experiments, registers best model
                         ▼
┌─────────────────────────────────────────────────────────┐
│  MLflow Tracking Server        (Docker container)       │
│  • Experiment tracking, model registry                  │
│  • Proxied artifact storage (client uploads via HTTP)   │
└───────────────────────┬─────────────────────────────────┘
                         │ loads "Production" model at startup
                         ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Inference API         (Docker container)       │
│  • POST /predict — classification + confidence          │
│  • GET  /health  — liveness check                       │
│  • GET  /metrics — Prometheus-format instrumentation    │
└───────────────────────┬─────────────────────────────────┘
                         │ scraped every 15s
                         ▼
┌──────────────────────────────────────────────────────────┐ 
│  Prometheus (native)  →  Grafana (native)                │
│  • Request rate, p95/p99 latency                         │
│  • Prediction class distribution, confidence distribution│
└──────────────────────────────────────────────────────────┘
```

## What's Working

- **Training pipeline** — 3 models trained and compared (Logistic Regression
  won on test log-loss: 1.0028 vs 1.1356 RF vs 1.3396 XGBoost); MLflow tracks
  every run and manages the registry.
- **Containerized serving** — `mlflow` + `api` services via Docker Compose,
  stable in continuous operation for 6+ days.
- **Load testing** — Locust-driven concurrency tests up to 50 simultaneous
  users: **p95 latency 49ms, 0 failures across ~4,000 requests.** Identified
  and explained a cold-start latency pattern (first ~30s of traffic only —
  same phenomenon as AWS Lambda cold starts), distinguishing it from a real
  bottleneck.
- **Live monitoring** — Prometheus scrapes custom metrics (request count,
  latency histogram, prediction class distribution, confidence distribution)
  from the API; Grafana renders real-time dashboards proven against live
  traffic, not mock data.
- **23 unit tests** covering the training/inference pipeline.

---

## Load Test Results

| Concurrency | Requests | Failures | Median | p95 | p99* |
|---|---|---|---|---|---|
| 10 users | 292 | 0 | 17ms | 30ms | 2100ms* |
| 50 users | 1,997 | 0 | 18ms | 49ms | 2100ms* |

\* *p99/max latency spikes were isolated entirely to the first ~30 seconds of
each test run (confirmed via Locust's response-time chart) — a one-time
cold-start cost from lazy model/library initialization, not a sustained
bottleneck. Real-world mitigation: send warm-up requests immediately after
deployment, before opening traffic to real users.*

**SLA achieved:** p95 < 100ms, sustained, at up to 50 concurrent users, zero
failures across ~4,000 requests.

---

## Repository Structure

```
├── data/                      # training_variants.csv, training_text.csv
├── main.py                    # FastAPI inference server
├── train.py                   # Training pipeline (3-model comparison)
├── train_fast.py              # Quick single-model retrain for verification
├── preprocess.py               # Shared text-cleaning / feature-building
├── promote_model.py            # Auto-promotes latest MLflow version to Production
├── test_pipeline.py             # 23 unit tests
├── locustfile.py                # Load testing scenarios
├── LOAD_TEST_RESULTS.md         # Documented load test findings
├── Dockerfile                  # FastAPI service image
├── Dockerfile.mlflow            # MLflow tracking server image
├── docker-compose.yml           # 2-service orchestration (mlflow + api)
├── init.sql                    # MySQL schema (designed, not yet wired in)
├── prometheus.yml               # Prometheus scrape config
├── aws_setup.sh                 # EC2 provisioning script (scaffolding)
├── ci_cd.yml                    # GitHub Actions pipeline (scaffolding)
└── requirements.txt
```