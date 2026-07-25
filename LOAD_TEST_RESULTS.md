# Load Test Results — Cancer Mutation Classifier API

## Test Setup
- **Tool**: Locust
- **Endpoint tested**: `POST /predict`
- **Server**: `uvicorn main:app --port 8000` (no `--reload`)
- **Model**: Logistic Regression, loaded via MLflow Model Registry (`Production` stage)

---

## Test 1 — 10 concurrent users, ~2 min sustained
| Metric | Value |
|---|---|
| Total requests | 292 (`/predict`) |
| Failures | 0 |
| Median | 17ms |
| p95 | 30ms |
| p99 | 2100ms (cold-start only, see finding below) |
| Max | 2092ms |

---

## Test 2 — 50 concurrent users, ~3 min sustained
| Metric | Value |
|---|---|
| Total requests | 1997 (`/predict`) |
| Failures | 0 |
| Median | 18ms |
| p95 | 49ms |
| p99 | 2100ms (cold-start only) |
| Max | 2137ms |
| Sustained throughput | ~24 RPS |

**Observation**: Going from 10 → 50 concurrent users (5x load), median latency
barely moved (17ms → 18ms) and p95 stayed well under the 100ms target (30ms →
49ms), with zero failures across ~4000 total requests. This indicates the API
has meaningful headroom above 50 concurrent users before real degradation.

---

## Finding: Cold-start latency, not a sustained bottleneck
In both tests, the p99/max spike is isolated entirely to the **first ~30
seconds** of the test (confirmed via Locust's Response Times chart — the 95th
percentile line spikes early, then drops to near-zero and stays flat for the
remainder of the sustained run). After the warm-up window, p95/p99 stayed
consistently low with zero failures.

**Root cause (likely)**: first-request lazy initialization in sklearn/joblib,
OS file-cache warming for the transformer files, and FastAPI's thread pool
spinning up on first use.

**Real-world implication**: this is the same class of issue as AWS Lambda
cold starts. Mitigation in production is operational, not code — send a
few warm-up requests immediately after deployment, before opening traffic
to real users.

---

## SLA Definition (based on measured data, not assumed)
> **p95 < 100ms post-warm-up, sustained, up to at least 50 concurrent users.**
> Confirmed met in Test 2 (p95 = 49ms).

## Next Steps
- [ ] Test at 100 concurrent users to find the actual breaking point
- [ ] Add a warm-up step to deployment (few dummy requests before serving real traffic)
- [ ] Cross-reference client-side Locust numbers against server-side Prometheus metrics once that's wired up