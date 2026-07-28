# promote_model.py
# ─────────────────────────────────────────────────────────────
# Promotes the LATEST registered version of cancer-classifier
# to the "Production" stage — no need to hardcode a version
# number, which changes every time you retrain.
#
# Usage:
#   python promote_model.py
# ─────────────────────────────────────────────────────────────

import os
import mlflow

# Point at the same tracking server main.py / docker-compose use.
# Override with an env var if needed, e.g. for local sqlite testing.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME   = "cancer-classifier"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Find the latest version registered, regardless of its current stage.
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not all_versions:
    print(f"No versions found for model '{MODEL_NAME}'. Did training run against {TRACKING_URI}?")
    raise SystemExit(1)

latest_version = max(all_versions, key=lambda v: int(v.version))

print(f"Tracking URI : {TRACKING_URI}")
print(f"Model        : {MODEL_NAME}")
print(f"Latest version found: v{latest_version.version} (run_id={latest_version.run_id})")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True,  # old "Production" versions get archived, not left dangling
)

print(f"✓ Promoted v{latest_version.version} of '{MODEL_NAME}' to Production")