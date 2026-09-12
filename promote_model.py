import os
import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME   = "cancer-classifier"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

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
    archive_existing_versions=True,  
)

print(f"Promoted v{latest_version.version} of '{MODEL_NAME}' to Production")