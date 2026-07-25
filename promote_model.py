import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="cancer-classifier",
    version=1,
    stage="Production"
)