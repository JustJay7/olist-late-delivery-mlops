import os
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

MODEL_NAME = "olist-late-delivery"
MODEL_ALIAS = "champion"


@dataclass(frozen=True)
class LoadedModel:
    model: Pipeline
    threshold: float
    version: str


def load_champion(
    current_version: str | None = None,
) -> LoadedModel | None:

    # Test-only override: bypass the MLflow registry.
    override_uri = os.getenv("MODEL_URI")

    if override_uri:
        if current_version == "test":
            return None

        model = mlflow.sklearn.load_model(override_uri)

        metadata = mlflow.models.Model.load(override_uri).metadata

        threshold = float(metadata["flag_threshold"])

        return LoadedModel(
            model=model,
            threshold=threshold,
            version="test",
        )

    # Normal operation: load the registered champion.
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    version = str(champion.version)

    if version == current_version:
        return None

    run = client.get_run(champion.run_id)
    threshold = float(run.data.metrics["flag_threshold"])

    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    return LoadedModel(
        model=model,
        threshold=threshold,
        version=version,
    )
