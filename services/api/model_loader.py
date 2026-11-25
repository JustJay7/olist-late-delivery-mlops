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
