"""Monitor recent predictions against the serving baseline."""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from prefect import flow, task
from olist_mlops.config import ROOT
from olist_mlops.schema import INPUT_COLUMNS

load_dotenv(ROOT / ".env")
BASELINE_PATH = ROOT / "data" / "reference" / "serving_baseline.parquet"
WINDOW_SIZE = 500


def get_db_connection():
    """Connect to the application PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


@task
def load_recent_predictions():
    """Load the latest predictions from PostgreSQL."""
    query = """
        SELECT
            id,
            created_at,
            order_id,
            features,
            late_probability,
            flagged,
            model_version,
            source
        FROM predictions
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (WINDOW_SIZE,))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    finally:
        conn.close()


@task
def load_serving_baseline():
    """Load the reference data used for drift monitoring."""
    baseline = pd.read_parquet(BASELINE_PATH)

    return baseline[INPUT_COLUMNS].copy()


@task
def prepare_current_features(predictions):
    """Convert stored JSON features into model input columns."""
    if predictions.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)

    features = pd.DataFrame(predictions["features"].tolist())

    return features[INPUT_COLUMNS].copy()


@flow(name="monitoring-flow")
def monitoring_flow():
    """Load reference and current data for monitoring."""
    predictions = load_recent_predictions()
    baseline = load_serving_baseline()
    current = prepare_current_features(predictions)

    print(f"Reference data shape: {baseline.shape}")
    print(f"Current data shape: {current.shape}")

    if current.empty:
        print("No predictions available yet.")
        return

    print("Monitoring data loaded successfully.")


if __name__ == "__main__":
    monitoring_flow()
