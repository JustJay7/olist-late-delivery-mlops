from pathlib import Path
import sys

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from olist_mlops.features import make_preprocess
from olist_mlops.schema import INPUT_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data/sample/orders_sample.parquet"
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test-model"


def main():
    orders = pd.read_parquet(SAMPLE_PATH)

    X = orders[INPUT_COLUMNS]
    y = orders[TARGET]

    model = Pipeline(
        [
            ("pre", make_preprocess(min_frequency=20, drop_month=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(np.quantile(probabilities, 0.9))

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(MODEL_PATH),
        serialization_format="cloudpickle",
        metadata={"flag_threshold": threshold},
    )

    print(f"Test model saved to: {MODEL_PATH}")
    print(f"Training orders: {len(X)}")
    print(f"Flag threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
