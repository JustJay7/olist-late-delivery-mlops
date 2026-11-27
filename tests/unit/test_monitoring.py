import pandas as pd

from flows.monitoring_flow import (
    load_serving_baseline,
    prepare_current_features,
)
from olist_mlops.schema import INPUT_COLUMNS


def test_load_serving_baseline():
    baseline = load_serving_baseline.fn()

    assert baseline.shape == (5000, 19)
    assert list(baseline.columns) == INPUT_COLUMNS


def test_prepare_empty_predictions():
    predictions = pd.DataFrame(columns=["features"])

    current = prepare_current_features.fn(predictions)

    assert current.shape == (0, 19)
    assert list(current.columns) == INPUT_COLUMNS


def test_prepare_current_features():
    baseline = load_serving_baseline.fn()
    sample = baseline.iloc[0].to_dict()

    predictions = pd.DataFrame({"features": [sample]})
    current = prepare_current_features.fn(predictions)

    assert current.shape == (1, 19)
    assert list(current.columns) == INPUT_COLUMNS

    pd.testing.assert_frame_equal(
        current,
        baseline.iloc[[0]].reset_index(drop=True),
        check_dtype=False,
    )
