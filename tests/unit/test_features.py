"""Checks that the shared schema and the 674 preprocessing agree, before any data is committed."""

import numpy as np
import pandas as pd

from olist_mlops.features import CAT_COLS, DIST_COLS, LOG_COLS, OTHER_NUM, WEIGHT_COLS
from olist_mlops.features import make_preprocess
from olist_mlops.schema import BRAZILIAN_STATES, CATEGORICAL_COLUMNS, COLUMN_SPECS
from olist_mlops.schema import INPUT_COLUMNS, NULLABLE_COLUMNS


def _orders(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Small made-up orders inside the schema's ranges, with the nullable columns partly missing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "promised_days_from_approval": rng.uniform(5, 40, n),
            "approval_lag_hours": rng.uniform(0, 48, n),
            "purchase_weekday": rng.integers(0, 7, n),
            "purchase_hour": rng.integers(0, 24, n),
            "item_count": rng.integers(1, 4, n),
            "distinct_product_count": 1,
            "seller_count": 1,
            "total_price": rng.uniform(10, 500, n),
            "total_freight": rng.uniform(5, 60, n),
            "mean_product_weight_g": rng.uniform(100, 5000, n),
            "total_product_volume_cm3": rng.uniform(500, 50_000, n),
            "mean_distance_km": rng.uniform(5, 3000, n),
            "same_state_item_share": rng.integers(0, 2, n).astype(float),
            "customer_state": rng.choice(["SP", "RJ", "MG", "BA"], n),
            "any_seller_state": rng.choice(["SP", "PR"], n),
            "primary_product_category": rng.choice(["bed_bath_table", "toys"], n).astype(object),
        }
    )
    df["max_product_weight_g"] = df["mean_product_weight_g"]
    df["max_distance_km"] = df["mean_distance_km"]
    df["freight_to_price_ratio"] = df["total_freight"] / df["total_price"]
    df.loc[:4, NULLABLE_COLUMNS] = np.nan
    return df[INPUT_COLUMNS]


def test_schema_has_the_19_inputs_of_the_674_model():
    assert len(INPUT_COLUMNS) == 19
    assert len(set(INPUT_COLUMNS)) == 19
    assert "purchase_month" not in INPUT_COLUMNS


def test_preprocessing_column_groups_cover_exactly_the_schema():
    # A column in the schema but in no group would be silently dropped by the pipeline.
    used = LOG_COLS + DIST_COLS + WEIGHT_COLS + [c for c in OTHER_NUM if c != "purchase_month"]
    used += CAT_COLS
    assert sorted(used) == sorted(INPUT_COLUMNS)
    assert sorted(CAT_COLS) == sorted(CATEGORICAL_COLUMNS)


def test_preprocessing_leaves_no_missing_values():
    X = make_preprocess(min_frequency=1).fit_transform(_orders())
    assert not np.isnan(X.astype(float)).any()


def test_unseen_category_at_serving_does_not_fail():
    pre = make_preprocess(min_frequency=1).fit(_orders())
    new = _orders(n=1, seed=1)
    new["primary_product_category"] = "a_category_never_seen_in_training"
    assert pre.transform(new).shape[0] == 1


def test_states_are_restricted_to_brazil():
    # validate() and the API reject an order whose state is not one of the 27 units.
    for col in ["customer_state", "any_seller_state"]:
        assert COLUMN_SPECS[col].allowed == BRAZILIAN_STATES
    assert len(BRAZILIAN_STATES) == 27


def test_purchase_month_is_dropped_by_default():
    # Every model trains without purchase_month (decision 5), even if a caller passes it in.
    orders = _orders()
    orders["purchase_month"] = 5
    pre = make_preprocess(min_frequency=1).fit(orders)
    assert not any("purchase_month" in name for name in pre.get_feature_names_out())
