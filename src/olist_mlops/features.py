"""Preprocessing shared by training and serving.

make_preprocess() is copied from our INSY 674 notebook 05 (section 3.4 pipeline) without
changes to the steps, so the model trains exactly as it did last course and its CV results
stay comparable. The column groups say which transformation each column gets.
"""

from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from olist_mlops.ingest import REFERENCE_DIR, haversine_km
from olist_mlops.schema import COLUMN_SPECS, INPUT_COLUMNS

DIST_COLS = ["mean_distance_km", "max_distance_km"]
WEIGHT_COLS = ["mean_product_weight_g", "max_product_weight_g"]
# Right-skewed with no missing values: logged, then scaled
LOG_COLS = [
    "total_price",
    "seller_count",
    "total_freight",
    "item_count",
    "distinct_product_count",
    "total_product_volume_cm3",
    "approval_lag_hours",
    "freight_to_price_ratio",
    "promised_days_from_approval",
]
# Only scaled. purchase_month stays listed so drop_month=False still reproduces the 674
# experiment that kept it, but it is not one of the 19 model inputs.
OTHER_NUM = ["purchase_month", "purchase_weekday", "purchase_hour", "same_state_item_share"]
CAT_COLS = ["customer_state", "any_seller_state", "primary_product_category"]


def make_preprocess(min_frequency: int = 100, drop_month: bool = True) -> ColumnTransformer:
    """674 section 3.4 pipeline. `min_frequency` and `drop_month` were tuning choices in 5.1.

    The final model uses min_frequency=20. drop_month defaults to True here (False in the
    notebook) because locked decision 5 trains every candidate without purchase_month.
    With drop_month=True, purchase_month is left out of the column lists, and
    ColumnTransformer's default remainder="drop" removes it if present.
    """
    other_num = [c for c in OTHER_NUM if not (drop_month and c == "purchase_month")]
    log_scale = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    fill_log_scale = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    # Rare and unseen categories become one "infrequent" column instead of failing at serving.
    cat_pipe = Pipeline(
        [
            ("fill", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("log_scale", log_scale, LOG_COLS),
            ("fill_log_scale", fill_log_scale, DIST_COLS + WEIGHT_COLS),
            # A missing distance means the zip was not geocoded, which may itself carry risk.
            ("dist_flag", MissingIndicator(features="all"), DIST_COLS),
            ("scale", StandardScaler(), other_num),
            ("categories", cat_pipe, CAT_COLS),
        ],
        sparse_threshold=0,
    )


# Typical approval lag in the 2017-Mar 2018 training orders (median, in hours). The form has no
# approval time, so a single order is scored as if it was approved after the usual delay.
TYPICAL_APPROVAL_LAG_HOURS = 0.3


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check that df has the 19 inputs with allowed values; raise a ValueError naming the column.

    Uses the same rules as the API (both read schema.py), so training data and live orders
    agree on what a valid order is. Returns df unchanged so it can be chained.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col, spec in COLUMN_SPECS.items():
        values = df[col]
        if values.isna().any() and not spec.nullable:
            raise ValueError(f"Column '{col}' has missing values but cannot be empty")
        present = values.dropna()
        if spec.kind == "numeric":
            numbers = pd.to_numeric(present, errors="coerce")
            if numbers.isna().any():
                raise ValueError(f"Column '{col}' must be numeric")
            if ((numbers < spec.min) | (numbers > spec.max)).any():
                raise ValueError(f"Column '{col}' has values outside [{spec.min}, {spec.max}]")
            if spec.integer and (numbers % 1 != 0).any():
                raise ValueError(f"Column '{col}' must hold whole numbers")
        elif spec.allowed is not None:
            bad = sorted(set(present) - set(spec.allowed))
            if bad:
                raise ValueError(f"Column '{col}' has values that are not allowed: {bad}")
    return df


@lru_cache(maxsize=1)
def _zip_lookup() -> pd.DataFrame:
    # Read once per process: the API and the frontend call single_item_order() per request.
    return pd.read_parquet(REFERENCE_DIR / "zip_lookup.parquet").set_index("zip_code_prefix")


def _zip_distance_km(customer_zip: int, seller_zip: int) -> float:
    zips = _zip_lookup()
    if customer_zip not in zips.index or seller_zip not in zips.index:
        # Same as training: an unknown zip leaves the distance missing, and the model's
        # missing-distance flag handles it.
        return np.nan
    c, s = zips.loc[customer_zip], zips.loc[seller_zip]
    return float(haversine_km(c.lat, c.lon, s.lat, s.lon))


def single_item_order(
    customer_zip_prefix: int,
    customer_state: str,
    seller_zip_prefix: int,
    seller_state: str,
    product_category: str | None,
    price: float,
    freight: float,
    weight_g: float | None,
    length_cm: float,
    height_cm: float,
    width_cm: float,
    promised_days: float,
    purchased_at: datetime | None = None,
) -> pd.DataFrame:
    """Turn the frontend's one-item form into one row of the 19 model inputs.

    A one-item order has item count 1, one product and one seller, so totals equal the item's
    values and mean equals max. Purchase time defaults to now.
    """
    if price <= 0:
        # The freight-to-price ratio divides by price; a zero would crash the API with a 500.
        raise ValueError("Column 'total_price': price must be greater than 0")

    purchased_at = purchased_at or datetime.now()
    distance = _zip_distance_km(int(customer_zip_prefix), int(seller_zip_prefix))
    row = {
        "promised_days_from_approval": promised_days,
        "approval_lag_hours": TYPICAL_APPROVAL_LAG_HOURS,
        "purchase_weekday": purchased_at.weekday(),
        "purchase_hour": purchased_at.hour,
        "item_count": 1,
        "distinct_product_count": 1,
        "seller_count": 1,
        "total_price": price,
        "total_freight": freight,
        "freight_to_price_ratio": freight / price,
        "mean_product_weight_g": weight_g,
        "max_product_weight_g": weight_g,
        "total_product_volume_cm3": length_cm * height_cm * width_cm,
        "mean_distance_km": distance,
        "max_distance_km": distance,
        "same_state_item_share": 1.0 if customer_state == seller_state else 0.0,
        "customer_state": customer_state,
        "any_seller_state": seller_state,
        "primary_product_category": product_category,
    }
    return validate(pd.DataFrame([row], columns=INPUT_COLUMNS))
