"""Preprocessing shared by training and serving.

make_preprocess() is copied from our INSY 674 notebook 05 (section 3.4 pipeline) without
changes to the steps, so the model trains exactly as it did last course and its CV results
stay comparable. The column groups say which transformation each column gets.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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
