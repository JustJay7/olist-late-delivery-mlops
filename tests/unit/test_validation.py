"""validate() and single_item_order(): the checks every order passes before it reaches the model."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from olist_mlops.features import BRAZIL_TZ, make_preprocess, single_item_order, validate
from olist_mlops.ingest import SAMPLE_DIR, build_orders
from olist_mlops.schema import INPUT_COLUMNS, TARGET

FORM = dict(
    customer_zip_prefix=1310,
    customer_state="SP",
    seller_zip_prefix=13023,
    seller_state="SP",
    product_category="toys",
    price=100.0,
    freight=15.0,
    weight_g=800.0,
    length_cm=30.0,
    height_cm=20.0,
    width_cm=10.0,
    promised_days=20.0,
    purchased_at=datetime(2018, 5, 14, 10, 30),  # a Monday
)


@pytest.fixture(scope="module")
def sample():
    return pd.read_parquet(SAMPLE_DIR / "orders_sample.parquet")


def test_all_85371_orders_pass_validation():
    # Data validation from my task card: columns present, no missing target, values in range.
    orders = build_orders()
    assert len(orders) == 85_371
    assert orders[TARGET].notna().all()
    validate(orders)


def test_missing_column_is_named(sample):
    with pytest.raises(ValueError, match="total_price"):
        validate(sample.drop(columns="total_price"))


def test_out_of_range_value_is_named(sample):
    bad = sample.copy()
    bad.loc[bad.index[0], "purchase_hour"] = 25
    with pytest.raises(ValueError, match="purchase_hour"):
        validate(bad)


def test_unknown_state_is_rejected(sample):
    bad = sample.copy()
    bad.loc[bad.index[0], "customer_state"] = "XX"
    with pytest.raises(ValueError, match="customer_state"):
        validate(bad)


def test_missing_value_only_allowed_where_the_schema_says(sample):
    ok = sample.copy()
    ok.loc[ok.index[0], "mean_distance_km"] = np.nan
    validate(ok)
    bad = sample.copy()
    bad.loc[bad.index[0], "total_price"] = np.nan
    with pytest.raises(ValueError, match="total_price"):
        validate(bad)


def test_single_item_order_builds_the_19_inputs():
    row = single_item_order(**FORM).iloc[0]
    assert list(row.index) == INPUT_COLUMNS
    assert (row.item_count, row.distinct_product_count, row.seller_count) == (1, 1, 1)
    assert row.freight_to_price_ratio == pytest.approx(0.15)
    assert row.total_product_volume_cm3 == 6000
    assert (row.purchase_weekday, row.purchase_hour) == (0, 10)
    assert row.same_state_item_share == 1.0
    # Sao Paulo centre to Campinas: about 85 km in a straight line.
    assert row.mean_distance_km == pytest.approx(85, abs=5)


def test_single_item_order_across_states_and_unknown_zip():
    other_state = single_item_order(**{**FORM, "seller_state": "RJ"}).iloc[0]
    assert other_state.same_state_item_share == 0.0
    unknown_zip = single_item_order(**{**FORM, "seller_zip_prefix": 99999}).iloc[0]
    assert np.isnan(unknown_zip.mean_distance_km)


def test_form_order_goes_through_preprocessing(sample):
    # A preprocessing fitted on real orders turns the form row into features with no gaps.
    pre = make_preprocess(min_frequency=20).fit(sample[INPUT_COLUMNS])
    X = pre.transform(single_item_order(**FORM))
    assert X.shape[0] == 1 and not np.isnan(X).any()


@pytest.mark.parametrize("price", [0, -5.0])
def test_single_item_order_rejects_a_price_of_zero_or_less(price):
    with pytest.raises(ValueError, match="price"):
        single_item_order(**{**FORM, "price": price})


def test_default_purchase_time_is_brazil_time():
    # The model learned purchase_hour from Brazil timestamps, so "now" must be Brazil time too.
    form = {k: v for k, v in FORM.items() if k != "purchased_at"}
    row = single_item_order(**form).iloc[0]
    now = datetime.now(BRAZIL_TZ)
    assert (row.purchase_weekday, row.purchase_hour) == (now.weekday(), now.hour)
