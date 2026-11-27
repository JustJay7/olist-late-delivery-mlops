import json
from datetime import datetime

import pytest

from olist_mlops.features import TYPICAL_APPROVAL_LAG_HOURS
from olist_mlops.schema import INPUT_COLUMNS
from services.api.schemas import OrderInput
from services.frontend.order import EXAMPLE_FORM, FormError, build_order


@pytest.fixture
def form():
    return dict(EXAMPLE_FORM)


def test_build_order_matches_api_contract(form):
    order = build_order(form)

    assert list(order) == INPUT_COLUMNS
    OrderInput(**order)  # raises if the frontend and the API disagree on any column


def test_build_order_single_item_features(form):
    order = build_order(form)

    assert order["item_count"] == order["distinct_product_count"] == order["seller_count"] == 1
    assert order["total_product_volume_cm3"] == 20 * 15 * 10
    assert order["freight_to_price_ratio"] == pytest.approx(17.50 / 89.90)
    assert order["same_state_item_share"] == 0.0  # RJ customer, SP seller
    assert 300 < order["mean_distance_km"] < 450  # Rio de Janeiro to Campinas
    assert order["max_distance_km"] == order["mean_distance_km"]
    assert order["approval_lag_hours"] == TYPICAL_APPROVAL_LAG_HOURS  # same as the simulator


def test_build_order_same_state_and_unknown_zip(form):
    form.update(seller_state="RJ", seller_zip="99999")

    order = build_order(form)

    assert order["same_state_item_share"] == 1.0
    assert order["mean_distance_km"] is None  # zip not geocoded: the model imputes it


def test_build_order_is_plain_json(form):
    form.update(seller_zip="99999", weight_g="")  # missing values come back as NaN

    order = build_order(form)

    json.dumps(order, allow_nan=False)  # raises on numpy numbers or NaN
    assert order["mean_product_weight_g"] is None


def test_build_order_uses_brazilian_time(form):
    # Monday 02:30 UTC is still Sunday 23:30 in Sao Paulo, where the training timestamps are
    order = build_order(form, now=datetime.fromisoformat("2026-11-30T02:30:00+00:00"))

    assert (order["purchase_weekday"], order["purchase_hour"]) == (6, 23)


@pytest.mark.parametrize(
    "field, value",
    [
        ("price", "abc"),
        ("price", "0"),
        ("price", "-5"),
        ("freight", ""),
        ("customer_zip", "RJ-1"),
        ("promised_days", "400"),
    ],
)
def test_build_order_names_bad_field(form, field, value):
    form[field] = value

    with pytest.raises(FormError) as err:
        build_order(form)

    assert err.value.field == field


def test_out_of_range_value_is_put_back_on_its_field(form):
    form["promised_days"] = "400"

    with pytest.raises(FormError) as err:
        build_order(form)

    assert str(err.value) == (
        "Promised delivery (days): must be between 0 and 365 (promised_days_from_approval)."
    )
