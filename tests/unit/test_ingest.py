"""The in-memory ingest must rebuild last course's dataset exactly, with no CSV step."""

import pytest

from olist_mlops.ingest import build_orders, haversine_km
from olist_mlops.integration import APPROVAL_FEATURES


@pytest.fixture(scope="module")
def orders():
    # Built once for the module: it reads all 9 raw tables, which takes a few seconds.
    return build_orders()


def test_build_orders_gives_the_674_dataset(orders):
    assert len(orders) == 85_371
    assert orders["order_id"].is_unique
    assert int(orders["is_late"].sum()) == 5_986  # 7.0% late, as in the 674 README
    assert set(APPROVAL_FEATURES) <= set(orders.columns)


def test_build_orders_is_sorted_by_purchase_time(orders):
    # The time split and the time-series CV both assume this order.
    assert orders["order_purchase_timestamp"].is_monotonic_increasing


def test_haversine_matches_a_known_distance():
    # Sao Paulo to Rio de Janeiro city centres: about 360 km in a straight line.
    assert haversine_km(-23.55, -46.63, -22.91, -43.17) == pytest.approx(360, abs=10)
