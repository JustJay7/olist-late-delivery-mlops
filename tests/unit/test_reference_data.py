"""The committed reference files must match what ingest builds, so no one uses a stale copy."""

import pandas as pd

from olist_mlops import config
from olist_mlops.ingest import REFERENCE_DIR, SAMPLE_DIR, build_orders, split_serving
from olist_mlops.schema import INPUT_COLUMNS, TARGET


def test_serving_split_is_5000_baseline_and_16333_replay():
    baseline, replay = split_serving(build_orders())
    assert (len(baseline), len(replay)) == (5_000, 16_333)
    assert set(baseline["order_id"]).isdisjoint(replay["order_id"])
    # Both parts come from Apr-Jul 2018, which training never sees.
    assert baseline["order_purchase_timestamp"].min() >= config.TEST_START


def test_committed_baseline_matches_the_split():
    committed = pd.read_parquet(REFERENCE_DIR / "serving_baseline.parquet")
    baseline, _ = split_serving(build_orders())
    assert committed["order_id"].tolist() == baseline["order_id"].tolist()
    assert {"order_id", TARGET, *INPUT_COLUMNS} <= set(committed.columns)


def test_ci_sample_has_500_training_orders_with_both_classes():
    sample = pd.read_parquet(SAMPLE_DIR / "orders_sample.parquet")
    assert len(sample) == 500
    assert set(INPUT_COLUMNS) <= set(sample.columns)
    assert (sample["order_purchase_timestamp"] < config.TEST_START).all()
    # A CI model needs late and on-time orders to train on.
    assert sample[TARGET].nunique() == 2


def test_zip_lookup_has_one_row_per_prefix():
    zips = pd.read_parquet(REFERENCE_DIR / "zip_lookup.parquet")
    assert zips["zip_code_prefix"].is_unique
    assert zips[["lat", "lon"]].notna().all().all()
