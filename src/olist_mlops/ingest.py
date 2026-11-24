"""Builds the modelling dataset from the committed raw tables, in memory.

Ported from the 674 repo's scripts/build_olist_model_table.py. The joins and features are
unchanged; the differences are that it returns a DataFrame instead of writing a CSV
(locked decision 2), so no step depends on a file built by hand, and that payments are left
out because 674 never used them (availability at approval is not documented).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from olist_mlops import config
from olist_mlops.data import build_population, load_raw
from olist_mlops.integration import integrate_features
from olist_mlops.schema import INPUT_COLUMNS, TARGET

NAMES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "geo": "olist_geolocation_dataset.csv",
}


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / NAMES[name]
    if not path.exists():
        path = folder / (NAMES[name] + ".gz")
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} table (.csv or .csv.gz) under {folder}")
    return pd.read_csv(path, low_memory=False)


def zip_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    """One median lat/lon per zip prefix; a prefix has many raw rows, some of them noisy."""
    return geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median")
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance on the earth's surface; the road network is not in the data."""
    a, b = np.radians(lat1), np.radians(lat2)
    da, dl = a - b, np.radians(lon1 - lon2)
    h = np.sin(da / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build_order_table(folder: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per delivered order with the 674 approval-time features."""
    orders = _read(folder, "orders")
    customers = _read(folder, "customers")
    items = _read(folder, "items")
    products = _read(folder, "products")
    sellers = _read(folder, "sellers")
    translation = _read(folder, "translation")
    geo = _read(folder, "geo")
    for c in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    # Prediction is made at approval; labels exist only for completed deliveries.
    orders = orders.loc[
        orders.order_status.eq("delivered")
        & orders.order_approved_at.notna()
        & orders.order_delivered_customer_date.notna()
        & orders.order_estimated_delivery_date.notna()
    ].copy()
    orders["is_late"] = (
        orders.order_delivered_customer_date.dt.normalize()
        > orders.order_estimated_delivery_date.dt.normalize()
    ).astype("int8")
    promised = orders.order_estimated_delivery_date - orders.order_approved_at
    orders["promised_days_from_approval"] = promised.dt.total_seconds() / 86400
    lag = orders.order_approved_at - orders.order_purchase_timestamp
    orders["approval_lag_hours"] = lag.dt.total_seconds() / 3600
    orders["purchase_month"] = orders.order_purchase_timestamp.dt.month
    orders["purchase_weekday"] = orders.order_purchase_timestamp.dt.day_of_week
    orders["purchase_hour"] = orders.order_purchase_timestamp.dt.hour
    orders["approval_month"] = orders.order_approved_at.dt.month
    # Outcome dates stay for audits and the time split, never as model features.
    orders = orders.drop(columns=["order_delivered_carrier_date", "order_status"])

    customers = customers[
        ["customer_id", "customer_zip_code_prefix", "customer_state"]
    ].drop_duplicates("customer_id")
    out = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    products = products.merge(
        translation.drop_duplicates("product_category_name"),
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )
    items = items.merge(products, on="product_id", how="left", validate="many_to_one")
    items = items.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]].drop_duplicates(
            "seller_id"
        ),
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    items["product_volume_cm3"] = (
        items.product_length_cm * items.product_height_cm * items.product_width_cm
    )

    centers = zip_centroids(geo)
    seller_geo = centers.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "lat": "seller_lat",
            "lon": "seller_lon",
        }
    )
    customer_geo = centers.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "lat": "customer_lat",
            "lon": "customer_lon",
        }
    )
    items = items.merge(seller_geo, on="seller_zip_code_prefix", how="left", validate="many_to_one")
    # Customer zip per item for the distance, without creating item-level duplicates of orders.
    loc = out[["order_id", "customer_zip_code_prefix", "customer_state"]]
    items = items.merge(loc, on="order_id", how="inner", validate="many_to_one")
    items = items.merge(
        customer_geo, on="customer_zip_code_prefix", how="left", validate="many_to_one"
    )
    items["distance_km"] = haversine_km(
        items.seller_lat, items.seller_lon, items.customer_lat, items.customer_lon
    )
    items["same_state"] = np.where(
        items.seller_state.notna() & items.customer_state.notna(),
        (items.seller_state == items.customer_state).astype(float),
        np.nan,
    )
    # An order with several categories is summarised by its most expensive item's category.
    cat = (
        items.assign(_price=items.price.fillna(0))
        .sort_values(["order_id", "_price"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "product_category_name_english"]]
        .rename(columns={"product_category_name_english": "primary_product_category"})
    )
    by_order = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "size"),
        distinct_product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        mean_product_weight_g=("product_weight_g", "mean"),
        max_product_weight_g=("product_weight_g", "max"),
        total_product_volume_cm3=("product_volume_cm3", "sum"),
        mean_distance_km=("distance_km", "mean"),
        max_distance_km=("distance_km", "max"),
        same_state_item_share=("same_state", "mean"),
        any_seller_state=("seller_state", "first"),
    )
    by_order = by_order.merge(cat, on="order_id", how="left", validate="one_to_one")
    out = out.merge(by_order, on="order_id", how="left", validate="one_to_one")
    out["freight_to_price_ratio"] = out.total_freight / out.total_price.replace(0, np.nan)
    if not out.order_id.is_unique:
        raise ValueError("Unexpected duplicate order IDs")
    return out


def build_orders() -> pd.DataFrame:
    """The 85,371 modelling orders: 674 population and label, with the approval-time features.

    Sorted by purchase time, which the time split and the time-series CV rely on.
    """
    population, _ = build_population(load_raw("orders"))
    return integrate_features(population, build_order_table())


# Committed reference files (locked decision 3). Rebuild them with: python -m olist_mlops.ingest
REFERENCE_DIR = config.ROOT / "data" / "reference"
SAMPLE_DIR = config.ROOT / "data" / "sample"
BASELINE_SIZE = 5_000
SAMPLE_SIZE = 500


def split_serving(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the Apr-Jul 2018 orders once: a drift baseline, and the orders the simulator replays.

    Neither part is ever used for training. The seed is fixed so every laptop and container
    gets the same 5,000 baseline orders and the same 16,333 replay orders.
    """
    serving = orders[orders["order_purchase_timestamp"] >= config.TEST_START]
    baseline = serving.sample(n=BASELINE_SIZE, random_state=config.RANDOM_STATE).sort_index()
    replay = serving.drop(index=baseline.index)
    return baseline, replay


def write_reference_files() -> None:
    """Write the zip lookup, the serving baseline and the 500-order CI sample."""
    keep = ["order_id", "order_purchase_timestamp", *INPUT_COLUMNS, TARGET]
    orders = build_orders()
    baseline, _ = split_serving(orders)
    train = orders[orders["order_purchase_timestamp"] < config.TEST_START]
    sample = train.sample(n=SAMPLE_SIZE, random_state=config.RANDOM_STATE).sort_index()
    zips = zip_centroids(_read(config.RAW_DIR, "geo")).rename(
        columns={"geolocation_zip_code_prefix": "zip_code_prefix"}
    )

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    zips.to_parquet(REFERENCE_DIR / "zip_lookup.parquet", index=False)
    baseline[keep].to_parquet(REFERENCE_DIR / "serving_baseline.parquet", index=False)
    sample[keep].to_parquet(SAMPLE_DIR / "orders_sample.parquet", index=False)


if __name__ == "__main__":
    write_reference_files()
