"""Turns the one-item web form into the 19 model inputs the API expects.

Stand-in for features.single_item_order() (issue #8): once that merges, build_order() should
call it instead, so the frontend and the simulator can't compute features differently.
"""

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

from olist_mlops import config
from olist_mlops.ingest import NAMES, REFERENCE_DIR, haversine_km

# Olist timestamps are Brazilian local time, so the weekday and hour must be too.
BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")

# Form field name -> label shown to the user, in the order the form shows them.
FORM_FIELDS = {
    "customer_zip": "Customer zip prefix",
    "customer_state": "Customer state",
    "seller_zip": "Seller zip prefix",
    "seller_state": "Seller state",
    "category": "Product category",
    "price": "Price (R$)",
    "freight": "Freight (R$)",
    "weight_g": "Weight (g)",
    "length_cm": "Length (cm)",
    "width_cm": "Width (cm)",
    "height_cm": "Height (cm)",
    "promised_days": "Promised delivery (days)",
}

# A typical order from the training data, so the form can be submitted as it opens.
EXAMPLE_FORM = {
    "customer_zip": "22790",
    "customer_state": "RJ",
    "seller_zip": "13023",
    "seller_state": "SP",
    "category": "health_beauty",
    "price": "89.90",
    "freight": "17.50",
    "weight_g": "650",
    "length_cm": "20",
    "width_cm": "15",
    "height_cm": "10",
    "promised_days": "23",
}

# Which form field each model input comes from, so an API 422 can name what the user typed.
COLUMN_TO_FIELD = {
    "promised_days_from_approval": "promised_days",
    "total_price": "price",
    "total_freight": "freight",
    "freight_to_price_ratio": "freight",
    "mean_product_weight_g": "weight_g",
    "max_product_weight_g": "weight_g",
    "total_product_volume_cm3": "length_cm",
    "mean_distance_km": "customer_zip",
    "max_distance_km": "customer_zip",
    "customer_state": "customer_state",
    "any_seller_state": "seller_state",
    "primary_product_category": "category",
}


class FormError(ValueError):
    """A form value that can't be read, named by its form field."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{FORM_FIELDS[field]}: {message}")
        self.field = field


@lru_cache(maxsize=1)
def zip_lookup() -> dict[int, tuple[float, float]]:
    zips = pd.read_parquet(REFERENCE_DIR / "zip_lookup.parquet")
    return dict(zip(zips.zip_code_prefix, zip(zips.lat, zips.lon)))


@lru_cache(maxsize=1)
def categories() -> list[str]:
    path = config.RAW_DIR / (NAMES["translation"] + ".gz")
    names = pd.read_csv(path, encoding="utf-8-sig").product_category_name_english
    return sorted(names.dropna().unique())


def _number(form, field: str, required: bool = True) -> float | None:
    raw = (form.get(field) or "").strip().replace(",", ".")
    if not raw:
        if required:
            raise FormError(field, "this field is required.")
        return None
    try:
        return float(raw)
    except ValueError:
        raise FormError(field, f"'{raw}' is not a number.") from None


def _zip(form, field: str) -> int:
    raw = (form.get(field) or "").strip()
    if not raw.isdigit():
        raise FormError(field, "enter the first 5 digits of the zip code.")
    return int(raw)


def distance_km(customer_zip: int, seller_zip: int) -> float | None:
    """Straight-line route length; None when either zip isn't geocoded, as in training."""
    lookup = zip_lookup()
    if customer_zip not in lookup or seller_zip not in lookup:
        return None
    (c_lat, c_lon), (s_lat, s_lon) = lookup[customer_zip], lookup[seller_zip]
    return float(haversine_km(s_lat, s_lon, c_lat, c_lon))


def build_order(form, now: datetime | None = None) -> dict:
    """One single-item order as the 19 model inputs. Range checks are left to the API."""
    price = _number(form, "price")
    if price == 0:
        # The ratio below is undefined at 0, and the model was never trained on free items.
        raise FormError("price", "must be more than 0.")
    freight = _number(form, "freight")
    weight = _number(form, "weight_g", required=False)
    volume = _number(form, "length_cm") * _number(form, "width_cm") * _number(form, "height_cm")
    distance = distance_km(_zip(form, "customer_zip"), _zip(form, "seller_zip"))
    customer_state = form.get("customer_state", "")
    seller_state = form.get("seller_state", "")
    # The order is scored as it is approved, so purchase and approval are both "now".
    now = (now or datetime.now(BRAZIL_TZ)).astimezone(BRAZIL_TZ)

    order = {
        "promised_days_from_approval": _number(form, "promised_days"),
        "approval_lag_hours": 0.0,
        "purchase_weekday": now.weekday(),
        "purchase_hour": now.hour,
        "item_count": 1,
        "distinct_product_count": 1,
        "seller_count": 1,
        "total_price": price,
        "total_freight": freight,
        "freight_to_price_ratio": freight / price,
        "mean_product_weight_g": weight,
        "max_product_weight_g": weight,
        "total_product_volume_cm3": volume,
        "mean_distance_km": distance,
        "max_distance_km": distance,
        "same_state_item_share": 1.0 if customer_state == seller_state else 0.0,
        "customer_state": customer_state,
        "any_seller_state": seller_state,
        "primary_product_category": form.get("category") or None,
    }
    return order
