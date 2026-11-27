"""Turns the one-item web form into the 19 model inputs the API expects.

The features come from features.single_item_order(), the same function the simulator uses, so
the frontend can't compute them differently. This module only reads the form and puts its
errors back on the field the user typed.
"""

import re
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from olist_mlops import config
from olist_mlops.features import BRAZIL_TZ, single_item_order
from olist_mlops.ingest import NAMES
from olist_mlops.schema import COLUMN_SPECS

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

# Which form field each model input comes from, so a rejected input can name what the user typed.
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

# single_item_order() and validate() start every error with the column they reject.
COLUMN_ERROR = re.compile(r"Column '(\w+)'[:\s]*(.*)")


class FormError(ValueError):
    """A form value that can't be read, named by its form field."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{FORM_FIELDS[field]}: {message}")
        self.field = field


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


def _form_error(err: ValueError) -> FormError:
    """Put a rejected model input back on the form field it came from."""
    match = COLUMN_ERROR.match(str(err))
    if not match or match.group(1) not in COLUMN_TO_FIELD:
        raise err  # not something the user typed, so a bug: let it show
    column, reason = match.groups()
    if reason.startswith("has values outside"):
        spec = COLUMN_SPECS[column]
        reason = f"must be between {spec.min:g} and {spec.max:g}"
    elif reason.startswith("has values that are not allowed"):
        reason = "is not one of the allowed values"
    return FormError(COLUMN_TO_FIELD[column], f"{reason} ({column}).")


def _plain(value):
    """numpy numbers and NaN can't go into JSON; the API wants Python numbers and null."""
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def build_order(form, now: datetime | None = None) -> dict:
    """One single-item order as the 19 model inputs, checked with the same rules as the API."""
    inputs = dict(
        customer_zip_prefix=_zip(form, "customer_zip"),
        customer_state=form.get("customer_state", ""),
        seller_zip_prefix=_zip(form, "seller_zip"),
        seller_state=form.get("seller_state", ""),
        product_category=form.get("category") or None,
        price=_number(form, "price"),
        freight=_number(form, "freight"),
        weight_g=_number(form, "weight_g", required=False),
        length_cm=_number(form, "length_cm"),
        width_cm=_number(form, "width_cm"),
        height_cm=_number(form, "height_cm"),
        promised_days=_number(form, "promised_days"),
        # Olist timestamps are Brazilian local time, so the weekday and hour must be too.
        purchased_at=(now or datetime.now(BRAZIL_TZ)).astimezone(BRAZIL_TZ),
    )
    try:
        row = single_item_order(**inputs)
    except ValueError as err:
        raise _form_error(err) from None
    return {column: _plain(value) for column, value in row.iloc[0].items()}
