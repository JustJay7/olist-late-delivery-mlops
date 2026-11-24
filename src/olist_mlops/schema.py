"""The 19 model inputs, shared by training, the API, the frontend, the simulator and monitoring.

Import names from here instead of retyping them, so a column can only be renamed in one place.
These are the inputs of our INSY 674 final model: the 22 approval-time features minus
customer_zip_code_prefix and approval_month (dropped in 674 section 3) and purchase_month
(dropped in 674 section 5.1).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """Type and valid range of one input column."""

    kind: str  # "numeric" or "categorical"
    min: float | None = None
    max: float | None = None
    integer: bool = False
    # Only distances, weights and category can be missing; the model imputes them.
    nullable: bool = False
    # For categoricals with a closed set of values; None means any text is accepted.
    allowed: tuple[str, ...] | None = None


# Brazil's 27 federative units. Customers come from all of them; sellers from 22 in our data.
BRAZILIAN_STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)  # fmt: skip

# Ranges reject impossible values, not unusual ones: every one of the 85,371 orders from
# last course fits inside them, with room for real orders we have not seen yet.
COLUMN_SPECS: dict[str, ColumnSpec] = {
    # Order timing
    "promised_days_from_approval": ColumnSpec("numeric", min=0, max=365),
    "approval_lag_hours": ColumnSpec("numeric", min=0, max=24 * 90),
    "purchase_weekday": ColumnSpec("numeric", min=0, max=6, integer=True),
    "purchase_hour": ColumnSpec("numeric", min=0, max=23, integer=True),
    # Basket
    "item_count": ColumnSpec("numeric", min=1, max=100, integer=True),
    "distinct_product_count": ColumnSpec("numeric", min=1, max=100, integer=True),
    "seller_count": ColumnSpec("numeric", min=1, max=100, integer=True),
    "total_price": ColumnSpec("numeric", min=0, max=100_000),
    "total_freight": ColumnSpec("numeric", min=0, max=10_000),
    "freight_to_price_ratio": ColumnSpec("numeric", min=0, max=100),
    # Product size; weight is missing when the seller left it blank
    "mean_product_weight_g": ColumnSpec("numeric", min=0, max=100_000, nullable=True),
    "max_product_weight_g": ColumnSpec("numeric", min=0, max=100_000, nullable=True),
    "total_product_volume_cm3": ColumnSpec("numeric", min=0, max=10_000_000),
    # Route; distance is missing when a zip prefix is not in the geolocation table.
    # The upper bound is above Brazil's width because the raw geolocation has a few bad
    # coordinates (674 section 2.1) that the model was trained with.
    "mean_distance_km": ColumnSpec("numeric", min=0, max=10_000, nullable=True),
    "max_distance_km": ColumnSpec("numeric", min=0, max=10_000, nullable=True),
    "same_state_item_share": ColumnSpec("numeric", min=0, max=1),
    # Where the order goes, where it ships from, and what it is
    "customer_state": ColumnSpec("categorical", allowed=BRAZILIAN_STATES),
    "any_seller_state": ColumnSpec("categorical", allowed=BRAZILIAN_STATES),
    "primary_product_category": ColumnSpec("categorical", nullable=True),
}

INPUT_COLUMNS: list[str] = list(COLUMN_SPECS)
NUMERIC_COLUMNS: list[str] = [c for c, s in COLUMN_SPECS.items() if s.kind == "numeric"]
CATEGORICAL_COLUMNS: list[str] = [c for c, s in COLUMN_SPECS.items() if s.kind == "categorical"]
NULLABLE_COLUMNS: list[str] = [c for c, s in COLUMN_SPECS.items() if s.nullable]

TARGET = "is_late"
