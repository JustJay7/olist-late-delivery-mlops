from typing import Literal

from pydantic import Field, create_model

from olist_mlops.schema import COLUMN_SPECS, INPUT_COLUMNS

fields = {}

for column in INPUT_COLUMNS:
    spec = COLUMN_SPECS[column]

    if spec.kind == "numeric":
        field_type = int if spec.integer else float
        constraints = {"ge": spec.min, "le": spec.max}
    else:
        field_type = Literal[spec.allowed] if spec.allowed else str
        constraints = {}

    if spec.nullable:
        field_type = field_type | None

    fields[column] = (
        field_type,
        Field(
            default=None if spec.nullable else ...,
            **constraints,
        ),
    )


fields["order_id"] = (str | None, None)
fields["source"] = (
    Literal["frontend", "simulator", "simulator-drift"],
    "frontend",
)


OrderInput = create_model("OrderInput", **fields)
