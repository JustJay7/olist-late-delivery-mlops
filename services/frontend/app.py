"""Web form for one order: builds the model inputs, asks the API, shows the risk."""

import os

import requests
from flask import Flask, render_template, request

from olist_mlops.schema import BRAZILIAN_STATES
from services.frontend.order import (
    COLUMN_TO_FIELD,
    EXAMPLE_FORM,
    FORM_FIELDS,
    FormError,
    build_order,
    categories,
)

# Inside Docker the API is reached by its service name, not localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Long enough for a cold model load, short enough that a dead API doesn't hang the page.
API_TIMEOUT_SECONDS = 10

app = Flask(__name__)


def show_form(values, error=None, field=None, status=200):
    page = render_template(
        "form.html",
        fields=FORM_FIELDS,
        values=values,
        states=BRAZILIAN_STATES,
        categories=categories(),
        error=error,
        error_field=field,
    )
    return page, status


def explain_422(response) -> tuple[str, str | None]:
    """The API names the bad model input; translate it back to the form field."""
    try:
        first = response.json()["detail"][0]
        column, reason = first["loc"][-1], first["msg"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "The API rejected this order as invalid.", None
    field = COLUMN_TO_FIELD.get(column)
    label = FORM_FIELDS[field] if field else column
    return f"The API rejected {label} ({column}): {reason}.", field


@app.get("/")
def index():
    return show_form(EXAMPLE_FORM)


@app.post("/predict")
def predict():
    values = request.form.to_dict()
    try:
        order = build_order(values)
    except FormError as err:
        return show_form(values, str(err), err.field, status=400)

    try:
        response = requests.post(
            f"{API_URL}/predict", json={**order, "source": "frontend"}, timeout=API_TIMEOUT_SECONDS
        )
    except requests.RequestException:
        message = (
            f"The prediction API at {API_URL} can't be reached. If the stack has just started, "
            "wait a minute and try again."
        )
        return show_form(values, message, status=502)

    if response.status_code == 422:
        message, field = explain_422(response)
        return show_form(values, message, field, status=422)
    if response.status_code == 503:
        message = (
            "The API is running but has no model loaded yet. The first training run takes a "
            "few minutes; try again shortly."
        )
        return show_form(values, message, status=503)
    if response.status_code != 200:
        message = f"The API returned an unexpected error (HTTP {response.status_code})."
        return show_form(values, message, status=502)

    result = response.json()
    return render_template(
        "result.html",
        result=result,
        risk_pct=round(result["late_probability"] * 100),
        order=order,
    )
