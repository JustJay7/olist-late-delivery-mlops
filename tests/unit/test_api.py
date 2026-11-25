from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from olist_mlops.schema import INPUT_COLUMNS
from services.api.main import app
from services.api.model_loader import LoadedModel
from services.api.schemas import OrderInput


@pytest.fixture
def client(monkeypatch):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.8, 0.2]])

    loaded_model = LoadedModel(
        model=model,
        threshold=0.15,
        version="1",
    )

    monkeypatch.setattr(
        "services.api.main.load_champion",
        lambda current_version=None: (loaded_model if current_version != "1" else None),
    )

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_order():
    return {
        "promised_days_from_approval": 10,
        "approval_lag_hours": 2,
        "purchase_weekday": 3,
        "purchase_hour": 14,
        "item_count": 1,
        "distinct_product_count": 1,
        "seller_count": 1,
        "total_price": 100.0,
        "total_freight": 15.0,
        "freight_to_price_ratio": 0.15,
        "mean_product_weight_g": 500.0,
        "max_product_weight_g": 500.0,
        "total_product_volume_cm3": 1000.0,
        "mean_distance_km": 200.0,
        "max_distance_km": 200.0,
        "same_state_item_share": 1.0,
        "customer_state": "SP",
        "any_seller_state": "SP",
        "primary_product_category": "electronics",
    }


# Health endpoint tests


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Pydantic schema tests


def test_schema_columns():
    assert set(OrderInput.model_fields) == set(INPUT_COLUMNS) | {
        "order_id",
        "source",
    }


def test_valid_order(valid_order):
    order = OrderInput(**valid_order)

    assert order.total_price == 100.0
    assert order.source == "frontend"
    assert order.order_id is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("purchase_hour", 25),
        ("purchase_weekday", -1),
        ("item_count", 0),
        ("item_count", 2.5),
        ("total_price", -100),
        ("customer_state", "Canada"),
        ("same_state_item_share", 1.5),
        ("total_price", None),
        ("source", "invalid"),
    ],
)
def test_invalid_order(valid_order, field, value):
    valid_order[field] = value

    with pytest.raises(ValidationError) as exc:
        OrderInput(**valid_order)

    assert exc.value.errors()[0]["loc"] == (field,)


def test_missing_required_field(valid_order):
    del valid_order["total_price"]

    with pytest.raises(ValidationError):
        OrderInput(**valid_order)


def test_nullable_fields(valid_order):
    valid_order["mean_distance_km"] = None
    valid_order["mean_product_weight_g"] = None
    valid_order["primary_product_category"] = None

    order = OrderInput(**valid_order)

    assert order.mean_distance_km is None
    assert order.mean_product_weight_g is None
    assert order.primary_product_category is None


# Prediction endpoint tests


def test_predict_valid_order(client, valid_order):
    valid_order["order_id"] = "test-001"

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 200

    result = response.json()

    assert result["order_id"] == "test-001"
    assert 0 <= result["late_probability"] <= 1
    assert result["late_probability"] == pytest.approx(0.2)
    assert result["flagged"] is True
    assert result["model_version"] == "1"


def test_predict_invalid_order(client, valid_order):
    valid_order["purchase_hour"] = 25

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "body",
        "purchase_hour",
    ]


def test_predict_generates_order_id(client, valid_order):
    response = client.post("/predict", json=valid_order)

    assert response.status_code == 200
    assert response.json()["order_id"]


def test_predict_missing_required_field(client, valid_order):
    del valid_order["total_price"]

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 422


def test_predict_model_unavailable(client, valid_order):
    app.state.loaded_model = None

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 503


# Prometheus metrics tests


def test_metrics(client):
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
