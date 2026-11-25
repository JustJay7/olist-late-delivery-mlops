import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.model_loader import load_champion

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def test_model_path(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "test-model"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_test_model.py"),
            str(model_path),
        ],
        check=True,
        cwd=ROOT,
    )

    return model_path


@pytest.fixture
def real_client(monkeypatch, test_model_path):
    monkeypatch.setenv("MODEL_URI", str(test_model_path))

    with TestClient(app) as client:
        yield client


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
        "total_price": 100,
        "total_freight": 15,
        "freight_to_price_ratio": 0.15,
        "mean_product_weight_g": 500,
        "max_product_weight_g": 500,
        "total_product_volume_cm3": 1000,
        "mean_distance_km": 200,
        "max_distance_km": 200,
        "same_state_item_share": 1,
        "customer_state": "SP",
        "any_seller_state": "SP",
        "primary_product_category": "electronics",
    }


def test_model_override(monkeypatch, test_model_path):
    monkeypatch.setenv("MODEL_URI", str(test_model_path))

    loaded = load_champion()

    assert loaded.version == "test"
    assert 0 <= loaded.threshold <= 1
    assert load_champion(current_version="test") is None


def test_probability_range(real_client, valid_order):
    response = real_client.post("/predict", json=valid_order)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["late_probability"] <= 1
    assert result["model_version"] == "test"


def test_shorter_window_increases_risk(real_client, valid_order):
    probabilities = []

    for days in [2, 10, 30]:
        order = {
            **valid_order,
            "promised_days_from_approval": days,
        }

        response = real_client.post("/predict", json=order)

        assert response.status_code == 200

        probabilities.append(response.json()["late_probability"])

    assert probabilities[0] > probabilities[1] > probabilities[2]
