import pytest
import requests

from services.frontend import app as frontend
from services.frontend.order import EXAMPLE_FORM

client = frontend.app.test_client()


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def form():
    return dict(EXAMPLE_FORM)


@pytest.fixture
def api(monkeypatch):
    """Replaces the HTTP call to the API; set .reply to a response or an exception."""

    class Api:
        def __init__(self):
            self.calls = []
            self.reply = FakeResponse(
                200,
                {
                    "order_id": "abc-1",
                    "late_probability": 0.23,
                    "flagged": True,
                    "model_version": "3",
                },
            )

        def post(self, url, json, timeout):
            self.calls.append(json)
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    fake = Api()
    monkeypatch.setattr(frontend.requests, "post", fake.post)
    return fake


def test_form_page_renders():
    response = client.get("/")

    assert response.status_code == 200
    assert b"Customer zip prefix" in response.data
    assert b"health_beauty" in response.data


def test_prediction_is_readable(form, api):
    response = client.post("/predict", data=form)

    assert response.status_code == 200
    assert b"Likely late: 23% risk, in the riskiest 10%" in response.data
    assert b"abc-1" in response.data
    assert api.calls[0]["source"] == "frontend"


def test_unflagged_prediction(form, api):
    api.reply = FakeResponse(
        200, {"order_id": "abc-2", "late_probability": 0.04, "flagged": False, "model_version": "3"}
    )

    response = client.post("/predict", data=form)

    assert b"Probably on time: 4% risk, not in the riskiest 10%" in response.data


def test_api_422_names_the_field(form, api):
    api.reply = FakeResponse(
        422,
        {
            "detail": [
                {
                    "loc": ["body", "promised_days_from_approval"],
                    "msg": "Input should be less than or equal to 365",
                }
            ]
        },
    )

    response = client.post("/predict", data=form)

    assert response.status_code == 422
    assert b"Promised delivery (days)" in response.data
    assert b"less than or equal to 365" in response.data


def test_api_unreachable_shows_message(form, api):
    api.reply = requests.ConnectionError("connection refused")

    response = client.post("/predict", data=form)

    assert response.status_code == 502
    assert b"can&#39;t be reached" in response.data


def test_api_without_model_shows_message(form, api):
    api.reply = FakeResponse(503, {"detail": "model not loaded"})

    response = client.post("/predict", data=form)

    assert response.status_code == 503
    assert b"no model loaded yet" in response.data


def test_bad_form_value_never_reaches_api(form, api):
    form["price"] = "lots"

    response = client.post("/predict", data=form)

    assert response.status_code == 400
    assert b"Price (R$): &#39;lots&#39; is not a number." in response.data
    assert api.calls == []
